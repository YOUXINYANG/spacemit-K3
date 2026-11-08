# K3 TCM 拷贝微基准

用于固定版本 `58745bdb86b759940577e0a8226cfbfa9987c286` 的初步实验。
调用已有动态库里的 `spacemit_kernels::rvv::memcpy1d`，与标准库 `memcpy` 对照。
已完成热缓冲区与大源数据集的 K3 实测。最新版本加入两个实验性预取包装函数，尚未在 K3 验证，也未修改模型使用的算子库。

## 测试范围

- 按固定版本 ime.cpp 的方式，必要时先向 `/proc/set_ai_thread` 写入 `0`，再将单线程固定在 CPU 8，使用真实 TCM 块 0；这是当前实例的映射。
- 普通内存到 TCM，大小为 16384、18432、20480 字节。
- 源地址按 128 字节对齐；目标为 TCM 起点加 128 或 136 字节，打印实际对齐余数。
- 每个组合使用三种数据模式，逐字节比对，并检查复制区域之外仍是保护值。轮转模式抽查首、中、末源块。
- 默认只检查正确性。`--bench` 才计时，热缓冲区模式预热 64 次，每轮复制 200000 次，测三轮，交替函数顺序。初轮实验为 20000 次；因部分结果有波动，后续增加计时长度。
- 加上 `--stream` 后申请 64MiB 源缓冲区，按当前复制大小划分源块，顺序读取并循环回绕。实际工作集为不超过 64MiB 的完整源块总大小，18KiB 时为 3640 块。计时前先完整遍历一轮源块。
- 两种模式均在计时循环中更新源指针。输出追加 `source_slots` 和 `source_working_set_bytes`，原前八列顺序不变。
- 最新版加入 `rvv_prefetch64`、`rvv_prefetch128`：分别按 64、128 字节间隔预取当前源块，再调用原 RVV 函数。间隔是实验参数，不代表已确认的硬件缓存行大小。
- 预取与相关循环开销均在计时范围内。四个函数都执行正确性检查，每轮交替正序和逆序测量。
- 计时区域不包含内存申请、数据初始化和正确性检查。
- 输出 `payload_GB_s = 复制字节数 / 纳秒数`，按十进制 GB/s 计，只计有效载荷一次。

默认反复使用同一源块，源数据很可能驻留缓存。轮转模式增大源工作集，
用于考察缓存条件的影响；没有强制清缓存，不能保证每次访问都来自 DDR。
这两种模式都不能等同于 Qwen 全模型的权重流式搬运性能，也不能直接推算模型加速比。
计时包含函数调用、指针更新和回绕判断；应使用同一版程序比较热缓冲区与轮转结果。
后续再加入多线程及更多尺寸。

## 文件

- `main.cpp`：硬件测试程序。
- `check_copy.h`：检查数据与目标区域边界。
- `test_check_copy.cpp`：验证检查器能发现数据错误和前后越界写。
- `copy_workload.h`、`test_copy_workload.cpp`：源指针轮转逻辑及其回绕、最终复制结果测试。
- `prefetch_copy.h`、`prefetch_copy.cpp`：实验性预取包装函数；单独使用 `-march=rv64gcv_zicbop` 编译。
- `CMakeLists.txt`：链接系统 TCM 库及现有 baseline 动态库。

## K3 上构建

将整个目录上传为 `~/qwen/tcm-copy`。以下命令都在 K3 上执行：

```bash
cmake -S ~/qwen/tcm-copy -B ~/qwen/builds/tcm-copy -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DK3_LLAMA_BIN="$HOME/qwen/builds/ime2-baseline/bin"
cmake --build ~/qwen/builds/tcm-copy --parallel 2
ctest --test-dir ~/qwen/builds/tcm-copy --output-on-failure
```

## 先检查正确性

确保没有其他 TCM 使用者，并且当前显示 8/8 空闲。使用普通 bianbu 用户。

```bash
mkdir -p ~/qwen/logs
spacemit-tcm-smi
set -o pipefail
env SPACEMIT_DISABLE_TCM=0 ~/qwen/builds/tcm-copy/tcm-copy \
  2>&1 | tee ~/qwen/logs/tcm-copy-check.log
k3_copy_rc=$?
echo "正确性测试退出码：$k3_copy_rc"
spacemit-tcm-smi
```

预期有六条 `PASS`（每条汇总当前四种函数、三种数据模式），
`tcm_free_result=0 test_result=0`，退出码 0，测试后 TCM 恢复 8/8 空闲。
CPU 绑定失败时，程序在申请 TCM 之前退出；请保留错误输出用于排查。
不要通过 GDB 在持有 TCM 时终止程序，也不要为日常运行添加强制释放命令。

## 正确性通过后计时

```bash
set -o pipefail
env SPACEMIT_DISABLE_TCM=0 ~/qwen/builds/tcm-copy/tcm-copy --bench \
  > ~/qwen/logs/tcm-copy-bench.csv \
  2> ~/qwen/logs/tcm-copy-bench.stderr.log
k3_copy_rc=$?
echo "计时测试退出码：$k3_copy_rc"
cat ~/qwen/logs/tcm-copy-bench.stderr.log
cat ~/qwen/logs/tcm-copy-bench.csv
spacemit-tcm-smi
```

最新四函数版本的 CSV 应包含 72 行测量记录；此前两函数版本为 36 行。若依赖库向标准输出打印启动日志，需先剔除非 CSV 行。
先查看三轮结果是否稳定，再比较同尺寸、同对齐条件下的两个函数。

## 大源数据集对照

先运行新模式的正确性检查，再用同一版程序测量两种模式：

```bash
env SPACEMIT_DISABLE_TCM=0 ~/qwen/builds/tcm-copy/tcm-copy --stream
env SPACEMIT_DISABLE_TCM=0 ~/qwen/builds/tcm-copy/tcm-copy --bench \
  > ~/qwen/logs/tcm-copy-v3-hot.csv 2> ~/qwen/logs/tcm-copy-v3-hot.stderr.log
env SPACEMIT_DISABLE_TCM=0 ~/qwen/builds/tcm-copy/tcm-copy --bench --stream \
  > ~/qwen/logs/tcm-copy-v3-stream.csv 2> ~/qwen/logs/tcm-copy-v3-stream.stderr.log
```

每个程序结束后检查退出码和 TCM 是否恢复 8/8 空闲；某步失败时先排查，不继续计时。

## 当前验证情况

在本地通过检查器测试，确认其能检测数据损坏、复制区域前后的越界写和无效范围。
主程序已在 K3 编译通过，检查器测试通过。初版直接绑定 CPU 8 时返回
`Invalid argument`，在申请 TCM 之前退出，测试后仍有 8/8 空闲块。
核对固定版本 ime.cpp 后，补上 `/proc/set_ai_thread` 注册步骤。修订版在 K3 上通过全部六组正确性检查，CPU=8，真实 TCM，正常释放，退出码 0，结束后 8/8 空闲。
20000 次版本的计时也成功完成；18KiB、目标地址余数为 8 时，三轮平均耗时 RVV 为 515.841ns、标准库为 1560.595ns，标准库耗时约为 RVV 的 3.025 倍。结果仅代表本次热缓冲区微基准。
200000 次版本已独立运行三次并成功结束，每次 TCM 均恢复 8/8 空闲。
18KiB、目标地址余数为 8 的九个计时样本：RVV 平均 455.776ns，标准库平均 1563.344ns，
耗时之比约 3.430；RVV 样本范围 418.900–478.730ns，样本变异系数约 4.00%。
两函数轮转版本已在 K3 编译通过，两项辅助测试通过，轮转正确性检查通过，hot/stream 计时均退出 0，结束后均为 8/8 空闲。

18KiB、目标地址余数为 8，三轮平均结果：

| 模式 | RVV 耗时/ns | RVV 吞吐/GB/s | 标准库耗时/ns | 标准库吞吐/GB/s |
| --- | ---: | ---: | ---: | ---: |
| hot | 460.951 | 39.987 | 1566.380 | 11.767 |
| stream | 950.441 | 19.393 | 1715.765 | 10.743 |

轮转源工作集为 67092480 字节、3640 块。RVV 的轮转耗时约为热缓冲区的 2.062 倍；
在轮转模式下，标准库耗时约为 RVV 的 1.805 倍。
结果说明源数据缓存条件显著影响本次测试，但不能据此认定所有耗时来自 DDR 或预测模型整体加速。

新增预取候选的语法检查在本地通过，硬件正确性、预取指令生成及性能尚待 K3 验证。
该候选仅检查“先预取整个源块再复制”是否值得继续研究。预取可能因为指令开销或缓存干扰而减速；没有收益时不应集成到模型算子。

## 预取候选实验

更新完整目录并重新构建后，先检查生成的指令：

```bash
objdump -d ~/qwen/builds/tcm-copy/CMakeFiles/tcm-copy.dir/prefetch_copy.cpp.o \
  | grep -F 'prefetch.r'
```

若没有输出，先排查编译产物与反汇编结果，不直接声称正在测量硬件预取。
然后运行 `tcm-copy --stream` 检查所有候选正确性，成功并确认 TCM 空闲后运行：

```bash
env SPACEMIT_DISABLE_TCM=0 ~/qwen/builds/tcm-copy/tcm-copy --bench --stream \
  > ~/qwen/logs/tcm-copy-v4-stream.csv \
  2> ~/qwen/logs/tcm-copy-v4-stream.stderr.log
```

主要比较同一 CSV 内的候选与 `rvv_memcpy1d`，将预取开销计入结果；不将与旧版本、旧运行的差异当作候选收益。
若有收益，再进行独立复测、热缓冲区回归和更广尺寸检查，然后才考虑算子集成。
