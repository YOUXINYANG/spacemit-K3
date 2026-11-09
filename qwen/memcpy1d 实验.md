# memcpy1d 实验

## 目标与环境

 **`memcpy1d` 通常指“一维内存拷贝”函数** ，用于把一段连续的数据从源地址复制到目标地址，例如复制一个数组或缓冲区。这里的“一维”表示按连续字节复制，相对地，二维拷贝通常按行复制，还需要指定行数、每行长度和行间跨度。

此前发现 Qwen 生成阶段的 `memcpy1d` 是热点。上一阶段将它单独拿出来测试，了解数据搬入 TCM 的耗时，以及地址对齐和缓存条件对性能的影响，为后续优化提供依据。

| 项目           | 配置                                                         |
| -------------- | ------------------------------------------------------------ |
| 平台           | K3，Bianbu 4.0，riscv64，GCC 15.2.0                          |
| llama.cpp 源码 | `58745bdb86b759940577e0a8226cfbfa9987c286`                 |
| 线程与存储     | 单线程固定 CPU 8，真实 TCM 块 0，块大小 393216 字节          |
| 复制大小       | 16384、18432、20480 字节                                     |
| 地址条件       | 源地址按 128 字节对齐；目标地址除以 128 的余数为 0 或 8      |
| 计时方法       | 每组测三轮，交替函数顺序；申请内存、初始化和正确性检查不计时 |

程序直接链接现有 `libggml-cpu.so.0`，没有修改模型使用的算子库。

## 确认接口与构建

以下命令在 K3 上执行：

```bash
cd ~/qwen/src/llama.cpp-spacemit
git rev-parse HEAD
git status --short
git grep -n 'memcpy1d' -- ggml/src/ggml-cpu/spacemit/rvv_kernels.h tests
nm -D -C ~/qwen/builds/ime2-baseline/bin/libggml-cpu.so.0 \
  | grep 'spacemit_kernels::rvv::memcpy1d'
```

结果：源码版本正确、工作区干净，动态库已导出函数：

```cpp
void memcpy1d(void *dst, const void *src, int64_t size);
```

参考系统的 TCM 申请和释放示例：

```bash
cat /usr/share/spine_tcm/examples/multithread_direct_link/main.c
cat /usr/share/spine_tcm/examples/multithread_direct_link/CMakeLists.txt
```

测试源码在本地 `qwen/benchmarks/tcm-copy/`，上传后位于 K3 的 `~/qwen/tcm-copy/`：

```bash
cmake -S ~/qwen/tcm-copy -B ~/qwen/builds/tcm-copy -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DK3_LLAMA_BIN="$HOME/qwen/builds/ime2-baseline/bin"
cmake --build ~/qwen/builds/tcm-copy --parallel 2
ctest --test-dir ~/qwen/builds/tcm-copy --output-on-failure
```

初版的数据检查器测试通过；加入源数据轮转后，两项辅助测试均通过。

## CPU 绑定问题与正确性验证

初版直接绑定 CPU 8，出现：

```text
Cannot bind CPU 8; no TCM block requested: Invalid argument
```

程序退出码为 1，尚未申请 TCM，8 个块仍全部空闲。

核对固定版本 `ime.cpp` 后发现漏了平台专用步骤：程序需要在必要时先向 `/proc/set_ai_thread` 写入 `0`，再设置 CPU 亲和性。补齐后，运行成功：

```bash
mkdir -p ~/qwen/logs
set -o pipefail
env SPACEMIT_DISABLE_TCM=0 ~/qwen/builds/tcm-copy/tcm-copy \
  2>&1 | tee ~/qwen/logs/tcm-copy-check-v2.log
k3_copy_rc=$?
echo "正确性测试退出码：$k3_copy_rc"
spacemit-tcm-smi
```

结果：CPU 为 8，真实 TCM；六组检查均 `PASS`，覆盖两个复制函数及三种数据模式，并检查目标区域前后的越界写。最后输出：

```text
tcm_free_result=0 test_result=0
正确性测试退出码：0
available_blocks=8/8
```

启动时仍有 `/dev/tcm_sync_mem` 缺失、屏障回退到堆内存的提示，本次测试在该回退路径下成功。

## 热缓冲区测试

反复读取同一源块，源数据很可能驻留缓存。首轮每组复制 20000 次：

```bash
env SPACEMIT_DISABLE_TCM=0 ~/qwen/builds/tcm-copy/tcm-copy --bench \
  > ~/qwen/logs/tcm-copy-bench.csv \
  2> ~/qwen/logs/tcm-copy-bench.stderr.log
```

三轮平均结果如下；“耗时比”为标准库耗时除以 RVV 耗时：

| 大小  | 目标余数 | RVV 耗时/ns | 标准库耗时/ns | 耗时比 |
| ----- | -------: | ----------: | ------------: | -----: |
| 16KiB |        0 |       459.4 |        1389.8 |   3.03 |
| 16KiB |        8 |       489.4 |        1391.4 |   2.84 |
| 18KiB |        0 |       526.7 |        1561.9 |   2.97 |
| 18KiB |        8 |       515.8 |        1560.6 |   3.03 |
| 20KiB |        0 |       546.0 |        1733.2 |   3.17 |
| 20KiB |        8 |       595.9 |        1733.6 |   2.91 |

部分 RVV 耗时有波动，因此把每轮次数增加到 200000，再独立运行三次：

```bash
sed -i 's/iterations = 20000;/iterations = 200000;/' ~/qwen/tcm-copy/main.cpp
cmake --build ~/qwen/builds/tcm-copy --parallel 2

for k3_run in 1 2 3; do
  env SPACEMIT_DISABLE_TCM=0 ~/qwen/builds/tcm-copy/tcm-copy --bench \
    > "$HOME/qwen/logs/tcm-copy-long-${k3_run}.csv" \
    2> "$HOME/qwen/logs/tcm-copy-long-${k3_run}.stderr.log"
  k3_copy_rc=$?
  echo "第 ${k3_run} 次退出码：$k3_copy_rc"
  spacemit-tcm-smi
  if [ "$k3_copy_rc" -ne 0 ]; then break; fi
done
grep -H ',18432,0,8,' ~/qwen/logs/tcm-copy-long-*.csv
```

三次均退出 0，TCM 均恢复 8/8。18KiB、目标余数为 8 的九个样本：

| 指标                      |              RVV |      标准库 memcpy |
| ------------------------- | ---------------: | -----------------: |
| 平均耗时/ns               |          455.776 |           1563.344 |
| 耗时范围/ns               | 418.900–478.730 | 1562.594–1564.774 |
| 按平均耗时计算的吞吐/GB/s |           40.441 |             11.790 |

标准库耗时约为 RVV 的 **3.43 倍**；RVV 的样本变异系数约 4%，仍有一定波动。当前结果没有显示目标偏移 8 字节会造成明显降速。

## 大源数据集轮转测试

增加 `--stream`：申请 64MiB 源缓冲区，分块顺序读取，读到末尾后重新开始。18KiB 时使用 **3640 块，共 67092480 字节**。

先运行 `tcm-copy --stream`，六组正确性检查通过；然后使用同一版程序比较两种模式。关键计时命令为：

```bash
env SPACEMIT_DISABLE_TCM=0 ~/qwen/builds/tcm-copy/tcm-copy --bench \
  > ~/qwen/logs/tcm-copy-v3-hot.csv \
  2> ~/qwen/logs/tcm-copy-v3-hot.stderr.log

env SPACEMIT_DISABLE_TCM=0 ~/qwen/builds/tcm-copy/tcm-copy --bench --stream \
  > ~/qwen/logs/tcm-copy-v3-stream.csv \
  2> ~/qwen/logs/tcm-copy-v3-stream.stderr.log

grep -H ',18432,0,8,' ~/qwen/logs/tcm-copy-v3-*.csv
```

实际每次运行后均检查退出码和 TCM：均为 0、8/8 空闲。18KiB、目标余数为 8 的三轮平均结果：

| 模式       | RVV 耗时/ns | RVV 吞吐/GB/s | 标准库耗时/ns | 标准库吞吐/GB/s |
| ---------- | ----------: | ------------: | ------------: | --------------: |
| 热缓冲区   |     460.951 |        39.987 |      1566.380 |          11.767 |
| 源数据轮转 |     950.441 |        19.393 |      1715.765 |          10.743 |

轮转模式下 RVV 耗时约为热缓冲区的 **2.06 倍**；标准库耗时仍约为 RVV 的 **1.81 倍**。源数据的缓存条件对本实验影响明显。

## 实验结论

* **现有 RVV 复制优于标准库 `memcpy`** ，目前没有依据直接替换成标准库函数。
* **源数据的缓存条件影响明显** ：18KiB 拷贝从热缓冲区约 **461ns** 增加到大数据集轮转约  **950ns** ，说明数据搬运值得继续优化。
* **地址偏移 8 字节未表现出明显降速** ，暂不作为主要优化方向。
