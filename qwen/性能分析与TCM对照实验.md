# Qwen 性能分析与 TCM 对照实验

## 实验环境

- K3 云实例，Bianbu 4.0，16GB 内存。
- 模型：Qwen3-0.6B Q4_0，8 线程。
- 源码提交：`58745bdb86b759940577e0a8226cfbfa9987c286`。
- 测速程序：`~/qwen/builds/ime2-baseline/bin/llama-bench`。
- 分析程序：`~/qwen/builds/ime2-profile/bin/llama-bench`。
- 命令均在 K3 实例执行；测试时停止其他推理任务。

## 检查 perf

perf 是 Linux 下的性能分析工具。用来回答：程序慢在哪里？CPU 时间主要花在哪些函数上？缓存和分支预测效率如何？

它通过 Linux 内核的性能事件接口，结合 CPU 硬件计数器等机制采集数据。常见用法：

- perf stat：统计运行时间、CPU 周期、指令数、缓存未命中等指标。
- perf record：采样程序运行情况，记录热点函数和调用栈。
- perf report：查看采样结果，找出耗时最多的代码。
- perf top：实时观察热点，类似性能分析版的 top。

```bash
perf version --build-options
perf stat -e task-clock -- sleep 1
perf stat -e cycles,instructions -- sleep 1
```

结果：perf 版本为 `6.18.3.g27275ec8240c`，软件和硬件事件探测均返回 0。已启用 libdw 和 DWARF 回溯支持。

`sleep` 仅用于检查权限，不用于评价性能。

**DWARF 是一种调试信息格式** 。编译器可以把源代码与机器指令的对应关系、函数名、变量类型、调用栈展开信息等，以 DWARF 格式写入程序文件或独立的调试文件。调试器和性能分析工具利用这些信息，把机器地址还原成更容易理解的源码位置。编译时常用 `-g` 生成调试信息。

**libdw 是读取和处理 DWARF 调试信息的库** ，属于 elfutils 项目。它为工具提供解析接口，用来查询函数、源码行号、变量信息等，也可配合相关接口进行调用栈展开。

## 分别采样 prefill 和 decode

Prefill 是输入处理阶段，decode 是逐 token 生成阶段。使用软件时钟采样用户态 CPU 时间，分别观察热点。

```bash
mkdir -p ~/qwen/profiles ~/qwen/logs
set -o pipefail

perf record -e cpu-clock:u -F 99 \
  -o ~/qwen/profiles/prefill.data \
  -- ~/qwen/builds/ime2-baseline/bin/llama-bench \
  -m ~/qwen/models/Qwen3-0.6B-Q4_0.gguf \
  -t 8 -p 128 -n 0 -r 50 \
  2>&1 | tee ~/qwen/logs/profile-prefill.log

perf record -e cpu-clock:u -F 99 \
  -o ~/qwen/profiles/decode.data \
  -- ~/qwen/builds/ime2-baseline/bin/llama-bench \
  -m ~/qwen/models/Qwen3-0.6B-Q4_0.gguf \
  -t 8 -p 0 -n 128 -r 8 \
  2>&1 | tee ~/qwen/logs/profile-decode.log

perf report --stdio --no-children --sort=dso,symbol \
  --percent-limit 1 -i ~/qwen/profiles/decode.data
```

查看 prefill 时，将报告输入文件换成 `prefill.data`。

结果：prefill 获得 **16607** 个样本，decode 获得 **19804** 个样本；两次退出码均为 0，报告均显示丢失样本为 0。

| 阶段    | 热点及后续定位结果                            | 采样占比 |
| ------- | --------------------------------------------- | -------: |
| Prefill | RVV 矩阵乘 `tinyBLAS_RVV…gemm`             |   36.51% |
| Prefill | IME2 内核 `gemm_kernel_i8i4_hp_m4` 内部循环 |   27.10% |
| Decode  | 自定义复制函数 `rvv::memcpy1d` 内部循环     |   32.36% |
| Decode  | FP16 点积 `ggml_vec_dot_f16`                |   15.04% |
| Decode  | IME2 内核 `gemm_kernel_i8i4_hp_m1` 内部循环 |    8.28% |

这些占比近似反映用户态 CPU 时间分布，不是模型总延迟占比。采样覆盖加载、预热和正式计算，尚未完全排除初始化影响。

## 将热点定位到源码

perf 中的部分名称是汇编内部标签，不能直接当作独立函数。

```bash
cd ~/qwen/src/llama.cpp-spacemit

git grep -n -E 'memcpy_main_loop|KsubBLK|INNER_BLK_LOOP' -- ggml
git grep -n -E 'tinyBLAS_RVV|ggml_vec_dot_f16' -- ggml

nm -n -S -C --defined-only \
  ~/qwen/builds/ime2-baseline/bin/libggml-cpu.so \
  > ~/qwen/profiles/cpu-symbols.txt

grep -B 6 -A 6 -E \
  'memcpy_main_loop34|_KsubBLK_LPST979|INNER_BLK_LOOP756' \
  ~/qwen/profiles/cpu-symbols.txt
```

主要文件：

- `ggml/src/ggml-cpu/spacemit/rvv_kernels.cpp`：复制内核。
- `ggml/src/ggml-cpu/spacemit/ime2_kernels.cpp`：量化矩阵乘内核。
- `ggml/src/ggml-cpu/spacemit/ime.cpp`：算子调度、TCM 搬运和线程绑定。
- `ggml/src/ggml-cpu/llamafile/sgemm.cpp`：RVV 矩阵乘。

## 调用栈分析与工具问题

另外构建了分析版：沿用 baseline 配置和 Release 优化，新增以下两项，将构建目录设为 `~/qwen/builds/ime2-profile`。

```text
-DCMAKE_C_FLAGS_RELEASE="-O3 -DNDEBUG -g"
-DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG -g"
```

```bash
cmake --build ~/qwen/builds/ime2-profile \
  --target llama-bench --parallel 4

readelf -S -W ~/qwen/builds/ime2-profile/bin/libggml-cpu.so \
  | grep -E '\.debug_info|\.debug_line|\.eh_frame'
```

调试信息生成成功，但该版本 `llama-bench` 不支持 `--version`，实际提交以测试输出末尾的 `build` 信息确认。

尝试 DWARF 调用栈采样：

```bash
perf record -e cpu-clock:u -F 99 --call-graph dwarf,16384 \
  -o ~/qwen/profiles/decode-callgraph.data \
  -- ~/qwen/builds/ime2-profile/bin/llama-bench \
  -m ~/qwen/models/Qwen3-0.6B-Q4_0.gguf \
  -t 8 -p 0 -n 128 -r 8
```

记录了 **19921** 个样本，退出码为 0，但反复出现 IP/SP 寄存器识别错误：

```text
Fail to find SP register for arch riscv64, returns 0
Fail to find IP register for arch riscv64, returns 0
```

因此不采用该数据的调用栈结论。问题与 perf 的架构名称兼容性有关的可能性较高，尚未修复；不能简单归因于未安装 libunwind。

关闭调用栈后，平面报告仍可正常读取：

```bash
perf report --stdio --no-children --call-graph none \
  --sort=dso,symbol --percent-limit 1 \
  -i ~/qwen/profiles/decode-callgraph.data
```

分析版复制循环占比 **32.61%**，FP16 点积占比 **14.26%**，与此前热点趋势一致。

随后使用 GDB 在 `spacemit_kernels::rvv::memcpy1d` 入口观察调用现场，关键检查命令为 `bt 12` 和 `info args`。三个样本均来自同一 Q4_0 矩阵乘算子的不同线程，每次复制 **18432 字节（18KiB）**：

```text
GGML 计算图 → Q4_0 forward_mul_mat → ime.cpp:446 → memcpy1d
```

该源码位置将量化权重块搬入 TCM。前三次调用证明路径存在，但不能代表整个 decode 阶段。

## TCM 残留占用与恢复

TCM（Tightly Coupled Memory，紧耦合内存）是直接连接到处理器内核、访问延迟低且较为确定的一小块高速存储器，通常由片上 SRAM 实现。

它常用于存放对速度或实时性要求高的代码和数据，例如中断处理程序、关键计算函数和频繁访问的数据。

它与常见内存的区别是：

- 相比 DDR：容量较小，但访问通常更快、延迟更稳定。
- 相比 Cache：存放什么通常由软件显式安排，更容易控制，Cache 通常由硬件自动管理。

原 GDB 脚本在断点处直接退出，终止了被调试程序。随后 TCM 开启测试出现 `wait tcm buffer failed`，退出码为 **134**。崩溃后的 ptrace 错误属于回溯过程，最先需要处理的是 TCM 获取失败。

检查发现：无残留推理进程，但 8 个 TCM 块全部 busy，登记的线程编号与此前 GDB 调试线程对应。

```bash
ps -eo user,pid,ppid,stat,comm,args \
  | grep -E '[l]lama|[g]db|[p]erf'
sudo fuser -v /dev/tcm
spacemit-tcm-smi -v
spacemit-tcm-smi
```

确认无活跃占用者后，以普通用户恢复：

```bash
spacemit-tcm-smi -c
spacemit-tcm-smi
```

结果为 `force released 8/8 blocks`，状态恢复到 `available_blocks=8/8`。恢复后 TCM 开启短测成功，tg16 为 **42.74 token/s**，退出码为 0。

## TCM 开关对照结果

使用同一 baseline 程序，按“开启 → 关闭 → 再开启”测试，每组重复 5 次。每组结束后查询 TCM 状态。

```bash
set -o pipefail
for k3_case in on1 off on2; do
  if [ "$k3_case" = off ]; then
    k3_disable_tcm=1
  else
    k3_disable_tcm=0
  fi

  timeout 120s env SPACEMIT_DISABLE_TCM="$k3_disable_tcm" \
    ~/qwen/builds/ime2-baseline/bin/llama-bench \
    -m ~/qwen/models/Qwen3-0.6B-Q4_0.gguf \
    -t 8 -p 0 -n 128 -r 5 \
    2>&1 | tee "$HOME/qwen/logs/tcm-retry-${k3_case}-bench.log"

  k3_test_rc=$?
  echo "测试退出码：$k3_test_rc"
  spacemit-tcm-smi
  if [ "$k3_test_rc" -ne 0 ]; then break; fi
done
```

| 测试组   | tg128（token/s） | 退出码 | 测试后空闲块 |
| -------- | ---------------: | -----: | ------------ |
| 开启 on1 |    42.05 ± 0.04 |      0 | 8/8          |
| 关闭 off |    39.40 ± 0.05 |      0 | 8/8          |
| 开启 on2 |    42.12 ± 0.07 |      0 | 8/8          |

两次开启结果相差约 **0.17%**；开启的平均吞吐为 **42.085 token/s**，比关闭高约 **6.81%**。

## 阶段结论与下一步

1. Decode 中的复制循环是稳定的热点，已确认存在 Q4_0 矩阵乘向 TCM 搬运权重的路径。
2. 当前配置下，开启 TCM 的整条执行路径更快，应保留其收益，再研究搬运成本。开关还影响计算组织和线程绑定，不能把差异全部归因于 TCM 存储速度。
3. 尚未修改算子代码，也未取得算子优化成果。采样比例不能直接换算成可实现的加速比例。
4. 下一步设计复制微基准：使用真实 TCM，覆盖 18KiB 及附近尺寸、地址对齐、单线程和多线程，检查数据一致性及资源正常释放。
