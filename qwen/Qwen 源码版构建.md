# Qwen 源码版构建

## 源码版

从源码构建可以让我们能够追溯、修改并验证推理程序，自己掌握程序的构建过程。

## 工作概览

完成 SpacemiT llama.cpp 源码版本固定、CMake 配置、工具编译、Qwen 推理验证、性能基线测试和核心调度检查，为后续算子 profiling 与优化建立实验环境。

以下命令均在 **K3 云实例**上执行，路径以用户 `bianbu` 的主目录为准。

## 实验环境与源码版本

- 平台：K3 云实例，16GB 内存、128GB 磁盘；Bianbu 4.0，内核 `6.18.3-generic`。
- 工具：GCC/G++ 15.2.0、CMake 4.2.3、Ninja 1.13.2、pkg-config 2.5.1。
- 模型：`Qwen3-0.6B-Q4_0.gguf`，模型权重大小 358.78 MiB。
- 仓库：[https://github.com/spacemit-com/llama.cpp](https://github.com/spacemit-com/llama.cpp)。
- 源分支：`add-spacemit-backend-ime2`；本地研究分支：`k3-qwen-baseline`。
- 固定提交：`58745bdb86b759940577e0a8226cfbfa9987c286`。
- 源码版程序版本：`9083 (58745bdb)`。

预装版提交 `a5b514468` 未能在本次检查的官方仓库和上游仓库中定位，因此采用公开源码重新建立独立基线。

关键版本记录命令（创建分支的命令仅需首次执行）：

```bash
cd ~/qwen/src/llama.cpp-spacemit
git switch -c k3-qwen-baseline origin/add-spacemit-backend-ime2
mkdir -p ~/qwen/logs
git rev-parse HEAD | tee ~/qwen/logs/source-commit.txt
git log -1 --format=fuller
git status --short --branch
```

## 配置与编译

使用实例中的 GCC/G++ 本机编译，构建目录独立于源码目录。

```bash
cd ~/qwen/src/llama.cpp-spacemit
mkdir -p ~/qwen/builds ~/qwen/logs
set -o pipefail

cmake -S . -B ~/qwen/builds/ime2-baseline -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/gcc \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DGGML_NATIVE=OFF \
  -DGGML_CPU_RISCV64_SPACEMIT=ON \
  -DGGML_CPU_REPACK=OFF \
  -DGGML_RVV=ON \
  -DGGML_RV_ZFH=ON \
  -DGGML_RV_ZVFH=ON \
  -DGGML_RV_ZICBOP=ON \
  -DGGML_RV_ZIHINTPAUSE=ON \
  -DGGML_RV_ZBA=ON \
  -DGGML_OPENMP=ON \
  -DLLAMA_OPENSSL=OFF \
  -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_WEBUI=OFF \
  2>&1 | tee ~/qwen/logs/configure-ime2-baseline.log

cmake --build ~/qwen/builds/ime2-baseline \
  --target llama-cli llama-bench llama-server \
  --parallel 4 \
  2>&1 | tee ~/qwen/logs/build-ime2-baseline.log
```

配置检测通过，包含 IME1/IME2 编译支持、RVV 和 OpenMP。三个目标程序均编译成功。

名词解释：

* IME：Integrated Matrix Extension，集成矩阵扩展，通过专门的矩阵指令，提高矩阵乘法等计算的效率。
* RVV：RISC-V Vector Extension，RISC-V 向量扩展，让 CPU 用一条向量指令，对多个数据执行相同操作，例如同时计算多组加法、乘法。它可以加速向量计算，以及模型中的部分量化、数据转换和矩阵计算。
* OpenMP：共享内存并行编程接口。开发者通过编译指令和运行库，把计算任务分给多个线程，让多个 CPU 核心同时执行。

验证程序版本与动态库来源：

```bash
~/qwen/builds/ime2-baseline/bin/llama-cli --version \
  2>&1 | tee ~/qwen/logs/source-version.log

ldd ~/qwen/builds/ime2-baseline/bin/llama-bench
ls -l /dev/tcm* /dev/hugetlb_1g 2>&1
```

结果：程序运行时 `use_ime2: 1`。相关 llama/GGML 动态库来自本次构建目录。

## 运行环境差异

实例存在 `/dev/tcm`，后端检测到真实 TCM 可用，但不存在 `/dev/tcm_sync_mem`，出现以下日志：

```text
mem_backend: HPAGE
alloc_chunk: open(/dev/tcm_sync_mem) failed, errno=2
failed to allocate init_barrier from shared mem, falling back to heap
```

源码中的初始化同步屏障申请失败后回退到堆内存。本次短测和标准测试均正常结束，该回退对性能的影响尚未单独测量。`/dev/hugetlb_1g` 也不存在，但当前选择的是 HPAGE 路径。

## 推理与性能验证

短测使用 `-p 32 -n 16 -r 1`，退出码为 0，得到 pp32 **279.47 token/s**、tg16 **42.05 token/s**。仅重复一次，不能据此判断波动。

标准性能测试命令：

```bash
set -o pipefail
~/qwen/builds/ime2-baseline/bin/llama-bench \
  -m ~/qwen/models/Qwen3-0.6B-Q4_0.gguf \
  -t 8 -p 128 -n 128 -r 3 \
  2>&1 | tee ~/qwen/logs/source-bench-t8.log
echo "测试退出码：$?"
```

| 测试  |       预装版 a5b514468 |        源码版 58745bdb |
| ----- | ---------------------: | ---------------------: |
| pp128 | 279.06 ± 0.83 token/s | 319.06 ± 0.97 token/s |
| tg128 |  39.73 ± 0.01 token/s |  42.37 ± 0.08 token/s |

源码版标准测试退出码为 0。两版存在源码、编译器和内存管理路径等差异，不能将速度变化归因于某一算子。后续优化以源码版为对照。

文本生成验证命令：

```bash
~/qwen/builds/ime2-baseline/bin/llama-cli \
  -m ~/qwen/models/Qwen3-0.6B-Q4_0.gguf \
  -t 8 -c 2048 -n 256 --no-mmap \
  -p '请用中文简要介绍你自己。/no_think' \
  2>&1 | tee ~/qwen/logs/source-generation.log
```

程序生成了连贯的中文回答，显示 Prompt **287.8 token/s**、Generation **27.5 token/s**。这是功能检查，不能替代数值正确性测试，也不与 benchmark 直接比较。

## 核心调度验证

后台启动诊断任务并采集线程信息的关键命令：

```bash
~/qwen/builds/ime2-baseline/bin/llama-bench \
  -m ~/qwen/models/Qwen3-0.6B-Q4_0.gguf \
  -t 8 -p 128 -n 256 -r 3 \
  > ~/qwen/logs/core-check-bench.log 2>&1 &
k3_bench_pid=$!

sleep 2
for k3_status_file in /proc/"$k3_bench_pid"/task/*/status; do
  if [ -r "$k3_status_file" ]; then
    awk '/^(Name|Pid|Cpus_allowed_list):/' "$k3_status_file"
  fi
done

for k3_sample in 1 2 3; do
  ps -L -p "$k3_bench_pid" -o pid,tid,psr,pcpu,comm
  sleep 2
done

wait "$k3_bench_pid"
echo "诊断进程退出码：$?"
```

首次采集仅见主线程，允许核心为 0–7。后两次采样显示，包含主线程在内的 8 个线程分别在 CPU **8–15** 执行，与后端优选掩码 `ff00` 一致。未采集到计算期间各线程的亲和性列表，因此实际运行位置以 PSR 采样结果为依据。

诊断任务退出码为 0，pp128 为 **318.58 ± 1.30 token/s**，tg256 为 **42.04 ± 0.03 token/s**。该任务用于调度观察，不替代标准性能基线。
