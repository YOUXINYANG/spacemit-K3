# Qwen 部署

## 确认 K3 云实例系统环境

检查操作系统与架构：

```Shell
cat /etc/os-release
uname -m
uname -r
```

检查 CPU 与核心信息：

```Shell
lscpu
```

检查可用内存和磁盘：

```Shell
free -h
df -h / "$HOME"
```

检查是否已经安装 inference 工具和编译工具：

```Shell
for tool in llama-cli llama-server llama-bench gcc g++ cmake git wget curl; do
    command -v "$tool" || true
done
```

## 确认 llama.cpp 版本和后端

查看版本和启动时的后端信息：

```Shell
llama-cli --version 2>&1
```

确认工具的软件包来源：

```Shell
dpkg-query -S /usr/bin/llama-cli /usr/bin/llama-server /usr/bin/llama-bench
```

查看已经安装的相关软件包版本：

```Shell
dpkg-query -W '*llama*' '*spacemit*' 2>/dev/null
```

查看推理进程允许使用的 CPU：

```Shell
grep Cpus_allowed_list /proc/self/status
```

## Qwen3-0.6B

### 跑通推理

创建模型和日志目录：

```Shell
mkdir -p ~/qwen/models ~/qwen/logs
cd ~/qwen
```

下载 Q4_0 量化的 GGUF 模型，支持断点续传：

```Shell
wget -c \
  'https://www.modelscope.cn/models/unsloth/Qwen3-0.6B-GGUF/resolve/master/Qwen3-0.6B-Q4_0.gguf' \
  -O models/Qwen3-0.6B-Q4_0.gguf
```

启动推理，并保留完整日志：

```Shell
llama-cli \
  -m ~/qwen/models/Qwen3-0.6B-Q4_0.gguf \
  -t 8 \
  -c 2048 \
  -n 256 \
  --no-mmap \
  -p '请用中文简要介绍你自己。/no_think' \
  2>&1 | tee ~/qwen/logs/first-run.log
```

这里的参数含义：

* `-t` ：计算线程数；
* `-c` ：上下文容量；
* `-n` ：限制本次生成长度；

Qwen3-0.6B 成功跑通推理，本次显示提示词处理速度和生成速度为：

```
[ Prompt: 244.5 t/s | Generation: 23.3 t/s ]
```

### 测试性能 baseline

执行测试：

```Shell
cd ~/qwen

llama-bench \
  -m ~/qwen/models/Qwen3-0.6B-Q4_0.gguf \
  -t 8 \
  -p 128 \
  -n 128 \
  -r 3 \
  2>&1 | tee ~/qwen/logs/bench-t8.log
```

参数解读：

* `-p 128`：测试处理 128 个输入 token 的速度，即  **prefill** 。
* `-n 128`：测试生成 128 个 token 的速度，即  **decode** 。
* `-r 3`：重复三次，观察结果的稳定性。

测试结果：

```
CPU_RISCV64_SPACEMIT: tcm is available, blk_size: 393216, blk_num: 8, is_fake_tcm: 0
CPU_RISCV64_SPACEMIT: num_cores: 16, num_perfer_cores: 8, perfer_core_arch_id: a064, exclude_main_thread: 0, use_ime1: 0, use_ime2: 1, cpu_mask: ff00
| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
llama_model_load_from_file_impl: llama_model_load took 1.92 seconds
| qwen3 0.6B Q4_0                | 358.78 MiB |   596.05 M | CPU        |       8 |           pp128 |        279.06 ± 0.83 |
| qwen3 0.6B Q4_0                | 358.78 MiB |   596.05 M | CPU        |       8 |           tg128 |         39.73 ± 0.01 |
```

baseline：

* Prefill（`pp128`）：**279.06 ± 0.83 token/s**
* Decode（`tg128`）：**39.73 ± 0.01 token/s**

## 启动 Qwen API 服务

首先在当前 terminal 中运行：

```Shell
cd ~/qwen

llama-server \
  -m ~/qwen/models/Qwen3-0.6B-Q4_0.gguf \
  -t 8 \
  -c 2048 \
  --n-gpu-layers 0 \
  --host 127.0.0.1 \
  --port 8080 \
  --no-mmap \
  2>&1 | tee ~/qwen/logs/server.log
```

然后新建一个新的 terminal，检查服务是否建立：

```Shell
curl -sS http://127.0.0.1:8080/health
```

模型加载完成后，预期返回包含 `"status":"ok"` 的结果。接着发送一次对话请求：

```Shell
curl -sS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3-0.6B",
    "messages": [
      {"role": "user", "content": "请用中文一句话解释什么是算子优化。/no_think"}
    ],
    "max_tokens": 256,
    "temperature": 0,
    "stream": false
  }'
```
