# SpacemiT K3 学习笔记

本仓库记录在 SpacemiT K3 云实例上的学习与实践，涵盖 K3 平台环境、spine-triton 算子编程、Qwen 模型部署、源码构建、性能分析与 TCM 对照实验，并收录推理优化相关的论文笔记和毕设研究计划。

## 阅读建议

先阅读 [K3 平台笔记](ops/K3.md)，了解平台与 spine-triton。再从 GEMM、Reduction 和 Softmax 入手，按需阅读归一化、Attention 等算子笔记。阅读示例时，建议具备 Python、PyTorch 和张量运算的基础知识。

如需在 K3 云实例上部署模型，可先阅读 [Qwen 预装版部署笔记](qwen/Qwen%20预装版构建.md)，按顺序完成环境检查、模型下载、推理验证、性能测试与 API 服务启动。随后阅读 [Qwen 源码版构建与基线验证](qwen/Qwen%20源码版构建.md)，了解版本固定、编译配置、推理验证及核心调度检查，为后续性能分析与优化建立源码基线。

建立源码基线后，可继续阅读 [Qwen 性能分析与 TCM 对照实验](qwen/性能分析与TCM对照实验.md)，了解 prefill 与 decode 的 perf 热点采样、源码定位、调用栈分析的工具限制，以及 TCM 占用恢复和开关对照结果。

关注推理性能时，可从 [SMEPilot 论文解读](papers-study/SMEPilot.md) 了解 prefill 和 decode 的性能瓶颈，再参照 [毕设课题研究计划](qwen/毕设课题研究计划.md) 查看 K3 上的源码分析、性能测量与优化实验安排。

## 文档索引

### 平台与资料

| 文档                                            | 内容                                 |
| ----------------------------------------------- | ------------------------------------ |
| [K3 平台笔记](ops/K3.md)                         | K3 平台、spine-triton 与相关学习资料 |
| [K3 中文数据手册](datasheet/k3_datasheet_zh.pdf) | 芯片规格与硬件参考资料（PDF）        |

### 算子笔记

| 算子                                          | 内容                           |
| --------------------------------------------- | ------------------------------ |
| [GEMM](ops/GEMM.md)                            | 通用矩阵乘法的计算原理与实现   |
| [LayerNorm](ops/LayerNorm.md)                  | 层归一化：均值、方差与缩放平移 |
| [RMSNorm](ops/RMSNorm.md)                      | 均方根归一化的计算与实现       |
| [Softmax](ops/Softmax.md)                      | 将分数归一化为权重的计算过程   |
| [GEMV](ops/GEMV.md)                            | 矩阵与向量乘法                 |
| [Reduction](ops/Reduction.md)                  | 求和、最大值等归约运算         |
| [Elementwise 与融合](ops/Elementwise-fused.md) | 逐元素运算与算子融合           |
| [Attention](ops/Attention.md)                  | Q、K、V 与注意力计算           |
| [Cumsum / Scan](ops/Cumsum%20%26%20Scan.md)     | 累积和与前缀扫描               |
| [Cat / Concat](ops/Cat%20%26%20Concat.md)       | 张量拼接与数据搬运             |

### Qwen 模型部署、源码构建与性能分析

| 文档 | 内容 |
| ---- | ---- |
| [Qwen 预装版部署笔记](qwen/Qwen%20预装版构建.md) | K3 云实例环境检查、预装 llama.cpp 版本与后端确认、Qwen3-0.6B Q4_0 GGUF 模型推理、性能基线测试与 API 服务部署 |
| [Qwen 源码版构建与基线验证](qwen/Qwen%20源码版构建.md) | SpacemiT llama.cpp 版本固定、CMake 配置与编译、运行环境差异、推理与性能验证、预装版基线对照及核心调度检查 |
| [Qwen 性能分析与 TCM 对照实验](qwen/性能分析与TCM对照实验.md) | prefill/decode 热点采样与源码定位、DWARF 回溯限制、GDB 复制路径观察、TCM 占用恢复及开关对照实验 |

### 论文解读与研究计划

| 文档 | 内容 |
| ---- | ---- |
| [SMEPilot 论文解读](papers-study/SMEPilot.md) | 可伸缩矩阵扩展下的 LLM 推理分析与优化，涵盖 prefill 与 decode 瓶颈、Roofline 模型、矩阵分块、注意力流水线与数据布局 |
| [毕设课题研究计划](qwen/毕设课题研究计划.md) | Qwen 推理流程、K3 上的 llama.cpp 调用路径、热点分析、算子优化与端到端验证 |

## 官方 spine-triton 教程

- [spine-triton 教程](https://github.com/spacemit-com/spine-triton-tutorial)：系统学习 spine-triton 基础与算子实践。
