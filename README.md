# SpacemiT K3 学习笔记

本项目记录 SpacemiT K3 平台、spine-triton 算子编程与 Qwen 模型部署的学习内容，涵盖平台基础、常见算子的计算原理、实现方式与优化思路，以及模型推理与服务部署实践。

本项目 base K3 云实例进行。

## 阅读建议

先阅读 [K3 平台笔记](ops/K3.md)，了解平台与 spine-triton。再从 GEMM、Reduction 和 Softmax 入手，按需阅读归一化、Attention 等算子笔记。阅读示例时，建议具备 Python、PyTorch 和张量运算的基础知识。

如需在 K3 云实例上部署模型，可阅读 [Qwen 部署笔记](qwen/Qwen%20部署.md)，按顺序完成环境检查、模型下载、推理验证、性能测试与 API 服务启动。

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
| [Cumsum / Scan](<ops/Cumsum%20%3A%20Scan.md>)  | 累积和与前缀扫描               |
| [Cat / Concat](<ops/Cat%20%3A%20Concat.md>)    | 张量拼接与数据搬运             |

### Qwen 模型部署

| 文档 | 内容 |
| ---- | ---- |
| [Qwen 部署笔记](qwen/Qwen%20部署.md) | K3 云实例环境检查、llama.cpp 版本与后端确认、Qwen3-0.6B Q4_0 GGUF 模型推理、性能基线测试与 API 服务部署 |

## 官方 spine-triton 教程

- [spine-triton 教程](https://github.com/spacemit-com/spine-triton-tutorial)：系统学习 spine-triton 基础与算子实践。
