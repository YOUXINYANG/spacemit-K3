# Elementwise\-fused

## 简介

### Elementwise 

逐元素计算。

以向量加法为例：

```Plain Text
a = [1, 2, 3, 4]
b = [5, 6, 7, 8]

y = a + b
  = [6, 8, 10, 12]
```

每个位置独立计算：

```Plain Text
y[i] = a[i] + b[i]
```

计算 `y[0]` 不需要知道 `y[1]` 的结果，也不需要其他位置的输入。

### SiLU

SiLU 对每个输入做下面的变换：

```math
\mathrm{SiLU}(x)=x\cdot\mathrm{sigmoid}(x) =\frac{x}{1+e^{-x}}
```

虽然公式变复杂了，但不同位置之间仍然互不依赖。可以使用相似的“分块 → 读取 → 计算 → 写回”框架。

### SwiGLU

本教程实现的 SwiGLU 核心计算是：

```math
y_i=\mathrm{SiLU}(g_i)\cdot u_i
```

这里的 `g` 和 `u` 是两个输入张量。

教程把它放在大模型 MLP 的背景下：

```Python
输入 x
  → gate_proj(x) 得到 g
  → up_proj(x)   得到 u

然后计算 silu(g) * u
最后交给 down_proj
```

SwiGLU kernel 负责的是中间的 `silu(g) * u`，两个投影矩阵乘法不包含在其中。

### 融合算子

融合算子节省了什么内容？

假设分两步执行，并把中间张量存入内存：

```Plain Text
t = silu(g)
y = t * u
```

需要的数据流是：

```Plain Text
第一步：读 g → 计算 SiLU → 写 t
第二步：读 t、u → 相乘 → 写 y
```

融合后，一个 kernel 直接完成：

```Plain Text
读 g、u → 计算 silu(g) * u → 写 y
```

主要变化是：中间结果 `t` 不再作为完整张量写入内存，然后又被读回来。

## SwiGLU 完整可运行程序

```Python
import torch
import triton
import triton.language as tl
from triton.backends.spine_triton.driver import CPUDriver

# 激活 spine-triton 的 CPU 后端
triton.runtime.driver.set_active(CPUDriver())


@triton.jit(do_not_specialize=["N"])
def swiglu_kernel(G, U, Y, N, BLOCK: tl.constexpr, MAX_GRID: tl.constexpr):
    pid = tl.program_id(0)

    # 每个 program 覆盖 BLOCK * MAX_GRID 个元素
    base = pid * (BLOCK * MAX_GRID)

    # 每次处理 BLOCK 个元素，共循环 MAX_GRID 次
    for start in tl.range(0, BLOCK * MAX_GRID, BLOCK):
        offs = base + start + tl.arange(0, BLOCK)
        mask = offs < N

        # 安全读取，并在 sigmoid 前提升到 float32
        g = tl.load(G + offs, mask=mask, other=0.).to(tl.float32)
        u = tl.load(U + offs, mask=mask, other=0.).to(tl.float32)

        # 融合计算：SiLU(g) * u
        y = (g * tl.sigmoid(g)) * u

        # 转回输出类型，只写入有效位置
        tl.store(Y + offs, y.to(Y.dtype.element_ty), mask=mask)


def swiglu(g, u):
    # 本例要求 g、u 形状一致且内存连续
    N = g.numel()
    y = torch.empty_like(g)

    BLOCK = 1024
    MAX_GRID = 512
    grid = triton.cdiv(N, BLOCK * MAX_GRID)  # 本例启动 6 个 program

    swiglu_kernel[(grid,)](
        g, u, y, N,
        BLOCK=BLOCK,
        MAX_GRID=MAX_GRID,
    )
    return y


if __name__ == "__main__":
    torch.manual_seed(0)

    # 创建 CPU 上的 float16 输入
    N = 3072 * 1024
    g = torch.randn(N, dtype=torch.float16)
    u = torch.randn(N, dtype=torch.float16)

    y = swiglu(g, u)

    # 参考结果：float32 计算，最后转回 float16
    y_ref = (torch.nn.functional.silu(g.float()) * u.float()).half()

    torch.testing.assert_close(y, y_ref, atol=1e-2, rtol=1e-2)
    print("PASS  max diff =", (y - y_ref).abs().max().item())
```

运行结果：

```Python
PASS  max diff = 0.0009765625
```

## SwiGLU 官方程序讲解

### 核心计算

```Python
y = (g * tl.sigmoid(g)) * u
```

分步拆开看就是：

```Python
s = tl.sigmoid(g)
t = g * s               # SiLU(g)
y = t * u               # SwiGLU
```

### 数据类型变化

f16 → f32 → f16

输入由 host 创建为 `float16`：

```Plain Text
g = torch.randn(N, dtype=torch.float16)
```

kernel 读取后立即提升：

```Plain Text
g = tl.load(...).to(tl.float32)
u = tl.load(...).to(tl.float32)
```

按教程所用的接口约定，`tl.sigmoid` 内部涉及的数学函数要求输入为 fp32/fp64，需要先转换。SiLU 和后续乘法都在 f32 中完成，减少中间步骤的舍入误差。

最后写回时再降精度：

```Plain Text
y.to(Y.dtype.element_ty)
```

`Y.dtype.element_ty` 表示输出指针所指向的元素类型。

## epilogue 融合版本

进一步把逐元素计算接到 GEMM 的末尾。矩阵乘法算完后，直接在累加结果上加 bias、做 SiLU，最后只写出一次。

表达式是：

```math
Y_{m,n}=\mathrm{SiLU}\left((AB)_{m,n}+b_n\right)
```

融合的是 GEMM \+ bias \+ SiLU。它可以对应 MLP 中 gate 分支的一段计算，完整 SwiGLU 还需要与 up 分支的结果相乘。

上述融合省掉了哪些读写？

假设 GEMM 和后续的 bias、SiLU 分成两个 kernel：

```Plain Text
GEMM kernel：
    计算 acc → 写出中间矩阵 T

逐元素 kernel：
    读 T → 加 bias → SiLU → 写 Y
```

融合后：

```Plain Text
GEMM kernel：
    计算 acc → 加 bias → SiLU → 写 Y
```

