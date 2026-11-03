# RMSNorm

## 简介

Root Mean Square Normalization，均方根归一化。

作用：根据一组数据的均方根大小，对数据进行缩放，让数值尺度更加稳定。

计算公式：

$\boxed{y_i=\frac{x_i}{\sqrt{\frac{1}{N}\sum_{j=1}^{N}x_j^2+\epsilon}}\cdot w_i}$

- $x_i$：输入向量的第 $i$个元素

- $N$：输入向量的元素数量

- $\epsilon$：防止分母为零的极小常数，通常为 $10^{-5}$

- $w_i$：可学习的缩放参数（weight）

- $y_i$：输出向量的第 $i$个元素

分为三步计算：

1. 计算均方值：$ms=\frac{1^2+2^2+3^2+4^2}{4}=7.5$

2. 计算归一化系数：$rstd=\frac{1}{\sqrt{7.5}}\approx0.3651$

3. 逐元素缩放：$y=x\cdot rstd\cdot w$

均方值 ms 是一个标量，整行数据共同计算得到一个值。

权重 w 是一个向量，每个元素都有自己的缩放参数。

### **RMSNorm 和 LayerNorm**

LayerNorm 的计算公式：

$y_i=\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}\gamma_i+\beta_i$

RMSNorm 的计算公式：

$y_i=\frac{x_i}{\sqrt{ms+\epsilon}}w_i$

RMSNorm 不需要减去均值，减少了一次归约计算，也简化了 kernal 实现。

对于大语言模型而言，RMSNorm 在许多应用中可以实现与 LayerNorm 相近的训练效果，同时减少计算量，因此被广泛采用。

## fused\_add\_rmsnorm

在 Transformer 模型中，RMSNorm 经常与残差加法一起出现。

例如：

```Plain Text
residual = residual + x
y = rmsnorm(residual, weight)
```

这里有两个操作：

- Add：将输入 `x` 与之前保留的残差 `residual` 相加。作用：保留原始信息，并帮助深层神经网络更稳定地训练。

- RMSNorm：对相加后的结果进行归一化。

把这两个操作融合进同一个 Kernel，就叫做：

fused\_add\_rmsnorm = Add \+ RMSNorm 算子融合。

```plaintext
未融合：两个 Kernel                  融合：一个 Kernel

      x + residual                         x + residual
           |                                    |
           v                                    v
+------------------------+           +------------------------+
| Kernel 1: Add          |           | Fused Kernel           |
+------------------------+           |                        |
           |                         |          Add           |
           v                         |           |            |
      写入中间结果                   |           v            |
           |                         |        RMSNorm         |
           v                         +------------------------+
+------------------------+                      |
| 内存                   |                      v
+------------------------+           输出 y 和更新后的 residual
           |
        再次读取
           |
           v
+------------------------+
| Kernel 2: RMSNorm      |
+------------------------+
           |
           v
         输出 y
```

## Pytorch 实现

```Python
import torch

x = torch.tensor([1., 2., 3., 4.])
residual = torch.ones(4)
w = torch.ones(4)
eps = 1e-5

# 残差加法
r_new = residual + x

# RMSNorm
ms = torch.mean(r_new * r_new)
rstd = torch.rsqrt(ms + eps)
y = r_new * rstd * w

print("Input x:", x)
print("Updated residual:", r_new)
print("Mean Square:", ms.item())
print("Output y:", y)
```

输出：

```Python
Input x: tensor([1., 2., 3., 4.])
Updated residual: tensor([2., 3., 4., 5.])
Mean Square: 13.5
Output y: tensor([0.5443, 0.8165, 1.0887, 1.3608])
```

## Triton 实现

### 官方代码

rmsnorm\.py

```Python
# rmsnorm.py — 每 program 一行的 RMSNorm forward
import torch
import triton
import triton.language as tl
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())


@triton.jit
def rmsnorm_kernel(X, Y, W, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    X += row * stride
    Y += row * stride

    # ── 趟1: sum(x²)，f32 累加器 ──────────────────────────
    _acc = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        # other=0. 在这里是安全的：0² = 0，越界 lane 天然不贡献
        # （对比 layernorm 需要 tl.where 归零——平方对 0 是“吸收”的）
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _acc += x * x
    ms = tl.sum(_acc, axis=0) / N              # mean square，标量
    rstd = 1 / tl.sqrt(ms + eps)

    # ── 趟2: y = x * rstd * w ─────────────────────────────
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        w = tl.load(W + cols, mask=mask)
        tl.store(Y + cols, (x * rstd * w).to(Y.dtype.element_ty), mask=mask)


def rmsnorm(x, weight, eps=1e-5):
    x_arg = x.reshape(-1, x.shape[-1])
    M, N = x_arg.shape
    y = torch.empty_like(x)
    BLOCK = min(65536 // x.element_size(), triton.next_power_of_2(N))
    rmsnorm_kernel[(M,)](x_arg, y, weight, x_arg.stride(0), N, eps, BLOCK=BLOCK)
    return y


if __name__ == "__main__":
    torch.manual_seed(0)
    M, N, eps = 128, 1024, 1e-5
    x = torch.randn(M, N, dtype=torch.float16)
    w = torch.rand(N, dtype=torch.float16)

    y = rmsnorm(x, w, eps)

    xf = x.float()
    y_ref = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
             * w.float()).to(torch.float16)
    torch.testing.assert_close(y, y_ref, atol=1e-2, rtol=1e-2)
    print("PASS  max diff =", (y - y_ref).abs().max().item())
```

### Python 包装函数

```Python
def rmsnorm(x, weight, eps=1e-5):
    x_arg = x.reshape(-1, x.shape[-1])
    M, N = x_arg.shape
    y = torch.empty_like(x)

    BLOCK = min(
        65536 // x.element_size(),
        triton.next_power_of_2(N)
    )

    rmsnorm_kernel[(M,)](
        x_arg, y, weight,
        x_arg.stride(0), N, eps,
        BLOCK=BLOCK
    )
    return y
```

```Python
x_arg = x.reshape(-1, x.shape[-1])
```

含义：保留最后一个维度，将前面所有维度合并成行。原因是 RMSNorm 沿最后一个维度计算。

例如：

```Python
原始形状：    (2, 3, 1024)
整理后形状：  (6, 1024)

M = 6
N = 1024
```

`torch.empty_like(x)` 则分配输出空间，形状和类型与输入相同。里面的数值随后由内核写入。

`[(M,)]`表示启动 M 个逻辑任务，启动语法：

```Python
rmsnorm_kernel[(M,)](...)
```

表示使用一维 grid，包含 `M` 个 program。

教程让每个 program 负责一行，内核通过下面这句知道自己负责哪一行，括号中的 `0` 指 grid 的第 0 个维度。

```Python
row = tl.program_id(0)
```

program 是 Triton 的逻辑执行单元，不应直接理解为一个 CPU 核或一个线程。 实际如何调度，由后端和运行时决定。

### 内存指针

Kernal 开头：

```Python
@triton.jit
def rmsnorm_kernel(X, Y, W, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    X += row * stride
    Y += row * stride
```

`X`、`Y`、`W` 在这里是内存指针。

例如考虑连续的`(128, 1024)` 输入：

```Python
stride = 1024

第 0 行起点：X + 0
第 1 行起点：X + 1024
第 2 行起点：X + 2048
```

这里的偏移量以元素个数为单位，不是字节数。

```Python
X += row * stride
```

当前 program 中的 `X` 就指向它负责的那一行。后面访问 `X + cols`，就是在访问这一行的各列。

### Block：每次处理的列数

```Python
BLOCK = min(
    65536 // x.element_size(),
    triton.next_power_of_2(N)
)
```

理解：每个 Program 扫描一行时，每次处理的元素块大小。

例如，输入为 `float16`，每个元素占 2 字节：

```Plain Text
N = 1024 → BLOCK = 1024
N = 1000 → BLOCK = 1024
```

`next_power_of_2(1000)` 得到 `1024`，因此第二种情况会多出 24 个无效位置，后面需要用 `mask` 处理。

如果 `N > BLOCK`，一个 program 就要循环处理多个块。

内核工作：

- Grid 决定有多少个 Program

- Row 决定当前 Program 处理哪一行

- Block 决定它每次处理多少列

### Pass 1

```Python
_acc = tl.zeros([BLOCK], dtype=tl.float32)

for off in range(0, N, BLOCK):
    cols = off + tl.arange(0, BLOCK)
    x = tl.load(
        X + cols,
        mask=cols < N,
        other=0.
    ).to(tl.float32)
    _acc += x * x

ms = tl.sum(_acc, axis=0) / N
rstd = 1 / tl.sqrt(ms + eps)
```

`tl.arange` ：一次生成一组列索引。这个之前学习过。

```Python
cols = off + tl.arange(0, BLOCK)
```

`tl.arange(0, BLOCK)` 生成：

```Plain Text
[0, 1, 2, ..., BLOCK-1]
```

循环变量 `off` 决定当前处理哪个块。假设 `N=10`、`BLOCK=4`，则依次得到：

```Plain Text
off = 0 → cols = [0, 1, 2, 3]
off = 4 → cols = [4, 5, 6, 7]
off = 8 → cols = [8, 9, 10, 11]
```

`mask` ：控制哪些位置可以读取。这个之前学习过。

```Python
x = tl.load(X + cols, mask=cols < N, other=0.)
```

`tl.load` 的行为是：

- `mask=True`：读取对应内存位置。

- `mask=False`：不读取该位置，返回 `other` 指定的值，即 `0`。

`_acc` ：是向量累加器。这个之前学习过。

```Python
_acc = tl.zeros([BLOCK], dtype=tl.float32)
```

这会创建长度为 `BLOCK` 的零向量。

每轮执行：

```Plain Text
_acc += x * x
```

是在对应位置上累加，还没有把所有位置加成一个数。

例如：

```Python
x = [1, 2, 3, 4, 5, 6]
N = 6
BLOCK = 4
```

累加过程为：

```Python
初始：
_acc = [0, 0, 0, 0]

第一块读取 [1, 2, 3, 4]：
_acc = [1, 4, 9, 16]

第二块读取 [5, 6, 0, 0]：
_acc = [26, 40, 9, 16]
```

循环结束后，才通过：

```Plain Text
tl.sum(_acc, axis=0)
```

把向量归约成标量：

```Plain Text
26 + 40 + 9 + 16 = 91
```

**FP32 使用时机**

加载后立马转成 float32。

```Python
x = tl.load(...).to(tl.float32)
```

这使得后面的平方和累加都使用 float32。

原因：中间计算需要更大的数值范围和更好的精读。

计算 rstd：

```Python
rstd = 1 / tl.sqrt(ms + eps)
```

### Pass 2

```Python
for off in range(0, N, BLOCK):
    cols = off + tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
    w = tl.load(W + cols, mask=mask)

    tl.store(
        Y + cols,
        (x * rstd * w).to(Y.dtype.element_ty),
        mask=mask,
    )
```

加载当前 block 的输入和权重：

```Python
x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
w = tl.load(W + cols, mask=mask)
```

核心计算是：

```Plain Text
x * rstd * w
```

其中 `rstd` 是标量，但会自动广播到当前块的每个元素。

最后写回：

```Python
tl.store(
    Y + cols,
    (x * rstd * w).to(Y.dtype.element_ty),
    mask=mask,
)
```

### 完整数据流

RMSNorm 的完整数据流是：

```Plain Text
X（第一次读取）→ 求 sum(x²) → ms → rstd
X（第二次读取）+ W → x * rstd * w → Y
```

## smt\_rvisa 版本

### Raw kernal 源码

```Python
@tle.raw_kernel
def rms_norm_1d_kernel(X: tle.mem(f16), out: tle.mem(f32, out=True), N: tle.index):
    nvl = tle.vconfig(-1, 1)
    Nfloor = (N // nvl) * nvl
    acc = tle.vzero(f32)
    for i in tle.range(0, Nfloor, nvl):
        vx = tle.cast(tle.vload(X, i), f32)
        acc = acc + vx * vx
    for i in tle.range(Nfloor, N, nvl):        # tail（整除时空循环）
        tle.vconfig(N - i, 1)
        tx = tle.cast(tle.vload(X, i), f32)
        acc = acc + tx * tx
    ms = tle.vreduce_sum(acc) / N              # ← f32 标量 / index 标量
    scale = tle.rsqrt(ms + EPS)                # ← rsqrt 作用在标量上
    # 归一化循环用独立临时名（nx/mx），别复用 vx/tx——历史 bug 会把出作用域
    # 的同名变量误当 loop iter_arg
    for i in tle.range(0, Nfloor, nvl):
        nx = tle.cast(tle.vload(X, i), f32)
        tle.vstore(out, i, nx * scale)         # ← 标量广播进向量乘
    for i in tle.range(Nfloor, N, nvl):
        tle.vconfig(N - i, 1)
        mx = tle.cast(tle.vload(X, i), f32)
        tle.vstore(out, i, mx * scale)
```

示例处理的是一个一维向量，没有乘权重 `w`，输入为 float16，输出为 float32。 它计算的是：

$out_i=\frac{x_i}{\sqrt{\frac{1}{N}\sum_jx_j^2+\mathrm{EPS}}}$

### Kernal 开头

```Python
@tle.raw_kernel
def rms_norm_1d_kernel(
    X: tle.mem(f16),
    out: tle.mem(f32, out=True),
    N: tle.index,
):
    nvl = tle.vconfig(-1, 1)
    Nfloor = (N // nvl) * nvl
    acc = tle.vzero(f32)
```

```Python
X: tle.mem(f16)
```

表示 `X` 是一块保存 float16 元素的连续内存。

```Python
out: tle.mem(f32, out=True)
```

表示输出内存保存 float32 元素，`out=True` 将这个参数标记为只写。

```Python
N: tle.index
```

表示 `N` 是索引类型的标量，用于描述长度、偏移和循环边界。它与保存数据的向量是两种不同的对象。

```Python
nvl = tle.vconfig(-1, 1)
```

`nvl` 是一次向量操作处理的元素数。`-1` 请求最大向量长度，并将这个长度赋给 `nvl`。

```Python
Nfloor = (N // nvl) * nvl
```

`Nfloor` 把完整块和尾部拆开。这会把 `N` 向下取整到 `nvl` 的整数倍。

例如，假设 `N=1000`、`nvl=32`：

```Plain Text
Nfloor = (1000 // 32) * 32
       = 31 * 32
       = 992

完整部分：索引 0～991，共 31 个向量块
尾部部分：索引 992～999，共 8 个元素
```

```Python
acc = tle.vzero(f32)
```

`acc` 仍然是向量累加器。它创建全零的 float32 向量。后续每次读取一块数据，将对应位置的平方加到 `acc` 上。全部扫描完成后，才将 `acc` 的各位置加起来，得到一个标量。

### Pass 1

```Python
acc = tle.vzero(f32)

for i in tle.range(0, Nfloor, nvl):
    vx = tle.cast(tle.vload(X, i), f32)
    acc = acc + vx * vx

for i in tle.range(Nfloor, N, nvl):
    tle.vconfig(N - i, 1)
    tx = tle.cast(tle.vload(X, i), f32)
    acc = acc + tx * tx
```

第一个循环处理完整块，第二个循环处理尾部剩余元素。

```Python
tle.vload(X, i)
```

理解：表示从偏移 `i` 开始加载一个向量，加载的元素数量由当前向量配置决定。

两个循环之后的计算：

```Python
ms = tle.vreduce_sum(acc) / N
scale = tle.rsqrt(ms + EPS)
```

循环内部计算：

```Python
acc = acc + vx * vx
```

进行的是对应向量位置之间的累加。`acc` 仍然是向量，尚未得到整条输入的平方和。

```Python
tle.vreduce_sum(acc)
```

执行上述，才会把各位置的部分合并成一个 float32 标量。再除以实际元素数 `N`，得到均方：

$ms=\frac{\sum_{j=0}^{N-1}x_j^2}{N}$

然后计算：

```Python
scale = tle.rsqrt(ms + EPS)
```

$scale=\frac{1}{\sqrt{ms+\mathrm{EPS}}}$

### Pass 2

```Python
for i in tle.range(0, Nfloor, nvl):
    nx = tle.cast(tle.vload(X, i), f32)
    tle.vstore(out, i, nx * scale)

for i in tle.range(Nfloor, N, nvl):
    tle.vconfig(N - i, 1)
    mx = tle.cast(tle.vload(X, i), f32)
    tle.vstore(out, i, mx * scale)
```

循环结构与第一趟相同，但这次加载输入后，直接乘已经求出的 `scale`，然后写回输出。

### 使用 raw kernal 的完整代码

```Python
import torch
import triton

from triton.backends.spine_triton.driver import CPUDriver
import triton.language.extra.smt_rvisa as tle
from triton.language.extra.smt_rvisa import call as _sr_call


triton.runtime.driver.set_active(CPUDriver())

f16, f32 = tle.f16, tle.f32
EPS = 1e-5


@tle.raw_kernel
def rms_norm_1d_kernel(
    X: tle.mem(f16),
    out: tle.mem(f32, out=True),
    N: tle.index,
):
    nvl = tle.vconfig(-1, 1)
    Nfloor = (N // nvl) * nvl

    # 第一趟：累加平方和
    acc = tle.vzero(f32)

    for i in tle.range(0, Nfloor, nvl):
        vx = tle.cast(tle.vload(X, i), f32)
        acc = acc + vx * vx

    for i in tle.range(Nfloor, N, nvl):
        tle.vconfig(N - i, 1)
        tx = tle.cast(tle.vload(X, i), f32)
        acc = acc + tx * tx

    # 向量归约，计算标量缩放系数
    ms = tle.vreduce_sum(acc) / N
    scale = tle.rsqrt(ms + EPS)

    # 第二趟：归一化并写出
    for i in tle.range(0, Nfloor, nvl):
        nx = tle.cast(tle.vload(X, i), f32)
        tle.vstore(out, i, nx * scale)

    for i in tle.range(Nfloor, N, nvl):
        tle.vconfig(N - i, 1)
        mx = tle.cast(tle.vload(X, i), f32)
        tle.vstore(out, i, mx * scale)


@triton.jit
def rms_norm_1d_host(X, out, N):
    _sr_call(
        rms_norm_1d_kernel,
        outputs=[],
        inputs=[X, out, N],
    )


if __name__ == "__main__":
    torch.manual_seed(0)

    # 使用 length，避免创建与内核形参同名的全局变量 N
    for length in (64, 70):
        x = torch.randn(length, dtype=torch.float16)
        out = torch.zeros(length, dtype=torch.float32)

        rms_norm_1d_host[(1,)](x, out, length)

        # PyTorch 参照：无权重，输出保持 float32
        xf = x.float()
        ref = xf * torch.rsqrt((xf * xf).mean() + EPS)

        torch.testing.assert_close(
            out, ref, rtol=1e-2, atol=1e-2
        )

        max_diff = (out - ref).abs().max().item()
        print(f"PASS  N={length}  max diff={max_diff:.8f}")
```

