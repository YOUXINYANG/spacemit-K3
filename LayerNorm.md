# LayerNorm

## 简介

LayerNorm 全称 Layer Normalization（层归一化），是 Transformer 等神经网络中常见的算子。

作用：将一组数据按照自身的均值和方差进行标准化，再进行可学习的缩放和平移。

例如，输入是一个二维矩阵：

$X=\begin{bmatrix} 1&2&3&4\\ 5&6&7&8 \end{bmatrix}$

LayerNorm 按行进行计算。第一行与第二行独立处理。

对于长度为 N 的一行数据，计算过程分为三步。

第一步：计算均值（Mean）

$\mu=\frac{1}{N}\sum_{i=1}^{N}x_i $

第二步：计算方差（Variance）

$\sigma^2=\frac{1}{N}\sum_{i=1}^{N}(x_i-\mu)^2$

第三步：归一化并缩放平移

$y_i=\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}\gamma_i+\beta_i$

其中：

- $\epsilon$：很小的正数，防止除零。

- $\gamma_i$（weight）：可学习的缩放参数。

- $\beta_i$（bias）：可学习的平移参数。

## 计算特点

LayerNorm 的均值和方差依赖整行数据，不能只看到部分元素就直接计算最终结果。

这对硬件实现的启发：

1. 需要对一行中的元素进行求和归约 reduction；

2. 要获得整行均值后，才能计算以该均值为中心的方差；

3. 最后才能进行归一化和仿射变换；

## Pytorch 实现

```Python
python3 - <<'PY'
import torch

# 输入数据
x = torch.tensor([1., 2., 3., 4.])

# 第一步：均值
mean = x.mean()

# 第二步：方差
var = ((x - mean) ** 2).mean()

# 第三步：归一化
eps = 1e-5
y = (x - mean) / torch.sqrt(var + eps)

print("mean =", mean.item())
print("var =", var.item())
print("y =", y)
PY
```

输出：

```Python
mean = 2.5
var = 1.25
y = tensor([-1.3416, -0.4472,  0.4472,  1.3416])
```

## 算子融合 Kernel Fusion

LayerNorm 的计算过程：

$\mu=E[x],\qquad \sigma^2=E[(x-\mu)^2]$

$y=\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}\cdot w+b$

如果直接使用多个独立的 pytorch 操作，可能写成：

```Python
mean = x.mean(dim=-1, keepdim=True)
var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
rstd = torch.rsqrt(var + eps)
y = (x - mean) * rstd * w + b
```

这里包含均值计算、方差计算、减法、乘法和加法等多个操作。

在没有算子融合的情况下，不同操作之间可能需要频繁地读写中间张量。

```plaintext
未融合：多个 Kernel                    融合：一个 Kernel

读 X → 算 Mean                         +------------------------+
      |                                | 加载 X                 |
      v                                |   |                    |
读 X → 算 Var                          |   v                    |
      |                                | Mean + Var             |
      v                                |   |                    |
读中间结果 → 归一化                    |   v                    |
      |                                | 归一化 + 仿射          |
      v                                |   |                    |
读中间结果 → 仿射变换                  |   v                    |
                                       | 写入 Y                 |
多个独立操作与中间结果                 +------------------------+
                                       一个 Kernel，减少中间访存
```

算子融合**主要目的**：不是改变计算公式，而是把原本分散的计算集中执行，尽可能减少中间数据的内存读写。

## Memory\-bound

一个 kernal 的执行性能通常受到两种主要因素限制：

对于 LayerNorm，每个元素只需要进行少量加减乘除运算，但是需要加载输入数据、读取参数，并写出结果。

**关键**：CPU 可能花费更多时间等待数据，而不是执行算术计算。

## LayerNorm 如何并行执行

LayerNorm 每行独立计算，不同行之间没有数据依赖。

因此，可以让一个 triton program 负责一整行。

```plaintext
输入矩阵：每行独立计算

第 0 行   [ | | | | | ... ]  --->  Program 0
第 1 行   [ | | | | | ... ]  --->  Program 1
第 2 行   [ | | | | | ... ]  --->  Program 2
   ...              ...                ...
第 M-1 行 [ | | | | | ... ]  --->  Program M-1

每个 Program 完成对应行的均值、方差和归一化计算，
不需要与其他 Program 交换数据。
```

在 triton 中，使用：

```Python
grid = (M,)
```

表示：在一维 grid 上启动 M 个 Program 实例。

```Python
row = tl.program_id(0)
```

表示：获取当前 Program 的编号。

例如：`row=2`，说明当前 Program 负责第 2 行（从 0 开始计数）。

## 行内分块

```Python
M = 128
N = 1024
BLOCK_SIZE = 256
```

一个 Program 负责 1024 个元素，但一次按 `BLOCK_SIZE=256` 处理数据。

一行分成 4 块：

```plaintext
Program 0 处理第 0 行

+-------------+-------------+-------------+-------------+
|   Block 0   |   Block 1   |   Block 2   |   Block 3   |
|    0～255   |   256～511  |   512～767  |  768～1023  |
+-------------+-------------+-------------+-------------+

一行 1024 个元素，分成 4 个连续的数据块
```

在 triton 中，分块循环的程序实现：

```Python
for off in range(0, N, BLOCK_SIZE):
    cols = off + tl.arange(0, BLOCK_SIZE)
```

- `off`：当前块在这一行中的起始偏移。

- `tl.arange(0, BLOCK_SIZE)`：生成当前块内的列偏移向量。

- `cols`：当前块需要访问的实际列下标。

第一次循环：

```Python
off = 0
cols = [0, 1, 2, ..., 255]
```

第二次循环：

```Python
off = 256
cols = [256, 257, ..., 511]
```

## 三趟扫描策略

官方教程采用三趟扫描（three\-PASS）实现 LayerNorm。

### PASS 1：Mean

扫描整行，累加所有元素，再除以 N，得到均值。

$\mu=\frac{1}{N}\sum x_i$

### PASS 2：Variance

重新扫描整行，计算各个元素与均值之差的平方，再归约得到方差。

$\sigma^2=\frac{1}{N}\sum(x_i-\mu)^2$

### PASS 3：Output

再次扫描整行，使用均值和方差完成归一化、仿射变换，写回输出。

$y_i=(x_i-\mu)\cdot rstd\cdot w_i+b_i$

$rstd=\frac{1}{\sqrt{\sigma^2+\epsilon}}$

`rstd` 是标准差的倒数，提前计算可以避免为每个元素重复执行平方根和除法。

## 官方代码

```Python
# layernorm.py — 每 program 一行的 fused LayerNorm forward
import torch
import triton
import triton.language as tl
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())


@triton.jit
def _layer_norm_fwd_fused(
    X,          # 输入 (M, N)，按行连续
    Y,          # 输出，同形状
    W, B,       # weight/bias，各 (N,)
    Mean, Rstd, # 每行的 mean / 1-std（f32），缓存给 backward 用
    stride,     # 行 stride（元素数）
    N,          # 列数
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)          # 我是第几行
    X += row * stride               # 指针移到本行开头
    Y += row * stride

    # ── 趟1: mean（f32 累加器）─────────────────────────────
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _mean += x
    mean = tl.sum(_mean, axis=0) / N        # 向量 → 标量

    # ── 趟2: var ──────────────────────────────────────────
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.)   # 越界 lane 强制贡献 0
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    # 缓存 mean/rstd（backward 要用，避免重算）
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)

    # ── 趟3: 归一化 + 仿射 + 写出 ─────────────────────────
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        y = (x - mean) * rstd * w + b
        tl.store(Y + cols, y, mask=mask)      # 自动 cast 回 Y 的 dtype


def layer_norm(x, weight, bias, eps=1e-5):
    x_arg = x.reshape(-1, x.shape[-1])
    M, N = x_arg.shape
    y = torch.empty_like(x)
    mean = torch.empty((M,), dtype=torch.float32)
    rstd = torch.empty((M,), dtype=torch.float32)

    # 官方教程的 BLOCK 启发式：单 feature 不超 64KB
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))

    _layer_norm_fwd_fused[(M,)](
        x_arg, y, weight, bias, mean, rstd,
        x_arg.stride(0), N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y, mean, rstd


if __name__ == "__main__":
    torch.manual_seed(0)
    M, N, eps = 128, 1024, 1e-5
    x = torch.randn(M, N, dtype=torch.float16)
    w = torch.rand(N, dtype=torch.float16)
    b = torch.rand(N, dtype=torch.float16)

    y, mean, rstd = layer_norm(x, w, b, eps)

    # golden：torch CPU 的 layer_norm 不吃 fp16，用 f32 算再降回来
    y_ref = torch.nn.functional.layer_norm(
        x.float(), (N,), w.float(), b.float(), eps).to(torch.float16)
    torch.testing.assert_close(y, y_ref, atol=1e-2, rtol=1e-2)
    print("PASS  y max diff =", (y - y_ref).abs().max().item())
```

## 运行官方代码

官方代码放在了教程的 github 网页中，首先下载教程然后提取其中的 Python 代码：

```Bash
curl -fL https://raw.githubusercontent.com/spacemit-com/spine-triton-tutorial/main/ops/02-layernorm.md -o 02-layernorm.md
```

```Bash
python3 - <<'PY'
from pathlib import Path
import re

text = Path("02-layernorm.md").read_text()
section = text.split("## 2.4 完整可运行程序", 1)[1]

match = re.search(r"```python\s*\n(.*?)\n```", section, re.S)
assert match, "未找到 2.4 节 Python 代码"

Path("layernorm.py").write_text(match.group(1) + "\n")
print("Created layernorm.py")
PY
```



运行官方程序：

```Bash
python3 layernorm.py
```

输出：

```Python
PASS  y max diff = 0.001953125
```

## 理解源码

### 定义 kernal

```Python
@triton.jit
def _layer_norm_fwd_fused(
    X,
    Y,
    W, B,
    Mean, Rstd,
    stride,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
```

参数含义：

`@triton.jit` 表示该函数会被 Triton 编译为设备执行的 Kernel。

`tl.constexpr` 表示 BLOCK\_SIZE 是编译期常量，编译器可以根据其值进行专门优化。

### Program ID 和指针偏移

```Python
row = tl.program_id(0)
X += row * stride
Y += row * stride
```

假设：

$X\in\mathbb{R}^{128\times1024}$

输入矩阵按行连续存放，因此：

```Plain Text
stride = 1024
```

如果当前执行的是 Program 2，那么：

```Plain Text
row = 2
```

其输入起始地址变成：

$X_{\text{row2}}=X_{\text{base}}+2\times1024$

由于指针已经被移动到当前行开头，后续访问本行对应列的数据，只需要：

```Plain Text
tl.load(X + cols)
```

### Pass 1

第一趟扫描计算均值。

```Python
_mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

for off in range(0, N, BLOCK_SIZE):
    cols = off + tl.arange(0, BLOCK_SIZE)

    x = tl.load(
        X + cols,
        mask=cols < N,
        other=0.
    ).to(tl.float32)

    _mean += x

mean = tl.sum(_mean, axis=0) / N
```

1. 初始化向量累加器。

```Python
_mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
```

假设 BLOCK\_SIZE = 4，那么：

```Plain Text
_mean = [0, 0, 0, 0]
```

2. 读取一个数据块

```Python
cols = off + tl.arange(0, BLOCK_SIZE)
```

假设：

```Python
N = 8BLOCK_SIZE = 4
```

第一次循环：

```Plain Text
off = 0
cols = [0, 1, 2, 3]
```

第二次循环：

```Plain Text
off = 4
cols = [4, 5, 6, 7]
```

随后执行：

```Plain Text
x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
```

读取当前块的数据，并立即转换为 float32，以提高后续累加精度。

3. 向量累加

假设输入一行是：

$X=[1,2,3,4,5,6,7,8]$

循环实现：

```plaintext
第一次累加（off = 0）

          +-----+-----+-----+-----+
          |  1  |  2  |  3  |  4  |
          +-----+-----+-----+-----+
_mean = [1, 2, 3, 4]

                   |
                   v

第二次累加（off = 4）

          +-----+-----+-----+-----+
          |  6  |  8  | 10  | 12  |
          +-----+-----+-----+-----+
_mean = [1+5, 2+6, 3+7, 4+8]
```

循环完成后，执行：

```Python
mean = tl.sum(_mean, axis=0) / N
```

将向量的所有元素相加，最终计算得到标量 Mean。

**FP32 和 FP16 的使用时机**

官方代码在 `tl.load` 后立即使用：

```Python
.to(tl.float32)
```

原因是输入可能使用 FP16，如果直接以半精度进行大量累加，容易引入明显的数值误差。

虽然输入输出都是 FP16，但均值和方差计算通常应该使用更高精度。

### Pass 2

官方实现代码：

```Python
_var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

for off in range(0, N, BLOCK_SIZE):
    cols = off + tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
    x = tl.where(cols < N, x - mean, 0.)
    _var += x * x

var = tl.sum(_var, axis=0) / N
rstd = 1 / tl.sqrt(var + eps)
```

**为什么需要 ****`tl.where`****？**

假设一行有 5 个元素：

$X=[1,2,3,4,5]$

其均值：

$mean=3$

假设 `BLOCK_SIZE=4`，则需要两个数据块：

```plaintext
第一个 Block

+--------+--------+--------+--------+
|   1    |   2    |   3    |   4    |
+--------+--------+--------+--------+

第二个 Block（mask 越界加载）

+--------+--------+--------+--------+
|   5    |   0    |   0    |   0    |
|  有效  |  填充  |  填充  |  填充  |
+--------+--------+--------+--------+
```

由于 `tl.load` 指定了 `other=0.`，越界的三个元素会被填充为 0。

如果直接计算：

```Python
x = x - mean
```

第二个 Block 就变成：

$[5,0,0,0]-3=[2,-3,-3,-3]$

平方以后：

$[4,9,9,9]$

那么 0 元素也参与了方差的计算，导致计算错误。

正确实现方法：

```Python
x = tl.where(cols < N, x - mean, 0.)
```

逻辑理解：

```Python
if col < N:
    result = x - mean
else:
    result = 0
```

避免了越界元素参与了方差计算。

**为什么计算 rstd？**

```Plain Text
rstd = 1 / tl.sqrt(var + eps)
```

其中：

$rstd=\frac{1}{\sqrt{var+\epsilon}}$

`rstd` 是标准差的倒数。

将它提前计算出来，后续每个元素只需要乘以 `rstd`，就可以完成归一化，而不必重复进行平方根和除法。随后官方程序把 `mean` 和 `rstd` 写到对应的输出缓冲区：

```Python
tl.store(Mean + row, mean)
tl.store(Rstd + row, rstd)
```

这样后续的反向传播可以复用这些结果。

### Pass 3

第三趟扫描的目的是为了计算最终输出：

$y_i=(x_i-\mu)\times rstd\times w_i+b_i$

官方代码实现：

```Python
for off in range(0, N, BLOCK_SIZE):
    cols = off + tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    w = tl.load(W + cols, mask=mask)
    b = tl.load(B + cols, mask=mask)
    x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
    y = (x - mean) * rstd * w + b
    tl.store(Y + cols, y, mask=mask)
```

**加载 w 和 b**

```Python
w = tl.load(W + cols, mask=mask)
b = tl.load(B + cols, mask=mask)
```

`W` 和 `B` 都是一维张量，长度为 N。

不同的行可以共享同一组 `W` 和 `B`，但每一列具有自己对应的 weight 和 bias。

**计算输出**

```Python
y = (x - mean) * rstd * w + b
```

**写回结果**

```Python
tl.store(Y + cols, y, mask=mask)
```

`tl.store` 将计算结果写入输出张量 Y。

### Host 函数

Host 和 kernal 的区别：

- Host：负责准备数据、配置参数和发起调用。

- Kernel：负责实际计算。

LayerNorm 的 host 函数：

```Python
def layer_norm(x, weight, bias, eps=1e-5):
```

确定输入矩阵维度。

```Python
x_arg = x.reshape(-1, x.shape[-1])
M, N = x_arg.shape
```

配置 BLOCK\_SIZE。官方使用的策略：

```Python
MAX_FUSED_SIZE = 65536 // x.element_size()
BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
```

`x.element_size()` 表示每个输入元素占用的字节数。

`triton.next_power_of_2(N)` ：返回不小于 N 的最小 2 的幂。

例如：

启动 kernal。核心调用方式：

```Python
_layer_norm_fwd_fused[(M,)](
    x_arg, y, weight, bias, mean, rstd,
    x_arg.stride(0), N, eps,
    BLOCK_SIZE=BLOCK_SIZE,
)
```

`[(M,)]`指定一维启动网格，创建 M 个 Program 实例。

## smt\_rvisa

标准 triton kernal 的编写方式比较简洁。这些高级操作最终如何转换成 RISC\-V 向量指令，主要由编译器决定。

`smt_rvisa` 是 Spine Triton 提供的一套 Python 嵌入式领域专用语言（eDSL），用于更直接地描述 K3 上的 RISC\-V 向量计算。

`smt_rvisa` 就提供了更直接的控制方式，更精细地控制底层向量操作，例如：

- 每次加载多少个元素？

- 使用什么精度的向量累加器？

- 什么时候执行水平归约？

- 怎样处理不足一个向量寄存器的尾部数据？

```plaintext
标准 Triton                         smt_rvisa

+------------------------+          +------------------------+
| tl.load / tl.sum       |          | vload / vreduce_sum    |
+------------------------+          +------------------------+
            |                                  |
            v                                  v
+------------------------+          +------------------------+
| 编译器自动向量化       |          | 显式描述向量操作       |
+------------------------+          +------------------------+
            |                                  |
            v                                  v
+------------------------+          +------------------------+
| RISC-V 向量指令        |          | RISC-V 向量指令        |
+------------------------+          +------------------------+
```

### 核心 API

使用官方接口：

```Python
import triton.language.extra.smt_rvisa as tle
```

这里 `tle` 相当于底层向量编程接口。

常见函数：

**vconfig**

官方教程以 K3 的 1024\-bit 向量寄存器为例。

```plaintext
一个 1024-bit 向量寄存器（LMUL=1）

FP32：每个元素 32 bit，共 32 个元素
+--------+--------+--------+-----+--------+
| 元素 0 | 元素 1 | 元素 2 | ... | 元素31 |
+--------+--------+--------+-----+--------+
                  1024 bit

FP16：每个元素 16 bit，共 64 个元素
+----+----+----+----+----+-----+----+
| 0  | 1  | 2  | 3  | 4  | ... | 63 |
+----+----+----+----+----+-----+----+
               1024 bit
```

```Python
nvl = tle.vconfig(-1, 1)
```

表示按照该实现的向量配置获取最大可处理元素。在教程所用的 FP32、LMUL=1 配置下，`nvl=32`。

如果 LayerNorm 有 256 个元素，处理完整一行通常需要：

$256/32=8\text{ 次向量迭代}$

这与标准 Triton 直接使用长度为 256 的 `tl.arange` 不同。在 raw 版本中，程序员需要自己明确组织这些向量循环。

### Raw LayerNorm

官方提供的 raw 版本位于：

```Python
python/tests/raw/test_raw_layernorm.py
```

该版本比较简化：

- 输入：一维 float16 数组。

- 输出：一维 float32 数组。

- 不包含 weight 和 bias。

- 使用三趟扫描完成 LayerNorm。

**第一趟：计算 Mean**

首先使用一个 FP32 向量累加器对输入求和，最后执行水平归约：

$mean=\frac{\sum_i x_i}{N}$

这里需要使用 `tle.vload`、`tle.cast` 和 `tle.vreduce_sum`。

**第二趟：计算 Variance**

这里与之前的标准 Triton 版本有一个重要区别。

标准版使用：

$var=E[(x-mean)^2]$

而 raw 版本使用等价的数学公式：

$\boxed{var=E[x^2]-mean^2}$

这一方案使得 padding 0 不会贡献额外的平方和，而且平方和本身不依赖已经计算好的 mean。

两种公式在实数数学中等价，在有限精度浮点计算中不完全等价。 当输入数值很大、方差却很小时，`E[x²] - mean²` 可能发生严重的消去误差。因此不能简单认为 raw 版本在数值上更优。

**第三趟：计算最终输出**

$y_i=(x_i-mean)\times\frac{1}{\sqrt{var+\epsilon}}$

由于这个简化版本没有 weight 和 bias，所以不需要执行额外的仿射变换。

使用 `tle.vstore` 将结果写回 FP32 输出数组。

**如何调用 raw kernal？**

`@tle.raw_kernel` 定义的函数不会直接替代整个 Triton 启动流程。

官方使用 `_sr_call`，在普通 `@triton.jit` Kernel 内部调用 raw kernel。

它们之间的关系是：

```plaintext
+--------------------------------------------------+
| Python Host                                      |
| 准备输入、输出并启动 grid                        |
+--------------------------------------------------+
                         |
                         v
+--------------------------------------------------+
| @triton.jit Host Kernel                          |
| 通过 _sr_call 调用 raw kernel                    |
+--------------------------------------------------+
                         |
                         v
+--------------------------------------------------+
| @tle.raw_kernel                                  |
| 执行显式向量加载、运算、归约和写回               |
+--------------------------------------------------+
```

~~目前 raw kernal 环境尚未完善，无法运行官方现有测试。~~

已解决：将 triton 从 a5 版本升级到 b0 即可使用 smt\_rvisa。

