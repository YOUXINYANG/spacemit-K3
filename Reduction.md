# Reduction

## 简介

### 归约

Reduction 归约，按照某种规则，把一组元素合并成一个结果。

例如：

```Python
x = [3, 1, 4, 1, 5]

sum(x)    = 14       求和
mean(x)   = 2.8      平均值
max(x)    = 5        最大值
argmax(x) = 4        最大值所在的下标，下标从 0 开始
```

### 单位元

参与计算后，不改变原来结果的值。

例如：

```Python
求和：   a + 0 = a        → 单位元是 0
求积：   a × 1 = a        → 单位元是 1
求最大： max(a, -∞) = a   → 单位元是 -∞
求最小： min(a, +∞) = a   → 单位元是 +∞
```

## 任务分配

### 行归约

每个 Program 负责一行。

假设输入矩阵有 `M=3` 行、`N=4` 列：

```Plain Text
输入矩阵 X                       输出 Y

第 0 行：[ 1,  2,  3,  4]  →      10
第 1 行：[ 5,  6,  7,  8]  →      26
第 2 行：[ 9, 10, 11, 12]  →      42
```

每行可以独立求和，因此分配如下：

```Plain Text
program 0 → 处理第 0 行 → 写入 Y[0]
program 1 → 处理第 1 行 → 写入 Y[1]
program 2 → 处理第 2 行 → 写入 Y[2]
```

在教程代码中，启动方式是：

```Python
row_sum_kernel[(M,)](x, y, N, BLOCK=BLOCK)
```

### 全归约

先分块求和，再合并。

假设任务变成：把一个很长的数组求和，只输出一个数。如果只启动一个Program扫完整个数组，就难以充分利用多个 program 的并行能力。

教程采用两段式：

```Plain Text
原始数组
   ↓ 分块
每个 program 计算自己那一块的和
   ↓
局部结果 partial
   ↓ 再次求和
最终结果
```

例如：

```Plain Text
X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
N = 10
BLOCK = 4
```

第一段：

```Plain Text
program 0： [1, 2, 3, 4]   → partial[0] = 10
program 1： [5, 6, 7, 8]   → partial[1] = 26
program 2： [9, 10, 0, 0]  → partial[2] = 19
```

最后一块只有两个有效元素，其余位置通过 `mask` 填 `0`，这就用到了上一部分的单位元。

第二段只需计算：

```Plain Text
partial = [10, 26, 19]

最终结果 = 10 + 26 + 19 = 55
```

教程直接使用 PyTorch 完成第二段：

```Plain Text
return partial.sum()
```

这个实现的第一段由 Triton 计算，第二段由 PyTorch 合并局部结果。

## 运行官方程序

### Debug

在直接运行官方原版程序时，出现了几个问题：

1. ~~编译中间文件解析失败，程序在编译阶段报错：~~

```Python
ptr.to_ptr ... invalid kind of attribute specified
... memorySpace ... MemorySpaceAttrInterface
```

解决办法：将 Triton 升级到官方 `3.6.0+spacemit.b1`，并重新验证。

2. 大 grid 启动后，输出没有被写入。官方全归约失败。当前环境在这些测试中，`grid=977` 的启动没有产生预期写入。具体原因不知道。

暂时的解决办法：

把第一段的 977 个 program 拆成多批，每批最多 128 个。对这个案例，共启动 8 批：前 7 批各 128 个，最后一批 81 个。

### 修改后的完整可运行程序

```Python
import torch
import triton
import triton.language as tl
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())


# 行归约：每个 program 处理一行。
@triton.jit(do_not_specialize=["N"])
def row_sum_kernel(X, Y, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    X += row * N

    acc = tl.zeros([BLOCK], dtype=tl.float32)

    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        x = tl.load(
            X + cols,
            mask=cols < N,
            other=0.,
        ).to(tl.float32)
        acc += x

    tl.store(Y + row, tl.sum(acc, axis=0))


# 全归约第一段：每个 program 计算一个块的和。
@triton.jit(do_not_specialize=["N"])
def partial_sum_kernel(X, P, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)

    x = tl.load(
        X + offs,
        mask=offs < N,
        other=0.,
    ).to(tl.float32)

    tl.store(P + pid, tl.sum(x, axis=0))


def row_sum(x):
    M, N = x.shape
    y = torch.empty(M, dtype=torch.float32)

    BLOCK = min(1024, triton.next_power_of_2(N))
    row_sum_kernel[(M,)](x, y, N, BLOCK=BLOCK)

    return y


def full_sum(x):
    x = x.contiguous().view(-1)
    N = x.numel()

    BLOCK = 1024
    MAX_PROGRAMS_PER_LAUNCH = 128
    n_parts = triton.cdiv(N, BLOCK)

    # 保留 NaN 初始化，便于发现缺失写入。
    partial = torch.full(
        (n_parts,), float("nan"), dtype=torch.float32
    )

    # 分批启动，每批最多 128 个 program。
    for part_start in range(
        0, n_parts, MAX_PROGRAMS_PER_LAUNCH
    ):
        part_end = min(
            part_start + MAX_PROGRAMS_PER_LAUNCH,
            n_parts,
        )

        element_start = part_start * BLOCK
        element_end = min(part_end * BLOCK, N)

        # 切片将本批输入、输出的起始指针移到对应位置。
        x_batch = x[element_start:element_end]
        p_batch = partial[part_start:part_end]

        partial_sum_kernel[(part_end - part_start,)](
            x_batch,
            p_batch,
            x_batch.numel(),
            BLOCK=BLOCK,
        )

    # 第二段：合并所有批次产生的局部和。
    return partial.sum()


if __name__ == "__main__":
    torch.manual_seed(0)

    # 测试 1：行归约。
    x = torch.randn(128, 1000, dtype=torch.float16)
    got_rows = row_sum(x)
    ref_rows = x.float().sum(dim=1)

    torch.testing.assert_close(
        got_rows,
        ref_rows,
        atol=1e-2,
        rtol=1e-2,
    )
    print("PASS row_sum 128x1000", flush=True)

    # 测试 2：分批启动的两段式全归约。
    big = torch.randn(1_000_000, dtype=torch.float16)
    got = full_sum(big)
    ref = big.float().sum()

    print(
        "full_sum:",
        "triton =", got.item(),
        "torch =", ref.item(),
        "绝对误差 =", (got - ref).abs().item(),
        flush=True,
    )

    torch.testing.assert_close(
        got,
        ref,
        atol=1e-1,
        rtol=1e-3,
    )
    print("PASS full_sum 1M", flush=True)
```

运行结果：

```Python
PASS row_sum 128x1000
full_sum: triton = -999.4791870117188 torch = -999.4793701171875 绝对误差 = 0.00018310546875
PASS full_sum 1M
```

## 代码解读

### 激活 K3 后端

```Python
triton.runtime.driver.set_active(CPUDriver())
```

让 Triton 使用 Spine Triton 的 CPU 后端编译和运行 kernel。

### 行归约

```Plain Text
row_sum_kernel[(M,)](x, y, N, BLOCK=BLOCK)
```

对于 `(128, 1000)` 的输入，启动 128 个 program，每个负责一行。

kernel 中：

```Plain Text
row = tl.program_id(0)
X += row * N
```

`row` 是当前 program 的编号，`row * N` 将输入指针移到对应行的开头。

分块加载，向量累加，最后求和：

```Python
acc = tl.zeros([BLOCK], dtype=tl.float32)

for off in range(0, N, BLOCK):
    cols = off + tl.arange(0, BLOCK)
    x = tl.load(
        X + cols, mask=cols < N, other=0.
    ).to(tl.float32)
    acc += x

tl.store(Y + row, tl.sum(acc, axis=0))
```

### 全归约

每个 Program 负责一个块。

```Python
pid = tl.program_id(0)
offs = pid * BLOCK + tl.arange(0, BLOCK)
```

与行归约不同，这里的 `pid` 对应块编号。每个 program 读取最多 1024 个元素，求和后写入：

```Python
tl.store(P + pid, tl.sum(x, axis=0))
```

### 分批启动再合并

当前版本每次最多启动 128 个 program：

```Plain Text
x_batch = x[element_start:element_end]
p_batch = partial[part_start:part_end]
```

切片让每批 kernel 从正确的输入、输出位置开始，kernel 内的 `pid` 每批都从 0 开始。各批完成后，将全部 977 个局部和合并成最终结果：

```Python
return partial.sum()
```

