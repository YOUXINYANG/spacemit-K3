# Cumsum / Scan

## 简介

Cumsum 是 cumulative sum 的简称，累积和或前缀和。

例如：

```Python
输入 x = [3, 1, 4, 1, 5]
输出 y = [3, 4, 8, 9, 14]
```

逐个位置计算过程：

```Python
y[0] = 3                 = 3
y[1] = 3 + 1             = 4
y[2] = 3 + 1 + 4         = 8
y[3] = 3 + 1 + 4 + 1     = 9
y[4] = 3 + 1 + 4 + 1 + 5 = 14
```

数学表达式：

```math
y[i]=\sum_{j=0}^{i}x[j]
```

## 需要解决的核心问题

串行代码可以写成：

```Plain Text
acc = acc + x[i]
y[i] = acc
```

但这样每一步都需要等待前一步的 `acc`，形成一条依赖链。教程后面的重点，就是如何组织并行计算，同时让每个位置获得正确的累计结果。

## 任务划分

先分成两块，分别计算：

```Python
输入：
[3, 1, 4, 1 | 5, 9, 2, 6]
```

让两个任务分别做块内 cumsum：

```Plain Text
第0块：[3, 1, 4, 1] → [3, 4, 8, 9]
第1块：[5, 9, 2, 6] → [5, 14, 16, 22]
```

两个块可以独立计算，所以能够并行。但是不能直接拼起来，还缺少前面块的累加。

第 0 块的总和是 9，所以，第 1 块的每个局部结果都需要加上 9。这个需要补上的值，教程称为 `base`，可以理解为“进入当前块之前，已经累计了多少”。

对任意一块：最终结果 = 块内前缀和 \+ 前面所有块的总和。

## 两趟计算

第一趟：各块独立计算，保存两类结果。

```Plain Text
out：         [3, 4, 8, 9 | 5, 14, 16, 22]
partial_sum： [9, 22]
```

其中：

- `out` 保存每块的局部前缀和。

- `partial_sum` 保存每块的总和，每块只占一个元素。

第二趟：给各块加上对应的 base。

```Plain Text
第0块：无需修改
第1块：每个元素加上 partial_sum[0]，即 9
```

最终得到：

```Python
out = [3, 4, 8, 9, 14, 23, 25, 31]
```

上述两趟计算分别对应两个 kernal：

```Python
scan_part_sum_kernel
    → 计算块内前缀和，同时保存块总和

add_base_sum_kernel
    → 给块内结果加上 base
```

## 官方代码解读

### scan\_part\_sum\_kernel

第一个kernal负责：每块计算局部前缀和，并保存该块的总和。

```SQL
@triton.jit(do_not_specialize=["n_elements", "part_num"])
def scan_part_sum_kernel(
    inp, out, partial_sum,
    n_elements, part_num,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    inp_vals = tl.load(inp + offset, mask=mask, other=0.)

    result = tl.cumsum(inp_vals, axis=0)
    tl.store(out + offset, result, mask=mask)

    tl.store(partial_sum + pid, tl.sum(inp_vals))
```

`tl.cumsum`：只计算当前块的前缀和。

`tl.sum`：另外保存每块的总和。

### add\_base\_sum\_kernel

第二个kernal负责：给后一块加上 base。

```SQL
@triton.jit(do_not_specialize=["n_elements", "part_num"])
def add_base_sum_kernel(
    out, partial_sum,
    n_elements, part_num,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    if pid > 0:
        base = tl.load(partial_sum + pid - 1)
        out_vals = tl.load(out + offset, mask=mask, other=0.)
        tl.store(
            out + offset,
            (out_vals + base).to(out_vals.dtype),
            mask=mask,
        )
```

计算条件：

```Plain Text
if pid > 0:
```

- `pid=0`：第一块已经是正确的全局前缀和，无需修改。

- `pid=1`：第二块需要加上第一块的总和。

### Host 函数

准备内存并启动两个 kernal：

```Python
def cumsum(x, part_num=2):
    n = x.numel()

    BLOCK_SIZE = triton.cdiv(n, part_num)
    BLOCK_SIZE = triton.next_power_of_2(BLOCK_SIZE)

    out = torch.empty_like(x)
    partial_sum = torch.empty(part_num, dtype=x.dtype)

    grid = (triton.cdiv(n, BLOCK_SIZE),)

    scan_part_sum_kernel[grid](
        x, out, partial_sum, n, part_num,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    add_base_sum_kernel[grid](
        out, partial_sum, n, part_num,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out
```

## 完整可运行程序

```Python
import torch
import triton
import triton.language as tl
from triton.backends.spine_triton.driver import CPUDriver

# 启用 Spine Triton CPU 驱动
triton.runtime.driver.set_active(CPUDriver())


@triton.jit(do_not_specialize=["n_elements"])
def scan_part_kernel(
    inp, local_out, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    # 第一阶段：各块独立计算前缀和，越界位置补零
    values = tl.load(inp + offset, mask=mask, other=0.0)
    result = tl.cumsum(values, axis=0)
    tl.store(local_out + offset, result, mask=mask)


@triton.jit(do_not_specialize=["n_elements"])
def add_base_kernel(
    local_out, offsets, out, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    # 第三阶段：局部前缀和加上当前块之前所有块的总和
    base = tl.load(offsets + pid)
    values = tl.load(local_out + offset, mask=mask, other=0.0)
    tl.store(out + offset, values + base, mask=mask)


def cumsum(x, debug=False):
    if (
        x.device.type != "cpu"
        or x.dtype != torch.float32
        or x.ndim != 1
        or not x.is_contiguous()
        or x.numel() == 0
    ):
        raise ValueError(
            "输入必须是非空、连续的一维 CPU float32 张量"
        )

    n = x.numel()

    # 块大小取 2 的幂，并限制为最多 512
    block_size = min(
        512,
        triton.next_power_of_2(triton.cdiv(n, 2)),
    )
    num_blocks = triton.cdiv(n, block_size)
    grid = (num_blocks,)

    local_out = torch.empty_like(x)
    out = torch.empty_like(x)
    offsets = torch.zeros(num_blocks, dtype=x.dtype)

    scan_part_kernel[grid](
        x, local_out, n,
        BLOCK_SIZE=block_size,
    )

    if num_blocks > 1:
        # 完整块的末尾局部结果就是块总和；
        # 最后一块的总和不参与任何后续块的 base。
        block_sums = local_out[
            block_size - 1::block_size
        ][:num_blocks - 1].contiguous()

        # 第二阶段：用 PyTorch 计算块总和的前缀和
        # offsets[0]=0，offsets[p] 为前 p 个块的总和。
        offsets[1:] = torch.cumsum(block_sums, dim=0)

    add_base_kernel[grid](
        local_out, offsets, out, n,
        BLOCK_SIZE=block_size,
    )

    if debug:
        print(
            f"n={n}, BLOCK_SIZE={block_size}, grid={grid}",
            flush=True,
        )
        print("各块 base：", offsets.tolist(), flush=True)
        print("结果末尾：", out[-5:].tolist(), flush=True)

    return out


if __name__ == "__main__":
    print(f"torch={torch.__version__}", flush=True)
    print(f"triton={triton.__version__}", flush=True)
    print(f"triton path={triton.__file__}", flush=True)

    torch.manual_seed(0)

    lengths = [
        5, 1, 2, 3, 4, 6, 7, 8,
        64, 511, 512, 513, 1000,
        1024, 1025, 2048, 4096,
    ]

    for n in lengths:
        print(f"\n开始测试 n={n}", flush=True)

        x = torch.randn(n, dtype=torch.float32)
        got = cumsum(x, debug=n in (5, 4096))
        ref = torch.cumsum(x, dim=0)

        torch.testing.assert_close(
            got, ref, atol=1e-3, rtol=1e-3,
        )

        print(
            f"PASS f32 n={n}  "
            f"max diff={(got - ref).abs().max().item():.2e}",
            flush=True,
        )
```

上述程序是根据官方程序修改得到的。原来的官方程序在当前环境中运行出现两个问题：

1. ~~块总和为 NaN，污染最终结果。~~

解决方式：不再使用 `tl.sum(inp_vals)` 生成块总和，改为读取局部 scan 的块末尾结果。

2. ~~大块 scan 发生段错误。~~~~`n=2048、BLOCK_SIZE=1024`~~~~ 通过；~~~~`n=4096、BLOCK_SIZE=2048`~~~~ 在第一趟 kernel 调用期间崩溃。~~

解决方式：将块大小限制为 `512`，把 `4096` 个元素拆成 8 块。

为了支持更多块，我还补充了块总和的前缀扫描。官方教学版默认两块，后一块直接加前一块总和即可。改成多块后，每块需要的是“前面所有块的总和”。当前通过下面这一步生成正确的 base：

```Python
offsets[1:] = torch.cumsum(block_sums, dim=0)
```

修改后的完整计算过程是：

```Plain Text
Spine Triton：各块局部 scan
        ↓
PyTorch：计算块总和的前缀和
        ↓
Spine Triton：每块加上对应 base
```

运行结果：

```Python
torch=2.8.0+spacemit.0
triton=3.6.0
triton path=/home/bianbu/spine-venv312/lib/python3.12/site-packages/triton/__init__.py

开始测试 n=5
n=5, BLOCK_SIZE=4, grid=(2,)
各块 base： [0.0, -0.36279094219207764]
结果末尾： [1.5409960746765137, 1.2475671768188477, -0.9312222003936768, -0.36279094219207764, -1.4473133087158203]
PASS f32 n=5  max diff=0.00e+00

开始测试 n=1
PASS f32 n=1  max diff=0.00e+00

开始测试 n=2
PASS f32 n=2  max diff=0.00e+00

开始测试 n=3
PASS f32 n=3  max diff=0.00e+00

开始测试 n=4
PASS f32 n=4  max diff=2.98e-08

开始测试 n=6
PASS f32 n=6  max diff=0.00e+00

开始测试 n=7
PASS f32 n=7  max diff=0.00e+00

开始测试 n=8
PASS f32 n=8  max diff=2.38e-07

开始测试 n=64
PASS f32 n=64  max diff=9.54e-07

开始测试 n=511
PASS f32 n=511  max diff=4.05e-06

开始测试 n=512
PASS f32 n=512  max diff=3.81e-06

开始测试 n=513
PASS f32 n=513  max diff=6.68e-06

开始测试 n=1000
PASS f32 n=1000  max diff=4.77e-06

开始测试 n=1024
PASS f32 n=1024  max diff=7.63e-06

开始测试 n=1025
PASS f32 n=1025  max diff=7.63e-06

开始测试 n=2048
PASS f32 n=2048  max diff=1.03e-05

开始测试 n=4096
n=4096, BLOCK_SIZE=512, grid=(8,)
各块 base： [0.0, -13.161455154418945, 6.927219390869141, -28.03194808959961, -36.397247314453125, -27.118091583251953, -42.0013313293457, -49.36884689331055]
结果末尾： [-70.60450744628906, -70.6566162109375, -70.39283752441406, -70.33534240722656, -70.34789276123047]
PASS f32 n=4096  max diff=7.63e-06
```

