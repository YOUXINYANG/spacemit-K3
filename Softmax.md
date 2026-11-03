# Softmax

## 简介

对于一行输入 $x$，Softmax 的定义是：

$y_i=\frac{e^{x_i}}{\sum_j e^{x_j}}$

作用：在 attention 中，负责把分数转换成权重。

以 $x=[1,2,3]$ 为例：

```Plain Text
取指数：  [2.718, 7.389, 20.086]
求总和：  30.193
归一化：  [0.090, 0.245, 0.665]
```

输出特点：

1. 每项为正，整行之和是 1；

2. 输入越大，对应的权重越大。

由于直接计算指数可能产生溢出，教程中采用下面的稳定写法，与原公式等价。

$m=\max_j x_j,\qquad y_i=\frac{e^{x_i-m}}{\sum_j e^{x_j-m}}$

## Pytorch 实现

```Python
import torch

x = torch.tensor([[1000., 1001., 1002.]], dtype=torch.float32)

# 直接按定义计算
exp_x = torch.exp(x)
naive = exp_x / exp_x.sum(dim=-1, keepdim=True)

# 先减去每行最大值，再计算
row_max = x.max(dim=-1, keepdim=True).values
shifted = x - row_max
numerator = torch.exp(shifted)
stable = numerator / numerator.sum(dim=-1, keepdim=True)

print("直接取指数：", exp_x)
print("直接归一化：", naive)
print("减去最大值：", shifted)
print("稳定版结果：", stable)
print("每行之和：  ", stable.sum(dim=-1))
print("PyTorch结果：", torch.softmax(x, dim=-1))
```

`dim=-1` 表示沿最后一维计算。

`keepdim=True` 保留 `(行数, 1)` 的形状，方便把每行的最大值或总和应用到该行所有元素。

数据计算流程：

1. 求每行最大值；

2. 相减并取指数；

3. 求每行指数之和；

4. 相除；

## one\_tile 版完整源代码

```Python
# softmax.py — 每 program 一行的 softmax（行装得下单 block）
import torch
import triton
import triton.language as tl
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())


@triton.jit
def softmax_kernel(output_ptr, input_ptr, in_stride, out_stride, n_cols,
                   BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)                    # 每 program 一行
    row_start_ptr = input_ptr + row_idx * in_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # 整行 load 进寄存器。关键：other=-inf ——
    # 越界 lane 必须是 max 的“吸收元”（-inf 不可能是最大值），
    # 且 exp(-inf)=0 对分母也无贡献。填 0 就全错了。
    row = tl.load(row_start_ptr + col_offsets,
                  mask=col_offsets < n_cols, other=-float('inf'))

    row_f32 = row.to(tl.float32)                  # ← 固定规则：exp 前先升 f32！
    row_minus_max = row_f32 - tl.max(row_f32, axis=0)   # 数值稳定
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    out_ptrs = output_ptr + row_idx * out_stride + col_offsets
    tl.store(out_ptrs, softmax_output.to(output_ptr.dtype.element_ty),
             mask=col_offsets < n_cols)


def softmax(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)   # 一行装进一个 block
    y = torch.empty_like(x)
    softmax_kernel[(n_rows,)](y, x, x.stride(0), y.stride(0), n_cols,
                              BLOCK_SIZE=BLOCK_SIZE)
    return y


if __name__ == "__main__":
    torch.manual_seed(0)
    # f32 场景（仓库示例同款，1823×781 非 2 幂行长）
    x = torch.randn(1823, 781)
    y_triton = softmax(x)
    y_torch = torch.softmax(x, axis=1)
    torch.testing.assert_close(y_triton, y_torch)
    print("PASS f32 1823x781")

    # f16 场景：kernel 内先 .to(tl.float32) 再 exp，才过得了 dtype 检查
    x16 = torch.randn(128, 256, dtype=torch.float16) * 5   # ×5 制造大 logits
    y16 = softmax(x16)
    ref16 = torch.softmax(x16.float(), dim=-1).half()
    torch.testing.assert_close(y16, ref16, atol=1e-3, rtol=1e-3)
    print("PASS f16 128x256 (large logits)")
```

## 学习源码

### Program 和 block 分配

Softmax 也是按行完成计算。每个 Program 负责一行计算。

启动 kernal ：

```Python
softmax_kernel[(n_rows,)](...)
```

`(n_rows,)` 是一维启动网格。3 行就启动 3 个 program，kernel 内通过下面这句知道自己处理哪一行：

```Python
row_idx = tl.program_id(0)
```

每个 Program 用一个 block 容纳整行数据。

one\_tile 版本中，每个 program 一次处理整行数据。

教程将 block 的长度设为列数向上取整到最近的 2 的幂：

```Plain Text
BLOCK_SIZE = triton.next_power_of_2(n_cols)
```

例如：

kernel 内：

```Python
col_offsets = tl.arange(0, BLOCK_SIZE)
```

当 `BLOCK_SIZE=8` 时，它生成：

```Plain Text
列偏移： [0, 1, 2, 3, 4, 5, 6, 7]
有效性： [✓, ✓, ✓, ✓, ✓, ✗, ✗, ✗]
```

多出来的位置，用 mask 和 `-inf` 处理。

```Python
row = tl.load(
    row_start_ptr + col_offsets,
    mask=col_offsets < n_cols,
    other=-float("inf"),
)
```

- `mask` 决定哪些位置真正读取内存，避免越界访问。

- `other` 决定无效位置在后续运算中使用什么值。

填 `-inf` 的原因：同时满足后面的两步计算。

- 求最大值：\-inf 不影响有效数据的最大值。

- 取指数：exp\(\-inf\) = 0，不影响分母求和计算。

### 计算过程

1. 从内存加载一行

2. 求最大值

3. 减去最大值，取指数，求和

4. 除以总和

5. 将结果写回内存

### Kernal 定义和数据加载

```Python
@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    in_stride,
    out_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * in_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)

    row = tl.load(
        row_start_ptr + col_offsets,
        mask=col_offsets < n_cols,
        other=-float("inf"),
    )
```

传进 kernal 的参数含义：

内存指针赋值的部分和之前算子相似。不重复解释。

### 在 kernal 内稳定完成 Softmax

```Python
row_f32 = row.to(tl.float32)
row_minus_max = row_f32 - tl.max(row_f32, axis=0)
numerator = tl.exp(row_minus_max)
denominator = tl.sum(numerator, axis=0)
softmax_output = numerator / denominator
```

转换为 FP32：

```Python
row_f32 = row.to(tl.float32)
```

在本教程使用的 Triton 接口中，`tl.exp` 要求输入为 float32 或 float64，所以 float16 输入需要先显式转换。

求最大值，再从每个元素中减去它：

```Python
row_minus_max = row_f32 - tl.max(row_f32, axis=0)
```

`axis` 描述当前 Triton 张量的维度，不是原始输入矩阵的维度。

对每个位置取指数：

```Python
numerator = tl.exp(row_minus_max)
```

求指数和：

```Python
denominator = tl.sum(numerator, axis=0)
```

每个指数值除以总和：

```Python
softmax_output = numerator / denominator
```

计算中的形状变化：

```Python
row_f32          向量 (8,)
    ↓ tl.max
row_max          标量
    ↓ 向量减标量
row_minus_max    向量 (8,)
    ↓ tl.exp
numerator        向量 (8,)
    ↓ tl.sum
denominator      标量
    ↓ 向量除标量
softmax_output   向量 (8,)
```

### 结果写回

计算输出地址，和输入地址计算保持一致。

```Python
out_ptrs = output_ptr + row_idx * out_stride + col_offsets
```

转换输出类型，再写入内存：

```Python
tl.store(
    out_ptrs,
    softmax_output.to(output_ptr.dtype.element_ty),
    mask=col_offsets < n_cols,
)
```

### 启动 kernal

教程中的包装函数：

```Python
def softmax(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    y = torch.empty_like(x)

    softmax_kernel[(n_rows,)](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return y
```

```Python
y = torch.empty_like(x)
```

它分配与 `x` 形状、dtype 和设备相同的输出张量。

`empty` 表示初始内容未初始化，kernel 会覆盖所有有效输出位置，所以不需要先填零。

参数对应关系：

## 运行源码

在k3实例中新建 softmax\.py ，写入：

```Python
import torch
import triton
import triton.language as tl
from triton.backends.spine_triton.driver import CPUDriver

# 选择 Spine Triton 的 CPU 后端
triton.runtime.driver.set_active(CPUDriver())


@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    in_stride,
    out_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # 每个 program 负责一行
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # 读取当前行，补齐位置填 -inf
    row_start_ptr = input_ptr + row_idx * in_stride
    row = tl.load(
        row_start_ptr + col_offsets,
        mask=mask,
        other=-float("inf"),
    )

    # 在 float32 下计算稳定 Softmax
    row_f32 = row.to(tl.float32)
    shifted = row_f32 - tl.max(row_f32, axis=0)
    numerator = tl.exp(shifted)
    denominator = tl.sum(numerator, axis=0)
    result = numerator / denominator

    # 转换为输出类型，只写回有效位置
    out_ptrs = output_ptr + row_idx * out_stride + col_offsets
    tl.store(
        out_ptrs,
        result.to(output_ptr.dtype.element_ty),
        mask=mask,
    )


def softmax(x):
    n_rows, n_cols = x.shape
    block_size = triton.next_power_of_2(n_cols)
    y = torch.empty_like(x)

    softmax_kernel[(n_rows,)](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_cols,
        BLOCK_SIZE=block_size,
    )
    return y


if __name__ == "__main__":
    torch.manual_seed(0)

    # float32：非 2 的幂列数，同时验证 mask
    x32 = torch.randn(1823, 781, dtype=torch.float32)
    y32 = softmax(x32)
    ref32 = torch.softmax(x32, dim=-1)

    torch.testing.assert_close(y32, ref32)
    print("PASS f32 1823x781")

    # float16：放大输入，验证升精度后的计算
    x16 = torch.randn(128, 256, dtype=torch.float16) * 5
    y16 = softmax(x16)
    ref16 = torch.softmax(x16.float(), dim=-1).half()

    torch.testing.assert_close(
        y16, ref16, atol=1e-3, rtol=1e-3
    )
    print("PASS f16 128x256 (large logits)")

    # 打印一些便于观察的结果
    print("f32 第一行前 5 项：", y32[0, :5])
    print("f32 前 3 行的行和：", y32[:3].sum(dim=-1))
    print("f16 输出类型：", y16.dtype)
```

该代码沿用教程 4\.3 节的计算逻辑，增加了最后三行打印方便观察。

测试失败：实际测试在 float32 结果比较处失败，几乎全部元素不匹配。

排查发现两个问题：

1. 原示例的大 grid 启动没有写入输出：对 `(1823, 781)` 单次启动，全部输出保留初始化值 `-123`，同一输入改为每批最多 16 行后通过。

2. 短 block 计算产生 nan：对相同 `(1, 5)` 输入，`BLOCK=8` 输出全 NaN，改为 `BLOCK=32` 后通过。

