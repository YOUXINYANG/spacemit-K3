# GEMM

## 简介

通用矩阵乘法，最基础的算子

数学定义：`C[M,N] = A[M,K] @ B[K,N]`

总计算量是 2MNK 次浮点运算，一次乘和一次加算两次浮点运算。

## Pytorch 参考实现

用以下命令创建文件。

程序主要功能是使用Pytorch生成两个随机矩阵，完成一次 GEMM，并打印矩阵的形状。

```Python
cat > gemm_torch.py <<'PY'
import torch

torch.manual_seed(0)

M, N, K = 256, 256, 256

A = torch.randn((M, K), dtype=torch.float16)
B = torch.randn((K, N), dtype=torch.float16)

C = torch.matmul(A, B)

print("A.shape =", A.shape)
print("B.shape =", B.shape)
print("C.shape =", C.shape)
print("GEMM completed!")
PY

python3 gemm_torch.py
```

## mm\_block\_ptr

官方源码：mm\_block\_ptr\.py

是一个使用 `tl.make_block_ptr` 实现矩阵乘法的完整示例。

```plaintext
(spine-venv312) bianbu@k3:~/spine-triton$ python3 python/examples/mm_block_ptr.py
c_triton tensor([[-16.2812, -38.6250,   1.8633,  ...,  -2.6855,  -6.2422,  12.2031],
                 [-16.2969,  -8.4922,  -6.5547,  ..., -18.4688, -17.4688,   9.3281],
                 [  3.1328,   8.9609,  -0.1647,  ..., -16.4531,   4.2656,  16.1875],
                 ...,
                 [-23.5781,  -0.9995,  12.7422,  ...,  32.1562, -19.0938, -23.2812],
                 [ -4.4805,  -6.9414,   7.3438,  ...,  16.6094,  -9.5547, -13.1484],
                 [-31.7500, -17.2031,  -6.4180,  ..., -12.3750, -12.0469,  -2.6700]],
                dtype=torch.float16)
c_torch tensor([[-16.2812, -38.6250,   1.8633,  ...,  -2.6855,  -6.2422,  12.2031],
                [-16.2969,  -8.4922,  -6.5547,  ..., -18.4688, -17.4688,   9.3281],
                [  3.1328,   8.9609,  -0.1647,  ..., -16.4531,   4.2656,  16.1875],
                ...,
                [-23.5781,  -0.9995,  12.7422,  ...,  32.1562, -19.0938, -23.2812],
                [ -4.4805,  -6.9414,   7.3438,  ...,  16.6094,  -9.5547, -13.1484],
                [-31.7500, -17.2031,  -6.4180,  ..., -12.3750, -12.0469,  -2.6700]],
               dtype=torch.float16)
✅PASS: mm_block_ptr (256x256) @ (256x256) = (256x256)
```

上图显示 spine triton 的结果和 pytorch 计算结果一致。

## 矩阵分块

假设需要计算：

```math
C_{256\times256}=A_{256\times256}\times B_{256\times256}
```

在 Pytorch 中，只需要：

```Python
C = torch.matmul(A, B)
```

在 triton 中，通常会将输出矩阵 C 划分成若干小矩阵，每个 program 负责计算其中一个小块，称为 tile。

假设每个 tile 是 32\*32，那么原本的 256\*256 矩阵被划分成 8\*8 64个 tile。

每个 program 负责 C 中的一块，执行同一个 kernal，但通过不同的 ID 确定自己需要计算哪一块。

## 理解官方源码

源码：

```Python
*"""*
*最简单的矩阵乘法示例，使用 tl.make_block_ptr 进行内存访问。*
*C = A @ B, 其中 A: [M, K], B: [K, N], C: [M, N]*
*"""*

import torch
import triton
import triton.language as tl
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())


@triton.jit # 将函数声明为 triton kernal
def mm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # K == BLOCK_K，单次加载完整 K 维度
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )
    a = tl.load(a_block_ptr, boundary_check=(0, 1))
    b = tl.load(b_block_ptr, boundary_check=(0, 1))
    acc = tl.dot(a, b)

    # 存储结果
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(c_block_ptr, acc.to(c_ptr.dtype.element_ty), boundary_check=(0, 1))


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"incompatible dimensions: {K} vs {K2}"
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, triton.next_power_of_2(K)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    mm_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return c


if __name__ == "__main__":
    M, N, K = 256, 256, 256
    a = torch.randn((M, K), device="cpu", dtype=torch.float16)
    b = torch.randn((K, N), device="cpu", dtype=torch.float16)

    c_triton = matmul(a, b)
    c_torch = torch.matmul(a, b)

    print("c_triton", c_triton)
    print("c_torch", c_torch)

    torch.testing.assert_close(c_triton, c_torch, atol=1e-2, rtol=1e-2)
    print(f"✅ PASS: mm_block_ptr ({M}x{K}) @ ({K}x{N}) = ({M}x{N})")
```

### 设置 spine triton 后端

```Python
import torch
import triton
import triton.language as tl

from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
```

前三行导入 Pytorch 和 Triton。

- `triton`：提供 JIT 编译、Kernel 启动等功能。

- `triton.language`：提供 `tl.load`、`tl.dot`、`tl.store` 等编写 Kernel 所需的接口。

- `CPUDriver()`：选择 Spine\-Triton 的 CPU 后端，使其能够通过对应的编译和运行通路在 K3 平台执行。

### 定义计算内核

```Python
@triton.jit
def mm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    ...
):
```

`@triton.jit` 表示使用 Triton 的 JIT 编译机制处理这个函数。

`a_ptr`、`b_ptr`、`c_ptr` 分别表示三个矩阵的地址。

与普通 Python 函数不同，`mm_kernel()` 中的计算会由 Triton 编译器编译，然后通过后端执行。

### 获取 program ID

```Python
pid_m = tl.program_id(0)
pid_n = tl.program_id(1)
```

这两行代码用来确定当前 Program 负责的 Tile。

例如：

- `(pid_m, pid_n) = (0, 0)`：负责 C\[0:32, 0:32\]。

- `(pid_m, pid_n) = (0, 1)`：负责 C\[0:32, 32:64\]。

- `(pid_m, pid_n) = (1, 0)`：负责 C\[32:64, 0:32\]。

每个 Program 通过自己的 ID 定位要处理的数据。

### 加载输入矩阵

```Bash
a_block_ptr = tl.make_block_ptr(
    base=a_ptr,
    shape=(M, K),
    strides=(stride_am, stride_ak),
    offsets=(pid_m * BLOCK_M, 0),
    block_shape=(BLOCK_M, BLOCK_K),
    order=(1, 0),
)
```

`tl.make_block_ptr()`的作用是描述矩阵中的一个数据块，包括它的起始地址、大小、布局和偏移量。

```Bash
a = tl.load(a_block_ptr, boundary_check=(0, 1))
b = tl.load(b_block_ptr, boundary_check=(0, 1))
```

把这两个矩阵块读取出来。`boundary_check` 表示对两个维度进行越界检查。

### 执行矩阵乘法

```Bash
acc = tl.dot(a, b)
```

`tl.dot()` 执行块级矩阵乘法，其结果就是 C 中的一个 32×32 Tile。

在 K3 的 FP16 GEMM 路径中，教程采用 FP32 累加以保持计算精度。

### 保存计算结果

```Bash
tl.store(
    c_block_ptr,
    acc.to(c_ptr.dtype.element_ty),
    boundary_check=(0, 1)
)
```

这里的 `acc.to(...)` 将计算结果转换为输出矩阵 C 对应的数据类型，即本例的 FP16。

随后 `tl.store()` 把结果写回 C 对应的 Tile。

到此，一个 program 的工作已经完成。

### Kernal 如何被启动

```Python
BLOCK_M, BLOCK_N, BLOCK_K = (
    32, 32, triton.next_power_of_2(K)
)

grid = (
    triton.cdiv(M, BLOCK_M),
    triton.cdiv(N, BLOCK_N)
)
```

对于 M=N=K=256：

- `BLOCK_M=32`

- `BLOCK_N=32`

- `BLOCK_K=256`

- `grid=(8,8)`

`triton.cdiv()` 表示向上取整除法。

```Python
mm_kernel[grid](...)
```

表示：按照指定的 grid 组织 kernal 的执行。

可以理解为：一共有 64 个计算任务，每个任务得到对应的 Program ID，执行 `mm_kernel()` 并计算输出矩阵 C 的一小块。

## K\-loop

在上述程序中，如果维度 K 很大，无法一次加载完整的输入数据时，怎样把矩阵乘法拆成多次小规模计算？

### 基本原理

假设：

```math
C=A\times B
```

其中：

- A 为 256×256

- B 为 256×256

- 输出 Tile 为 32×32

- `BLOCK_K=64`

因为 K=256，所以循环次数是 256/64 = 4\.

每次执行：

```math
(32\times64)\times(64\times32) =(32\times32)
```

每次计算得到一个部分结果，再把四个部分结果累加，得到最终的 C Tile。

### 在 K3 上实现 K\-loop GEMM

Python 程序：

```Python
import torch
import triton
import triton.language as tl

from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())


@triton.jit
def gemm_kloop_kernel(
    A, B, C,
    M, N, K,
    sa0, sa1, sb0, sb1, sc0, sc1,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    # 1. 当前 Program 的二维 ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # 2. 当前输出 Tile 的行列下标
    rows = pid_m * BM + tl.arange(0, BM)
    cols = pid_n * BN + tl.arange(0, BN)
    depth = tl.arange(0, BK)

    # 3. 创建 FP32 累加器
    acc = tl.zeros((BM, BN), dtype=tl.float32)

    # 4. 沿 K 维分块循环
    for t in range(tl.cdiv(K, BK)):
        k = t * BK + depth

        # A 的 BM×BK 子块地址
        a_ptrs = A + rows[:, None] * sa0 + k[None, :] * sa1

        # B 的 BK×BN 子块地址
        b_ptrs = B + k[:, None] * sb0 + cols[None, :] * sb1

        # 加载 A、B，越界部分用 0 填充
        a = tl.load(
            a_ptrs,
            mask=(rows[:, None] < M) & (k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(k[:, None] < K) & (cols[None, :] < N),
            other=0.0,
        )

        # 矩阵乘法并累加
        acc += tl.dot(a, b)

    # 5. 写回结果
    c_ptrs = C + rows[:, None] * sc0 + cols[None, :] * sc1
    tl.store(
        c_ptrs,
        acc.to(C.dtype.element_ty),
        mask=(rows[:, None] < M) & (cols[None, :] < N),
    )


def matmul(a, b):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2

    c = torch.empty((M, N), dtype=a.dtype, device=a.device)

    BM, BN, BK = 32, 32, 64
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))

    gemm_kloop_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BM=BM, BN=BN, BK=BK,
    )
    return c


if __name__ == "__main__":
    torch.manual_seed(0)

    M, N, K = 256, 256, 256
    A = torch.randn((M, K), dtype=torch.float16)
    B = torch.randn((K, N), dtype=torch.float16)

    C_triton = matmul(A, B)
    C_torch = torch.matmul(A, B)

    torch.testing.assert_close(
        C_triton, C_torch, atol=1e-2, rtol=1e-2
    )

    error = (C_triton.float() - C_torch.float()).abs().max()

    print("M, N, K =", M, N, K)
    print("BLOCK_K =", 64)
    print("K-loop rounds =", triton.cdiv(K, 64))
    print("Maximum absolute error =", error.item())
    print("PASS: K-loop GEMM")
```

这份程序沿用官方教程中手写矩阵指针和 `tl.load` 的思路，但对 K 维掩码进行了调整：每一轮使用当前实际的 K 坐标，以便正确处理最后一个不足 BK 大小的块。

输出：

```Plain Text
M, N, K = 256 256 256
BLOCK_K = 64
K-loop rounds = 4
Maximum absolute error = 0.015625
PASS: K-loop GEMM
```

最大绝对误差的具体数值与编译实现有关，不必完全为零。`PASS` 说明它通过了设定容差下的 PyTorch 正确性检查。

## 理解 K\-loop 核心代码

### `tl.arange()`：生成索引向量

```Python
rows = pid_m * BM + tl.arange(0, BM)
cols = pid_n * BN + tl.arange(0, BN)
depth = tl.arange(0, BK)
```

假设当前：

```Python
pid_m = 0
pid_n = 0

BM = 32
BN = 32
BK = 64
```

那么生成：

```Python
rows  = [0, 1, 2, ..., 31]
cols  = [0, 1, 2, ..., 31]
depth = [0, 1, 2, ..., 63]
```

`rows` 和 `cols` 决定了当前 Program 需要计算 C 的哪一块，而 `depth` 决定每轮沿 K 方向读取哪些元素。

### `[:, None]`：广播机制

```Python
a_ptrs = A + rows[:, None] * sa0 + k[None, :] * sa1
```

这里涉及 Python / NumPy / PyTorch 中常见的广播机制（Broadcasting）。

- `rows[:, None]`：将形状为 `(32,)` 的向量变成 `(32,1)`。

- `k[None, :]`：将形状为 `(64,)` 的向量变成 `(1,64)`。

两者广播后，形成一个 32×64 的地址矩阵。

假设 A 是行优先连续存储，访问 `A[i,j]` 的元素地址可以表示为：

```math
\mathrm{Addr}(A[i,j]) =\mathrm{Base}(A)+i\cdot sa0+j\cdot sa1
```

`sa0` 和 `sa1` 分别表示矩阵 A 在两个维度上的 步长（Stride）。

- `sa0`：行步长，表示从 A 的一行移动到下一行，需要跨过多少个元素。

- `sa1`：列步长，表示从 A 的一列移动到下一列，需要跨过多少个元素。

这里的 stride 以元素为单位，由 Triton 张量指针运算自动考虑元素字节宽度。

因此，`a_ptrs` 就包含了当前 A 子块中所有元素对应的地址。

### `mask` 和 `other=0.0`：防止越界

```Python
a = tl.load(
    a_ptrs,
    mask=(rows[:, None] < M) & (k[None, :] < K),
    other=0.0,
)
```

上述代码含义：只有矩阵下标处于有效范围时，才进行加载，对于掩码为假的位置，返回0。

这保证了 K 不是 BK 整数倍时也能完成正确计算。

### `acc += tl.dot(a,b)`：保持 FP32 累加

```Python
acc = tl.zeros((BM, BN), dtype=tl.float32)

for t in range(tl.cdiv(K, BK)):
    ...
    acc += tl.dot(a, b)
```

虽然输入 A、B 使用 FP16，但累加器 `acc` 使用 FP32。原因是浮点数精度问题。

FP16 只有约 11 位有效二进制精度，而 FP32 有约 24 位。矩阵乘法需要进行大量乘加，如果每次部分和都转换为 FP16，再进行下一次累加，就可能引入更明显的舍入误差。

因此，正确的设计思路是：

> FP16 输入矩阵 A,B
> 
> ↓
> 
> 多次 FP16 矩阵乘法
> 
> ↓
> 
> FP32 累加器，保留每一轮的部分和
> 
> ↓
> 
> 最终转换为 FP16 写回 C
> 
> 

官方教程特别强调了这一点：早期实现如果在 K\-loop 内将每个部分和过早截断为 FP16，可能导致明显的数值错误。

## K\-loop 与 K3 硬件的关系

K3 的矩阵计算实现受到片上临时存储空间、输出 Tile 规模以及数据类型等限制。

教程给出的典型约束包括：

基本矛盾：Tile 越大，可能带来更高的数据复用率；但 Tile 太大，又会受到片上存储和计算资源的约束。

优化目标：在计算并行度、内存访问、数据复用和硬件资源之间寻找平衡。

## K3 矩阵引擎显式编排

下面学习：如何使用 K3 专用的 smt 接口，显式控制矩阵分块、数据布局、片上存储与计算过程。

### test\_smt\_mm

这是官方源码：

```Python
import torch
import triton
import triton.language as tl
import triton.language.extra.smt as smt
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())


@triton.jit
def mm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SPLIT_M: tl.constexpr,
    SPLIT_MN: tl.constexpr,
    SPLIT_K: tl.constexpr,
    SUB_BLK_M: tl.constexpr,
    SUB_BLK_N: tl.constexpr,
    SUB_BLK_K: tl.constexpr,
    MICRO_M: tl.constexpr,
    MICRO_K: tl.constexpr,
    MICRO_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=[M, K],
        strides=[stride_am, stride_ak],
        offsets=[pid_m * BLOCK_SIZE_M, 0],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
        order=[1, 0],
    )

    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=[K, N],
        strides=[stride_bk, stride_bn],
        offsets=[0, pid_n * BLOCK_SIZE_N],
        block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
        order=[1, 0],
    )

    if SPLIT_M:
        b_descriptor_load = smt.descriptor_load(b_block_ptr, (0, 0))
        b = smt.view(b_descriptor_load, (0, 0), (BLOCK_SIZE_K, BLOCK_SIZE_N), (MICRO_K, MICRO_N))
        sub_num = (min(BLOCK_SIZE_M, M - BLOCK_SIZE_M * pid_m) + SUB_BLK_M - 1) // SUB_BLK_M
        for s in smt.parallel(0, sub_num):
            a_descriptor_load = smt.descriptor_load(a_block_ptr, (0, 0))
            a = smt.view(a_descriptor_load, (s * SUB_BLK_M, 0), (SUB_BLK_M, BLOCK_SIZE_K), (MICRO_M, MICRO_K))
            accumulator = smt.dot(a, b)
            accumulator = smt.view(accumulator, (0, 0), (SUB_BLK_M, BLOCK_SIZE_N), (1, 1))
            c = accumulator.to(c_ptr.dtype.element_ty)
            c_block_ptr = tl.make_block_ptr(
                base=c_ptr,
                shape=[M, N],
                strides=[stride_cm, stride_cn],
                offsets=[pid_m * BLOCK_SIZE_M + s * SUB_BLK_M, pid_n * BLOCK_SIZE_N],
                block_shape=[SUB_BLK_M, BLOCK_SIZE_N],
                order=[1, 0],
            )
            tl.store(c_block_ptr, c, boundary_check=(0, 1))

    elif SPLIT_MN:
        sub_num_m = (min(BLOCK_SIZE_M, M - BLOCK_SIZE_M * pid_m) + SUB_BLK_M - 1) // SUB_BLK_M
        sub_num_n = (min(BLOCK_SIZE_N, N - BLOCK_SIZE_N * pid_n) + SUB_BLK_N - 1) // SUB_BLK_N
        total_sub_blocks = sub_num_m * sub_num_n
        b_alloc_ptr = smt.alloc(shape=[BLOCK_SIZE_K, BLOCK_SIZE_N], type=b_ptr.dtype.element_ty)
        b_alloc_view_ptr = smt.view(b_alloc_ptr, (0, 0), (BLOCK_SIZE_K, BLOCK_SIZE_N), (MICRO_K, MICRO_N))
        for s in smt.parallel(0, total_sub_blocks):
            s_m = s // sub_num_n
            s_n = s % sub_num_n
            a_descriptor_load = smt.descriptor_load(a_block_ptr, (0, 0))
            a = smt.view(a_descriptor_load, (s_m * SUB_BLK_M, 0), (SUB_BLK_M, BLOCK_SIZE_K), (MICRO_M, MICRO_K))
            b_alloc_sub_ptr = smt.view(b_alloc_view_ptr, (0, s_n * SUB_BLK_N), (BLOCK_SIZE_K, SUB_BLK_N))
            if s_m == 0:
                b_descriptor_load = smt.descriptor_load(b_block_ptr, (0, 0))
                b = smt.view(b_descriptor_load, (0, s_n * SUB_BLK_N), (BLOCK_SIZE_K, SUB_BLK_N), (MICRO_K, MICRO_N))
                tl.store(b_alloc_sub_ptr, b, boundary_check=(0, 1, 2, 3))

            b_alloc = tl.load(b_alloc_sub_ptr, boundary_check=(0, 1, 2, 3))
            accumulator = smt.dot(a, b_alloc)
            accumulator = smt.view(accumulator, (0, 0), (SUB_BLK_M, SUB_BLK_N), (1, 1))
            c = accumulator.to(c_ptr.dtype.element_ty)
            c_block_ptr = tl.make_block_ptr(
                base=c_ptr,
                shape=[M, N],
                strides=[stride_cm, stride_cn],
                offsets=[pid_m * BLOCK_SIZE_M + s_m * SUB_BLK_M, pid_n * BLOCK_SIZE_N + s_n * SUB_BLK_N],
                block_shape=[SUB_BLK_M, SUB_BLK_N],
                order=[1, 0],
            )
            tl.store(c_block_ptr, c, boundary_check=(0, 1))

    elif SPLIT_K:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=a_ptr.type.element_ty)
        accumulator = smt.view(accumulator, (0, 0), (BLOCK_SIZE_M, BLOCK_SIZE_N), (MICRO_M, MICRO_N))
        sub_num = (K + SUB_BLK_K - 1) // SUB_BLK_K
        for k in tl.range(0, sub_num):
            a_descriptor_load = smt.descriptor_load(a_block_ptr, (0, 0))
            a = smt.view(a_descriptor_load, (0, k * SUB_BLK_K), (BLOCK_SIZE_M, SUB_BLK_K), (MICRO_M, MICRO_K))
            b_descriptor_load = smt.descriptor_load(b_block_ptr, (0, 0))
            b = smt.view(b_descriptor_load, (k * SUB_BLK_K, 0), (SUB_BLK_K, BLOCK_SIZE_N), (MICRO_K, MICRO_N))
            accumulator += smt.dot(a, b)
        accumulator = smt.view(accumulator, (0, 0), (BLOCK_SIZE_M, BLOCK_SIZE_N), (1, 1))
        c = accumulator.to(c_ptr.dtype.element_ty)

        c_block_ptr = tl.make_block_ptr(
            base=c_ptr,
            shape=[M, N],
            strides=[stride_cm, stride_cn],
            offsets=[pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
            order=[1, 0],
        )
        tl.store(c_block_ptr, c, boundary_check=(0, 1))


def triton_mm(a, b):
    # handle non-contiguous inputs if necessary
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    # checks constraints
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    # allocates output
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    # launch kernel
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    BLOCK_SIZE_K = triton.next_power_of_2(K)

    mm_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        SPLIT_M=0,
        SPLIT_MN=0,
        SPLIT_K=1,
        SUB_BLK_M=8,
        SUB_BLK_N=64,
        SUB_BLK_K=256,
        MICRO_M=16,
        MICRO_N=32,
        MICRO_K=8,
    )
    return c


def _run_correctness(M: int, N: int, K: int, dtype=torch.float16, device: str = "cpu"):
    A = torch.randn((M, K), dtype=dtype, device=device, requires_grad=False)
    B = torch.randn((K, N), dtype=dtype, device=device, requires_grad=False)

    C = triton_mm(A, B)
    C_ref = torch.mm(A, B)
    torch.testing.assert_close(C, C_ref, atol=1e-1, rtol=0)
    print("PASS")


# -----------------------------
# Performance benchmark (GOPS)
# -----------------------------
def benchmark_gops(
    M: int = 512,
    N: int = 512,
    K: int = 512,
    iters: int = 100,
    warmup: int = 10,
    dtype=torch.float16,
    device: str = "cpu",
):
    *"""Run matmul benchmark and report GOPS.*

*    GOPS definition here uses GEMM FLOP count: 2 * M * N * K operations.*
*    """*
*    *import time

    A = torch.randn((M, K), dtype=dtype, device=device, requires_grad=False)
    B = torch.randn((K, N), dtype=dtype, device=device, requires_grad=False)

    # Warmup
    for _ in range(warmup):
        _ = triton_mm(A, B)

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = triton_mm(A, B)
    t1 = time.perf_counter()

    avg_s = (t1 - t0) / iters
    ops = 2.0 * float(M) * float(N) * float(K)
    gops = ops / avg_s / 1e9

    print(f"[PERF] M={M}, N={N}, K={K}, iters={iters}, "
          f"avg={avg_s * 1e3:.3f} ms, GOPS={gops:.3f}")


# Run perf by env var:
#   RUN_PERF=1 python test_smt_mm.py
if __name__ == "__main__":
    import os

    M, N, K = 512, 512, 512
    dtype = torch.float16
    device = "cpu"

    _run_correctness(M, N, K, dtype=dtype, device=device)

    if os.getenv("RUN_PERF", "0") == "1":
        benchmark_gops(M=M, N=N, K=K, iters=100, dtype=dtype, device=device)
```

### 运行官方程序

```Python
python3 python/examples/test_smt_mm.py
```

官方源码默认配置为：

```Python
M, N, K = 512, 512, 512

dtype = torch.float16
device = "cpu"
```

这里的 `device="cpu"` 表示使用 CPU 设备上的 PyTorch 张量，实际 Triton Kernel 则通过 Spine\-Triton 的 `CPUDriver` 在 K3 平台执行。

输出：PASS

这里的 `PASS` 表示与 PyTorch 参考结果的误差满足官方测试条件，不表示二者每个元素完全相等。

### SMT 五个核心接口

在程序开头会看到一个新的库。这个库提供了 K3 矩阵计算的专用接口。

```Python
import triton.language.extra.smt as smt
```

### MICRO\-tile

源码中有：

```Python
MICRO_M = 16
MICRO_K = 8
MICRO_N = 32
```

这些参数描述 K3 FP16 矩阵计算使用的基本 MICRO\-tile 形状。

例如一个微块乘法：

```math
A_{16\times8}\times B_{8\times32} =C_{16\times32}
```

但要注意，实际矩阵引擎的执行不一定表现为程序中一次完整的 16×8 与 8×32 乘法指令。这里描述的是供编译器和矩阵引擎使用的数据打包粒度。

### Pack 和 unpack

假设有一个普通的 32\*64 矩阵。

矩阵在内存中通常是按行连续排列的，但为了让硬件矩阵计算单元**高效读取数据**，可能需要把它组织成多个 **MICRO\-tile**。

```plaintext
普通矩阵 32×64                      MICRO-tile: 16×32

+--------------------------+        +------------+------------+
|                          |        |            |            |
|                          |        |   Tile 0   |   Tile 1   |
|                          |        |            |            |
|                          |  --->  +------------+------------+
|                          |        |            |            |
|                          |        |   Tile 2   |   Tile 3   |
|                          |        |            |            |
+--------------------------+        +------------+------------+
```

对于一个 32×64 的矩阵，如果 MICRO\-tile 是 16×32，那么逻辑上会形成 4 个微块。

```Python
packed = smt.view(
    x,
    (0, 0),
    (32, 64),
    (16, 32)
)
```

可以理解为：从矩阵 `x` 的 `(0,0)` 位置开始，提取一个 32×64 的区域，并组织为以 16×32 为粒度的 packed 布局。

如果之后需要回复普通布局，就要用到 unpack 。

```Python
unpacked = smt.view(
    packed,
    (0, 0),
    (32, 64),
    (1, 1)
)
```

`(1,1)` 在官方示例中用于 Unpack。

**注意**

`smt.dot()` 返回的是 packed 布局的数据，不能不经转换就把它当作普通二维矩阵写回。

### SPLIT\_K 策略

`SPLIT_K` 也是沿 K 维分块计算，和 K\-loop 类似，但这一次使用显式矩阵引擎接口。

test\_smt\_mm\.py 中的官方参数是：

```Python
BLOCK_SIZE_M = 128
BLOCK_SIZE_N = 128

SPLIT_M = 0
SPLIT_MN = 0
SPLIT_K = 1

SUB_BLK_K = 256

MICRO_M = 16
MICRO_K = 8
MICRO_N = 32
```

这些是调用 Kernel 时传入的参数，这里用变量形式展示。

输入矩阵大小是 512×512，输出 Tile 是 128×128，总共启动 16 个逻辑 Program。

对于每个 Program：

```math
A_{128\times512}\times B_{512\times128}
```

`SUB_BLK_K=256`，因此沿 K 维分成两次：

```plaintext
SPLIT_K 两轮计算演示：第 1 轮

       A: 128×512                    B: 512×128
+-------------+-------------+       +-------------+
|             |             |       | K=0～255    | <-- 本轮
| K=0～255    | K=256～511   |       | 256×128     |
| 128×256     | 128×256      |   ×   +-------------+  =  +-------------+
|             |             |       | K=256～511  |     | 部分和 1    |
+-------------+-------------+       | 256×128     |     | 128×128     |
      ^                             +-------------+     +-------------+
      本轮

第一轮：K=0～255
(128 × 256) × (256 × 128)
计算第一个部分和，存入累加器。
```

```plaintext
SPLIT_K 两轮计算演示：第 2 轮

       A: 128×512                    B: 512×128
+-------------+-------------+       +-------------+
|             |             |       | K=0～255    |
| K=0～255    | K=256～511   |       | 256×128     |
| 128×256     | 128×256      |   ×   +-------------+  =  +-------------+
|             |             |       | K=256～511  |     | 部分和 2    |
+-------------+-------------+       | 256×128     |     | 128×128     |
                    ^               +-------------+     +-------------+
                    本轮                   ^
                                          本轮

第二轮：K=256～511
(128 × 256) × (256 × 128)
计算第二个部分和，与第一轮的结果相加。
```

官方代码中的 SPLIT\_K 核心部分如下：

```Python
accumulator = tl.zeros(
    (BLOCK_SIZE_M, BLOCK_SIZE_N),
    dtype=a_ptr.type.element_ty
)

accumulator = smt.view(
    accumulator,
    (0, 0),
    (BLOCK_SIZE_M, BLOCK_SIZE_N),
    (MICRO_M, MICRO_N)
)

sub_num = (K + SUB_BLK_K - 1) // SUB_BLK_K
```

第一部分先创建一个零矩阵，然后使用 `smt.view()` 转换成 packed 布局。

原因是`smt.dot()` 计算出的部分结果也是 packed 布局。提前把累加器变为相同布局，就可以直接累加，而不必每轮反复 Unpack。

接下来是实现 K\-loop：

```Python
for k in tl.range(0, sub_num):

    a_descriptor_load = smt.descriptor_load(
        a_block_ptr, (0, 0)
    )

    a = smt.view(
        a_descriptor_load,
        (0, k * SUB_BLK_K),
        (BLOCK_SIZE_M, SUB_BLK_K),
        (MICRO_M, MICRO_K)
    )

    b_descriptor_load = smt.descriptor_load(
        b_block_ptr, (0, 0)
    )

    b = smt.view(
        b_descriptor_load,
        (k * SUB_BLK_K, 0),
        (SUB_BLK_K, BLOCK_SIZE_N),
        (MICRO_K, MICRO_N)
    )

    accumulator += smt.dot(a, b)
```

上述代码可以分为 3 steps。

**Step A**：确定本轮读取的矩阵区域

例如第二轮 `k=1` 时：

```Python
a_offset = (0, 256)
b_offset = (256, 0)
```

理解：A 沿列方向偏移 256，B 沿行方向偏移 256。

**Step B**：将矩阵数据组织成 MICRO\-tile

使用 `smt.view()` 将 A、B 的子块组织为矩阵引擎需要的 packed 布局。

**Step C**：执行乘法并累加

```Python
accumulator += smt.dot(a, b)
```

和 `tl.dot()` 非常相似，只是现在是显式管理数据布局。

循环结束后，执行：

```Python
accumulator = smt.view(
    accumulator,
    (0, 0),
    (BLOCK_SIZE_M, BLOCK_SIZE_N),
    (1, 1)
)

c = accumulator.to(c_ptr.dtype.element_ty)
```

这里进行 unpack，并将结果转换为输出矩阵类型，之后使用 `tl.store()` 写回。

在官方程序中，A 是 FP16，所以累加器也是 FP16。和之前使用的 FP32 累加器不同。

原因是，这是官方演示代码的简化设计，测试也相应采用了较宽松的标准：

```Python
atol=1e-1
rtol=0
```

实际开发中，对于较长 K 维的 GEMM，应优先关注 FP32 跨 K 块累加的数值正确性。

除了默认的 `SPLIT_K`，官方还实现了 `SPLIT_M` 和 `SPLIT_MN`两种策略。

### SPLIT\_M 策略

总结：沿行方向拆分

假设一个 Program 原本负责 C 的 128×128 Tile，现在将其沿 M 方向拆成多个 8×128 子块。

对于每一个子块，都需要与 B 中相同的列进行乘法。因此可以将 B 加载和打包一次，随后让多个 A 子块共享这份 B 数据。

目的：复用 B，减少重复读取和重新打包数据的开销。

代码实现：

```Python
b = smt.view(
    b_descriptor_load,
    (0, 0),
    (BLOCK_SIZE_K, BLOCK_SIZE_N),
    (MICRO_K, MICRO_N)
)

for s in smt.parallel(0, sub_num):
    # 获取第 s 个 A 子块
    # smt.dot(a, b)
    # unpack 并写回结果
```

这里 `smt.parallel()` 表示子块级循环，在教程对应后端配置中默认按普通顺序循环执行，不能直接理解为每个子块都并行运行。

### SPLIT\_MN 策略

总结：沿两个维度拆分，并使用 TCM。

**主要思想**：用片上存储空间换取更少的外部内存访问。

**TCM**

Tightly Coupled Memory，紧耦合存储器。

可以简单理解为 CPU 或计算单元附近的一块低延迟片上存储空间，可用于临时数据缓冲。

工作示意图：

```plaintext
SPLIT_MN：B 数据通过 TCM 复用

                    +------------------+
                    | DDR 内存中的 B   |
                    +------------------+
                             |
                             | 首次读取 B 子块
                             v
                  +----------------------+
                  | TCM：暂存 packed B   |
                  +----------------------+
                             |
             +---------------+---------------+
             |               |               |
             v               v               v
         A 子块 0        A 子块 1        A 子块 2
          复用 B          复用 B          复用 B
         smt.dot()       smt.dot()       smt.dot()

逻辑示意：多个不同的 A 子块可以使用先前存入 TCM 的 B 数据。
```

官方代码：

```Python
b_alloc_ptr = smt.alloc(
    shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
    type=b_ptr.dtype.element_ty
)
```

理解：为 B 的数据申请片上临时缓冲区

`type` ：当前 B 是 FP16，如果没有显式指定元素类型，可能因为默认类型不同而产生存储类型不匹配错误。

对于 B 矩阵中的某个列子块，当它第一次与 A 矩阵的第 0 个行子块进行乘法时，将 B 从 DDR 加载到 TCM。之后其他 A 行子块与同一个 B 列子块相乘时，就可以直接复用 TCM 中的数据。

### 运行性能测试

```Bash
cd spine-triton
RUN_PERF=1 python3 python/examples/test_smt_mm.py
```

输出：

```Bash
PASS
[PERF] M=512, N=512, K=512, iters=100, avg=0.443 ms, GOPS=606.401
```

性能测试结果：

- 矩阵计算规模：512 × 512 × 512；

- 重复测试次数：100

- 平均执行时间：0\.443 ms

- 计算吞吐率：606\.401 GOPS，约 0\.606 TOPS

修改 SUB\_BLK\_K，观察性能变化。通过该实验，可以验证 K 分块粒度对实际运行时间的影响。

创建实验版本的代码：

```Bash
cd ~/spine-triton

mkdir -p my_gemm

python3 - <<'PY'
from pathlib import Path

src = Path("python/examples/test_smt_mm.py")
dst = Path("my_gemm/test_smt_mm_bk128.py")

code = src.read_text()

old = "SUB_BLK_K=256,"
new = "SUB_BLK_K=128,"

assert old in code, "未找到目标参数，请检查源码"

dst.write_text(code.replace(old, new))

print("Created:", dst)
PY
```

运行测试：

```Bash
RUN_PERF=1 python3 my_gemm/test_smt_mm_bk128.py
```

输出：

```Bash
PASS
[PERF] M=512, N=512, K=512, iters=100, avg=0.418 ms, GOPS=642.159
```

两次性能结果对比：

结论：将 `SUB_BLK_K` 从 256 减小到 128 后，GEMM 性能有所提升，平均耗时降低 5\.64%，计算吞吐率提升 5\.9%。

