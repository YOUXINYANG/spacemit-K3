# Cat / Concat

## 简介

举例说明，假设有两个二维张量：

```Python
A = [[1, 2],
     [3, 4]]     形状：(2, 2)，2 行 2 列

B = [[5, 6]]     形状：(1, 2)，1 行 2 列
```

执行：

```Plain Text
C = torch.cat([A, B], dim=0)
```

结果是：

```Plain Text
C = [[1, 2],
     [3, 4],
     [5, 6]]     形状：(3, 2)
```

### 形状变化

```Python
A.shape = (M_A, N)
B.shape = (M_B, N)

C.shape = (M_A + M_B, N)
```

## 无分支 mask\-select 版本代码理解

### 任务划分

教程的划分方法：一个Program 负责一行中的一个列块。

假设：

```Python
A.shape = (2, 6)
B.shape = (1, 6)
C.shape = (3, 6)
```

假设每个列块包含 4 个元素，即 `BLOCK_N = 4`：

```Python
第 0 个列块       第 1 个列块
                    列 0、1、2、3     列 4、5

C 第 0 行（来自 A）  [ program(0,0) ] [ program(0,1) ]
C 第 1 行（来自 A）  [ program(1,0) ] [ program(1,1) ]
C 第 2 行（来自 B）  [ program(2,0) ] [ program(2,1) ]
```

`grid` 是 program 的组织尺寸。

### 读取 A 矩阵

计算索引和掩码：

```Python
row = tl.program_id(0)
col = tl.program_id(1)
offs_n = col * BLOCK_N + tl.arange(0, BLOCK_N)
n_mask = offs_n < N
```

读取 A：

```Python
va = tl.load(
    A + row * N + offs_n,
    mask=n_mask & (row < M_A),
    other=0.,
)
```

### 读取 B 矩阵

读取 B 矩阵时，行号需要减去 `M_A`：

```Python
vb = tl.load(
    B + (row - M_A) * N + offs_n,
    mask=n_mask & (row >= M_A),
    other=0.,
)
```

### 计算并写回 C 矩阵

```Python
tl.store(
    C + row * N + offs_n,
    va + vb,
    mask=n_mask,
)
```

## 运行官方完整程序

```Python
import torch
import triton
import triton.language as tl
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())


@triton.jit(do_not_specialize=["M_A", "M_B", "N"])
def cat0_kernel(A, B, C, M_A, M_B, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1)

    offs_n = col * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N

    va = tl.load(
        A + row * N + offs_n,
        mask=n_mask & (row < M_A),
        other=0.,
    )
    vb = tl.load(
        B + (row - M_A) * N + offs_n,
        mask=n_mask & (row >= M_A),
        other=0.,
    )

    tl.store(
        C + row * N + offs_n,
        va + vb,
        mask=n_mask,
    )


def cat0(a, b):
    M_A, N = a.shape
    M_B, _ = b.shape

    c = torch.empty(
        (M_A + M_B, N),
        dtype=a.dtype,
        device=a.device,
    )

    BLOCK_N = min(256, triton.next_power_of_2(N))

    cat0_kernel[(M_A + M_B, triton.cdiv(N, BLOCK_N))](
        a, b, c, M_A, M_B, N, BLOCK_N=BLOCK_N,
    )
    return c


if __name__ == "__main__":
    torch.manual_seed(0)

    for M_A, M_B, N in [
        (2, 1, 2),
        (100, 37, 256),
        (64, 64, 1000),
    ]:
        a = torch.randn(M_A, N, dtype=torch.float16)
        b = torch.randn(M_B, N, dtype=torch.float16)

        got = cat0(a, b)
        ref = torch.cat([a, b], dim=0)

        assert torch.equal(got, ref), (
            f"mismatch at {(M_A, M_B, N)}"
        )
        print(f"PASS cat0 {M_A}x{N} + {M_B}x{N}  (exact)")
```

运行结果：

```Python
PASS cat0 2x2 + 1x2  (exact)
PASS cat0 100x256 + 37x256  (exact)
PASS cat0 64x1000 + 64x1000  (exact)
```

## 沿 dim= 1 拼接

### 从增加行数变成增加列数

```Python
A = [[1, 2],
     [3, 4]]       形状：(2, 2)

B = [[5],
     [6]]          形状：(2, 1)
```

执行：

```Python
C = torch.cat([A, B], dim=1)
```

得到：

```Python
C = [[1, 2, 5],
     [3, 4, 6]]    形状：(2, 3)
```

形状变化：

```Python
A.shape = (M, N_A)
B.shape = (M, N_B)

C.shape = (M, N_A + N_B)
```

### 由列号决定输入来源

```Python
如果 c < N_A：
    C[r, c] = A[r, c]

如果 c >= N_A：
    C[r, c] = B[r, c - N_A]
```

### 任务划分

仍然按一行中的一个列块划分。沿用之前的输出切块方式：

```Python
N = N_A + N_B
grid = (M, triton.cdiv(N, BLOCK_N))
```

### 两路 mask

延续教程的两路读取思路：

```Python
a_mask = n_mask & (offs_n < N_A)
b_mask = n_mask & (offs_n >= N_A)
```

## dim=1 拼接完整程序

```Python
import torch
import triton
import triton.language as tl
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())


@triton.jit(do_not_specialize=["M", "N_A", "N_B"])
def cat1_kernel(A, B, C, M, N_A, N_B, BLOCK_N: tl.constexpr):
    # 当前 program 负责的输出行和列块
    row = tl.program_id(0)
    col_block = tl.program_id(1)

    # 输出列下标
    N = N_A + N_B
    offs_n = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N

    # 输出前 N_A 列来自 A
    va = tl.load(
        A + row * N_A + offs_n,
        mask=n_mask & (offs_n < N_A),
        other=0.,
    )

    # 输出后 N_B 列来自 B
    vb = tl.load(
        B + row * N_B + (offs_n - N_A),
        mask=n_mask & (offs_n >= N_A),
        other=0.,
    )

    # 延续教程用“有效值 + 另一路的 0”合并结果
    tl.store(
        C + row * N + offs_n,
        va + vb,
        mask=n_mask,
    )


def cat1(a, b):
    assert a.ndim == 2 and b.ndim == 2, "输入必须是二维张量"
    assert a.shape[0] == b.shape[0], "输入行数必须相同"
    assert a.dtype == b.dtype, "输入 dtype 必须相同"
    assert a.device == b.device, "输入设备必须相同"
    assert a.is_contiguous() and b.is_contiguous(), "输入必须连续"
    assert a.shape[0] > 0, "本示例要求行数大于 0"
    assert a.shape[1] > 0 and b.shape[1] > 0, "本示例要求列数大于 0"

    M, N_A = a.shape
    _, N_B = b.shape
    N = N_A + N_B

    c = torch.empty((M, N), dtype=a.dtype, device=a.device)

    BLOCK_N = min(256, triton.next_power_of_2(N))
    grid = (M, triton.cdiv(N, BLOCK_N))

    cat1_kernel[grid](
        a, b, c, M, N_A, N_B,
        BLOCK_N=BLOCK_N,
    )
    return c


if __name__ == "__main__":
    # 固定数据：直接观察拼接顺序
    a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float16)
    b = torch.tensor([[5], [6]], dtype=torch.float16)

    got = cat1(a, b)
    ref = torch.cat([a, b], dim=1)

    print("固定数据结果：")
    print(got)
    assert torch.equal(got, ref)
    print("PASS 固定数据")

    # 随机数据：覆盖不同分块情况
    torch.manual_seed(0)

    for M, N_A, N_B in [
        (3, 5, 2),
        (8, 128, 128),
        (4, 300, 211),
    ]:
        a = torch.randn(M, N_A, dtype=torch.float16)
        b = torch.randn(M, N_B, dtype=torch.float16)

        got = cat1(a, b)
        ref = torch.cat([a, b], dim=1)

        assert torch.equal(got, ref), (
            f"mismatch at {(M, N_A, N_B)}"
        )
        print(f"PASS cat1 {M}x{N_A} + {M}x{N_B}")
```

运行结果：

```Python
PASS cat1 3x5 + 3x2
PASS cat1 8x128 + 8x128
PASS cat1 4x300 + 4x211
```

## 三输入拼接

```Python
import torch
import triton
import triton.language as tl
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())


@triton.jit(
    do_not_specialize=["M_A", "M_B", "M_D", "N"]
)
def cat3_kernel(
    A, B, D, C,
    M_A, M_B, M_D, N,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    col_block = tl.program_id(1)

    offs_n = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N

    # 输出行分别属于哪个输入
    from_a = row < M_A
    from_b = (row >= M_A) & (row < M_A + M_B)
    from_d = row >= M_A + M_B

    va = tl.load(
        A + row * N + offs_n,
        mask=n_mask & from_a,
        other=0.,
    )
    vb = tl.load(
        B + (row - M_A) * N + offs_n,
        mask=n_mask & from_b,
        other=0.,
    )
    vd = tl.load(
        D + (row - M_A - M_B) * N + offs_n,
        mask=n_mask & from_d,
        other=0.,
    )

    tl.store(
        C + row * N + offs_n,
        va + vb + vd,
        mask=n_mask,
    )


def cat3(a, b, d):
    inputs = (a, b, d)

    assert all(x.ndim == 2 for x in inputs), "输入必须是二维张量"
    assert all(x.is_contiguous() for x in inputs), "输入必须连续"
    assert all(x.dtype == a.dtype for x in inputs), "dtype 必须相同"
    assert all(x.device == a.device for x in inputs), "设备必须相同"
    assert all(x.shape[1] == a.shape[1] for x in inputs), "列数必须相同"
    assert all(x.shape[0] > 0 for x in inputs), "本示例要求输入行数大于 0"
    assert a.shape[1] > 0, "本示例要求列数大于 0"

    M_A, N = a.shape
    M_B = b.shape[0]
    M_D = d.shape[0]
    M = M_A + M_B + M_D

    c = torch.empty((M, N), dtype=a.dtype, device=a.device)

    BLOCK_N = min(256, triton.next_power_of_2(N))
    grid = (M, triton.cdiv(N, BLOCK_N))

    cat3_kernel[grid](
        a, b, d, c,
        M_A, M_B, M_D, N,
        BLOCK_N=BLOCK_N,
    )
    return c


if __name__ == "__main__":
    # 固定数据：观察三个输入的排列顺序
    a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float16)
    b = torch.tensor([[5, 6]], dtype=torch.float16)
    d = torch.tensor([[7, 8], [9, 10]], dtype=torch.float16)

    got = cat3(a, b, d)
    ref = torch.cat([a, b, d], dim=0)

    print("固定数据结果：")
    print(got)
    assert torch.equal(got, ref)
    print("PASS 固定数据")

    # 分别检查三个输出区间
    M_A = a.shape[0]
    M_B = b.shape[0]

    assert torch.equal(got[:M_A], a)
    assert torch.equal(got[M_A:M_A + M_B], b)
    assert torch.equal(got[M_A + M_B:], d)
    print("PASS 三段输出区间")

    # 随机数据：验证不同形状和列块边界
    torch.manual_seed(0)

    for M_A, M_B, M_D, N in [
        (2, 1, 2, 2),
        (7, 3, 5, 256),
        (4, 2, 3, 1000),
    ]:
        a = torch.randn(M_A, N, dtype=torch.float16)
        b = torch.randn(M_B, N, dtype=torch.float16)
        d = torch.randn(M_D, N, dtype=torch.float16)

        got = cat3(a, b, d)
        ref = torch.cat([a, b, d], dim=0)

        assert torch.equal(got, ref), (
            f"mismatch at {(M_A, M_B, M_D, N)}"
        )
        print(
            f"PASS cat3 "
            f"{M_A}x{N} + {M_B}x{N} + {M_D}x{N}"
        )
```

运行结果：

```Python
固定数据结果：
tensor([[ 1.,  2.],
        [ 3.,  4.],
        [ 5.,  6.],
        [ 7.,  8.],
        [ 9., 10.]], dtype=torch.float16)
PASS 固定数据
PASS 三段输出区间
PASS cat3 2x2 + 1x2 + 2x2
PASS cat3 7x256 + 3x256 + 5x256
PASS cat3 4x1000 + 2x1000 + 3x1000
```

## 多输入拼接的实现方案取舍

### 方案1：单次启动，在kernal内选择输入

```Python
启动一次 kernel
    ↓
每个 program 根据输出行号判断数据来源
    ↓
使用各路 mask 读取，再写入输出
```

两个输入时有两路读取，三个输入时有三路读取。

它的优点是只需一次 kernel 启动。但输入越多，地址计算、mask 判断和结果合并也越复杂。被屏蔽的读取不会访问输入数据，但表达这些候选路径仍可能产生指令开销。因此，一次启动不等于执行成本一定最低。

### 方案2：逐个输入，分别启动复制kernal

在 Python 侧安排好每个输入的输出区域：

```Plain Text
A → C 的第一段
B → C 的第二段
D → C 的第三段
```

再分别执行复制：

```Plain Text
启动 copy kernel：A → C 的第一段
启动 copy kernel：B → C 的第二段
启动 copy kernel：D → C 的第三段
```

这样，每次启动的 kernel 已经知道“从哪个输入复制到哪个输出区域”，无需再选择输入。

这也是教程介绍的 FlagGems 生产实现所采用的组织思路。具体实现还会检查连续性，并处理不适合直接复制的情况。

### 时间模型

```Python
L：一次 kernel 启动的开销
T：实际搬运数据所需的时间
S：单 kernel 内选择多个输入的额外开销
K：输入数量
```

两种方案的时间模型分别为：

```Python
单 kernel 选路：
    总时间 ≈ L + T + S

每个输入启动一次 copy：
    总时间 ≈ K × L + T
```

### 为什么不写 `if/elif`

从逻辑上可以这样表达：

```Python
如果输出行属于 A：
    选择 A 的地址
否则如果属于 B：
    选择 B 的地址
否则：
    选择 D 的地址

从选好的地址读取
```

但是教程记录过：历史版本的编译器对这种通过控制流选择指针的写法支持不完整，可能报错。

## Cat 和相关算子区别

### Cat 和 stack 

`cat` 沿已有维度拼接，`stack` 会新增一个维度。

例如，两个输入都是 `(2, 3)`：

```Plain Text
a = torch.ones(2, 3)
b = torch.zeros(2, 3)
```

结果形状分别是：

约束差别：`stack` 要求所有输入形状相同；`cat` 允许拼接维度的长度不同。

