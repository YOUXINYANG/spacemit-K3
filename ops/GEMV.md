# GEMV

## 简介

一个矩阵乘一个向量。

```math
y = W x,\qquad W\in\mathbb{R}^{N\times K},\quad x\in\mathbb{R}^{K},\quad y\in\mathbb{R}^{N}
```

- `N`：矩阵的行数，也是输出元素个数。

- `K`：矩阵的列数，也是每个点积的长度。

- `y[i]`：矩阵第 `i` 行与向量 `x` 的点积。

具体计算公式：

```math
y[i]=\sum_{k=0}^{K-1}W[i,k]x[k]
```

特点：各行的计算互不依赖，但都使用同一个 x。

## 标准 tl 实现代码

```Python
import torch
import triton
import triton.language as tl
from triton.backends.spine_triton.driver import CPUDriver

# 启用 Spine-Triton CPU 后端。
triton.runtime.driver.set_active(CPUDriver())


@triton.jit(do_not_specialize=["N", "K"])
def mv_kernel(
    W, x, y, N, K,
    stride_wn, stride_wk, stride_xk, stride_yn,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)

    # 当前 program 的行号，以及列块内的偏移。
    offset_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)[:, None]
    offset_k = tl.arange(0, BLOCK_K)[None, :]
    n_mask = offset_n < N

    W_ptrs = W + offset_n * stride_wn + offset_k * stride_wk
    x_ptrs = x + offset_k * stride_xk

    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_mask = k + offset_k < K

        w = tl.load(
            W_ptrs, mask=n_mask & k_mask, other=0.0
        ).to(tl.float32)
        v = tl.load(
            x_ptrs, mask=k_mask, other=0.0
        ).to(tl.float32)

        acc += w * v

        W_ptrs += BLOCK_K * stride_wk
        x_ptrs += BLOCK_K * stride_xk

    row_sum = tl.sum(acc, axis=1)
    tl.store(
        y + offset_n * stride_yn,
        row_sum[:, None],
        mask=n_mask,
    )


def mv(W, x):
    N, K = W.shape
    y = torch.empty((N,), device=W.device, dtype=W.dtype)

    BLOCK_N = 32
    BLOCK_K = min(128, triton.next_power_of_2(K))
    grid = (triton.cdiv(N, BLOCK_N),)

    mv_kernel[grid](
        W, x, y, N, K,
        W.stride(0), W.stride(1),
        x.stride(0), y.stride(0),
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return y


if __name__ == "__main__":
    torch.manual_seed(0)

    N, K = 1024, 512
    W = torch.randn(N, K, dtype=torch.float16)
    x = torch.randn(K, dtype=torch.float16)

    y = mv(W, x)
    y_ref = torch.mv(W.float(), x.float()).half()

    torch.testing.assert_close(y, y_ref, atol=1e-2, rtol=1e-2)

    print("PASS")
    print("output shape:", tuple(y.shape))
    print("max diff:", (y.float() - y_ref.float()).abs().max().item())
```

运行结果：

```Python
PASS
output shape: (1024,)
max diff: 3.0517578125e-05
```

将 N 和 K 改变，验证不能被分块大小整除时，代码是否仍然正确。

运行结果：

```Python
PASS
output shape: (100,)
max diff: 0.0
```

## 代码解读

### 按照行划分任务

一个 triton Program 负责若干行。官方代码使用：

```Python
BLOCK_N = 32
grid = (triton.cdiv(N, BLOCK_N),)
```

`BLOCK_N` 表示一个 program 处理多少行；`triton.cdiv` 表示向上取整的除法。

各个 program 都要使用向量 `x`。这是教程标准实现的任务划分方式。

### 分块处理

每个任务内部，沿 K 分块。一个 program 要算 32 行，每行有 512 个元素。教程并非一次处理完整的 `(32, 512)`，而是设置：

```Plain Text
BLOCK_K = 128
```

每个 program 分四轮读取：

```Plain Text
第 1 轮：这 32 行的第   0～127 列，以及 x[0:128]
第 2 轮：这 32 行的第 128～255 列，以及 x[128:256]
第 3 轮：这 32 行的第 256～383 列，以及 x[256:384]
第 4 轮：这 32 行的第 384～511 列，以及 x[384:512]
```

基本计算结构：

```Python
整个 W[N, K]
    ↓ 按 N 分配 program
每个 program 负责 BLOCK_N 行
    ↓ 沿 K 循环
每轮处理一个 BLOCK_N × BLOCK_K 数据块
    ↓ 完成点积
写出 BLOCK_N 个结果
```

### 地址计算

一个 Program 如何在内存中找到自己负责的行？

```Python
pid = tl.program_id(0)

offset_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)[:, None]
offset_k = tl.arange(0, BLOCK_K)[None, :]
```

### Stride

Stride 表示：沿某一个维度走一步，要跨过多少个元素。和之前 gemm 中相似。

教程中 kernal 接收这些参数：

```Python
stride_wn, stride_wk, stride_xk, stride_yn
```

这些参数由 pytorch 张量提供：

```Python
W.stride(0), W.stride(1), x.stride(0), y.stride(0)
```

`W[i, k]`地址可以表达为：

```Python
W + i * stride_wn + k * stride_wk
```

### 整块数据地址

```Python
W_ptrs = W + offset_n * stride_wn + offset_k * stride_wk
x_ptrs = x + offset_k * stride_xk
```

之前地址计算得到的：

```Python
offset_n.shape = (32, 1)
offset_k.shape = (1, 128)
```

通过广播，`W_ptrs`的形状成为 `(32, 128)`。其中每个位置都保存对应元素的地址：

```Python
列偏移
               0     1    ...   127
行号 64      地址   地址   ...   地址
行号 65      地址   地址   ...   地址
  ...
行号 95      地址   地址   ...   地址
```

这里生成的是数据地址，真正读取数据要靠 `tl.load`。

每轮结束，指针推进到下一块：

```Python
W_ptrs += BLOCK_K * stride_wk
x_ptrs += BLOCK_K * stride_xk
```

### Mask

Mask 决定哪些地址可以读取。和之前教程中类似。

教程分别检查行和列：

```Python
n_mask = offset_n < N

for k in range(0, K, BLOCK_K):
    k_mask = k + offset_k < K
```

读取矩阵时，行列必须同时有效：

```Python
w = tl.load(
    W_ptrs,
    mask=n_mask & k_mask,
    other=0.0,
).to(tl.float32)
```

两个 mask 广播后，得到 `(32, 128)` 的有效性判断。`mask=False` 的位置不读取对应内存，直接使用 `other=0.0`。

读取 `x` 时只需要检查列：

```Plain Text
v = tl.load(
    x_ptrs,
    mask=k_mask,
    other=0.0,
).to(tl.float32)
```

### 累加器

初始化累加器：

```Python
acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
```

当 `BLOCK_N = 32`、`BLOCK_K = 128` 时，`acc` 包含 32 行，每行有 128 个累加位置。计算过程中，每个列位置先保存自己负责的部分和：

```Plain Text
acc += w * v
```

循环结束后，再执行：

```Python
row_sum = tl.sum(acc, axis=1)
```

得到完整点积结果。

### 写回结果

```Python
tl.store(
    y + offset_n * stride_yn,
    row_sum[:, None],
    mask=n_mask,
)
```

## smt\_rvisa 版本代码

官方代码在 spine\-triton 项目中的地址：python/tests/raw/test\_raw\_mv\_svector\.py

```Python
"""Raw 向量版 GEMV：C = B @ A。

B: [N, K] FP16，行主序；A: [K] FP16；C: [N] FP32。
style2：直接加载向量；style3：先打包 B，再加载向量。
每组处理 4 行，K 尾部通过收窄 VL 处理，C 扩容以容纳尾部行写入。
普通内存使用扁平元素偏移，packed_B 使用多维下标。
"""

import torch
import triton
import triton.language as tl
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import pytest
import triton.language.extra.smt_rvisa as tle
from triton.language.extra.smt_rvisa import call as _sr_call

f16 = tle.f16
f32 = tle.f32


# style2：直接加载向量
@tle.raw_kernel
def mv_block_style2(B: tle.mem(f16), A: tle.mem(f16), C: tle.mem(f32, out=True), K: tle.index, row_base: tle.index,
                    row_end: tle.index):
    # 当前 program 处理 [row_base, row_end)，每组 4 行。
    nvl = tle.vconfig(-1, 1)  # FP16、LMUL=1 时 VLMAX=64
    Kfloor = (K // nvl) * nvl  # 满向量块的结束位置
    for ni in tle.range(row_base, row_end, 4):
        acc0 = tle.vzero(f32)
        acc1 = tle.vzero(f32)
        acc2 = tle.vzero(f32)
        acc3 = tle.vzero(f32)
        for ki in tle.range(0, Kfloor, nvl):  # 主循环：完整向量块
            va = tle.vload(A, ki)
            vb0 = tle.vload(B, ni * K + ki)
            vb1 = tle.vload(B, (ni + 1) * K + ki)
            vb2 = tle.vload(B, (ni + 2) * K + ki)
            vb3 = tle.vload(B, (ni + 3) * K + ki)
            acc0 = tle.vmacc(acc0, vb0, va)
            acc1 = tle.vmacc(acc1, vb1, va)
            acc2 = tle.vmacc(acc2, vb2, va)
            acc3 = tle.vmacc(acc3, vb3, va)
        for ki in tle.range(Kfloor, K, nvl):  # 尾循环：0 或 1 次
            nvl = tle.vconfig(K - ki, 1)
            # 使用独立临时名，避免编译器误判循环携带值。
            ta = tle.vload(A, ki)
            tb0 = tle.vload(B, ni * K + ki)
            tb1 = tle.vload(B, (ni + 1) * K + ki)
            tb2 = tle.vload(B, (ni + 2) * K + ki)
            tb3 = tle.vload(B, (ni + 3) * K + ki)
            acc0 = tle.vmacc(acc0, tb0, ta)
            acc1 = tle.vmacc(acc1, tb1, ta)
            acc2 = tle.vmacc(acc2, tb2, ta)
            acc3 = tle.vmacc(acc3, tb3, ta)
        tle.sstore(C, ni, tle.vreduce_sum(acc0))
        tle.sstore(C, ni + 1, tle.vreduce_sum(acc1))
        tle.sstore(C, ni + 2, tle.vreduce_sum(acc2))
        tle.sstore(C, ni + 3, tle.vreduce_sum(acc3))


@triton.jit(do_not_specialize=["K", "N"])
def _mv_sv_host_style2(B, A, C, K, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row_base = pid * BLOCK
    row_end = min(row_base + BLOCK, N)
    _sr_call(mv_block_style2, outputs=[], inputs=[B, A, C, K, row_base, row_end])


# style3：预打包 B 后加载向量
@tle.raw_kernel
def mv_block_style3(B: tle.mem(f16), A: tle.mem(f16), C: tle.mem(f32, out=True), K: tle.index, row_base: tle.index,
                    row_end: tle.index):
    nvl = tle.vconfig(-1, 1)  # FP16、LMUL=1 时 VLMAX=64
    # 向上取整，确保 scratch 能容纳尾块。
    kct = (K + nvl - 1) // nvl
    packed_B = tle.alloc((1, kct, 4, nvl), f16)
    Kfloor = (K // nvl) * nvl
    for ni in tle.range(row_base, row_end, 4):
        acc0 = tle.vzero(f32)
        acc1 = tle.vzero(f32)
        acc2 = tle.vzero(f32)
        acc3 = tle.vzero(f32)
        tle.pack(B, (ni, 0), packed_B, (1, kct, 4, nvl), K)
        for ki in tle.range(0, Kfloor, nvl):  # 主循环：完整向量块
            vb0 = tle.vload(packed_B, (0, ki // nvl, 0, 0))
            vb1 = tle.vload(packed_B, (0, ki // nvl, 1, 0))
            vb2 = tle.vload(packed_B, (0, ki // nvl, 2, 0))
            vb3 = tle.vload(packed_B, (0, ki // nvl, 3, 0))
            va = tle.vload(A, ki)
            acc0 = tle.vmacc(acc0, vb0, va)
            acc1 = tle.vmacc(acc1, vb1, va)
            acc2 = tle.vmacc(acc2, vb2, va)
            acc3 = tle.vmacc(acc3, vb3, va)
        for ki in tle.range(Kfloor, K, nvl):  # 尾循环：收窄 VL
            nvl = tle.vconfig(K - ki, 1)
            tb0 = tle.vload(packed_B, (0, ki // nvl, 0, 0))
            tb1 = tle.vload(packed_B, (0, ki // nvl, 1, 0))
            tb2 = tle.vload(packed_B, (0, ki // nvl, 2, 0))
            tb3 = tle.vload(packed_B, (0, ki // nvl, 3, 0))
            ta = tle.vload(A, ki)
            acc0 = tle.vmacc(acc0, tb0, ta)
            acc1 = tle.vmacc(acc1, tb1, ta)
            acc2 = tle.vmacc(acc2, tb2, ta)
            acc3 = tle.vmacc(acc3, tb3, ta)
        tle.sstore(C, ni, tle.vreduce_sum(acc0))
        tle.sstore(C, ni + 1, tle.vreduce_sum(acc1))
        tle.sstore(C, ni + 2, tle.vreduce_sum(acc2))
        tle.sstore(C, ni + 3, tle.vreduce_sum(acc3))


@triton.jit(do_not_specialize=["K", "N"])
def _mv_sv_host_style3(B, A, C, K, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row_base = pid * BLOCK
    row_end = min(row_base + BLOCK, N)
    _sr_call(mv_block_style3, outputs=[], inputs=[B, A, C, K, row_base, row_end])


def _run(host, N, K, BLOCK=4):
    # C 按 BLOCK 向上扩容，容纳末组 4 行的额外写入；结果取前 N 项。
    Np = ((N + BLOCK - 1) // BLOCK) * BLOCK
    B = torch.randn(N, K, dtype=torch.float16)
    A = torch.randn(K, dtype=torch.float16)
    C = torch.empty(Np, dtype=torch.float32)
    grid = (Np // BLOCK, )
    host[grid](B.contiguous().reshape(-1), A.contiguous(), C, K, N, BLOCK=BLOCK)
    got = C[:N]
    ref = torch.mv(B.float(), A.float())
    max_diff = (got - ref).abs().max().item()
    assert torch.allclose(got, ref, rtol=1e-2, atol=1e-2), \
        f"N={N} K={K} max_diff={max_diff:.4e}"


# 完整分块形状
_SHAPES = [(4, 64), (8, 128), (16, 256), (32, 512), (64, 64), (128, 256)]


@pytest.mark.parametrize("N, K", _SHAPES)
def test_raw_mv_svector_style2(N, K):
    _run(_mv_sv_host_style2, N, K)


@pytest.mark.parametrize("N, K", _SHAPES)
def test_raw_mv_svector_style3(N, K):
    _run(_mv_sv_host_style3, N, K)


# 尾部形状：覆盖 K 非 64 倍数、N 非 4 倍数及二者同时出现的情况。
_SHAPES_ARB = [(4, 60), (4, 100), (8, 65), (4, 63), (8, 127), (4, 200), (16, 130), (12, 50), (7, 64), (33, 65),
               (50, 130), (100, 100), (6, 60), (13, 200), (37, 130)]


@pytest.mark.parametrize("N, K", _SHAPES_ARB)
def test_raw_mv_svector_style2_arb(N, K):
    _run(_mv_sv_host_style2, N, K)


@pytest.mark.parametrize("N, K", _SHAPES_ARB)
def test_raw_mv_svector_style3_arb(N, K):
    _run(_mv_sv_host_style3, N, K)
```

运行官方代码：

```Python
python3 -m pytest python/tests/raw/test_raw_mv_svector.py -q
```

输出：

```Python
..........................................                      [100%]
42 passed in 21.27s
```

## 代码解读

### 显式操作向量

5\.2 中，我们用：

```Plain Text
acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
```

描述一个二维累加器，由编译器处理后续映射。

5\.4 则通过 `tle` 的 raw 接口，更直接地描述向量加载、乘加和归约。注意教程中的参数命名发生了变化：

### 四行展开

核心结构：

```Python
for ni in tle.range(row_base, row_end, 4):
    acc0 = tle.vzero(f32)
    acc1 = tle.vzero(f32)
    acc2 = tle.vzero(f32)
    acc3 = tle.vzero(f32)
```

每轮处理矩阵的四行：

```Plain Text
第 ni     行 → acc0
第 ni + 1 行 → acc1
第 ni + 2 行 → acc2
第 ni + 3 行 → acc3
```

四个 `acc` 都是向量累加器，还不是最终的四个标量结果。

然后每次沿 K 方向取一段：

```Plain Text
va = tle.vload(A, ki)

vb0 = tle.vload(B, ni * K + ki)
vb1 = tle.vload(B, (ni + 1) * K + ki)
vb2 = tle.vload(B, (ni + 2) * K + ki)
vb3 = tle.vload(B, (ni + 3) * K + ki)

acc0 = tle.vmacc(acc0, vb0, va)
acc1 = tle.vmacc(acc1, vb1, va)
acc2 = tle.vmacc(acc2, vb2, va)
acc3 = tle.vmacc(acc3, vb3, va)
```

这里按连续、行主序矩阵计算地址，`ni * K + ki` 指向第 `ni` 行、第 `ki` 列。

`va` 只加载一次，随后供四行使用。示意图：

```Python
┌→ 第 ni     行的片段 → acc0
x 的一个片段 va ────┼→ 第 ni + 1 行的片段 → acc1
                     ├→ 第 ni + 2 行的片段 → acc2
                     └→ 第 ni + 3 行的片段 → acc3
```

这样组织的**原因**：如果四行分别处理、每行都单独加载相同的 `x` 片段，就会出现四次向量加载。四行展开把它减少为一次。

等四行的 K 方向全部处理完成，再分别归约：

```Plain Text
tle.sstore(C, ni,     tle.vreduce_sum(acc0))
tle.sstore(C, ni + 1, tle.vreduce_sum(acc1))
tle.sstore(C, ni + 2, tle.vreduce_sum(acc2))
tle.sstore(C, ni + 3, tle.vreduce_sum(acc3))
```

它与 5\.2 仍然遵循同一个思路：循环内保留向量部分和，循环结束后再归约。

## vfwmacc 加宽乘累加版本

`vfwmacc` 是 RISC\-V 向量指令中的浮点加宽乘累加。这里关注的形式可以理解为：

```Plain Text
FP16 × FP16 → 累加到 FP32
```

低精度输入，高精度累加。

### 使用条件

教程列出的条件如下。依赖固定的计算块和布局，不能直接认为能处理前面测试过的所有形状。

```Python
M % 32 == 0
N % 256 == 0
输入类型为 FP16
输入连续存储
```

### 运行官方代码

源码暂时没找到。

## CBM 矩阵引擎版本

### 运行官方代码

官方源码：

```Python
"""Raw CBM GEMV：C[M] = B[M,K] @ A[K]。

B 经 vpack 转为 cube 布局，A 经 spread 和 vbroadcast 广播。
FP32 累加，输出还原为行主序并转换为 FP16，取第 0 列。
M/K 尾部由 valid_rows、vpack 和 spread 补零处理。
"""

import numpy as np
import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import triton.language as tl
import pytest
import triton.language.extra.smt_rvisa as tle
from triton.language.extra.smt_rvisa import call as _sr_call

f16 = tle.f16
f32 = tle.f32

CM, CN, CK = tle.mma_cube(f16)  # FP16 cube：(8, 8, 8)
VL = CN * CK  # 64 lanes
MB = 2 * CM  # 每 program 处理 16 行
Npad = 4 * CN  # 广播维固定为 32
B1, B2 = MB // CM, Npad // CN  # 2, 4


def make_mv(M, K):
    """为给定 M/K 生成 host；按 shape 命名以区分 JIT 缓存。"""
    Kp = ((K + CK - 1) // CK) * CK  # K 向上对齐
    KC = Kp // CK
    Mtot = M
    grid_blocks = (M + MB - 1) // MB

    @tle.raw_kernel
    def mv(B: tle.mem(f16), A: tle.mem(f16), C: tle.mem(f16, out=True), row_base: tle.index):
        tle.vconfig(VL, 1)

        # 打包 B，有效行之外及 K 尾部补零。
        vr = tle.imin(MB, Mtot - row_base)
        Bcube = tle.vpack(B, inner_tiles=(MB, CK), stride=K, rows=MB, offset=row_base * K, valid_rows=vr)

        # 广播 A，K 尾部补零。
        scrA = tle.spread(A, cube_shape=(KC, CN, CK), k_real=K)
        acc = tle.vzero(f32, group=B1 * B2)  # <8×64xf32>
        for kc in tle.range(0, KC, 1):
            vb = tle.vload(Bcube, (0, kc), group=B1)  # <2×64xf16>
            va1 = tle.vload(scrA, (kc, 0))  # <64xf16>
            va = tle.vbroadcast(va1, B2)  # <4×64xf16>
            acc = tle.vmadot(acc, vb, va)

        # 将 cube 输出还原为行主序。
        c1 = tle.vpack(acc, CN)  # <8×64> → <4×128>
        c2 = tle.vpack(c1, 2 * CN)  # <4×128> → <2×256>
        cf = tle.vshape(c2, (MB, Npad))
        c = tle.cast(cf, f16)
        tle.vstore(C, row_base * Npad, c, shape=(MB, Npad))

    mv._fn.__name__ = f"mv_cbm_{M}_{K}"

    @triton.jit
    def host(B, A, C, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        row_base = pid * BLOCK
        _sr_call(mv, outputs=[], inputs=[B, A, C, row_base])

    host.__name__ = f"_mv_cbm_host_{M}_{K}"
    host.fn.__name__ = host.__name__
    host._grid_blocks = grid_blocks
    return host


def _run(M, K):
    rng = np.random.default_rng(0)
    Blog = rng.standard_normal((M, K)).astype(np.float16)
    Alog = rng.standard_normal((K, )).astype(np.float16)
    golden = Blog.astype(np.float64) @ Alog.astype(np.float64)
    B = torch.tensor(Blog.reshape(-1))
    A = torch.tensor(Alog.reshape(-1))

    # 按完整行块分配输出，读取时丢弃补齐行。
    Mp = ((M + MB - 1) // MB) * MB
    C = torch.zeros(Mp, Npad, dtype=torch.float16)
    host = make_mv(M, K)
    host[(host._grid_blocks, )](B.contiguous(), A.contiguous(), C, BLOCK=MB)
    got = C[:M, 0].float().numpy().astype(np.float64)
    diff = np.abs(got - golden).max()
    assert diff < 5e-2, f"M={M} K={K} max_diff={diff:.4e}\ngot={got[:4]}\ngold={golden[:4]}"


# 覆盖完整分块、K 尾部、M 小于一块及多块行尾部。
_SHAPES = [(64, 64), (128, 64), (64, 128), (256, 64),
           (64, 60), (128, 100), (64, 40),
           (12, 60), (12, 64), (4, 40),
           (20, 64), (20, 60), (100, 64), (50, 100)]


@pytest.mark.parametrize("M, K", _SHAPES)
def test_mv_cbm(M, K):
    _run(M, K)
```

测试官方源码：

```Python
python3 -m pytest python/tests/raw/test_raw_mv_cbm.py -q
```

