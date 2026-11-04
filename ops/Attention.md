# Attention

## 简介

### QKV

- Q（Query）：当前位置用来查询的向量。

- K（Key）：各位置用来参与匹配的向量。

- V（Value）：匹配完成后，实际被加权汇总的内容。

### Attention

设序列有 $M$ 个 token，每个向量有 $D$ 个元素，本教程中三个输入的形状都是 $(M,D)$。

Attention 计算分成三步完成：

$S=\frac{QK^T}{\sqrt D},\qquad P=\mathrm{softmax}(S),\qquad O=PV $

计算理解：先算匹配分数，再把分数转换成权重，最后按权重汇总 V。

### 形状变化

```Python
Q (M,D) × Kᵀ (D,M) → S (M,M)
S 每行做 softmax    → P (M,M)
P (M,M) × V (M,D)   → O (M,D)
```

### 手算

教程使用：

$Q=\begin{bmatrix}1&0\\0&1\end{bmatrix},\quad K=\begin{bmatrix}1&0\\1&1\end{bmatrix},\quad V=\begin{bmatrix}10&20\\30&40\end{bmatrix}$

先计算匹配分数：

$S=QK^T= \begin{bmatrix}1&1\\0&1\end{bmatrix}$

接着加 causal mask（因果掩码）。序列位置从 0 开始，第 `i` 个位置只能访问 `j≤i` 的位置，因此：

$S_{\text{masked}}= \begin{bmatrix} 1&-\infty\\ 0&1 \end{bmatrix}$

第一行不能看第二个位置，所以把对应分数设成 $-\infty$。由于 $e^{-\infty}=0$，它经过 softmax 后权重为零。

逐行做 softmax：

$P= \begin{bmatrix} 1&0\\ \frac{1}{1+e}&\frac{e}{1+e} \end{bmatrix} \approx \begin{bmatrix} 1&0\\ 0.26894&0.73106 \end{bmatrix}$

最后用权重汇总 V：

$O[0]=1\times[10,20]+0\times[30,40]=[10,20]$

$O[1]=0.26894\times[10,20]+0.73106\times[30,40] \approx[24.6212,34.6212]$

## Pytorch 验证

```Python
import torch

q = torch.tensor([[1., 0.], [0., 1.]])
k = torch.tensor([[1., 0.], [1., 1.]])
v = torch.tensor([[10., 20.], [30., 40.]])

# 与教程手算一致：scale=1
scores = q @ k.T

# 行是 query 位置，列是 key 位置
positions = torch.arange(2)
visible = positions[:, None] >= positions[None, :]
masked_scores = scores.masked_fill(~visible, float("-inf"))

weights = torch.softmax(masked_scores, dim=-1)
output = weights @ v

print("原始分数:\n", scores)
print("掩码后分数:\n", masked_scores)
print("注意力权重:\n", weights)
print("输出:\n", output)

expected = torch.tensor([[10., 20.], [24.6212, 34.6212]])
torch.testing.assert_close(output, expected, atol=1e-4, rtol=0)
```

打印内容：

```Python
原始分数:
 tensor([[1., 1.],
        [0., 1.]])
掩码后分数:
 tensor([[1., -inf],
        [0., 1.]])
注意力权重:
 tensor([[1.0000, 0.0000],
        [0.2689, 0.7311]])
输出:
 tensor([[10.0000, 20.0000],
        [24.6212, 34.6212]])
```

## Kernal 负责部分

上述简介中的 attention 计算是：

$O=\mathrm{softmax}(QK^T/\sqrt D)V$

但模型中的完整 attention 模块还包括生成 QKV 投影，以及输出投影。完整流程大致是：

```Python
输入 X
  ↓
Q、K、V 投影
  ↓
QKᵀ → mask → softmax → PV
  ↓
输出投影
```

本教程中实现的 kernal 只负责中间的计算核心，输入已经是 QKV。

官方教程给出的特定 Qwen3\-0\.6B prefill 测量中，Attention 模块的时间主要花在投影矩阵乘上。

## Multi\-head 

上一部分只看了一个 head，QKV 都是 `(M,D)`。

教程程序中使用四维张量：

```Python
(B, H, M, D)
```

## `(batch, head)` 并行方案

为什么可以从 batch 和 head 两个方面并行？

原因：因为一个 head 的计算，不需要其他 head 的中间结果，不同 batch 样本也相互独立。

每个 triton Program 负责一条序列中的一个 head，完成这个 head 的全部 attention 计算。

官方使用：

```Python
grid = (B * H,)
```

## 官方完整可运行程序

```Python
# attention.py — 单 tile fused causal attention（B=1, H=4, M=64, D=64）
import torch
import triton
import triton.language as tl
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())


@triton.jit(do_not_specialize=["M", "D"])
def attn_fwd_kernel(Q, K, V, O, sm_scale, M, D,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                    BLOCK_D: tl.constexpr):
    pid = tl.program_id(0)               # 每 program 一个 (batch, head)
    base = pid * M * D                   # Q/K/V/O 都 reshape 成 (B*H, M, D)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    m_mask = offs_m < M                  # 行方向越界 mask
    n_mask = offs_n < M                  # 列（key）方向越界 mask

    # load Q tile (BLOCK_M, BLOCK_D)、K tile (BLOCK_N, BLOCK_D)
    q = tl.load(Q + base + offs_m[:, None] * D + offs_d[None, :],
                mask=m_mask[:, None], other=0.)
    k = tl.load(K + base + offs_n[:, None] * D + offs_d[None, :],
                mask=n_mask[:, None], other=0.)

    # S = Q·K^T * scale —— tl.dot 输出 f32（规则自动满足）
    qk = tl.dot(q, tl.trans(k)) * sm_scale

    # causal mask：只保留 j <= i，其余 -inf（softmax 后权重为 0）
    causal = offs_m[:, None] >= offs_n[None, :]
    qk = tl.where(causal, qk, float('-inf'))

    # 数值稳定 softmax（ops/04 §4.1 的三趟，在寄存器里完成）
    m_i = tl.max(qk, axis=1)             # (BLOCK_M,) 每行 max
    p = tl.exp(qk - m_i[:, None])        # f32，永不溢出
    l_i = tl.sum(p, axis=1)              # (BLOCK_M,) 每行分母

    # O = P·V / l_i
    v = tl.load(V + base + offs_n[:, None] * D + offs_d[None, :],
                mask=n_mask[:, None], other=0.)
    acc = tl.dot(p.to(tl.float16), v) / l_i[:, None]

    tl.store(O + base + offs_m[:, None] * D + offs_d[None, :],
             acc.to(O.dtype.element_ty), mask=m_mask[:, None])


def attention(q, k, v, causal=True):
    # q/k/v: (B, H, M, D) f16 contiguous
    B, H, M, D = q.shape
    o = torch.empty_like(q)
    sm_scale = D ** -0.5
    BLOCK_M = BLOCK_N = triton.next_power_of_2(M)
    BLOCK_D = triton.next_power_of_2(D)
    attn_fwd_kernel[(B * H,)](
        q.reshape(B * H, M, D), k.reshape(B * H, M, D),
        v.reshape(B * H, M, D), o.reshape(B * H, M, D),
        sm_scale, M, D,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D)
    return o


if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, M, D = 1, 4, 64, 64
    q = torch.randn(B, H, M, D, dtype=torch.float16)
    k = torch.randn(B, H, M, D, dtype=torch.float16)
    v = torch.randn(B, H, M, D, dtype=torch.float16)

    got = attention(q, k, v)

    # golden：manual f32 分解（不依赖任何 attention 库）
    scale = D ** -0.5
    s = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    mask = torch.tril(torch.ones(M, M, dtype=torch.bool))
    s = s.masked_fill(~mask, float('-inf'))
    ref = torch.matmul(torch.softmax(s, dim=-1), v.float()).half()

    torch.testing.assert_close(got, ref, atol=1e-2, rtol=1e-2)
    print("PASS attention B=%d H=%d M=%d D=%d  max diff=%.4f"
          % (B, H, M, D, (got - ref).abs().max().item()))
```

运行官方程序：

```Python
python attention.py
```

输出：

```Python
PASS attention B=1 H=4 M=64 D=64  max diff=0.0010
```

## 单 tile Kernal 实现解析

### Program 定位负责的数据

教程把张量看成：

```Python
(B * H, M, D)
```

在 kernal 中，定位：

```Python
pid = tl.program_id(0)
base = pid * M * D
```

`base` 是当前 head 起始位置的元素偏移量，不是字节数。

### 生成编号

```Python
offs_m = tl.arange(0, BLOCK_M)
offs_n = tl.arange(0, BLOCK_N)
offs_d = tl.arange(0, BLOCK_D)
```

`BLOCK_*` 是一个 program 要处理的 tile 尺寸。教程设置：

```Plain Text
BLOCK_M = BLOCK_N = triton.next_power_of_2(M)
BLOCK_D = triton.next_power_of_2(D)
```

在本章默认的 `M=D=64` 下，这三个值都是 64。

### 广播生成一整块地址

加载 Q 的核心表达式是：

```Python
Q + base + offs_m[:, None] * D + offs_d[None, :]
```

拆开看：

```Python
offs_m                 形状：(BLOCK_M,)
offs_m[:, None]        形状：(BLOCK_M, 1)

offs_d                 形状：(BLOCK_D,)
offs_d[None, :]        形状：(1, BLOCK_D)
```

两个方向相加，广播得到：

```Plain Text
(BLOCK_M, 1) + (1, BLOCK_D)
                  ↓
          (BLOCK_M, BLOCK_D)
```

例如，假设 `M=4, D=2`，忽略 `base`：

```Plain Text
行偏移：                 列偏移：

[[0],                    [[0, 1]]
 [2],
 [4],
 [6]]
```

相加后：

```Plain Text
[[0, 1],
 [2, 3],
 [4, 5],
 [6, 7]]
```

### 读入 QK

```Python
m_mask = offs_m < M
n_mask = offs_n < M

q = tl.load(
    Q + base + offs_m[:, None] * D + offs_d[None, :],
    mask=m_mask[:, None],
    other=0.,
)

k = tl.load(
    K + base + offs_n[:, None] * D + offs_d[None, :],
    mask=n_mask[:, None],
    other=0.,
)
```

加载后的矩阵形状：

```Python
q 的形状：(BLOCK_M, BLOCK_D)
k 的形状：(BLOCK_N, BLOCK_D)
```

现在加载的 K 还是原来的方向，未进行转置。

### 计算 QK 

先进行 K 的转置，然后计算点积：

```Python
qk = tl.dot(q, tl.trans(k))
```

`tl.trans(k)` 表达 tile 的转置，这里不需要先把转置后的 K 写回内存。对于教程使用的 float16 输入，`tl.dot` 默认以 float32 累加，得到 float32 的分数。

### 乘 `sm_scale`

```Python
sm_scale = D ** -0.5
```

$\text{sm\_scale}=\frac{1}{\sqrt D}$

```Python
qk = tl.dot(q, tl.trans(k)) * sm_scale
```

为什么需要缩放？

原因：点积会把 D 个乘积加起来，D 较大时，分数的波动可能变大。softmax 对分数差异非常敏感，容易让权重过度集中。除以 $\sqrt D$ 可以调节这个尺度。

### Causal 条件

$\text{causal}[i,j]=(i\ge j)$

```Python
causal = offs_m[:, None] >= offs_n[None, :]
```

### `tl.where` 完成掩码

```Python
qk = tl.where(causal, qk, float('-inf'))
```

逻辑：

```Python
causal=True  → 保留原分数
causal=False → 改成 -∞
```

### Softmax 计算

```Python
m_i = tl.max(qk, axis=1)
p = tl.exp(qk - m_i[:, None])
l_i = tl.sum(p, axis=1)
```

先求每行最大值：

```Python
m_i = tl.max(qk, axis=1)
```

`qk` 的形状是：

```Plain Text
(BLOCK_M, BLOCK_N)
```

`axis=1` 表示沿 key 列方向归约，因此得到：

```Plain Text
m_i：(BLOCK_M,)
```

`m_i[:, None]` 把形状变成 `(BLOCK_M, 1)`，广播到每一列，让每行减去自己的最大值，再求指数。

```Python
p = tl.exp(qk - m_i[:, None])
```

求出每行指数之和，作为归一化分母：

```Python
l_i = tl.sum(p, axis=1)
```

```Python
l_i：(BLOCK_M,)
```

对 p 归一化，得到最终的注意力权重：

```Python
weights = p / l_i[:, None]
```

### 计算 V 和写回结果

```SQL
v = tl.load(
    V + base + offs_n[:, None] * D + offs_d[None, :],
    mask=n_mask[:, None],
    other=0.,
)

acc = tl.dot(p.to(tl.float16), v) / l_i[:, None]

tl.store(
    O + base + offs_m[:, None] * D + offs_d[None, :],
    acc.to(O.dtype.element_ty),
    mask=m_mask[:, None],
)
```

K 与 V 的行一一对应：第 `j` 个 key 得到的权重，要用于第 `j` 个 value。因此，V 和 K 一样，沿 `offs_n` 加载：

```Python
V + base + offs_n[:, None] * D + offs_d[None, :]
```

地址计算与之前加载 Q、K 的方式相同。

上一部分得到：p ，但是尚未归一化的指数值。

```Python
p：(BLOCK_M, BLOCK_N)
```

现在计算：

```Python
p                     v
(BLOCK_M, BLOCK_N) × (BLOCK_N, BLOCK_D)
                         ↓
                (BLOCK_M, BLOCK_D)
```

展开到单个元素就是：

$(pV)_{i,d}=\sum_j p_{i,j}V_{j,d}$

含义是：对于 query 位置 `i`，把各个位置的 V 按对应权重累加起来。

然后再除以该行分母：

$O_{i,d}=\frac{\sum_j p_{i,j}V_{j,d}}{l_i}$

```Python
acc = tl.dot(p.to(tl.float16), v) / l_i[:, None]
```

`l_i[:, None]` 的形状是 `(BLOCK_M, 1)`，广播后，同一输出行里的所有维度都除以同一个分母。

**为什么将 p 转成 float16？**

教程使用 float16 的 QKV。第一次 `tl.dot` 得到的分数是 float32 ，随后求指数得到的 `p` 也是 float32。

第二次矩阵乘前，把左操作数转成 float16，与 V 配合进行矩阵乘。乘法的累加结果仍采用 float32，随后的除法也在 float32 中完成。

```Plain Text
p.to(tl.float16)
```

整体数据类型变化：

```Python
Q、K：float16
       ↓ 第一次 dot，以 float32 累加
qk：float32
       ↓ 减最大值、exp、sum
p、l_i：float32
       ↓ p 转成 float16，与 float16 的 V 做 dot
累加结果：float32
       ↓ 除以 l_i
acc：float32
       ↓ 写出前转换
O：float16
```

输出地址：

```Python
O + base + offs_m[:, None] * D + offs_d[None, :]
```

这里使用 `offs_m`，因为每个 query 位置产生一行输出。

例如第 `i` 个 query 的第 `d` 个输出分量，写到：

```Plain Text
O + base + i * D + d
```

写入前把结果转换成输出张量的元素类型。教程中，O 由 `torch.empty_like(q)` 创建，所以它与 Q 一样是 float16。

```Python
acc.to(O.dtype.element_ty)
```

另外再加上掩码机制防止写出越界：

```Python
mask=m_mask[:, None]
```

## 非 casual Attention 测试

修改好的 attention\_noncausal\.py：

```Python
import torch
import triton
import triton.language as tl
from triton.backends.spine_triton.driver import CPUDriver


triton.runtime.driver.set_active(CPUDriver())


@triton.jit(do_not_specialize=["M", "D"])
def attn_fwd_kernel(
    Q, K, V, O, sm_scale, M, D,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # 每个 program 负责一个 (batch, head)
    pid = tl.program_id(0)
    base = pid * M * D

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    m_mask = offs_m < M
    n_mask = offs_n < M
    d_mask = offs_d < D

    # Q: (BLOCK_M, BLOCK_D)
    q = tl.load(
        Q + base + offs_m[:, None] * D + offs_d[None, :],
        mask=m_mask[:, None] & d_mask[None, :],
        other=0.,
    )

    # K: (BLOCK_N, BLOCK_D)
    k = tl.load(
        K + base + offs_n[:, None] * D + offs_d[None, :],
        mask=n_mask[:, None] & d_mask[None, :],
        other=0.,
    )

    # 非因果 attention：保留所有有效 key，不限制 j <= i
    qk = tl.dot(q, tl.trans(k)) * sm_scale

    # 仅屏蔽 padding key，避免它们参与 softmax
    qk = tl.where(n_mask[None, :], qk, float("-inf"))

    # 数值稳定的 softmax，暂不归一化
    m_i = tl.max(qk, axis=1)
    p = tl.exp(qk - m_i[:, None])
    l_i = tl.sum(p, axis=1)

    # V: (BLOCK_N, BLOCK_D)
    v = tl.load(
        V + base + offs_n[:, None] * D + offs_d[None, :],
        mask=n_mask[:, None] & d_mask[None, :],
        other=0.,
    )

    # 加权汇总 V，再按行归一化
    acc = tl.dot(p.to(tl.float16), v) / l_i[:, None]

    tl.store(
        O + base + offs_m[:, None] * D + offs_d[None, :],
        acc.to(O.dtype.element_ty),
        mask=m_mask[:, None] & d_mask[None, :],
    )


def attention(q, k, v):
    # 本实验使用相同形状、连续存储的 float16 Q/K/V
    B, H, M, D = q.shape
    o = torch.empty_like(q)
    sm_scale = D ** -0.5

    BLOCK_M = BLOCK_N = triton.next_power_of_2(M)
    BLOCK_D = triton.next_power_of_2(D)

    attn_fwd_kernel[(B * H,)](
        q.reshape(B * H, M, D),
        k.reshape(B * H, M, D),
        v.reshape(B * H, M, D),
        o.reshape(B * H, M, D),
        sm_scale, M, D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )
    return o


if __name__ == "__main__":
    torch.manual_seed(0)

    B, H, M, D = 1, 4, 64, 64
    q = torch.randn(B, H, M, D, dtype=torch.float16)
    k = torch.randn(B, H, M, D, dtype=torch.float16)
    v = torch.randn(B, H, M, D, dtype=torch.float16)

    got = attention(q, k, v)

    # float32 参考实现：不添加因果掩码
    scale = D ** -0.5
    scores = torch.matmul(
        q.float(), k.float().transpose(-1, -2)
    ) * scale
    weights = torch.softmax(scores, dim=-1)
    ref = torch.matmul(weights, v.float()).half()

    torch.testing.assert_close(got, ref, atol=1e-2, rtol=1e-2)

    max_diff = (got.float() - ref.float()).abs().max().item()
    print(
        f"PASS noncausal attention B={B} H={H} M={M} D={D}"
        f"  max diff={max_diff:.4f}"
    )

    # 非因果模式下，第一个 query 也能访问后面的 key/value
    first_diff = (
        got[0, 0, 0].float() - v[0, 0, 0].float()
    ).abs().max().item()
    print(f"第一个位置的输出与 V[0] 的最大差异: {first_diff:.4f}")
```

输出：

```Python
PASS noncausal attention B=1 H=4 M=64 D=64  max diff=0.0005
第一个位置的输出与 V[0] 的最大差异: 3.9868
```

结果符合预期，去掉因果掩码后，第一个 query 可以关注整个序列，输出变成多个 V 向量的加权汇总。

## 2\*2 小输入测试非 casual

代码：

```Python
import torch

# 导入时也会执行原文件中的 CPUDriver 激活代码；
# 原文件的 __main__ 测试不会执行。
from attention_noncausal import attn_fwd_kernel


def test_noncausal_small():
    # 形状：(B=1, H=1, M=2, D=2)
    q = torch.tensor(
        [[[[1., 0.],
           [0., 1.]]]],
        dtype=torch.float16,
    )
    k = torch.tensor(
        [[[[1., 0.],
           [1., 1.]]]],
        dtype=torch.float16,
    )
    v = torch.tensor(
        [[[[10., 20.],
           [30., 40.]]]],
        dtype=torch.float16,
    )
    out = torch.empty_like(q)

    # 一个 head，所以只启动一个 program。
    # 明确使用 scale=1，与手算保持一致。
    attn_fwd_kernel[(1,)](
        q, k, v, out,
        1.0, 2, 2,
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_D=64,
    )

    # 直接写入手算答案，不用另一套 attention 实现生成答案。
    expected = torch.tensor(
        [[[[20., 30.],
           [24.6211716, 34.6211716]]]],
        dtype=torch.float32,
    )

    print("kernel 输出:")
    print(out[0, 0])
    print("手算期望:")
    print(expected[0, 0])

    # 在这个数值范围内，需容许 float16 输出舍入及中间转换误差。
    torch.testing.assert_close(
        out.float(), expected,
        atol=0.02,
        rtol=0,
    )
    print("PASS：2×2 非因果 attention 手算测试")


if __name__ == "__main__":
    test_noncausal_small()
```

输出：

```Python
kernel 输出:
tensor([[20.0000, 30.0000],
        [24.6250, 34.6250]], dtype=torch.float16)
手算期望:
tensor([[20.0000, 30.0000],
        [24.6212, 34.6212]])
PASS：2×2 非因果 attention 手算测试
```

输出符合手算的预期。

这个测试验证了当前配置下的 padding 屏蔽（让多出来的位置不参与计算）。测试中，实际只有 2 个 key，虽然使用了 64\*64 tile，多出来的位置没有错误地分走注意力权重。

## 单 tile 方案的瓶颈

目前的 kernel，一个 program 处理一个 head 的整段序列：

```Plain Text
加载整个 Q、K
      ↓
生成整个 M×M 分数矩阵
      ↓
softmax
      ↓
乘 V，输出结果
```

但是分数矩阵大小随序列长度的平方增长。除此之外，还需要 QKV 、指数值、输出累加器等中间数据。

解决方向：每次只处理一小块分数矩阵，处理完就累积结果。

## 分块处理的方案

### 把 Q 按行分块

假设：

```Plain Text
序列长度 M = 256
向量维度 D = 64
BLOCK_M = 64
```

可以把 Q 分成四块：

```Plain Text
Q 块 0：query 位置   0～63
Q 块 1：query 位置  64～127
Q 块 2：query 位置 128～191
Q 块 3：query 位置 192～255
```

每块 Q 的形状都是：

```Plain Text
(BLOCK_M, D) = (64, 64)
```

这些行块对应不同的输出行，可以分别计算。

任务分配细化了：

```Python
(batch, head) --> (batch, head, query 行块)
```

### KV 也按序列分块

对于一个固定的 Q 块，不一次加载完整 K、V，而是逐块扫描：

```Plain Text
固定一个 Q 块
    ↓
读取 K/V 块 0 → 计算本块分数和贡献
    ↓
读取 K/V 块 1 → 更新累积结果
    ↓
读取 K/V 块 2 → 更新累积结果
    ↓
读取 K/V 块 3 → 更新累积结果
    ↓
输出这个 Q 块对应的结果
```

例如 `BLOCK_N=64`，每次矩阵乘的形状是：

```Plain Text
Q 块：64×64
K 块转置：64×64
          ↓
局部分数：64×64
```

局部分数只需要 `16 KiB`，不再一次生成整个 `256×256` 分数矩阵。

这里要区分：

- `BLOCK_M`：一次处理多少个 query。

- `BLOCK_N`：一次扫描多少个 key/value。

- `D`：每个向量的维度。

两种块大小可以不同。

### 分块后的 causal mask

必须比较整段序列中的位置，不能只比较块内部编号。

例如当前处理：

```Plain Text
Q 位置：64～127
K 位置： 0～63
```

这些 key 都在 query 之前，因此整个块都可见。

如果当前处理：

```Plain Text
Q 位置：64～127
K 位置：64～127
```

这是对角块，需要逐元素判断：

```Plain Text
query_global_position >= key_global_position
```

如果 K 块的位置全部晚于当前 Q 块，则 causal Attention 可以直接跳过它。

### Softmax 难点

假设某个 query 对四个 key 的分数如下，竖线表示切成两块。

```Plain Text
[1, 2 | 3, 4]
```

如果分别做 softmax：

```Plain Text
softmax([1, 2]) → 两个权重之和为 1
softmax([3, 4]) → 两个权重之和也为 1
```

合起来权重总和就是 2，显然不是整行 softmax。

两块独立归一化后，已经丢失了它们之间的相对大小关系。

矩阵乘容易分块，softmax 的分母却依赖整行所有 key。

## Online Softmax

问题：分块扫描 KV 时，如何得到整行 attention 的结果。

### 保存内容

对于一个 query，假设已经处理过一些 key，保存三个状态：

$m=\max_j s_j$

$l=\sum_j e^{s_j-m}$

$acc=\sum_j e^{s_j-m}V_j$

这里的 $j$ 只包含目前已经扫描过的位置。

- `m`：目前的最大分数。

- `l`：以这个最大值为基准的指数和。

- `acc`：使用相同指数权重计算的 V 加权和，是一个 D 维向量。

如果现在已经扫描完整行，输出就是：

$O=\frac{acc}{l}$

难点在于：下一块可能带来更大的最大值，旧的 `l` 和 `acc` 就需要换算到新的基准。

### 用例子理解更新

假设一行分数及其对应的 V 是：

```Plain Text
分数：[ 1,  2 |  3,  4]
V：  [10, 20 | 30, 40]
```

这里为方便手算，让每个 V 只有一个元素。真实 kernel 中，V 是向量，推导相同。

先处理第一块：

```Plain Text
分数：[1, 2]
V：  [10, 20]
```

最大值：

$m_{\text{old}}=2$

减去最大值后，指数权重为：

```Plain Text
[e⁻¹, 1]
```

因此保存：

$l_{\text{old}}=e^{-1}+1$

$acc_{\text{old}}=10e^{-1}+20$

暂时不做最终归一化，继续扫描。

下面处理第二块。最大值发生改变。

第二块分数是 `[3,4]`，所以新的全局最大值为：

$m_{\text{new}}=\max(2,4)=4$

第二块以 4 为基准计算：

```Plain Text
[e^(3−4), e^(4−4)] = [e⁻¹, 1]
```

但旧结果以 2 为基准，两者不能直接相加。

例如第一个位置，旧权重是：

$e^{1-2}=e^{-1}$

换成新的基准后应该是：

$e^{1-4}=e^{-3}$

它们之间差一个共同系数：

$\alpha=e^{m_{\text{old}}-m_{\text{new}}}=e^{-2}$

因为：

$e^{s-m_{\text{old}}}\alpha =e^{s-m_{\text{new}}}$

所以，所有旧权重都只需要乘同一个 $\alpha$，无需重新读取旧分数。

`l` 和 `acc` 都由旧权重累加而来，因此必须一起修正：

### 代码逻辑

下面是示意代码，省略了加载、mask 和类型转换：

```Python
# 当前 Q 行块对应的状态
m = -inf
l = 0
acc = 0

for 每个有效的_KV块:
    scores = Q块 @ K块.T * scale

    new_m = maximum(m, 本块每行最大分数)
    alpha = exp(m - new_m)
    p = exp(scores - new_m)

    l = alpha * l + 本块每行的sum(p)
    acc = alpha * acc + p @ V块
    m = new_m

output = acc / l
```

### 作用

处理完一个 K/V 块后，只需要保留 `m、l、acc`，本块的分数和指数 tile 就可以被后续块复用。

## 循环扫描 KV

### 每个 Program 的任务

Q 块在循环外加载一次，K/V 块在循环内不断更换。

假设：

```Plain Text
M = 256
D = 64
BLOCK_M = 64
BLOCK_N = 64
```

一个 program 负责其中 64 个 query，例如位置 `64～127`：

```Python
固定 Q[64:128]
      ↓
扫描 K/V[0:64]
      ↓
扫描 K/V[64:128]
      ↓
扫描 K/V[128:192]
      ↓
扫描 K/V[192:256]
      ↓
写回 O[64:128]
```

### 循环更新的 triton 代码

假设 `q` 已加载，循环中的 `k、v、n_mask` 也已按当前 K/V 块准备好。假设序列非空，遍历的每个 K/V 块至少有一个有效 key。加入 causal mask 后，还需要处理跳过不可见块等情况。

```Python
# 循环外：为每个 query 初始化状态
m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
l_i = tl.zeros((BLOCK_M,), tl.float32)
acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

# 循环内：处理当前 K/V 块
scores = tl.dot(q, tl.trans(k)) * sm_scale
scores = tl.where(n_mask[None, :], scores, float("-inf"))

# 每行的新最大值
m_new = tl.maximum(m_i, tl.max(scores, axis=1))

# 旧状态换算到新最大值对应的尺度
alpha = tl.exp(m_i - m_new)

# 当前块相对于新最大值的指数权重
p = tl.exp(scores - m_new[:, None])

# 更新每行分母
l_i = l_i * alpha + tl.sum(p, axis=1)

# 更新每行的 V 加权和
acc = acc * alpha[:, None]
acc = acc + tl.dot(p.to(tl.float16), v)

# 保存新最大值，供下一块使用
m_i = m_new

# 所有 K/V 块处理完后，在循环外归一化
out = acc / l_i[:, None]
```

### KV 地址计算

KV 的地址如何随循环改变？

之前单 tile 的 key 位置是：

```Python
offs_n = tl.arange(0, BLOCK_N)
```

现在需要加上当前块的起始位置：

```Python
offs_n = start_n + tl.arange(0, BLOCK_N)
```

例如：

```Plain Text
start_n=0   → offs_n 为   0～63
start_n=64  → offs_n 为  64～127
start_n=128 → offs_n 为 128～191
start_n=192 → offs_n 为 192～255
```

再用全局位置计算地址：

```Python
K + base + offs_n[:, None] * D + offs_d[None, :]
```

边界检查也使用全局位置：

```Python
n_mask = offs_n < M
```

## 和单 tile 版本的对比

原来单 tile 版本：

```Python
一次算完全部分数
→ 求整行最大值和分母
→ 乘 V
→ 归一化
```

当前版本：

```Python
固定一个 Q 块
→ 循环读取 K/V 块
→ 每块更新 m_i、l_i、acc
→ 扫描结束后归一化
```

最终写回的仍然是当前 query 块对应的输出行。

## 非 causal 分块 代码实现

固定 Q 行块，循环扫描 KV，并用 online Softmax 更新结果。

这是根据官方教程 8\.5 的思路实现的，目前使用 `tl.dot`，没有使用 `smt.dot` 和任务调度。

完整代码：

```Python
import torch
import triton
import triton.language as tl
from triton.backends.spine_triton.driver import CPUDriver


triton.runtime.driver.set_active(CPUDriver())


@triton.jit(do_not_specialize=["M", "D"])
def attn_tiled_kernel(
    Q, K, V, O,
    sm_scale, M, D,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # grid 第 0 维：query 行块
    # grid 第 1 维：(batch, head)
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    base = pid_bh * M * D

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    m_mask = offs_m < M
    d_mask = offs_d < D

    # Q 块只加载一次
    q = tl.load(
        Q + base + offs_m[:, None] * D + offs_d[None, :],
        mask=m_mask[:, None] & d_mask[None, :],
        other=0.,
    )

    # 每个 query 行分别保存 online softmax 状态
    m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    # 沿序列方向扫描 K/V
    for start_n in range(0, M, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < M

        k = tl.load(
            K + base + offs_n[:, None] * D + offs_d[None, :],
            mask=n_mask[:, None] & d_mask[None, :],
            other=0.,
        )
        v = tl.load(
            V + base + offs_n[:, None] * D + offs_d[None, :],
            mask=n_mask[:, None] & d_mask[None, :],
            other=0.,
        )

        # 当前 Q 块与当前 K 块的匹配分数
        scores = tl.dot(q, tl.trans(k)) * sm_scale

        # 非因果模式：只屏蔽 padding key
        scores = tl.where(
            n_mask[None, :], scores, float("-inf")
        )

        # 更新行最大值
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))

        # 旧状态的尺度修正系数
        alpha = tl.exp(m_i - m_new)

        # 当前块使用新的行最大值计算指数
        p = tl.exp(scores - m_new[:, None])

        # 分母、输出累加器必须同时修正
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = acc + tl.dot(p.to(tl.float16), v)

        m_i = m_new

    # 所有 K/V 块处理完后，再归一化
    out = acc / l_i[:, None]

    tl.store(
        O + base + offs_m[:, None] * D + offs_d[None, :],
        out.to(O.dtype.element_ty),
        mask=m_mask[:, None] & d_mask[None, :],
    )


def attention_tiled(q, k, v):
    # 本教学版本：CPU、连续布局、等长 Q/K/V、相同 head 数
    assert q.ndim == 4
    assert q.shape == k.shape == v.shape
    assert all(x.dtype == torch.float16 for x in (q, k, v))
    assert all(x.device.type == "cpu" for x in (q, k, v))
    assert all(x.is_contiguous() for x in (q, k, v))

    B, H, M, D = q.shape
    assert B > 0 and H > 0 and M > 0
    assert D == 64, "本轮实验先固定 D=64"

    o = torch.empty_like(q)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = 64

    grid = (triton.cdiv(M, BLOCK_M), B * H)

    attn_tiled_kernel[grid](
        q, k, v, o,
        D ** -0.5, M, D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )
    return o


if __name__ == "__main__":
    torch.manual_seed(0)

    B, H, M, D = 1, 4, 256, 64
    q = torch.randn(B, H, M, D, dtype=torch.float16)
    k = torch.randn(B, H, M, D, dtype=torch.float16)
    v = torch.randn(B, H, M, D, dtype=torch.float16)

    got = attention_tiled(q, k, v)

    # 完整 float32 参考计算，不加 causal mask
    scores = torch.matmul(
        q.float(), k.float().transpose(-1, -2)
    ) * (D ** -0.5)
    ref = torch.matmul(
        torch.softmax(scores, dim=-1), v.float()
    ).half()

    # 先明确排除 NaN、Inf，再检查数值误差
    assert torch.isfinite(got).all().item(), "输出包含 NaN 或 Inf"
    torch.testing.assert_close(
        got, ref, atol=1e-2, rtol=1e-2
    )

    max_diff = (got.float() - ref.float()).abs().max().item()
    print(
        f"PASS tiled noncausal attention "
        f"B={B} H={H} M={M} D={D}  max diff={max_diff:.4f}"
    )
```

输出：

```Python
PASS tiled noncausal attention B=1 H=4 M=256 D=64  max diff=0.0005
```

将 M 改成 250，验证序列长度不是块大小整数倍时，最后一块能否正确处理。

```Python
B, H, M, D = 1, 4, 256, 64 --> B, H, M, D = 1, 4, 250, 64
```

输出：

```Python
PASS tiled noncausal attention B=1 H=4 M=250 D=64  max diff=0.0002
```

## causal 分块 代码实现

```Python
import torch
import triton
import triton.language as tl
from triton.backends.spine_triton.driver import CPUDriver


triton.runtime.driver.set_active(CPUDriver())


@triton.jit(do_not_specialize=["M", "D"])
def attn_tiled_causal_kernel(
    Q, K, V, O,
    sm_scale, M, D,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # 一个 program 负责一个 (batch, head) 中的一个 Q 行块
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    base = pid_bh * M * D

    # 整段序列中的全局 query 位置
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    m_mask = offs_m < M
    d_mask = offs_d < D

    # 当前 Q 块只加载一次
    q = tl.load(
        Q + base + offs_m[:, None] * D + offs_d[None, :],
        mask=m_mask[:, None] & d_mask[None, :],
        other=0.,
    )

    # 每行的 online softmax 状态
    m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    # 只扫描当前 Q 块可能访问的 K/V 范围
    end_n = tl.minimum((pid_m + 1) * BLOCK_M, M)

    for start_n in range(0, end_n, BLOCK_N):
        # 整段序列中的全局 key/value 位置
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < M

        k = tl.load(
            K + base + offs_n[:, None] * D + offs_d[None, :],
            mask=n_mask[:, None] & d_mask[None, :],
            other=0.,
        )
        v = tl.load(
            V + base + offs_n[:, None] * D + offs_d[None, :],
            mask=n_mask[:, None] & d_mask[None, :],
            other=0.,
        )

        # 当前 Q 块与 K 块的匹配分数
        scores = tl.dot(q, tl.trans(k)) * sm_scale

        # 同时屏蔽 padding key 和未来位置
        causal = offs_m[:, None] >= offs_n[None, :]
        visible = n_mask[None, :] & causal
        scores = tl.where(visible, scores, float("-inf"))

        # 更新行最大值
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))

        # 将旧分母、旧累加器换算到新最大值对应的尺度
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = acc + tl.dot(p.to(tl.float16), v)

        m_i = m_new

    # 扫描完所有可见 K/V 块，再进行归一化
    out = acc / l_i[:, None]

    tl.store(
        O + base + offs_m[:, None] * D + offs_d[None, :],
        out.to(O.dtype.element_ty),
        mask=m_mask[:, None] & d_mask[None, :],
    )


def attention_tiled_causal(q, k, v):
    # 本教学版本：
    # CPU、float16、连续布局、Q/K/V 等长且 head 数相同
    assert q.ndim == 4
    assert q.shape == k.shape == v.shape
    assert all(x.dtype == torch.float16 for x in (q, k, v))
    assert all(x.device.type == "cpu" for x in (q, k, v))
    assert all(x.is_contiguous() for x in (q, k, v))

    B, H, M, D = q.shape
    assert B > 0 and H > 0 and M > 0
    assert D == 64, "本轮实验固定 D=64"

    o = torch.empty_like(q)

    # 本版本保持 Q、K/V 的序列块大小相同
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = 64

    grid = (triton.cdiv(M, BLOCK_M), B * H)

    attn_tiled_causal_kernel[grid](
        q, k, v, o,
        D ** -0.5, M, D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )
    return o


if __name__ == "__main__":
    torch.manual_seed(0)

    # 非整块长度，同时验证 padding 边界
    B, H, M, D = 1, 4, 250, 64

    q = torch.randn(B, H, M, D, dtype=torch.float16)
    k = torch.randn(B, H, M, D, dtype=torch.float16)
    v = torch.randn(B, H, M, D, dtype=torch.float16)

    got = attention_tiled_causal(q, k, v)

    # 完整 float32 因果 attention 参考实现
    scores = torch.matmul(
        q.float(), k.float().transpose(-1, -2)
    ) * (D ** -0.5)

    causal_mask = torch.tril(
        torch.ones(M, M, dtype=torch.bool)
    )
    scores = scores.masked_fill(
        ~causal_mask, float("-inf")
    )

    ref = torch.matmul(
        torch.softmax(scores, dim=-1), v.float()
    ).half()

    assert torch.isfinite(got).all().item(), "输出包含 NaN 或 Inf"
    torch.testing.assert_close(
        got, ref, atol=1e-2, rtol=1e-2
    )

    max_diff = (got.float() - ref.float()).abs().max().item()
    print(
        f"PASS tiled causal attention "
        f"B={B} H={H} M={M} D={D}  max diff={max_diff:.4f}"
    )

    # 因果模式：第一个 query 只能访问第一个 value。
    # 检查所有 batch、head 的第一个位置。
    first_diff = (
        got[:, :, 0, :].float() - v[:, :, 0, :].float()
    ).abs().max().item()

    torch.testing.assert_close(
        got[:, :, 0, :].float(),
        v[:, :, 0, :].float(),
        atol=1e-2,
        rtol=1e-2,
    )
    print(f"第一个位置的输出与对应 V[0] 的最大差异: {first_diff:.4f}")
```

输出：

```Python
PASS tiled causal attention B=1 H=4 M=250 D=64  max diff=0.0010
第一个位置的输出与对应 V[0] 的最大差异: 0.0000
```

## GQA

### 简介

Grouped\-Query Attention，分组查询注意力。多个 Q head 共享一组 K/V head。

大模型逐 token 生成时，历史 K/V 的存储和读取成本很高。让多个 Q head 共享 K/V，可以降低这些成本，同时保留多个 Q head 的查询能力。

模型接下来逐个生成 token 时，每一步通常只产生一个新位置的 Q，但它需要关注此前所有位置的 K/V。

为了避免反复计算，模型会保存每层历史位置的 K/V，这就是 KV cache：

```Python
生成新 token
    ↓
计算当前位置的 Q、K、V
    ↓
把新 K、V 追加到 KV cache
    ↓
用新 Q 查询历史 K/V
```

随着上下文变长，缓存越来越大，每一步需要访问的历史数据也越来越多。

普通多头注意力 MHA 中，每个 Q head 都对应独立的 K/V head。

假设有 32 个 Q head：

GQA 位于 MHA 与 MQA 之间，用 K/V head 数量调节存储开销和模型表达能力。

目前的 head 是一一对应的。已经跑通的版本要求：

```Python
Q：(B, H, M, D)
K：(B, H, M, D)
V：(B, H, M, D)
```

每个 Q head 使用同编号的 K/V head：

```Plain Text
Q head 0 → K/V head 0
Q head 1 → K/V head 1
Q head 2 → K/V head 2
Q head 3 → K/V head 3
```

GQA 让 Q head 数量多于 K/V head 数量，一组 Q head 共享一个 K/V head。

### 举例

接下来使用：

```Python
B = 1
Hq = 4
Hkv = 2
M = 250
D = 64
```

输入形状为：

```Python
Q：(1, 4, 250, 64)
K：(1, 2, 250, 64)
V：(1, 2, 250, 64)
```

每组包含的 Q head 数：

```Plain Text
GROUP_SIZE = Hq // Hkv
```

这里为 2，映射关系是：

共享的是 K/V 数据，不是注意力权重。 Q head 0 和 Q head 1 使用相同的 K/V，但各自的 Q 不同，仍分别计算分数、softmax 和输出。

### 输出形状由 Q 决定

每个 query 都需要自己的输出，因此：

```Plain Text
O：(B, Hq, M, D)
```

本例输出仍是：

```Plain Text
(1, 4, 250, 64)
```

每个 program 的 online softmax 状态也仍独立维护：

```Python
m_i、l_i、acc
```

GQA 不需要改变这些更新公式，主要改变的是 K/V head 的寻址。

### K V 寻址

原代码使用：

```Plain Text
base = pid_bh * M * D
```

Q、K、V、O 都使用这个偏移，是因为它们的 head 数相同。

GQA 中必须先从任务编号解出 batch 和 Q head：

```Plain Text
batch = pid_bh // Hq
hq = pid_bh % Hq

hkv = hq // GROUP_SIZE
```

然后分别计算偏移：

```Plain Text
q_base = (batch * Hq + hq) * M * D
kv_base = (batch * Hkv + hkv) * M * D
```

使用方式：

```Plain Text
加载 Q、写出 O → q_base
加载 K、V     → kv_base
```

### Program 数量计算

任务划分仍然是：

```Plain Text
一个 program
    ↓
一个 batch、一个 Q head、一个 query 行块
```

所以 grid 是：

```Plain Text
grid = (triton.cdiv(M, BLOCK_M), B * Hq)
```

不能改成 `B * Hkv`，因为每个 Q head 都需要独立计算输出。

### Causal \+ 分块 \+ GQA 代码实现

```Python
import torch
import triton
import triton.language as tl
from triton.backends.spine_triton.driver import CPUDriver


triton.runtime.driver.set_active(CPUDriver())


@triton.jit(do_not_specialize=["M", "D"])
def attn_tiled_gqa_kernel(
    Q, K, V, O,
    sm_scale, M, D,
    Hq: tl.constexpr,
    Hkv: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    # 从任务编号解析 batch 和 Q head
    batch = pid_bh // Hq
    hq = pid_bh % Hq

    # 多个 Q head 映射到同一个 K/V head
    group_size = Hq // Hkv
    hkv = hq // group_size

    # Q/O 与 K/V 的 head 数不同，分别计算偏移
    q_base = (batch * Hq + hq) * M * D
    kv_base = (batch * Hkv + hkv) * M * D

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    m_mask = offs_m < M
    d_mask = offs_d < D

    q = tl.load(
        Q + q_base + offs_m[:, None] * D + offs_d[None, :],
        mask=m_mask[:, None] & d_mask[None, :],
        other=0.,
    )

    m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    # 标准等长 causal attention：跳过完全位于未来的块
    end_n = tl.minimum((pid_m + 1) * BLOCK_M, M)

    for start_n in range(0, end_n, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < M

        k = tl.load(
            K + kv_base + offs_n[:, None] * D + offs_d[None, :],
            mask=n_mask[:, None] & d_mask[None, :],
            other=0.,
        )
        v = tl.load(
            V + kv_base + offs_n[:, None] * D + offs_d[None, :],
            mask=n_mask[:, None] & d_mask[None, :],
            other=0.,
        )

        scores = tl.dot(q, tl.trans(k)) * sm_scale

        causal = offs_m[:, None] >= offs_n[None, :]
        visible = n_mask[None, :] & causal
        scores = tl.where(visible, scores, float("-inf"))

        # Online softmax：公式与普通多头版本相同
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = acc + tl.dot(p.to(tl.float16), v)

        m_i = m_new

    out = acc / l_i[:, None]

    # 输出 head 数与 Q 相同，所以使用 q_base
    tl.store(
        O + q_base + offs_m[:, None] * D + offs_d[None, :],
        out.to(O.dtype.element_ty),
        mask=m_mask[:, None] & d_mask[None, :],
    )


def attention_tiled_gqa(q, k, v):
    # 本教学版本：等长 causal attention，不支持 KV cache decode
    assert q.ndim == k.ndim == v.ndim == 4
    assert k.shape == v.shape
    assert all(x.dtype == torch.float16 for x in (q, k, v))
    assert all(x.device.type == "cpu" for x in (q, k, v))
    assert all(x.is_contiguous() for x in (q, k, v))

    B, Hq, M, D = q.shape
    Bkv, Hkv, Mkv, Dkv = k.shape

    assert B > 0 and Hq > 0 and Hkv > 0 and M > 0
    assert Bkv == B and Mkv == M and Dkv == D
    assert Hq % Hkv == 0, "Hq 必须能被 Hkv 整除"
    assert D == 64, "本轮实验固定 D=64"

    o = torch.empty_like(q)

    BLOCK_M = BLOCK_N = BLOCK_D = 64
    grid = (triton.cdiv(M, BLOCK_M), B * Hq)

    attn_tiled_gqa_kernel[grid](
        q, k, v, o,
        D ** -0.5, M, D,
        Hq=Hq,
        Hkv=Hkv,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )
    return o


if __name__ == "__main__":
    torch.manual_seed(0)

    B, Hq, Hkv, M, D = 1, 4, 2, 250, 64

    q = torch.randn(B, Hq, M, D, dtype=torch.float16)
    k = torch.randn(B, Hkv, M, D, dtype=torch.float16)
    v = torch.randn(B, Hkv, M, D, dtype=torch.float16)

    got = attention_tiled_gqa(q, k, v)

    # 仅参考路径显式重复 K/V，方便核对 GQA 映射
    # head 顺序：[0, 1] -> [0, 0, 1, 1]
    group_size = Hq // Hkv
    k_expanded = k.float().repeat_interleave(group_size, dim=1)
    v_expanded = v.float().repeat_interleave(group_size, dim=1)

    scores = torch.matmul(
        q.float(), k_expanded.transpose(-1, -2)
    ) * (D ** -0.5)

    causal_mask = torch.tril(
        torch.ones(M, M, dtype=torch.bool)
    )
    scores = scores.masked_fill(~causal_mask, float("-inf"))

    ref = torch.matmul(
        torch.softmax(scores, dim=-1), v_expanded
    ).half()

    assert torch.isfinite(got).all().item(), "输出包含 NaN 或 Inf"
    torch.testing.assert_close(
        got, ref, atol=1e-2, rtol=1e-2
    )

    max_diff = (got.float() - ref.float()).abs().max().item()
    print(
        f"PASS tiled causal GQA "
        f"B={B} Hq={Hq} Hkv={Hkv} M={M} D={D}"
        f"  max diff={max_diff:.4f}"
    )

    # 第一个 query 只能看到位置 0：
    # 每个 Q head 的输出应等于对应 K/V head 的 V[0]
    first_expected = v_expanded[:, :, 0, :]
    first_got = got[:, :, 0, :].float()

    torch.testing.assert_close(
        first_got, first_expected, atol=1e-2, rtol=1e-2
    )
    first_diff = (first_got - first_expected).abs().max().item()
    print(f"第一个位置与对应 V[0] 的最大差异: {first_diff:.4f}")
```

输出：

```Python
PASS tiled causal GQA B=1 Hq=4 Hkv=2 M=250 D=64  max diff=0.0010
第一个位置与对应 V[0] 的最大差异: 0.0000
```

这组测试验证了 `Hq=4、Hkv=2` 时的分组映射 `[0,0,1,1]`，以及它与因果掩码、尾块处理和 online softmax 的配合。

将上述代码中的 B 改为 2，只改变 batch 的数量，检查 GQA 的寻址是否能正确区分两条序列。

```Python
B, Hq, Hkv, M, D = 1, 4, 2, 250, 64 -->
B, Hq, Hkv, M, D = 2, 4, 2, 250, 64
```

运行结果：

```Python
PASS tiled causal GQA B=2 Hq=4 Hkv=2 M=250 D=64  max diff=0.0010
第一个位置与对应 V[0] 的最大差异: 0.0000
```

