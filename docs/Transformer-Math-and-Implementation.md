# Transformer 数学原理与实现 —— 从线性代数到代码

> **前置知识**：矩阵运算、softmax 函数、梯度下降、Python 基础

---

## 目录

1. [引言：为什么 Transformer 需要数学？](#1-引言为什么-transformer-需要数学)
2. [核心思想：注意力作为矩阵运算](#2-核心思想注意力作为矩阵运算)
3. [Scaled Dot-Product Attention 的数学推导](#3-scaled-dot-product-attention-的数学推导)
4. [Multi-Head Attention 的线性代数解释](#4-multi-head-attention-的线性代数解释)
5. [位置编码的傅里叶视角](#5-位置编码的傅里叶视角)
6. [从数学到代码：完整实现](#6-从数学到代码完整实现)
7. [实践中的关键技巧](#7-实践中的关键技巧)
8. [练习与思考题](#8-练习与思考题)

---

## 1. 引言：为什么 Transformer 需要数学？

### 1.1 问题的数学表述

**自然语言处理的核心问题**：给定输入序列 $X = (x_1, x_2, \ldots, x_n)$，如何学习一个函数 $f$ 使得输出 $Y = f(X)$ 有意义？

**RNN 的方案**：递归定义
$$
h_t = \sigma(W_h h_{t-1} + W_x x_t + b)
$$
**问题**：
- 计算无法并行（$h_t$ 依赖 $h_{t-1}$）
- 长距离依赖需要 $O(n)$ 次矩阵乘法
- 梯度 $\frac{\partial h_t}{\partial h_1} = \prod_{k=2}^t W_h^\top \sigma'(\cdot)$ 易消失

**Transformer 的方案**：直接建立任意两位置的关系
$$
\text{Attention}(x_i, x_j) = \text{similarity}(x_i, x_j) \cdot v_j
$$
**优势**：
- 任意两位置距离为 $O(1)$
- 可表示为矩阵运算，完全并行

### 1.2 本科数学知识映射表

| 数学概念 | Transformer 中的应用 | 代码对应 |
|---------|---------------------|---------|
| 矩阵乘法 $AB$ | 注意力分数计算 | `torch.matmul(Q, K.T)` |
| Softmax 函数 | 注意力权重归一化 | `F.softmax(scores, dim=-1)` |
| 特征值分解 | 多头注意力的子空间投影 | `nn.Linear(d_model, d_k)` |
| 傅里叶级数 | 位置编码的正余弦函数 | `torch.sin(pos / ...)` |
| 残差连接 | 梯度流动 | `x + sublayer(x)` |
| 层归一化 | 训练稳定性 | `nn.LayerNorm()` |

---

## 2. 核心思想：注意力作为矩阵运算

### 2.1 从向量到矩阵

假设我们有 $n$ 个词，每个词表示为 $d$ 维向量。

**输入矩阵**：
$$
X = \begin{bmatrix}
x_1^\top \\
x_2^\top \\
\vdots \\
x_n^\top
\end{bmatrix} \in \mathbb{R}^{n \times d}
$$

**关键问题**：如何让每个词 $x_i$ 聚合其他词的信息？

### 2.2 注意力权重的计算

**步骤 1：线性投影**

对每个词 $x_i$，学习三个线性变换：
$$
q_i = W_Q x_i, \quad k_i = W_K x_i, \quad v_i = W_V x_i
$$

其中 $W_Q, W_K \in \mathbb{R}^{d_k \times d}$，$W_V \in \mathbb{R}^{d_v \times d}$。

**矩阵形式**：
$$
Q = XW_Q^\top \in \mathbb{R}^{n \times d_k}, \quad K = XW_K^\top \in \mathbb{R}^{n \times d_k}, \quad V = XW_V^\top \in \mathbb{R}^{n \times d_v}
$$

**步骤 2：相似度计算**

词 $i$ 对词 $j$ 的注意力分数：
$$
\text{score}_{ij} = q_i^\top k_j = \langle q_i, k_j \rangle
$$

**矩阵形式**：
$$
\text{Scores} = QK^\top \in \mathbb{R}^{n \times n}
$$

> **数学直觉**：$QK^\top$ 的第 $(i,j)$ 元素是 $q_i$ 和 $k_j$ 的内积，衡量两者的"匹配程度"。

**步骤 3：归一化**

使用 softmax 将分数转为概率：
$$
\alpha_{ij} = \frac{\exp(\text{score}_{ij})}{\sum_{k=1}^n \exp(\text{score}_{ik})}
$$

**矩阵形式**：
$$
A = \text{softmax}(QK^\top, \text{dim}=-1) \in \mathbb{R}^{n \times n}
$$

其中 $A_{ij}$ 表示词 $i$ 关注词 $j$ 的权重。

**步骤 4：加权求和**

$$
\text{output}_i = \sum_{j=1}^n \alpha_{ij} v_j
$$

**矩阵形式**：
$$
O = AV = \text{softmax}(QK^\top)V \in \mathbb{R}^{n \times d_v}
$$

### 2.3 完整公式

$$
\boxed{\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V}
$$

---

## 3. Scaled Dot-Product Attention 的数学推导

### 3.1 为什么需要缩放因子 $\sqrt{d_k}$？

**问题**：当 $d_k$ 较大时，点积 $q_i^\top k_j$ 的方差会很大。

**推导**：

假设 $q_i$ 和 $k_j$ 的每个元素独立同分布，均值 0，方差 1：
$$
\mathbb{E}[q_{im}] = 0, \quad \text{Var}(q_{im}) = 1
$$

点积的方差：
$$
\begin{aligned}
\text{Var}(q_i^\top k_j) &= \text{Var}\left(\sum_{m=1}^{d_k} q_{im} k_{jm}\right) \\
&= \sum_{m=1}^{d_k} \text{Var}(q_{im} k_{jm}) \\
&= \sum_{m=1}^{d_k} \mathbb{E}[q_{im}^2]\mathbb{E}[k_{jm}^2] \quad (\text{独立性})\\
&= \sum_{m=1}^{d_k} 1 \cdot 1 = d_k
\end{aligned}
$$

**结论**：点积的标准差为 $\sqrt{d_k}$。

**缩放后**：
$$
\text{Var}\left(\frac{q_i^\top k_j}{\sqrt{d_k}}\right) = \frac{d_k}{d_k} = 1
$$

### 3.2 Softmax 梯度的分析

Softmax 函数：
$$
p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

梯度：
$$
\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j)
$$

**问题**：当 $z_i$ 很大时，softmax 趋近于 one-hot，梯度趋近于 0。

**示例**：
```python
import torch
import torch.nn.functional as F

# 未缩放的情况（d_k = 512）
z_large = torch.randn(512) * 22.6  # 标准差约为 sqrt(512) ≈ 22.6
p_large = F.softmax(z_large, dim=0)
print(f"Max probability (unscaled): {p_large.max().item():.4f}")  # 接近 1.0

# 缩放后的情况
z_scaled = z_large / 22.6
p_scaled = F.softmax(z_scaled, dim=0)
print(f"Max probability (scaled): {p_scaled.max().item():.4f}")  # 更均匀
```

---

## 4. Multi-Head Attention 的线性代数解释

### 4.1 子空间投影的思想

**核心思想**：将 $d_{model}$ 维空间投影到 $h$ 个不同的 $d_k$ 维子空间，在每个子空间独立计算注意力。

**数学表述**：

对于第 $i$ 个头：
$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

其中投影矩阵：
$$
W_i^Q \in \mathbb{R}^{d_{model} \times d_k}, \quad W_i^K \in \mathbb{R}^{d_{model} \times d_k}, \quad W_i^V \in \mathbb{R}^{d_{model} \times d_v}
$$

**拼接与输出投影**：
$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O
$$

其中 $W^O \in \mathbb{R}^{h \cdot d_v \times d_{model}}$。

### 4.2 参数效率分析

**原始论文配置**：
- $d_{model} = 512$
- $h = 8$ 头
- $d_k = d_v = 64$

**参数量计算**：

单头注意力参数：
$$
\begin{aligned}
\text{Params}_{\text{single}} &= d_{model} \cdot d_k \times 3 + d_k \cdot d_{model} \\
&= 512 \times 64 \times 3 + 64 \times 512 = 131,072
\end{aligned}
$$

多头注意力参数：
$$
\begin{aligned}
\text{Params}_{\text{multi}} &= h \cdot (d_{model} \cdot d_k \times 3) + (h \cdot d_v) \cdot d_{model} \\
&= 8 \times (512 \times 64 \times 3) + (8 \times 64) \times 512 = 1,048,576
\end{aligned}
$$

**等价性**：如果 $h \cdot d_k = d_{model}$，多头注意力的表达能力理论上等价于单头注意力，但实际中多头表现更好，因为：
1. 不同头可以学习不同的注意力模式
2. 优化更容易（更小的梯度方差）

### 4.3 多头的几何解释

**单头注意力**：在一个高维空间中计算相似度

**多头注意力**：在多个低维子空间中并行计算相似度，然后融合

```
d_model = 512 维空间
    |
    |--- 头 1: 投影到 64 维子空间 → 注意力 1
    |--- 头 2: 投影到 64 维子空间 → 注意力 2
    |--- ...
    |--- 头 8: 投影到 64 维子空间 → 注意力 8
    |
    拼接 → 512 维 → 线性投影 → 输出
```

---

## 5. 位置编码的傅里叶视角

### 5.1 正弦位置编码的公式

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

### 5.2 为什么使用正余弦函数？

**性质 1：相对位置可表示为线性变换**

对于位置 $pos$ 和 $pos+k$：
$$
\begin{aligned}
PE_{(pos+k, 2i)} &= \sin\left(\frac{pos+k}{10000^{2i/d}}\right) \\
&= \sin\left(\frac{pos}{10000^{2i/d}} + \frac{k}{10000^{2i/d}}\right) \\
&= \sin\left(\frac{pos}{10000^{2i/d}}\right)\cos\left(\frac{k}{10000^{2i/d}}\right) + \cos\left(\frac{pos}{10000^{2i/d}}\right)\sin\left(\frac{k}{10000^{2i/d}}\right)
\end{aligned}
$$

这意味着 $PE_{pos+k}$ 可以表示为 $PE_{pos}$ 的线性组合。

**性质 2：不同频率捕获不同尺度的位置信息**

波长序列：
$$
\lambda_i = 2\pi \cdot 10000^{2i/d_{model}}
$$

- $i=0$：波长 $2\pi \cdot 10000^0 = 2\pi$（高频，捕获局部位置）
- $i=d/2$：波长 $2\pi \cdot 10000^1 = 20000\pi$（低频，捕获全局位置）

### 5.3 与傅里叶级数的联系

**傅里叶级数**：任何周期函数可表示为正余弦函数的和：
$$
f(x) = \sum_{n=0}^{\infty} a_n \sin(n\omega x) + b_n \cos(n\omega x)
$$

**位置编码**：使用几何级数的频率：
$$
\omega_i = 10000^{-2i/d_{model}}
$$

这使得模型可以学习不同尺度的位置依赖关系。

### 5.4 代码实现与可视化

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

def get_positional_encoding(seq_len, d_model):
    """
    计算正弦位置编码
    
    参数:
        seq_len: 序列长度 n
        d_model: 嵌入维度 d
    
    返回:
        PE: (seq_len, d_model) 位置编码矩阵
    """
    position = torch.arange(seq_len).unsqueeze(1)  # (n, 1)
    
    # 计算频率：10000^(-2i/d_model)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    
    # 计算位置编码
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度用 sin
    pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度用 cos
    
    return pe

# 可视化
d_model = 512
seq_len = 100
pe = get_positional_encoding(seq_len, d_model)

# 绘制前 8 个维度的位置编码
plt.figure(figsize=(12, 6))
for i in range(8):
    plt.plot(pe[:, i].numpy(), label=f'dim {i}')
plt.xlabel('Position')
plt.ylabel('Encoding Value')
plt.title('Positional Encoding (First 8 Dimensions)')
plt.legend()
plt.savefig('positional_encoding_viz.png', dpi=150)
plt.show()
```

---

## 6. 从数学到代码：完整实现

### 6.1 Scaled Dot-Product Attention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class ScaledDotProductAttention(nn.Module):
    """
    缩放点积注意力
    
    数学公式:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    
    参数:
        d_k: 查询/键的维度
        dropout: dropout 概率
    """
    def __init__(self, d_k: int, dropout: float = 0.1):
        super().__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        Q: torch.Tensor,  # (batch, heads, seq_len, d_k)
        K: torch.Tensor,  # (batch, heads, seq_len, d_k)
        V: torch.Tensor,  # (batch, heads, seq_len, d_v)
        mask: Optional[torch.Tensor] = None  # (batch, 1, 1, seq_len) 或 (batch, 1, seq_len, seq_len)
    ) -> torch.Tensor:
        # 1. 计算注意力分数：QK^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        # scores shape: (batch, heads, seq_len, seq_len)
        
        # 2. 应用掩码（如果有）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 3. Softmax 归一化
        attention_weights = F.softmax(scores, dim=-1)
        # attention_weights shape: (batch, heads, seq_len, seq_len)
        
        # 4. Dropout
        attention_weights = self.dropout(attention_weights)
        
        # 5. 加权求和：Attention * V
        output = torch.matmul(attention_weights, V)
        # output shape: (batch, heads, seq_len, d_v)
        
        return output, attention_weights
```

### 6.2 Multi-Head Attention

```python
class MultiHeadAttention(nn.Module):
    """
    多头注意力
    
    数学公式:
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
        head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    
    参数:
        d_model: 模型维度（输入/输出维度）
        num_heads: 注意力头数
        dropout: dropout 概率
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度
        
        # 线性投影层
        self.W_Q = nn.Linear(d_model, d_model)  # 投影到 Q
        self.W_K = nn.Linear(d_model, d_model)  # 投影到 K
        self.W_V = nn.Linear(d_model, d_model)  # 投影到 V
        self.W_O = nn.Linear(d_model, d_model)  # 输出投影
        
        # 注意力机制
        self.attention = ScaledDotProductAttention(self.d_k, dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        Q: torch.Tensor,  # (batch, seq_len, d_model)
        K: torch.Tensor,  # (batch, seq_len, d_model)
        V: torch.Tensor,  # (batch, seq_len, d_model)
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = Q.size(0)
        seq_len = Q.size(1)
        
        # 1. 线性投影并分割成多头
        # Q, K, V shape: (batch, seq_len, d_model)
        Q_proj = self.W_Q(Q)  # (batch, seq_len, d_model)
        K_proj = self.W_K(K)  # (batch, seq_len, d_model)
        V_proj = self.W_V(V)  # (batch, seq_len, d_model)
        
        # 重塑为多头：(batch, num_heads, seq_len, d_k)
        Q_heads = Q_proj.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K_heads = K_proj.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V_heads = V_proj.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. 应用缩放点积注意力
        attention_output, attention_weights = self.attention(Q_heads, K_heads, V_heads, mask)
        # attention_output shape: (batch, num_heads, seq_len, d_k)
        
        # 3. 拼接多头
        concatenated = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        # concatenated shape: (batch, seq_len, d_model)
        
        # 4. 输出投影
        output = self.W_O(concatenated)
        output = self.dropout(output)
        
        return output, attention_weights
```

### 6.3 Position-wise Feed-Forward Network

```python
class PositionwiseFeedForward(nn.Module):
    """
    位置前馈网络
    
    数学公式:
        FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
               = ReLU(xW_1 + b_1)W_2 + b_2
    
    参数:
        d_model: 输入/输出维度
        d_ff: 隐藏层维度（通常为 d_model 的 4 倍）
        dropout: dropout 概率
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, d_model)
        x = self.linear1(x)  # (batch, seq_len, d_ff)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)  # (batch, seq_len, d_model)
        return x
```

### 6.4 Positional Encoding

```python
class PositionalEncoding(nn.Module):
    """
    正弦位置编码
    
    数学公式:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 注册为 buffer（不参与梯度更新）
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
```

### 6.5 Encoder Layer

```python
class EncoderLayer(nn.Module):
    """
    Encoder 层
    
    结构:
        x -> [Multi-Head Attention] -> [Add & Norm] -> [Feed Forward] -> [Add & Norm] -> output
    
    数学公式:
        x1 = LayerNorm(x + MultiHeadAttention(x, x, x))
        output = LayerNorm(x1 + FFN(x1))
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 1. 自注意力 + 残差 + 归一化
        attn_output, _ = self.self_attn(x, x, x, mask)
        x1 = self.norm1(x + self.dropout(attn_output))
        
        # 2. 前馈网络 + 残差 + 归一化
        ff_output = self.feed_forward(x1)
        output = self.norm2(x1 + self.dropout(ff_output))
        
        return output
```

### 6.6 完整 Encoder

```python
class Encoder(nn.Module):
    """
    Transformer Encoder
    
    结构:
        [Embedding + Positional Encoding] -> [Encoder Layer x N] -> output
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 5000,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x shape: (batch, seq_len)
        x = self.embedding(x)  # (batch, seq_len, d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        return x  # (batch, seq_len, d_model)
```

### 6.7 使用示例

```python
# 测试代码
if __name__ == "__main__":
    # 超参数
    vocab_size = 10000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    batch_size = 32
    seq_len = 100
    
    # 创建模型
    encoder = Encoder(vocab_size, d_model, num_heads, num_layers, d_ff)
    
    # 创建假数据
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # 前向传播
    output = encoder(x)
    
    print(f"输入形状：{x.shape}")
    print(f"输出形状：{output.shape}")
    print(f"参数量：{sum(p.numel() for p in encoder.parameters()):,}")
```

---

## 7. 实践中的关键技巧

### 7.1 学习率调度器

原始论文使用的 warmup + decay 策略：

```python
import math

class TransformerLRScheduler:
    """
    Transformer 学习率调度器
    
    公式:
        lrate = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))
    """
    def __init__(self, optimizer, d_model: int, warmup_steps: int = 4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0
    
    def step(self):
        self.step_num += 1
        lr = self.d_model ** (-0.5) * min(
            self.step_num ** (-0.5),
            self.step_num * self.warmup_steps ** (-1.5)
        )
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

# 使用示例
optimizer = torch.optim.Adam(encoder.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
scheduler = TransformerLRScheduler(optimizer, d_model=512, warmup_steps=4000)

for epoch in range(num_epochs):
    for batch in dataloader:
        # 训练步骤
        loss = compute_loss(...)
        loss.backward()
        optimizer.step()
        
        # 更新学习率
        current_lr = scheduler.step()
```

### 7.2 Label Smoothing

防止模型过于自信：

```python
class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing 交叉熵损失
    
    公式:
        L_smooth = (1 - ε) * L_CE + ε * (1 / |V|)
    
    参数:
        epsilon: smoothing 系数（原始论文用 0.1）
        vocab_size: 词表大小
    """
    def __init__(self, epsilon: float = 0.1, vocab_size: int = 10000):
        super().__init__()
        self.epsilon = epsilon
        self.vocab_size = vocab_size
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, predictions, targets):
        # predictions: (batch * seq_len, vocab_size)
        # targets: (batch * seq_len)
        
        # 标准交叉熵损失
        ce_loss = self.ce_loss(predictions, targets)
        
        # smoothing 项：均匀分布的损失
        uniform_loss = -torch.log(torch.ones_like(predictions) / self.vocab_size).mean()
        
        # 加权组合
        loss = (1 - self.epsilon) * ce_loss + self.epsilon * uniform_loss
        
        return loss
```

### 7.3 梯度裁剪

防止梯度爆炸：

```python
# 在训练循环中
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## 8. 练习与思考题

### 8.1 数学推导练习

**练习 1**：证明 Multi-Head Attention 的参数量计算

假设 $d_{model} = 512$，$h = 8$，$d_k = d_v = 64$，计算：
1. $W_Q, W_K, W_V$ 的参数量
2. $W^O$ 的参数量
3. 总参数量

**练习 2**：推导位置编码的线性关系

证明对于任意位置偏移 $k$，存在矩阵 $M_k$ 使得：
$$
PE_{pos+k} = M_k \cdot PE_{pos}
$$

**练习 3**：分析注意力复杂度

证明 Self-Attention 的时间复杂度为 $O(n^2 \cdot d)$，空间复杂度为 $O(n^2 + n \cdot d)$。

### 8.2 代码实现练习

**练习 4**：实现 Causal Mask

```python
def generate_causal_mask(seq_len: int) -> torch.Tensor:
    """
    生成因果掩码（防止看到未来位置）
    
    返回:
        mask: (1, 1, seq_len, seq_len) 的布尔矩阵
              mask[i, j] = 1 表示可以关注，0 表示不能关注
    """
    # TODO: 实现上三角掩码
    pass
```

**练习 5**：实现 Cross-Attention

修改 `MultiHeadAttention` 类，支持 Encoder-Decoder 注意力：
- $Q$ 来自 Decoder
- $K, V$ 来自 Encoder 输出

### 8.3 思考题

**思考 1**：为什么 Transformer 使用 LayerNorm 而不是 BatchNorm？

**思考 2**：如果序列长度 $n$ 增加到 10 倍，注意力机制的计算量会增加多少？有什么优化方法？

**思考 3**：位置编码使用可学习参数（Learned Positional Encoding）与正弦编码各有什么优缺点？

---

## 附录：符号表

| 符号 | 含义 | 典型值 |
|-----|------|--------|
| $n$ | 序列长度 | 512 |
| $d_{model}$ | 模型维度 | 512 |
| $d_k$ | 查询/键维度 | 64 |
| $d_v$ | 值维度 | 64 |
| $d_{ff}$ | 前馈网络隐藏层维度 | 2048 |
| $h$ | 注意力头数 | 8 |
| $Q$ | 查询矩阵 | $\mathbb{R}^{n \times d_k}$ |
| $K$ | 键矩阵 | $\mathbb{R}^{n \times d_k}$ |
| $V$ | 值矩阵 | $\mathbb{R}^{n \times d_v}$ |
| $W_Q, W_K, W_V$ | 投影矩阵 | $\mathbb{R}^{d_{model} \times d_k}$ |
| $W^O$ | 输出投影矩阵 | $\mathbb{R}^{h \cdot d_v \times d_{model}}$ |

---

## 参考文献

1. Vaswani, A., et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). NeurIPS 2017.
2. [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Jay Alammar
3. [Attention Mechanism in Deep Learning](https://distill.pub/2016/augmented-rnns/) - Distill
4. [Stanford CS224N: Natural Language Processing](https://web.stanford.edu/class/cs224n/) - Lecture 10

---

最后更新：2026-03-11
