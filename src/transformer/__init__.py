"""
Transformer 数学原理与实现 —— 从线性代数到代码

这是配套代码实现，包含：
1. Scaled Dot-Product Attention
2. Multi-Head Attention
3. Position-wise Feed-Forward Network
4. Positional Encoding
5. Encoder Layer & Encoder

作者：OpenClaw Engineer (AI + Mathematics Professor)
日期：2026-03-11
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import math


# ============================================================================
# 1. Scaled Dot-Product Attention
# ============================================================================

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        返回:
            output: 注意力输出 (batch, heads, seq_len, d_v)
            attention_weights: 注意力权重 (batch, heads, seq_len, seq_len)
        """
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


# ============================================================================
# 2. Multi-Head Attention
# ============================================================================

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        返回:
            output: 多头注意力输出 (batch, seq_len, d_model)
            attention_weights: 注意力权重 (batch, heads, seq_len, seq_len)
        """
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


# ============================================================================
# 3. Position-wise Feed-Forward Network
# ============================================================================

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
        """
        前向传播
        
        参数:
            x: 输入 (batch, seq_len, d_model)
        
        返回:
            output: 输出 (batch, seq_len, d_model)
        """
        # x shape: (batch, seq_len, d_model)
        x = self.linear1(x)  # (batch, seq_len, d_ff)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)  # (batch, seq_len, d_model)
        return x


# ============================================================================
# 4. Positional Encoding
# ============================================================================

class PositionalEncoding(nn.Module):
    """
    正弦位置编码
    
    数学公式:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    参数:
        d_model: 嵌入维度
        max_len: 最大序列长度
        dropout: dropout 概率
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
        """
        前向传播
        
        参数:
            x: 输入嵌入 (batch, seq_len, d_model)
        
        返回:
            output: 加上位置编码的嵌入 (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ============================================================================
# 5. Encoder Layer
# ============================================================================

class EncoderLayer(nn.Module):
    """
    Encoder 层
    
    结构:
        x -> [Multi-Head Attention] -> [Add & Norm] -> [Feed Forward] -> [Add & Norm] -> output
    
    数学公式:
        x1 = LayerNorm(x + MultiHeadAttention(x, x, x))
        output = LayerNorm(x1 + FFN(x1))
    
    参数:
        d_model: 模型维度
        num_heads: 注意力头数
        d_ff: 前馈网络隐藏层维度
        dropout: dropout 概率
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 输入 (batch, seq_len, d_model)
            mask: 注意力掩码
        
        返回:
            output: 输出 (batch, seq_len, d_model)
        """
        # 1. 自注意力 + 残差 + 归一化
        attn_output, _ = self.self_attn(x, x, x, mask)
        x1 = self.norm1(x + self.dropout(attn_output))
        
        # 2. 前馈网络 + 残差 + 归一化
        ff_output = self.feed_forward(x1)
        output = self.norm2(x1 + self.dropout(ff_output))
        
        return output


# ============================================================================
# 6. Complete Encoder
# ============================================================================

class Encoder(nn.Module):
    """
    Transformer Encoder
    
    结构:
        [Embedding + Positional Encoding] -> [Encoder Layer x N] -> output
    
    参数:
        vocab_size: 词表大小
        d_model: 模型维度（默认 512）
        num_heads: 注意力头数（默认 8）
        num_layers: Encoder 层数（默认 6）
        d_ff: 前馈网络隐藏层维度（默认 2048）
        max_len: 最大序列长度（默认 5000）
        dropout: dropout 概率（默认 0.1）
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
        """
        前向传播
        
        参数:
            x: 输入 token IDs (batch, seq_len)
            mask: 注意力掩码
        
        返回:
            output: Encoder 输出 (batch, seq_len, d_model)
        """
        # x shape: (batch, seq_len)
        x = self.embedding(x)  # (batch, seq_len, d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        return x  # (batch, seq_len, d_model)


# ============================================================================
# 7. Utility Functions
# ============================================================================

def generate_causal_mask(seq_len: int) -> torch.Tensor:
    """
    生成因果掩码（防止看到未来位置）
    
    用于 Decoder 的自注意力，确保位置 i 只能关注位置 j <= i
    
    参数:
        seq_len: 序列长度
    
    返回:
        mask: (1, 1, seq_len, seq_len) 的布尔矩阵
              mask[i, j] = 1 表示可以关注，0 表示不能关注
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask.masked_fill(mask == 1, 0)
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)


def generate_padding_mask(pad_idx: int, x: torch.Tensor) -> torch.Tensor:
    """
    生成填充掩码
    
    参数:
        pad_idx: 填充 token 的 ID
        x: 输入 (batch, seq_len)
    
    返回:
        mask: (batch, 1, 1, seq_len) 的布尔矩阵
              mask[i, j] = 0 表示位置 j 是填充，需要被掩码
    """
    mask = (x != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask


# ============================================================================
# 8. Test Functions
# ============================================================================

def test_attention():
    """测试注意力机制"""
    print("=" * 60)
    print("测试 Scaled Dot-Product Attention")
    print("=" * 60)
    
    batch_size, seq_len, d_k, d_v = 2, 10, 64, 64
    
    Q = torch.randn(batch_size, seq_len, d_k)
    K = torch.randn(batch_size, seq_len, d_k)
    V = torch.randn(batch_size, seq_len, d_v)
    
    attention = ScaledDotProductAttention(d_k)
    output, weights = attention(Q, K, V)
    
    print(f"输入 Q 形状：{Q.shape}")
    print(f"输入 K 形状：{K.shape}")
    print(f"输入 V 形状：{V.shape}")
    print(f"输出形状：{output.shape}")
    print(f"注意力权重形状：{weights.shape}")
    print(f"注意力权重和（每行）：{weights.sum(dim=-1)}")
    print()


def test_multihead_attention():
    """测试多头注意力"""
    print("=" * 60)
    print("测试 Multi-Head Attention")
    print("=" * 60)
    
    batch_size, seq_len, d_model, num_heads = 2, 10, 512, 8
    
    Q = torch.randn(batch_size, seq_len, d_model)
    K = torch.randn(batch_size, seq_len, d_model)
    V = torch.randn(batch_size, seq_len, d_model)
    
    attention = MultiHeadAttention(d_model, num_heads)
    output, weights = attention(Q, K, V)
    
    print(f"输入形状：{Q.shape}")
    print(f"输出形状：{output.shape}")
    print(f"注意力头数：{num_heads}")
    print(f"每个头维度：{d_model // num_heads}")
    print(f"参数量：{sum(p.numel() for p in attention.parameters()):,}")
    print()


def test_positional_encoding():
    """测试位置编码"""
    print("=" * 60)
    print("测试 Positional Encoding")
    print("=" * 60)
    
    d_model, max_len = 512, 100
    pe = PositionalEncoding(d_model, max_len)
    
    # 创建假嵌入
    batch_size, seq_len = 2, 20
    x = torch.randn(batch_size, seq_len, d_model)
    
    output = pe(x)
    
    print(f"输入嵌入形状：{x.shape}")
    print(f"输出形状：{output.shape}")
    print(f"位置编码形状：{pe.pe.shape}")
    
    # 可视化前 8 个维度
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 6))
    for i in range(8):
        plt.plot(pe.pe[0, :50, i].numpy(), label=f'dim {i}')
    plt.xlabel('Position')
    plt.ylabel('Encoding Value')
    plt.title('Positional Encoding (First 8 Dimensions)')
    plt.legend()
    plt.savefig('positional_encoding_viz.png', dpi=150)
    print("位置编码可视化已保存：positional_encoding_viz.png")
    print()


def test_encoder():
    """测试完整 Encoder"""
    print("=" * 60)
    print("测试 Transformer Encoder")
    print("=" * 60)
    
    vocab_size = 10000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    batch_size = 32
    seq_len = 100
    
    encoder = Encoder(vocab_size, d_model, num_heads, num_layers, d_ff)
    
    # 创建假数据
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # 前向传播
    output = encoder(x)
    
    print(f"词表大小：{vocab_size}")
    print(f"输入形状：{x.shape}")
    print(f"输出形状：{output.shape}")
    print(f"参数量：{sum(p.numel() for p in encoder.parameters()):,}")
    
    # 参数量分解
    print("\n参数量分解:")
    print(f"  Embedding: {sum(p.numel() for p in encoder.embedding.parameters()):,}")
    print(f"  Encoder Layers: {sum(p.numel() for p in encoder.layers.parameters()):,}")
    print()


def test_mathematical_properties():
    """测试数学性质"""
    print("=" * 60)
    print("测试数学性质")
    print("=" * 60)
    
    # 1. 测试缩放因子的作用
    print("1. 缩放因子对 Softmax 的影响:")
    d_k = 512
    z_unscaled = torch.randn(d_k) * math.sqrt(d_k)  # 未缩放
    z_scaled = z_unscaled / math.sqrt(d_k)  # 缩放后
    
    p_unscaled = F.softmax(z_unscaled, dim=0)
    p_scaled = F.softmax(z_scaled, dim=0)
    
    print(f"   未缩放最大概率：{p_unscaled.max().item():.4f}")
    print(f"   缩放后最大概率：{p_scaled.max().item():.4f}")
    print(f"   未缩放熵：{-(p_unscaled * torch.log(p_unscaled + 1e-9)).sum().item():.4f}")
    print(f"   缩放后熵：{-(p_scaled * torch.log(p_scaled + 1e-9)).sum().item():.4f}")
    print()
    
    # 2. 测试多头注意力的子空间
    print("2. 多头注意力的子空间投影:")
    d_model, num_heads = 512, 8
    d_k = d_model // num_heads
    
    W_Q = nn.Linear(d_model, d_model)
    print(f"   d_model = {d_model}")
    print(f"   num_heads = {num_heads}")
    print(f"   每个头维度 d_k = {d_k}")
    print(f"   投影矩阵 W_Q 形状：{W_Q.weight.shape}")
    print()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Transformer 数学原理与实现 —— 测试套件")
    print("=" * 60 + "\n")
    
    # 运行所有测试
    test_attention()
    test_multihead_attention()
    test_positional_encoding()
    test_encoder()
    test_mathematical_properties()
    
    print("=" * 60)
    print("所有测试完成！")
    print("=" * 60)
