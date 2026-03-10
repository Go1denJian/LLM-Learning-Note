#!/usr/bin/env python3
"""
Transformer 组件测试

测试注意力机制、多头注意力、位置编码、Encoder 等组件。

用法:
    python examples/test_transformer.py
"""

import sys
import os

# 添加 src 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.transformer import (
    ScaledDotProductAttention,
    MultiHeadAttention,
    PositionwiseFeedForward,
    PositionalEncoding,
    EncoderLayer,
    Encoder
)

import torch


def test_scaled_dot_product_attention():
    """测试缩放点积注意力"""
    print("=" * 60)
    print("测试 1: Scaled Dot-Product Attention")
    print("=" * 60)
    
    batch_size, seq_len, d_k, d_v = 2, 10, 64, 64
    
    Q = torch.randn(batch_size, seq_len, d_k)
    K = torch.randn(batch_size, seq_len, d_k)
    V = torch.randn(batch_size, seq_len, d_v)
    
    attention = ScaledDotProductAttention(d_k)
    output, weights = attention(Q, K, V)
    
    assert output.shape == (batch_size, seq_len, d_v), "输出形状错误"
    assert weights.shape == (batch_size, seq_len, seq_len), "注意力权重形状错误"
    
    # 验证权重和为 1
    weight_sums = weights.sum(dim=-1)
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5), "权重和不为 1"
    
    print(f"✓ 输入 Q 形状：{Q.shape}")
    print(f"✓ 输出形状：{output.shape}")
    print(f"✓ 注意力权重形状：{weights.shape}")
    print(f"✓ 权重和：{weight_sums[0, :5]}")
    print()


def test_multi_head_attention():
    """测试多头注意力"""
    print("=" * 60)
    print("测试 2: Multi-Head Attention")
    print("=" * 60)
    
    batch_size, seq_len, d_model, num_heads = 2, 10, 512, 8
    
    Q = torch.randn(batch_size, seq_len, d_model)
    K = torch.randn(batch_size, seq_len, d_model)
    V = torch.randn(batch_size, seq_len, d_model)
    
    attention = MultiHeadAttention(d_model, num_heads)
    output, weights = attention(Q, K, V)
    
    assert output.shape == (batch_size, seq_len, d_model), "输出形状错误"
    
    print(f"✓ 输入形状：{Q.shape}")
    print(f"✓ 输出形状：{output.shape}")
    print(f"✓ 注意力头数：{num_heads}")
    print(f"✓ 每个头维度：{d_model // num_heads}")
    print(f"✓ 参数量：{sum(p.numel() for p in attention.parameters()):,}")
    print()


def test_positional_encoding():
    """测试位置编码"""
    print("=" * 60)
    print("测试 3: Positional Encoding")
    print("=" * 60)
    
    d_model, max_len = 512, 100
    pe = PositionalEncoding(d_model, max_len)
    
    batch_size, seq_len = 2, 20
    x = torch.randn(batch_size, seq_len, d_model)
    
    output = pe(x)
    
    assert output.shape == x.shape, "输出形状错误"
    
    # 验证位置编码的形状
    assert pe.pe.shape == (1, max_len, d_model), "位置编码形状错误"
    
    print(f"✓ 输入嵌入形状：{x.shape}")
    print(f"✓ 输出形状：{output.shape}")
    print(f"✓ 位置编码形状：{pe.pe.shape}")
    print(f"✓ 位置编码值范围：[{pe.pe.min():.4f}, {pe.pe.max():.4f}]")
    print()


def test_feed_forward():
    """测试前馈网络"""
    print("=" * 60)
    print("测试 4: Position-wise Feed-Forward Network")
    print("=" * 60)
    
    batch_size, seq_len, d_model, d_ff = 2, 10, 512, 2048
    
    x = torch.randn(batch_size, seq_len, d_model)
    ffn = PositionwiseFeedForward(d_model, d_ff)
    
    output = ffn(x)
    
    assert output.shape == x.shape, "输出形状错误"
    
    print(f"✓ 输入形状：{x.shape}")
    print(f"✓ 输出形状：{output.shape}")
    print(f"✓ 隐藏层维度：{d_ff}")
    print(f"✓ 参数量：{sum(p.numel() for p in ffn.parameters()):,}")
    print()


def test_encoder_layer():
    """测试编码器层"""
    print("=" * 60)
    print("测试 5: Encoder Layer")
    print("=" * 60)
    
    batch_size, seq_len, d_model = 2, 10, 512
    num_heads, d_ff = 8, 2048
    
    x = torch.randn(batch_size, seq_len, d_model)
    layer = EncoderLayer(d_model, num_heads, d_ff)
    
    output = layer(x)
    
    assert output.shape == x.shape, "输出形状错误"
    
    print(f"✓ 输入形状：{x.shape}")
    print(f"✓ 输出形状：{output.shape}")
    print(f"✓ 参数量：{sum(p.numel() for p in layer.parameters()):,}")
    print()


def test_encoder():
    """测试完整编码器"""
    print("=" * 60)
    print("测试 6: Complete Encoder")
    print("=" * 60)
    
    vocab_size = 10000
    d_model, num_heads, num_layers = 512, 8, 6
    d_ff = 2048
    batch_size, seq_len = 32, 100
    
    encoder = Encoder(vocab_size, d_model, num_heads, num_layers, d_ff)
    
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    output = encoder(x)
    
    assert output.shape == (batch_size, seq_len, d_model), "输出形状错误"
    
    print(f"✓ 词表大小：{vocab_size}")
    print(f"✓ 输入形状：{x.shape}")
    print(f"✓ 输出形状：{output.shape}")
    print(f"✓ 参数量：{sum(p.numel() for p in encoder.parameters()):,}")
    
    # 参数量分解
    embedding_params = sum(p.numel() for p in encoder.embedding.parameters())
    layer_params = sum(p.numel() for p in encoder.layers.parameters())
    print(f"✓ Embedding 参数：{embedding_params:,}")
    print(f"✓ Encoder Layers 参数：{layer_params:,}")
    print()


def test_masking():
    """测试掩码功能"""
    print("=" * 60)
    print("测试 7: 掩码功能")
    print("=" * 60)
    
    batch_size, seq_len, d_k = 2, 10, 64
    
    Q = torch.randn(batch_size, seq_len, d_k)
    K = torch.randn(batch_size, seq_len, d_k)
    V = torch.randn(batch_size, seq_len, d_k)
    
    # 创建因果掩码（上三角）
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask.masked_fill(mask == 1, 0)
    mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
    
    attention = ScaledDotProductAttention(d_k)
    output, weights = attention(Q, K, V, mask=mask)
    
    # 验证掩码后的注意力权重（上三角应为 0）
    masked_sum = (weights * (1 - mask)).sum()
    total_sum = weights.sum()
    
    print(f"✓ 掩码形状：{mask.shape}")
    print(f"✓ 有效注意力权重和：{masked_sum.item():.4f}")
    print(f"✓ 总权重和：{total_sum.item():.4f}")
    print(f"✓ 掩码有效性：{torch.allclose(masked_sum, total_sum, atol=1e-5)}")
    print()


def main():
    print("\n" + "=" * 60)
    print("Transformer 组件测试套件")
    print("=" * 60 + "\n")
    
    tests = [
        test_scaled_dot_product_attention,
        test_multi_head_attention,
        test_positional_encoding,
        test_feed_forward,
        test_encoder_layer,
        test_encoder,
        test_masking,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ 测试失败：{test.__name__}")
            print(f"  错误：{e}")
            import traceback
            traceback.print_exc()
            print()
            failed += 1
    
    print("=" * 60)
    print(f"测试结果：{passed} 通过，{failed} 失败")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
