#!/usr/bin/env python3
"""
Transformer Component Tests

Tests attention mechanism, multi-head attention, positional encoding, Encoder, etc.

Usage:
    python examples/test_transformer.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

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
    """Test scaled dot-product attention"""
    print("=" * 60)
    print("Test 1: Scaled Dot-Product Attention")
    print("=" * 60)
    
    batch_size, seq_len, d_k, d_v = 2, 10, 64, 64
    
    Q = torch.randn(batch_size, seq_len, d_k)
    K = torch.randn(batch_size, seq_len, d_k)
    V = torch.randn(batch_size, seq_len, d_v)
    
    attention = ScaledDotProductAttention(d_k)
    output, weights = attention(Q, K, V)
    
    assert output.shape == (batch_size, seq_len, d_v), "Output shape incorrect"
    assert weights.shape == (batch_size, seq_len, seq_len), "Attention weight shape incorrect"
    
    # Verify weights sum to 1
    weight_sums = weights.sum(dim=-1)
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5), "Weights do not sum to 1"
    
    print(f"✓ Input Q shape: {Q.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Attention weight shape: {weights.shape}")
    print(f"✓ Weight sums: {weight_sums[0, :5]}")
    print()


def test_multi_head_attention():
    """Test multi-head attention"""
    print("=" * 60)
    print("Test 2: Multi-Head Attention")
    print("=" * 60)
    
    batch_size, seq_len, d_model, num_heads = 2, 10, 512, 8
    
    Q = torch.randn(batch_size, seq_len, d_model)
    K = torch.randn(batch_size, seq_len, d_model)
    V = torch.randn(batch_size, seq_len, d_model)
    
    attention = MultiHeadAttention(d_model, num_heads)
    output, weights = attention(Q, K, V)
    
    assert output.shape == (batch_size, seq_len, d_model), "Output shape incorrect"
    
    print(f"✓ Input shape: {Q.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Number of heads: {num_heads}")
    print(f"✓ Dimension per head: {d_model // num_heads}")
    print(f"✓ Parameters: {sum(p.numel() for p in attention.parameters()):,}")
    print()


def test_positional_encoding():
    """Test positional encoding"""
    print("=" * 60)
    print("Test 3: Positional Encoding")
    print("=" * 60)
    
    d_model, max_len = 512, 100
    pe = PositionalEncoding(d_model, max_len)
    
    batch_size, seq_len = 2, 20
    x = torch.randn(batch_size, seq_len, d_model)
    
    output = pe(x)
    
    assert output.shape == x.shape, "Output shape incorrect"
    
    # Verify positional encoding shape
    assert pe.pe.shape == (1, max_len, d_model), "Positional encoding shape incorrect"
    
    print(f"✓ Input embedding shape: {x.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Positional encoding shape: {pe.pe.shape}")
    print(f"✓ Positional encoding value range: [{pe.pe.min():.4f}, {pe.pe.max():.4f}]")
    print()


def test_feed_forward():
    """Test feed-forward network"""
    print("=" * 60)
    print("Test 4: Position-wise Feed-Forward Network")
    print("=" * 60)
    
    batch_size, seq_len, d_model, d_ff = 2, 10, 512, 2048
    
    x = torch.randn(batch_size, seq_len, d_model)
    ffn = PositionwiseFeedForward(d_model, d_ff)
    
    output = ffn(x)
    
    assert output.shape == x.shape, "Output shape incorrect"
    
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Hidden dimension: {d_ff}")
    print(f"✓ Parameters: {sum(p.numel() for p in ffn.parameters()):,}")
    print()


def test_encoder_layer():
    """Test encoder layer"""
    print("=" * 60)
    print("Test 5: Encoder Layer")
    print("=" * 60)
    
    batch_size, seq_len, d_model = 2, 10, 512
    num_heads, d_ff = 8, 2048
    
    x = torch.randn(batch_size, seq_len, d_model)
    layer = EncoderLayer(d_model, num_heads, d_ff)
    
    output = layer(x)
    
    assert output.shape == x.shape, "Output shape incorrect"
    
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Parameters: {sum(p.numel() for p in layer.parameters()):,}")
    print()


def test_encoder():
    """Test complete encoder"""
    print("=" * 60)
    print("Test 6: Complete Encoder")
    print("=" * 60)
    
    vocab_size = 10000
    d_model, num_heads, num_layers = 512, 8, 6
    d_ff = 2048
    batch_size, seq_len = 32, 100
    
    encoder = Encoder(vocab_size, d_model, num_heads, num_layers, d_ff)
    
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    output = encoder(x)
    
    assert output.shape == (batch_size, seq_len, d_model), "Output shape incorrect"
    
    print(f"✓ Vocabulary size: {vocab_size}")
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    
    # Parameter breakdown
    embedding_params = sum(p.numel() for p in encoder.embedding.parameters())
    layer_params = sum(p.numel() for p in encoder.layers.parameters())
    print(f"✓ Embedding parameters: {embedding_params:,}")
    print(f"✓ Encoder Layers parameters: {layer_params:,}")
    print()


def test_masking():
    """Test masking functionality"""
    print("=" * 60)
    print("Test 7: Masking")
    print("=" * 60)
    
    batch_size, seq_len, d_k = 2, 10, 64
    
    Q = torch.randn(batch_size, seq_len, d_k)
    K = torch.randn(batch_size, seq_len, d_k)
    V = torch.randn(batch_size, seq_len, d_k)
    
    # Create causal mask (upper triangular)
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask.masked_fill(mask == 1, 0)
    mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
    
    attention = ScaledDotProductAttention(d_k)
    output, weights = attention(Q, K, V, mask=mask)
    
    # Verify masked attention weights (upper triangular should be 0)
    masked_sum = (weights * (1 - mask)).sum()
    total_sum = weights.sum()
    
    print(f"✓ Mask shape: {mask.shape}")
    print(f"✓ Valid attention weight sum: {masked_sum.item():.4f}")
    print(f"✓ Total weight sum: {total_sum.item():.4f}")
    print(f"✓ Mask有效性：{torch.allclose(masked_sum, total_sum, atol=1e-5)}")
    print()


def main():
    print("\n" + "=" * 60)
    print("Transformer Component Test Suite")
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
            print(f"✗ Test failed: {test.__name__}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            print()
            failed += 1
    
    print("=" * 60)
    print(f"Test results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
