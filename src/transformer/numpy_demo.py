"""
Transformer: Mathematical Principles and Implementation (NumPy Version)

Pure NumPy implementation for mathematical verification, no PyTorch required.

Contains:
1. Scaled Dot-Product Attention (NumPy implementation)
2. Mathematical property verification
3. Visualization examples

Author: OpenClaw Engineer (AI + Mathematics Professor)
Date: 2026-03-11
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional


# ============================================================================
# 1. Scaled Dot-Product Attention (NumPy Implementation)
# ============================================================================

def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scaled dot-product attention - NumPy implementation
    
    Mathematical formula:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    
    Args:
        Q: Query matrix (seq_len, d_k)
        K: Key matrix (seq_len, d_k)
        V: Value matrix (seq_len, d_v)
        mask: Optional mask (seq_len, seq_len)
    
    Returns:
        output: Attention output (seq_len, d_v)
        attention_weights: Attention weights (seq_len, seq_len)
    """
    d_k = Q.shape[1]
    
    # 1. Compute attention scores: QK^T / sqrt(d_k)
    scores = np.matmul(Q, K.T) / np.sqrt(d_k)
    # scores shape: (seq_len, seq_len)
    
    # 2. Apply mask (if any)
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)
    
    # 3. Softmax normalization
    # Subtract max for numerical stability
    scores_max = np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attention_weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # attention_weights shape: (seq_len, seq_len)
    
    # 4. Weighted sum: Attention * V
    output = np.matmul(attention_weights, V)
    # output shape: (seq_len, d_v)
    
    return output, attention_weights


# ============================================================================
# 2. Mathematical Property Verification
# ============================================================================

def verify_scaling_factor():
    """Verify the effect of scaling factor"""
    print("=" * 60)
    print("Verifying scaling factor")
    print("=" * 60)
    
    d_k = 512
    
    # Generate random vectors
    np.random.seed(42)
    q = np.random.randn(d_k)
    k = np.random.randn(d_k)
    
    # Unscaled dot product
    dot_product_unscaled = np.dot(q, k)
    
    # Scaled dot product
    dot_product_scaled = np.dot(q, k) / np.sqrt(d_k)
    
    # Theoretical standard deviation
    theoretical_std = np.sqrt(d_k)
    
    print(f"d_k = {d_k}")
    print(f"Unscaled dot product: {dot_product_unscaled:.4f}")
    print(f"Scaled dot product: {dot_product_scaled:.4f}")
    print(f"Theoretical std: {theoretical_std:.4f}")
    print(f"Scaling factor: 1/{np.sqrt(d_k):.4f} = {1/np.sqrt(d_k):.4f}")
    print()
    
    # Simulate variance with multiple samples
    n_samples = 10000
    dot_products = []
    for _ in range(n_samples):
        q = np.random.randn(d_k)
        k = np.random.randn(d_k)
        dot_products.append(np.dot(q, k))
    
    dot_products = np.array(dot_products)
    print(f"Experimental variance (unscaled): {np.var(dot_products):.4f}")
    print(f"Theoretical variance (unscaled): {d_k:.4f}")
    print(f"Experimental std (unscaled): {np.std(dot_products):.4f}")
    print()


def verify_softmax_gradient():
    """Verify Softmax gradient issue"""
    print("=" * 60)
    print("Verifying Softmax gradient")
    print("=" * 60)
    
    d_k = 512
    np.random.seed(42)
    
    # Unscaled case
    z_unscaled = np.random.randn(d_k) * np.sqrt(d_k)
    
    # Scaled case
    z_scaled = z_unscaled / np.sqrt(d_k)
    
    # Softmax
    def softmax(z):
        z_max = np.max(z)
        exp_z = np.exp(z - z_max)
        return exp_z / np.sum(exp_z)
    
    p_unscaled = softmax(z_unscaled)
    p_scaled = softmax(z_scaled)
    
    # Entropy
    def entropy(p):
        return -np.sum(p * np.log(p + 1e-9))
    
    print(f"Unscaled max probability: {np.max(p_unscaled):.4f}")
    print(f"Scaled max probability: {np.max(p_scaled):.4f}")
    print(f"Unscaled entropy: {entropy(p_unscaled):.4f}")
    print(f"Scaled entropy: {entropy(p_scaled):.4f}")
    print(f"Maximum possible entropy (uniform): {np.log(d_k):.4f}")
    print()
    
    print("Conclusion: Scaled Softmax has more uniform distribution and larger gradients")
    print()


def visualize_attention():
    """Visualize attention weights"""
    print("=" * 60)
    print("Generating attention weight visualization")
    print("=" * 60)
    
    # Create example data
    np.random.seed(42)
    seq_len, d_k, d_v = 20, 64, 64
    
    Q = np.random.randn(seq_len, d_k)
    K = np.random.randn(seq_len, d_k)
    V = np.random.randn(seq_len, d_v)
    
    # Compute attention
    output, attention_weights = scaled_dot_product_attention(Q, K, V)
    
    # Visualization
    plt.figure(figsize=(10, 8))
    plt.imshow(attention_weights, cmap='YlOrRd', aspect='auto')
    plt.colorbar(label='Attention Weight')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.title('Self-Attention Weights Matrix')
    plt.savefig('attention_weights_viz.png', dpi=150, bbox_inches='tight')
    print("Attention weight visualization saved: attention_weights_viz.png")
    print()


def visualize_positional_encoding():
    """Visualize positional encoding"""
    print("=" * 60)
    print("Generating positional encoding visualization")
    print("=" * 60)
    
    seq_len, d_model = 100, 512
    pe = get_positional_encoding(seq_len, d_model)
    
    # Visualize first 16 dimensions
    plt.figure(figsize=(14, 8))
    for i in range(16):
        plt.plot(pe[:, i], label=f'dim {i}')
    
    plt.xlabel('Position')
    plt.ylabel('Encoding Value')
    plt.title('Positional Encoding (First 16 Dimensions)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.savefig('positional_encoding_viz.png', dpi=150, bbox_inches='tight')
    print("Positional encoding visualization saved: positional_encoding_viz.png")
    print()


def get_positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """
    Sinusoidal positional encoding - NumPy implementation
    
    Mathematical formula:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Args:
        seq_len: Sequence length
        d_model: Embedding dimension
    
    Returns:
        pe: Positional encoding matrix (seq_len, d_model)
    """
    pe = np.zeros((seq_len, d_model))
    
    position = np.arange(0, seq_len)[:, np.newaxis]  # (seq_len, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    
    return pe


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Transformer: Mathematical Principles (NumPy Verification)")
    print("=" * 60 + "\n")
    
    # Run all verifications
    verify_scaling_factor()
    verify_softmax_gradient()
    
    # Generate visualizations
    visualize_attention()
    visualize_positional_encoding()
    
    print("=" * 60)
    print("All verifications complete!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - attention_weights_viz.png")
    print("  - positional_encoding_viz.png")
