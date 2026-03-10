"""
Transformer 数学原理与实现 —— 纯 NumPy 版本

这是配套代码的纯 NumPy 实现，无需 PyTorch，可直接运行验证数学原理。

包含：
1. Scaled Dot-Product Attention (NumPy 实现)
2. 数学性质验证
3. 可视化示例

作者：OpenClaw Engineer (AI + Mathematics Professor)
日期：2026-03-11
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional


# ============================================================================
# 1. Scaled Dot-Product Attention (NumPy 实现)
# ============================================================================

def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    缩放点积注意力 - NumPy 实现
    
    数学公式:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    
    参数:
        Q: 查询矩阵 (seq_len, d_k)
        K: 键矩阵 (seq_len, d_k)
        V: 值矩阵 (seq_len, d_v)
        mask: 可选掩码 (seq_len, seq_len)
    
    返回:
        output: 注意力输出 (seq_len, d_v)
        attention_weights: 注意力权重 (seq_len, seq_len)
    """
    d_k = Q.shape[1]
    
    # 1. 计算注意力分数：QK^T / sqrt(d_k)
    scores = np.matmul(Q, K.T) / np.sqrt(d_k)
    # scores shape: (seq_len, seq_len)
    
    # 2. 应用掩码（如果有）
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)
    
    # 3. Softmax 归一化
    # 减去最大值防止数值溢出
    scores_max = np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attention_weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # attention_weights shape: (seq_len, seq_len)
    
    # 4. 加权求和：Attention * V
    output = np.matmul(attention_weights, V)
    # output shape: (seq_len, d_v)
    
    return output, attention_weights


def multi_head_attention(
    X: np.ndarray,
    W_Q: np.ndarray,
    W_K: np.ndarray,
    W_V: np.ndarray,
    W_O: np.ndarray,
    num_heads: int
) -> np.ndarray:
    """
    多头注意力 - NumPy 实现
    
    参数:
        X: 输入 (seq_len, d_model)
        W_Q, W_K, W_V: 投影矩阵 (d_model, d_model)
        W_O: 输出投影矩阵 (d_model, d_model)
        num_heads: 头数
    
    返回:
        output: 多头注意力输出 (seq_len, d_model)
    """
    seq_len, d_model = X.shape
    d_k = d_model // num_heads
    
    # 1. 线性投影
    Q_all = X @ W_Q.T  # (seq_len, d_model)
    K_all = X @ W_K.T  # (seq_len, d_model)
    V_all = X @ W_V.T  # (seq_len, d_model)
    
    # 2. 分割成多头
    Q_heads = Q_all.reshape(seq_len, num_heads, d_k).transpose(1, 0, 2)  # (heads, seq_len, d_k)
    K_heads = K_all.reshape(seq_len, num_heads, d_k).transpose(1, 0, 2)
    V_heads = V_all.reshape(seq_len, num_heads, d_k).transpose(1, 0, 2)
    
    # 3. 每个头独立计算注意力
    head_outputs = []
    for i in range(num_heads):
        Q_head = Q_heads[i]  # (seq_len, d_k)
        K_head = K_heads[i]
        V_head = V_heads[i]
        
        output, _ = scaled_dot_product_attention(Q_head, K_head, V_head)
        head_outputs.append(output)  # (seq_len, d_k)
    
    # 4. 拼接多头
    concatenated = np.concatenate(head_outputs, axis=1)  # (seq_len, d_model)
    
    # 5. 输出投影
    output = concatenated @ W_O.T  # (seq_len, d_model)
    
    return output


# ============================================================================
# 2. Positional Encoding
# ============================================================================

def get_positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """
    正弦位置编码 - NumPy 实现
    
    数学公式:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    参数:
        seq_len: 序列长度
        d_model: 嵌入维度
    
    返回:
        pe: 位置编码矩阵 (seq_len, d_model)
    """
    pe = np.zeros((seq_len, d_model))
    
    position = np.arange(0, seq_len)[:, np.newaxis]  # (seq_len, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    
    return pe


# ============================================================================
# 3. 数学性质验证
# ============================================================================

def verify_scaling_factor():
    """验证缩放因子的作用"""
    print("=" * 60)
    print("验证缩放因子的作用")
    print("=" * 60)
    
    d_k = 512
    
    # 生成随机向量
    np.random.seed(42)
    q = np.random.randn(d_k)
    k = np.random.randn(d_k)
    
    # 未缩放的点积
    dot_product_unscaled = np.dot(q, k)
    
    # 缩放后的点积
    dot_product_scaled = np.dot(q, k) / np.sqrt(d_k)
    
    # 理论标准差
    theoretical_std = np.sqrt(d_k)
    
    print(f"d_k = {d_k}")
    print(f"未缩放点积：{dot_product_unscaled:.4f}")
    print(f"缩放后点积：{dot_product_scaled:.4f}")
    print(f"理论标准差：{theoretical_std:.4f}")
    print(f"缩放因子：1/{np.sqrt(d_k):.4f} = {1/np.sqrt(d_k):.4f}")
    print()
    
    # 模拟多个样本的方差
    n_samples = 10000
    dot_products = []
    for _ in range(n_samples):
        q = np.random.randn(d_k)
        k = np.random.randn(d_k)
        dot_products.append(np.dot(q, k))
    
    dot_products = np.array(dot_products)
    print(f"实验方差（未缩放）：{np.var(dot_products):.4f}")
    print(f"理论方差（未缩放）：{d_k:.4f}")
    print(f"实验标准差（未缩放）：{np.std(dot_products):.4f}")
    print()


def verify_softmax_gradient():
    """验证 Softmax 梯度问题"""
    print("=" * 60)
    print("验证 Softmax 梯度问题")
    print("=" * 60)
    
    d_k = 512
    np.random.seed(42)
    
    # 未缩放的情况
    z_unscaled = np.random.randn(d_k) * np.sqrt(d_k)
    
    # 缩放后的情况
    z_scaled = z_unscaled / np.sqrt(d_k)
    
    # Softmax
    def softmax(z):
        z_max = np.max(z)
        exp_z = np.exp(z - z_max)
        return exp_z / np.sum(exp_z)
    
    p_unscaled = softmax(z_unscaled)
    p_scaled = softmax(z_scaled)
    
    # 熵
    def entropy(p):
        return -np.sum(p * np.log(p + 1e-9))
    
    print(f"未缩放最大概率：{np.max(p_unscaled):.4f}")
    print(f"缩放后最大概率：{np.max(p_scaled):.4f}")
    print(f"未缩放熵：{entropy(p_unscaled):.4f}")
    print(f"缩放后熵：{entropy(p_scaled):.4f}")
    print(f"最大可能熵（均匀分布）：{np.log(d_k):.4f}")
    print()
    
    print("结论：缩放后的 Softmax 分布更均匀，梯度更大，有利于训练")
    print()


def verify_positional_encoding_linearity():
    """验证位置编码的线性关系"""
    print("=" * 60)
    print("验证位置编码的线性关系")
    print("=" * 60)
    
    seq_len, d_model = 100, 512
    pe = get_positional_encoding(seq_len, d_model)
    
    # 验证 PE[pos+k] 可以表示为 PE[pos] 的线性组合
    pos, k = 10, 5
    
    # 对于每个维度 i，验证 sin((pos+k)/factor) 与 sin(pos/factor), cos(pos/factor) 的关系
    i = 0  # 第一个维度
    factor = 10000 ** (2 * i / d_model)
    
    sin_pos = np.sin(pos / factor)
    cos_pos = np.cos(pos / factor)
    sin_pos_k = np.sin((pos + k) / factor)
    cos_pos_k = np.cos((pos + k) / factor)
    
    # 使用和角公式验证
    sin_k = np.sin(k / factor)
    cos_k = np.cos(k / factor)
    
    sin_pos_k_formula = sin_pos * cos_k + cos_pos * sin_k
    cos_pos_k_formula = cos_pos * cos_k - sin_pos * sin_k
    
    print(f"位置 {pos}, 偏移 {k}, 维度 {i}")
    print(f"sin((pos+k)/factor) = {sin_pos_k:.6f}")
    print(f"sin(pos/factor)*cos(k/factor) + cos(pos/factor)*sin(k/factor) = {sin_pos_k_formula:.6f}")
    print(f"误差：{abs(sin_pos_k - sin_pos_k_formula):.10f}")
    print()
    
    print("结论：位置编码满足和角公式，PE[pos+k] 可表示为 PE[pos] 的线性组合")
    print("这使得模型可以学习相对位置关系")
    print()


def visualize_attention():
    """可视化注意力权重"""
    print("=" * 60)
    print("生成注意力权重可视化")
    print("=" * 60)
    
    # 创建示例数据
    np.random.seed(42)
    seq_len, d_k, d_v = 20, 64, 64
    
    Q = np.random.randn(seq_len, d_k)
    K = np.random.randn(seq_len, d_k)
    V = np.random.randn(seq_len, d_v)
    
    # 计算注意力
    output, attention_weights = scaled_dot_product_attention(Q, K, V)
    
    # 可视化
    plt.figure(figsize=(10, 8))
    plt.imshow(attention_weights, cmap='YlOrRd', aspect='auto')
    plt.colorbar(label='Attention Weight')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.title('Self-Attention Weights Matrix')
    plt.savefig('attention_weights_viz.png', dpi=150, bbox_inches='tight')
    print("注意力权重可视化已保存：attention_weights_viz.png")
    print()


def visualize_positional_encoding():
    """可视化位置编码"""
    print("=" * 60)
    print("生成位置编码可视化")
    print("=" * 60)
    
    seq_len, d_model = 100, 512
    pe = get_positional_encoding(seq_len, d_model)
    
    # 可视化前 16 个维度
    plt.figure(figsize=(14, 8))
    for i in range(16):
        plt.plot(pe[:, i], label=f'dim {i}')
    
    plt.xlabel('Position')
    plt.ylabel('Encoding Value')
    plt.title('Positional Encoding (First 16 Dimensions)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.savefig('positional_encoding_viz.png', dpi=150, bbox_inches='tight')
    print("位置编码可视化已保存：positional_encoding_viz.png")
    print()


def compute_parameter_count():
    """计算 Transformer 参数量"""
    print("=" * 60)
    print("Transformer 参数量计算")
    print("=" * 60)
    
    # 原始论文配置
    d_model = 512
    num_heads = 8
    d_k = d_model // num_heads  # 64
    d_v = d_k  # 64
    d_ff = 2048
    vocab_size = 30000
    num_layers = 6
    
    # Embedding
    embedding_params = vocab_size * d_model
    
    # 每个 Encoder 层的参数
    # Multi-Head Attention
    W_QKV_params = 3 * d_model * d_model  # W_Q, W_K, W_V
    W_O_params = d_model * d_model  # 输出投影
    attn_params = W_QKV_params + W_O_params
    
    # Feed-Forward
    ff_params = d_model * d_ff + d_ff * d_model  # W_1, W_2
    
    # LayerNorm (2 个)
    norm_params = 2 * 2 * d_model  # weight + bias
    
    # 每层总参数
    layer_params = attn_params + ff_params + norm_params
    
    # Encoder 总参数
    encoder_params = embedding_params + num_layers * layer_params
    
    print(f"配置:")
    print(f"  d_model = {d_model}")
    print(f"  num_heads = {num_heads}")
    print(f"  d_k = d_v = {d_k}")
    print(f"  d_ff = {d_ff}")
    print(f"  vocab_size = {vocab_size}")
    print(f"  num_layers = {num_layers}")
    print()
    
    print(f"参数量分解:")
    print(f"  Embedding: {embedding_params:,}")
    print(f"  Multi-Head Attention (每层): {attn_params:,}")
    print(f"    - W_Q, W_K, W_V: {W_QKV_params:,}")
    print(f"    - W_O: {W_O_params:,}")
    print(f"  Feed-Forward (每层): {ff_params:,}")
    print(f"  LayerNorm (每层): {norm_params:,}")
    print()
    print(f"  每层总参数：{layer_params:,}")
    print(f"  Encoder 总参数：{encoder_params:,}")
    print(f"  ≈ {encoder_params / 1e6:.2f}M")
    print()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Transformer 数学原理与实现 —— 纯 NumPy 验证")
    print("=" * 60 + "\n")
    
    # 运行所有验证
    verify_scaling_factor()
    verify_softmax_gradient()
    verify_positional_encoding_linearity()
    compute_parameter_count()
    
    # 生成可视化
    visualize_attention()
    visualize_positional_encoding()
    
    print("=" * 60)
    print("所有验证完成！")
    print("=" * 60)
    print("\n生成的文件:")
    print("  - attention_weights_viz.png")
    print("  - positional_encoding_viz.png")
    print("\n下一步:")
    print("  1. 阅读 Transformer-Math-and-Implementation.md")
    print("  2. 安装 PyTorch 运行 transformer_implementation.py")
    print("  3. 完成练习与思考题")
