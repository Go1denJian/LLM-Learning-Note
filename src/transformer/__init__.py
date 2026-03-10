"""
Transformer: Mathematical Principles and Implementation

Implementation code containing:
1. Scaled Dot-Product Attention
2. Multi-Head Attention
3. Position-wise Feed-Forward Network
4. Positional Encoding
5. Encoder Layer & Encoder

Author: OpenClaw Engineer (AI + Mathematics Professor)
Date: 2026-03-11
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
    Scaled dot-product attention
    
    Mathematical formula:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    
    Args:
        d_k: Dimension of query/key
        dropout: Dropout probability
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
        mask: Optional[torch.Tensor] = None  # (batch, 1, 1, seq_len) or (batch, 1, seq_len, seq_len)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Returns:
            output: Attention output (batch, heads, seq_len, d_v)
            attention_weights: Attention weights (batch, heads, seq_len, seq_len)
        """
        # 1. Compute attention scores: QK^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        # scores shape: (batch, heads, seq_len, seq_len)
        
        # 2. Apply mask (if any)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 3. Softmax normalization
        attention_weights = F.softmax(scores, dim=-1)
        # attention_weights shape: (batch, heads, seq_len, seq_len)
        
        # 4. Dropout
        attention_weights = self.dropout(attention_weights)
        
        # 5. Weighted sum: Attention * V
        output = torch.matmul(attention_weights, V)
        # output shape: (batch, heads, seq_len, d_v)
        
        return output, attention_weights


# ============================================================================
# 2. Multi-Head Attention
# ============================================================================

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention
    
    Mathematical formula:
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
        head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    
    Args:
        d_model: Model dimension (input/output dimension)
        num_heads: Number of attention heads
        dropout: Dropout probability
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        
        # Linear projection layers
        self.W_Q = nn.Linear(d_model, d_model)  # Project to Q
        self.W_K = nn.Linear(d_model, d_model)  # Project to K
        self.W_V = nn.Linear(d_model, d_model)  # Project to V
        self.W_O = nn.Linear(d_model, d_model)  # Output projection
        
        # Attention mechanism
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
        Forward pass
        
        Returns:
            output: Multi-head attention output (batch, seq_len, d_model)
            attention_weights: Attention weights (batch, heads, seq_len, seq_len)
        """
        batch_size = Q.size(0)
        seq_len = Q.size(1)
        
        # 1. Linear projection and split into heads
        # Q, K, V shape: (batch, seq_len, d_model)
        Q_proj = self.W_Q(Q)  # (batch, seq_len, d_model)
        K_proj = self.W_K(K)  # (batch, seq_len, d_model)
        V_proj = self.W_V(V)  # (batch, seq_len, d_model)
        
        # Reshape to multi-head: (batch, num_heads, seq_len, d_k)
        Q_heads = Q_proj.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K_heads = K_proj.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V_heads = V_proj.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. Apply scaled dot-product attention
        attention_output, attention_weights = self.attention(Q_heads, K_heads, V_heads, mask)
        # attention_output shape: (batch, num_heads, seq_len, d_k)
        
        # 3. Concatenate heads
        concatenated = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        # concatenated shape: (batch, seq_len, d_model)
        
        # 4. Output projection
        output = self.W_O(concatenated)
        output = self.dropout(output)
        
        return output, attention_weights


# ============================================================================
# 3. Position-wise Feed-Forward Network
# ============================================================================

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise feed-forward network
    
    Mathematical formula:
        FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
               = ReLU(xW_1 + b_1)W_2 + b_2
    
    Args:
        d_model: Input/output dimension
        d_ff: Hidden layer dimension (typically 4x d_model)
        dropout: Dropout probability
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input (batch, seq_len, d_model)
        
        Returns:
            output: Output (batch, seq_len, d_model)
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
    Sinusoidal positional encoding
    
    Mathematical formula:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Args:
        d_model: Embedding dimension
        max_len: Maximum sequence length
        dropout: Dropout probability
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (no gradient update)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input embeddings (batch, seq_len, d_model)
        
        Returns:
            output: Embeddings with positional encoding (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ============================================================================
# 5. Encoder Layer
# ============================================================================

class EncoderLayer(nn.Module):
    """
    Encoder layer
    
    Structure:
        x -> [Multi-Head Attention] -> [Add & Norm] -> [Feed Forward] -> [Add & Norm] -> output
    
    Mathematical formula:
        x1 = LayerNorm(x + MultiHeadAttention(x, x, x))
        output = LayerNorm(x1 + FFN(x1))
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        dropout: Dropout probability
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
        Forward pass
        
        Args:
            x: Input (batch, seq_len, d_model)
            mask: Attention mask
        
        Returns:
            output: Output (batch, seq_len, d_model)
        """
        # 1. Self-attention + residual + normalization
        attn_output, _ = self.self_attn(x, x, x, mask)
        x1 = self.norm1(x + self.dropout(attn_output))
        
        # 2. Feed-forward + residual + normalization
        ff_output = self.feed_forward(x1)
        output = self.norm2(x1 + self.dropout(ff_output))
        
        return output


# ============================================================================
# 6. Complete Encoder
# ============================================================================

class Encoder(nn.Module):
    """
    Transformer Encoder
    
    Structure:
        [Embedding + Positional Encoding] -> [Encoder Layer x N] -> output
    
    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension (default 512)
        num_heads: Number of attention heads (default 8)
        num_layers: Number of encoder layers (default 6)
        d_ff: Feed-forward hidden dimension (default 2048)
        max_len: Maximum sequence length (default 5000)
        dropout: Dropout probability (default 0.1)
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
        Forward pass
        
        Args:
            x: Input token IDs (batch, seq_len)
            mask: Attention mask
        
        Returns:
            output: Encoder output (batch, seq_len, d_model)
        """
        # x shape: (batch, seq_len)
        x = self.embedding(x)  # (batch, seq_len, d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        return x  # (batch, seq_len, d_model)
