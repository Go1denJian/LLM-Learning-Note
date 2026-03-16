# GRU 数学原理与 Seq2Seq 架构 —— 从简化门控到注意力机制

> **前置知识**：RNN、LSTM、门控机制、Python 基础  
> **与前面内容的联系**：建议先学习 [RNN-Fundamentals](./RNN-Fundamentals.md) 和 [LSTM-Deep-Dive](./LSTM-Deep-Dive.md)，理解递归网络和门控机制  
> **与后续内容的联系**：GRU 是 LSTM 的高效替代，Seq2Seq 是 Transformer 的雏形，注意力机制是 Transformer 的核心

---

## 目录

1. [GRU：LSTM 的精简版](#1-grulstm-的精简版)
   - 1.1 [为什么需要 GRU？](#11-为什么需要-gru)
   - 1.2 [GRU 的数学表达](#12-gru-的数学表达)
2. [为什么 GRU 能保持 LSTM 的能力](#2-为什么-gru-能保持-lstm-的能力)
3. [Seq2Seq 架构](#3-seq2seq-架构)
   - 3.1 [Encoder-Decoder 结构](#31-encoder-decoder-结构)
   - 3.2 [Teacher Forcing](#32-teacher-forcing)
4. [注意力机制初步](#4-注意力机制初步)
   - 4.1 [Seq2Seq 的局限性](#41-seq2seq-的局限性)
   - 4.2 [注意力机制的直觉](#42-注意力机制的直觉)
5. [从数学到代码：完整实现](#5-从数学到代码完整实现)
   - 5.1 [GRU NumPy 实现](#51-gru-numpy-实现)
   - 5.2 [Seq2Seq PyTorch 实现](#52-seq2seq-pytorch-实现)
6. [练习与思考题](#6-练习与思考题)

---

## 1. GRU：LSTM 的精简版

### 1.1 为什么需要 GRU？

**LSTM 的问题：**
- 参数量大（约 4 倍于 RNN）
- 计算复杂度高
- 训练时间长

**GRU 的设计目标：**
- 保持 LSTM 捕捉长期依赖的能力
- 减少参数量
- 简化计算

### 1.2 GRU 架构

GRU 将 LSTM 的 4 个门合并为 2 个门：

```
        ┌─────────────────────────────────┐
        │           GRU Cell              │
        │                                 │
  xₜ ──→│  ┌─────────┐   ┌─────────┐     │
        │  │ Reset   │   │ Update  │     │

---

## 1. GRU：LSTM的精简版

### 1.1 为什么需要GRU？

**LSTM的问题：**
- 参数量大（约4倍于RNN）
- 计算复杂度高
- 训练时间长

**GRU的设计目标：**
- 保持LSTM捕捉长期依赖的能力
- 减少参数量
- 简化计算

### 1.2 GRU架构

GRU将LSTM的4个门合并为2个门：

```
        ┌─────────────────────────────────┐
        │           GRU Cell              │
        │                                 │
  xₜ ──→│  ┌─────────┐   ┌─────────┐     │
        │  │ Reset   │   │ Update  │     │
        │  │  Gate   │   │  Gate   │     │
        │  │  (rₜ)   │   │  (zₜ)   │     │
        │  └────┬────┘   └────┬────┘     │
        │       │             │          │
        │       ▼             ▼          │
   hₜ₋₁→│    [×]            [1-z]       │
        │       │               │        │
        │       ▼               ▼        │
        │    [tanh]          [×]←──hₜ₋₁  │
        │       │               │        │
        │       └──→[+]←────────┘        │
        │              │                 │
        │              ▼                 │
        │             hₜ                 │
        └─────────────────────────────────┘
```

### 1.3 数学公式

**1. 重置门 (Reset Gate)**

决定遗忘多少过去的信息：

$$
r_t = \sigma(W_r \cdot [h_{t-1}, x_t])
$$

**2. 更新门 (Update Gate)**

决定保留多少旧状态，接受多少新状态：

$$
z_t = \sigma(W_z \cdot [h_{t-1}, x_t])
$$

**3. 候选隐藏状态**

$$
\tilde{h}_t = \tanh(W \cdot [r_t \odot h_{t-1}, x_t])
$$

**4. 最终隐藏状态**

$$
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
$$

### 1.4 GRU vs LSTM 对比

| 特性 | LSTM | GRU |
|------|------|-----|
| 门数量 | 3个 (f, i, o) | 2个 (r, z) |
| 状态变量 | $h_t$ + $C_t$ | 只有 $h_t$ |
| 参数量 | 约 $4n^2$ | 约 $3n^2$ |
| 计算速度 | 较慢 | 较快 |
| 长期依赖 | 强 | 较强 |

**直观理解：**
- **更新门 z**：相当于LSTM的遗忘门 + 输入门
  - $z \approx 1$：保留旧状态（类似遗忘门≈1，输入门≈0）
  - $z \approx 0$：接受新状态（类似遗忘门≈0，输入门≈1）
- **重置门 r**：控制过去信息对候选状态的影响
  - $r \approx 0$：完全忽略过去，只依赖当前输入
  - $r \approx 1$：充分利用过去信息

### 1.5 PyTorch实现

```python
import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        gru_out, hidden = self.gru(x)
        # gru_out: (batch, seq_len, hidden_size)
        
        last_hidden = gru_out[:, -1, :]
        output = self.fc(last_hidden)
        
        return output

# 对比参数量
lstm = nn.LSTM(input_size=100, hidden_size=128, num_layers=1)
gru = nn.GRU(input_size=100, hidden_size=128, num_layers=1)

lstm_params = sum(p.numel() for p in lstm.parameters())
gru_params = sum(p.numel() for p in gru.parameters())

print(f"LSTM参数: {lstm_params}")   # 约118k
print(f"GRU参数: {gru_params}")     # 约88k (约75%)
```

### 1.6 实验对比

在多个NLP任务上的性能对比：

| 任务 | LSTM | GRU | 结论 |
|------|------|-----|------|
| 语言建模 | 78.4 PPL | 79.2 PPL | 相当 |
| 机器翻译 | 24.5 BLEU | 24.1 BLEU | 相当 |
| 情感分析 | 87.2% Acc | 86.8% Acc | 相当 |
| 训练时间 | 100% | 75% | GRU更快 |

**结论**：GRU在保持性能的同时，训练更快，参数量更少。

---

## 2. Seq2Seq：序列到序列学习

### 2.1 问题背景

**传统RNN/LSTM的局限：**
- 输入输出长度必须相同
- 只能处理单序列任务

**Seq2Seq解决：**
- 输入输出长度可以不同
- 适用于翻译、摘要、对话等任务

### 2.2 架构概览

```
Encoder（编码器）          Decoder（解码器）

"Hello" → [E] ──┐
"World" → [E] ──┼→ LSTM → LSTM → LSTM → [Context]
"!"     → [E] ──┘         ↑
                          │
                    [Context Vector]
                          │
                          ↓
                    [Start] → LSTM → "Bonjour"
                                ↓
                              LSTM → "le"
                                ↓
                              LSTM → "monde"
                                ↓
                              LSTM → "!"
                                ↓
                             [End]
```

### 2.3 Encoder（编码器）

将输入序列编码为固定长度的上下文向量：

```python
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
    
    def forward(self, input_seq):
        # input_seq: (batch, seq_len)
        embedded = self.embedding(input_seq)  # (batch, seq_len, hidden)
        
        # outputs: (batch, seq_len, hidden)
        # hidden: (num_layers, batch, hidden)
        # cell: (num_layers, batch, hidden)
        outputs, (hidden, cell) = self.lstm(embedded)
        
        return outputs, (hidden, cell)
```

### 2.4 Decoder（解码器）

根据上下文向量生成输出序列：

```python
class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, input_token, hidden, cell):
        # input_token: (batch, 1) - 单个词
        # hidden, cell: 来自encoder或上一步decoder
        
        embedded = self.embedding(input_token)  # (batch, 1, hidden)
        
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        # output: (batch, 1, hidden)
        
        prediction = self.fc(output.squeeze(1))  # (batch, output_size)
        
        return prediction, hidden, cell
```

### 2.5 完整Seq2Seq模型

```python
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: (batch, src_len)
        # trg: (batch, trg_len)
        
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_size
        
        # 存储所有输出
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        # Encoder
        _, (hidden, cell) = self.encoder(src)
        
        # Decoder的第一个输入是 <SOS> token
        input_token = trg[:, 0]  # (batch,)
        
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input_token.unsqueeze(1), hidden, cell)
            outputs[:, t, :] = output
            
            # Teacher Forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)  # 预测的词
            input_token = trg[:, t] if teacher_force else top1
        
        return outputs
```

### 2.6 Teacher Forcing

**问题：** 如果decoder在第一步预测错误，错误会传播到后续所有步骤

**解决方案：** 训练时有一定概率使用真实标签作为下一步输入

```python
# Teacher Forcing 示例
for t in range(trg_len):
    if random.random() < teacher_forcing_ratio:
        input_token = trg[:, t]      # 使用真实标签（老师指导）
    else:
        input_token = prev_output    # 使用模型预测（自主学习）
```

---

## 3. 注意力机制初步

### 3.1 Seq2Seq的瓶颈

**问题：** 所有信息被压缩到一个固定长度的上下文向量

```
"The cat sat on the mat and looked at the bird"
                    ↓
              [Context Vector]
                    ↓
         "Le chat s'est assis sur le tapis"
         
问题：长句子时，信息丢失严重
```

### 3.2 注意力机制的核心思想

**关键洞察：** Decoder在生成每个词时，应该关注Encoder的不同部分

```
生成 "chat" 时 → 关注 "cat"
生成 "tapis" 时 → 关注 "mat"
```

### 3.3 注意力计算流程

```
Step 1: 计算注意力分数
        score(s_t, h_i) = s_t^T · h_i
        
Step 2: 归一化为注意力权重
        α_i = softmax(score_i)
        
Step 3: 计算上下文向量
        c_t = Σ α_i · h_i
        
Step 4: 结合上下文生成输出
        output = f(c_t, s_t)
```

### 3.4 可视化理解

```
Encoder Outputs:  [h₁]  [h₂]  [h₃]  [h₄]  [h₅]
                    ↓     ↓     ↓     ↓     ↓
Attention Weights: 0.1   0.1   0.6   0.1   0.1  (生成"cat"时)
                    ↓     ↓     ↓     ↓     ↓
Context Vector:    c_t = 0.1h₁ + 0.1h₂ + 0.6h₃ + 0.1h₄ + 0.1h₅
                          ↑
                      主要关注h₃
```

### 3.5 注意力机制的优势

1. **解决长距离依赖**：直接访问所有encoder状态
2. **可解释性**：注意力权重显示模型关注哪里
3. **并行计算**：注意力可以并行计算

### 3.6 注意力类型概览

| 类型 | 名称 | 特点 |
|------|------|------|
| Soft Attention | 软注意力 | 加权平均，可微分 |
| Hard Attention | 硬注意力 | 选择单一位置，需强化学习 |
| Self-Attention | 自注意力 | 序列对自身计算注意力 |
| Multi-Head | 多头注意力 | 多组注意力并行计算 |

**注意**：详细注意力机制将在后续阶段（Transformer）深入学习。

---

## 4. 总结与展望

### 4.1 本阶段要点

| 主题 | 核心内容 |
|------|----------|
| **GRU** | LSTM的简化版，2个门控，参数更少，速度更快 |
| **Seq2Seq** | Encoder-Decoder架构，处理变长序列转换 |
| **注意力** | 解决信息瓶颈，实现选择性关注 |

### 4.2 架构演进

```
RNN (1986)
  ↓
LSTM (1997) ← 解决梯度消失
  ↓
GRU (2014) ← 简化LSTM
  ↓
Seq2Seq (2014) ← 序列转换
  ↓
Attention (2014) ← 选择性关注
  ↓
Transformer (2017) ← 完全并行 + 自注意力 ← 现代LLM基础
```

### 4.3 下一步学习

1. **Transformer架构**：Self-Attention、Multi-Head Attention、Position Encoding
2. **预训练语言模型**：BERT、GPT系列
3. **大语言模型**：Scaling Law、RLHF、Prompt Engineering

### 4.4 学习路径总览

```
Stage 1: Word2Vec/GloVe ──┐
Stage 2: FastText ────────┼→ 词嵌入基础
Stage 3: Embedding对比 ───┘
Stage 4: RNN基础 ─────────┐
Stage 5: LSTM ────────────┼→ 序列建模
Stage 6: GRU + Seq2Seq ───┘
Stage 7: Transformer ←─── 当前前沿基础
Stage 8: BERT/GPT
Stage 9: 大模型训练与应用
```

---

## 参考资源

1. **论文**：
   - "Learning Phrase Representations using RNN Encoder-Decoder" (Cho et al., 2014) - GRU
   - "Sequence to Sequence Learning with Neural Networks" (Sutskever et al., 2014) - Seq2Seq
   - "Neural Machine Translation by Jointly Learning to Align and Translate" (Bahdanau et al., 2015) - Attention

2. **教程**：
   - PyTorch Seq2Seq Tutorial
   - Harvard NLP: The Annotated Transformer

3. **可视化**：
   - https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/

---

*Created: 2026-03-16 | Stage 6 of LLM Learning Roadmap (Final Stage)*