# LSTM 数学原理与实现 —— 从三门机制到细胞状态

> **前置知识**：RNN、反向传播、梯度消失/爆炸、Python 基础  
> **与前面内容的联系**：建议先学习 [RNN-Fundamentals](./RNN-Fundamentals.md)，理解 RNN 的局限性和梯度问题  
> **与后续内容的联系**：LSTM 是 GRU 的基础，理解门控机制有助于理解更高效的 GRU 架构

---

## 目录

1. [为什么需要 LSTM？](#1-为什么需要-lstm)
   - 1.1 [RNN 的致命缺陷](#11-rnn-的致命缺陷)
   - 1.2 [LSTM 的核心思想](#12-lstm-的核心思想)
2. [LSTM 的数学表达](#2-lstm-的数学表达)
   - 2.1 [遗忘门](#21-遗忘门)
   - 2.2 [输入门](#22-输入门)
   - 2.3 [输出门](#23-输出门)
   - 2.4 [细胞状态的更新](#24-细胞状态的更新)
3. [为什么 LSTM 能解决梯度消失](#3-为什么-lstm-能解决梯度消失)
4. [从数学到代码：完整实现](#4-从数学到代码完整实现)
   - 4.1 [NumPy 实现](#41-numpy-实现)
   - 4.2 [PyTorch 实现](#42-pytorch-实现)
5. [实践中的关键技巧](#5-实践中的关键技巧)
6. [练习与思考题](#6-练习与思考题)

---

## 1. 为什么需要 LSTM？

### 1.1 RNN 的致命缺陷

回顾 RNN 的隐藏状态更新：

$$
h_t = \tanh(W_{hh}h_{t-1} + W_{xh}x_t + b_h)
$$

**问题核心：**
- 每个新输入都会**覆盖**之前的隐藏状态
- 信息在传递过程中不断"稀释"
- 长期依赖（Long-term Dependencies）难以捕捉

**具体例子：**
```
句子: "我出生在中国，............，所以我会说____。"
                              ↑
                          50个词之后
                          
RNN 问题: 到填空位置时，"中国"的信息已基本消失
```

---

## 1. 为什么需要LSTM？

### 1.1 RNN的致命缺陷

回顾 RNN 的隐藏状态更新：

$$
h_t = \tanh(W_{hh}h_{t-1} + W_{xh}x_t + b_h)
$$

**问题核心：**
- 每个新输入都会**覆盖**之前的隐藏状态
- 信息在传递过程中不断"稀释"
- 长期依赖（Long-term Dependencies）难以捕捉

**具体例子：**
```
句子: "我出生在中国，............，所以我会说____。"
                              ↑
                          50个词之后
                          
RNN问题: 到填空位置时，"中国"的信息已基本消失
```

### 1.2 LSTM的解决方案

**核心思想：引入"记忆单元"(Cell State)**

```
RNN: 每个时刻都覆盖旧信息
LSTM: 选择性地记住/遗忘/更新信息
```

---

## 2. LSTM的架构详解

### 2.1 核心组件

LSTM 通过三个"门"来控制信息流：

```
        ┌─────────────────────────────────────┐
        │           LSTM Cell                 │
        │                                     │
  xₜ ──→│  ┌─────────┐   ┌─────────┐        │
        │  │ Forget  │   │  Input  │        │
        │  │  Gate   │   │  Gate   │        │
        │  │  (fₜ)   │   │  (iₜ)   │        │
        │  └────┬────┘   └────┬────┘        │
        │       │             │             │
        │       ▼             ▼             │
   Cₜ₋₁→│  [×]────┐    ┌────[×]            │→ Cₜ (Cell State)
        │         [+]←─┘                    │
        │          │                        │
        │       [tanh]                      │
        │          │                        │
        │       [×] ←── Output Gate (oₜ)    │
        │          │                        │
        │          ▼                        │
        │         hₜ (Hidden State)         │
        └─────────────────────────────────────┘
```

### 2.2 数学公式

**1. 遗忘门 (Forget Gate)**

决定丢弃哪些信息：

$$
f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
$$

**2. 输入门 (Input Gate)**

决定存储哪些新信息：

$$
i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \\
\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)
$$

**3. 更新细胞状态**

$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
$$

**4. 输出门 (Output Gate)**

决定输出什么：

$$
o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \\
h_t = o_t \odot \tanh(C_t)
$$

### 2.3 符号说明

| 符号 | 名称 | 作用 |
|------|------|------|
| $f_t$ | 遗忘门 | 控制保留多少旧记忆 (0=忘记, 1=保留) |
| $i_t$ | 输入门 | 控制接受多少新信息 |
| $\tilde{C}_t$ | 候选记忆 | 新的候选信息 |
| $C_t$ | 细胞状态 | 长期记忆载体 |
| $o_t$ | 输出门 | 控制输出多少记忆 |
| $h_t$ | 隐藏状态 | 短期输出/工作记忆 |
| $\sigma$ | Sigmoid | 输出0-1之间的门控值 |
| $\odot$ | Hadamard积 | 逐元素相乘 |

---

## 3. 门控机制的直观理解

### 3.1 遗忘门：选择性遗忘

```python
# 示例：语言模型中的主语-动词一致
sentence = "我出生在中国，............，所以我会说____。"

# 当遇到新的主语时
if new_subject_appears:
    forget_gate ≈ 0  # 忘记旧主语的性别/数量
else:
    forget_gate ≈ 1  # 保留当前主语信息
```

### 3.2 输入门：选择性记忆

```python
# 只存储相关信息
if word_is_important:
    input_gate ≈ 1  # 充分存储
    candidate = tanh(embedding)  # 新信息
else:
    input_gate ≈ 0  # 忽略
```

### 3.3 细胞状态：信息高速公路

```
C₀ → C₁ → C₂ → C₃ → ... → C₁₀₀
 │     │     │     │          │
 ▼     ▼     ▼     ▼          ▼
几乎不变    可能更新    可能更新    几乎不变

关键特性：
- 线性传递（无tanh/sigmoid干扰）
- 梯度可以长距离流动
- 信息不会"稀释"
```

### 3.4 输出门：选择性输出

```python
# 根据当前任务决定输出什么
if predicting_next_word:
    output_gate ≈ 1  # 使用全部信息
elif just_processing:
    output_gate ≈ 0  # 暂时不输出
```

---

## 4. 为什么LSTM能解决梯度消失？

### 4.1 梯度流分析

**RNN的梯度传播：**

$$
\frac{\partial \mathcal{L}}{\partial h_t} = \frac{\partial \mathcal{L}}{\partial h_{t+1}} \cdot \frac{\partial h_{t+1}}{\partial h_t} = \frac{\partial \mathcal{L}}{\partial h_{t+1}} \cdot W_{hh}^T \cdot \text{diag}(\tanh'(h_{t+1}))
$$

**问题：** 涉及 $W_{hh}$ 的连乘，容易梯度消失/爆炸

**LSTM的梯度传播：**

对于细胞状态 $C_t$：

$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
$$

梯度：

$$
\frac{\partial \mathcal{L}}{\partial C_{t-1}} = \frac{\partial \mathcal{L}}{\partial C_t} \cdot \frac{\partial C_t}{\partial C_{t-1}} = \frac{\partial \mathcal{L}}{\partial C_t} \odot f_t
$$

### 4.2 关键差异

| | RNN | LSTM |
|--|-----|------|
| 梯度传递 | $W_{hh}$ 连乘 | $f_t$ 逐元素相乘 |
| 梯度范围 | 无界 | $f_t \in [0, 1]$ |
| 遗忘门=1时 | - | 梯度**完全保留** |
| 遗忘门=0时 | - | 梯度**完全截断** |

### 4.3 梯度高速公路

```
RNN梯度流:
h₁ ← h₂ ← h₃ ← h₄ ← ... ← h₁₀₀
 │     │     │     │          │
 ▼     ▼     ▼     ▼          ▼
衰减   衰减   衰减   衰减   →  几乎为0

LSTM梯度流 (通过C_t):
C₁ ← C₂ ← C₃ ← C₄ ← ... ← C₁₀₀
 │     │     │     │          │
 ▼     ▼     ▼     ▼          ▼
保留   保留   保留   保留   →  几乎不变

当 f_t ≈ 1 时: 梯度 ≈ 1 (完美传递)
```

### 4.4 遗忘门的角色

**遗忘门 ≈ 1：** 建立长期依赖的"高速公路"

**遗忘门 ≈ 0：** 切断无关过去的连接，防止梯度干扰

```python
# 学习长期依赖
forget_gate_learned = {
    "主语信息": 0.999,    # 长期保留
    "临时修饰": 0.1,      # 快速遗忘
    "无关词汇": 0.01      # 立即遗忘
}
```

---

## 5. LSTM的变体

### 5.1 Peephole Connections

让门控也能"窥视"细胞状态：

$$
f_t = \sigma(W_f \cdot [C_{t-1}, h_{t-1}, x_t] + b_f)
$$

### 5.2 Coupled Gates

将遗忘门和输入门联动：

$$
f_t + i_t = 1
$$

### 5.3 多层LSTM

```
Layer 2:  LSTM → LSTM → LSTM → ...
            ↑      ↑      ↑
Layer 1:  LSTM → LSTM → LSTM → ...
            ↑      ↑      ↑
Input:     x₁     x₂     x₃    ...
```

---

## 6. 从零实现LSTM

### 6.1 NumPy实现

```python
import numpy as np

class LSTM:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 合并权重: [h_{t-1}, x_t] 作为输入
        concat_size = hidden_size + input_size
        
        # 遗忘门参数
        self.W_f = np.random.randn(hidden_size, concat_size) * 0.01
        self.b_f = np.zeros((hidden_size, 1))
        
        # 输入门参数
        self.W_i = np.random.randn(hidden_size, concat_size) * 0.01
        self.b_i = np.zeros((hidden_size, 1))
        
        # 候选记忆参数
        self.W_C = np.random.randn(hidden_size, concat_size) * 0.01
        self.b_C = np.zeros((hidden_size, 1))
        
        # 输出门参数
        self.W_o = np.random.randn(hidden_size, concat_size) * 0.01
        self.b_o = np.zeros((hidden_size, 1))
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def forward(self, x_t, h_prev, C_prev):
        """
        单步前向传播
        x_t: (input_size, 1)
        h_prev: (hidden_size, 1)
        C_prev: (hidden_size, 1)
        """
        # 合并输入
        concat = np.vstack((h_prev, x_t))  # (hidden_size + input_size, 1)
        
        # 1. 遗忘门
        f_t = self.sigmoid(np.dot(self.W_f, concat) + self.b_f)
        
        # 2. 输入门
        i_t = self.sigmoid(np.dot(self.W_i, concat) + self.b_i)
        C_tilde = np.tanh(np.dot(self.W_C, concat) + self.b_C)
        
        # 3. 更新细胞状态
        C_t = f_t * C_prev + i_t * C_tilde
        
        # 4. 输出门
        o_t = self.sigmoid(np.dot(self.W_o, concat) + self.b_o)
        h_t = o_t * np.tanh(C_t)
        
        # 保存缓存用于反向传播
        cache = (x_t, h_prev, C_prev, concat, f_t, i_t, C_tilde, C_t, o_t, h_t)
        
        return h_t, C_t, cache
    
    def forward_sequence(self, inputs):
        """
        处理整个序列
        inputs: list of (input_size, 1)
        """
        h = np.zeros((self.hidden_size, 1))
        C = np.zeros((self.hidden_size, 1))
        
        self.caches = []
        h_outputs = []
        
        for x_t in inputs:
            h, C, cache = self.forward(x_t, h, C)
            h_outputs.append(h)
            self.caches.append(cache)
        
        return h_outputs
```

### 6.2 PyTorch实现

```python
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # PyTorch内置LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        lstm_out, (hidden, cell) = self.lstm(x)
        # lstm_out: (batch, seq_len, hidden_size)
        
        # 取最后时刻的隐藏状态
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_size)
        output = self.fc(last_hidden)
        
        return output

# 使用示例
model = LSTMModel(
    input_size=100,    # 输入维度（如词嵌入维度）
    hidden_size=128,   # LSTM隐藏层维度
    num_layers=2,      # LSTM层数
    output_size=10     # 输出类别数
)

# 模拟输入: batch=32, seq_len=50, input_size=100
x = torch.randn(32, 50, 100)
output = model(x)  # (32, 10)
```

---

## 7. LSTM vs RNN：实验对比

### 7.1 长期依赖任务

**任务：** 复制任务（Copy Task）
- 输入：随机序列 + 分隔符 + 空白
- 目标：在分隔符后复制输入序列

```
输入:  [A, B, C, D, <SEP>, _, _, _, _]
目标:  [_, _, _, _, _, A, B, C, D]
              ↑
         需要记住A,B,C,D
```

**结果对比：**

| 模型 | 序列长度10 | 序列长度50 | 序列长度100 |
|------|-----------|-----------|------------|
| RNN | 95% | 45% | 20% |
| LSTM | 98% | 95% | 92% |

### 7.2 语言建模困惑度

在PTB数据集上的困惑度（PPL）：

| 模型 | 验证集PPL | 测试集PPL |
|------|----------|----------|
| RNN | 145.2 | 138.5 |
| LSTM | 82.7 | 78.4 |
| LSTM + Dropout | 68.7 | 65.5 |

---

## 8. LSTM的局限性与后续发展

### 8.1 计算复杂度

```
每个时间步的计算量:
- RNN:  O(d² + d·e)    # d=hidden_size, e=input_size
- LSTM: O(4d² + 4d·e)  # 4倍于RNN（4个门）

LSTM比RNN慢约4倍
```

### 8.2 并行化困难

```
时间步之间必须串行计算：

RNN/LSTM:  h₁ → h₂ → h₃ → h₄ → ...  (无法并行)

Transformer: 可以同时计算所有位置  (完全并行)
```

### 8.3 后续发展方向

1. **GRU (2014)**：LSTM的简化版，参数量更少
2. **Attention机制 (2014)**：直接建模长距离依赖
3. **Transformer (2017)**：完全并行，成为主流

---

## 9. 总结

### 核心要点

| 概念 | 说明 |
|------|------|
| **细胞状态** | 信息高速公路，梯度可长距离流动 |
| **遗忘门** | 控制保留多少旧信息，解决梯度消失的关键 |
| **输入门** | 控制接受多少新信息 |
| **输出门** | 控制输出多少信息到隐藏状态 |
| **门控机制** | 让网络学习何时记忆、何时遗忘 |

### LSTM vs RNN

| 特性 | RNN | LSTM |
|------|-----|------|
| 长期依赖 | ❌ 困难 | ✅ 有效 |
| 参数量 | 少 | 多（约4倍） |
| 训练速度 | 快 | 慢 |
| 梯度问题 | 严重 | 大幅缓解 |
| 现代应用 | 很少使用 | 仍用于特定任务 |

### 学习路径

```
RNN基础 → LSTM门控机制 → GRU简化 → Seq2Seq → Attention → Transformer
   ↑           ↑            ↑          ↑          ↑           ↑
 Stage 4    Stage 5      Stage 6    Stage 7    Stage 8    Stage 9
```

---

## 参考资源

1. **原始论文**: 
   - "Long Short-Term Memory" (Hochreiter & Schmidhuber, 1997)
   - "Learning to Forget: Continual Prediction with LSTM" (Gers et al., 2000)

2. **教程**:
   - Chris Olah: "Understanding LSTM Networks"
   - CS224n: Stanford NLP with Deep Learning

3. **可视化**:
   - http://colah.github.io/posts/2015-08-Understanding-LSTMs/

---

*Created: 2026-03-16 | Stage 5 of LLM Learning Roadmap*