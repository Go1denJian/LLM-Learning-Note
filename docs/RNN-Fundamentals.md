# RNN 数学原理与实现 —— 从全连接到循环

> **前置知识**：全连接神经网络、反向传播、梯度下降、Python 基础  
> **与前面内容的联系**：建议先学习 [Word-Embedding](./Word-Embedding-Math-and-Implementation.md)，理解词向量表示  
> **与后续内容的联系**：RNN 是 LSTM、GRU、Transformer 等序列模型的基础，理解 RNN 的局限性有助于理解后续改进

---

## 目录

1. [从 MLP 到 RNN：为什么需要循环结构？](#1-从-mlp-到-rnn为什么需要循环结构)
   - 1.1 [传统神经网络的局限](#11-传统神经网络的局限)
   - 1.2 [RNN 的核心思想](#12-rnn-的核心思想)
2. [RNN 的数学表达](#2-rnn-的数学表达)
   - 2.1 [前向传播](#21-前向传播)
   - 2.2 [反向传播 (BPTT)](#22-反向传播-bptt)
3. [梯度消失与梯度爆炸](#3-梯度消失与梯度爆炸)
   - 3.1 [数学本质](#31-数学本质)
   - 3.2 [影响与解决方案](#32-影响与解决方案)
4. [从数学到代码：完整实现](#4-从数学到代码完整实现)
   - 4.1 [NumPy 实现](#41-numpy-实现)
   - 4.2 [PyTorch 实现](#42-pytorch-实现)
5. [实践中的关键技巧](#5-实践中的关键技巧)
6. [练习与思考题](#6-练习与思考题)

---

## 1. 从 MLP 到 RNN：为什么需要循环结构？

### 1.1 传统神经网络的局限

**MLP (Multi-Layer Perceptron) 的问题：**
- 输入输出维度固定
- 无法处理变长序列
- 缺乏时间/顺序感知能力

**序列数据的挑战：**
```
输入: "The cat sat on the..."
目标: 预测下一个词 "mat"

关键洞察: 词序信息至关重要
"The cat sat on the mat" ≠ "mat the on sat cat The"
```

### 1.2 RNN 的核心思想

**关键创新：隐藏状态 (Hidden State)**

RNN 引入了一个"记忆单元"，将过去的信息传递到未来：

```
        x₁        x₂        x₃        x₄
        ↓         ↓         ↓         ↓
    ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐
    │  RNN  │→│  RNN  │→│  RNN  │→│  RNN  │
    │ Cell  │ │ Cell  │ │ Cell  │ │ Cell  │
    └───┬───┘ └───┬───┘ └───┬───┘ └───┬───┘
        ↓         ↓         ↓         ↓
        h₁   →    h₂   →    h₃   →    h₄
        ↓         ↓         ↓         ↓
        y₁        y₂        y₃        y₄
        
        hₜ = f(hₜ₋₁, xₜ)  # 核心递推公式
```

---

## 1. 从MLP到RNN：为什么需要循环结构？

### 1.1 传统神经网络的局限

**MLP (Multi-Layer Perceptron) 的问题：**
- 输入输出维度固定
- 无法处理变长序列
- 缺乏时间/顺序感知能力

**序列数据的挑战：**
```
输入: "The cat sat on the..."
目标: 预测下一个词 "mat"

关键洞察: 词序信息至关重要
"The cat sat on the mat" ≠ "mat the on sat cat The"
```

### 1.2 RNN的核心思想

**关键创新：隐藏状态 (Hidden State)**

RNN引入了一个"记忆单元"，将过去的信息传递到未来：

```
        x₁        x₂        x₃        x₄
        ↓         ↓         ↓         ↓
    ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐
    │  RNN  │→│  RNN  │→│  RNN  │→│  RNN  │
    │ Cell  │ │ Cell  │ │ Cell  │ │ Cell  │
    └───┬───┘ └───┬───┘ └───┬───┘ └───┬───┘
        ↓         ↓         ↓         ↓
        h₁   →    h₂   →    h₃   →    h₄
        ↓         ↓         ↓         ↓
        y₁        y₂        y₃        y₄
        
        hₜ = f(hₜ₋₁, xₜ)  # 核心递推公式
```

---

## 2. RNN的数学表达

### 2.1 前向传播 (Forward Propagation)

**隐藏状态更新：**

$$
h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)
$$

**输出计算：**

$$
y_t = W_{hy} h_t + b_y \quad \text{(或)} \quad \hat{y}_t = \text{softmax}(W_{hy} h_t + b_y)
$$

**参数说明：**
| 符号 | 维度 | 含义 |
|------|------|------|
| $x_t$ | $(d_{in}, 1)$ | 时刻t的输入向量 |
| $h_t$ | $(d_{hidden}, 1)$ | 时刻t的隐藏状态 |
| $y_t$ | $(d_{out}, 1)$ | 时刻t的输出 |
| $W_{xh}$ | $(d_{hidden}, d_{in})$ | 输入到隐藏的权重 |
| $W_{hh}$ | $(d_{hidden}, d_{hidden})$ | 隐藏到隐藏的权重（循环权重）|
| $W_{hy}$ | $(d_{out}, d_{hidden})$ | 隐藏到输出的权重 |
| $b_h, b_y$ | 偏置项 | - |

### 2.2 展开视角 (Unrolled View)

将RNN在时间上展开，可以看作一个深层网络：

```
x₁ → [RNN] → h₁ → [RNN] → h₂ → [RNN] → h₃ → ... → h_T
         ↓          ↓          ↓                ↓
         y₁         y₂         y₃               y_T
         
共享参数: W_xh, W_hh, W_hy, b_h, b_y 在所有时间步相同
```

**参数共享的意义：**
- 减少参数量
- 模型可以处理任意长度的序列
- 学习到的模式具有时间平移不变性

---

## 3. BPTT：随时间反向传播

### 3.1 损失函数定义

对于序列任务，总损失是各时刻损失的累加：

$$
\mathcal{L} = \sum_{t=1}^{T} \mathcal{L}_t = \sum_{t=1}^{T} \text{loss}(y_t, \hat{y}_t)
$$

### 3.2 梯度计算

**输出层梯度：**

$$
\frac{\partial \mathcal{L}}{\partial W_{hy}} = \sum_{t=1}^{T} \frac{\partial \mathcal{L}_t}{\partial W_{hy}} = \sum_{t=1}^{T} (\hat{y}_t - y_t) \cdot h_t^T
$$

**隐藏层梯度（关键！）：**

对于时刻t的隐藏状态，梯度来自两部分：
1. 当前时刻的直接损失
2. 未来时刻通过h_t传播的梯度

$$
\frac{\partial \mathcal{L}}{\partial h_t} = \underbrace{W_{hy}^T \cdot (\hat{y}_t - y_t)}_{\text{当前时刻}} + \underbrace{W_{hh}^T \cdot \frac{\partial \mathcal{L}}{\partial h_{t+1}} \odot (1 - \tanh^2(h_{t+1}))}_{\text{未来时刻传播}}
$$

### 3.3 梯度展开

将梯度从时刻T反向传播到时刻t：

$$
\frac{\partial \mathcal{L}}{\partial h_t} = \sum_{k=t}^{T} \left( \prod_{j=t+1}^{k} W_{hh}^T \cdot \text{diag}(1 - \tanh^2(h_j)) \right) \cdot \frac{\partial \mathcal{L}_k}{\partial h_k}
$$

**核心观察：** 梯度计算涉及 $W_{hh}$ 的连乘！

---

## 4. 梯度消失与梯度爆炸

### 4.1 问题本质

考虑从时刻t传播到时刻1的梯度：

$$
\frac{\partial \mathcal{L}}{\partial h_1} \propto \prod_{i=2}^{t} W_{hh}^T \cdot \text{diag}(\tanh'(h_i))
$$

**梯度范数分析：**

令 $\gamma$ 为 $\tanh'$ 的最大值（≈1），$\lambda_{max}$ 为 $W_{hh}$ 的最大奇异值：

$$
\left\| \frac{\partial \mathcal{L}}{\partial h_1} \right\| \approx \|W_{hh}\|^{t-1} \cdot \gamma^{t-1} = \lambda_{max}^{t-1} \cdot \gamma^{t-1}
$$

### 4.2 梯度消失 (Vanishing Gradient)

**条件：** $\lambda_{max} < 1$

**现象：**
```
序列长度: 10   → 梯度 ≈ (0.9)^10  ≈ 0.35
序列长度: 50   → 梯度 ≈ (0.9)^50  ≈ 0.005
序列长度: 100  → 梯度 ≈ (0.9)^100 ≈ 0.000026
```

**后果：**
- 远距离依赖无法学习
- 模型只能捕捉短期模式
- 长序列建模失败

### 4.3 梯度爆炸 (Exploding Gradient)

**条件：** $\lambda_{max} > 1$

**现象：**
```
序列长度: 10   → 梯度 ≈ (1.1)^10  ≈ 2.6
序列长度: 50   → 梯度 ≈ (1.1)^50  ≈ 117
序列长度: 100  → 梯度 ≈ (1.1)^100 ≈ 13780
```

**后果：**
- 数值溢出 (NaN/Inf)
- 参数更新不稳定
- 训练无法收敛

### 4.4 可视化理解

```
梯度传播路径:

h₁ ← h₂ ← h₃ ← ... ← h₁₀₀
 │     │     │          │
 ▼     ▼     ▼          ▼
小    较小   中等   →  大 (梯度爆炸)

或:

h₁ ← h₂ ← h₃ ← ... ← h₁₀₀
 │     │     │          │
 ▼     ▼     ▼          ▼
极小  很小   小     →  正常 (梯度消失)
```

---

## 5. 解决方案概览

### 5.1 梯度裁剪 (Gradient Clipping)

解决梯度爆炸的简单方法：

```python
if ||gradient|| > threshold:
    gradient = gradient * (threshold / ||gradient||)
```

### 5.2 更好的架构

**LSTM (Long Short-Term Memory)**
- 引入门控机制控制信息流
- 解决梯度消失问题

**GRU (Gated Recurrent Unit)**
- LSTM的简化变体
- 参数更少，计算更快

**将在后续阶段详细学习...**

---

## 6. 从零实现基础RNN

### 6.1 NumPy实现

```python
import numpy as np

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        # 初始化权重
        self.W_xh = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_hy = np.random.randn(output_size, hidden_size) * 0.01
        
        # 偏置
        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((output_size, 1))
        
        self.hidden_size = hidden_size
        
    def forward(self, inputs):
        """
        inputs: 序列列表，每个元素是 (input_size, 1) 的向量
        返回: (outputs, hidden_states)
        """
        h = np.zeros((self.hidden_size, 1))
        self.hidden_states = [h]
        self.inputs = inputs
        outputs = []
        
        for x in inputs:
            # h_t = tanh(W_xh @ x + W_hh @ h + b_h)
            h = np.tanh(np.dot(self.W_xh, x) + 
                       np.dot(self.W_hh, h) + self.b_h)
            self.hidden_states.append(h)
            
            # y_t = W_hy @ h + b_y
            y = np.dot(self.W_hy, h) + self.b_y
            outputs.append(y)
            
        return outputs, self.hidden_states[1:]
    
    def backward(self, targets, outputs, learning_rate=0.1):
        """BPTT实现"""
        T = len(outputs)
        
        # 梯度初始化
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)
        
        # 从最后时刻开始反向传播
        dh_next = np.zeros((self.hidden_size, 1))
        
        for t in reversed(range(T)):
            # 输出层梯度
            dy = outputs[t] - targets[t]  # 假设MSE损失
            dW_hy += np.dot(dy, self.hidden_states[t+1].T)
            db_y += dy
            
            # 隐藏层梯度
            dh = np.dot(self.W_hy.T, dy) + dh_next
            dh_raw = dh * (1 - self.hidden_states[t+1] ** 2)  # tanh导数
            
            dW_xh += np.dot(dh_raw, self.inputs[t].T)
            dW_hh += np.dot(dh_raw, self.hidden_states[t].T)
            db_h += dh_raw
            
            dh_next = np.dot(self.W_hh.T, dh_raw)
        
        # 梯度裁剪
        for grad in [dW_xh, dW_hh, dW_hy, db_h, db_y]:
            np.clip(grad, -5, 5, out=grad)
        
        # 参数更新
        self.W_xh -= learning_rate * dW_xh
        self.W_hh -= learning_rate * dW_hh
        self.W_hy -= learning_rate * dW_hy
        self.b_h -= learning_rate * db_h
        self.b_y -= learning_rate * db_y
```

### 6.2 PyTorch实现

```python
import torch
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, hidden = self.rnn(x)
        # out: (batch, seq_len, hidden_size)
        out = self.fc(out)
        return out, hidden

# 使用示例
model = RNNModel(input_size=100, hidden_size=128, output_size=10000)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练循环
for epoch in range(num_epochs):
    outputs, _ = model(inputs)
    loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
    
    optimizer.zero_grad()
    loss.backward()
    # PyTorch自动处理梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
    optimizer.step()
```

---

## 7. RNN的变体架构

### 7.1 单向 vs 双向

```
单向RNN (Left-to-Right):
→ → → → →
x₁ x₂ x₃ x₄ x₅

双向RNN (BiRNN):
← ← ← ← ←
x₁ x₂ x₃ x₄ x₅
→ → → → →

输出: [h_forward; h_backward] (拼接)
```

### 7.2 深层RNN

```
Layer 3:  h₃₁ → h₃₂ → h₃₃ → ...
           ↑     ↑     ↑
Layer 2:  h₂₁ → h₂₂ → h₂₃ → ...
           ↑     ↑     ↑
Layer 1:  h₁₁ → h₁₂ → h₁₃ → ...
           ↑     ↑     ↑
Input:     x₁    x₂    x₃    ...
```

---

## 8. 总结与关键要点

### 核心概念

1. **循环连接**：RNN通过隐藏状态的自我连接实现序列建模
2. **参数共享**：同一组参数在所有时间步使用
3. **BPTT**：反向传播通过时间展开进行
4. **梯度问题**：长序列导致梯度消失或爆炸

### 数学要点

| 概念 | 公式 | 关键点 |
|------|------|--------|
| 前向传播 | $h_t = \tanh(W_{hh}h_{t-1} + W_{xh}x_t + b_h)$ | 隐藏状态更新 |
| 梯度传播 | 涉及 $W_{hh}$ 的连乘 | 梯度问题的根源 |
| 梯度消失 | $\lambda_{max} < 1$ | 长期依赖难学习 |
| 梯度爆炸 | $\lambda_{max} > 1$ | 数值不稳定 |

### 下一步学习

- **Stage 5**: LSTM - 解决梯度消失的门控机制
- **Stage 6**: GRU - 更简洁的门控方案
- **Stage 7**: Seq2Seq与注意力机制

---

## 参考资源

1. **论文**: 
   - "Learning representations by back-propagating errors" (Rumelhart et al., 1986)
   - "Backpropagation Through Time: What It Does and How to Do It" (Werbos, 1990)

2. **教材**:
   - 《深度学习》(Goodfellow et al.) - 第10章
   - 《神经网络与深度学习》(邱锡鹏)

3. **在线资源**:
   - Andrej Karpathy: "The Unreasonable Effectiveness of RNNs"
   - CS224n: Natural Language Processing with Deep Learning

---

*Created: 2026-03-16 | Stage 4 of LLM Learning Roadmap*
