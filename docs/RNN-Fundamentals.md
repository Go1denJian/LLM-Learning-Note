# RNN Fundamentals (循环神经网络基础)

> 学习目标：理解RNN的数学原理、训练机制，以及梯度问题的本质

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

**为什么使用 tanh？**

`tanh` 是一种激活函数，它将输入压缩到 (-1, 1) 区间。选择它的原因：
1. **输出范围对称**：输出在零附近对称，有利于梯度传播
2. **非线性**：引入非线性，使网络能学习复杂模式
3. **平滑可导**：便于梯度计算

> 注：也可以用 sigmoid 或 ReLU，但 tanh 在RNN中更常见，因为它的输出范围更适合表示"状态"

**输出计算：**

$$
y_t = W_{hy} h_t + b_y \quad \text{(或)} \quad \hat{y}_t = \text{softmax}(W_{hy} h_t + b_y)
$$

**符号说明：**
- $y_t$：原始输出（logits），未归一化的分数
- $\hat{y}_t$（读作"y-hat"）：预测概率分布，通过 softmax 将 logits 转换为概率

例如，在词预测任务中：
- $y_t$ = [2.5, -1.0, 0.3, ...] （每个词一个分数）
- $\hat{y}_t$ = [0.7, 0.05, 0.15, ...] （概率和为1）

**参数说明：**
| 符号 | 维度 | 含义 |
|------|------|------|
| $x_t$ | $(d_{in}, 1)$ | 时刻t的输入向量 |
| $h_t$ | $(d_{hidden}, 1)$ | 时刻t的隐藏状态 |
| $y_t$ | $(d_{out}, 1)$ | 时刻t的原始输出（logits） |
| $\hat{y}_t$ | $(d_{out}, 1)$ | 时刻t的预测概率分布 |
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

**具体例子：交叉熵损失（用于分类任务）**

假设我们在做词预测，词汇表大小为 V，真实标签是 one-hot 向量 $y_t$（只有正确词的位置为1，其余为0）：

$$
\mathcal{L}_t = -\sum_{i=1}^{V} y_{t,i} \cdot \log(\hat{y}_{t,i})
$$

由于 $y_t$ 是 one-hot，只有一个位置为1（假设是位置 k）：

$$
\mathcal{L}_t = -\log(\hat{y}_{t,k})
$$

**直观理解：** 模型对正确答案给出的概率越高，损失越小。

**数值示例：**
```
真实词: "cat" (位置 k=5)
预测概率: ŷ = [0.1, 0.05, 0.03, 0.02, 0.6, 0.2, ...]

损失: L = -log(0.6) ≈ 0.51

如果预测更准: ŷ = [0.01, 0.01, 0.01, 0.01, 0.9, 0.05, ...]
损失: L = -log(0.9) ≈ 0.105 （更小，更好）
```

### 3.2 梯度计算

**输出层梯度：**

对于交叉熵损失 + softmax 输出，梯度有简洁形式：

$$
\frac{\partial \mathcal{L}_t}{\partial W_{hy}} = (\hat{y}_t - y_t) \cdot h_t^T
$$

其中 $(\hat{y}_t - y_t)$ 是预测与真实的误差向量。

**隐藏层梯度（关键！）**

对于时刻t的隐藏状态，梯度来自两部分：
1. 当前时刻的直接损失
2. 未来时刻通过 $h_t$ 传播的梯度

$$
\frac{\partial \mathcal{L}}{\partial h_t} = \underbrace{W_{hy}^T \cdot (\hat{y}_t - y_t)}_{\text{当前时刻}} + \underbrace{W_{hh}^T \cdot \frac{\partial \mathcal{L}}{\partial h_{t+1}} \odot (1 - \tanh^2(h_{t+1}))}_{\text{未来时刻传播}}
$$

**符号 $\odot$ 说明：**

$\odot$ 表示**逐元素乘法**（Hadamard积），两个相同维度的向量/矩阵对应位置相乘。

例如：
```
a = [1, 2, 3]
b = [4, 5, 6]
a ⊙ b = [1×4, 2×5, 3×6] = [4, 10, 18]
```

为什么这里用逐元素乘法？因为 tanh 的导数是 $1 - \tanh^2(x)$，需要对每个隐藏单元单独计算。

### 3.3 梯度展开详解

让我们逐步展开从时刻 T 到时刻 t 的梯度传播：

**时刻 T（最后一个时刻）：**

$$
\frac{\partial \mathcal{L}}{\partial h_T} = W_{hy}^T \cdot (\hat{y}_T - y_T)
$$

只有当前时刻的损失，没有未来时刻。

**时刻 T-1：**

$$
\frac{\partial \mathcal{L}}{\partial h_{T-1}} = W_{hy}^T \cdot (\hat{y}_{T-1} - y_{T-1}) + W_{hh}^T \cdot \frac{\partial \mathcal{L}}{\partial h_T} \odot (1 - \tanh^2(h_T))
$$

**时刻 T-2：**

$$
\frac{\partial \mathcal{L}}{\partial h_{T-2}} = W_{hy}^T \cdot (\hat{y}_{T-2} - y_{T-2}) + W_{hh}^T \cdot \frac{\partial \mathcal{L}}{\partial h_{T-1}} \odot (1 - \tanh^2(h_{T-1}))
$$

**一般形式（时刻 t）：**

将梯度从时刻 T 反向传播到时刻 t：

$$
\frac{\partial \mathcal{L}}{\partial h_t} = \sum_{k=t}^{T} \left( \prod_{j=t+1}^{k} W_{hh}^T \cdot \text{diag}(1 - \tanh^2(h_j)) \right) \cdot W_{hy}^T \cdot (\hat{y}_k - y_k)
$$

**核心观察：** 梯度计算涉及 $W_{hh}$ 的连乘！

**数值示例（梯度传播）：**

假设一个简单情况，T=3，我们要计算 $\frac{\partial \mathcal{L}}{\partial h_1}$：

```
路径1: h₁ → h₂ → h₃ → loss₃
路径2: h₁ → h₂ → loss₂
路径3: h₁ → loss₁

∂L/∂h₁ = [W_hhᵀ · diag(1-tanh²(h₂)) · W_hhᵀ · diag(1-tanh²(h₃)) · ∂L/∂h₃]  ← 路径1
       + [W_hhᵀ · diag(1-tanh²(h₂)) · ∂L/∂h₂]  ← 路径2
       + [∂L/∂h₁的直接部分]  ← 路径3
```

可以看到，从时刻1到时刻3，$W_{hh}^T$ 出现了两次连乘！

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

---

## 发展脉络与下一步学习

RNN虽然是序列建模的重要里程碑，但梯度消失问题严重限制了它捕捉长期依赖的能力。这一局限推动了研究者们探索更强大的架构：

**1997年**，Hochreiter 和 Schmidhuber 提出了 **LSTM**（Long Short-Term Memory），通过引入门控机制（输入门、遗忘门、输出门）和细胞状态，从根本上解决了梯度消失问题，使得模型能够有效学习跨越数百个时间步的依赖关系。

**2014年**，Cho 等人提出了 **GRU**（Gated Recurrent Unit），作为LSTM的简化变体，它将门控机制精简为更新门和重置门，在保持相近性能的同时减少了参数量，成为许多应用中的实用选择。

**2014-2015年**，随着LSTM和GRU的成熟，研究者们开始将它们应用于更复杂的任务。**Seq2Seq**（Sequence-to-Sequence）架构应运而生，它使用编码器-解码器结构处理输入输出长度不同的任务（如机器翻译）。然而，Seq2Seq在处理长序列时面临信息瓶颈——所有输入信息必须压缩到一个固定长度的向量中。

这一瓶颈催生了**注意力机制**（Attention Mechanism，2015年，Bahdanau et al.），它允许解码器在生成每个输出时"关注"输入序列的不同部分，彻底改变了序列建模的范式。注意力机制的成功最终导向了**Transformer**架构的诞生（2017年，"Attention Is All You Need"），完全摒弃了循环结构，仅通过自注意力机制就实现了并行化和长距离依赖建模，成为现代大语言模型的基础。

**建议学习路径：**
1. **LSTM-Deep-Dive.md** - 理解门控机制如何解决梯度消失
2. **GRU-and-Seq2Seq.md** - 学习简化门控和序列到序列建模
3. 注意力机制与Transformer（后续文档）

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

*Created: 2026-03-16 | RNN Fundamentals - LLM Learning Roadmap*
