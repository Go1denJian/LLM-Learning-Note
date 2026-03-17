# GRU 数学原理与 Seq2Seq 架构

> **前置知识**：RNN 基础、LSTM 门控机制
> **学习目标**：理解 GRU 的简化设计、Seq2Seq 架构、注意力机制原理

---

## 目录

1. [引言：为什么需要 GRU？](#1-引言为什么需要-gru)
2. [GRU 的数学表达](#2-gru-的数学表达)
3. [核心算法：门控机制](#3-核心算法门控机制)
4. [梯度推导与参数更新](#4-梯度推导与参数更新)
5. [训练优化方法总结](#5-训练优化方法总结)
6. [从数学到代码：完整实现](#6-从数学到代码完整实现)
7. [Seq2Seq 架构](#7-seq2seq-架构)
8. [注意力机制初步](#8-注意力机制初步)
9. [扩展阅读与实现](#扩展阅读与实现)
10. [参考资源](#参考资源)
附录：[符号表](#附录符号表)

---

## 1. 引言：为什么需要 GRU？

### 1.1 LSTM 的局限性

**问题**：
- 参数量大（约 4 倍于 RNN）
- 计算复杂度高
- 训练时间长

### 1.2 GRU 的设计目标

- 保持 LSTM 捕捉长期依赖的能力
- 减少参数量（约 3 倍于 RNN）
- 简化计算

---

## 2. GRU 的数学表达

### 2.1 门控机制

GRU 将 LSTM 的 4 个门合并为 2 个门：

**重置门（Reset Gate）**：
$$
r_t = \sigma(W_{xr} x_t + W_{hr} h_{t-1} + b_r)
$$

**更新门（Update Gate）**：
$$
z_t = \sigma(W_{xz} x_t + W_{hz} h_{t-1} + b_z)
$$

**候选隐藏状态**：
$$
\tilde{h}_t = \tanh(W_{xh} x_t + W_{hh} (r_t \odot h_{t-1}) + b_h)
$$

**隐藏状态更新**：
$$
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
$$

### 2.2 与 LSTM 的对比

| 特性 | LSTM | GRU |
|------|------|-----|
| 门数量 | 3个（输入/遗忘/输出） | 2个（重置/更新） |
| 参数量 | 约 4 倍 RNN | 约 3 倍 RNN |
| 细胞状态 | 有 | 无（合并到隐藏状态） |
| 输出门 | 有 | 无 |

---

## 3. 核心算法：门控机制

### 3.1 重置门的作用

**功能**：控制前一时刻隐藏状态的信息流入

**直觉**：
- $r_t \approx 0$：忽略历史，关注当前输入
- $r_t \approx 1$：保留历史，类似 RNN

### 3.2 更新门的作用

**功能**：控制新信息与旧信息的混合比例

**直觉**：
- $z_t \approx 0$：保持旧状态（$h_t \approx h_{t-1}$）
- $z_t \approx 1$：接受新状态（$h_t \approx \tilde{h}_t$）

---

## 4. 梯度推导与参数更新

### 4.1 输出层梯度

$$
\frac{\partial \mathcal{L}_t}{\partial W_{hy}} = (\hat{y}_t - y_t) \cdot h_t^T
$$

### 4.2 隐藏层梯度

**对 $h_t$ 的梯度**：

$$
\frac{\partial \mathcal{L}}{\partial h_t} = W_{hy}^T (\hat{y}_t - y_t) + \frac{\partial \mathcal{L}}{\partial h_{t+1}} \cdot \frac{\partial h_{t+1}}{\partial h_t}
$$

**其中**：

$$
\frac{\partial h_{t+1}}{\partial h_t} = (1 - z_{t+1}) + \frac{\partial \tilde{h}_{t+1}}{\partial h_t} \cdot z_{t+1}
$$

### 4.3 门控梯度

**更新门梯度**：

$$
\frac{\partial \mathcal{L}}{\partial z_t} = \frac{\partial \mathcal{L}}{\partial h_t} \odot (\tilde{h}_t - h_{t-1})
$$

**重置门梯度**：

$$
\frac{\partial \mathcal{L}}{\partial r_t} = \frac{\partial \mathcal{L}}{\partial \tilde{h}_t} \odot (W_{hh}^T \cdot h_{t-1}) \odot (1 - \tanh^2(\cdot))
$$

---

## 5. 训练优化方法总结

### 5.1 梯度问题

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| 梯度消失 | 更新门 $z_t \approx 0$ | 门控初始化优化 |
| 梯度爆炸 | 连乘效应 | 梯度裁剪 |

### 5.2 优化策略

- **门控初始化**：使用较小初始值（如 0.1）
- **梯度裁剪**：限制梯度范数
- **学习率调整**：使用 Adam 优化器

---

## 6. 从数学到代码：完整实现

### 6.1 NumPy 实现

```python
import numpy as np

class GRU:
    def __init__(self, input_size, hidden_size, output_size):
        # 重置门参数
        self.W_xr = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hr = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_r = np.zeros((hidden_size, 1))
        
        # 更新门参数
        self.W_xz = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hz = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_z = np.zeros((hidden_size, 1))
        
        # 候选状态参数
        self.W_xh = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_h = np.zeros((hidden_size, 1))
        
        # 输出层参数
        self.W_hy = np.random.randn(output_size, hidden_size) * 0.01
        self.b_y = np.zeros((output_size, 1))
        
        self.hidden_size = hidden_size
    
    def forward(self, x, h_prev):
        """前向传播"""
        # 重置门
        r = 1 / (1 + np.exp(-(self.W_xr @ x + self.W_hr @ h_prev + self.b_r)))
        
        # 更新门
        z = 1 / (1 + np.exp(-(self.W_xz @ x + self.W_hz @ h_prev + self.b_z)))
        
        # 候选隐藏状态
        h_tilde = np.tanh(self.W_xh @ x + self.W_hh @ (r * h_prev) + self.b_h)
        
        # 隐藏状态
        h = (1 - z) * h_prev + z * h_tilde
        
        # 输出
        y = self.W_hy @ h + self.b_y
        
        return y, h, (r, z, h_tilde, h_prev, x)
    
    def backward(self, dy, dh_next, cache):
        """反向传播"""
        r, z, h_tilde, h_prev, x = cache
        
        # 输出层梯度
        dW_hy = dy @ h_prev.T
        db_y = dy
        dh = self.W_hy.T @ dy + dh_next
        
        # 门控梯度
        dz = dh * (h_tilde - h_prev)
        dh_tilde = dh * z
        dh_prev = dh * (1 - z)
        
        # 候选状态梯度
        dh_tilde_raw = dh_tilde * (1 - h_tilde ** 2)
        dW_xh = dh_tilde_raw @ x.T
        dW_hh = dh_tilde_raw @ (r * h_prev).T
        
        return dW_hy, db_y, dW_xh, dW_hh
```

### 6.2 PyTorch 实现

```python
import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, hidden = self.gru(x)
        out = self.fc(out)
        return out, hidden

# 使用示例
model = GRUModel(input_size=100, hidden_size=128, output_size=10000)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

---

## 7. Seq2Seq 架构

### 7.1 Encoder-Decoder 结构

**编码器（Encoder）**：
- 将输入序列压缩为上下文向量
- 使用 RNN/GRU/LSTM

**解码器（Decoder）**：
- 根据上下文向量生成输出序列
- 使用 RNN/GRU/LSTM

```
输入序列: [x₁, x₂, x₃] → Encoder → 上下文向量 c → Decoder → [y₁, y₂, y₃, y₄]
```

### 7.2 Teacher Forcing

**训练时**：使用真实标签作为下一时刻输入
**推理时**：使用模型预测作为下一时刻输入

---

## 8. 注意力机制初步

### 8.1 Seq2Seq 的局限性

**信息瓶颈**：所有信息压缩到固定长度向量

### 8.2 注意力机制的直觉

**核心思想**：解码器在生成每个词时"关注"编码器的不同部分

**注意力权重**：
$$
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_k \exp(e_{ik})}
$$

其中 $e_{ij}$ 是解码器状态 $s_i$ 与编码器状态 $h_j$ 的相似度。

---

## 9. 扩展阅读与实现

### 问题 1：GRU 与 LSTM 的参数对比

**问题**：计算 GRU 和 LSTM 的参数数量。

**解答**：

**LSTM 参数量**：
- 4 个门 × (输入权重 + 隐藏权重 + 偏置)
- $4 \times (d_{in} \cdot d_{hidden} + d_{hidden} \cdot d_{hidden} + d_{hidden})$

**GRU 参数量**：
- 3 个门 × (输入权重 + 隐藏权重 + 偏置)
- $3 \times (d_{in} \cdot d_{hidden} + d_{hidden} \cdot d_{hidden} + d_{hidden})$

**结论**：GRU 参数量约为 LSTM 的 75%。

---

### 问题 2：门控初始化对梯度的影响

**问题**：为什么门控需要较小的初始值？

**解答**：

**分析**：
- 初始化 $z_t \approx 0.5$
- 如果初始化过大，$z_t \approx 1$，网络退化为标准 RNN
- 如果初始化过小，$z_t \approx 0$，网络无法学习新信息

**工程建议**：
- 使用 $U(-0.1, 0.1)$ 均匀分布初始化
- 或使用 Xavier/Glorot 初始化

---

### 问题 3：注意力机制的计算复杂度

**问题**：分析 Seq2Seq + Attention 的计算复杂度。

**解答**：

**标准 Seq2Seq**：$O(T_{enc} + T_{dec})$

**Seq2Seq + Attention**：$O(T_{enc} \cdot T_{dec})$

**结论**：注意力机制增加了计算量，但解决了信息瓶颈问题。

---

## 10. 参考资源

### 经典论文

1. Cho, K., et al. (2014). [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078). EMNLP 2014.
   - **贡献**：GRU 和 Seq2Seq 的原始论文

2. Chung, J., et al. (2014). [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](https://arxiv.org/abs/1412.3555). NIPS 2014 Workshop.
   - **贡献**：GRU 与 LSTM 的对比实验

3. Bahdanau, D., et al. (2015). [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473). ICLR 2015.
   - **贡献**：注意力机制的原始论文

### 在线资源

4. [PyTorch GRU Documentation](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html)
5. [Seq2Seq with Attention Tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)

---

## 附录：符号表

| 符号 | 含义 | 维度 |
|------|------|------|
| $x_t$ | 时刻 $t$ 的输入 | $(d_{in}, 1)$ |
| $h_t$ | 时刻 $t$ 的隐藏状态 | $(d_{hidden}, 1)$ |
| $\tilde{h}_t$ | 候选隐藏状态 | $(d_{hidden}, 1)$ |
| $r_t$ | 重置门输出 | $(d_{hidden}, 1)$ |
| $z_t$ | 更新门输出 | $(d_{hidden}, 1)$ |
| $W_{xr}, W_{hr}$ | 重置门权重 | $(d_{hidden}, d_{in})$, $(d_{hidden}, d_{hidden