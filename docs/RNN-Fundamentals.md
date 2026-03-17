# RNN Fundamentals (循环神经网络基础)

> 学习目标：理解RNN的数学原理、训练机制，以及梯度问题的本质

---

## 目录

1. [从MLP到RNN：为什么需要循环结构？](#1-从mlp到rnn为什么需要循环结构)
2. [RNN的数学表达](#2-rnn的数学表达)
3. [BPTT：随时间反向传播](#3-bptt随时间反向传播)
4. [梯度消失与梯度爆炸](#4-梯度消失与梯度爆炸)
5. [从数学到代码：RNN完整实现](#5-从数学到代码rnn完整实现)
6. [RNN的变体架构](#6-rnn的变体架构)
7. [总结与关键要点](#7-总结与关键要点)
8. [与其他模型的关系](#与其他模型的关系)
9. [扩展阅读与实现](#扩展阅读与实现)
10. [参考资源](#参考资源)
11. [附录：符号表](#附录符号表)

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

**参数说明：** 参见文档开头的[符号说明](#符号说明)部分。

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

#### 符号约定（重要）

为了清晰区分不同层次的损失，本文档使用以下约定：

| 符号 | 名称 | 定义 | 说明 |
|------|------|------|------|
| $\ell(\hat{y}, y)$ | **损失函数** | $\ell: \mathbb{R}^{d_{out}} \times \mathbb{R}^{d_{out}} \rightarrow \mathbb{R}$ | 函数本身，输入为预测值和真实值 |
| $\mathcal{L}_t$ | **时刻 $t$ 的局部损失** | $\mathcal{L}_t = \ell(\hat{y}_t, y_t)$ | 第 $t$ 个时间步的损失**标量值** |
| $\mathcal{L}_{\text{seq}}$ | **序列总损失** | $\mathcal{L}_{\text{seq}} = \sum_{t=1}^{T} \mathcal{L}_t$ | 单个序列所有时刻损失之和 |

**关键理解**：
- $\ell(\cdot, \cdot)$ 是**函数**（映射规则）
- $\mathcal{L}_t$ 和 $\mathcal{L}_{\text{seq}}$ 是**标量值**（具体数值）
- 下角标 $t$ 表示**时间步位置**，不是函数参数

#### 损失函数的显式定义

以交叉熵损失为例，损失函数的完整定义为：

$$
\ell(\hat{y}, y) = -\sum_{i=1}^{d_{out}} y_i \log(\hat{y}_i)
$$

其中：
- 输入 $\hat{y} \in \mathbb{R}^{d_{out}}$：模型预测的概率分布（softmax输出）
- 输入 $y \in \mathbb{R}^{d_{out}}$：真实标签（one-hot向量）
- 输出：标量损失值

#### 各时刻局部损失

对于序列中的第 $t$ 个时间步：

$$
\mathcal{L}_t = \ell(\hat{y}_t, y_t) = -\sum_{i=1}^{d_{out}} y_{t,i} \log(\hat{y}_{t,i})
$$

由于 $y_t$ 是 one-hot（只有正确类别位置为1）：

$$
\mathcal{L}_t = -\log(\hat{y}_{t, k}) \quad \text{（其中 $k$ 是正确类别索引）}
$$

#### 序列总损失

$$
\mathcal{L}_{\text{seq}} = \sum_{t=1}^{T} \mathcal{L}_t = \sum_{t=1}^{T} \ell(\hat{y}_t, y_t)
$$

**注意**：在后续推导中，为简洁起见，我们用 $\mathcal{L}$ 表示 $\mathcal{L}_{\text{seq}}$（序列总损失）。

<!-- 
为什么叫"交叉熵"？

交叉熵（Cross-Entropy）源于信息论，衡量两个概率分布之间的差异：
- $y_t$：真实分布（目标分布）
- $\hat{y}_t$：模型预测的分布

公式 $H(y, \hat{y}) = -\sum_i y_i \log(\hat{y}_i)$ 表示：用模型分布 $\hat{y}$ 来编码真实分布 $y$ 所需的平均比特数。

当两个分布完全一致时，交叉熵等于真实分布的熵（最小值）；
当两个分布差异越大，交叉熵越大。

在分类任务中，由于 $y$ 是 one-hot（确定性的），交叉熵简化为 $-\log(\hat{y}_{\text{正确}})$，即负对数似然。
-->

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

### 3.3 BPTT 梯度推导详解

BPTT（Backpropagation Through Time）是RNN训练的核心算法。让我们从数学上详细推导梯度计算过程。

#### 3.3.1 问题设定

**前向传播回顾：**

$$
h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h) \\
\hat{y}_t = \text{softmax}(W_{hy} h_t + b_y)
$$

**损失函数（交叉熵）：**

$$
\mathcal{L}_t = -\sum_{i=1}^{d_{out}} y_{t,i} \log(\hat{y}_{t,i})
$$

#### 3.3.2 输出层梯度推导

**步骤 1：Softmax + 交叉熵的梯度**

对于Softmax输出和交叉熵损失，梯度有简洁形式：

$$
\frac{\partial \mathcal{L}_t}{\partial z_t} = \hat{y}_t - y_t
$$

其中 $z_t = W_{hy} h_t + b_y$ 是softmax前的logits。

**推导过程：**

Softmax定义：
$$
\hat{y}_{t,i} = \frac{e^{z_{t,i}}}{\sum_j e^{z_{t,j}}}
$$

交叉熵损失：
$$
\mathcal{L}_t = -\sum_i y_{t,i} \log(\hat{y}_{t,i})
$$

对 $z_{t,k}$ 求导：

$$
\frac{\partial \mathcal{L}_t}{\partial z_{t,k}} = -\sum_i y_{t,i} \frac{\partial \log(\hat{y}_{t,i})}{\partial z_{t,k}}
$$

分两种情况：

**情况 A：$i = k$**

$$
\frac{\partial \hat{y}_{t,k}}{\partial z_{t,k}} = \hat{y}_{t,k}(1 - \hat{y}_{t,k})
$$

**情况 B：$i \neq k$**

$$
\frac{\partial \hat{y}_{t,i}}{\partial z_{t,k}} = -\hat{y}_{t,i} \hat{y}_{t,k}
$$

**综合：**

$$
\begin{aligned}
\frac{\partial \mathcal{L}_t}{\partial z_{t,k}} &= -y_{t,k} \cdot \frac{1}{\hat{y}_{t,k}} \cdot \hat{y}_{t,k}(1 - \hat{y}_{t,k}) - \sum_{i \neq k} y_{t,i} \cdot \frac{1}{\hat{y}_{t,i}} \cdot (-\hat{y}_{t,i}\hat{y}_{t,k}) \\
&= -y_{t,k}(1 - \hat{y}_{t,k}) + \sum_{i \neq k} y_{t,i} \hat{y}_{t,k} \\
&= -y_{t,k} + y_{t,k}\hat{y}_{t,k} + \sum_{i \neq k} y_{t,i} \hat{y}_{t,k} \\
&= -y_{t,k} + \hat{y}_{t,k} \sum_i y_{t,i} \\
&= \hat{y}_{t,k} - y_{t,k} \quad (\text{因为 } \sum_i y_{t,i} = 1)
\end{aligned}
$$

**向量形式：**

$$
\frac{\partial \mathcal{L}_t}{\partial z_t} = \hat{y}_t - y_t \in \mathbb{R}^{d_{out}}
$$

**步骤 2：对 $W_{hy}$ 和 $b_y$ 的梯度**

由 $z_t = W_{hy} h_t + b_y$：

$$
\frac{\partial \mathcal{L}_t}{\partial W_{hy}} = (\hat{y}_t - y_t) h_t^T \in \mathbb{R}^{d_{out} \times d_{hidden}}
$$

$$
\frac{\partial \mathcal{L}_t}{\partial b_y} = \hat{y}_t - y_t \in \mathbb{R}^{d_{out}}
$$

#### 3.3.3 隐藏层梯度推导（核心）

**关键挑战：** 隐藏状态 $h_t$ 影响损失的两条路径：

1. **直接路径**：$h_t \rightarrow z_t \rightarrow \mathcal{L}_t$（当前时刻损失）
2. **间接路径**：$h_t \rightarrow h_{t+1} \rightarrow \ldots \rightarrow \mathcal{L}_{t+1}, \ldots$（未来时刻损失）

**步骤 1：对 $h_t$ 的梯度**

$$
\frac{\partial \mathcal{L}_{\text{seq}}}{\partial h_t} = \underbrace{\frac{\partial \mathcal{L}_t}{\partial h_t}}_{\text{直接梯度}} + \underbrace{\frac{\partial \mathcal{L}_{\text{seq}}}{\partial h_{t+1}} \cdot\frac{\partial h_{t+1}}{\partial h_t}}_{\text{传播梯度}}
$$

> **【讨论：关于梯度求导的深入理解】**
>
> **Q:** $\frac{\partial \mathcal{L}_t}{\partial h_t}$ 和 $\frac{\partial \mathcal{L}_{t+N}}{\partial h_{t+N}}$ 有什么区别？
>
> **A:** 从函数形式上看，它们完全相同！
>
> > $\mathcal{L}_t(h_t) = \ell(\text{softmax}(W_{hy} h_t + b_y), y_t)$  
> > $\mathcal{L}_{t+N}(h_{t+N}) = \ell(\text{softmax}(W_{hy} h_{t+N} + b_y), y_{t+N})$
>
> 两者都是同一个损失函数 $\ell$ 与 softmax、线性变换的复合，只是输入的隐藏状态和真实标签不同。
>
> 求导结果的形式完全相同：
> - $\frac{\partial \mathcal{L}_t}{\partial h_t} = W_{hy}^T (\hat{y}_t - y_t)$
> - $\frac{\partial \mathcal{L}_{t+N}}{\partial h_{t+N}} = W_{hy}^T (\hat{y}_{t+N} - y_{t+N})$
>
> 角标 $t$ 和 $t+N$ 只是标记"在哪个时刻求值"，不改变函数本身的结构。
>
> ---
>
> **Q:** 那为什么 $\frac{\partial \mathcal{L}_{t+1}}{\partial h_t}$ 和 $\frac{\partial \mathcal{L}_{t+1}}{\partial h_{t+1}}$ 不同？
>
> **A:** 关键区别在于"对谁求导"！
>
> $\frac{\partial \mathcal{L}_{t+1}}{\partial h_{t+1}}$：直接对 $h_{t+1}$ 求导  
> $= W_{hy}^T (\hat{y}_{t+1} - y_{t+1})$ （标准形式）
>
> $\frac{\partial \mathcal{L}_{t+1}}{\partial h_t}$：对 $h_t$ 求导，需要通过链式法则  
> $= \frac{\partial \mathcal{L}_{t+1}}{\partial h_{t+1}} \cdot \frac{\partial h_{t+1}}{\partial h_t}$
>
> 多了一个雅可比矩阵 $\frac{\partial h_{t+1}}{\partial h_t} = \text{diag}(1-\tanh^2) \cdot W_{hh}$，这就是梯度在时间上递归传播的数学本质。
>
> ---
>
> **Q:** 为什么公式中只显式出现 $h_{t+1}$，而没有 $h_{t+2}, h_{t+3}, ...$？
>
> **A:** 这是递归定义的巧妙之处，既是数学恒等变形，也有实际计算意义。
>
> 总梯度展开：  
> $\frac{\partial \mathcal{L}_{\text{seq}}}{\partial h_t} = \frac{\partial \mathcal{L}_t}{\partial h_t} + \frac{\partial \mathcal{L}_{t+1}}{\partial h_t} + \frac{\partial \mathcal{L}_{t+2}}{\partial h_t} + ... + \frac{\partial \mathcal{L}_T}{\partial h_t}$
>
> 重新组合：  
> $\frac{\partial \mathcal{L}_{\text{seq}}}{\partial h_t} = \frac{\partial \mathcal{L}_t}{\partial h_t} + \underbrace{\left( \frac{\partial \mathcal{L}_{t+1}}{\partial h_{t+1}} + \frac{\partial \mathcal{L}_{t+2}}{\partial h_{t+1}} + ... \right)}_{= \frac{\partial \mathcal{L}_{\text{seq}}}{\partial h_{t+1}}} \cdot \frac{\partial h_{t+1}}{\partial h_t}$
>
> 括号里的正是 $\frac{\partial \mathcal{L}_{\text{seq}}}{\partial h_{t+1}}$，它已经包含了从 $t+1$ 到 $T$ 的所有梯度信息。
>
> **计算优势**：从最后一个时刻开始反向计算，每一步只需计算两项，时间复杂度 $O(T)$ 而非 $O(T^2)$。
>
> ---
>
> **类比**：多米诺骨牌
> - $h_t$ 是第一块骨牌
> - 推倒第一块会连锁推倒第二块、第三块...
> - 我们只需要知道"第一块如何推第二块"（$\partial h_{t+1}/\partial h_t$）
> - "第二块及以后的影响"已经包含在 $\partial \mathcal{L}_{\text{seq}}/\partial h_{t+1}$ 中

**直接梯度：**

由 $z_t = W_{hy} h_t + b_y$：

$$
\frac{\partial \mathcal{L}_t}{\partial h_t} = W_{hy}^T (\hat{y}_t - y_t) \in \mathbb{R}^{d_{hidden}}
$$

**雅可比矩阵 $\frac{\partial h_{t+1}}{\partial h_t}$：**

由 $h_{t+1} = \tanh(W_{hh} h_t + W_{xh} x_{t+1} + b_h)$：

令 $a_{t+1} = W_{hh} h_t + W_{xh} x_{t+1} + b_h$，则 $h_{t+1} = \tanh(a_{t+1})$

$$
\frac{\partial h_{t+1}}{\partial h_t} = \text{diag}(1 - \tanh^2(a_{t+1})) \cdot W_{hh} \in \mathbb{R}^{d_{hidden} \times d_{hidden}}
$$

其中 $\text{diag}(1 - \tanh^2(a_{t+1}))$ 是对角矩阵，对角线元素为 $1 - \tanh^2(a_{t+1,i})$。

**综合：**

$$
\frac{\partial \mathcal{L}}{\partial h_t} = W_{hy}^T (\hat{y}_t - y_t) + W_{hh}^T \cdot \text{diag}(1 - \tanh^2(a_{t+1})) \cdot \frac{\partial \mathcal{L}}{\partial h_{t+1}}
$$

**步骤 2：对 $W_{hh}$ 和 $W_{xh}$ 的梯度**

由 $h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$：

$$
\frac{\partial \mathcal{L}}{\partial W_{hh}} = \sum_{t=1}^T \frac{\partial \mathcal{L}}{\partial h_t} \odot (1 - \tanh^2(a_t)) \cdot h_{t-1}^T
$$

$$
\frac{\partial \mathcal{L}}{\partial W_{xh}} = \sum_{t=1}^T \frac{\partial \mathcal{L}}{\partial h_t} \odot (1 - \tanh^2(a_t)) \cdot x_t^T
$$

$$
\frac{\partial \mathcal{L}}{\partial b_h} = \sum_{t=1}^T \frac{\partial \mathcal{L}}{\partial h_t} \odot (1 - \tanh^2(a_t))
$$

其中 $\odot$ 表示逐元素乘法。

#### 3.3.4 梯度展开与连乘问题

让我们逐步展开从时刻 $T$ 到时刻 $t$ 的梯度传播：

**时刻 $T$（最后一个时刻）：**

$$
\frac{\partial \mathcal{L}}{\partial h_T} = W_{hy}^T (\hat{y}_T - y_T)
$$

只有当前时刻的损失，没有未来时刻。

**时刻 $T-1$：**

$$
\frac{\partial \mathcal{L}}{\partial h_{T-1}} = W_{hy}^T (\hat{y}_{T-1} - y_{T-1}) + W_{hh}^T \cdot \text{diag}(1 - \tanh^2(a_T)) \cdot \frac{\partial \mathcal{L}}{\partial h_T}
$$

**时刻 $T-2$：**

$$
\frac{\partial \mathcal{L}}{\partial h_{T-2}} = W_{hy}^T (\hat{y}_{T-2} - y_{T-2}) + W_{hh}^T \cdot \text{diag}(1 - \tanh^2(a_{T-1})) \cdot \frac{\partial \mathcal{L}}{\partial h_{T-1}}
$$

**一般形式（时刻 $t$）：**

将梯度从时刻 $T$ 反向传播到时刻 $t$：

$$
\frac{\partial \mathcal{L}}{\partial h_t} = \sum_{k=t}^{T} \left( \prod_{j=t+1}^{k} W_{hh}^T \cdot \text{diag}(1 - \tanh^2(a_j)) \right) \cdot W_{hy}^T (\hat{y}_k - y_k)
$$

**核心观察：** 梯度计算涉及 $W_{hh}$ 的连乘！

#### 3.3.5 数值示例（梯度传播）

假设一个简单情况，$T=3$，我们要计算 $\frac{\partial \mathcal{L}}{\partial h_1}$：

**路径分解：**

```
路径1: h₁ → h₂ → h₃ → loss₃
路径2: h₁ → h₂ → loss₂
路径3: h₁ → loss₁
```

**梯度计算：**

$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial h_1} &= \underbrace{W_{hy}^T (\hat{y}_1 - y_1)}_{\text{路径3：直接损失}} \\
&+ \underbrace{W_{hh}^T \cdot D_2 \cdot W_{hy}^T (\hat{y}_2 - y_2)}_{\text{路径2：经h₂传播}} \\
&+ \underbrace{W_{hh}^T \cdot D_2 \cdot W_{hh}^T \cdot D_3 \cdot W_{hy}^T (\hat{y}_3 - y_3)}_{\text{路径1：经h₂→h₃传播}}
\end{aligned}
$$

其中 $D_t = \text{diag}(1 - \tanh^2(a_t))$。

**关键问题：** 从时刻1到时刻3，$W_{hh}^T$ 出现了**两次连乘**！

对于长度为 $T$ 的序列，从时刻1到时刻 $T$ 的梯度传播涉及 $W_{hh}^T$ 的 $(T-1)$ 次连乘：

$$
\frac{\partial \mathcal{L}}{\partial h_1} \propto (W_{hh}^T)^{T-1}
$$

这就是**梯度消失/爆炸**的数学根源！

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

## 5. 从数学到代码：RNN完整实现

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

## 6. RNN的变体架构

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

## 7. 总结与关键要点

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

## 与其他模型的关系

RNN虽然是序列建模的重要里程碑，但梯度消失问题严重限制了它捕捉长期依赖的能力。这一局限推动了研究者们探索更强大的架构：

RNN的梯度问题推动了更强大架构的发展：

| 时间 | 架构 | 核心贡献 |
|------|------|---------|
| 1997 | **LSTM** | 门控机制解决梯度消失 |
| 2014 | **GRU** | 简化门控，参数更少 |
| 2014-15 | **Seq2Seq** | 编码器-解码器架构 |
| 2015 | **Attention** | 选择性关注输入部分 |
| 2017 | **Transformer** | 完全并行，自注意力机制 |

**建议学习路径：**
1. **LSTM-Deep-Dive.md** - 门控机制详解
2. **GRU-and-Seq2Seq.md** - 序列到序列建模
3. **Transformer** - 现代大语言模型基础

---

## 扩展阅读与实现

本章提供详细的数学推导、深度分析和工程实践问题，帮助巩固前面学习的知识。

### 问题 1：BPTT 梯度完整推导

**问题**：给定RNN的前向传播公式：
$$
h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h) \\
\hat{y}_t = \text{softmax}(W_{hy} h_t + b_y)
$$

和交叉熵损失：
$$
\mathcal{L}_{\text{seq}} = -\sum_{t=1}^{T} \sum_{i=1}^{d_{out}} y_{t,i} \log(\hat{y}_{t,i})
$$

推导 $\frac{\partial \mathcal{L}_{\text{seq}}}{\partial W_{hh}}$ 的完整表达式。

**解答**：

**步骤 1：链式法则展开**

$$
\frac{\partial \mathcal{L}_{\text{seq}}}{\partial W_{hh}} = \sum_{t=1}^{T} \frac{\partial \mathcal{L}_{\text{seq}}}{\partial h_t} \cdot \frac{\partial h_t}{\partial W_{hh}}
$$

**步骤 2：计算 $\frac{\partial h_t}{\partial W_{hh}}$**

由于 $h_t = \tanh(a_t)$，其中 $a_t = W_{hh} h_{t-1} + W_{xh} x_t + b_h$：

$$
\frac{\partial h_t}{\partial W_{hh}} = \text{diag}(1 - \tanh^2(a_t)) \cdot \left( h_{t-1} + W_{hh} \frac{\partial h_{t-1}}{\partial W_{hh}} \right)
$$

这是一个递归关系！

**步骤 3：展开递归**

从 $t=1$ 到 $T$ 展开：

$$
\frac{\partial h_1}{\partial W_{hh}} = \text{diag}(1-\tanh^2(a_1)) \cdot h_0
$$

$$
\frac{\partial h_2}{\partial W_{hh}} = \text{diag}(1-\tanh^2(a_2)) \cdot \left( h_1 + W_{hh} \frac{\partial h_1}{\partial W_{hh}} \right)
$$

$$
\vdots
$$

**步骤 4：综合结果**

$$
\boxed{
\frac{\partial \mathcal{L}_{\text{seq}}}{\partial W_{hh}} = \sum_{t=1}^{T} \frac{\partial \mathcal{L}_{\text{seq}}}{\partial h_t} \odot (1 - \tanh^2(a_t)) \cdot h_{t-1}^T
}
$$

其中 $\frac{\partial \mathcal{L}_{\text{seq}}}{\partial h_t}$ 通过BPTT递归计算。

---

### 问题 2：梯度消失问题的数学分析

**问题**：证明当 $W_{hh}$ 的最大奇异值 $\lambda_{max} < 1$ 时，长序列会出现梯度消失。

**解答**：

**步骤 1：梯度传播的连乘形式**

从时刻 $t$ 传播到时刻 $1$ 的梯度：

$$
\frac{\partial \mathcal{L}_t}{\partial h_1} = \frac{\partial \mathcal{L}_t}{\partial h_t} \cdot \prod_{i=2}^{t} \frac{\partial h_i}{\partial h_{i-1}}
$$

**步骤 2：分析雅可比矩阵**

$$
\frac{\partial h_i}{\partial h_{i-1}} = \text{diag}(1 - \tanh^2(a_i)) \cdot W_{hh}
$$

令 $\gamma = \max_i |1 - \tanh^2(a_i)| \leq 1$（因为 $\tanh$ 输出在 $(-1, 1)$）

**步骤 3：梯度范数估计**

$$
\left\| \frac{\partial \mathcal{L}_t}{\partial h_1} \right\| \leq \left\| \frac{\partial \mathcal{L}_t}{\partial h_t} \right\| \cdot (\gamma \cdot \lambda_{max})^{t-1}
$$

**步骤 4：结论**

当 $\lambda_{max} < 1$ 且 $\gamma \leq 1$：

$$
\lim_{t \to \infty} (\gamma \cdot \lambda_{max})^{t-1} = 0
$$

**梯度随序列长度指数衰减！**

---

### 问题 3：参数共享的梯度计算

**问题**：RNN在所有时间步共享参数 $W_{hh}$，求导时如何处理？

**解答**：

**关键洞察**：$W_{hh}$ 影响**所有时间步**的隐藏状态。

**步骤 1：展开依赖关系**

$W_{hh}$ 通过多条路径影响损失：

```
W_hh → h_1 → L_1
     → h_1 → h_2 → L_2
     → h_1 → h_2 → h_3 → L_3
     → ...
```

**步骤 2：总梯度是所有路径之和**

$$
\frac{\partial \mathcal{L}_{\text{seq}}}{\partial W_{hh}} = \sum_{t=1}^{T} \sum_{k=1}^{t} \frac{\partial \mathcal{L}_t}{\partial h_k} \cdot \frac{\partial h_k}{\partial W_{hh}}
$$

**步骤 3：重新组合**

按时间步重组：

$$
\frac{\partial \mathcal{L}_{\text{seq}}}{\partial W_{hh}} = \sum_{k=1}^{T} \left( \sum_{t=k}^{T} \frac{\partial \mathcal{L}_t}{\partial h_k} \right) \cdot \frac{\partial h_k}{\partial W_{hh}}
$$

括号内正是 $\frac{\partial \mathcal{L}_{\text{seq}}}{\partial h_k}$！

**步骤 4：最终公式**

$$
\boxed{
\frac{\partial \mathcal{L}_{\text{seq}}}{\partial W_{hh}} = \sum_{t=1}^{T} \frac{\partial \mathcal{L}_{\text{seq}}}{\partial h_t} \odot (1 - \tanh^2(a_t)) \cdot h_{t-1}^T
}
$$

**工程意义**：每个时间步的梯度都要累加到 $W_{hh}$，这是RNN训练比前馈网络慢的原因之一。

---

### 问题 4：梯度裁剪的数学原理

**问题**：为什么梯度裁剪能解决梯度爆炸问题？给出数学分析。

**解答**：

**步骤 1：梯度爆炸的条件**

当 $\lambda_{max} > 1$ 时：

$$
\left\| \frac{\partial \mathcal{L}_t}{\partial h_1} \right\| \approx \lambda_{max}^{t-1} \to \infty \quad \text{当 } t \to \infty
$$

**步骤 2：裁剪策略**

给定阈值 $\tau$，如果 $\|g\| > \tau$：

$$
g_{\text{clip}} = g \cdot \frac{\tau}{\|g\|}
$$

**步骤 3：数学性质**

- **保持方向**：$g_{\text{clip}}$ 与 $g$ 同方向
- **限制范数**：$\|g_{\text{clip}}\| = \tau$
- **避免溢出**：防止数值爆炸

**步骤 4：为什么有效**

即使原始梯度 $\|g\| = 10^6$，裁剪后：

$$
\|g_{\text{clip}}\| = \tau \quad \text{(如 } \tau = 5 \text{)}
$$

参数更新稳定可控！

---

### 问题 5：双向RNN的前向与反向传播

**问题**：推导双向RNN的梯度计算公式。

**解答**：

**步骤 1：前向传播**

双向RNN有两个独立的隐藏层：

**前向层**（Left-to-Right）：

$$
\vec{h}_t = \tanh(W_{\rightarrow} \vec{h}_{t-1} + W_{xh}^{\rightarrow} x_t + b_h^{\rightarrow})
$$

**反向层**（Right-to-Left）：

$$
\overleftarrow{h}_t = \tanh(W_{\leftarrow} \overleftarrow{h}_{t+1} + W_{xh}^{\leftarrow} x_t + b_h^{\leftarrow})
$$

**输出**：

$$
\hat{y}_t = \text{softmax}(W_{hy} [\vec{h}_t; \overleftarrow{h}_t] + b_y)
$$

**步骤 2：梯度计算**

**前向层梯度**（标准BPTT）：

$$
\frac{\partial \mathcal{L}}{\partial \vec{h}_t} = \frac{\partial \mathcal{L}_t}{\partial \vec{h}_t} + \frac{\partial \mathcal{L}}{\partial \vec{h}_{t+1}} \cdot \frac{\partial \vec{h}_{t+1}}{\partial \vec{h}_t}
$$

**反向层梯度**（反向BPTT）：

$$
\frac{\partial \mathcal{L}}{\partial \overleftarrow{h}_t} = \frac{\partial \mathcal{L}_t}{\partial \overleftarrow{h}_t} + \frac{\partial \mathcal{L}}{\partial \overleftarrow{h}_{t-1}} \cdot \frac{\partial \overleftarrow{h}_{t-1}}{\partial \overleftarrow{h}_t}
$$

**步骤 3：关键区别**

- 前向层：从 $t=T$ 到 $t=1$ 反向计算
- 反向层：从 $t=1$ 到 $t=T$ 正向计算（或理解为从右向左反向传播）

**步骤 4：参数更新**

两个方向的权重独立更新：

$$
W_{\rightarrow} \leftarrow W_{\rightarrow} - \eta \frac{\partial \mathcal{L}}{\partial W_{\rightarrow}}
$$

$$
W_{\leftarrow} \leftarrow W_{\leftarrow} - \eta \frac{\partial \mathcal{L}}{\partial W_{\leftarrow}}
$$

**工程注意**：需要遍历序列两次（前向+反向），计算量是单向RNN的两倍。

---

### 问题 6：RNN训练优化方法总结

**问题**：总结RNN训练中的主要优化方法。

**解答**：

**1. 梯度裁剪 (Gradient Clipping)**

解决梯度爆炸：

```python
if ||gradient|| > threshold:
    gradient = gradient * (threshold / ||gradient||)
```

**2. 更好的架构**

| 方法 | 核心思想 | 优点 |
|------|---------|------|
| **LSTM** | 门控机制（输入门、遗忘门、输出门）| 解决梯度消失 |
| **GRU** | 简化门控（更新门、重置门）| 参数更少，计算更快 |

**3. 正则化方法**

- **Dropout**：在隐藏层应用（需要特殊变体如Recurrent Dropout）
- **权重衰减**：L2正则化
- **早停**：防止过拟合

**4. 初始化策略**

- **Xavier/Glorot初始化**：保持梯度稳定
- **正交初始化**：$W_{hh}$使用正交矩阵

---

## 参考资源

### 经典论文

1. Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). [Learning representations by back-propagating errors](https://www.nature.com/articles/323533a0). Nature, 323(6088), 533-536.
   - **贡献**：反向传播算法的奠基性论文

2. Werbos, P. J. (1990). [Backpropagation Through Time: What It Does and How to Do It](https://ieeexplore.ieee.org/document/58337). Proceedings of the IEEE, 78(10), 1550-1560.
   - **贡献**：BPTT算法的完整描述

3. Elman, J. L. (1990). [Finding Structure in Time](https://crl.ucsd.edu/~elman/Papers/fsit.pdf). Cognitive Science, 14(2), 179-211.
   - **贡献**：Simple RNN（Elman网络）的提出

4. Hochreiter, S., & Schmidhuber, J. (1997). [Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf). Neural Computation, 9(8), 1735-1780.
   - **贡献**：LSTM的原始论文，解决梯度消失问题

5. Cho, K., et al. (2014). [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078). EMNLP 2014.
   - **贡献**：GRU和Seq2Seq架构的提出

### 教材与书籍

6. Goodfellow, I., Bengio, Y., & Courville, A. (2016). [Deep Learning](https://www.deeplearningbook.org/) - Chapter 10: Sequence Modeling: Recurrent and Recursive Nets. MIT Press.
   - **章节**：第10章详细讲解RNN和BPTT

7. 邱锡鹏. (2020). [神经网络与深度学习](https://nndl.github.io/) - 第6章：循环神经网络. 
   - **章节**：第6章中文详解RNN原理

### 在线资源与教程

8. Karpathy, A. (2015). [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/).
   - **内容**：RNN的直观解释和char-rnn实现

9. Stanford CS224n: [Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/).
   - **内容**：NLP深度学习的经典课程，包含RNN/LSTM/Attention

10. Olah, C. (2015). [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/).
    - **内容**：LSTM和GRU的可视化解释

### 代码实现

11. [PyTorch RNN Documentation](https://pytorch.org/docs/stable/nn.html#rnn)
12. [TensorFlow RNN Tutorial](https://www.tensorflow.org/tutorials/text/text_classification_rnn)



## 附录：符号表

本文档使用以下符号约定：

| 符号 | 含义 | 维度 |
|------|------|------|
| $T$ | 序列长度 | 标量 |
| $t$ | 当前时刻 | 标量 |
| $x_t$ | 时刻 $t$ 的输入向量 | $(d_{in}, 1)$ |
| $h_t$ | 时刻 $t$ 的隐藏状态 | $(d_{hidden}, 1)$ |
| $y_t$ | 时刻 $t$ 的真实标签 | $(d_{out}, 1)$ |
| $\hat{y}_t$ | 时刻 $t$ 的预测输出 | $(d_{out}, 1)$ |
| $z_t$ | softmax前的logits | $(d_{out}, 1)$ |
| $W_{xh}$ | 输入到隐藏的权重 | $(d_{hidden}, d_{in})$ |
| $W_{hh}$ | 隐藏到隐藏的权重（循环权重）| $(d_{hidden}, d_{hidden})$ |
| $W_{hy}$ | 隐藏到输出的权重 | $(d_{out}, d_{hidden})$ |
| $b_h, b_y$ | 偏置项 | - |
| $\mathcal{L}_t$ | 时刻 $t$ 的局部损失 | 标量 |
| $\mathcal{L}_{\text{seq}}$ | 序列总损失 | 标量 |
| $\ell(\cdot, \cdot)$ | 损失函数 | 函数 |
| $\odot$ | 逐元素乘法（Hadamard积）| - |
| $\sigma$ | sigmoid函数 | - |
| $\tanh$ | 双曲正切函数 | - |
| $\lambda_{max}$ | 最大奇异值 | 标量 |

**典型维度示例：**
- $d_{in} = 300$（词嵌入维度）
- $d_{hidden} = 128$（隐藏状态维度）
- $d_{out} = 10,000$（词汇表大小）

---

*Created: 2026-03-16 | RNN Fundamentals - LLM Learning Roadmap*
