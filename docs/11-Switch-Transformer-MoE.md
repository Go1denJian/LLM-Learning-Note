# Switch Transformer 数学原理与实现 —— 稀疏激活 MoE 的完整推导

> **前置知识**：Transformer 架构、FFN 层、Softmax 路由、交叉熵损失、Python 基础  
> **与前面内容的联系**：建议先学习 [Transformer-Math-and-Implementation](./06-Transformer-Math-and-Implementation.md) 和 [GPT3-Scaling-and-InContext](./10-GPT3-Scaling-and-InContext.md)，理解标准 FFN 层和大规模模型训练  
> **与后续内容的联系**：Switch Transformer 的 MoE 稀疏激活思想直接影响了后续 GLaM、GShard 以及 DeepSeek-MoE 等模型的设计

---

## 目录

1. [引言：为什么需要稀疏激活？](#1-引言为什么需要稀疏激活)
   - 1.1 [Dense 模型的计算瓶颈](#11-dense-模型的计算瓶颈)
   - 1.2 [Mixture of Experts 的核心洞察](#12-mixture-of-experts-的核心洞察)
   - 1.3 [Switch Transformer 的关键创新](#13-switch-transformer-的关键创新)
   - 1.4 [本科数学知识映射表](#14-本科数学知识映射表)
2. [MoE 基础：从 Dense 到 Sparse](#2-moe-基础从-dense-到-sparse)
   - 2.1 [标准 FFN 回顾](#21-标准-ffn-回顾)
   - 2.2 [经典 MoE 的数学定义](#22-经典-moe-的数学定义)
   - 2.3 [稀疏激活：Top-k 路由](#23-稀疏激活top-k-路由)
   - 2.4 [Switch Routing：极简 Top-1](#24-switch-routingtop-1)
3. [Switch Transformer 架构的数学描述](#3-switch-transformer-架构的数学描述)
   - 3.1 [整体架构总览](#31-整体架构总览)
   - 3.2 [Switch FFN 层的前向传播](#32-switch-ffn-层的前向传播)
   - 3.3 [路由器的数学定义](#33-路由器的数学定义)
   - 3.4 [参数量与 FLOPs 分析](#34-参数量与-flops-分析)
4. [负载均衡损失与训练稳定性](#4-负载均衡损失与训练稳定性)
   - 4.1 [负载不均衡问题](#41-负载不均衡问题)
   - 4.2 [辅助负载均衡损失](#42-辅助负载均衡损失)
   - 4.3 [专家容量因子与溢出处理](#43-专家容量因子与溢出处理)
   - 4.4 [梯度推导：路由器参数更新](#44-梯度推导路由器参数更新)
5. [训练优化方法总结](#5-训练优化方法总结)
   - 5.1 [选择性精度训练](#51-选择性精度训练)
   - 5.2 [专家并行与通信](#52-专家并行与通信)
   - 5.3 [初始化与正则化策略](#53-初始化与正则化策略)
   - 5.4 [学习率调度与超参数](#54-学习率调度与超参数)
6. [从数学到代码：完整实现](#6-从数学到代码完整实现)
   - 6.1 [NumPy 实现核心组件](#61-numpy-实现核心组件)
   - 6.2 [PyTorch 完整实现](#62-pytorch-完整实现)
7. [MoE 可视化与实践技巧](#7-moe-可视化与实践技巧)
   - 7.1 [路由分布可视化](#71-路由分布可视化)
   - 7.2 [实践调参建议](#72-实践调参建议)
8. [与其他模型的关系](#8-与其他模型的关系)
   - 8.1 [从 Dense Transformer 到 MoE Transformer](#81-从-dense-transformer-到-moe-transformer)
   - 8.2 [Switch Transformer 在大模型发展中的定位](#82-switch-transformer-在大模型发展中的定位)
   - 8.3 [MoE 的后续发展](#83-moe-的后续发展)

[扩展阅读与实现](#扩展阅读与实现)

[参考资源](#参考资源)

附录：[符号表](#附录符号表)

---

## 1. 引言：为什么需要稀疏激活？

### 1.1 Dense 模型的计算瓶颈

在 GPT-3 之后，语言模型的性能与参数量之间的 Scaling Laws 已经确立：

$$
\mathcal{L}(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}
$$

但 Dense 模型面临一个根本矛盾：

| 问题 | 具体表现 |
|------|---------|
| **参数=计算** | Dense 模型中，增加参数必然增加每个 token 的计算量 |
| **计算预算有限** | 训练 175B 参数需要 $\sim 3.14 \times 10^{23}$ FLOPs |
| **推理延迟** | 每个 token 都必须经过全部参数 |

核心问题用数学表达：

$$
C_{\text{Dense}} = 6ND \quad \Rightarrow \quad N \uparrow \;\Longleftrightarrow\; C \uparrow
$$

**能否打破这个线性关系？** 即增加参数量 $N$ 的同时，保持每个 token 的计算量 $C_{\text{per-token}}$ 不变？

### 1.2 Mixture of Experts 的核心洞察

MoE 的核心思想来自一个简单的观察：

> **并非所有输入都需要所有参数。** 不同的输入可以由不同的"专家"子网络处理。

$$
\boxed{N_{\text{total}} \gg N_{\text{active}} \quad \Longrightarrow \quad \text{参数量与计算量解耦}}
$$

| 模型类型 | 总参数 $N_{\text{total}}$ | 活跃参数 $N_{\text{active}}$ | 每 token FLOPs |
|---------|:------------------------:|:---------------------------:|:--------------:|
| GPT-3 (Dense) | 175B | 175B | $\sim 350$ TFLOP |
| Switch-Base (MoE) | 7.4B | 0.2B | $\sim 0.4$ TFLOP |
| Switch-XXL (MoE) | 1.6T | 0.2B | $\sim 0.4$ TFLOP |

Switch Transformer 用 **1.6 万亿参数** 实现了与 Dense 模型相当的每 token 计算量，但模型容量增大了 **数倍**。

### 1.3 Switch Transformer 的关键创新

经典 MoE（Shazeer et al., 2017）使用 Top-2 路由：每个 token 发送到 2 个专家。Switch Transformer 的核心简化：

$$
\boxed{\text{Top-2 路由} \xrightarrow{\text{简化}} \text{Top-1 路由（Switch Routing）}}
$$

**三大创新**：

1. **Top-1 路由**：每个 token 只路由到 **1 个**专家，减少通信和计算
2. **简化的负载均衡损失**：辅助损失确保 token 均匀分配
3. **选择性精度训练**：路由器和门控使用 FP32，专家使用 BF16

### 1.4 本科数学知识映射表

| 数学概念 | Switch Transformer 中的应用 | 代码对应 |
|---------|--------------------------|---------|
| Softmax $\sigma(z_i)$ | 路由概率计算 | `F.softmax(router_logits)` |
| $\arg\max$ | Top-1 专家选择 | `torch.argmax(probs)` |
| 加权求和 $\sum_i w_i f_i(x)$ | 经典 MoE 输出 | `sum(w_i * expert_i(x))` |
| 均匀分布 $U(1/E, \ldots, 1/E)$ | 理想负载均衡目标 | `1.0 / num_experts` |
| 内积 $f_i \cdot P_i$ | 负载均衡损失 | `torch.dot(f, P)` |
| 辅助损失 $\alpha \cdot \mathcal{L}_{\text{aux}}$ | 正则化训练 | `loss += alpha * aux_loss` |
| 容量约束 $C = \lceil T/E \cdot c \rceil$ | 专家缓冲区大小 | `capacity = ceil(T/E * cf)` |
| 稀疏矩阵 | 路由分配矩阵 | `one_hot(expert_idx)` |

---

## 2. MoE 基础：从 Dense 到 Sparse

### 2.1 标准 FFN 回顾

在标准 Transformer 中，每个 token $x \in \mathbb{R}^{d}$ 经过 FFN 层：

$$
\text{FFN}(x) = W_2 \cdot \text{GELU}(W_1 x + b_1) + b_2
$$

其中 $W_1 \in \mathbb{R}^{d_{ff} \times d}$，$W_2 \in \mathbb{R}^{d \times d_{ff}}$，FFN 层通常占 Transformer 总参数的 **2/3**。

**关键观察**：研究表明 FFN 层的神经元激活是**稀疏**的——对于任意输入 $x$，只有少部分神经元被显著激活。

$$
\|\text{GELU}(W_1 x + b_1)\|_0 \ll d_{ff}
$$

这意味着大量参数在每次前向传播中是"浪费"的。

### 2.2 经典 MoE 的数学定义

Shazeer et al. (2017) 提出将单个 FFN 替换为 $E$ 个并行的"专家"FFN：

$$
\text{MoE}(x) = \sum_{i=1}^{E} g_i(x) \cdot \text{FFN}_i(x)
$$

其中 $g_i(x)$ 是**门控函数**（Gating Function），决定每个专家的贡献权重。

**门控函数的定义**：

$$
g(x) = \text{Softmax}(W_g x + \epsilon) \in \mathbb{R}^E
$$

其中 $W_g \in \mathbb{R}^{E \times d}$ 是路由器的可学习参数，$\epsilon$ 是可选的噪声项。

**问题**：如果所有 $g_i > 0$，则每个 token 需要经过**全部** $E$ 个专家——计算量反而增加了 $E$ 倍！

### 2.3 稀疏激活：Top-k 路由

解决方案：**稀疏门控**——只激活 Top-k 个专家：

$$
\boxed{g_i(x) = \begin{cases} \frac{\exp(h_i(x))}{\sum_{j \in \text{Top-k}} \exp(h_j(x))} & \text{if } i \in \text{Top-k}(h(x)) \\ 0 & \text{otherwise} \end{cases}}
$$

其中 $h(x) = W_g x$ 是路由器的 logits。

**Shazeer (2017) 使用 Top-2**：

$$
\text{MoE}_{\text{Top-2}}(x) = g_{i_1}(x) \cdot \text{FFN}_{i_1}(x) + g_{i_2}(x) \cdot \text{FFN}_{i_2}(x)
$$

其中 $i_1, i_2 = \text{Top-2}(\text{Softmax}(W_g x))$。

**Top-2 的问题**：

1. 计算量翻倍（每个 token 需要 2 个 FFN 前向）
2. 通信量翻倍（分布式训练中 token 需发送到 2 台设备）
3. 门控权重的组合引入额外复杂性

### 2.4 Switch Routing：极简 Top-1

Switch Transformer 的核心创新——**只路由到 1 个专家**：

$$
\boxed{\text{Switch}(x) = p_{i^*}(x) \cdot \text{FFN}_{i^*}(x), \quad i^* = \arg\max_{i} \, p_i(x)}
$$

其中路由概率为：

$$
p(x) = \text{Softmax}(W_g x) \in \mathbb{R}^E
$$

**为什么 Top-1 可以工作？**

| 对比 | Top-2 (Shazeer, 2017) | Top-1 (Switch, 2021) |
|------|:---------------------:|:--------------------:|
| 每 token 计算 | $2 \times \text{FFN}$ | $1 \times \text{FFN}$ |
| 通信量 | $2 \times$ | $1 \times$ |
| 路由复杂度 | 需要归一化两个权重 | 直接 argmax |
| 实际效果 | 基线 | 相当或更优（配合改进的训练策略） |

Fedus et al. (2021) 的实验表明：**在相同计算预算下，Top-1 路由配合更多专家数 $E$，比 Top-2 路由获得更好的效果**。

---

## 3. Switch Transformer 架构的数学描述

### 3.1 整体架构总览

Switch Transformer 的结构与标准 Transformer 几乎相同，**唯一的区别**是将部分（或全部）FFN 层替换为 Switch FFN 层：

```
输入 Token Sequence: x_1, x_2, ..., x_T
  ↓
[Token Embedding + Position Embedding]
  ↓
[Transformer Block × L]
  │
  │  每个 Block:
  │  ┌─────────────────────────────┐
  │  │ LayerNorm → Self-Attention  │  ← 标准多头注意力（不变）
  │  │ + Residual Connection       │
  │  ├─────────────────────────────┤
  │  │ LayerNorm → Switch FFN      │  ← 替换标准 FFN
  │  │ + Residual Connection       │
  │  └─────────────────────────────┘
  ↓
[Final LayerNorm] → [Output Head]
```

**注意力层不变**：MoE 只替换 FFN 层，因为 FFN 层占参数量的 2/3，是扩展效率最高的部分。

### 3.2 Switch FFN 层的前向传播

对于一个 batch 中的 token $x \in \mathbb{R}^d$，Switch FFN 的完整前向传播：

**Step 1：计算路由概率**

$$
h = W_g x, \quad p = \text{Softmax}(h) \in \mathbb{R}^E
$$

**Step 2：选择专家**

$$
i^* = \arg\max_i \, p_i
$$

**Step 3：计算专家输出（乘以门控值）**

$$
\boxed{y = p_{i^*} \cdot \text{FFN}_{i^*}(x)}
$$

其中 $\text{FFN}_{i^*}(x) = W_2^{(i^*)} \cdot \text{GELU}(W_1^{(i^*)} x + b_1^{(i^*)}) + b_2^{(i^*)}$。

**关键设计**：输出乘以 $p_{i^*}$（而非直接输出 $\text{FFN}_{i^*}(x)$），原因有二：

1. **梯度传播**：$p_{i^*}$ 是可微的，允许梯度流回路由器 $W_g$
2. **置信度调节**：当路由器不确定时，$p_{i^*}$ 较小，自动降低输出幅度

### 3.3 路由器的数学定义

路由器是一个简单的线性变换 + Softmax：

$$
\text{Router}(x) = \text{Softmax}(W_g x + b_g)
$$

其中 $W_g \in \mathbb{R}^{E \times d}$，$b_g \in \mathbb{R}^E$（部分实现中省略偏置项）。

**路由器参数量**：

$$
P_{\text{router}} = E \times d \quad \text{（远小于单个专家的参数量 } 2 d \cdot d_{ff} \text{）}
$$

对于 $E = 128$，$d = 768$：$P_{\text{router}} = 98{,}304$，而单个专家 $P_{\text{expert}} = 2 \times 768 \times 3072 = 4{,}718{,}592$。路由器开销不到总参数的 **0.02%**。

### 3.4 参数量与 FLOPs 分析

**参数量对比**（以 T5-Base 为基础）：

| 组件 | Dense T5-Base | Switch-Base (128 experts) |
|------|:------------:|:------------------------:|
| Embedding | $V \times d$ | $V \times d$（不变） |
| Self-Attention | $4d^2 \times L$ | $4d^2 \times L$（不变） |
| FFN | $2d \cdot d_{ff} \times L$ | $2d \cdot d_{ff} \times E \times L$（$\times E$） |
| Router | — | $E \times d \times L$（新增，极小） |
| **总参数** | **~223M** | **~7.4B**（$\sim 33\times$） |

**FLOPs 对比**（每个 token）：

$$
\boxed{
\begin{aligned}
\text{FLOPs}_{\text{Dense}} &= L \times (4d^2 \cdot n + 2d \cdot d_{ff}) \\
\text{FLOPs}_{\text{Switch}} &= L \times (4d^2 \cdot n + 2d \cdot d_{ff} + \underbrace{E \cdot d}_{\text{router}})
\end{aligned}
}
$$

由于 $E \cdot d \ll 2d \cdot d_{ff}$，**Switch Transformer 每 token 的 FLOPs 几乎与 Dense 模型相同**。

**参数效率比**：

$$
\text{参数效率} = \frac{N_{\text{total}}}{N_{\text{active}}} = \frac{P_{\text{shared}} + E \times P_{\text{expert}}}{P_{\text{shared}} + 1 \times P_{\text{expert}}} \approx E
$$

128 个专家的 Switch Transformer：参数量约为 Dense 的 $33\times$，计算量几乎不变。

---

## 4. 负载均衡损失与训练稳定性

### 4.1 负载不均衡问题

稀疏路由的最大挑战：**路由器可能将大多数 token 发送到少数几个专家**（"赢家通吃"现象）。

数学描述：定义专家 $i$ 的负载分数（fraction）为：

$$
f_i = \frac{1}{T} \sum_{t=1}^{T} \mathbb{1}[i^*(x_t) = i]
$$

理想情况下 $f_i = 1/E$（均匀分布），但实际中常出现：

$$
f_{\text{hot}} \gg 1/E, \quad f_{\text{cold}} \approx 0
$$

**后果**：

1. **训练效率下降**：热门专家处理大量 token，冷门专家闲置
2. **模型容量浪费**：冷门专家的参数永远不会被充分训练
3. **分布式训练瓶颈**：负载不均导致 GPU 利用率失衡
4. **正反馈循环**：热门专家因更多训练信号变得更强，吸引更多 token

### 4.2 辅助负载均衡损失

Switch Transformer 设计了一个简洁的辅助损失来鼓励均匀分配。

**定义两个量**：

1. **实际分配分数**（每个专家收到的 token 比例）：

$$
f_i = \frac{1}{T} \sum_{t=1}^{T} \mathbb{1}[\arg\max \, p(x_t) = i]
$$

2. **平均路由概率**（每个专家的平均被选择概率）：

$$
P_i = \frac{1}{T} \sum_{t=1}^{T} p_i(x_t)
$$

**辅助负载均衡损失**：

$$
\boxed{\mathcal{L}_{\text{aux}} = \alpha \cdot E \cdot \sum_{i=1}^{E} f_i \cdot P_i}
$$

其中 $\alpha$ 是平衡系数（论文建议 $\alpha = 10^{-2}$）。

**为什么这个损失有效？**

- 当所有专家均匀分配时：$f_i = P_i = 1/E$，$\mathcal{L}_{\text{aux}} = \alpha$
- 当 token 集中到少数专家时：$f_{\text{hot}} \cdot P_{\text{hot}} \gg 1/E^2$，损失增大

> **Q:** 为什么不直接最小化 $\sum_i (f_i - 1/E)^2$？
>
> **A:** 因为 $f_i$ 中的 $\arg\max$ 操作不可微！而 $P_i$ 是可微的（来自 Softmax），所以 $f_i \cdot P_i$ 的乘积形式允许梯度通过 $P_i$ 流回路由器。$f_i$ 作为常数权重引导优化方向。

**总训练损失**：

$$
\boxed{\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \mathcal{L}_{\text{aux}} = -\frac{1}{T}\sum_{t=1}^{T} \log P_\theta(y_t \mid y_{<t}) + \alpha \cdot E \cdot \sum_{i=1}^{E} f_i \cdot P_i}
$$

### 4.3 专家容量因子与溢出处理

即使有辅助损失，短期内仍可能出现负载不均。**容量因子**（Capacity Factor, CF）控制每个专家的最大处理 token 数：

$$
\boxed{C_i = \left\lceil \frac{T}{E} \cdot \text{CF} \right\rceil}
$$

其中 $T$ 是 batch 中的 token 总数，$T/E$ 是理想均匀分配数，CF 是容量放大系数。

| CF 值 | 含义 | 效果 |
|:-----:|------|------|
| 1.0 | 恰好均匀分配 | 可能有大量溢出 |
| 1.25 | 允许 25% 余量 | 论文推荐，平衡效率和质量 |
| 1.5 | 允许 50% 余量 | 几乎无溢出，但浪费内存 |
| 2.0+ | 大余量 | 接近无溢出 |

**溢出 token 的处理**：

当专家 $i$ 已满（收到 $C_i$ 个 token），后续路由到该专家的 token 直接通过**残差连接**跳过 FFN：

$$
y_t = \begin{cases}
p_{i^*}(x_t) \cdot \text{FFN}_{i^*}(x_t) & \text{if expert } i^* \text{ has capacity} \\
x_t & \text{if expert } i^* \text{ is full (overflow)}
\end{cases}
$$

这是一种**优雅的降级策略**：溢出 token 不会丢失信息（通过残差连接保留），只是少了一层 FFN 变换。

### 4.4 梯度推导：路由器参数更新

路由器 $W_g$ 通过两条路径接收梯度：

**路径 1：主损失 → 门控值 → 路由器**

$$
\frac{\partial \mathcal{L}_{\text{CE}}}{\partial W_g} = \frac{\partial \mathcal{L}_{\text{CE}}}{\partial y} \cdot \frac{\partial y}{\partial p_{i^*}} \cdot \frac{\partial p_{i^*}}{\partial W_g}
$$

由于 $y = p_{i^*} \cdot \text{FFN}_{i^*}(x)$：

$$
\frac{\partial y}{\partial p_{i^*}} = \text{FFN}_{i^*}(x)
$$

而 $p_{i^*}$ 是 Softmax 的第 $i^*$ 个分量：

$$
\frac{\partial p_{i^*}}{\partial h_j} = p_{i^*}(\delta_{i^*j} - p_j)
$$

$$
\frac{\partial h_j}{\partial W_g} = x^\top \quad \text{（线性层的标准梯度）}
$$

**路径 2：辅助损失 → 路由概率 → 路由器**

$$
\frac{\partial \mathcal{L}_{\text{aux}}}{\partial W_g} = \alpha \cdot E \cdot \sum_{i=1}^{E} f_i \cdot \frac{\partial P_i}{\partial W_g}
$$

其中 $f_i$ 视为常数（$\arg\max$ 不可微），$P_i = \frac{1}{T}\sum_t p_i(x_t)$：

$$
\boxed{\frac{\partial \mathcal{L}_{\text{aux}}}{\partial W_g} = \frac{\alpha \cdot E}{T} \sum_{t=1}^{T} \sum_{i=1}^{E} f_i \cdot \frac{\partial p_i(x_t)}{\partial W_g}}
$$

辅助损失的梯度会推动路由器**降低**热门专家的概率、**提高**冷门专家的概率，从而实现均衡。

---

## 5. 训练优化方法总结

### 5.1 选择性精度训练

Switch Transformer 发现标准混合精度（全 BF16）在 MoE 中不稳定，提出**选择性精度**策略：

$$
\boxed{
\begin{aligned}
\text{路由器计算} &: \text{FP32} \quad \text{（Softmax 需要高精度）} \\
\text{专家 FFN} &: \text{BF16} \quad \text{（计算密集，节省显存）} \\
\text{注意力层} &: \text{BF16} \\
\text{参数主副本} &: \text{FP32} \quad \text{（优化器状态）}
\end{aligned}
}
$$

**为什么路由器需要 FP32？**

Softmax 的数值稳定性对指数运算的精度敏感：

$$
p_i = \frac{\exp(h_i)}{\sum_j \exp(h_j)}
$$

当 $h_i$ 的差异较小时，BF16 的 7 位尾数可能导致路由决策的随机翻转，破坏训练稳定性。

### 5.2 专家并行与通信

在分布式训练中，$E$ 个专家分布在 $E$ 台设备上（每台设备 1 个专家）：

```
Device 1: Expert 1    Device 2: Expert 2    ...    Device E: Expert E
    ↑                      ↑                            ↑
    └──── All-to-All Communication (tokens ↔ experts) ────┘
```

**通信模式**：

$$
\boxed{
\begin{aligned}
\textbf{1. Dispatch:} &\quad \text{All-to-All}(\text{tokens} \to \text{experts}) \\
\textbf{2. Compute:} &\quad \text{FFN}_{i}(\text{assigned tokens}) \quad \text{各设备独立} \\
\textbf{3. Combine:} &\quad \text{All-to-All}(\text{outputs} \to \text{original devices})
\end{aligned}
}
$$

**通信量分析**：

$$
V_{\text{comm}} = 2 \times B \times T \times d \quad \text{（dispatch + combine，与 Dense 模型的 AllReduce 相当）}
$$

**Switch Transformer 的优势**：Top-1 路由比 Top-2 减少了 **50%** 的通信量。

### 5.3 初始化与正则化策略

**路由器初始化**：

$$
W_g \sim \mathcal{N}\left(0, \frac{1}{\sqrt{d}}\right)
$$

使初始路由概率接近均匀分布 $p_i \approx 1/E$。

**专家 FFN 初始化**（与标准 Transformer 相同）：

$$
W_1^{(i)} \sim \mathcal{N}\left(0, \frac{0.02}{\sqrt{2L}}\right), \quad W_2^{(i)} \sim \mathcal{N}\left(0, \frac{0.02}{\sqrt{2L}}\right)
$$

**Dropout 策略**：

| 组件 | Dense 模型 | Switch Transformer |
|------|:---------:|:-----------------:|
| Attention | 0.1 | 0.1 |
| FFN 输入 | 0.1 | 0.1 |
| 专家内部 | — | 增大（如 0.4） |

**关键发现**：MoE 模型容易在小数据集上过拟合，增大**专家 Dropout** 是有效的正则化手段。

### 5.4 学习率调度与超参数

Switch Transformer 沿用 T5 的 **逆平方根学习率调度**：

$$
\boxed{\eta(t) = \frac{\eta_{\max}}{\sqrt{\max(t, t_w)}}}
$$

**关键超参数**：

| 参数 | 值 | 说明 |
|------|---|------|
| 优化器 | Adafactor | 内存效率优于 Adam |
| 学习率 $\eta_{\max}$ | $1 \times 10^{-2}$ | T5 默认 |
| 负载均衡系数 $\alpha$ | $10^{-2}$ | 辅助损失权重 |
| 容量因子 CF | 1.0~1.5 | 1.25 为默认 |
| 专家数 $E$ | 8~2048 | 取决于设备数 |
| Warmup 步数 $t_w$ | $10^4$ | |
| 训练步数 | $50\text{K}$~$500\text{K}$ | 取决于规模 |

---

## 6. 从数学到代码：完整实现

### 6.1 NumPy 实现核心组件

```python
import numpy as np


def softmax(x, axis=-1):
    """数值稳定的 Softmax: softmax(x_i) = exp(x_i - max) / Σexp(x_j - max)"""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def gelu(x):
    """GELU(x) = x · Φ(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))"""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def layer_norm(x, gamma, beta, eps=1e-5):
    """LayerNorm(x) = γ * (x - μ) / √(σ² + ε) + β"""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


def router_forward(x, W_g):
    """
    路由器前向传播

    数学公式: p = Softmax(W_g · x)
    选择: i* = argmax(p)

    参数:
        x: (batch, seq_len, d_model) — 输入 token 表示
        W_g: (num_experts, d_model) — 路由器权重

    返回:
        expert_idx: (batch, seq_len) — 每个 token 选择的专家索引
        gate_values: (batch, seq_len) — 对应的门控值 p_{i*}
        probs: (batch, seq_len, num_experts) — 完整路由概率
    """
    # h = x @ W_g^T, shape: (batch, seq_len, num_experts)
    logits = np.einsum('bsd,ed->bse', x, W_g)
    probs = softmax(logits, axis=-1)

    # Top-1 选择
    expert_idx = np.argmax(probs, axis=-1)  # (batch, seq_len)

    # 提取门控值 p_{i*}
    B, T = expert_idx.shape
    gate_values = probs[
        np.arange(B)[:, None],
        np.arange(T)[None, :],
        expert_idx
    ]  # (batch, seq_len)

    return expert_idx, gate_values, probs


def expert_ffn(x, W1, b1, W2, b2):
    """
    单个专家 FFN: FFN(x) = W2 · GELU(W1 · x + b1) + b2

    参数:
        x: (..., d_model)
        W1: (d_ff, d_model), b1: (d_ff,)
        W2: (d_model, d_ff), b2: (d_model,)
    """
    hidden = gelu(np.dot(x, W1.T) + b1)
    return np.dot(hidden, W2.T) + b2


def load_balance_loss(expert_idx, probs, num_experts):
    """
    辅助负载均衡损失

    数学公式: L_aux = α · E · Σ_i (f_i · P_i)

    其中:
        f_i = (1/T) Σ_t 1[argmax p(x_t) = i]  — 实际分配分数
        P_i = (1/T) Σ_t p_i(x_t)               — 平均路由概率

    参数:
        expert_idx: (batch, seq_len) — 每个 token 选择的专家
        probs: (batch, seq_len, num_experts) — 路由概率
        num_experts: 专家数量 E
    """
    # 展平 batch 和 seq 维度
    flat_idx = expert_idx.reshape(-1)  # (B*T,)
    flat_probs = probs.reshape(-1, num_experts)  # (B*T, E)
    T_total = flat_idx.shape[0]

    # f_i: 实际分配到每个专家的 token 比例
    f = np.zeros(num_experts)
    for i in range(num_experts):
        f[i] = np.sum(flat_idx == i) / T_total

    # P_i: 每个专家的平均路由概率
    P = np.mean(flat_probs, axis=0)  # (E,)

    # L_aux = E · Σ(f_i · P_i)
    loss = num_experts * np.sum(f * P)
    return loss, f, P


def switch_ffn_forward(x, W_g, experts_params, capacity_factor=1.25):
    """
    Switch FFN 层完整前向传播（含容量约束和溢出处理）

    数学公式:
        y_t = p_{i*}(x_t) · FFN_{i*}(x_t)  (if expert has capacity)
        y_t = x_t                             (if overflow)

    参数:
        x: (batch, seq_len, d_model)
        W_g: (num_experts, d_model)
        experts_params: list of (W1, b1, W2, b2) for each expert
        capacity_factor: 容量因子 CF

    返回:
        output: (batch, seq_len, d_model)
        aux_loss: 辅助负载均衡损失值
        metadata: dict 包含路由统计信息
    """
    B, T, d = x.shape
    E = len(experts_params)

    # Step 1: 路由
    expert_idx, gate_values, probs = router_forward(x, W_g)

    # Step 2: 计算容量
    capacity = int(np.ceil(T / E * capacity_factor))

    # Step 3: 逐专家处理（含容量约束）
    output = np.copy(x)  # 默认残差（溢出 token 保持原值）
    expert_counts = np.zeros(E, dtype=int)
    overflow_count = 0

    for b in range(B):
        for t in range(T):
            ei = expert_idx[b, t]
            if expert_counts[ei] < capacity:
                # 专家有容量：计算 FFN 输出
                W1, b1, W2, b2 = experts_params[ei]
                ffn_out = expert_ffn(x[b, t], W1, b1, W2, b2)
                output[b, t] = gate_values[b, t] * ffn_out
                expert_counts[ei] += 1
            else:
                # 溢出：残差跳过（output[b,t] 已是 x[b,t]）
                overflow_count += 1

    # Step 4: 辅助损失
    aux_loss, f, P = load_balance_loss(expert_idx, probs, E)

    metadata = {
        'expert_counts': expert_counts,
        'overflow_count': overflow_count,
        'overflow_rate': overflow_count / (B * T),
        'f': f,  # 实际分配分数
        'P': P,  # 平均路由概率
    }

    return output, aux_loss, metadata


# ========== 测试 ==========
if __name__ == "__main__":
    np.random.seed(42)

    B, T, d = 2, 16, 64
    E = 4           # 专家数量
    d_ff = d * 4    # FFN 隐藏层维度

    # 初始化路由器
    W_g = np.random.randn(E, d) / np.sqrt(d)

    # 初始化专家参数
    experts_params = []
    for i in range(E):
        W1 = np.random.randn(d_ff, d) * 0.02
        b1 = np.zeros(d_ff)
        W2 = np.random.randn(d, d_ff) * 0.02
        b2 = np.zeros(d)
        experts_params.append((W1, b1, W2, b2))

    # 1. 路由器测试
    x = np.random.randn(B, T, d)
    expert_idx, gate_values, probs = router_forward(x, W_g)
    print(f"路由器测试:")
    print(f"  专家选择分布: {[np.sum(expert_idx == i) for i in range(E)]}")
    print(f"  门控值范围: [{gate_values.min():.4f}, {gate_values.max():.4f}]")
    print(f"  概率和≈1: {np.allclose(probs.sum(axis=-1), 1.0)}")

    # 2. 负载均衡损失
    aux_loss, f, P = load_balance_loss(expert_idx, probs, E)
    print(f"\n负载均衡损失:")
    print(f"  f (实际分配): {f}")
    print(f"  P (平均概率): {P}")
    print(f"  L_aux = {aux_loss:.4f}")
    print(f"  理想值 (均匀): {E * E * (1/E * 1/E):.4f} = {1.0:.4f}")

    # 3. Switch FFN 完整前向传播
    output, aux_loss, metadata = switch_ffn_forward(
        x, W_g, experts_params, capacity_factor=1.5
    )
    print(f"\nSwitch FFN 前向传播:")
    print(f"  输出形状: {output.shape}")
    print(f"  专家负载: {metadata['expert_counts']}")
    print(f"  溢出率: {metadata['overflow_rate']:.2%}")
    print(f"  辅助损失: {aux_loss:.4f}")

    # 4. 验证溢出处理（极端容量因子）
    _, _, meta_tight = switch_ffn_forward(x, W_g, experts_params, capacity_factor=0.5)
    _, _, meta_loose = switch_ffn_forward(x, W_g, experts_params, capacity_factor=3.0)
    print(f"\n容量因子对比:")
    print(f"  CF=0.5 溢出率: {meta_tight['overflow_rate']:.2%}")
    print(f"  CF=1.5 溢出率: {metadata['overflow_rate']:.2%}")
    print(f"  CF=3.0 溢出率: {meta_loose['overflow_rate']:.2%}")

    # 5. 门控值乘法验证
    assert output.shape == x.shape, "输出形状应与输入相同"
    diff = np.abs(output - x).max()
    print(f"\n门控值乘法: 输出与输入差异={diff:.6f} > 0 ✓")

    print("\n✅ Switch Transformer NumPy 核心组件测试通过！")
```

### 6.2 PyTorch 完整实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict


class SwitchRouter(nn.Module):
    """
    Switch 路由器: p = Softmax(W_g · x), i* = argmax(p)

    路由器是 MoE 的核心——决定每个 token 发送到哪个专家。
    """
    def __init__(self, d_model: int, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        # 初始化使初始路由接近均匀
        nn.init.normal_(self.gate.weight, mean=0, std=1.0 / math.sqrt(d_model))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        参数:
            x: (batch, seq_len, d_model)
        返回:
            expert_idx: (batch, seq_len) — 选择的专家索引
            gate_values: (batch, seq_len) — 门控值 p_{i*}
            probs: (batch, seq_len, num_experts) — 完整路由概率
        """
        # FP32 路由（数值稳定性）
        logits = self.gate(x.float())
        probs = F.softmax(logits, dim=-1)

        # Top-1 选择
        gate_values, expert_idx = probs.max(dim=-1)

        return expert_idx, gate_values, probs


class ExpertFFN(nn.Module):
    """单个专家 FFN: FFN(x) = W2 · GELU(W1 · x + b1) + b2"""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class SwitchFFN(nn.Module):
    """
    Switch FFN 层: 用 E 个专家 FFN 替换标准 FFN

    数学公式:
        y_t = p_{i*}(x_t) · FFN_{i*}(x_t)  (有容量)
        y_t = x_t                             (溢出)

    L_aux = α · E · Σ(f_i · P_i)
    """
    def __init__(self, d_model: int, d_ff: int, num_experts: int,
                 capacity_factor: float = 1.25, aux_loss_alpha: float = 0.01,
                 expert_dropout: float = 0.0):
        super().__init__()
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.aux_loss_alpha = aux_loss_alpha

        self.router = SwitchRouter(d_model, num_experts)
        self.experts = nn.ModuleList([
            ExpertFFN(d_model, d_ff, dropout=expert_dropout)
            for _ in range(num_experts)
        ])

    def _compute_aux_loss(self, expert_idx: torch.Tensor,
                          probs: torch.Tensor) -> torch.Tensor:
        """
        辅助负载均衡损失: L_aux = α · E · Σ(f_i · P_i)

        f_i: 实际分配到专家 i 的 token 比例（不可微）
        P_i: 专家 i 的平均路由概率（可微）
        """
        E = self.num_experts

        # f_i: 不可微，作为常数
        one_hot = F.one_hot(expert_idx, E).float()  # (B, T, E)
        f = one_hot.mean(dim=[0, 1])  # (E,)

        # P_i: 可微
        P = probs.mean(dim=[0, 1])  # (E,)

        # L_aux = α · E · Σ(f_i · P_i)
        return self.aux_loss_alpha * E * torch.sum(f * P)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        参数:
            x: (batch, seq_len, d_model)
        返回:
            output: (batch, seq_len, d_model)
            aux_loss: 标量
            metadata: dict
        """
        B, T, d = x.shape
        E = self.num_experts

        # Step 1: 路由
        expert_idx, gate_values, probs = self.router(x)

        # Step 2: 容量计算
        capacity = int(math.ceil(T / E * self.capacity_factor))

        # Step 3: 分发和计算
        # 使用 one-hot 编码实现可微分的路由
        output = torch.zeros_like(x)
        expert_counts = torch.zeros(E, device=x.device)
        overflow_mask = torch.zeros(B, T, device=x.device, dtype=torch.bool)

        for i in range(E):
            # 找到路由到专家 i 的 token
            mask_i = (expert_idx == i)  # (B, T)

            if not mask_i.any():
                continue

            # 容量约束：只处理前 capacity 个 token
            # 展平处理
            flat_mask = mask_i.view(-1)
            token_positions = flat_mask.nonzero(as_tuple=True)[0]
            expert_counts[i] = len(token_positions)

            if len(token_positions) > capacity * B:
                # 标记溢出
                overflow_positions = token_positions[capacity * B:]
                flat_overflow = torch.zeros(B * T, device=x.device, dtype=torch.bool)
                flat_overflow[overflow_positions] = True
                overflow_mask |= flat_overflow.view(B, T)
                # 截断到容量
                token_positions = token_positions[:capacity * B]
                flat_mask = torch.zeros_like(flat_mask)
                flat_mask[token_positions] = True
                mask_i = flat_mask.view(B, T)

            if mask_i.any():
                # 提取 token，计算专家输出
                tokens = x[mask_i]  # (num_tokens, d)
                expert_out = self.experts[i](tokens)
                gates = gate_values[mask_i].unsqueeze(-1)  # (num_tokens, 1)
                output[mask_i] = gates * expert_out

        # 溢出 token 使用残差
        output[overflow_mask] = x[overflow_mask]

        # Step 4: 辅助损失
        aux_loss = self._compute_aux_loss(expert_idx, probs)

        metadata = {
            'expert_counts': expert_counts.detach(),
            'overflow_rate': overflow_mask.float().mean().item(),
            'gate_entropy': -(probs * (probs + 1e-10).log()).sum(-1).mean().item(),
        }

        return output, aux_loss, metadata


class SwitchTransformerBlock(nn.Module):
    """
    Switch Transformer 块 (Pre-Norm)

    a = x + Attn(LN(x))
    out = a + SwitchFFN(LN(a))
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 num_experts: int, capacity_factor: float = 1.25,
                 aux_loss_alpha: float = 0.01, dropout: float = 0.0,
                 expert_dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads,
                                          dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.switch_ffn = SwitchFFN(d_model, d_ff, num_experts,
                                    capacity_factor, aux_loss_alpha,
                                    expert_dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        # Pre-Norm Self-Attention
        normed = self.ln1(x)
        attn_out, _ = self.attn(normed, normed, normed,
                                attn_mask=attn_mask, is_causal=True)
        x = x + self.dropout(attn_out)

        # Pre-Norm Switch FFN
        normed2 = self.ln2(x)
        ffn_out, aux_loss, metadata = self.switch_ffn(normed2)
        x = x + self.dropout(ffn_out)

        return x, aux_loss, metadata


class SwitchTransformer(nn.Module):
    """
    完整 Switch Transformer 模型

    将标准 Transformer 的 FFN 层替换为 Switch FFN（MoE），
    每个 token 只路由到 1 个专家，实现参数量与计算量的解耦。
    """
    def __init__(self, vocab_size: int = 32000, d_model: int = 768,
                 num_heads: int = 12, num_layers: int = 12,
                 d_ff: int = 3072, num_experts: int = 8,
                 max_len: int = 512, capacity_factor: float = 1.25,
                 aux_loss_alpha: float = 0.01, dropout: float = 0.1,
                 expert_dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            SwitchTransformerBlock(
                d_model, num_heads, d_ff, num_experts,
                capacity_factor, aux_loss_alpha, dropout, expert_dropout
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # 权重共享

        self._init_weights()

    def _init_weights(self):
        """初始化: 标准 N(0, 0.02), 残差路径 N(0, 0.02/√(2L))"""
        std = 0.02
        res_std = std / math.sqrt(2 * self.num_layers)
        for name, p in self.named_parameters():
            if 'gate' in name:
                continue  # 路由器已在 SwitchRouter.__init__ 中初始化
            if p.dim() >= 2:
                if 'fc2' in name or 'out_proj' in name:
                    nn.init.normal_(p, mean=0, std=res_std)
                else:
                    nn.init.normal_(p, mean=0, std=std)
            elif 'bias' in name:
                nn.init.zeros_(p)

    def forward(self, input_ids: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> Dict:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(pos))

        total_aux_loss = 0.0
        all_metadata = []

        for block in self.blocks:
            x, aux_loss, metadata = block(x)
            total_aux_loss = total_aux_loss + aux_loss
            all_metadata.append(metadata)

        logits = self.lm_head(self.final_norm(x))

        loss = None
        if labels is not None:
            ce_loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, self.vocab_size),
                labels[:, 1:].contiguous().view(-1), ignore_index=-100
            )
            loss = ce_loss + total_aux_loss

        return {
            "logits": logits,
            "loss": loss,
            "ce_loss": ce_loss if labels is not None else None,
            "aux_loss": total_aux_loss,
            "metadata": all_metadata,
        }

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 50,
                 temperature: float = 1.0, top_k: int = 0) -> torch.Tensor:
        """自回归生成（支持 top-k 采样）"""
        max_len = self.pos_emb.num_embeddings
        for _ in range(max_new_tokens):
            idx = input_ids if input_ids.size(1) <= max_len else input_ids[:, -max_len:]
            out = self.forward(idx)
            logits = out["logits"][:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            next_tok = torch.multinomial(F.softmax(logits, dim=-1), 1)
            input_ids = torch.cat([input_ids, next_tok], dim=1)
        return input_ids

    def get_expert_stats(self) -> Dict:
        """获取所有层的专家使用统计"""
        stats = {}
        for i, block in enumerate(self.blocks):
            router = block.switch_ffn.router
            stats[f'layer_{i}'] = {
                'router_weight_norm': router.gate.weight.norm().item(),
                'num_experts': block.switch_ffn.num_experts,
            }
        return stats


# ========== 测试 ==========
if __name__ == "__main__":
    torch.manual_seed(42)

    V, d, H, L, d_ff = 1000, 128, 4, 4, 512
    E = 8   # 专家数量
    B, T = 4, 32

    model = SwitchTransformer(V, d, H, L, d_ff, num_experts=E,
                              max_len=256, capacity_factor=1.5)

    # 参数量统计
    total_params = sum(p.numel() for p in model.parameters())
    expert_params = sum(
        sum(p.numel() for p in block.switch_ffn.experts.parameters())
        for block in model.blocks
    )
    router_params = sum(
        sum(p.numel() for p in block.switch_ffn.router.parameters())
        for block in model.blocks
    )
    shared_params = total_params - expert_params - router_params
    print(f"参数量统计:")
    print(f"  总参数: {total_params:,}")
    print(f"  专家参数: {expert_params:,} ({expert_params/total_params:.1%})")
    print(f"  路由器参数: {router_params:,} ({router_params/total_params:.1%})")
    print(f"  共享参数: {shared_params:,} ({shared_params/total_params:.1%})")

    # 等效 Dense 模型参数
    dense_params = shared_params + expert_params // E
    print(f"  等效 Dense 参数: {dense_params:,}")
    print(f"  参数膨胀比: {total_params/dense_params:.1f}x")

    # 前向传播
    ids = torch.randint(0, V, (B, T))
    model.eval()
    with torch.no_grad():
        out = model(ids, ids)

    print(f"\n前向传播:")
    print(f"  Logits: {out['logits'].shape}")
    print(f"  CE Loss: {out['ce_loss'].item():.4f}")
    print(f"  Aux Loss: {out['aux_loss'].item():.4f}")
    print(f"  Total Loss: {out['loss'].item():.4f}")

    # 路由统计
    print(f"\n路由统计 (各层):")
    for i, meta in enumerate(out['metadata']):
        counts = meta['expert_counts'].tolist()
        overflow = meta['overflow_rate']
        entropy = meta['gate_entropy']
        print(f"  Layer {i}: counts={[int(c) for c in counts]}, "
              f"overflow={overflow:.2%}, entropy={entropy:.2f}")

    # 因果性验证
    with torch.no_grad():
        full = model(ids)["logits"]
        part = model(ids[:, :16])["logits"]
    diff = (full[:, :16, :] - part).abs().max().item()
    print(f"\n因果性: 前16位差异={diff:.6f}")

    # 权重共享验证
    assert torch.equal(model.tok_emb.weight.data, model.lm_head.weight.data)
    print("权重共享 ✓")

    # 训练一步
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    out = model(ids, ids)
    out["loss"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # 验证路由器梯度存在
    for block in model.blocks:
        g = block.switch_ffn.router.gate.weight.grad
        assert g is not None and g.abs().max() > 0
    print("路由器梯度 ✓")

    opt.step()
    print(f"训练后 Loss: {out['loss'].item():.4f}")

    # 生成
    model.eval()
    gen = model.generate(torch.randint(0, V, (1, 5)), max_new_tokens=10, top_k=50)
    print(f"生成: {gen[0].tolist()}")

    print(f"\n✅ Switch Transformer 模型测试通过！")
```

---

## 7. MoE 可视化与实践技巧

### 7.1 路由分布可视化

```python
import numpy as np
import matplotlib.pyplot as plt


def plot_routing_distribution():
    """可视化路由分布：均匀 vs 不均匀 vs 训练后"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    E = 8  # 专家数量

    # (1) 理想均匀分布
    f_uniform = np.ones(E) / E
    axes[0].bar(range(E), f_uniform, color='steelblue', alpha=0.8)
    axes[0].axhline(y=1/E, color='red', linestyle='--', label=f'Ideal = 1/{E}')
    axes[0].set(xlabel='Expert Index', ylabel='Token Fraction',
                title='Ideal Uniform Distribution', ylim=(0, 0.5))
    axes[0].legend()

    # (2) 不均衡分布（未训练）
    f_skewed = np.array([0.35, 0.25, 0.15, 0.10, 0.05, 0.05, 0.03, 0.02])
    colors = ['red' if f > 0.2 else 'orange' if f > 0.1 else 'steelblue' for f in f_skewed]
    axes[1].bar(range(E), f_skewed, color=colors, alpha=0.8)
    axes[1].axhline(y=1/E, color='red', linestyle='--', label=f'Ideal = 1/{E}')
    axes[1].set(xlabel='Expert Index', ylabel='Token Fraction',
                title='Skewed Distribution (Without Balance Loss)', ylim=(0, 0.5))
    axes[1].legend()

    # (3) 训练后的分布（接近均匀）
    np.random.seed(42)
    f_trained = np.random.dirichlet(np.ones(E) * 20)  # 接近均匀的 Dirichlet
    axes[2].bar(range(E), f_trained, color='steelblue', alpha=0.8)
    axes[2].axhline(y=1/E, color='red', linestyle='--', label=f'Ideal = 1/{E}')
    axes[2].set(xlabel='Expert Index', ylabel='Token Fraction',
                title='Balanced Distribution (With Balance Loss)', ylim=(0, 0.5))
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("switch_routing_distribution.png", dpi=150, bbox_inches='tight')
    plt.show()


def plot_capacity_factor_effect():
    """可视化容量因子对溢出率的影响"""
    T, E = 1024, 8
    ideal = T / E  # = 128

    # 模拟不同程度的不均衡
    np.random.seed(42)
    skew_levels = [1.0, 2.0, 5.0]  # Dirichlet 参数（越小越不均衡）
    cfs = np.linspace(0.5, 3.0, 50)

    fig, ax = plt.subplots(figsize=(10, 6))
    for alpha_d in skew_levels:
        overflow_rates = []
        for cf in cfs:
            cap = int(np.ceil(ideal * cf))
            # 模拟多次
            rates = []
            for _ in range(100):
                counts = np.random.multinomial(T, np.random.dirichlet(np.ones(E) * alpha_d))
                overflow = np.maximum(counts - cap, 0).sum()
                rates.append(overflow / T)
            overflow_rates.append(np.mean(rates))
        ax.plot(cfs, overflow_rates, lw=2, label=f'Skew α={alpha_d}')

    ax.axvline(x=1.25, color='red', linestyle='--', alpha=0.7, label='Default CF=1.25')
    ax.set(xlabel='Capacity Factor (CF)', ylabel='Overflow Rate',
           title='Overflow Rate vs Capacity Factor')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("switch_capacity_factor.png", dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    plot_routing_distribution()
    plot_capacity_factor_effect()
```

### 7.2 实践调参建议

**专家数量选择**：

| 场景 | 推荐专家数 $E$ | 说明 |
|------|:------------:|------|
| 单 GPU 实验 | 4~8 | 验证 MoE 架构有效性 |
| 多 GPU 训练 | GPU 数量的倍数 | 确保专家并行效率 |
| 大规模训练 | 64~2048 | 参数量可达万亿级 |

**容量因子经验值**：

$$
\boxed{\text{CF} = \begin{cases}
1.0 & \text{推理（节省内存）} \\
1.25 & \text{训练默认} \\
1.5 & \text{负载严重不均时} \\
2.0 & \text{调试或小规模实验}
\end{cases}}
$$

**负载均衡系数 $\alpha$ 调优**：

| $\alpha$ 值 | 效果 |
|:----------:|------|
| $10^{-3}$ | 轻微均衡，路由更自由 |
| $10^{-2}$ | 论文默认，推荐起点 |
| $10^{-1}$ | 强均衡，可能损害模型质量 |
| $1$ | 过强，专家退化为随机分配 |

**MoE 过拟合缓解**：

| 策略 | 效果 | 代价 |
|------|------|------|
| 增大专家 Dropout | 显著 | 训练速度略降 |
| 减少专家数 $E$ | 显著 | 模型容量降低 |
| 增大训练数据 | 根本解决 | 数据获取成本 |
| 增大 $\alpha$ | 中等 | 可能限制专家专业化 |

---

## 8. 与其他模型的关系

### 8.1 从 Dense Transformer 到 MoE Transformer

| 维度 | Dense Transformer | MoE Transformer |
|------|:-----------------:|:---------------:|
| **参数利用** | 100% 参数参与每次计算 | 仅 $\sim 1/E$ 参数活跃 |
| **计算-参数关系** | $C \propto N$ | $C \propto N_{\text{active}} \ll N_{\text{total}}$ |
| **训练效率** | 基线 | 同等计算下更好的效果 |
| **推理效率** | 每 token FLOPs 固定 | 每 token FLOPs 不变，但需更多内存 |
| **扩展方式** | 加宽/加深 | 增加专家数 |

$$
\boxed{\underbrace{\text{Dense}}_{\text{参数=计算}} \xrightarrow{\text{MoE 稀疏化}} \underbrace{\text{Sparse}}_{\text{参数} \gg \text{计算}}}
$$

### 8.2 Switch Transformer 在大模型发展中的定位

**稀疏化范式的三次演进**：

```
Phase 1: Dense MoE (Jacobs, 1991)       ← 所有专家参与，不稀疏
Phase 2: Sparse MoE (Shazeer, 2017)     ← Top-2 稀疏路由
Phase 3: Switch Routing (Fedus, 2021)   ← Top-1 极简路由 + 训练稳定性改进
```

**Switch Transformer 的核心贡献**：

1. **简化路由**：Top-1 替代 Top-2，减少计算和通信
2. **训练稳定性**：选择性精度 + 改进的负载均衡损失
3. **规模验证**：首次训练 1.6T 参数的稀疏模型
4. **效率证明**：在相同 FLOPs 下，MoE 显著优于 Dense

### 8.3 MoE 的后续发展

```
Switch Transformer (2021) ── Top-1 MoE, 1.6T 参数
  ├── GShard (2021) ── Top-2 + 随机路由, 600B 参数
  ├── GLaM (2022) ── 1.2T 参数, 能效优化
  ├── ST-MoE (2022) ── 改进训练稳定性
  ├── Mixtral (2024) ── Top-2 MoE + 开源, 46.7B 参数
  └── DeepSeek-MoE (2024) ── 细粒度专家 + 共享专家
       └── DeepSeek-V2/V3 (2024) ── 236B/671B 参数
```

| Switch 遗留问题 | 后续解决方案 |
|:------------:|:----------:|
| Top-1 信息丢失 | Mixtral 恢复 Top-2 + 开源 |
| 推理需全部专家在内存 | Expert Offloading |
| 负载不均衡 | DeepSeek-MoE 共享专家设计 |
| 训练不稳定 | ST-MoE 路由器 z-loss |
| 专家冗余 | DeepSeek-MoE 细粒度分割 |

**MoE 在 2024 年的主导地位**：

几乎所有前沿大模型都采用了 MoE 架构：

$$
\boxed{\text{GPT-4} \xrightarrow{\text{传闻 MoE}} \text{Mixtral} \xrightarrow{\text{开源}} \text{DeepSeek-V3} \xrightarrow{\text{671B, Top-8/256}} \text{MoE 成为默认选择}}
$$

---

## 扩展阅读与实现

### 问题 1：为什么 Top-1 路由不会导致信息瓶颈？

三个关键因素：(1) **门控缩放** $y = p_{i^*} \cdot \text{FFN}_{i^*}(x)$，低置信度自动降低影响；(2) **残差连接**保留原始信息；(3) **多层堆叠**——不同层路由独立，经过 $L$ 层后 token 实际"访问"了 $L$ 个不同专家。

### 问题 2：负载均衡损失的变体比较

| 方法 | 公式 | 特点 |
|------|------|------|
| Switch (2021) | $E \sum f_i P_i$ | 简洁有效，$f_i$ 不可微 |
| z-loss (ST-MoE) | $\frac{1}{T}\sum_t (\log\sum_i \exp h_i^{(t)})^2$ | 全可微，惩罚 logits 过大 |
| 两者结合 | Load Balance + z-loss | 更稳定，两个超参数 |

### 问题 3：MoE 的内存-计算权衡

Switch Transformer 的核心权衡——内存 $\propto N_{\text{total}}$，计算 $\propto N_{\text{active}}$：

| 指标 | Dense T5-Base | Switch-Base 128E |
|------|:------------:|:----------:|
| 总参数 | 223M | 7.4B |
| 活跃参数/token | 223M | 223M |
| 显存需求 | 0.9 GB | 29.6 GB |
| 预训练速度 | 基线 | 快 7× |

**推理优化**：Expert Offloading——将不活跃专家放在 CPU/SSD 上，按需加载。

### 问题 4：专家专业化现象

训练充分后，不同专家会自发"专业化"——例如 Expert 0 处理标点、Expert 1 处理动词、Expert 2 处理名词等。这种专业化是**自发涌现**的，没有显式监督信号，说明路由器学会了基于 token 语义特征进行有意义的分组。

### 问题 5：Switch Transformer 的 Scaling 行为

MoE 的 Scaling Law：$\mathcal{L}_{\text{MoE}}(N_{\text{total}}, E) \approx \mathcal{L}_{\text{Dense}}(N_{\text{active}}) - \Delta(E)$，其中增益 $\Delta(E) \propto \log(E)$（递减）。这意味着从 1→8 专家提升显著，8→64 中等，64→2048 趋缓。实际中专家数量受限于设备数和通信带宽。

---

## 参考资源

### 经典论文

1. Fedus, W., Zoph, B., & Shazeer, N. (2021). [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961). JMLR 2022.
   - **贡献**：提出 Switch Routing (Top-1 MoE)，首次训练 1.6T 参数稀疏模型

2. Shazeer, N., Mirhoseini, A., Maziarz, K., et al. (2017). [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538). ICLR 2017.
   - **贡献**：提出稀疏门控 MoE 层（Top-2 路由），奠定现代 MoE 基础

3. Lepikhin, D., Lee, H., Xu, Y., et al. (2021). [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668). ICLR 2021.
   - **贡献**：提出 GShard 分布式 MoE 训练框架

4. Zoph, B., Bello, I., Kumar, S., et al. (2022). [ST-MoE: Designing Stable and Transferable Sparse Expert Models](https://arxiv.org/abs/2202.08906). arXiv.
   - **贡献**：系统研究 MoE 训练稳定性，提出 Router z-loss

5. Jiang, A. Q., Sablayrolles, A., Roux, A., et al. (2024). [Mixtral of Experts](https://arxiv.org/abs/2401.04088). arXiv.
   - **贡献**：开源 MoE 模型，Top-2 路由 + 8 专家，性能优异

### 教材与书籍

6. Jacobs, R. A., Jordan, M. I., Nowlan, S. J., & Hinton, G. E. (1991). [Adaptive Mixtures of Local Experts](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf). Neural Computation.
   - **章节**：MoE 的开创性论文，提出竞争性专家学习

### 在线资源与教程

7. Hugging Face. [Mixture of Experts Explained](https://huggingface.co/blog/moe).
   - **内容**：MoE 架构的直觉解释和实践指南

8. Cameron R. Wolfe. [A Visual Guide to Mixture of Experts](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts).
   - **内容**：MoE 路由机制和负载均衡的可视化讲解

---

## 附录：符号表

| 符号 | 含义 | 维度/类型 |
|------|------|----------|
| $T$ | 序列长度（batch 中 token 总数） | 标量 |
| $d$ ($d_{\text{model}}$) | 隐藏维度 | 标量 |
| $d_{ff}$ | FFN 隐藏层维度 | 标量，通常 $4d$ |
| $d_k$ | 每个注意力头的维度 | 标量 |
| $L$ | Transformer 层数 | 标量 |
| $A$ | 注意力头数 | 标量 |
| $E$ | 专家数量 | 标量 |
| $K$ | Top-k 路由的 $k$ 值 | 标量，Switch 中 $K=1$ |
| $\|V\|$ | 词表大小 | 标量 |
| $x$ | 输入 token 表示 | $(d,)$ |
| $W_g$ | 路由器权重矩阵 | $(E, d)$ |
| $p_i(x)$ | 路由概率 $\text{Softmax}(W_g x)_i$ | 标量，$\sum_i p_i = 1$ |
| $i^*(x)$ | 选中专家的索引 | 标量，$\arg\max_i p_i$ |
| $f_i$ | 专家 $i$ 的实际 token 分配比例 | 标量，$\sum f_i = 1$ |
| $P_i$ | 专家 $i$ 的平均路由概率 | 标量，$\sum P_i = 1$ |
| $\text{Expert}_i(\cdot)$ | 第 $i$ 个专家 FFN | 函数 $\mathbb{R}^d \to \mathbb{R}^d$ |
| $W_1^{(i)}, W_2^{(i)}$ | 专家 $i$ 的 FFN 权重 | $(d_{ff}, d)$, $(d, d_{ff})$ |
| $\mathcal{L}_{\text{aux}}$ | 辅助负载均衡损失 | 标量 |
| $\ell(\cdot, \cdot)$ | 损失函数 | 函数 |
| $\alpha$ | 辅助损失权重系数 | 标量，通常 $10^{-2}$ |
| $\text{CF}$ | 容量因子 (Capacity Factor) | 标量，通常 1.0~1.5 |
| $C_{\text{expert}}$ | 专家容量上限 | 标量，$\text{CF} \cdot T/E$ |
| $N_{\text{total}}$ | 模型总参数量 | 标量 |
| $N_{\text{active}}$ | 每 token 活跃参数量 | 标量，$\ll N_{\text{total}}$ |
| $\eta$ | 学习率 | 标量 |

**典型维度示例（Switch-Base, 128 experts）：**
- $d = 768$，$d_{ff} = 3072$，$d_k = 64$
- $L = 12$，$A = 12$，$E = 128$
- $|V| = 32{,}000$，$\text{CF} = 1.25$，$\alpha = 0.01$
- $N_{\text{total}} = 7.4\text{B}$，$N_{\text{active}} \approx 223\text{M}$

---

最后更新：2026-03-19
