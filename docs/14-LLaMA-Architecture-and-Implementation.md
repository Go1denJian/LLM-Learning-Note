# LLaMA 架构与实现 —— 开放高效大模型的完整数学推导

> **前置知识**：Transformer 架构（自注意力、多头注意力）、GPT 系列（因果语言模型）、LayerNorm、RLHF 基础、Python 基础  
> **与前面内容的联系**：建议先学习 [Transformer-Math-and-Implementation](./06-Transformer-Math-and-Implementation.md) 理解注意力机制，以及 [RLHF-Math-and-Implementation](./13-RLHF-Math-and-Implementation.md) 理解对齐训练  
> **与后续内容的联系**：LLaMA 的架构创新（RMSNorm、SwiGLU、RoPE）成为后续开源大模型的标准组件，直接影响 LLaMA 2/3、Mistral、DeepSeek 等模型

---

## 目录

1. [引言：为什么需要开放高效的大模型？](#1-引言为什么需要开放高效的大模型)
   - 1.1 [大模型的封闭困境](#11-大模型的封闭困境)
   - 1.2 [LLaMA 的核心思想：更多数据胜过更大模型](#12-llama-的核心思想更多数据胜过更大模型)
   - 1.3 [三大架构创新概览](#13-三大架构创新概览)
   - 1.4 [本科数学知识映射表](#14-本科数学知识映射表)
2. [基础概念：LLaMA 架构总览](#2-基础概念llama-架构总览)
   - 2.1 [从标准 Transformer 到 LLaMA 的改进](#21-从标准-transformer-到-llama-的改进)
   - 2.2 [Pre-Norm vs Post-Norm 架构](#22-pre-norm-vs-post-norm-架构)
   - 2.3 [模型规模与训练配置](#23-模型规模与训练配置)
3. [核心创新一：RMSNorm 替代 LayerNorm](#3-核心创新一rmsnorm-替代-layernorm)
   - 3.1 [LayerNorm 回顾](#31-layernorm-回顾)
   - 3.2 [RMSNorm 的数学定义](#32-rmsnorm-的数学定义)
   - 3.3 [RMSNorm 的梯度推导](#33-rmsnorm-的梯度推导)
   - 3.4 [RMSNorm vs LayerNorm 的效率分析](#34-rmsnorm-vs-layernorm-的效率分析)
4. [核心创新二：SwiGLU 激活函数](#4-核心创新二swiglu-激活函数)
   - 4.1 [从 ReLU 到 GLU 家族](#41-从-relu-到-glu-家族)
   - 4.2 [SwiGLU 的数学定义](#42-swiglu-的数学定义)
   - 4.3 [SwiGLU 的梯度推导](#43-swiglu-的梯度推导)
   - 4.4 [FFN 维度调整：从 4d 到 8d/3](#44-ffn-维度调整从-4d-到-8d3)
5. [核心创新三：RoPE 旋转位置编码](#5-核心创新三rope-旋转位置编码)
   - 5.1 [位置编码的发展脉络](#51-位置编码的发展脉络)
   - 5.2 [RoPE 的数学推导](#52-rope-的数学推导)
   - 5.3 [旋转矩阵的性质与相对位置编码](#53-旋转矩阵的性质与相对位置编码)
   - 5.4 [高效实现：复数形式](#54-高效实现复数形式)
   - 5.5 [RoPE 的长程衰减特性](#55-rope-的长程衰减特性)
6. [从数学到代码：完整实现](#6-从数学到代码完整实现)
   - 6.1 [NumPy 实现核心组件](#61-numpy-实现核心组件)
   - 6.2 [PyTorch 完整 LLaMA 实现](#62-pytorch-完整-llama-实现)
7. [实践技巧与可视化](#7-实践技巧与可视化)
   - 7.1 [RoPE 位置编码可视化](#71-rope-位置编码可视化)
   - 7.2 [SwiGLU 激活函数对比](#72-swiglu-激活函数对比)
   - 7.3 [训练配置与工程细节](#73-训练配置与工程细节)
8. [与其他模型的关系](#8-与其他模型的关系)
   - 8.1 [从 GPT 到 LLaMA 的架构演进](#81-从-gpt-到-llama-的架构演进)
   - 8.2 [LLaMA 家族与开源生态](#82-llama-家族与开源生态)
   - 8.3 [Scaling Laws 的重新审视](#83-scaling-laws-的重新审视)

[扩展阅读与实现](#扩展阅读与实现)

[参考资源](#参考资源)

附录：[符号表](#附录符号表)

---

## 1. 引言：为什么需要开放高效的大模型？

### 1.1 大模型的封闭困境

2023 年初，大模型领域面临一个悖论：

| 问题 | 表现 | 影响 |
|------|------|------|
| **封闭性** | GPT-4、PaLM 等闭源 | 学术界无法复现和研究 |
| **高成本** | 训练 175B 需数百万美元 | 仅少数机构负担得起 |
| **效率低** | 追求更大模型，忽视训练效率 | 资源浪费严重 |
| **不透明** | 训练数据、方法不公开 | 安全性和偏见无法审计 |

> **核心矛盾**：学术界需要开放的大模型来推进研究，但训练大模型的成本和资源门槛极高。

### 1.2 LLaMA 的核心思想：更多数据胜过更大模型

LLaMA（Touvron et al., 2023）的关键洞察来自 **Chinchilla Scaling Laws**（Hoffmann et al., 2022）：

**传统思路**（GPT-3）：固定计算预算，尽可能增大模型

$$
\text{GPT-3}: \quad N = 175B, \quad D = 300B \text{ tokens}
$$

**Chinchilla 发现**：模型大小和数据量应同步扩展

$$
\boxed{N_{\text{opt}} \propto C^{0.5}, \quad D_{\text{opt}} \propto C^{0.5}}
$$

其中 $C$ 是总计算量（FLOPs），$N$ 是参数量，$D$ 是训练 token 数。

**LLaMA 的更激进策略**：超越 Chinchilla 最优点，用更多数据训练较小模型

| 模型 | 参数量 | 训练 Tokens | Chinchilla 最优 Tokens | 实际/最优比 |
|------|:------:|:-----------:|:---------------------:|:-----------:|
| LLaMA-7B | 6.7B | 1.0T | ~140B | **7.1×** |
| LLaMA-13B | 13.0B | 1.0T | ~260B | **3.8×** |
| LLaMA-33B | 32.5B | 1.4T | ~650B | **2.2×** |
| LLaMA-65B | 65.2B | 1.4T | ~1.3T | **1.1×** |

> **关键结论**：LLaMA-13B 在大多数基准上匹配或超越 GPT-3（175B），参数量仅为后者的 **7.4%**。
>
> $$\boxed{\text{LLaMA-13B} \approx_{\text{benchmark}} \text{GPT-3 (175B)}}$$

这意味着推理成本大幅降低：在单块 GPU 上即可运行 13B 模型。

### 1.3 三大架构创新概览

LLaMA 在标准 Transformer 基础上引入了三个关键改进：

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LLaMA 三大架构创新                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. RMSNorm 替代 LayerNorm                                         │
│  ┌──────────────────────────────────────────┐                       │
│  │  去除均值中心化，仅保留缩放归一化             │                       │
│  │  计算量减少 ~10-15%，训练更稳定              │                       │
│  │  数学: RMSNorm(x) = x / RMS(x) * γ       │                       │
│  └──────────────────────────────────────────┘                       │
│                                                                     │
│  2. SwiGLU 激活函数                                                 │
│  ┌──────────────────────────────────────────┐                       │
│  │  GLU 门控 + Swish 激活的组合               │                       │
│  │  比 ReLU/GELU 更好的性能                   │                       │
│  │  数学: SwiGLU(x) = Swish(xW₁) ⊙ (xW₃)  │                       │
│  └──────────────────────────────────────────┘                       │
│                                                                     │
│  3. RoPE 旋转位置编码                                               │
│  ┌──────────────────────────────────────────┐                       │
│  │  通过旋转矩阵编码相对位置                   │                       │
│  │  自然支持外推到更长序列                     │                       │
│  │  数学: q_m = R_m · q, k_n = R_n · k      │                       │
│  └──────────────────────────────────────────┘                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.4 本科数学知识映射表

| LLaMA 概念 | 对应数学 | 本科课程 |
|-----------|----------|----------|
| RMSNorm | 均方根、归一化 | 线性代数、统计学 |
| SwiGLU 激活 | Sigmoid、逐元素乘法 | 微积分、线性代数 |
| RoPE 旋转编码 | 旋转矩阵、复数乘法 | 线性代数、复变函数 |
| 注意力分数衰减 | 内积、三角函数求和 | 线性代数、傅里叶分析 |
| Scaling Laws | 幂律关系、对数线性回归 | 统计学 |
| 因果语言模型 | 条件概率链式法则 | 概率论 |

---

## 2. 基础概念：LLaMA 架构总览

### 2.1 从标准 Transformer 到 LLaMA 的改进

LLaMA 基于 GPT 风格的 **decoder-only** Transformer，但做了多项关键修改：

| 组件 | 标准 Transformer (GPT) | LLaMA |
|------|----------------------|-------|
| 归一化 | Post-Norm (LayerNorm) | **Pre-Norm (RMSNorm)** |
| 激活函数 | ReLU / GELU | **SwiGLU** |
| 位置编码 | 绝对正弦/学习式 | **RoPE（旋转位置编码）** |
| FFN 维度 | $4d$ | **$\frac{8d}{3}$**（取最近的 256 倍数） |
| 偏置项 | 有 bias | **无 bias**（注意力和 FFN 均去除） |
| 词表 | BPE | **SentencePiece (BPE)** |

LLaMA 单层 Transformer Block 的计算流程：

$$
\begin{aligned}
& \text{// 自注意力子层 (Pre-Norm)} \\
& h'_l = h_{l-1} + \text{SelfAttn}(\text{RMSNorm}(h_{l-1})) \\
& \text{// FFN 子层 (Pre-Norm)} \\
& h_l = h'_l + \text{SwiGLU-FFN}(\text{RMSNorm}(h'_l))
\end{aligned}
$$

### 2.2 Pre-Norm vs Post-Norm 架构

**Post-Norm**（原始 Transformer）：

$$
h_l = \text{LayerNorm}(h_{l-1} + \text{SubLayer}(h_{l-1}))
$$

**Pre-Norm**（LLaMA 采用）：

$$
h_l = h_{l-1} + \text{SubLayer}(\text{RMSNorm}(h_{l-1}))
$$

Pre-Norm 的优势：

1. **梯度流更稳定**：残差连接直接传递梯度，归一化不在残差路径上
2. **无需 warmup**：训练初期不易出现梯度爆炸
3. **更适合深层网络**：LLaMA-65B 有 80 层，Pre-Norm 确保梯度可以顺畅回传

$$
\boxed{\frac{\partial h_L}{\partial h_l} = I + \sum_{k=l+1}^{L} \prod_{j=l+1}^{k} \frac{\partial \text{SubLayer}_j}{\partial h_{j-1}} \approx I \quad \text{(Pre-Norm 下残差梯度主导)}}
$$

### 2.3 模型规模与训练配置

| 配置 | LLaMA-7B | LLaMA-13B | LLaMA-33B | LLaMA-65B |
|------|:--------:|:---------:|:---------:|:---------:|
| 维度 $d$ | 4096 | 5120 | 6656 | 8192 |
| 层数 $L$ | 32 | 40 | 60 | 80 |
| 注意力头 $n_h$ | 32 | 40 | 52 | 64 |
| 头维度 $d_h$ | 128 | 128 | 128 | 128 |
| FFN 维度 | 11008 | 13824 | 17920 | 22016 |
| 学习率 | 3e-4 | 3e-4 | 1.5e-4 | 1.5e-4 |
| Batch Size | 4M tokens | 4M tokens | 4M tokens | 4M tokens |
| 训练 Tokens | 1.0T | 1.0T | 1.4T | 1.4T |

**训练数据**：

| 数据集 | 比例 | Token 数 | 采样倍率 |
|--------|:----:|:--------:|:--------:|
| CommonCrawl | 67.0% | ~927B | ~1.10 |
| C4 | 15.0% | ~207B | ~1.06 |
| GitHub | 4.5% | ~62B | ~0.64 |
| Wikipedia | 4.5% | ~62B | ~2.45 |
| Books | 4.5% | ~62B | ~2.23 |
| ArXiv | 2.5% | ~35B | ~1.06 |
| StackExchange | 2.0% | ~28B | ~1.03 |

> **注意**：Wikipedia 和 Books 的采样倍率 > 1，意味着这些高质量数据在训练中被多次使用。

---

## 3. 核心创新一：RMSNorm 替代 LayerNorm

### 3.1 LayerNorm 回顾

标准 LayerNorm（Ba et al., 2016）对每个样本的隐藏维度做归一化：

给定输入向量 $x \in \mathbb{R}^d$：

$$
\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

其中：

$$
\mu = \frac{1}{d} \sum_{i=1}^d x_i, \quad \sigma^2 = \frac{1}{d} \sum_{i=1}^d (x_i - \mu)^2
$$

- $\gamma, \beta \in \mathbb{R}^d$ 是可学习的缩放和平移参数
- $\epsilon$ 是数值稳定常数（通常 $10^{-6}$）
- 计算复杂度：需要两次遍历数据（计算 $\mu$，再计算 $\sigma^2$）

### 3.2 RMSNorm 的数学定义

RMSNorm（Zhang & Sennrich, 2019）的核心洞察：**归一化中的均值中心化（减去 $\mu$）对模型效果影响不大，可以去除**。

$$
\boxed{\text{RMSNorm}(x) = \gamma \odot \frac{x}{\text{RMS}(x)}}
$$

其中 RMS（Root Mean Square）定义为：

$$
\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}
$$

**关键简化**：
1. **去除均值中心化**：不计算 $\mu$，不做 $x - \mu$
2. **去除偏置参数**：没有 $\beta$，仅保留缩放 $\gamma$
3. **一次遍历**：只需计算 $\sum x_i^2$

展开写出第 $i$ 个分量：

$$
\text{RMSNorm}(x)_i = \frac{\gamma_i \cdot x_i}{\sqrt{\frac{1}{d} \sum_{j=1}^d x_j^2 + \epsilon}}
$$

### 3.3 RMSNorm 的梯度推导

为了推导反向传播，我们需要计算 $\frac{\partial \mathcal{L}}{\partial x}$。

设 $y = \text{RMSNorm}(x)$，$s = \text{RMS}(x) = \sqrt{\frac{1}{d} \|x\|^2 + \epsilon}$，则 $y_i = \gamma_i x_i / s$。

**Step 1：计算 $\frac{\partial y_i}{\partial x_j}$**

$$
\frac{\partial y_i}{\partial x_j} = \frac{\gamma_i}{s} \cdot \delta_{ij} + \gamma_i x_i \cdot \frac{\partial}{\partial x_j}\left(\frac{1}{s}\right)
$$

其中：

$$
\frac{\partial}{\partial x_j}\left(\frac{1}{s}\right) = -\frac{1}{s^2} \cdot \frac{\partial s}{\partial x_j} = -\frac{1}{s^2} \cdot \frac{x_j}{d \cdot s} = -\frac{x_j}{d \cdot s^3}
$$

因此：

$$
\frac{\partial y_i}{\partial x_j} = \frac{\gamma_i \delta_{ij}}{s} - \frac{\gamma_i x_i x_j}{d \cdot s^3}
$$

**Step 2：链式法则**

设上游梯度 $g_i = \frac{\partial \mathcal{L}}{\partial y_i}$，则：

$$
\frac{\partial \mathcal{L}}{\partial x_j} = \sum_{i=1}^d g_i \frac{\partial y_i}{\partial x_j} = \frac{g_j \gamma_j}{s} - \frac{x_j}{d \cdot s^3} \sum_{i=1}^d g_i \gamma_i x_i
$$

记 $\hat{g}_i = g_i \gamma_i$（吸收缩放参数），$c = \frac{1}{d \cdot s^2} \sum_{i=1}^d \hat{g}_i x_i$，则：

$$
\boxed{\frac{\partial \mathcal{L}}{\partial x_j} = \frac{1}{s}\left(\hat{g}_j - x_j \cdot c\right)}
$$

> **与 LayerNorm 梯度对比**：RMSNorm 梯度中没有 $\bar{g}$（梯度均值项），计算更简单。LayerNorm 的梯度还需要减去 $\bar{g} = \frac{1}{d}\sum_i \hat{g}_i$。

**缩放参数 $\gamma$ 的梯度**：

$$
\boxed{\frac{\partial \mathcal{L}}{\partial \gamma_i} = g_i \cdot \frac{x_i}{s}}
$$

### 3.4 RMSNorm vs LayerNorm 的效率分析

| 维度 | LayerNorm | RMSNorm | 优势 |
|------|-----------|---------|------|
| **统计量计算** | $\mu$ + $\sigma^2$（两次遍历） | 仅 $\text{RMS}$（一次遍历） | ~减少 50% 统计计算 |
| **可学习参数** | $\gamma, \beta$（$2d$ 个） | 仅 $\gamma$（$d$ 个） | 参数量减半 |
| **反向传播** | 需要 $\bar{g}$ 和 $\bar{gx}$ | 仅需 $\bar{gx}$ | 梯度计算更简 |
| **数值稳定性** | 减均值有助稳定 | 无均值中心化 | LayerNorm 略优 |
| **实际加速** | — | **~7-15%** | RMSNorm 更快 |

**FLOPs 对比**（单次前向，维度 $d$）：

$$
\text{LayerNorm FLOPs} \approx 5d + 2 \quad \text{(均值 + 方差 + 归一化 + 仿射)}
$$

$$
\text{RMSNorm FLOPs} \approx 3d + 1 \quad \text{(平方和 + 归一化 + 缩放)}
$$

$$
\boxed{\text{FLOPs 节省} \approx \frac{2d}{5d} = 40\%}
$$

> **关键结论**：RMSNorm 在几乎不损失模型质量的前提下，显著减少了归一化层的计算开销。对于 LLaMA-65B（80 层，每层 2 个归一化），这意味着 160 个归一化操作都变得更高效。

---

## 4. 核心创新二：SwiGLU 激活函数

### 4.1 从 ReLU 到 GLU 家族

**激活函数发展脉络**：

| 激活函数 | 数学定义 | 问题 |
|----------|----------|------|
| ReLU | $\max(0, x)$ | 死神经元、非平滑 |
| GELU | $x \cdot \Phi(x)$ | 计算较复杂 |
| Swish | $x \cdot \sigma(\beta x)$ | 单门控 |
| **GLU** | $\sigma(xW_1) \odot (xW_2)$ | **门控线性单元** |
| **SwiGLU** | $\text{Swish}(xW_1) \odot (xW_3)$ | **LLaMA 采用** |

**GLU（Gated Linear Unit）的核心思想**（Dauphin et al., 2017）：

将输入分为两路：一路作为"门"控制信息流，另一路作为"值"：

$$
\text{GLU}(x) = \sigma(xW_1 + b_1) \odot (xW_2 + b_2)
$$

其中 $\sigma$ 是 sigmoid 函数，$\odot$ 是逐元素乘法。

### 4.2 SwiGLU 的数学定义

SwiGLU（Shazeer, 2020）将 GLU 中的 sigmoid 门替换为 **Swish 激活**：

**Swish 函数**（也称 SiLU）：

$$
\text{Swish}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
$$

**SwiGLU 前馈网络**：

$$
\boxed{\text{SwiGLU-FFN}(x) = \left(\text{Swish}(xW_1) \odot xW_3\right) W_2}
$$

其中：
- $W_1 \in \mathbb{R}^{d \times d_{ff}}$：门控投影（gate projection）
- $W_3 \in \mathbb{R}^{d \times d_{ff}}$：上投影（up projection）
- $W_2 \in \mathbb{R}^{d_{ff} \times d}$：下投影（down projection）
- LLaMA 中不使用偏置项 $b$

展开写出完整计算：

$$
\begin{aligned}
\text{gate} &= \text{Swish}(xW_1) \in \mathbb{R}^{d_{ff}} \\
\text{up} &= xW_3 \in \mathbb{R}^{d_{ff}} \\
\text{hidden} &= \text{gate} \odot \text{up} \in \mathbb{R}^{d_{ff}} \\
\text{output} &= \text{hidden} \cdot W_2 \in \mathbb{R}^{d}
\end{aligned}
$$

### 4.3 SwiGLU 的梯度推导

**Swish 函数的导数**：

$$
\frac{d}{dx} \text{Swish}(x) = \frac{d}{dx} [x \cdot \sigma(x)] = \sigma(x) + x \cdot \sigma(x)(1 - \sigma(x))
$$

$$
\boxed{\text{Swish}'(x) = \sigma(x) + x \cdot \sigma(x)(1 - \sigma(x)) = \text{Swish}(x) + \sigma(x)(1 - \text{Swish}(x))}
$$

> **性质**：Swish 的导数在 $x=0$ 处为 $0.5$（而非 ReLU 的不连续），确保了更平滑的梯度流。

**SwiGLU 对各参数的梯度**：

设 $g_1 = xW_1$，$g_3 = xW_3$，$h = \text{Swish}(g_1) \odot g_3$，$y = hW_2$。

上游梯度 $\frac{\partial \mathcal{L}}{\partial y} = \delta_y \in \mathbb{R}^{d}$，则：

$$
\frac{\partial \mathcal{L}}{\partial W_2} = h^\top \delta_y
$$

$$
\frac{\partial \mathcal{L}}{\partial h} = \delta_y W_2^\top \in \mathbb{R}^{d_{ff}}
$$

$$
\frac{\partial \mathcal{L}}{\partial g_3} = \frac{\partial \mathcal{L}}{\partial h} \odot \text{Swish}(g_1)
$$

$$
\boxed{\frac{\partial \mathcal{L}}{\partial g_1} = \frac{\partial \mathcal{L}}{\partial h} \odot g_3 \odot \text{Swish}'(g_1)}
$$

$$
\frac{\partial \mathcal{L}}{\partial W_1} = x^\top \frac{\partial \mathcal{L}}{\partial g_1}, \quad \frac{\partial \mathcal{L}}{\partial W_3} = x^\top \frac{\partial \mathcal{L}}{\partial g_3}
$$

### 4.4 FFN 维度调整：从 4d 到 8d/3

标准 Transformer 的 FFN 使用 2 个权重矩阵（$W_1 \in \mathbb{R}^{d \times 4d}$，$W_2 \in \mathbb{R}^{4d \times d}$），参数量为 $8d^2$。

SwiGLU 引入第三个矩阵 $W_3$，为保持参数量不变，需调整 $d_{ff}$：

$$
\text{标准 FFN}: \quad 2 \times d \times 4d = 8d^2
$$

$$
\text{SwiGLU FFN}: \quad 3 \times d \times d_{ff} = 3d \cdot d_{ff}
$$

令 $3d \cdot d_{ff} = 8d^2$，解得：

$$
\boxed{d_{ff} = \frac{8d}{3}}
$$

实际中 LLaMA 取 $\frac{8d}{3}$ 的最近 256 倍数：

| 模型 | $d$ | 理论 $\frac{8d}{3}$ | 实际 $d_{ff}$ |
|------|:---:|:---:|:---:|
| LLaMA-7B | 4096 | 10923 | 11008 |
| LLaMA-13B | 5120 | 13653 | 13824 |
| LLaMA-33B | 6656 | 17749 | 17920 |
| LLaMA-65B | 8192 | 21845 | 22016 |

> **为什么取 256 的倍数？** GPU 张量核心（Tensor Core）以 256 字节为粒度运算，对齐到 256 倍数可以最大化硬件利用率。

---

## 5. 核心创新三：RoPE 旋转位置编码

### 5.1 位置编码的发展脉络

| 方法 | 代表模型 | 特点 | 局限 |
|------|----------|------|------|
| 正弦绝对编码 | Transformer | 固定，无需学习 | 无法捕捉相对位置 |
| 学习式绝对编码 | GPT-2, BERT | 灵活 | 无法外推超出训练长度 |
| 相对位置编码 | Transformer-XL | 捕捉相对距离 | 实现复杂，计算开销大 |
| ALiBi | BLOOM | 线性衰减 | 仅在注意力偏置中 |
| **RoPE** | **LLaMA** | **旋转矩阵编码相对位置** | **优雅高效** |

### 5.2 RoPE 的数学推导

**核心目标**：设计一个位置编码函数 $f(x, m)$，使得两个位置 $m, n$ 的查询和键的内积**仅依赖于相对位置** $m - n$：

$$
\langle f(q, m), f(k, n) \rangle = g(q, k, m - n)
$$

**从二维情况出发**：

考虑 $q, k \in \mathbb{R}^2$，将它们视为复数 $q = q_1 + q_2 i$，$k = k_1 + k_2 i$。

**关键洞察**：如果我们对位置 $m$ 的向量乘以 $e^{im\theta}$（复数旋转），则：

$$
f(q, m) = q \cdot e^{im\theta}, \quad f(k, n) = k \cdot e^{in\theta}
$$

内积变为：

$$
\text{Re}\left[f(q, m) \cdot \overline{f(k, n)}\right] = \text{Re}\left[q \cdot e^{im\theta} \cdot \overline{k \cdot e^{in\theta}}\right] = \text{Re}\left[q \bar{k} \cdot e^{i(m-n)\theta}\right]
$$

$$
\boxed{\langle f(q, m), f(k, n) \rangle = g(q, k, m-n) \quad \checkmark}
$$

**推广到高维**：

对于 $d$ 维向量，将其分为 $d/2$ 对，每对应用不同频率的旋转：

$$
f(x, m) = R_m \cdot x
$$

其中旋转矩阵 $R_m$ 为：

$$
\boxed{R_m = \begin{pmatrix} \cos m\theta_1 & -\sin m\theta_1 & & & \\ \sin m\theta_1 & \cos m\theta_1 & & & \\ & & \cos m\theta_2 & -\sin m\theta_2 & \\ & & \sin m\theta_2 & \cos m\theta_2 & \\ & & & & \ddots \end{pmatrix}}
$$

频率参数定义为（与原始 Transformer 正弦编码类似）：

$$
\theta_i = 10000^{-2(i-1)/d}, \quad i = 1, 2, \ldots, d/2
$$

### 5.3 旋转矩阵的性质与相对位置编码

**性质 1：正交性**

$$
R_m^\top R_m = I \quad \text{（旋转矩阵是正交矩阵）}
$$

这意味着 RoPE **不改变向量的模长**：$\|R_m x\| = \|x\|$。

**性质 2：相对位置编码**

$$
R_m^\top R_n = R_{n-m}
$$

因此：

$$
\langle R_m q, R_n k \rangle = q^\top R_m^\top R_n k = q^\top R_{n-m} k
$$

$$
\boxed{\text{注意力分数仅依赖相对位置: } \quad \text{score}(m, n) = (R_m q)^\top (R_n k) / \sqrt{d_h} = q^\top R_{n-m} k / \sqrt{d_h}}
$$

**性质 3：远程衰减**

内积随相对距离 $|m-n|$ 增大而**自然衰减**（由于不同频率的旋转导致"去相关"效应）。

### 5.4 高效实现：复数形式

直接构造和应用旋转矩阵 $R_m$ 需要 $O(d^2)$ 运算，但利用其分块对角结构可以优化为 $O(d)$。

**复数实现**：

将 $d$ 维向量 $x = [x_1, x_2, x_3, x_4, \ldots, x_{d-1}, x_d]$ 重新解释为 $d/2$ 个复数：

$$
\tilde{x}_i = x_{2i-1} + x_{2i} \cdot j, \quad i = 1, \ldots, d/2
$$

RoPE 变为逐元素复数乘法：

$$
\boxed{f(\tilde{x}, m)_i = \tilde{x}_i \cdot e^{jm\theta_i} = (x_{2i-1} + x_{2i} j)(\cos m\theta_i + j \sin m\theta_i)}
$$

展开为实数运算：

$$
\begin{aligned}
f(x, m)_{2i-1} &= x_{2i-1} \cos m\theta_i - x_{2i} \sin m\theta_i \\
f(x, m)_{2i} &= x_{2i-1} \sin m\theta_i + x_{2i} \cos m\theta_i
\end{aligned}
$$

**计算复杂度**：

| 操作 | 矩阵形式 | 复数形式 |
|------|:--------:|:--------:|
| 乘法次数 | $O(d^2)$ | $O(d)$ |
| 加法次数 | $O(d^2)$ | $O(d)$ |
| 额外存储 | $O(d^2)$ | $O(d)$ |

> **实际实现**：预计算 $\cos m\theta_i$ 和 $\sin m\theta_i$ 的表（$m = 0, 1, \ldots, T_{\max}-1$），运行时只需查表和逐元素运算。

### 5.5 RoPE 的长程衰减特性

对于位置 $m$ 和 $n$ 的内积，展开后可以分析远程行为：

$$
\langle R_m q, R_n k \rangle = \sum_{i=1}^{d/2} \left[(q_{2i-1}k_{2i-1} + q_{2i}k_{2i})\cos(m-n)\theta_i + (q_{2i-1}k_{2i} - q_{2i}k_{2i-1})\sin(m-n)\theta_i\right]
$$

当 $|m-n|$ 很大时，由于不同频率 $\theta_i$ 的旋转角度不同，各项的 $\cos$ 和 $\sin$ 值趋于"随机"，导致求和后的结果衰减：

$$
\boxed{\mathbb{E}\left[\langle R_m q, R_n k \rangle\right] \to 0 \quad \text{as} \quad |m-n| \to \infty}
$$

这种**自然的远程衰减**使模型能够区分近距离和远距离 token 的关系，无需额外设计。

低频分量（$\theta_i$ 较小）变化缓慢，负责编码长程依赖；高频分量变化快速，编码局部位置关系。

---

## 6. 从数学到代码：完整实现

### 6.1 NumPy 实现核心组件

#### 6.1.1 RMSNorm

```python
import numpy as np

class RMSNormNumPy:
    """
    RMSNorm 的 NumPy 实现（含前向和反向传播）
    
    数学:
        y = γ ⊙ (x / RMS(x))
        RMS(x) = sqrt(mean(x²) + ε)
    """
    
    def __init__(self, d, eps=1e-6):
        self.gamma = np.ones(d)          # 可学习缩放参数，shape (d,)
        self.eps = eps
        # 缓存用于反向传播
        self.cache = {}
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: shape (B, T, d) — 输入张量
        返回:
            y: shape (B, T, d) — 归一化输出
        """
        # 计算 RMS: sqrt(mean(x²) + ε)
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.eps)  # (B, T, 1)
        x_norm = x / rms                  # (B, T, d) 归一化
        y = self.gamma * x_norm           # (B, T, d) 缩放
        
        self.cache = {'x': x, 'rms': rms, 'x_norm': x_norm}
        return y
    
    def backward(self, grad_y):
        """
        反向传播
        
        参数:
            grad_y: shape (B, T, d) — 上游梯度 ∂L/∂y
        返回:
            grad_x: shape (B, T, d) — 输入梯度 ∂L/∂x
        """
        x, rms, x_norm = self.cache['x'], self.cache['rms'], self.cache['x_norm']
        d = x.shape[-1]
        
        # ∂L/∂γ = Σ (grad_y ⊙ x_norm)，沿 batch 和 time 维度求和
        self.grad_gamma = np.sum(grad_y * x_norm, axis=(0, 1))
        
        # ∂L/∂x: 使用推导的公式
        grad_hat = grad_y * self.gamma             # (B, T, d) 吸收 γ
        # c = (1/d) * Σ(grad_hat ⊙ x) / rms²
        c = np.mean(grad_hat * x, axis=-1, keepdims=True) / (rms ** 2)  # (B, T, 1)
        grad_x = (grad_hat - x * c) / rms          # (B, T, d)
        
        return grad_x


# ===== 验证：与数值梯度对比 =====
np.random.seed(42)
B, T, d = 2, 3, 8
x = np.random.randn(B, T, d)

norm = RMSNormNumPy(d)
y = norm.forward(x)

# 数值梯度验证
grad_y = np.random.randn(B, T, d)
grad_x = norm.backward(grad_y)

eps_num = 1e-5
grad_x_numerical = np.zeros_like(x)
for i in range(B):
    for j in range(T):
        for k in range(d):
            x_plus = x.copy(); x_plus[i, j, k] += eps_num
            x_minus = x.copy(); x_minus[i, j, k] -= eps_num
            y_plus = norm.forward(x_plus)
            y_minus = norm.forward(x_minus)
            grad_x_numerical[i, j, k] = np.sum(grad_y * (y_plus - y_minus)) / (2 * eps_num)

error = np.max(np.abs(grad_x - grad_x_numerical))
print(f"RMSNorm 梯度验证 — 最大误差: {error:.2e} ({'✅ 通过' if error < 1e-5 else '❌ 失败'})")
print(f"输出统计: mean={y.mean():.4f}, std={y.std():.4f}, RMS≈{np.sqrt(np.mean(y**2)):.4f}")
```

#### 6.1.2 Swish 和 SwiGLU

```python
class SwiGLUNumPy:
    """
    SwiGLU 前馈网络的 NumPy 实现
    
    数学:
        SwiGLU(x) = (Swish(xW₁) ⊙ xW₃) W₂
        Swish(z) = z · σ(z)
    """
    
    def __init__(self, d, d_ff):
        scale = np.sqrt(2.0 / (d + d_ff))
        self.W1 = np.random.randn(d, d_ff) * scale    # 门控投影 (d, d_ff)
        self.W3 = np.random.randn(d, d_ff) * scale    # 上投影   (d, d_ff)
        self.W2 = np.random.randn(d_ff, d) * scale    # 下投影   (d_ff, d)
        self.cache = {}
    
    @staticmethod
    def swish(x):
        """Swish(x) = x · σ(x)"""
        sig = 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
        return x * sig
    
    @staticmethod
    def swish_grad(x):
        """Swish'(x) = σ(x) + x·σ(x)·(1-σ(x))"""
        sig = 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
        return sig + x * sig * (1.0 - sig)
    
    def forward(self, x):
        """
        参数:
            x: shape (B, T, d) — 输入
        返回:
            y: shape (B, T, d) — 输出
        """
        g1 = x @ self.W1              # (B, T, d_ff) — 门控预激活
        g3 = x @ self.W3              # (B, T, d_ff) — 上投影
        gate = self.swish(g1)          # (B, T, d_ff) — Swish 门控
        hidden = gate * g3             # (B, T, d_ff) — 逐元素门控
        y = hidden @ self.W2           # (B, T, d)    — 下投影
        
        self.cache = {'x': x, 'g1': g1, 'g3': g3, 'gate': gate, 'hidden': hidden}
        return y
    
    def backward(self, grad_y):
        """
        参数:
            grad_y: shape (B, T, d) — 上游梯度
        返回:
            grad_x: shape (B, T, d) — 输入梯度
        """
        x, g1, g3, gate, hidden = (
            self.cache['x'], self.cache['g1'], self.cache['g3'],
            self.cache['gate'], self.cache['hidden'])
        
        # ∂L/∂W₂ 和 ∂L/∂hidden
        grad_hidden = grad_y @ self.W2.T               # (B, T, d_ff)
        self.grad_W2 = np.einsum('btf,btd->fd', hidden, grad_y)
        
        # ∂L/∂g₃ 和 ∂L/∂g₁
        grad_g3 = grad_hidden * gate                    # (B, T, d_ff)
        grad_gate = grad_hidden * g3                    # (B, T, d_ff)
        grad_g1 = grad_gate * self.swish_grad(g1)      # (B, T, d_ff)
        
        # ∂L/∂W₁, ∂L/∂W₃, ∂L/∂x
        self.grad_W1 = np.einsum('btd,btf->df', x, grad_g1)
        self.grad_W3 = np.einsum('btd,btf->df', x, grad_g3)
        grad_x = grad_g1 @ self.W1.T + grad_g3 @ self.W3.T  # (B, T, d)
        
        return grad_x


# ===== 验证 =====
np.random.seed(42)
B, T, d, d_ff = 2, 3, 8, 16
x = np.random.randn(B, T, d) * 0.5

ffn = SwiGLUNumPy(d, d_ff)
y = ffn.forward(x)
print(f"SwiGLU 输出: shape={y.shape}, mean={y.mean():.4f}, std={y.std():.4f}")

# 验证 Swish 性质
z = np.linspace(-5, 5, 11)
swish_vals = SwiGLUNumPy.swish(z)
print(f"\nSwish 函数值:")
for zi, si in zip(z, swish_vals):
    print(f"  Swish({zi:+5.1f}) = {si:+.4f}")
print(f"  Swish(0) = {SwiGLUNumPy.swish(np.array([0.0]))[0]:.4f} (理论值: 0)")
```

#### 6.1.3 RoPE 旋转位置编码

```python
class RoPENumPy:
    """
    RoPE (Rotary Position Embedding) 的 NumPy 实现
    
    数学:
        f(x, m)_{2i-1} = x_{2i-1} cos(mθᵢ) - x_{2i} sin(mθᵢ)
        f(x, m)_{2i}   = x_{2i-1} sin(mθᵢ) + x_{2i} cos(mθᵢ)
        θᵢ = 10000^{-2(i-1)/d}
    """
    
    def __init__(self, d, max_len=2048, base=10000.0):
        self.d = d
        # 频率参数: θᵢ = base^{-2(i-1)/d}, i=1,...,d/2
        freqs = 1.0 / (base ** (np.arange(0, d, 2, dtype=np.float64) / d))  # (d/2,)
        # 位置序列
        positions = np.arange(max_len, dtype=np.float64)  # (max_len,)
        # 角度矩阵: m·θᵢ
        angles = np.outer(positions, freqs)  # (max_len, d/2)
        # 预计算 cos 和 sin
        self.cos_cache = np.cos(angles).astype(np.float32)  # (max_len, d/2)
        self.sin_cache = np.sin(angles).astype(np.float32)  # (max_len, d/2)
    
    def apply(self, x, start_pos=0):
        """
        对输入应用 RoPE
        
        参数:
            x: shape (B, T, n_h, d_h) — 查询或键向量
            start_pos: 起始位置（用于增量解码）
        返回:
            x_rotated: shape (B, T, n_h, d_h) — 旋转后的向量
        """
        B, T, n_h, d_h = x.shape
        cos = self.cos_cache[start_pos:start_pos+T, :]  # (T, d_h/2)
        sin = self.sin_cache[start_pos:start_pos+T, :]  # (T, d_h/2)
        
        # 将 x 分为偶数和奇数维度
        x_even = x[..., 0::2]  # (B, T, n_h, d_h/2) — x_{2i-1}
        x_odd  = x[..., 1::2]  # (B, T, n_h, d_h/2) — x_{2i}
        
        # 广播 cos/sin 到 (1, T, 1, d_h/2)
        cos = cos[np.newaxis, :, np.newaxis, :]
        sin = sin[np.newaxis, :, np.newaxis, :]
        
        # 旋转公式
        y_even = x_even * cos - x_odd * sin
        y_odd  = x_even * sin + x_odd * cos
        
        # 交错合并
        x_rotated = np.zeros_like(x)
        x_rotated[..., 0::2] = y_even
        x_rotated[..., 1::2] = y_odd
        
        return x_rotated


# ===== 验证 RoPE 的相对位置性质 =====
np.random.seed(42)
d_h = 8
rope = RoPENumPy(d_h, max_len=100)

# 创建两个固定向量 q, k
q = np.random.randn(1, 1, 1, d_h).astype(np.float32)
k = np.random.randn(1, 1, 1, d_h).astype(np.float32)

# 测试不同绝对位置但相同相对距离的内积
print("RoPE 相对位置验证 (相对距离=5):")
for m in [0, 10, 50, 90]:
    n = m + 5
    q_at_m = np.tile(q, (1, 1, 1, 1))
    k_at_n = np.tile(k, (1, 1, 1, 1))
    
    # 手动设置位置
    q_rotated = rope.apply(q_at_m, start_pos=m)
    k_rotated = rope.apply(k_at_n, start_pos=n)
    
    score = np.sum(q_rotated * k_rotated)
    print(f"  位置 ({m:2d}, {n:2d}): score = {score:.6f}")

print("\nRoPE 远程衰减验证:")
for dist in [1, 5, 10, 20, 50, 100]:
    scores = []
    for trial in range(100):
        q_rand = np.random.randn(1, 1, 1, d_h).astype(np.float32)
        k_rand = np.random.randn(1, 1, 1, d_h).astype(np.float32)
        q_rot = rope.apply(q_rand, start_pos=0)
        k_rot = rope.apply(k_rand, start_pos=dist)
        scores.append(np.sum(q_rot * k_rot))
    print(f"  距离 {dist:3d}: mean_score = {np.mean(scores):+.4f}, std = {np.std(scores):.4f}")
```

#### 6.1.4 因果自注意力（含 RoPE）

```python
def causal_attention_with_rope(Q, K, V, rope, start_pos=0):
    """
    带 RoPE 的因果自注意力 (NumPy)
    
    参数:
        Q: shape (B, T, n_h, d_h) — 查询
        K: shape (B, T, n_h, d_h) — 键
        V: shape (B, T, n_h, d_h) — 值
        rope: RoPENumPy 实例
        start_pos: 起始位置
    返回:
        output: shape (B, T, n_h, d_h) — 注意力输出
        attn_weights: shape (B, n_h, T, T) — 注意力权重
    
    数学:
        Attention(Q, K, V) = softmax(R_m Q (R_n K)^T / √d_h) V
    """
    B, T, n_h, d_h = Q.shape
    
    # 应用 RoPE
    Q_rot = rope.apply(Q, start_pos)  # (B, T, n_h, d_h)
    K_rot = rope.apply(K, start_pos)  # (B, T, n_h, d_h)
    
    # 转置为 (B, n_h, T, d_h) 方便矩阵乘法
    Q_rot = Q_rot.transpose(0, 2, 1, 3)
    K_rot = K_rot.transpose(0, 2, 1, 3)
    V_t   = V.transpose(0, 2, 1, 3)
    
    # 注意力分数: (B, n_h, T, T)
    scores = Q_rot @ K_rot.transpose(0, 1, 3, 2) / np.sqrt(d_h)
    
    # 因果掩码: 上三角设为 -inf
    causal_mask = np.triu(np.ones((T, T)), k=1) * (-1e9)
    scores = scores + causal_mask
    
    # Softmax
    scores_max = scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attn_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
    
    # 加权求和
    output = attn_weights @ V_t  # (B, n_h, T, d_h)
    output = output.transpose(0, 2, 1, 3)  # (B, T, n_h, d_h)
    
    return output, attn_weights


# ===== 验证 =====
np.random.seed(42)
B, T, n_h, d_h = 2, 8, 4, 16
rope = RoPENumPy(d_h, max_len=100)

Q = np.random.randn(B, T, n_h, d_h).astype(np.float32) * 0.1
K = np.random.randn(B, T, n_h, d_h).astype(np.float32) * 0.1
V = np.random.randn(B, T, n_h, d_h).astype(np.float32) * 0.1

output, attn_weights = causal_attention_with_rope(Q, K, V, rope)
print(f"注意力输出: shape={output.shape}")
print(f"注意力权重: shape={attn_weights.shape}")
print(f"因果掩码验证 — 权重上三角和: {attn_weights[0, 0][np.triu_indices(T, k=1)].sum():.6f} (应为 0)")
print(f"权重行和: {attn_weights[0, 0].sum(axis=-1)}")
```

### 6.2 PyTorch 完整 LLaMA 实现

#### 6.2.1 RMSNorm (PyTorch)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    """
    RMSNorm — LLaMA 使用的归一化层
    
    数学: y = γ ⊙ x / RMS(x), RMS(x) = √(mean(x²) + ε)
    """
    
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))  # 缩放参数 γ
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d)
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)
```

#### 6.2.2 RoPE (PyTorch)

```python
class RotaryPositionEmbedding(nn.Module):
    """
    RoPE — 旋转位置编码
    
    数学: f(x, m)_i = x_i · e^{j·m·θ_i} (复数乘法)
    """
    
    def __init__(self, d: int, max_len: int = 2048, base: float = 10000.0):
        super().__init__()
        # 频率: θ_i = base^{-2i/d}
        freqs = 1.0 / (base ** (torch.arange(0, d, 2).float() / d))
        positions = torch.arange(max_len).float()
        angles = torch.outer(positions, freqs)  # (max_len, d/2)
        
        # 预计算复数形式: e^{j·m·θ_i}
        self.register_buffer('cos', angles.cos())  # (max_len, d/2)
        self.register_buffer('sin', angles.sin())  # (max_len, d/2)
    
    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """
        参数:
            x: (B, T, n_h, d_h) — 查询或键
            start_pos: 起始位置
        返回:
            x_rotated: (B, T, n_h, d_h)
        """
        B, T, n_h, d_h = x.shape
        cos = self.cos[start_pos:start_pos+T].view(1, T, 1, -1)  # (1, T, 1, d_h/2)
        sin = self.sin[start_pos:start_pos+T].view(1, T, 1, -1)  # (1, T, 1, d_h/2)
        
        x_even = x[..., 0::2]  # (B, T, n_h, d_h/2)
        x_odd  = x[..., 1::2]  # (B, T, n_h, d_h/2)
        
        y_even = x_even * cos - x_odd * sin
        y_odd  = x_even * sin + x_odd * cos
        
        return torch.stack([y_even, y_odd], dim=-1).flatten(-2)
```

#### 6.2.3 SwiGLU FFN (PyTorch)

```python
class SwiGLU(nn.Module):
    """
    SwiGLU 前馈网络 — LLaMA 使用的 FFN
    
    数学: SwiGLU(x) = (Swish(xW₁) ⊙ xW₃) W₂
    """
    
    def __init__(self, d: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d, d_ff, bias=False)  # 门控投影
        self.w3 = nn.Linear(d, d_ff, bias=False)  # 上投影
        self.w2 = nn.Linear(d_ff, d, bias=False)  # 下投影
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Swish(xW₁) ⊙ xW₃ → 下投影
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

#### 6.2.4 LLaMA Attention (PyTorch)

```python
class LLaMAAttention(nn.Module):
    """
    LLaMA 多头注意力（含 RoPE + 因果掩码）
    
    数学: Attention = softmax((R_m·Q)(R_n·K)^T / √d_h) · V
    """
    
    def __init__(self, d: int, n_heads: int, max_len: int = 2048):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d // n_heads
        
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.wo = nn.Linear(d, d, bias=False)
        
        self.rope = RotaryPositionEmbedding(self.d_head, max_len)
    
    def forward(self, x: torch.Tensor, start_pos: int = 0,
                mask: torch.Tensor = None) -> torch.Tensor:
        B, T, d = x.shape
        
        # 线性投影
        q = self.wq(x).view(B, T, self.n_heads, self.d_head)
        k = self.wk(x).view(B, T, self.n_heads, self.d_head)
        v = self.wv(x).view(B, T, self.n_heads, self.d_head)
        
        # 应用 RoPE
        q = self.rope(q, start_pos)
        k = self.rope(k, start_pos)
        
        # 转置: (B, n_h, T, d_h)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 注意力分数
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        
        # 因果掩码
        if mask is None:
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Softmax + 加权求和
        attn = F.softmax(scores, dim=-1)
        output = (attn @ v).transpose(1, 2).contiguous().view(B, T, d)
        
        return self.wo(output)
```

#### 6.2.5 完整 LLaMA 模型

```python
class LLaMABlock(nn.Module):
    """
    LLaMA Transformer Block
    
    数学:
        h' = h + SelfAttn(RMSNorm(h))
        h_out = h' + SwiGLU-FFN(RMSNorm(h'))
    """
    
    def __init__(self, d: int, n_heads: int, d_ff: int, max_len: int = 2048):
        super().__init__()
        self.attention_norm = RMSNorm(d)
        self.attention = LLaMAAttention(d, n_heads, max_len)
        self.ffn_norm = RMSNorm(d)
        self.ffn = SwiGLU(d, d_ff)
    
    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        # Pre-Norm + 自注意力 + 残差
        h = x + self.attention(self.attention_norm(x), start_pos)
        # Pre-Norm + SwiGLU FFN + 残差
        out = h + self.ffn(self.ffn_norm(h))
        return out


class LLaMA(nn.Module):
    """
    完整 LLaMA 模型
    
    架构: Token Embedding → L × LLaMABlock → RMSNorm → Linear Head
    """
    
    def __init__(self, vocab_size: int, d: int = 4096, n_layers: int = 32,
                 n_heads: int = 32, max_len: int = 2048):
        super().__init__()
        d_ff = int(8 * d / 3)
        d_ff = ((d_ff + 255) // 256) * 256  # 对齐到 256 的倍数
        
        self.tok_emb = nn.Embedding(vocab_size, d)
        self.layers = nn.ModuleList([
            LLaMABlock(d, n_heads, d_ff, max_len) for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, vocab_size, bias=False)
        
        # 权重共享: 输出头与 embedding 共享
        self.head.weight = self.tok_emb.weight
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """
        参数:
            input_ids: (B, T) — token ids
            start_pos: 起始位置（增量解码用）
        返回:
            logits: (B, T, vocab_size)
        """
        x = self.tok_emb(input_ids)  # (B, T, d)
        
        for layer in self.layers:
            x = layer(x, start_pos)
        
        x = self.norm(x)             # 最终 RMSNorm
        logits = self.head(x)        # (B, T, vocab_size)
        return logits


# ===== 模型实例化与验证 =====
torch.manual_seed(42)
# 使用小规模配置验证
config = {
    'vocab_size': 1000,
    'd': 256,
    'n_layers': 4,
    'n_heads': 4,
    'max_len': 512,
}

model = LLaMA(**config)
n_params = sum(p.numel() for p in model.parameters())
print(f"LLaMA 模型参数量: {n_params:,} ({n_params/1e6:.1f}M)")
print(f"FFN 维度: {int(8*config['d']/3 + 255) // 256 * 256}")

# 前向传播测试
B, T = 4, 32
input_ids = torch.randint(0, config['vocab_size'], (B, T))
logits = model(input_ids)
print(f"输入: {input_ids.shape} → 输出: {logits.shape}")
print(f"Logits 统计: mean={logits.mean().item():.4f}, std={logits.std().item():.4f}")

# 因果语言模型损失
targets = torch.randint(0, config['vocab_size'], (B, T))
loss = F.cross_entropy(logits.view(-1, config['vocab_size']), targets.view(-1))
print(f"随机初始化损失: {loss.item():.4f} (理论值 ln({config['vocab_size']})={math.log(config['vocab_size']):.4f})")
```

#### 6.2.6 LLaMA 训练循环

```python
def train_llama(model, train_data, epochs=3, lr=3e-4, batch_size=8, max_len=128):
    """
    LLaMA 因果语言模型训练循环
    
    数学:
        L = -1/T Σ_t log p_θ(x_t | x_{<t})
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), 
                                   weight_decay=0.1)
    
    # Cosine 学习率调度（含 warmup）
    total_steps = epochs * (len(train_data) // batch_size)
    warmup_steps = min(2000, total_steps // 10)
    
    def lr_schedule(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    
    model.train()
    for epoch in range(epochs):
        total_loss, n_batches = 0.0, 0
        for i in range(0, len(train_data) - max_len, batch_size * max_len):
            # 构造 batch
            batch_ids = []
            for b in range(batch_size):
                start = i + b * max_len
                if start + max_len + 1 > len(train_data):
                    break
                batch_ids.append(train_data[start:start + max_len + 1])
            
            if len(batch_ids) == 0:
                break
            
            tokens = torch.stack(batch_ids)
            inputs = tokens[:, :-1]    # (B, T)
            targets = tokens[:, 1:]    # (B, T)
            
            logits = model(inputs)     # (B, T, V)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / max(n_batches, 1)
        ppl = math.exp(min(avg_loss, 20))
        print(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, ppl={ppl:.2f}, "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

# ===== 验证训练循环 =====
torch.manual_seed(42)
small_model = LLaMA(vocab_size=100, d=64, n_layers=2, n_heads=2, max_len=64)
fake_data = [torch.randint(0, 100, (256,)) for _ in range(10)]
fake_data = torch.cat(fake_data)
print(f"\n训练验证 (小规模):")
train_llama(small_model, fake_data, epochs=3, lr=1e-3, batch_size=4, max_len=32)
```

---

## 7. 实践技巧与可视化

### 7.1 RoPE 位置编码可视化

```python
import numpy as np

def visualize_rope_frequencies():
    """可视化 RoPE 不同频率分量的旋转角度"""
    d = 128
    max_pos = 100
    base = 10000.0
    
    freqs = 1.0 / (base ** (np.arange(0, d, 2) / d))
    positions = np.arange(max_pos)
    
    print("RoPE 频率分析:")
    print(f"{'维度对':>8} {'频率 θ':>12} {'周期 (tokens)':>15} {'类型':>8}")
    print("-" * 48)
    for i in [0, 1, 4, 16, 32, 63]:
        period = 2 * np.pi / freqs[i]
        kind = "高频" if i < 8 else ("中频" if i < 48 else "低频")
        print(f"{i:>8d} {freqs[i]:>12.6f} {period:>15.1f} {kind:>8}")
    
    # 位置内积衰减分析
    print(f"\n位置内积衰减 (d={d}):")
    for dist in [1, 5, 10, 50, 100, 500]:
        # 理论近似: 平均内积衰减
        cos_sum = np.mean([np.cos(dist * f) for f in freqs])
        print(f"  距离 {dist:>3d}: 平均 cos 分量 = {cos_sum:+.4f}")

visualize_rope_frequencies()
```

### 7.2 SwiGLU 激活函数对比

```python
def compare_activations():
    """对比不同激活函数的性质"""
    x = np.linspace(-4, 4, 9)
    
    relu = np.maximum(0, x)
    gelu = x * 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
    sigmoid = 1.0 / (1.0 + np.exp(-x))
    swish = x * sigmoid
    
    print(f"{'x':>6} {'ReLU':>8} {'GELU':>8} {'Swish':>8}")
    print("-" * 34)
    for i, xi in enumerate(x):
        print(f"{xi:>6.1f} {relu[i]:>8.4f} {gelu[i]:>8.4f} {swish[i]:>8.4f}")
    
    # 导数对比
    print(f"\n{'x':>6} {'ReLU导':>8} {'GELU导':>8} {'Swish导':>8}")
    print("-" * 34)
    dx = 1e-5
    for xi in [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]:
        r_grad = 1.0 if xi > 0 else 0.0
        g_p = (xi+dx) * 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * ((xi+dx) + 0.044715 * (xi+dx)**3)))
        g_m = (xi-dx) * 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * ((xi-dx) + 0.044715 * (xi-dx)**3)))
        g_grad = (g_p - g_m) / (2*dx)
        s = 1.0 / (1.0 + np.exp(-xi))
        s_grad = s + xi * s * (1 - s)
        print(f"{xi:>6.1f} {r_grad:>8.4f} {g_grad:>8.4f} {s_grad:>8.4f}")
    
    print("\n关键观察:")
    print("  • ReLU: x<0 梯度为 0 (死神经元问题)")
    print("  • GELU: x<0 有小负值 (非单调)")
    print("  • Swish: x<0 有小负值, x=0 处导数=0.5 (平滑过渡)")

compare_activations()
```

### 7.3 训练配置与工程细节

#### LLaMA 训练超参数

| 超参数 | 值 | 说明 |
|--------|-----|------|
| 优化器 | AdamW | $\beta_1=0.9$, $\beta_2=0.95$ |
| 权重衰减 | 0.1 | 应用于所有非嵌入权重 |
| 梯度裁剪 | 1.0 | 全局梯度范数裁剪 |
| 学习率调度 | Cosine | 最终学习率 = 初始的 10% |
| Warmup | 2000 steps | 线性预热 |
| Batch Size | 4M tokens | 固定 token 数，非固定序列数 |
| 序列长度 | 2048 | 所有模型统一 |
| 精度 | BFloat16 | 混合精度训练 |

#### 训练效率

| 模型 | GPU | 训练时间 | 总 GPU 时 | Token/s/GPU |
|------|:---:|:--------:|:---------:|:-----------:|
| LLaMA-7B | 2048× A100-80G | ~3 天 | ~82K | ~3350 |
| LLaMA-13B | 2048× A100-80G | ~5 天 | ~135K | ~2050 |
| LLaMA-33B | 2048× A100-80G | ~12 天 | ~530K | ~1100 |
| LLaMA-65B | 2048× A100-80G | ~21 天 | ~1022K | ~580 |

#### 训练稳定性技巧

1. **Pre-Norm**：归一化在子层之前，梯度更稳定
2. **无偏置项**：减少参数，简化实现
3. **权重共享**：输入 embedding 和输出 head 共享权重
4. **BFloat16**：比 Float16 有更大的指数范围，减少溢出

---

## 8. 与其他模型的关系

### 8.1 从 GPT 到 LLaMA 的架构演进

```
GPT-2 (2019)              GPT-3 (2020)              LLaMA (2023)
│                          │                          │
│ Post-Norm LayerNorm      │ Post-Norm LayerNorm      │ Pre-Norm RMSNorm
│ GELU 激活                │ GELU 激活                │ SwiGLU 激活
│ 绝对学习式位置编码         │ 绝对学习式位置编码         │ RoPE 旋转位置编码
│ FFN dim = 4d             │ FFN dim = 4d             │ FFN dim = 8d/3
│ 1.5B 参数                │ 175B 参数                │ 7B-65B 参数
│ 40GB 文本                │ 300B tokens              │ 1.0-1.4T tokens
│                          │                          │
└── 证明自回归可行 ──→      └── Scaling Laws ──→       └── 训练效率优先
```

**关键改进的影响**：

| 改进 | 效果 | 影响的后续模型 |
|------|------|---------------|
| RMSNorm | 训练加速 7-15% | LLaMA 2/3, Mistral, Qwen |
| SwiGLU | 下游任务性能 +1~2% | 几乎所有后续开源模型 |
| RoPE | 支持长度外推 | LLaMA 2/3, Mistral, DeepSeek |
| 更多训练数据 | 小模型达到大模型水平 | 整个开源社区 |

### 8.2 LLaMA 家族与开源生态

```
LLaMA (2023.02)
│
├── LLaMA 2 (2023.07)
│   ├── 扩展到 70B
│   ├── 训练 2T tokens
│   ├── GQA (Grouped-Query Attention)
│   └── 更宽松的开源许可
│
├── LLaMA 3 (2024.04)
│   ├── 扩展到 405B
│   ├── 训练 15T tokens
│   └── 128K 上下文窗口
│
├── 社区衍生
│   ├── Alpaca (Stanford) — SFT 微调
│   ├── Vicuna — ShareGPT 数据微调
│   ├── Llama.cpp — CPU 推理
│   └── GGML/GGUF — 量化格式
│
└── 架构影响
    ├── Mistral — 滑动窗口注意力 + GQA
    ├── Qwen — 类似架构 + 多语言
    └── DeepSeek — 类似架构 + MoE
```

### 8.3 Scaling Laws 的重新审视

LLaMA 的成功重新审视了 Scaling Laws 的应用方式：

**Kaplan et al. (2020) — OpenAI Scaling Laws**：

$$
L(N, D) = \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{D_c}{D}\right)^{\alpha_D} + L_\infty
$$

结论：优先增大模型（$N$），数据量（$D$）次之。

**Hoffmann et al. (2022) — Chinchilla Scaling Laws**：

$$
L(N, D) = A \cdot N^{-\alpha} + B \cdot D^{-\beta} + L_\infty
$$

结论：$N$ 和 $D$ 应等比例增长，$N_{\text{opt}} \approx D_{\text{opt}} / 20$。

**LLaMA 的实践结论**：

$$
\boxed{\text{推理最优} \neq \text{训练最优}: \quad D_{\text{LLaMA}} \gg D_{\text{Chinchilla-opt}}}
$$

> **关键洞察**：Chinchilla 最优是在固定计算预算下最小化训练损失，但 LLaMA 的目标是**最小化推理成本**——用更多训练计算换取更小的推理模型。对于部署而言，多训练一周 vs 永远运行一个更大的模型，前者更经济。

---

## 扩展阅读与实现

### Q1: RMSNorm 为什么不需要均值中心化？

> **Q:** LayerNorm 的均值中心化 ($x - \mu$) 在直觉上确保了零均值，去掉它真的没问题吗？
>
> **A:** Zhang & Sennrich (2019) 的实验表明，LayerNorm 的成功主要来自**缩放不变性**（re-scaling invariance），而非**重新中心化**（re-centering）。RMSNorm 保留了缩放不变性：
>
> $$\text{RMSNorm}(\alpha x) = \text{RMSNorm}(x) \quad \forall \alpha > 0$$
>
> 这意味着即使输入的绝对幅度变化，输出保持稳定。实验中，去除均值中心化后模型性能几乎无损（<0.1% 差异），但计算速度提升明显。

### Q2: SwiGLU 为什么比 GELU 更好？

> **Q:** SwiGLU 引入了额外的参数矩阵 $W_3$，它的优势是什么？
>
> **A:** SwiGLU 的优势来自**门控机制**：
>
> 1. **自适应特征选择**：$\text{Swish}(xW_1)$ 作为门控，可以选择性地激活 $xW_3$ 中的特征
> 2. **更丰富的非线性**：两个不同投影的逐元素乘法比单一非线性更具表达力
> 3. **梯度流更好**：Swish 在负值区域有非零梯度，避免 ReLU 的死神经元问题
>
> Shazeer (2020) 的消融实验显示，在控制参数量相同时，SwiGLU 在多个下游任务上一致优于 ReLU 和 GELU：
>
> $$\text{SwiGLU} > \text{GEGLU} > \text{ReGLU} > \text{GELU} > \text{ReLU}$$

### Q3: RoPE 如何支持长度外推？

> **Q:** LLaMA 训练时使用 2048 长度，但 RoPE 能否支持更长序列？
>
> **A:** RoPE 的数学结构使其天然支持外推：
>
> 1. **无学习参数**：频率 $\theta_i$ 是预设的，不依赖训练数据的长度
> 2. **连续位置编码**：位置 $m$ 可以取任意正整数，不像学习式编码有固定上限
> 3. **实践技巧**：
>    - **NTK-aware 缩放**：调整基频 $\text{base}' = \text{base} \times \alpha^{d/(d-2)}$ 来外推
>    - **YaRN**：结合 NTK 和注意力缩放
>    - LLaMA 2 使用此方法将上下文从 2K 扩展到 4K
>
> $$\text{NTK 缩放}: \quad \theta'_i = \frac{1}{\text{base}'^{2i/d}} = \frac{1}{(\text{base} \cdot \alpha^{d/(d-2)})^{2i/d}}$$

### Q4: LLaMA 的权重共享有什么数学意义？

> **Q:** 输入 embedding 和输出 head 共享权重 ($W_{\text{head}} = W_{\text{emb}}$) 的理论依据是什么？
>
> **A:** 权重共享基于以下直觉：
>
> - 输入 embedding 将 token 映射到语义空间：$e = W_{\text{emb}}[i] \in \mathbb{R}^d$
> - 输出 head 计算每个 token 的 logit：$z_i = h^\top W_{\text{head}}[i]$
> - 共享后：$z_i = h^\top W_{\text{emb}}[i] = h \cdot e_i$（语义相似度）
>
> 即输出 logit 就是隐藏状态与 token embedding 的内积，语义越接近的 token 分数越高。
> 此外，对于 LLaMA-7B，词表大小 32000 × 维度 4096 = 1.31 亿参数，共享节省约 2% 总参数。

### Q5: 为什么 LLaMA 去除了所有偏置项？

> **Q:** 标准 Transformer 中线性层都有偏置 $b$，LLaMA 去除它的原因是什么？
>
> **A:** 多个因素共同驱动了这一选择：
>
> 1. **与 RMSNorm 配合**：RMSNorm 没有偏移参数 $\beta$，整个模型保持一致的"无偏"设计
> 2. **简化实现**：减少需要管理的参数和超参数
> 3. **PaLM 的先例**：Google 的 PaLM (2022) 已验证去除偏置不影响性能
> 4. **张量并行友好**：无偏置使得模型并行更简单（不需要额外的 all-reduce）
>
> 参数节省量（以 LLaMA-7B 为例）：每层 ~20K 偏置参数 × 32 层 ≈ 640K，相对于 6.7B 总参数几乎可忽略，但工程上简化了很多。

---

## 参考资源

### 经典论文

1. Touvron et al. (2023). [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971). arXiv.
   - **贡献**：证明仅用公开数据即可训练出媲美闭源模型的开放大模型，推动了开源大模型生态

2. Hoffmann et al. (2022). [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556). NeurIPS 2022.
   - **贡献**：提出 Chinchilla Scaling Laws，证明数据量和模型大小应同步扩展

3. Zhang & Sennrich (2019). [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467). NeurIPS 2019.
   - **贡献**：提出 RMSNorm，证明归一化中均值中心化可以去除

4. Shazeer (2020). [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202). arXiv.
   - **贡献**：系统比较 GLU 变体，提出 SwiGLU 等高效激活函数

5. Su et al. (2024). [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864). Neurocomputing.
   - **贡献**：提出 RoPE，通过旋转矩阵优雅地实现相对位置编码

6. Touvron et al. (2023). [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288). arXiv.
   - **贡献**：扩展 LLaMA 至 70B，引入 GQA 和更宽松的开源许可

### 教材与书籍

7. Vaswani et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). NeurIPS 2017.
   - **章节**：原始 Transformer 架构，LLaMA 的基础

### 在线资源与教程

8. Meta AI. [LLaMA 官方代码](https://github.com/facebookresearch/llama).
   - **内容**：LLaMA 官方 PyTorch 实现和预训练权重

9. Karpathy. [llama2.c](https://github.com/karpathy/llama2.c).
   - **内容**：纯 C 语言实现的 LLaMA 2 推理，极简教学代码

10. Georgi Gerganov. [llama.cpp](https://github.com/ggerganov/llama.cpp).
    - **内容**：CPU 推理优化实现，支持量化和多平台部署

11. EleutherAI. [Rotary Embeddings: A Relative Revolution](https://blog.eleuther.ai/rotary-embeddings/).
    - **内容**：RoPE 的直观解释和数学推导

---

## 附录：符号表

| 符号 | 含义 | 维度/类型 |
|------|------|----------|
| $d$ ($d_{\text{model}}$) | 模型隐藏维度 | 标量 |
| $d_{ff}$ | FFN 中间维度 | 标量，$\approx \frac{8d}{3}$ |
| $d_h$ | 注意力头维度 | 标量，$d_h = d / n_h$ |
| $L$ | Transformer 层数 | 标量 |
| $n_h$ | 注意力头数 | 标量 |
| $T$ | 序列长度 | 标量 |
| $|\mathcal{V}|$ | 词表大小 | 标量 |
| $x$ | 隐藏状态输入 | $(B, T, d)$ |
| $h_l$ | 第 $l$ 层输出 | $(B, T, d)$ |
| $\gamma$ | RMSNorm 缩放参数 | $(d,)$ |
| $\text{RMS}(x)$ | 均方根值 | 标量 |
| $\epsilon$ | 数值稳定常数 | 标量，通常 $10^{-6}$ |
| $W_1$ | SwiGLU 门控投影 | $(d, d_{ff})$ |
| $W_2$ | SwiGLU 下投影 | $(d_{ff}, d)$ |
| $W_3$ | SwiGLU 上投影 | $(d, d_{ff})$ |
| $\sigma(\cdot)$ | Sigmoid 函数 | 函数 |
| $\text{Swish}(x)$ | Swish 激活：$x \cdot \sigma(x)$ | 函数 |
| $\theta_i$ | RoPE 第 $i$ 个频率参数 | 标量，$\theta_i = 10000^{-2(i-1)/d}$ |
| $R_m$ | 位置 $m$ 的旋转矩阵 | $(d, d)$ 正交矩阵 |
| $Q, K, V$ | 查询、键、值矩阵 | $(B, T, n_h, d_h)$ |
| $W_q, W_k, W_v, W_o$ | 注意力投影矩阵 | $(d, d)$ |
| $N$ | 模型参数量 | 标量 |
| $D$ | 训练数据量（tokens） | 标量 |
| $C$ | 总计算量（FLOPs） | 标量 |
| $\mathcal{L}$ | 损失函数值 | 标量 |

**典型维度示例（LLaMA-7B）：**
- $d = 4096$，$L = 32$，$n_h = 32$，$d_h = 128$
- $d_{ff} = 11008$（$\frac{8 \times 4096}{3} \approx 10923$，取 256 倍数）
- $|\mathcal{V}| = 32{,}000$（SentencePiece BPE）
- 训练数据：1.0T tokens（公开数据集混合）
- 训练硬件：2048× A100-80GB
- 总参数量：6.7B

---

最后更新：2026-03-19
