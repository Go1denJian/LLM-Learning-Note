# LLaMA 架构与实现 —— 开源高效大模型的完整数学推导

> **前置知识**：Transformer 架构（自注意力、前馈网络）、LayerNorm、位置编码基础、预训练范式、Python 基础  
> **与前面内容的联系**：建议先学习 [Transformer-Math-and-Implementation](./06-Transformer-Math-and-Implementation.md) 理解标准 Transformer 架构，以及 [RLHF-Math-and-Implementation](./13-RLHF-Math-and-Implementation.md) 理解对齐训练  
> **与后续内容的联系**：LLaMA 的架构创新（RMSNorm、SwiGLU、RoPE）成为后续几乎所有开源大模型的标准配置，直接影响 LLaMA-2、Mistral、DeepSeek 等模型

---

## 目录

1. [引言：为什么需要开源高效大模型？](#1-引言为什么需要开源高效大模型)
   - 1.1 [大模型的封闭困境](#11-大模型的封闭困境)
   - 1.2 [LLaMA 的设计哲学：推理效率优先](#12-llama-的设计哲学推理效率优先)
   - 1.3 [三大架构创新概览](#13-三大架构创新概览)
   - 1.4 [本科数学知识映射表](#14-本科数学知识映射表)
2. [基础概念：从标准 Transformer 到 LLaMA](#2-基础概念从标准-transformer-到-llama)
   - 2.1 [标准 Transformer Decoder 回顾](#21-标准-transformer-decoder-回顾)
   - 2.2 [LLaMA 的架构改进总览](#22-llama-的架构改进总览)
   - 2.3 [模型规模与训练数据](#23-模型规模与训练数据)
3. [核心创新一：RMSNorm 替代 LayerNorm](#3-核心创新一rmsnorm-替代-layernorm)
   - 3.1 [LayerNorm 的数学回顾](#31-layernorm-的数学回顾)
   - 3.2 [RMSNorm 的数学定义](#32-rmsnorm-的数学定义)
   - 3.3 [Pre-Norm 与 Post-Norm 的区别](#33-pre-norm-与-post-norm-的区别)
   - 3.4 [RMSNorm 的梯度推导](#34-rmsnorm-的梯度推导)
4. [核心创新二：SwiGLU 激活函数](#4-核心创新二swiglu-激活函数)
   - 4.1 [从 ReLU 到 Swish 的激活函数演进](#41-从-relu-到-swish-的激活函数演进)
   - 4.2 [GLU 门控线性单元](#42-glu-门控线性单元)
   - 4.3 [SwiGLU 的数学定义与性质](#43-swiglu-的数学定义与性质)
   - 4.4 [SwiGLU FFN 的梯度推导](#44-swiglu-ffn-的梯度推导)
   - 4.5 [隐藏层维度调整：参数量守恒](#45-隐藏层维度调整参数量守恒)
5. [核心创新三：RoPE 旋转位置编码](#5-核心创新三rope-旋转位置编码)
   - 5.1 [位置编码的需求分析](#51-位置编码的需求分析)
   - 5.2 [绝对位置编码的局限性](#52-绝对位置编码的局限性)
   - 5.3 [RoPE 的核心思想：旋转矩阵](#53-rope-的核心思想旋转矩阵)
   - 5.4 [RoPE 的完整数学推导](#54-rope-的完整数学推导)
   - 5.5 [RoPE 的长距离衰减性质](#55-rope-的长距离衰减性质)
   - 5.6 [RoPE 的高效实现](#56-rope-的高效实现)
6. [从数学到代码：完整实现](#6-从数学到代码完整实现)
   - 6.1 [NumPy 实现核心组件](#61-numpy-实现核心组件)
   - 6.2 [PyTorch 完整 LLaMA 实现](#62-pytorch-完整-llama-实现)
7. [实践技巧与可视化](#7-实践技巧与可视化)
   - 7.1 [RoPE 旋转可视化](#71-rope-旋转可视化)
   - 7.2 [SwiGLU 与 ReLU 的激活分布对比](#72-swiglu-与-relu-的激活分布对比)
   - 7.3 [训练效率与 Scaling 分析](#73-训练效率与-scaling-分析)
8. [与其他模型的关系](#8-与其他模型的关系)
   - 8.1 [从 GPT 到 LLaMA 的架构演进](#81-从-gpt-到-llama-的架构演进)
   - 8.2 [LLaMA 架构的后续影响](#82-llama-架构的后续影响)
   - 8.3 [开源大模型谱系](#83-开源大模型谱系)

[扩展阅读与实现](#扩展阅读与实现)

[参考资源](#参考资源)

附录：[符号表](#附录符号表)

---

## 1. 引言：为什么需要开源高效大模型？

### 1.1 大模型的封闭困境

2023 年初，大语言模型领域面临严峻的**封闭性问题**：

| 模型 | 参数量 | 是否开源 | 训练数据 | 可复现性 |
|------|:------:|:--------:|----------|:--------:|
| GPT-3 | 175B | ❌ | 未公开 | ❌ |
| PaLM | 540B | ❌ | 未公开 | ❌ |
| Chinchilla | 70B | ❌ | 未公开 | ❌ |
| ChatGPT | ~175B | ❌ | 未公开 | ❌ |

> **核心矛盾**：最强大的语言模型被少数机构垄断，学术界无法复现、验证和改进。

LLaMA（Large Language Model Meta AI）的发布打破了这一僵局：

- **完全开源**：模型权重对研究社区开放
- **高效训练**：仅用公开数据，训练效率极高
- **性能惊人**：LLaMA-13B 超越 GPT-3（175B）

### 1.2 LLaMA 的设计哲学：推理效率优先

Touvron et al. (2023) 的关键洞察来自 Hoffmann et al. (2022) 的 **Chinchilla Scaling Laws**：

**传统观点**（Kaplan et al., 2020）：
$$
\text{给定计算预算} \; C, \quad \text{增大模型} \; N \; \text{比增加数据} \; D \; \text{更有效}
$$

**Chinchilla 修正**：
$$
\boxed{N_{\text{opt}} \propto C^{0.5}, \quad D_{\text{opt}} \propto C^{0.5}}
$$

即模型大小和训练数据量应该**同比例增长**。

LLaMA 的进一步洞察：

$$
\boxed{\text{推理成本} \propto N, \quad \text{训练成本} \propto N \times D}
$$

> **关键决策**：在推理预算固定时，用**更小的模型 + 更多的训练数据**可以获得更好的性能。这意味着训练时间更长，但推理更便宜。

| 模型 | 参数量 | 训练 Tokens | 推理成本 | 性能 |
|------|:------:|:-----------:|:--------:|:----:|
| Chinchilla | 70B | 1.4T | 高 | 基准 |
| **LLaMA-13B** | **13B** | **1.0T** | **低** | **≈ GPT-3** |
| **LLaMA-65B** | **65B** | **1.4T** | **中** | **≈ Chinchilla** |

### 1.3 三大架构创新概览

LLaMA 相对于标准 GPT 架构引入了三个关键改进：

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LLaMA 架构创新                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. RMSNorm 替代 LayerNorm                                         │
│  ┌──────────────────────────────────────────┐                       │
│  │  去除均值中心化 → 减少 15% 计算量           │                       │
│  │  Pre-Norm 位置 → 训练更稳定                │                       │
│  └──────────────────────────────────────────┘                       │
│                                                                     │
│  2. SwiGLU 激活函数                                                  │
│  ┌──────────────────────────────────────────┐                       │
│  │  Swish + GLU 门控 → 比 ReLU/GELU 更优     │                       │
│  │  三矩阵 FFN → 更强的表达能力                │                       │
│  └──────────────────────────────────────────┘                       │
│                                                                     │
│  3. RoPE 旋转位置编码                                                │
│  ┌──────────────────────────────────────────┐                       │
│  │  旋转矩阵编码位置 → 天然相对位置感知         │                       │
│  │  长距离衰减 → 良好的外推性                  │                       │
│  └──────────────────────────────────────────┘                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.4 本科数学知识映射表

| LLaMA 概念 | 对应数学 | 本科课程 |
|------------|----------|----------|
| RMSNorm | 均方根、向量范数 | 线性代数 |
| SwiGLU | Sigmoid、门控机制、逐元素乘法 | 微积分、信号处理 |
| RoPE | 旋转矩阵、复数乘法、欧拉公式 | 线性代数、复变函数 |
| 注意力机制 | 矩阵乘法、softmax | 线性代数、概率论 |
| Scaling Laws | 幂律关系、回归分析 | 统计学 |
| 梯度推导 | 链式法则、矩阵微积分 | 高等数学 |

---

## 2. 基础概念：从标准 Transformer 到 LLaMA

### 2.1 标准 Transformer Decoder 回顾

GPT 系列使用的标准 Transformer Decoder 块：

$$
\text{GPT Block}(x) = x + \text{FFN}\left(\text{LN}\left(x + \text{MHA}\left(\text{LN}(x)\right)\right)\right)
$$

其中：
- $\text{LN}$：LayerNorm
- $\text{MHA}$：多头因果自注意力
- $\text{FFN}$：前馈网络 $\text{FFN}(x) = \max(0, xW_1 + b_1) W_2 + b_2$（ReLU 激活）

**标准自注意力**（单头）：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

其中 $Q = xW_Q$, $K = xW_K$, $V = xW_V$，$d_k = d_{\text{model}} / n_h$。

### 2.2 LLaMA 的架构改进总览

LLaMA 对标准 Decoder 做了以下修改：

| 组件 | GPT 标准 | LLaMA | 改进原因 |
|------|----------|-------|----------|
| **归一化** | LayerNorm (Post-Norm) | RMSNorm (Pre-Norm) | 计算更快，训练更稳定 |
| **激活函数** | ReLU / GELU | SwiGLU | 实验效果更好 |
| **位置编码** | 绝对位置嵌入 | RoPE | 更好的相对位置建模 |
| **偏置项** | 有偏置 | 无偏置 | 减少参数，配合 RMSNorm |
| **注意力** | 标准 MHA | MHA（后续 GQA） | 推理效率 |

**LLaMA Decoder 块**：

$$
h' = x + \text{MHA}\left(\text{RMSNorm}(x)\right)
$$

$$
\text{LLaMA Block}(x) = h' + \text{SwiGLU-FFN}\left(\text{RMSNorm}(h')\right)
$$

### 2.3 模型规模与训练数据

**LLaMA 模型家族**：

| 模型 | 参数量 | 层数 $L$ | $d_{\text{model}}$ | 头数 $n_h$ | $d_{\text{head}}$ | $d_{\text{ffn}}$ | 训练 Tokens |
|------|:------:|:--------:|:-------------------:|:----------:|:------------------:|:-----------------:|:-----------:|
| LLaMA-7B | 6.7B | 32 | 4096 | 32 | 128 | 11008 | 1.0T |
| LLaMA-13B | 13.0B | 40 | 5120 | 40 | 128 | 13824 | 1.0T |
| LLaMA-33B | 32.5B | 60 | 6656 | 52 | 128 | 17920 | 1.4T |
| LLaMA-65B | 65.2B | 80 | 8192 | 64 | 128 | 22016 | 1.4T |

**训练数据**（全部公开数据）：

| 数据集 | 采样比例 | 大小 | Epochs |
|--------|:--------:|:----:|:------:|
| CommonCrawl (CCNet) | 67.0% | 3.3T tokens | ~1.1 |
| C4 | 15.0% | 783B tokens | ~1.1 |
| GitHub | 4.5% | 328B tokens | ~0.6 |
| Wikipedia | 4.5% | 83B tokens | ~2.5 |
| Books | 4.5% | 85B tokens | ~2.2 |
| ArXiv | 2.5% | 92B tokens | ~1.1 |
| StackExchange | 2.0% | 78B tokens | ~1.1 |

> **重要发现**：即使训练 1T tokens 后，模型的 loss 仍在下降——表明更多数据仍然有益。

---

## 3. 核心创新一：RMSNorm 替代 LayerNorm

### 3.1 LayerNorm 的数学回顾

标准 LayerNorm 对隐藏向量 $x \in \mathbb{R}^d$ 做归一化：

$$
\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

其中：

$$
\mu = \frac{1}{d} \sum_{i=1}^d x_i, \quad \sigma^2 = \frac{1}{d} \sum_{i=1}^d (x_i - \mu)^2
$$

- $\gamma, \beta \in \mathbb{R}^d$：可学习的缩放和偏移参数
- $\epsilon$：数值稳定常数（如 $10^{-6}$）

**计算成本**：需要两次遍历（计算 $\mu$ 和 $\sigma^2$），涉及减法和平方运算。

### 3.2 RMSNorm 的数学定义

Zhang & Sennrich (2019) 提出 **RMSNorm**，去除均值中心化，仅使用均方根归一化：

$$
\boxed{\text{RMSNorm}(x) = \gamma \odot \frac{x}{\text{RMS}(x) + \epsilon}}
$$

其中均方根（Root Mean Square）：

$$
\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2}
$$

**关键区别**：

| 特性 | LayerNorm | RMSNorm |
|------|-----------|---------|
| 均值中心化 | ✅ 减去 $\mu$ | ❌ 不需要 |
| 方差归一化 | ✅ 除以 $\sigma$ | ✅ 除以 RMS |
| 偏移参数 $\beta$ | ✅ | ❌ |
| 缩放参数 $\gamma$ | ✅ | ✅ |
| 计算量 | $3d$ 次操作 | $2d$ 次操作 |
| 速度提升 | 基准 | **~15% 更快** |

> **为什么去掉均值中心化仍然有效？** Zhang & Sennrich (2019) 实验表明，LayerNorm 的成功主要归功于**缩放不变性**（rescaling invariance），而非**重新中心化**（re-centering）。RMSNorm 保留了缩放不变性，同时减少了计算。

**缩放不变性证明**：

对于任意常数 $a > 0$：

$$
\text{RMSNorm}(ax) = \gamma \odot \frac{ax}{\sqrt{\frac{1}{d}\sum_i (ax_i)^2}} = \gamma \odot \frac{ax}{a \cdot \text{RMS}(x)} = \text{RMSNorm}(x)
$$

### 3.3 Pre-Norm 与 Post-Norm 的区别

LLaMA 使用 **Pre-Norm**（归一化在子层之前），而非原始 Transformer 的 Post-Norm：

**Post-Norm**（原始 Transformer）：
$$
x_{l+1} = \text{Norm}(x_l + \text{SubLayer}(x_l))
$$

**Pre-Norm**（LLaMA 采用）：
$$
x_{l+1} = x_l + \text{SubLayer}(\text{Norm}(x_l))
$$

**Pre-Norm 的梯度优势**：

对于 $L$ 层 Pre-Norm 网络：

$$
\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L} + \sum_{k=l}^{L-1} \frac{\partial \mathcal{L}}{\partial x_L} \cdot \frac{\partial \text{SubLayer}_k}{\partial x_l}
$$

第一项 $\frac{\partial \mathcal{L}}{\partial x_L}$ 是**恒等映射**贡献的梯度直通路径，确保梯度不会在深层网络中消失。

> **LLaMA 的选择**：Pre-RMSNorm = 训练稳定性（Pre-Norm） + 计算效率（RMSNorm）

### 3.4 RMSNorm 的梯度推导

设 $y = \text{RMSNorm}(x) = \gamma \odot \frac{x}{\text{RMS}(x)}$，令 $s = \text{RMS}(x) = \sqrt{\frac{1}{d} \|x\|^2}$。

则 $y_i = \gamma_i \cdot \frac{x_i}{s}$。

**对 $x_j$ 求导**：

$$
\frac{\partial y_i}{\partial x_j} = \gamma_i \left( \frac{\delta_{ij}}{s} - \frac{x_i}{s^2} \cdot \frac{\partial s}{\partial x_j} \right)
$$

其中：

$$
\frac{\partial s}{\partial x_j} = \frac{1}{2} \cdot \frac{1}{\sqrt{\frac{1}{d}\|x\|^2}} \cdot \frac{2x_j}{d} = \frac{x_j}{d \cdot s}
$$

代入：

$$
\frac{\partial y_i}{\partial x_j} = \gamma_i \left( \frac{\delta_{ij}}{s} - \frac{x_i x_j}{d \cdot s^3} \right)
$$

用矩阵形式表达：

$$
\boxed{\frac{\partial y}{\partial x} = \frac{1}{s} \text{diag}(\gamma) \left( I - \frac{x x^\top}{d \cdot s^2} \right)}
$$

**反向传播**（给定上游梯度 $\frac{\partial \mathcal{L}}{\partial y}$）：

$$
\boxed{\frac{\partial \mathcal{L}}{\partial x_i} = \frac{\gamma_i}{s} \left( \frac{\partial \mathcal{L}}{\partial y_i} - \frac{y_i}{d \cdot s} \sum_{j=1}^d \frac{\partial \mathcal{L}}{\partial y_j} \cdot x_j \right)}
$$

> **计算复杂度**：RMSNorm 反向传播只需 $O(d)$ 操作（一次内积 + 逐元素运算），与 LayerNorm 相同量级但常数更小。

**对 $\gamma$ 求导**：

$$
\frac{\partial \mathcal{L}}{\partial \gamma_i} = \frac{\partial \mathcal{L}}{\partial y_i} \cdot \frac{x_i}{s}
$$

---

## 4. 核心创新二：SwiGLU 激活函数

### 4.1 从 ReLU 到 Swish 的激活函数演进

**ReLU**（最简单）：

$$
\text{ReLU}(x) = \max(0, x)
$$

- 优点：计算快，缓解梯度消失
- 缺点：$x < 0$ 时梯度为零（"死神经元"问题）

**GELU**（GPT/BERT 使用）：

$$
\text{GELU}(x) = x \cdot \Phi(x) \approx x \cdot \sigma(1.702x)
$$

其中 $\Phi(x)$ 是标准正态 CDF。

- 优点：平滑，有概率解释
- 缺点：计算较 ReLU 慢

**Swish**（也称 SiLU）：

$$
\boxed{\text{Swish}(x) = x \cdot \sigma(\beta x)}
$$

其中 $\sigma(z) = \frac{1}{1+e^{-z}}$ 是 sigmoid 函数。LLaMA 中 $\beta = 1$：

$$
\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
$$

**Swish 的导数**：

$$
\text{SiLU}'(x) = \sigma(x) + x \cdot \sigma(x)(1 - \sigma(x)) = \sigma(x)(1 + x(1 - \sigma(x)))
$$

$$
\boxed{\text{SiLU}'(x) = \sigma(x) + x \cdot \sigma(x) \cdot (1 - \sigma(x))}
$$

> **关键性质**：Swish 在 $x < 0$ 时不完全为零，允许少量负值通过（non-monotonic），这有助于梯度流动。

### 4.2 GLU 门控线性单元

Dauphin et al. (2017) 提出 **GLU（Gated Linear Unit）**：

$$
\text{GLU}(x) = (xW_1 + b_1) \otimes \sigma(xW_2 + b_2)
$$

其中 $\otimes$ 表示逐元素乘法（Hadamard 积）。

**GLU 的核心思想**：将输入分为两路——一路作为"内容"，另一路通过 sigmoid 作为"门控"，控制信息流。

**变体**：将 sigmoid 替换为其他激活函数：

| 名称 | 门控激活 | 公式 |
|------|----------|------|
| GLU | $\sigma$ | $(xW_1) \otimes \sigma(xW_2)$ |
| ReGLU | ReLU | $(xW_1) \otimes \text{ReLU}(xW_2)$ |
| GEGLU | GELU | $(xW_1) \otimes \text{GELU}(xW_2)$ |
| **SwiGLU** | **Swish** | $(\text{Swish}(xW_1)) \otimes (xW_2)$ |

### 4.3 SwiGLU 的数学定义与性质

Shazeer (2020) 提出 **SwiGLU**，在 LLaMA 中的具体形式为：

$$
\boxed{\text{SwiGLU-FFN}(x) = \left(\text{SiLU}(xW_{\text{gate}}) \otimes xW_{\text{up}}\right) W_{\text{down}}}
$$

展开：

$$
\text{SwiGLU-FFN}(x) = \left(\frac{xW_{\text{gate}}}{1 + e^{-xW_{\text{gate}}}} \otimes xW_{\text{up}}\right) W_{\text{down}}
$$

其中：
- $W_{\text{gate}} \in \mathbb{R}^{d \times d_{\text{ffn}}}$：门控投影矩阵
- $W_{\text{up}} \in \mathbb{R}^{d \times d_{\text{ffn}}}$：上投影矩阵
- $W_{\text{down}} \in \mathbb{R}^{d_{\text{ffn}} \times d}$：下投影矩阵

**对比标准 FFN**：

| 组件 | 标准 FFN | SwiGLU FFN |
|------|----------|------------|
| 投影矩阵 | 2 个 ($W_1, W_2$) | 3 个 ($W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$) |
| 激活函数 | ReLU/GELU | SiLU（内置在门控中） |
| 参数量 | $2 \times d \times d_{\text{ffn}}$ | $3 \times d \times d_{\text{ffn}}$ |
| 表达能力 | 标准 | 更强（门控机制） |

### 4.4 SwiGLU FFN 的梯度推导

设中间变量：
- $g = xW_{\text{gate}} \in \mathbb{R}^{d_{\text{ffn}}}$（门控预激活）
- $u = xW_{\text{up}} \in \mathbb{R}^{d_{\text{ffn}}}$（上投影）
- $a = \text{SiLU}(g) = g \cdot \sigma(g)$（门控激活）
- $m = a \otimes u$（门控结果）
- $y = mW_{\text{down}} \in \mathbb{R}^d$（输出）

**反向传播**（给定上游梯度 $\frac{\partial \mathcal{L}}{\partial y}$）：

$$
\frac{\partial \mathcal{L}}{\partial m} = \frac{\partial \mathcal{L}}{\partial y} W_{\text{down}}^\top
$$

$$
\frac{\partial \mathcal{L}}{\partial a} = \frac{\partial \mathcal{L}}{\partial m} \otimes u, \quad \frac{\partial \mathcal{L}}{\partial u} = \frac{\partial \mathcal{L}}{\partial m} \otimes a
$$

对于 SiLU 激活的梯度：

$$
\frac{\partial \mathcal{L}}{\partial g} = \frac{\partial \mathcal{L}}{\partial a} \otimes \text{SiLU}'(g) = \frac{\partial \mathcal{L}}{\partial a} \otimes \left[\sigma(g) + g \cdot \sigma(g)(1 - \sigma(g))\right]
$$

最终对输入和权重的梯度：

$$
\boxed{\frac{\partial \mathcal{L}}{\partial W_{\text{gate}}} = x^\top \frac{\partial \mathcal{L}}{\partial g}, \quad \frac{\partial \mathcal{L}}{\partial W_{\text{up}}} = x^\top \frac{\partial \mathcal{L}}{\partial u}, \quad \frac{\partial \mathcal{L}}{\partial W_{\text{down}}} = m^\top \frac{\partial \mathcal{L}}{\partial y}}
$$

$$
\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial g} W_{\text{gate}}^\top + \frac{\partial \mathcal{L}}{\partial u} W_{\text{up}}^\top
$$

### 4.5 隐藏层维度调整：参数量守恒

SwiGLU 引入了第三个权重矩阵，为保持与标准 FFN 相近的参数量，LLaMA 调整了 FFN 隐藏层维度：

**标准 FFN 参数量**（$d_{\text{ffn}} = 4d$）：

$$
P_{\text{std}} = 2 \times d \times 4d = 8d^2
$$

**SwiGLU FFN 参数量**（$d_{\text{ffn}} = d'$）：

$$
P_{\text{swiglu}} = 3 \times d \times d'
$$

令 $P_{\text{swiglu}} \approx P_{\text{std}}$：

$$
3d \cdot d' = 8d^2 \implies d' = \frac{8d}{3} \approx 2.67d
$$

LLaMA 实际使用 $d_{\text{ffn}} = \frac{2}{3} \times 4d$，并向上取整到 256 的倍数（硬件对齐优化）：

$$
\boxed{d_{\text{ffn}} = \text{round\_up}\left(\frac{8d}{3}, 256\right)}
$$

**验证**（以 LLaMA-7B 为例，$d = 4096$）：

$$
d_{\text{ffn}} = \text{round\_up}\left(\frac{8 \times 4096}{3}, 256\right) = \text{round\_up}(10922.67, 256) = 11008
$$

与论文中 $d_{\text{ffn}} = 11008$ 一致 ✅

---

## 5. 核心创新三：RoPE 旋转位置编码

### 5.1 位置编码的需求分析

自注意力机制本身是**位置无关**的：

$$
\text{Attention}(\{x_{\pi(1)}, \ldots, x_{\pi(T)}\}) = \pi^{-1}(\text{Attention}(\{x_1, \ldots, x_T\}))
$$

对于任意排列 $\pi$，注意力输出仅是输入排列的重新排列——**无法区分不同位置**。

因此需要位置编码将位置信息注入模型。理想的位置编码应满足：

1. **唯一性**：不同位置有不同编码
2. **相对性**：注意力应能感知位置间的相对距离
3. **泛化性**：能处理训练时未见过的序列长度
4. **高效性**：计算开销小

### 5.2 绝对位置编码的局限性

**正弦位置编码**（原始 Transformer）：

$$
\text{PE}(t, 2k) = \sin\left(\frac{t}{10000^{2k/d}}\right), \quad \text{PE}(t, 2k+1) = \cos\left(\frac{t}{10000^{2k/d}}\right)
$$

**可学习位置嵌入**（GPT 系列）：

$$
h_t^{(0)} = x_t + p_t, \quad p_t \in \mathbb{R}^d \text{ 是可学习参数}
$$

**绝对位置编码的问题**：

1. **加法注入 → 位置与内容纠缠**：
   $$
   q_m^\top k_n = (x_m + p_m)^\top W_Q^\top W_K (x_n + p_n)
   $$
   展开后产生四个交叉项，位置和内容的信息混合在一起。

2. **长度外推困难**：训练时见过 $T=2048$，推理时 $T=4096$ 需要新的位置嵌入。

3. **无法直接建模相对位置**：$q_m^\top k_n$ 依赖 $m, n$ 的绝对值，而非 $m - n$。

### 5.3 RoPE 的核心思想：旋转矩阵

Su et al. (2021) 提出的 **RoPE（Rotary Position Embedding）** 核心思想：

> **用旋转操作编码位置**，使得内积自然地只依赖相对位置。

**目标**：找到一种位置编码函数 $f(x, m)$，使得：

$$
\boxed{\langle f(q, m), f(k, n) \rangle = g(q, k, m - n)}
$$

即编码后的内积仅依赖**相对位置** $m - n$。

**二维情况的直觉**：

考虑二维向量 $x = [x_1, x_2]^\top$，将其视为复数 $x_1 + ix_2$。位置 $m$ 的旋转：

$$
f(x, m) = x \cdot e^{im\theta} = \begin{bmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}
$$

两个旋转后向量的内积：

$$
\langle f(q, m), f(k, n) \rangle = \langle R_m q, R_n k \rangle = q^\top R_m^\top R_n k = q^\top R_{n-m} k
$$

因为旋转矩阵 $R_m^\top R_n = R_{n-m}$（正交矩阵性质），内积**自然只依赖 $n - m$**。

### 5.4 RoPE 的完整数学推导

**高维推广**：将 $d$ 维向量分成 $d/2$ 对，每对独立旋转：

对于第 $k$ 对维度（$k = 0, 1, \ldots, d/2 - 1$），定义旋转角频率：

$$
\theta_k = \frac{1}{10000^{2k/d}}
$$

位置 $m$ 的旋转矩阵：

$$
\boxed{R_m = \begin{bmatrix} R_m^{(0)} & & & \\ & R_m^{(1)} & & \\ & & \ddots & \\ & & & R_m^{(d/2-1)} \end{bmatrix}}
$$

其中每个 $2 \times 2$ 块：

$$
R_m^{(k)} = \begin{bmatrix} \cos(m\theta_k) & -\sin(m\theta_k) \\ \sin(m\theta_k) & \cos(m\theta_k) \end{bmatrix}
$$

**RoPE 编码**：

$$
\boxed{f_{\text{RoPE}}(x, m) = R_m \cdot x}
$$

**注意力分数**：

$$
a_{m,n} = (R_m W_Q x_m)^\top (R_n W_K x_n) = x_m^\top W_Q^\top R_m^\top R_n W_K x_n
$$

$$
\boxed{a_{m,n} = x_m^\top W_Q^\top R_{n-m} W_K x_n = g(x_m, x_n, n-m)}
$$

> **证明内积只依赖相对位置**：
>
> $$R_m^\top R_n = \text{diag}(R_m^{(0)\top} R_n^{(0)}, \ldots, R_m^{(d/2-1)\top} R_n^{(d/2-1)})$$
>
> 对每个块：$R_m^{(k)\top} R_n^{(k)} = R_{n-m}^{(k)}$（旋转矩阵的正交性）
>
> 因此 $R_m^\top R_n = R_{n-m}$ ✅

**复数形式**（等价但更紧凑）：

将 $d$ 维实向量视为 $d/2$ 维复向量 $\tilde{x}_k = x_{2k} + i \cdot x_{2k+1}$，则：

$$
f_{\text{RoPE}}(\tilde{x}, m)_k = \tilde{x}_k \cdot e^{im\theta_k}
$$

注意力内积（取实部）：

$$
\text{Re}\left[\sum_{k=0}^{d/2-1} \tilde{q}_k \cdot \overline{\tilde{k}_k} \cdot e^{i(m-n)\theta_k}\right]
$$

### 5.5 RoPE 的长距离衰减性质

RoPE 的一个重要性质是**注意力分数随距离衰减**。

对于随机初始化的 $q, k$，注意力分数的期望：

$$
\mathbb{E}[a_{m,n}] = \mathbb{E}\left[\sum_{k=0}^{d/2-1} q^\top R_{m-n}^{(k)} k\right]
$$

当 $|m-n|$ 增大时，不同频率的旋转角 $\{(m-n)\theta_k\}$ 分散在 $[0, 2\pi]$ 上，导致求和的各项趋于**相互抵消**：

$$
\boxed{\mathbb{E}\left[\langle f(q, m), f(k, n) \rangle\right] \to 0 \quad \text{as} \quad |m - n| \to \infty}
$$

> **直觉**：RoPE 天然具有"近处关注多、远处关注少"的归纳偏置，与自然语言的局部性一致。

**频率设计的含义**：

- 低频维度（$k$ 小，$\theta_k$ 大）：旋转快，捕捉**短距离**位置关系
- 高频维度（$k$ 大，$\theta_k$ 小）：旋转慢，捕捉**长距离**位置关系

$$
\theta_0 = 1 \text{（最快旋转）}, \quad \theta_{d/2-1} = 10000^{-1+2/d} \approx 10000^{-1} \text{（最慢旋转）}
$$

### 5.6 RoPE 的高效实现

直接构造 $d \times d$ 旋转矩阵并做矩阵乘法的复杂度是 $O(d^2)$，但利用块对角结构可以降到 $O(d)$：

**高效实现公式**：

将向量 $x$ 分为前后两半 $x = [x_1, x_2, \ldots, x_{d/2}, x_{d/2+1}, \ldots, x_d]$：

$$
\boxed{f_{\text{RoPE}}(x, m) = x \otimes \cos(\Theta_m) + \text{rotate\_half}(x) \otimes \sin(\Theta_m)}
$$

其中：
- $\Theta_m = [m\theta_0, m\theta_0, m\theta_1, m\theta_1, \ldots, m\theta_{d/2-1}, m\theta_{d/2-1}]$
- $\text{rotate\_half}(x) = [-x_2, x_1, -x_4, x_3, \ldots, -x_d, x_{d-1}]$

> **复杂度**：仅需 $O(d)$ 的逐元素乘法和加法，与标准的线性投影相比开销可忽略。

---

## 6. 从数学到代码：完整实现

### 6.1 NumPy 实现核心组件

#### 6.1.1 RMSNorm

```python
import numpy as np

class RMSNormNumPy:
    """
    RMSNorm 的 NumPy 实现（含前向 + 反向传播）
    
    数学:
        y = γ ⊙ x / RMS(x)
        RMS(x) = sqrt(mean(x²) + ε)
    """
    
    def __init__(self, d, eps=1e-6):
        self.gamma = np.ones(d)           # 可学习缩放参数, shape (d,)
        self.eps = eps
        # 缓存前向传播中间值
        self.cache = {}
    
    def forward(self, x):
        """
        前向传播
        参数: x — shape (B, T, d) 或 (T, d) 或 (d,)
        返回: y — 同 x 的 shape
        """
        # 计算 RMS
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.eps)  # (..., 1)
        x_norm = x / rms                                                    # (..., d)
        y = self.gamma * x_norm                                             # (..., d)
        
        # 缓存用于反向传播
        self.cache = {'x': x, 'rms': rms, 'x_norm': x_norm}
        return y
    
    def backward(self, grad_y):
        """
        反向传播
        参数: grad_y — shape 同 y
        返回: grad_x — shape 同 x
                grad_gamma — shape (d,)
        """
        x = self.cache['x']
        rms = self.cache['rms']
        x_norm = self.cache['x_norm']
        d = x.shape[-1]
        
        # ∂L/∂γ = Σ (∂L/∂y · x_norm)
        grad_gamma = np.sum(grad_y * x_norm, axis=tuple(range(grad_y.ndim - 1)))
        
        # ∂L/∂x = γ/s · (∂L/∂y - x_norm · mean(∂L/∂y · x_norm))
        grad_x_norm = grad_y * self.gamma                     # (..., d)
        inner = np.sum(grad_x_norm * x_norm, axis=-1, keepdims=True) / d  # (..., 1)
        grad_x = (grad_x_norm - x_norm * inner) / rms         # (..., d)
        
        return grad_x, grad_gamma

# ===== 验证 =====
np.random.seed(42)
B, T, d = 2, 4, 8
x = np.random.randn(B, T, d)
rms_norm = RMSNormNumPy(d)
y = rms_norm.forward(x)

# 验证: 归一化后 RMS ≈ 1
rms_y = np.sqrt(np.mean(y ** 2, axis=-1))
print(f"输入 RMS: {np.sqrt(np.mean(x**2, axis=-1)).mean():.4f}")
print(f"输出 RMS: {rms_y.mean():.4f} (应接近 1.0)")

# 验证梯度（数值梯度检查）
grad_y = np.random.randn(B, T, d)
grad_x, grad_gamma = rms_norm.backward(grad_y)
print(f"梯度 shape: grad_x={grad_x.shape}, grad_gamma={grad_gamma.shape}")
```

#### 6.1.2 SwiGLU FFN

```python
def silu(x):
    """SiLU/Swish 激活函数: x · σ(x)"""
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -20, 20))))

def silu_grad(x):
    """SiLU 导数: σ(x) + x · σ(x) · (1 - σ(x))"""
    sig = 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
    return sig + x * sig * (1.0 - sig)

class SwiGLUFFNNumPy:
    """
    SwiGLU FFN 的 NumPy 实现
    
    数学:
        y = (SiLU(x @ W_gate) ⊙ (x @ W_up)) @ W_down
    """
    
    def __init__(self, d_model, d_ffn):
        scale = np.sqrt(2.0 / (d_model + d_ffn))
        self.W_gate = np.random.randn(d_model, d_ffn) * scale  # (d, d_ffn)
        self.W_up   = np.random.randn(d_model, d_ffn) * scale  # (d, d_ffn)
        self.W_down = np.random.randn(d_ffn, d_model) * scale  # (d_ffn, d)
        self.cache = {}
    
    def forward(self, x):
        """
        参数: x — shape (..., d_model)
        返回: y — shape (..., d_model)
        """
        g = x @ self.W_gate            # (..., d_ffn) 门控预激活
        u = x @ self.W_up              # (..., d_ffn) 上投影
        a = silu(g)                     # (..., d_ffn) SiLU 激活
        m = a * u                       # (..., d_ffn) 门控结果
        y = m @ self.W_down             # (..., d_model) 输出
        
        self.cache = {'x': x, 'g': g, 'u': u, 'a': a, 'm': m}
        return y
    
    def backward(self, grad_y):
        """
        反向传播
        返回: grad_x, grad_W_gate, grad_W_up, grad_W_down
        """
        x, g, u, a, m = (self.cache[k] for k in ['x', 'g', 'u', 'a', 'm'])
        
        # ∂L/∂m = ∂L/∂y @ W_down^T
        grad_m = grad_y @ self.W_down.T              # (..., d_ffn)
        
        # ∂L/∂a = ∂L/∂m ⊙ u,  ∂L/∂u = ∂L/∂m ⊙ a
        grad_a = grad_m * u                           # (..., d_ffn)
        grad_u = grad_m * a                           # (..., d_ffn)
        
        # ∂L/∂g = ∂L/∂a ⊙ SiLU'(g)
        grad_g = grad_a * silu_grad(g)                # (..., d_ffn)
        
        # ∂L/∂x = ∂L/∂g @ W_gate^T + ∂L/∂u @ W_up^T
        grad_x = grad_g @ self.W_gate.T + grad_u @ self.W_up.T
        
        # 权重梯度 (展平 batch 维度)
        x_flat = x.reshape(-1, x.shape[-1])
        grad_g_flat = grad_g.reshape(-1, grad_g.shape[-1])
        grad_u_flat = grad_u.reshape(-1, grad_u.shape[-1])
        m_flat = m.reshape(-1, m.shape[-1])
        grad_y_flat = grad_y.reshape(-1, grad_y.shape[-1])
        
        grad_W_gate = x_flat.T @ grad_g_flat
        grad_W_up   = x_flat.T @ grad_u_flat
        grad_W_down = m_flat.T @ grad_y_flat
        
        return grad_x, grad_W_gate, grad_W_up, grad_W_down

# ===== 验证 =====
np.random.seed(42)
B, T, d, d_ffn = 2, 4, 16, 11  # d_ffn ≈ 8d/3
ffn = SwiGLUFFNNumPy(d, d_ffn)
x = np.random.randn(B, T, d) * 0.1
y = ffn.forward(x)
print(f"SwiGLU FFN: input {x.shape} → output {y.shape}")
print(f"参数量: W_gate={ffn.W_gate.shape}, W_up={ffn.W_up.shape}, W_down={ffn.W_down.shape}")
print(f"总参数: {sum(w.size for w in [ffn.W_gate, ffn.W_up, ffn.W_down]):,}")
```

#### 6.1.3 RoPE 旋转位置编码

```python
def precompute_rope_frequencies(d, max_len=2048, base=10000.0):
    """
    预计算 RoPE 频率
    
    数学:
        θ_k = 1 / base^(2k/d),  k = 0, 1, ..., d/2 - 1
        Θ(m) = [m·θ_0, m·θ_1, ..., m·θ_{d/2-1}]
    
    参数:
        d:       头维度 (必须为偶数)
        max_len: 最大序列长度
        base:    频率基数 (默认 10000)
    返回:
        cos_cache: shape (max_len, d/2) — cos(m·θ_k) 值
        sin_cache: shape (max_len, d/2) — sin(m·θ_k) 值
    """
    assert d % 2 == 0, "RoPE 要求偶数维度"
    
    # θ_k = 1 / base^(2k/d)
    k = np.arange(0, d // 2)                      # (d/2,)
    theta = 1.0 / (base ** (2.0 * k / d))         # (d/2,)
    
    # m·θ_k 的外积
    positions = np.arange(max_len)                 # (max_len,)
    angles = np.outer(positions, theta)            # (max_len, d/2)
    
    cos_cache = np.cos(angles)                     # (max_len, d/2)
    sin_cache = np.sin(angles)                     # (max_len, d/2)
    
    return cos_cache, sin_cache

def apply_rope(x, cos_cache, sin_cache):
    """
    应用 RoPE 旋转位置编码
    
    数学:
        f(x, m) = x ⊙ cos(Θ_m) + rotate_half(x) ⊙ sin(Θ_m)
    
    参数:
        x:         shape (B, T, n_heads, d_head) 或 (B, T, d_head)
        cos_cache: shape (T, d_head/2) — 预计算的 cos 值
        sin_cache: shape (T, d_head/2) — 预计算的 sin 值
    返回:
        x_rotated: shape 同 x — 旋转后的向量
    """
    d = x.shape[-1]
    half_d = d // 2
    
    # 分为前后两半
    x1 = x[..., :half_d]              # (..., d/2)
    x2 = x[..., half_d:]             # (..., d/2)
    
    # 获取对应位置的 cos, sin
    T = x.shape[-3] if x.ndim >= 3 else x.shape[0]
    cos_m = cos_cache[:T]             # (T, d/2)
    sin_m = sin_cache[:T]             # (T, d/2)
    
    # 广播到 x 的 shape
    while cos_m.ndim < x.ndim:
        cos_m = np.expand_dims(cos_m, axis=0)
        sin_m = np.expand_dims(sin_m, axis=0)
    # 如果 x 有 n_heads 维度，在 head 维度也扩展
    if x.ndim == 4:
        cos_m = np.expand_dims(cos_m[..., 0, :], axis=-2)
        sin_m = np.expand_dims(sin_m[..., 0, :], axis=-2)
        cos_m = np.broadcast_to(cos_m, x1.shape)
        sin_m = np.broadcast_to(sin_m, x1.shape)
    
    # 旋转公式: [x1, x2] → [x1·cos - x2·sin, x2·cos + x1·sin]
    y1 = x1 * cos_m - x2 * sin_m
    y2 = x2 * cos_m + x1 * sin_m
    
    return np.concatenate([y1, y2], axis=-1)

def verify_rope_relative_position():
    """
    验证 RoPE 的核心性质: 内积只依赖相对位置
    """
    d = 8
    cos_cache, sin_cache = precompute_rope_frequencies(d, max_len=100)
    
    # 随机 q, k 向量
    np.random.seed(42)
    q = np.random.randn(d)
    k = np.random.randn(d)
    
    print("验证: <f(q,m), f(k,n)> 只依赖 (m-n)")
    print(f"{'m':>3} {'n':>3} {'m-n':>4} {'内积':>10}")
    print("-" * 30)
    
    for m, n in [(5, 3), (10, 8), (20, 18), (50, 48)]:
        # 应用 RoPE
        q_rot = apply_rope(q.reshape(1, 1, d), cos_cache, sin_cache)
        k_rot = apply_rope(k.reshape(1, 1, d), cos_cache, sin_cache)
        # 不对，需要分别在位置 m 和 n 应用
        q_at_m = q.copy()
        k_at_n = k.copy()
        half = d // 2
        q1, q2 = q_at_m[:half], q_at_m[half:]
        k1, k2 = k_at_n[:half], k_at_n[half:]
        
        cos_q = cos_cache[m]; sin_q = sin_cache[m]
        cos_k = cos_cache[n]; sin_k = sin_cache[n]
        
        q_rot_m = np.concatenate([q1*cos_q - q2*sin_q, q2*cos_q + q1*sin_q])
        k_rot_n = np.concatenate([k1*cos_k - k2*sin_k, k2*cos_k + k1*sin_k])
        
        dot = np.dot(q_rot_m, k_rot_n)
        print(f"{m:>3} {n:>3} {m-n:>4} {dot:>10.6f}")

# ===== 运行验证 =====
cos_c, sin_c = precompute_rope_frequencies(128, max_len=2048)
print(f"RoPE 频率: shape cos={cos_c.shape}, sin={sin_c.shape}")
print(f"θ_0 = {1.0:.4f} (最快旋转), θ_63 = {1.0/10000**(126/128):.6f} (最慢旋转)")

print()
verify_rope_relative_position()
```

#### 6.1.4 因果自注意力（带 RoPE）

```python
def causal_attention_with_rope(Q, K, V, cos_cache, sin_cache):
    """
    带 RoPE 的因果自注意力
    
    数学:
        Q' = RoPE(Q),  K' = RoPE(K)
        A = softmax(Q'K'^T / √d_k) ⊙ CausalMask
        O = A · V
    
    参数:
        Q, K, V: shape (B, T, n_heads, d_head)
        cos_cache, sin_cache: shape (T, d_head/2)
    返回:
        output: shape (B, T, n_heads, d_head)
        attn_weights: shape (B, n_heads, T, T)
    """
    B, T, n_heads, d_head = Q.shape
    scale = 1.0 / np.sqrt(d_head)
    
    # 应用 RoPE 到 Q, K (不对 V 应用)
    Q_rot = apply_rope(Q, cos_cache, sin_cache)  # (B, T, n_heads, d_head)
    K_rot = apply_rope(K, cos_cache, sin_cache)  # (B, T, n_heads, d_head)
    
    # 转置为 (B, n_heads, T, d_head) 便于矩阵乘法
    Q_rot = Q_rot.transpose(0, 2, 1, 3)  # (B, n_heads, T, d_head)
    K_rot = K_rot.transpose(0, 2, 1, 3)
    V_t   = V.transpose(0, 2, 1, 3)
    
    # 注意力分数
    scores = np.matmul(Q_rot, K_rot.transpose(0, 1, 3, 2)) * scale  # (B, n_heads, T, T)
    
    # 因果掩码
    causal_mask = np.triu(np.ones((T, T)), k=1) * (-1e9)
    scores = scores + causal_mask
    
    # Softmax
    scores_max = scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attn_weights = exp_scores / (exp_scores.sum(axis=-1, keepdims=True) + 1e-8)
    
    # 加权求和
    output = np.matmul(attn_weights, V_t)  # (B, n_heads, T, d_head)
    output = output.transpose(0, 2, 1, 3)  # (B, T, n_heads, d_head)
    
    return output, attn_weights

# ===== 验证 =====
np.random.seed(42)
B, T, n_heads, d_head = 2, 8, 4, 16
Q = np.random.randn(B, T, n_heads, d_head) * 0.1
K = np.random.randn(B, T, n_heads, d_head) * 0.1
V = np.random.randn(B, T, n_heads, d_head) * 0.1

cos_c, sin_c = precompute_rope_frequencies(d_head, max_len=T)
output, attn_w = causal_attention_with_rope(Q, K, V, cos_c, sin_c)
print(f"注意力输出: {output.shape}")
print(f"因果掩码验证 (attn[0,0] 上三角应为 0):")
print(f"  上三角最大值: {np.triu(attn_w[0, 0], k=1).max():.8f} (应为 ~0)")
print(f"  每行和: {attn_w[0, 0].sum(axis=-1)[:3]} (应为 ~1)")
```

### 6.2 PyTorch 完整 LLaMA 实现

#### 6.2.1 RMSNorm

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    """
    RMSNorm: 去除均值中心化的归一化层
    
    数学:
        y = γ ⊙ x / sqrt(mean(x²) + ε)
    """
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))  # γ
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 计算 RMS（使用 float32 保证数值稳定性）
        rms = torch.sqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_norm = x.float() / rms
        return (self.weight * x_norm).type_as(x)
```

#### 6.2.2 RoPE

```python
class RotaryPositionEmbedding(nn.Module):
    """
    RoPE: 旋转位置编码
    
    数学:
        f(x, m) = x ⊙ cos(Θ_m) + rotate_half(x) ⊙ sin(Θ_m)
        θ_k = 1 / base^(2k/d)
    """
    
    def __init__(self, d_head: int, max_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.d_head = d_head
        
        # 预计算频率
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer("inv_freq", inv_freq)  # (d_head/2,)
        
        # 预计算缓存
        self._build_cache(max_len)
    
    def _build_cache(self, max_len: int):
        positions = torch.arange(max_len, dtype=self.inv_freq.dtype)
        angles = torch.outer(positions, self.inv_freq)  # (max_len, d_head/2)
        # 拼接为 (max_len, d_head)
        emb = torch.cat([angles, angles], dim=-1)
        self.register_buffer("cos_cache", emb.cos(), persistent=False)
        self.register_buffer("sin_cache", emb.sin(), persistent=False)
    
    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """将向量的前半和后半交叉取负: [x1, x2] → [-x2, x1]"""
        d_half = x.shape[-1] // 2
        x1, x2 = x[..., :d_half], x[..., d_half:]
        return torch.cat([-x2, x1], dim=-1)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, 
                offset: int = 0) -> tuple:
        """
        参数:
            q: (B, T, n_heads, d_head)
            k: (B, T, n_heads, d_head)
            offset: 用于增量解码的位置偏移
        返回:
            q_rot, k_rot: 旋转后的 Q, K
        """
        T = q.shape[1]
        cos = self.cos_cache[offset:offset+T].unsqueeze(0).unsqueeze(2)  # (1, T, 1, d_head)
        sin = self.sin_cache[offset:offset+T].unsqueeze(0).unsqueeze(2)
        
        q_rot = q * cos + self.rotate_half(q) * sin
        k_rot = k * cos + self.rotate_half(k) * sin
        return q_rot, k_rot
```

#### 6.2.3 SwiGLU FFN

```python
class SwiGLUFFN(nn.Module):
    """
    SwiGLU 前馈网络
    
    数学:
        y = (SiLU(x @ W_gate) ⊙ (x @ W_up)) @ W_down
    """
    
    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ffn, bias=False)  # 门控投影
        self.w_up   = nn.Linear(d_model, d_ffn, bias=False)  # 上投影
        self.w_down = nn.Linear(d_ffn, d_model, bias=False)  # 下投影
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: SiLU(gate) ⊙ up → down
        gate = F.silu(self.w_gate(x))  # (B, T, d_ffn)
        up = self.w_up(x)              # (B, T, d_ffn)
        return self.w_down(gate * up)  # (B, T, d_model)
```

#### 6.2.4 多头因果自注意力

```python
class CausalSelfAttention(nn.Module):
    """
    带 RoPE 的多头因果自注意力
    
    数学:
        Q, K, V = xW_Q, xW_K, xW_V
        Q', K' = RoPE(Q, positions), RoPE(K, positions)
        Attn = softmax(Q'K'^T / √d_k + CausalMask) · V
    """
    
    def __init__(self, d_model: int, n_heads: int, max_len: int = 2048):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # QKV 投影 (无偏置)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        
        # RoPE
        self.rope = RotaryPositionEmbedding(self.d_head, max_len)
    
    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        B, T, d = x.shape
        
        # 线性投影 → 多头
        q = self.w_q(x).view(B, T, self.n_heads, self.d_head)
        k = self.w_k(x).view(B, T, self.n_heads, self.d_head)
        v = self.w_v(x).view(B, T, self.n_heads, self.d_head)
        
        # 应用 RoPE (仅 Q, K)
        q, k = self.rope(q, k, offset=offset)
        
        # 转置为 (B, n_heads, T, d_head)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 缩放点积注意力
        scale = 1.0 / math.sqrt(self.d_head)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # 因果掩码
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Softmax + 加权
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # (B, n_heads, T, d_head)
        
        # 合并头 + 输出投影
        out = out.transpose(1, 2).contiguous().view(B, T, d)
        return self.w_o(out)
```

#### 6.2.5 LLaMA Transformer 块

```python
class LLaMABlock(nn.Module):
    """
    LLaMA Transformer Decoder 块
    
    数学:
        h' = x + Attention(RMSNorm(x))
        out = h' + SwiGLU_FFN(RMSNorm(h'))
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ffn: int, 
                 max_len: int = 2048, norm_eps: float = 1e-6):
        super().__init__()
        self.attn_norm = RMSNorm(d_model, eps=norm_eps)
        self.attn = CausalSelfAttention(d_model, n_heads, max_len)
        self.ffn_norm = RMSNorm(d_model, eps=norm_eps)
        self.ffn = SwiGLUFFN(d_model, d_ffn)
    
    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        # Pre-RMSNorm + Attention + 残差
        h = x + self.attn(self.attn_norm(x), offset=offset)
        # Pre-RMSNorm + SwiGLU FFN + 残差
        out = h + self.ffn(self.ffn_norm(h))
        return out
```

#### 6.2.6 完整 LLaMA 模型

```python
class LLaMA(nn.Module):
    """
    完整 LLaMA 模型
    
    架构:
        Token Embedding → L × LLaMA Block → RMSNorm → LM Head
    
    关键设计:
        - Pre-RMSNorm (不用 LayerNorm)
        - SwiGLU FFN (不用 ReLU/GELU)
        - RoPE (不用绝对位置编码)
        - 无偏置项
    """
    
    def __init__(self, vocab_size: int, d_model: int, n_layers: int,
                 n_heads: int, d_ffn: int, max_len: int = 2048,
                 norm_eps: float = 1e-6):
        super().__init__()
        self.d_model = d_model
        
        # Token 嵌入 (无位置嵌入——RoPE 在注意力内部处理)
        self.token_emb = nn.Embedding(vocab_size, d_model)
        
        # Transformer 层
        self.layers = nn.ModuleList([
            LLaMABlock(d_model, n_heads, d_ffn, max_len, norm_eps)
            for _ in range(n_layers)
        ])
        
        # 最终归一化
        self.norm = RMSNorm(d_model, eps=norm_eps)
        
        # 语言模型头 (与嵌入权重共享)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.token_emb.weight = self.lm_head.weight  # 权重共享
        
        # 初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor, 
                offset: int = 0) -> torch.Tensor:
        """
        参数:
            input_ids: (B, T) — token ids
            offset: 位置偏移 (用于增量解码)
        返回:
            logits: (B, T, vocab_size)
        """
        # Token 嵌入
        h = self.token_emb(input_ids)  # (B, T, d_model)
        
        # L 层 Transformer
        for layer in self.layers:
            h = layer(h, offset=offset)
        
        # 最终归一化 + LM 头
        h = self.norm(h)
        logits = self.lm_head(h)  # (B, T, vocab_size)
        
        return logits
    
    def count_parameters(self) -> dict:
        """统计模型参数"""
        total = sum(p.numel() for p in self.parameters())
        # 去除权重共享的重复计数
        unique = total - self.token_emb.weight.numel()
        return {
            "total_params": total,
            "unique_params": unique,
            "embedding": self.token_emb.weight.numel(),
            "per_layer": sum(p.numel() for p in self.layers[0].parameters()),
        }

# ===== 验证: 构建一个缩小版 LLaMA =====
torch.manual_seed(42)
config = {
    "vocab_size": 32000,
    "d_model": 256,
    "n_layers": 4,
    "n_heads": 4,
    "d_ffn": 683,       # ≈ 8*256/3, 取整
    "max_len": 512,
}
model = LLaMA(**config)

# 参数统计
stats = model.count_parameters()
print(f"Mini-LLaMA 配置: {config}")
print(f"参数统计:")
print(f"  总参数: {stats['total_params']:,}")
print(f"  唯一参数: {stats['unique_params']:,}")
print(f"  嵌入层: {stats['embedding']:,}")
print(f"  每层: {stats['per_layer']:,}")

# 前向传播测试
B, T = 2, 32
input_ids = torch.randint(0, config["vocab_size"], (B, T))
logits = model(input_ids)
print(f"\n前向传播: input {input_ids.shape} → logits {logits.shape}")

# 简单训练步骤
labels = torch.randint(0, config["vocab_size"], (B, T))
loss = F.cross_entropy(logits.view(-1, config["vocab_size"]), labels.view(-1))
loss.backward()
print(f"损失: {loss.item():.4f} (随机初始化预期 ~{math.log(config['vocab_size']):.2f})")
```

#### 6.2.7 LLaMA 实际配置对照

```python
def llama_configs():
    """LLaMA 官方模型配置"""
    configs = {
        "LLaMA-7B": {
            "vocab_size": 32000, "d_model": 4096, "n_layers": 32,
            "n_heads": 32, "d_ffn": 11008, "max_len": 2048,
        },
        "LLaMA-13B": {
            "vocab_size": 32000, "d_model": 5120, "n_layers": 40,
            "n_heads": 40, "d_ffn": 13824, "max_len": 2048,
        },
        "LLaMA-33B": {
            "vocab_size": 32000, "d_model": 6656, "n_layers": 60,
            "n_heads": 52, "d_ffn": 17920, "max_len": 2048,
        },
        "LLaMA-65B": {
            "vocab_size": 32000, "d_model": 8192, "n_layers": 80,
            "n_heads": 64, "d_ffn": 22016, "max_len": 2048,
        },
    }
    
    print(f"{'模型':<12} {'参数量':>10} {'d_model':>8} {'层数':>5} {'头数':>5} {'d_ffn':>7} {'d_ffn/d':>8}")
    print("-" * 65)
    for name, cfg in configs.items():
        d, L, h, ffn = cfg["d_model"], cfg["n_layers"], cfg["n_heads"], cfg["d_ffn"]
        V = cfg["vocab_size"]
        # 参数量估算: embedding + L*(attn + ffn + 2*norm) + final_norm
        attn_params = 4 * d * d           # W_q, W_k, W_v, W_o
        ffn_params = 3 * d * ffn          # W_gate, W_up, W_down
        norm_params = 2 * d               # 2 个 RMSNorm
        layer_params = attn_params + ffn_params + norm_params
        total = V * d + L * layer_params + d  # embedding + layers + final_norm
        print(f"{name:<12} {total/1e9:>9.1f}B {d:>8} {L:>5} {h:>5} {ffn:>7} {ffn/d:>8.2f}")

llama_configs()
```

---

## 7. 实践技巧与可视化

### 7.1 RoPE 旋转可视化

```python
import numpy as np

def visualize_rope_rotation():
    """可视化 RoPE 在不同位置的旋转效果"""
    d = 128
    cos_cache, sin_cache = precompute_rope_frequencies(d, max_len=100)
    
    print("RoPE 频率分布 (d=128):")
    print(f"{'维度 k':>8} {'θ_k':>12} {'周期 (tokens)':>16}")
    print("-" * 40)
    for k in [0, 1, 4, 16, 32, 63]:
        theta = 1.0 / (10000 ** (2*k / d))
        period = 2 * np.pi / theta
        print(f"{k:>8} {theta:>12.6f} {period:>16.1f}")
    
    print("\n不同距离的注意力衰减效果:")
    np.random.seed(42)
    q = np.random.randn(d)
    k = np.random.randn(d)
    half = d // 2
    q1, q2 = q[:half], q[half:]
    k1, k2 = k[:half], k[half:]
    
    print(f"{'距离 |m-n|':>12} {'平均内积':>12} {'相对衰减':>12}")
    print("-" * 40)
    base_dot = None
    for dist in [0, 1, 5, 10, 50, 100, 500]:
        m, n = dist, 0
        cos_q = cos_cache[m]; sin_q = sin_cache[m]
        cos_k = cos_cache[n]; sin_k = sin_cache[n]
        q_rot = np.concatenate([q1*cos_q - q2*sin_q, q2*cos_q + q1*sin_q])
        k_rot = np.concatenate([k1*cos_k - k2*sin_k, k2*cos_k + k1*sin_k])
        dot = np.dot(q_rot, k_rot)
        if base_dot is None:
            base_dot = abs(dot)
        print(f"{dist:>12} {dot:>12.4f} {abs(dot)/base_dot:>12.4f}")

visualize_rope_rotation()
```

### 7.2 SwiGLU 与 ReLU 的激活分布对比

```python
def compare_activations():
    """对比不同激活函数的行为"""
    x = np.linspace(-4, 4, 1000)
    
    # 计算各激活函数
    relu = np.maximum(0, x)
    gelu = x * 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
    swish = x / (1 + np.exp(-x))
    
    # 统计特性
    print(f"{'激活函数':<10} {'负值比例':>10} {'均值':>10} {'最小值':>10} {'梯度消失区':>14}")
    print("-" * 60)
    
    for name, vals in [("ReLU", relu), ("GELU", gelu), ("Swish/SiLU", swish)]:
        neg_frac = np.mean(vals < 0)
        dead_zone = np.mean(np.abs(np.gradient(vals, x)) < 0.01)
        print(f"{name:<10} {neg_frac:>10.1%} {vals.mean():>10.4f} "
              f"{vals.min():>10.4f} {dead_zone:>14.1%}")
    
    print("\nSwiGLU 门控机制示意:")
    print("  gate = SiLU(x @ W_gate)  → 控制信息流")
    print("  up   = x @ W_up          → 内容信息")
    print("  out  = gate ⊙ up         → 门控选择性输出")
    print("  → 比纯激活函数更强的表达能力（乘法交互）")

compare_activations()
```

### 7.3 训练效率与 Scaling 分析

LLaMA 论文的核心实验结果：

| 模型 | 参数量 | 训练 Tokens | MMLU (5-shot) | HellaSwag | 推理速度相对值 |
|------|:------:|:-----------:|:-------------:|:---------:|:-------------:|
| GPT-3 | 175B | 300B | 43.9% | 78.9% | 1× |
| Chinchilla | 70B | 1.4T | 67.6% | 80.8% | 2.5× |
| PaLM | 540B | 780B | 69.3% | 83.4% | 0.3× |
| **LLaMA-7B** | **6.7B** | **1.0T** | **35.1%** | **76.1%** | **26×** |
| **LLaMA-13B** | **13B** | **1.0T** | **46.9%** | **79.2%** | **13×** |
| **LLaMA-33B** | **32.5B** | **1.4T** | **57.8%** | **82.8%** | **5×** |
| **LLaMA-65B** | **65B** | **1.4T** | **63.4%** | **84.2%** | **2.7×** |

> **关键发现**：LLaMA-13B（13B 参数）在大多数基准上超越 GPT-3（175B），推理速度快 13 倍。

**训练计算量分析**：

$$
\text{FLOPs} \approx 6 \times N \times D
$$

其中 $N$ 为参数量，$D$ 为训练 tokens 数。

| 模型 | FLOPs | GPU-hours (A100) |
|------|:-----:|:----------------:|
| LLaMA-7B | $4.0 \times 10^{22}$ | ~82K |
| LLaMA-13B | $7.8 \times 10^{22}$ | ~135K |
| LLaMA-33B | $2.7 \times 10^{23}$ | ~530K |
| LLaMA-65B | $5.5 \times 10^{23}$ | ~1022K |

---

## 8. 与其他模型的关系

### 8.1 从 GPT 到 LLaMA 的架构演进

```
GPT-2 (2019)              GPT-3 (2020)              LLaMA (2023)
│                          │                          │
│ Post-LayerNorm           │ Pre-LayerNorm            │ Pre-RMSNorm
│ GELU FFN                 │ GELU FFN                 │ SwiGLU FFN
│ 绝对位置嵌入              │ 绝对位置嵌入              │ RoPE
│ 有偏置                    │ 有偏置                    │ 无偏置
│ 1.5B                     │ 175B                     │ 7B-65B
│ 40GB 数据                │ 300B tokens              │ 1.0-1.4T tokens
│                          │                          │
└── 架构基础 ──→           └── 规模证明 ──→           └── 效率优化
```

**架构演进的核心趋势**：

$$
\boxed{\text{更简洁的组件} + \text{更多的训练数据} = \text{更高效的模型}}
$$

### 8.2 LLaMA 架构的后续影响

LLaMA 的三大创新已成为**开源大模型的事实标准**：

| 后续模型 | RMSNorm | SwiGLU | RoPE | 额外创新 |
|----------|:-------:|:------:|:----:|----------|
| LLaMA-2 (2023) | ✅ | ✅ | ✅ | GQA, 4K→128K 上下文 |
| Mistral-7B (2023) | ✅ | ✅ | ✅ | Sliding Window Attention, GQA |
| Qwen (2023) | ✅ | ✅ | ✅ | 多语言优化 |
| DeepSeek (2024) | ✅ | ✅ | ✅ | MLA (Multi-head Latent Attention) |
| LLaMA-3 (2024) | ✅ | ✅ | ✅ | 15T tokens, 128K 上下文 |

> **LLaMA 的历史意义**：不仅是一个模型，更是开源大模型的"Transformer 时刻"——定义了后续模型的标准架构。

### 8.3 开源大模型谱系

```
开源大模型谱系
│
├── LLaMA 系 (Meta)
│   ├── LLaMA (2023.02) ← RMSNorm + SwiGLU + RoPE
│   ├── LLaMA-2 (2023.07) ← + GQA + 扩展上下文
│   ├── Code Llama (2023.08) ← 代码特化
│   └── LLaMA-3 (2024.04) ← 15T tokens + 128K 上下文
│
├── LLaMA 衍生
│   ├── Alpaca (Stanford) ← SFT on LLaMA
│   ├── Vicuna (LMSYS) ← 对话微调
│   └── Koala (Berkeley) ← 对话数据
│
├── 独立架构 (采用 LLaMA 设计)
│   ├── Mistral/Mixtral ← + Sliding Window + MoE
│   ├── Qwen (阿里) ← + 多语言
│   ├── DeepSeek ← + MLA + GRPO
│   └── Yi (01.AI) ← + 长上下文
│
└── 对比: 非 LLaMA 架构
    ├── Falcon ← 多查询注意力
    ├── MPT ← ALiBi 位置编码
    └── RWKV ← 线性注意力
```

**LLaMA 在学习路径中的位置**：

| 时间线 | 模型 | 关键贡献 | 与 LLaMA 的关系 |
|--------|------|----------|----------------|
| 2017 | Transformer | 自注意力机制 | 基础架构 |
| 2018 | BERT | 双向预训练 | 编码器分支 |
| 2019 | GPT-2 | 因果 LM | 解码器分支 |
| 2020 | GPT-3 | Scaling Laws | 规模验证 |
| 2021 | LoRA | 参数高效微调 | 微调方法 |
| 2022 | InstructGPT | RLHF 对齐 | 训练范式 |
| **2023** | **LLaMA** | **高效架构 + 开源** | **架构标准** |
| 2024 | DeepSeek-R1 | GRPO 推理 | 建立在 LLaMA 架构上 |

---

## 扩展阅读与实现

### Q1: RMSNorm 和 LayerNorm 在训练中的实际差异有多大？

> **Q:** 去掉均值中心化真的没有影响吗？
>
> **A:** Zhang & Sennrich (2019) 在多个任务上的实验表明：
>
> - **性能差异极小**：RMSNorm 与 LayerNorm 在 BLEU、PPL 等指标上差异 < 0.5%
> - **速度提升显著**：RMSNorm 比 LayerNorm 快 10%~15%（因为省去了均值计算和偏移参数）
> - **内存节省**：每层少一个 $d$ 维参数向量（$\beta$）
>
> 对于 LLaMA-65B（80 层），节省的参数：$80 \times 2 \times 8192 = 1.3M$。虽然绝对量不大，但在推理时每一点加速都有意义。
>
> $$\text{RMSNorm 优势} = \underbrace{\text{等效性能}}_{\text{实验验证}} + \underbrace{\text{更快速度}}_{\text{~15\%}} + \underbrace{\text{更简代码}}_{\text{少一步操作}}$$

### Q2: SwiGLU 为什么比 GELU 更好？有理论解释吗？

> **Q:** Shazeer (2020) 的实验显示 SwiGLU 在多个任务上优于其他激活函数，背后的原因是什么？
>
> **A:** 目前没有完全的理论解释，但有几个直觉：
>
> 1. **门控机制**：SwiGLU 引入了乘法交互 $a \otimes u$，比单纯的逐元素激活更有表达力
> 2. **Swish 的平滑性**：相比 ReLU 的硬截断，Swish 允许小的负梯度流动
> 3. **参数效率**：虽然多了一个矩阵，但调整 $d_{\text{ffn}}$ 后总参数量相近，表达力更强
>
> Shazeer 的消融实验排名：
> $$\text{SwiGLU} > \text{GEGLU} > \text{ReGLU} > \text{GELU} > \text{ReLU}$$

### Q3: RoPE 的长度外推能力如何？

> **Q:** LLaMA 训练时使用 2048 长度，能否在更长序列上工作？
>
> **A:** 原始 RoPE 的外推能力有限，但有多种扩展方法：
>
> 1. **位置插值（PI, Chen et al., 2023）**：将位置缩放到训练范围内
>    $$f_{\text{PI}}(x, m) = f_{\text{RoPE}}\left(x, \frac{m \cdot L_{\text{train}}}{L_{\text{target}}}\right)$$
>
> 2. **NTK-aware 插值**：调整频率基数
>    $$\text{base}' = \text{base} \times \left(\frac{L_{\text{target}}}{L_{\text{train}}}\right)^{d/(d-2)}$$
>
> 3. **YaRN（Yet another RoPE extensioN）**：结合 NTK + 温度调节
>
> LLaMA-2 通过这些技术将上下文从 2K 扩展到 4K（训练时）和 128K（推理时）。

### Q4: LLaMA 为什么不使用 GQA（分组查询注意力）？

> **Q:** LLaMA-2 使用了 GQA，但原始 LLaMA 没有，为什么？
>
> **A:** GQA 是在 LLaMA 发布后才被广泛采用的优化。GQA 的核心是减少 KV 缓存：
>
> - **MHA**：$n_h$ 个 Q 头，$n_h$ 个 KV 头 → KV 缓存 $= 2 \times L \times n_h \times d_h \times T$
> - **GQA**：$n_h$ 个 Q 头，$n_g$ 个 KV 头（$n_g < n_h$）→ KV 缓存减少 $n_h / n_g$ 倍
> - **MQA**：$n_h$ 个 Q 头，1 个 KV 头 → KV 缓存最小但质量略降
>
> LLaMA-2-70B 使用 $n_g = 8$（8 组），在保持质量的同时显著加速推理。

### Q5: 权重共享（Tied Embeddings）的数学含义是什么？

> **Q:** LLaMA 将输入嵌入和输出 LM 头共享权重，这有什么数学意义？
>
> **A:** 设嵌入矩阵 $E \in \mathbb{R}^{V \times d}$，共享权重意味着：
>
> $$\text{Input: } h_0 = E[x_t] \quad \text{（查表）}$$
> $$\text{Output: } \text{logits} = h_L \cdot E^\top \quad \text{（内积）}$$
>
> 数学含义：输出 logit 衡量最后一层隐藏状态与每个词向量的**余弦相似度**（忽略缩放）。这创造了一个有意义的对称性——模型的"理解"（输入）和"表达"（输出）共享同一语义空间。
>
> 参数节省（LLaMA-7B）：$32000 \times 4096 = 131M$ 参数，占总参数量约 2%。

---

## 参考资源

### 经典论文

1. Touvron et al. (2023). [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971). arXiv.
   - **贡献**：提出 LLaMA 模型家族，证明仅用公开数据训练的小模型可以匹配甚至超越大型闭源模型

2. Zhang & Sennrich (2019). [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467). NeurIPS 2019.
   - **贡献**：提出 RMSNorm，证明 LayerNorm 的均值中心化不是必要的

3. Shazeer (2020). [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202). arXiv.
   - **贡献**：系统比较 GLU 变体，提出 SwiGLU 等激活函数

4. Su et al. (2021). [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864). arXiv.
   - **贡献**：提出 RoPE，用旋转矩阵编码位置信息，实现相对位置感知

5. Hoffmann et al. (2022). [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556). NeurIPS 2022.
   - **贡献**：提出 Chinchilla Scaling Laws，证明模型大小和数据量应同比例增长

6. Touvron et al. (2023). [LLaMA 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288). arXiv.
   - **贡献**：LLaMA 的改进版，引入 GQA、扩展上下文、RLHF 对齐

7. Dauphin et al. (2017). [Language Modeling with Gated Convolutional Networks](https://arxiv.org/abs/1612.08083). ICML 2017.
   - **贡献**：提出 GLU（门控线性单元），SwiGLU 的前身

### 教材与书籍

8. Vaswani et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). NeurIPS 2017.
   - **章节**：Transformer 基础架构，LLaMA 的改进基础

### 在线资源与教程

9. Meta AI. [LLaMA 官方代码](https://github.com/facebookresearch/llama).
   - **内容**：LLaMA 的官方 PyTorch 实现，简洁清晰

10. Eleuther AI. [GPT-NeoX](https://github.com/EleutherAI/gpt-neox).
    - **内容**：开源大模型训练框架，支持 RoPE、SwiGLU 等组件

11. Karpathy. [llama2.c](https://github.com/karpathy/llama2.c).
    - **内容**：极简 C 语言实现的 LLaMA-2 推理，适合理解架构细节

12. Chen et al. (2023). [Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595).
    - **内容**：RoPE 位置插值方法，将 LLaMA 上下文扩展到 32K+

---

## 附录：符号表

| 符号 | 含义 | 维度/类型 |
|------|------|----------|
| $d$ ($d_{\text{model}}$) | 模型隐藏维度 | 标量 |
| $d_{\text{head}}$ ($d_k$) | 注意力头维度 | 标量，$d / n_h$ |
| $d_{\text{ffn}}$ | FFN 中间维度 | 标量，$\approx 8d/3$ |
| $n_h$ | 注意力头数 | 标量 |
| $L$ | Transformer 层数 | 标量 |
| $T$ | 序列长度 | 标量 |
| $V$ ($\|\mathcal{V}\|$) | 词表大小 | 标量 |
| $x \in \mathbb{R}^d$ | 隐藏状态向量 | $(d,)$ |
| $\gamma \in \mathbb{R}^d$ | RMSNorm 缩放参数 | $(d,)$ |
| $\text{RMS}(x)$ | $x$ 的均方根 | 标量 |
| $\epsilon$ | 数值稳定常数 | 标量，通常 $10^{-6}$ |
| $\sigma(\cdot)$ | Sigmoid 函数 | 函数 |
| $\text{SiLU}(x)$ | $x \cdot \sigma(x)$，即 Swish | 函数 |
| $W_{\text{gate}}$ | SwiGLU 门控投影矩阵 | $(d, d_{\text{ffn}})$ |
| $W_{\text{up}}$ | SwiGLU 上投影矩阵 | $(d, d_{\text{ffn}})$ |
| $W_{\text{down}}$ | SwiGLU 下投影矩阵 | $(d_{\text{ffn}}, d)$ |
| $W_Q, W_K, W_V, W_O$ | 注意力投影矩阵 | $(d, d)$ |
| $\theta_k$ | RoPE 第 $k$ 维的角频率 | 标量，$10000^{-2k/d}$ |
| $R_m$ | 位置 $m$ 的旋转矩阵 | $(d, d)$ 块对角 |
| $R_m^{(k)}$ | 第 $k$ 对维度的 $2\times2$ 旋转块 | $(2, 2)$ |
| $\otimes$ | 逐元素乘法（Hadamard 积） | 运算符 |
| $\langle \cdot, \cdot \rangle$ | 内积 | 运算符 |
| $\mathcal{L}$ | 损失函数值 | 标量 |
| $N$ | 模型参数量 | 标量 |
| $D$ | 训练数据量（tokens） | 标量 |
| $C$ | 计算预算（FLOPs） | 标量 |

**典型维度示例（LLaMA-7B）：**
- $d = 4096$，$L = 32$，$n_h = 32$，$d_{\text{head}} = 128$
- $d_{\text{ffn}} = 11008$（$\approx \frac{8}{3} \times 4096$）
- $|\mathcal{V}| = 32{,}000$（SentencePiece BPE）
- 训练 Tokens：1.0T
- 训练硬件：2048 × A100-80GB
- 训练时间：~21 天
- 参数量：6.7B

---

最后更新：2026-03-19
