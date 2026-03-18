# GPT-2 数学原理与实现 —— 自回归生成与 Zero-shot 的完整推导

> **前置知识**：Transformer 解码器、因果注意力、交叉熵损失、Python 基础  
> **与前面内容的联系**：建议先学习 [BERT-Math-and-Implementation](./07-BERT-Math-and-Implementation.md)，理解预训练范式  
> **与后续内容的联系**：GPT-2 是 GPT-3 的直接前身，规模扩展 + In-context Learning 的起点

---

## 目录

1. [引言：为什么需要自回归生成模型？](#1-引言为什么需要自回归生成模型)
   - 1.1 [从 BERT 到 GPT-2：理解 vs 生成](#11-从-bert-到-gpt-2理解-vs-生成)
   - 1.2 [Zero-shot 的核心洞察](#12-zero-shot-的核心洞察)
   - 1.3 [本科数学知识映射表](#13-本科数学知识映射表)
2. [核心思想：语言模型即多任务学习器](#2-核心思想语言模型即多任务学习器)
   - 2.1 [自回归语言建模](#21-自回归语言建模)
   - 2.2 [任务条件化：从监督到 Zero-shot](#22-任务条件化从监督到-zero-shot)
   - 2.3 [数据质量与 WebText 数据集](#23-数据质量与-webtext-数据集)
3. [GPT-2 架构的数学描述](#3-gpt-2-架构的数学描述)
   - 3.1 [输入表示：Token + 位置嵌入](#31-输入表示token--位置嵌入)
   - 3.2 [因果注意力掩码（Causal Mask）](#32-因果注意力掩码causal-mask)
   - 3.3 [Pre-Norm Transformer 解码器层](#33-pre-norm-transformer-解码器层)
   - 3.4 [GPT-2 各规模配置](#34-gpt-2-各规模配置)
4. [自回归目标的数学推导](#4-自回归目标的数学推导)
   - 4.1 [语言模型损失函数](#41-语言模型损失函数)
   - 4.2 [梯度分析与因果约束](#42-梯度分析与因果约束)
   - 4.3 [困惑度（Perplexity）](#43-困惑度perplexity)
5. [训练优化方法总结](#5-训练优化方法总结)
   - 5.1 [预训练策略](#51-预训练策略)
   - 5.2 [优化器与学习率调度](#52-优化器与学习率调度)
   - 5.3 [训练稳定性技巧](#53-训练稳定性技巧)
6. [从数学到代码：完整实现](#6-从数学到代码完整实现)
   - 6.1 [NumPy 实现核心组件](#61-numpy-实现核心组件)
   - 6.2 [PyTorch 完整实现](#62-pytorch-完整实现)
7. [文本生成策略：采样与搜索](#7-文本生成策略采样与搜索)
   - 7.1 [贪心搜索与 Beam Search](#71-贪心搜索与-beam-search)
   - 7.2 [温度采样](#72-温度采样)
   - 7.3 [Top-k 与 Top-p (Nucleus) 采样](#73-top-k-与-top-p-nucleus-采样)
   - 7.4 [生成策略对比与可视化](#74-生成策略对比与可视化)
8. [与其他模型的关系](#8-与其他模型的关系)
   - 8.1 [GPT-1 → GPT-2 → GPT-3 演进](#81-gpt-1--gpt-2--gpt-3-演进)
   - 8.2 [GPT-2 vs BERT：生成 vs 理解](#82-gpt-2-vs-bert生成-vs-理解)
   - 8.3 [自回归范式的后续发展](#83-自回归范式的后续发展)

[扩展阅读与实现](#扩展阅读与实现)

[参考资源](#参考资源)

附录：[符号表](#附录符号表)

---

## 1. 引言：为什么需要自回归生成模型？

### 1.1 从 BERT 到 GPT-2：理解 vs 生成

在 [BERT](./07-BERT-Math-and-Implementation.md) 中，我们学习了**双向预训练**如何在 NLU 任务上取得突破。但 BERT 有一个根本限制：

$$
\text{BERT: } P(x_t \mid x_{\backslash t}) \quad \text{（完形填空，不能生成序列）}
$$

GPT-2 采用了截然不同的路线——**自回归语言模型**：

$$
\text{GPT-2: } P(x_1, x_2, \ldots, x_n) = \prod_{t=1}^{n} P(x_t \mid x_{<t})
$$

**核心哲学差异**：

| 维度 | BERT | GPT-2 |
|------|------|-------|
| 方向 | 双向（同时看左右） | 单向（只看左边） |
| 目标 | 填空 (MLM) | 预测下一个词 |
| 擅长 | 理解（分类、问答） | 生成（续写、翻译、摘要） |
| 微调 | 需要任务特定头 | Zero-shot，无需微调 |

**GPT-2 的革命性主张**：

> 一个足够好的语言模型，不需要任何监督训练，就能完成多种 NLP 任务。

### 1.2 Zero-shot 的核心洞察

传统 NLP 流程：

$$
\text{数据标注} \to \text{模型训练} \to \text{任务预测}
$$

GPT-2 的 Zero-shot 流程：

$$
\text{大规模无监督预训练} \to \text{自然语言提示} \to \text{任务预测}
$$

**关键数学洞察**：任何有监督任务都可以表示为条件语言模型：

$$
\boxed{P(\text{output} \mid \text{input}) = P(\text{output} \mid \text{input}, \text{task description})}
$$

**示例**：

| 任务 | 传统方式 | GPT-2 Zero-shot 提示 |
|------|---------|---------------------|
| 翻译 | 训练翻译模型 | "translate English to French: cat → " |
| 摘要 | 训练摘要模型 | "TL;DR: [长文本] → " |
| 问答 | 训练 QA 模型 | "Q: What is AI? A: " |

### 1.3 本科数学知识映射表

| 数学概念 | GPT-2 中的应用 | 代码对应 |
|---------|---------------|---------|
| 条件概率链式法则 | 自回归分解 | 逐 token 生成 |
| 交叉熵 $H(p, q)$ | 语言模型损失 | `F.cross_entropy()` |
| Softmax + 温度 | 采样策略 | `softmax(logits / T)` |
| 下三角矩阵 | 因果注意力掩码 | `torch.tril()` |
| 概率排序与累积分布 | Top-k / Top-p 采样 | `torch.sort()` + `cumsum()` |
| 困惑度 $2^H$ | 模型评估指标 | `torch.exp(loss)` |

---

## 2. 核心思想：语言模型即多任务学习器

### 2.1 自回归语言建模

GPT-2 的核心目标：给定前文 $x_{<t}$，预测下一个 token $x_t$。

**概率分解**（链式法则）：

$$
P(x_1, x_2, \ldots, x_n) = \prod_{t=1}^{n} P(x_t \mid x_1, x_2, \ldots, x_{t-1})
$$

每个条件概率由 Transformer 解码器参数化：

$$
P(x_t \mid x_{<t}; \theta) = \text{softmax}\left(W_e \cdot h_t^{(L)} + b\right)_{x_t}
$$

其中：
- $h_t^{(L)} \in \mathbb{R}^d$ 是第 $L$ 层 Transformer 在位置 $t$ 的隐藏状态
- $W_e \in \mathbb{R}^{|V| \times d}$ 是输出投影矩阵（与输入嵌入共享权重）
- $\theta$ 是模型所有参数

**与 BERT 的关键区别**：

$$
\underbrace{P_{\text{BERT}}(x_t \mid x_{\backslash t})}_{\text{双向，看到全部上下文}} \quad \text{vs} \quad \underbrace{P_{\text{GPT-2}}(x_t \mid x_{<t})}_{\text{单向，只看到左侧}}
$$

### 2.2 任务条件化：从监督到 Zero-shot

GPT-2 论文的核心论点：

> 当语言模型的容量足够大、训练数据足够多样时，它会**隐式学习**各种 NLP 任务。

**数学表述**：

有监督学习估计 $P(\text{output} \mid \text{input})$，但这忽略了任务信息。更一般的形式：

$$
P(\text{output} \mid \text{input}, \text{task})
$$

在语言模型框架下，任务描述、输入、输出都是 token 序列的一部分：

$$
\boxed{P(x_1, \ldots, x_n) = P(\underbrace{\text{task tokens}}_{\text{任务描述}}, \underbrace{\text{input tokens}}_{\text{输入}}, \underbrace{\text{output tokens}}_{\text{输出}})}
$$

语言模型在预测下一个 token 时，自然地条件化了前面所有信息（包括任务描述和输入）。

### 2.3 数据质量与 WebText 数据集

GPT-2 使用了精心筛选的 **WebText** 数据集：

| 属性 | 值 |
|------|-----|
| 来源 | Reddit 外链（≥3 karma） |
| 规模 | ~8M 文档，40GB 文本 |
| Token 数 | ~10B（BPE 编码后） |
| 去重 | 基于内容去重 |

**为什么数据质量如此重要？**

$$
\mathcal{L}(\theta) = -\mathbb{E}_{x \sim p_{\text{data}}} \left[\sum_{t=1}^{n} \log P(x_t \mid x_{<t}; \theta)\right]
$$

模型学到的分布 $P(\cdot; \theta)$ 逼近训练数据分布 $p_{\text{data}}$。如果 $p_{\text{data}}$ 质量低（含噪声、重复、低质量文本），模型也会学到这些模式。

---

## 3. GPT-2 架构的数学描述

### 3.1 输入表示：Token + 位置嵌入

与 BERT 的三种嵌入不同，GPT-2 只使用**两种嵌入**（无段落嵌入）：

$$
\boxed{E_{\text{input}}(t) = E_{\text{token}}(x_t) + E_{\text{position}}(t)}
$$

**Token 嵌入**：使用 **Byte Pair Encoding (BPE)** 分词：

$$
E_{\text{token}}(x_t) = W_e[x_t] \in \mathbb{R}^d
$$

其中 $W_e \in \mathbb{R}^{|V| \times d}$，GPT-2 的词表大小 $|V| = 50{,}257$。

**位置嵌入**：可学习的绝对位置编码：

$$
E_{\text{position}}(t) = W_p[t] \in \mathbb{R}^d
$$

其中 $W_p \in \mathbb{R}^{n_{\text{ctx}} \times d}$，$n_{\text{ctx}} = 1024$ 是最大上下文长度。

**BPE 分词示例**：

```
"GPT-2 is amazing!" → ["GPT", "-", "2", " is", " amazing", "!"]
"unhappiness"       → ["un", "happiness"]
```

> **Q:** 为什么 GPT-2 用 BPE 而不是 BERT 的 WordPiece？
>
> **A:** BPE 基于字节（byte-level），可以编码任意 Unicode 文本而无 OOV 问题。WordPiece 基于字符，需要预定义字符集。对于生成模型，处理任意输入的能力尤为重要。

### 3.2 因果注意力掩码（Causal Mask）

GPT-2 的核心约束：**位置 $t$ 只能看到位置 $1, 2, \ldots, t$**。

这通过**因果掩码矩阵**（下三角矩阵）实现：

$$
\boxed{
M^{\text{causal}}_{ij} = \begin{cases}
0 & \text{if } j \leq i \quad (\text{可见}) \\
-\infty & \text{if } j > i \quad (\text{不可见})
\end{cases}
}
$$

注意力分数加上掩码后：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M^{\text{causal}}\right) V
$$

**可视化**（$n = 5$）：

```
掩码矩阵 M:          softmax 后的注意力权重 A:
 0  -∞  -∞  -∞  -∞    [1.0   0    0    0    0  ]
 0   0  -∞  -∞  -∞    [0.5  0.5   0    0    0  ]  (示意)
 0   0   0  -∞  -∞    [0.3  0.3  0.4   0    0  ]
 0   0   0   0  -∞    [0.2  0.3  0.2  0.3   0  ]
 0   0   0   0   0    [0.2  0.2  0.2  0.2  0.2 ]
```

**因果掩码的数学保证**：

$$
A_{ij} = \frac{\exp\left(\frac{q_i^\top k_j}{\sqrt{d_k}} + M_{ij}\right)}{\sum_{m=1}^{n} \exp\left(\frac{q_i^\top k_m}{\sqrt{d_k}} + M_{im}\right)}
$$

当 $j > i$ 时，$M_{ij} = -\infty$，故 $\exp(-\infty) = 0$，即 $A_{ij} = 0$。

$$
\boxed{\text{因果约束: } A_{ij} = 0 \quad \forall j > i}
$$

这确保了 $h_t$ 只依赖于 $x_1, \ldots, x_t$，满足自回归条件。

### 3.3 Pre-Norm Transformer 解码器层

GPT-2 与原始 Transformer 的一个关键区别：**Pre-Norm**（LayerNorm 在子层之前，而非之后）。

**Post-Norm**（原始 Transformer / BERT）：

$$
h = x + \text{SubLayer}(x), \quad \text{then } \text{LayerNorm}(h)
$$

**Pre-Norm**（GPT-2）：

$$
h = x + \text{SubLayer}(\text{LayerNorm}(x))
$$

GPT-2 每层的完整计算：

$$
\boxed{
\begin{aligned}
a^{(l)} &= x^{(l-1)} + \text{MultiHeadAttn}\left(\text{LN}(x^{(l-1)})\right) \\
x^{(l)} &= a^{(l)} + \text{FFN}\left(\text{LN}(a^{(l)})\right)
\end{aligned}
}
$$

最终输出再加一层 LayerNorm：

$$
h = \text{LN}(x^{(L)})
$$

> **Q:** 为什么 Pre-Norm 比 Post-Norm 更稳定？
>
> **A:** Post-Norm 中残差路径上的值未经归一化，随着层数增加容易发散。Pre-Norm 确保每个子层的输入都经过归一化，梯度在残差路径上直接流动，训练更稳定。对于深层模型（如 GPT-2 的 48 层版本），这一点尤为关键。

### 3.4 GPT-2 各规模配置

| 参数 | Small (117M) | Medium (345M) | Large (762M) | XL (1.5B) |
|------|:-----------:|:-------------:|:------------:|:---------:|
| 层数 $L$ | 12 | 24 | 36 | 48 |
| 隐藏维度 $d$ | 768 | 1024 | 1280 | 1600 |
| 注意力头数 $A$ | 12 | 16 | 20 | 25 |
| 每头维度 $d_k$ | 64 | 64 | 64 | 64 |
| FFN 维度 $d_{ff}$ | 3072 | 4096 | 5120 | 6400 |
| 上下文长度 $n_{\text{ctx}}$ | 1024 | 1024 | 1024 | 1024 |
| 词表大小 $\|V\|$ | 50,257 | 50,257 | 50,257 | 50,257 |

**参数量估算**（GPT-2 Small）：

嵌入层：
$$
P_{\text{emb}} = |V| \cdot d + n_{\text{ctx}} \cdot d = 50257 \times 768 + 1024 \times 768 \approx 39.4\text{M}
$$

单层 Transformer（Pre-Norm）：
$$
P_{\text{layer}} = \underbrace{4d^2}_{\text{Attn}} + \underbrace{2 \cdot d \cdot d_{ff}}_{\text{FFN}} + \underbrace{4d + 2d_{ff} + 2d}_{\text{biases+LN}} \approx 7.1\text{M}
$$

总计：
$$
P_{\text{total}} = P_{\text{emb}} + L \cdot P_{\text{layer}} + d \approx 39.4\text{M} + 12 \times 7.1\text{M} \approx 124\text{M}
$$

> **注意**：GPT-2 共享输入嵌入和输出投影矩阵 $W_e$（权重绑定），因此输出层不额外计算参数。

---

## 4. 自回归目标的数学推导

### 4.1 语言模型损失函数

**目标**：最大化训练数据的对数似然：

$$
\max_\theta \sum_{x \in \mathcal{D}} \log P(x; \theta) = \max_\theta \sum_{x \in \mathcal{D}} \sum_{t=1}^{n} \log P(x_t \mid x_{<t}; \theta)
$$

等价于最小化**负对数似然**（交叉熵损失）：

$$
\boxed{\mathcal{L}(\theta) = -\frac{1}{n} \sum_{t=1}^{n} \log P(x_t \mid x_{<t}; \theta)}
$$

**前向计算流程**：

1. 嵌入：$E = W_e[x] + W_p[\text{pos}] \in \mathbb{R}^{n \times d}$
2. Transformer：$H = \text{TransformerDecoder}(E) \in \mathbb{R}^{n \times d}$
3. 输出投影（权重绑定）：$\text{logits} = H \cdot W_e^\top \in \mathbb{R}^{n \times |V|}$
4. 损失：对位置 $t$ 的 logits 与真实 token $x_{t+1}$ 计算交叉熵

**权重绑定（Weight Tying）**：

输出层复用输入嵌入矩阵：

$$
P(x_t = w \mid x_{<t}) = \text{softmax}(h_t^\top \cdot w_e^{(w)})
$$

其中 $w_e^{(w)}$ 是词 $w$ 在嵌入矩阵 $W_e$ 中的行向量。

> **直觉**：预测下一个词就是在嵌入空间中找到与当前隐藏状态最相似的词向量。

### 4.2 梯度分析与因果约束

对输出层的梯度（与 BERT 的 MLM 类似）：

$$
\frac{\partial \mathcal{L}}{\partial h_t} = -\frac{1}{n}\left(\mathbb{1}_{x_{t+1}} - P(\cdot \mid x_{\leq t})\right) \cdot W_e
$$

**因果约束对梯度的影响**：

由于因果掩码，$h_t$ 只依赖于 $x_1, \ldots, x_t$。反向传播时：

$$
\frac{\partial h_t}{\partial x_s} = 0 \quad \text{when } s > t
$$

这意味着位置 $t$ 的损失梯度**不会流向未来位置**的输入，保证了训练和推理的一致性。

### 4.3 困惑度（Perplexity）

**困惑度**是评估语言模型的标准指标：

$$
\boxed{\text{PPL} = \exp\left(-\frac{1}{n} \sum_{t=1}^{n} \log P(x_t \mid x_{<t})\right) = \exp(\mathcal{L})}
$$

**直觉**：困惑度可以理解为模型在每个位置上"平均需要从多少个等概率选项中选择"。

| 困惑度 | 含义 |
|--------|------|
| PPL = 1 | 完美预测（确定性） |
| PPL = 10 | 平均从 10 个选项中选择 |
| PPL = 100 | 平均从 100 个选项中选择 |
| PPL = $\|V\|$ | 均匀分布（随机猜测） |

**GPT-2 在各数据集上的困惑度**：

| 数据集 | GPT-2 Small | GPT-2 XL |
|--------|:----------:|:--------:|
| WikiText-103 | 29.41 | 17.48 |
| PTB | 65.85 | 35.76 |
| WebText (test) | 18.34 | 10.70 |

> **关键发现**：GPT-2 在多个数据集上**零样本**取得了当时的最优困惑度，且随模型规模增大持续下降。

---

## 5. 训练优化方法总结

### 5.1 预训练策略

**数据处理**：

| 属性 | 值 |
|------|-----|
| 分词方法 | Byte-level BPE |
| 词表大小 | 50,257 |
| 上下文长度 | 1024 tokens |
| 数据规模 | ~40GB 文本 |

**Byte Pair Encoding 核心思想**：

从字节级别开始，反复合并最高频的相邻 token 对：

$$
\text{vocab}_{k+1} = \text{vocab}_k \cup \{\text{merge}(\arg\max_{(a,b)} \text{freq}(a, b))\}
$$

直到词表达到目标大小。BPE 的优势在于可以编码任意 UTF-8 文本。

### 5.2 优化器与学习率调度

**Adam 优化器**（与 BERT 相同）：

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

$$
\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

**学习率调度**（余弦退火）：

$$
\boxed{\eta(t) = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t}{T}\pi\right)\right)}
$$

**超参数设置**（GPT-2 Small）：

| 超参数 | 值 |
|--------|-----|
| 学习率 $\eta_{\max}$ | $2.5 \times 10^{-4}$ |
| Batch Size | 512 |
| 上下文长度 | 1024 |
| 总步数 | ~800K |
| Warmup | 线性预热 |
| Adam $\beta_1$ | 0.9 |
| Adam $\beta_2$ | 0.999 |
| Weight Decay | 0.01 |

### 5.3 训练稳定性技巧

**1. Pre-Norm 结构**（前文 3.3 节已讨论）

**2. 残差连接初始化缩放**：

GPT-2 对残差路径的输出层进行缩放初始化：

$$
W_{\text{residual}} \sim \mathcal{N}\left(0, \frac{0.02}{\sqrt{2L}}\right)
$$

其中 $L$ 是层数。这确保初始化时残差路径的贡献随深度递减，避免前向传播中的信号爆炸。

**3. 梯度裁剪**：

$$
\hat{g} = \begin{cases}
g & \text{if } \|g\| \leq c \\
c \cdot \frac{g}{\|g\|} & \text{if } \|g\| > c
\end{cases}
$$

GPT-2 使用全局梯度范数裁剪，$c = 1.0$。

---

## 6. 从数学到代码：完整实现

### 6.1 NumPy 实现核心组件

```python
import numpy as np

def softmax(x, axis=-1):
    """
    数值稳定的 Softmax

    数学公式:
        softmax(x_i) = exp(x_i - max(x)) / Σ exp(x_j - max(x))
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def gelu(x):
    """
    GELU 激活函数（GPT-2 使用）

    数学公式:
        GELU(x) ≈ 0.5x(1 + tanh[√(2/π)(x + 0.044715x³)])
    """
    return 0.5 * x * (1.0 + np.tanh(
        np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))
    ))


def layer_norm(x, gamma, beta, eps=1e-5):
    """
    层归一化

    数学公式:
        LN(x) = γ * (x - μ) / √(σ² + ε) + β
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


def causal_mask(seq_len):
    """
    生成因果掩码（下三角矩阵）

    M[i,j] = 0 if j <= i else -inf
    """
    mask = np.triu(np.ones((seq_len, seq_len)) * (-1e9), k=1)
    return mask  # (seq_len, seq_len)


def causal_self_attention_numpy(Q, K, V, mask):
    """
    因果自注意力 (NumPy)

    数学公式:
        Attention(Q, K, V) = softmax(QK^T/√d_k + M_causal) V

    参数:
        Q: (batch, heads, seq_len, d_k)
        K: (batch, heads, seq_len, d_k)
        V: (batch, heads, seq_len, d_k)
        mask: (seq_len, seq_len) 因果掩码

    返回:
        output: (batch, heads, seq_len, d_k)
        weights: (batch, heads, seq_len, seq_len)
    """
    d_k = Q.shape[-1]
    scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    scores = scores + mask  # 广播加因果掩码
    weights = softmax(scores, axis=-1)
    output = np.matmul(weights, V)
    return output, weights


def gpt2_block_numpy(x, ln1_g, ln1_b, Wq, Wk, Wv, Wo,
                     ln2_g, ln2_b, W1, b1, W2, b2,
                     num_heads, mask):
    """
    GPT-2 Transformer 块 (Pre-Norm) (NumPy)

    数学公式:
        a = x + MultiHeadAttn(LN(x))
        output = a + FFN(LN(a))
    """
    batch, seq_len, d_model = x.shape
    d_k = d_model // num_heads

    # --- Pre-Norm → Self-Attention ---
    x_norm = layer_norm(x, ln1_g, ln1_b)

    Q = np.dot(x_norm, Wq).reshape(batch, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    K = np.dot(x_norm, Wk).reshape(batch, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    V = np.dot(x_norm, Wv).reshape(batch, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)

    attn_out, attn_w = causal_self_attention_numpy(Q, K, V, mask)
    attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch, seq_len, d_model)
    attn_out = np.dot(attn_out, Wo)

    a = x + attn_out  # 残差连接

    # --- Pre-Norm → FFN ---
    a_norm = layer_norm(a, ln2_g, ln2_b)
    ffn_out = gelu(np.dot(a_norm, W1) + b1)
    ffn_out = np.dot(ffn_out, W2) + b2

    output = a + ffn_out  # 残差连接
    return output, attn_w


def temperature_sampling_numpy(logits, temperature=1.0):
    """
    温度采样

    数学公式:
        P(x_t = w) = softmax(logits / T)
    """
    scaled = logits / temperature
    probs = softmax(scaled)
    return np.random.choice(len(probs), p=probs)


def top_k_sampling_numpy(logits, k=10, temperature=1.0):
    """
    Top-k 采样

    步骤:
        1. 取 logits 最大的 k 个
        2. 对这 k 个做温度缩放 + softmax
        3. 从中采样
    """
    top_k_idx = np.argsort(logits)[-k:]
    top_k_logits = logits[top_k_idx] / temperature
    top_k_probs = softmax(top_k_logits)
    chosen = np.random.choice(top_k_idx, p=top_k_probs)
    return chosen


def top_p_sampling_numpy(logits, p=0.9, temperature=1.0):
    """
    Top-p (Nucleus) 采样

    步骤:
        1. 对 logits 排序
        2. 累积概率达到 p 时截断
        3. 在截断后的候选集中采样
    """
    scaled = logits / temperature
    probs = softmax(scaled)
    sorted_idx = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_idx]
    cumsum = np.cumsum(sorted_probs)
    cutoff = np.searchsorted(cumsum, p) + 1
    top_idx = sorted_idx[:cutoff]
    top_probs = probs[top_idx]
    top_probs = top_probs / top_probs.sum()
    return np.random.choice(top_idx, p=top_probs)


# ========== 测试 NumPy 实现 ==========
if __name__ == "__main__":
    np.random.seed(42)
    batch, seq_len, d_model, num_heads = 2, 8, 64, 4
    d_k = d_model // num_heads
    vocab_size = 100

    # 测试因果掩码
    mask = causal_mask(seq_len)
    print(f"因果掩码形状: {mask.shape}")
    print(f"mask[0,:] = {mask[0, :4]}...")  # [0, -inf, -inf, ...]

    # 测试因果注意力
    Q = np.random.randn(batch, num_heads, seq_len, d_k)
    K = np.random.randn(batch, num_heads, seq_len, d_k)
    V = np.random.randn(batch, num_heads, seq_len, d_k)
    out, weights = causal_self_attention_numpy(Q, K, V, mask)
    print(f"注意力输出形状: {out.shape}")
    print(f"位置0的注意力权重: {weights[0, 0, 0, :4]}")  # 只有[0]非零
    print(f"因果性验证 - A[0,1]=0: {weights[0, 0, 0, 1] < 1e-6}")

    # 测试 GPT-2 块
    x = np.random.randn(batch, seq_len, d_model) * 0.02
    params = {
        'ln1_g': np.ones(d_model), 'ln1_b': np.zeros(d_model),
        'Wq': np.random.randn(d_model, d_model) * 0.02,
        'Wk': np.random.randn(d_model, d_model) * 0.02,
        'Wv': np.random.randn(d_model, d_model) * 0.02,
        'Wo': np.random.randn(d_model, d_model) * 0.02,
        'ln2_g': np.ones(d_model), 'ln2_b': np.zeros(d_model),
        'W1': np.random.randn(d_model, d_model * 4) * 0.02,
        'b1': np.zeros(d_model * 4),
        'W2': np.random.randn(d_model * 4, d_model) * 0.02,
        'b2': np.zeros(d_model),
    }
    block_out, _ = gpt2_block_numpy(
        x, params['ln1_g'], params['ln1_b'],
        params['Wq'], params['Wk'], params['Wv'], params['Wo'],
        params['ln2_g'], params['ln2_b'],
        params['W1'], params['b1'], params['W2'], params['b2'],
        num_heads, mask
    )
    print(f"GPT-2 块输出形状: {block_out.shape}")

    # 测试采样策略
    logits = np.random.randn(vocab_size)
    print(f"温度采样 (T=0.5): {temperature_sampling_numpy(logits, 0.5)}")
    print(f"Top-k 采样 (k=10): {top_k_sampling_numpy(logits, k=10)}")
    print(f"Top-p 采样 (p=0.9): {top_p_sampling_numpy(logits, p=0.9)}")
```

### 6.2 PyTorch 完整实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class GPT2Embedding(nn.Module):
    """
    GPT-2 输入嵌入层

    数学公式:
        E(t) = W_e[x_t] + W_p[t]

    参数:
        vocab_size: 词表大小 |V|
        d_model: 嵌入维度
        max_len: 最大上下文长度
        dropout: dropout 概率
    """
    def __init__(self, vocab_size: int, d_model: int,
                 max_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: (batch, seq_len) token 索引
        返回:
            (batch, seq_len, d_model) 嵌入向量
        """
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0)
        emb = self.token_embedding(x) + self.position_embedding(pos)
        return self.dropout(emb)


class CausalSelfAttention(nn.Module):
    """
    因果自注意力（GPT-2 核心组件）

    数学公式:
        Attention(Q, K, V) = softmax(QK^T/√d_k + M_causal) V

    参数:
        d_model: 模型维度
        num_heads: 注意力头数
        max_len: 最大序列长度（用于注册因果掩码）
        dropout: dropout 概率
    """
    def __init__(self, d_model: int, num_heads: int,
                 max_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 合并 Q/K/V 投影为单个矩阵（效率更高）
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # 注册因果掩码（不参与梯度计算）
        mask = torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        参数:
            x: (batch, seq_len, d_model)
        返回:
            output: (batch, seq_len, d_model)
            weights: (batch, heads, seq_len, seq_len)
        """
        B, T, C = x.size()

        # 一次性计算 Q, K, V
        qkv = self.c_attn(x)  # (B, T, 3*d_model)
        Q, K, V = qkv.split(self.d_model, dim=2)

        # 分割多头: (B, T, d_model) → (B, heads, T, d_k)
        Q = Q.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        # 注意力分数 + 因果掩码
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(self.causal_mask[:T, :T], float('-inf'))
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        # 加权求和 → 拼接 → 投影
        out = torch.matmul(weights, V)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.c_proj(out))

        return out, weights


class GPT2FFN(nn.Module):
    """
    GPT-2 前馈网络

    数学公式:
        FFN(x) = GELU(xW_1 + b_1)W_2 + b_2
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.c_fc = nn.Linear(d_model, d_ff)
        self.c_proj = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.c_fc(x))
        x = self.dropout(self.c_proj(x))
        return x


class GPT2Block(nn.Module):
    """
    GPT-2 Transformer 块 (Pre-Norm)

    数学公式:
        a = x + MultiHeadAttn(LN(x))
        output = a + FFN(LN(a))
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 max_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model, eps=1e-5)
        self.attn = CausalSelfAttention(d_model, num_heads, max_len, dropout)
        self.ln_2 = nn.LayerNorm(d_model, eps=1e-5)
        self.ffn = GPT2FFN(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        参数:
            x: (batch, seq_len, d_model)
        返回:
            output: (batch, seq_len, d_model)
            attn_weights: (batch, heads, seq_len, seq_len)
        """
        attn_out, attn_w = self.attn(self.ln_1(x))
        x = x + attn_out              # 残差连接 1
        x = x + self.ffn(self.ln_2(x))  # 残差连接 2
        return x, attn_w


class GPT2Model(nn.Module):
    """
    完整 GPT-2 模型

    结构:
        Input → [Embedding] → [Block x L] → [LN] → [LM Head]
    """
    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        d_ff: int = 3072,
        max_len: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model

        self.embedding = GPT2Embedding(vocab_size, d_model, max_len, dropout)
        self.blocks = nn.ModuleList([
            GPT2Block(d_model, num_heads, d_ff, max_len, dropout)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model, eps=1e-5)  # 最终 LayerNorm

        # 语言模型头（与嵌入共享权重）
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.token_embedding.weight

        self._init_weights()

    def _init_weights(self):
        """
        GPT-2 权重初始化: N(0, 0.02)
        残差投影层缩放: N(0, 0.02/√(2L))
        """
        num_layers = len(self.blocks)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                # 残差路径缩放
                if 'c_proj' in name:
                    nn.init.normal_(
                        module.weight,
                        mean=0.0,
                        std=0.02 / math.sqrt(2 * num_layers)
                    )
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor,
                targets: Optional[torch.Tensor] = None) -> dict:
        """
        参数:
            x: (batch, seq_len) token 索引
            targets: (batch, seq_len) 目标 token（训练时使用）
        返回:
            logits: (batch, seq_len, vocab_size)
            loss: 标量（如果提供 targets）
        """
        h = self.embedding(x)  # (B, T, d_model)

        all_attn = []
        for block in self.blocks:
            h, attn_w = block(h)
            all_attn.append(attn_w)

        h = self.ln_f(h)  # 最终 LayerNorm
        logits = self.lm_head(h)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # 移位：logits[t] 预测 targets[t] (即 x[t+1])
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100
            )

        return {"logits": logits, "loss": loss, "attention": all_attn}

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: Optional[int] = None,
                 top_p: Optional[float] = None) -> torch.Tensor:
        """
        自回归文本生成

        参数:
            idx: (batch, seq_len) 初始 token 序列
            max_new_tokens: 最大生成 token 数
            temperature: 温度参数 (T > 1 更随机, T < 1 更确定)
            top_k: Top-k 采样的 k 值
            top_p: Top-p 采样的阈值
        """
        for _ in range(max_new_tokens):
            # 截断到最大上下文长度
            idx_cond = idx if idx.size(1) <= 1024 else idx[:, -1024:]

            out = self.forward(idx_cond)
            logits = out["logits"][:, -1, :]  # 最后一个位置的 logits

            # 温度缩放
            logits = logits / temperature

            # Top-k 过滤
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Top-p (Nucleus) 过滤
            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                mask = cum_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[mask] = float('-inf')
                logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)

        return idx


# ========== 完整测试 ==========
if __name__ == "__main__":
    # 缩小版超参数
    vocab_size, d_model, num_heads = 1000, 128, 4
    num_layers, d_ff, max_len = 2, 512, 64
    batch_size, seq_len = 4, 32

    # 1. 创建模型
    model = GPT2Model(vocab_size, d_model, num_heads, num_layers,
                      d_ff, max_len)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {total_params:,}")

    # 2. 准备输入（自回归：输入右移一位作为目标）
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = torch.cat([input_ids[:, 1:],
                         torch.full((batch_size, 1), -100)], dim=1)

    # 3. 前向传播 + 损失
    model.eval()
    with torch.no_grad():
        out = model(input_ids, targets)
    print(f"Logits 形状: {out['logits'].shape}")
    print(f"Loss: {out['loss'].item():.4f}")
    print(f"PPL:  {torch.exp(out['loss']).item():.2f}")

    # 4. 验证因果性
    attn = out["attention"][0]  # 第 1 层
    print(f"\n因果性验证:")
    print(f"  A[0,3] (future): {attn[0, 0, 0, 3].item():.6f}")  # 应为 0
    print(f"  A[3,0] (past):   {attn[0, 0, 3, 0].item():.6f}")  # 应 > 0
    upper_tri = torch.triu(attn[0, 0], diagonal=1)
    print(f"  上三角最大值:     {upper_tri.max().item():.6f}")    # 应为 0

    # 5. 训练一步
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=2.5e-4, weight_decay=0.01)
    out = model(input_ids, targets)
    out["loss"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    opt.step()
    print(f"\n训练后 Loss: {out['loss'].item():.4f}")

    # 6. 文本生成测试
    model.eval()
    prompt = torch.randint(0, vocab_size, (1, 5))
    generated = model.generate(prompt, max_new_tokens=20,
                               temperature=0.8, top_k=50)
    print(f"\n生成序列长度: {generated.shape[1]} (prompt=5 + new=20)")
    print(f"生成的 token IDs: {generated[0, 5:].tolist()}")

    print("\n✅ GPT-2 模型测试通过！")
```

---

## 7. 文本生成策略：采样与搜索

### 7.1 贪心搜索与 Beam Search

**贪心搜索**：每步选择概率最高的 token：

$$
x_t = \arg\max_{w \in V} P(w \mid x_{<t})
$$

**优点**：简单高效。**缺点**：生成的文本重复、缺乏多样性。

**Beam Search**：维护 $B$ 个候选序列，每步扩展所有候选：

$$
\text{score}(y_{1:t}) = \sum_{i=1}^{t} \log P(y_i \mid y_{<i})
$$

最终选择得分最高的完整序列。典型 beam width $B \in \{4, 5, 10\}$。

**Beam Search 的长度惩罚**：

$$
\text{score}_{\text{normalized}} = \frac{\log P(y_{1:T})}{T^\alpha}
$$

其中 $\alpha \in [0.6, 1.0]$ 是长度惩罚因子。

> **问题**：对于开放式生成（如故事续写），贪心搜索和 Beam Search 都倾向于生成**安全但无趣**的文本。

### 7.2 温度采样

通过温度参数 $T$ 控制分布的"尖锐度"：

$$
\boxed{P(x_t = w \mid x_{<t}; T) = \frac{\exp(z_w / T)}{\sum_{w'} \exp(z_{w'} / T)}}
$$

其中 $z_w$ 是 logit 值。

**温度的效果**：

| 温度 $T$ | 分布特性 | 生成效果 |
|----------|---------|---------|
| $T \to 0$ | 趋向 one-hot（最大值） | 等价于贪心，确定性 |
| $T = 1$ | 原始分布 | 标准采样 |
| $T > 1$ | 更平坦（更均匀） | 更随机、更多样 |
| $T \to \infty$ | 趋向均匀分布 | 完全随机 |

**数学分析**：

当 $T \to 0$ 时，设 $z_{\max} = \max_w z_w$：

$$
\lim_{T \to 0} \frac{\exp(z_w / T)}{\sum_{w'} \exp(z_{w'} / T)} = \begin{cases} 1 & \text{if } w = \arg\max z \\ 0 & \text{otherwise} \end{cases}
$$

### 7.3 Top-k 与 Top-p (Nucleus) 采样

**Top-k 采样**（Fan et al., 2018）：

只保留概率最高的 $k$ 个 token，重新归一化后采样：

$$
P'(w) = \begin{cases}
\frac{P(w)}{\sum_{w' \in V_k} P(w')} & \text{if } w \in V_k \\
0 & \text{otherwise}
\end{cases}
$$

其中 $V_k$ 是概率最高的 $k$ 个 token 的集合。

**问题**：固定的 $k$ 不能适应不同位置的概率分布。当分布很集中时（如 "The capital of France is" → "Paris"），$k=50$ 会引入太多噪声；当分布很平坦时，$k=10$ 又会过度截断。

**Top-p (Nucleus) 采样**（Holtzman et al., 2019）：

动态选择最小的 token 集合，使其累积概率 $\geq p$：

$$
\boxed{V_p = \min \{V' \subseteq V : \sum_{w \in V'} P(w) \geq p\}}
$$

然后在 $V_p$ 中重新归一化采样。

**Top-p 的自适应性**：

| 场景 | 分布 | Top-k ($k$=10) | Top-p ($p$=0.9) |
|------|------|:-------------:|:---------------:|
| 确定性高 | 几乎 one-hot | 10 个候选（太多） | ~1-3 个候选 |
| 不确定性高 | 接近均匀 | 10 个候选（太少） | ~数百个候选 |

**组合策略**（GPT-2 推荐）：

$$
V_{\text{final}} = V_k \cap V_p
$$

同时使用 Top-k 和 Top-p，取交集。先用 Top-k 做粗过滤，再用 Top-p 做自适应截断。

### 7.4 生成策略对比与可视化

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_sampling_strategies(logits, k=10, p=0.9, temperatures=[0.5, 1.0, 1.5]):
    """
    可视化不同采样策略的概率分布

    参数:
        logits: 原始 logits 向量
        k: Top-k 的 k 值
        p: Top-p 的 p 值
        temperatures: 温度列表
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 原始分布
    probs = np.exp(logits - np.max(logits))
    probs = probs / probs.sum()
    sorted_idx = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_idx]

    # (1) 不同温度
    ax = axes[0, 0]
    for T in temperatures:
        scaled = logits / T
        p_t = np.exp(scaled - np.max(scaled))
        p_t = p_t / p_t.sum()
        p_sorted = p_t[sorted_idx]
        ax.plot(range(min(30, len(p_sorted))), p_sorted[:30], label=f'T={T}')
    ax.set_title('Temperature Sampling')
    ax.set_xlabel('Token rank')
    ax.set_ylabel('Probability')
    ax.legend()

    # (2) Top-k
    ax = axes[0, 1]
    ax.bar(range(min(30, len(sorted_probs))), sorted_probs[:30],
           color=['#2196F3' if i < k else '#BDBDBD' for i in range(30)])
    ax.axvline(x=k-0.5, color='red', linestyle='--', label=f'k={k}')
    ax.set_title(f'Top-k Sampling (k={k})')
    ax.set_xlabel('Token rank')
    ax.legend()

    # (3) Top-p
    ax = axes[1, 0]
    cumsum = np.cumsum(sorted_probs)
    cutoff = np.searchsorted(cumsum, p) + 1
    ax.bar(range(min(30, len(sorted_probs))), sorted_probs[:30],
           color=['#4CAF50' if i < cutoff else '#BDBDBD' for i in range(30)])
    ax.axvline(x=cutoff-0.5, color='red', linestyle='--',
               label=f'p={p}, cutoff={cutoff}')
    ax.set_title(f'Top-p (Nucleus) Sampling (p={p})')
    ax.set_xlabel('Token rank')
    ax.legend()

    # (4) 累积概率
    ax = axes[1, 1]
    ax.plot(range(min(50, len(cumsum))), cumsum[:50], 'b-')
    ax.axhline(y=p, color='red', linestyle='--', label=f'p={p}')
    ax.axvline(x=cutoff-0.5, color='green', linestyle='--',
               label=f'cutoff={cutoff}')
    ax.set_title('Cumulative Probability')
    ax.set_xlabel('Token rank')
    ax.set_ylabel('Cumulative prob')
    ax.legend()

    plt.tight_layout()
    plt.savefig("gpt2_sampling_strategies.png", dpi=150)
    plt.show()

# 示例
if __name__ == "__main__":
    np.random.seed(42)
    logits = np.random.randn(200) * 2
    logits[0] += 5  # 人为制造一个高概率 token
    visualize_sampling_strategies(logits)
```

---

## 8. 与其他模型的关系

### 8.1 GPT-1 → GPT-2 → GPT-3 演进

| 特性 | GPT-1 (2018) | GPT-2 (2019) | GPT-3 (2020) |
|------|:-----------:|:------------:|:------------:|
| 参数量 | 117M | 1.5B | 175B |
| 层数 | 12 | 48 | 96 |
| 上下文长度 | 512 | 1024 | 2048 |
| 训练数据 | BookCorpus | WebText (40GB) | ~570GB |
| 范式 | 预训练+微调 | Zero-shot | In-context Learning |

**演进的核心主题**：

$$
\boxed{\text{GPT-1: 预训练+微调} \xrightarrow{\text{去掉微调}} \text{GPT-2: Zero-shot} \xrightarrow{\text{示例学习}} \text{GPT-3: Few-shot}}
$$

**GPT-1 → GPT-2 的关键变化**：

1. **规模**：117M → 1.5B（约 13x）
2. **目标**：从"预训练+微调"转向"Zero-shot"
3. **架构**：Post-Norm → Pre-Norm
4. **数据**：BookCorpus → WebText（更大更多样）

### 8.2 GPT-2 vs BERT：生成 vs 理解

| 维度 | BERT | GPT-2 |
|------|------|-------|
| **架构** | Transformer Encoder | Transformer Decoder |
| **注意力** | 全连接（双向） | 因果掩码（单向） |
| **预训练** | MLM + NSP | 自回归 LM |
| **参数共享** | 无 | 输入/输出嵌入共享 |
| **LayerNorm** | Post-Norm | Pre-Norm |
| **NLU 任务** | ✅ 强（双向上下文） | ⚠️ 较弱（单向） |
| **NLG 任务** | ❌ 不适用 | ✅ 强（自回归生成） |

**互补性**：

$$
\underbrace{\text{BERT}}_{\text{理解}} + \underbrace{\text{GPT-2}}_{\text{生成}} \longrightarrow \underbrace{\text{T5 / GPT-3}}_{\text{统一}}
$$

### 8.3 自回归范式的后续发展

```
GPT-1 (2018) ── 预训练+微调
  └── GPT-2 (2019) ── Zero-shot
       └── GPT-3 (2020) ── In-context Learning / Few-shot
            ├── Codex (2021) ── 代码生成
            ├── InstructGPT (2022) ── RLHF 对齐
            │    └── ChatGPT (2022)
            ├── GPT-4 (2023) ── 多模态
            └── LLaMA (2023) ── 开源自回归 LLM
```

**GPT-2 在历史中的定位**：

GPT-2 是第一个令人信服地展示了**规模带来涌现能力**的模型。它证明了：

1. **足够大的语言模型可以 Zero-shot 完成多种任务**
2. **数据质量和规模同样重要**
3. **自回归范式的潜力远未被挖掘**

这些发现直接启发了 GPT-3 的 Scaling Laws 研究和后续大模型的发展方向。

---

## 扩展阅读与实现

### 问题 1：为什么 GPT-2 使用 Pre-Norm 而不是 Post-Norm？

**解答**：

在 Post-Norm 中：
$$
x^{(l)} = \text{LN}(x^{(l-1)} + f(x^{(l-1)}))
$$

残差路径 $x^{(l-1)}$ 在加上子层输出后才被归一化。当层数很深时，残差路径上的值可能不断累积，导致训练不稳定。

在 Pre-Norm 中：
$$
x^{(l)} = x^{(l-1)} + f(\text{LN}(x^{(l-1)}))
$$

残差路径是**直通的**，梯度可以无障碍地从最后一层流回第一层：

$$
\frac{\partial x^{(L)}}{\partial x^{(0)}} = I + \sum_{l=1}^{L} \frac{\partial f^{(l)}}{\partial x^{(0)}}
$$

恒等项 $I$ 确保梯度不会消失。实验表明，Pre-Norm 在深层模型（>12 层）中训练更稳定，无需精心调节学习率预热。

### 问题 2：权重绑定为什么有效？

**解答**：

权重绑定（Weight Tying）让输入嵌入矩阵 $W_e$ 同时作为输出投影矩阵：

$$
P(x_t = w) = \text{softmax}(h_t \cdot W_e^\top)_w = \text{softmax}(h_t \cdot e_w)
$$

**有效的原因**：

1. **参数效率**：嵌入矩阵通常是最大的参数组（GPT-2 中约 39M）。共享权重几乎将这部分参数减半。

2. **语义一致性**：输入空间和输出空间共享同一组词向量。预测"下一个词"就是在嵌入空间中寻找与 $h_t$ 最接近的词向量——这在语义上是自然的。

3. **正则化效果**：共享权重减少了过拟合风险，特别是在词表很大时。

### 问题 3：Byte-level BPE 如何处理任意文本？

**解答**：

传统 BPE 在字符级别操作，需要预定义字符集。GPT-2 的创新是使用 **Byte-level BPE**：

1. 将文本编码为 UTF-8 字节序列（256 个基本 token）
2. 在字节级别运行 BPE 合并

**优势**：
- 任何 UTF-8 文本都可以编码（中文、日文、emoji 等）
- 词表大小可控（50,257 个 token）
- 无需特殊的 `[UNK]` token

**示例**：

```
"Hello" → 字节 [72, 101, 108, 108, 111] → BPE ["Hello"]
"你好"  → 字节 [228, 189, 160, 229, 165, 189] → BPE ["ä½", "ł", "å¥", "½"]
```

### 问题 4：GPT-2 的 Zero-shot 能力有多强？

**解答**：

GPT-2 (1.5B) 在多个基准上的 Zero-shot 表现：

| 任务 | 数据集 | GPT-2 Zero-shot | 有监督 SOTA |
|------|--------|:---------------:|:-----------:|
| 语言建模 | WikiText-103 | 17.48 PPL | 15.79 PPL |
| 阅读理解 | CoQA | 55.0 F1 | 89.8 F1 |
| 翻译 (En→Fr) | WMT-14 | 5.0 BLEU | 45.6 BLEU |
| 摘要 | CNN/DM | 21.58 R-1 | 40.0 R-1 |

**关键观察**：
- 语言建模：接近甚至超过有监督 SOTA
- 理解类任务：远低于有监督方法，但作为 Zero-shot 已经令人印象深刻
- 生成类任务：质量有限，但展示了方向

### 问题 5：KV Cache 如何加速自回归生成？

**解答**：

自回归生成时，每生成一个新 token 都需要重新计算所有位置的注意力。但由于因果掩码，已生成 token 的 K、V 不会改变。

**KV Cache 策略**：

缓存已计算的 $K$ 和 $V$，每步只计算新 token 的 $Q$、$K$、$V$：

$$
K_{\text{cached}} = [k_1, k_2, \ldots, k_{t-1}], \quad V_{\text{cached}} = [v_1, v_2, \ldots, v_{t-1}]
$$

$$
K_t = [K_{\text{cached}}; k_t], \quad V_t = [V_{\text{cached}}; v_t]
$$

**复杂度对比**：

| 方法 | 每步计算量 | 内存 |
|------|----------|------|
| 无缓存 | $O(t^2 \cdot d)$ | $O(d)$ |
| KV Cache | $O(t \cdot d)$ | $O(t \cdot d)$ |

对于长序列生成，KV Cache 将时间复杂度从 $O(n^3)$ 降至 $O(n^2)$（总生成 $n$ 个 token）。

---

## 参考资源

### 经典论文

1. Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf). OpenAI Blog.
   - **贡献**：提出 GPT-2，证明大规模语言模型具有 Zero-shot 多任务能力

2. Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. (2018). [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf). OpenAI.
   - **贡献**：GPT-1，提出生成式预训练+判别式微调范式

3. Holtzman, A., Buys, J., Du, L., Forbes, M., & Choi, Y. (2019). [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751). ICLR 2020.
   - **贡献**：提出 Top-p (Nucleus) 采样，解决文本退化问题

4. Brown, T., et al. (2020). [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165). NeurIPS 2020.
   - **贡献**：GPT-3，将 GPT-2 的思路扩展到 175B 参数

### 教材与书籍

5. Jurafsky, D., & Martin, J. H. [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/). 3rd ed. (Draft).
   - **章节**：第 10 章详细讲解语言模型与文本生成

### 在线资源与教程

6. Alammar, J. [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/).
   - **内容**：GPT-2 架构和生成过程的直观图解

7. Karpathy, A. [nanoGPT](https://github.com/karpathy/nanoGPT).
   - **内容**：GPT-2 的简洁 PyTorch 复现，适合学习

8. Hugging Face. [GPT-2 Documentation](https://huggingface.co/docs/transformers/model_doc/gpt2).
   - **内容**：GPT-2 的工业级实现和使用指南

---

## 附录：符号表

| 符号 | 含义 | 维度/类型 |
|------|------|----------|
| $n$ | 输入序列长度 | 标量 |
| $n_{\text{ctx}}$ | 最大上下文长度 | 标量，1024 |
| $d$ | 隐藏维度（$d_{\text{model}}$） | 标量，GPT-2 Small: 768 |
| $d_k$ | 每个注意力头的维度 | 标量，64 |
| $d_{ff}$ | FFN 隐藏层维度 | 标量，3072 |
| $L$ | Transformer 层数 | 标量，GPT-2 Small: 12 |
| $A$ | 注意力头数 | 标量，GPT-2 Small: 12 |
| $\|V\|$ | BPE 词表大小 | 标量，50,257 |
| $x_t$ | 位置 $t$ 的 token | 整数索引 |
| $x_{<t}$ | 位置 $t$ 之前的所有 token | 序列 $(x_1, \ldots, x_{t-1})$ |
| $h_t^{(l)}$ | 第 $l$ 层位置 $t$ 的隐藏状态 | $(d,)$ |
| $W_e$ | Token 嵌入矩阵（输入/输出共享） | $(\|V\|, d)$ |
| $W_p$ | 位置嵌入矩阵 | $(n_{\text{ctx}}, d)$ |
| $M^{\text{causal}}$ | 因果注意力掩码 | $(n, n)$，上三角为 $-\infty$ |
| $Q, K, V$ | 查询、键、值矩阵 | $(n, d_k)$ |
| $A_{ij}$ | 注意力权重（位置 $i$ 对 $j$） | 标量，$A_{ij}=0$ 当 $j>i$ |
| $T$ | 温度参数 | 标量，$T > 0$ |
| $k$ | Top-k 采样参数 | 标量，正整数 |
| $p$ | Top-p 采样阈值 | 标量，$p \in (0, 1]$ |
| $\mathcal{L}$ | 语言模型损失（负对数似然） | 标量 |
| $\text{PPL}$ | 困惑度 $\exp(\mathcal{L})$ | 标量 |
| $\ell(\cdot, \cdot)$ | 交叉熵损失函数 | 函数 |
| $B$ | Beam Search 的 beam width | 标量 |

**典型维度示例（GPT-2 Small）：**
- $d = 768$（隐藏维度）
- $d_k = 64$（每头维度）
- $d_{ff} = 3072$（FFN 维度）
- $|V| = 50{,}257$（词表大小）
- $n_{\text{ctx}} = 1024$（最大上下文长度）

---

最后更新：2026-03-19
