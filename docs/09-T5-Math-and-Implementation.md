# T5 数学原理与实现 —— 统一 Text-to-Text 框架的完整推导

> **前置知识**：Transformer Encoder-Decoder、自注意力机制、交叉熵损失、Python 基础  
> **与前面内容的联系**：建议先学习 [BERT](./07-BERT-Math-and-Implementation.md)（编码器）和 [GPT-2](./08-GPT2-Math-and-Implementation.md)（解码器），T5 统一了两者  
> **与后续内容的联系**：T5 的 Text-to-Text 思想直接影响了 GPT-3 的 In-context Learning 和后续统一范式

---

## 目录

1. [引言：为什么需要统一的 Text-to-Text 框架？](#1-引言为什么需要统一的-text-to-text-框架)
   - 1.1 [NLP 范式的碎片化问题](#11-nlp-范式的碎片化问题)
   - 1.2 [T5 的核心洞察](#12-t5-的核心洞察)
   - 1.3 [本科数学知识映射表](#13-本科数学知识映射表)
2. [核心思想：一切皆 Text-to-Text](#2-核心思想一切皆-text-to-text)
   - 2.1 [统一任务格式](#21-统一任务格式)
   - 2.2 [任务前缀的数学意义](#22-任务前缀的数学意义)
   - 2.3 [C4 数据集与数据处理](#23-c4-数据集与数据处理)
3. [T5 架构的数学描述](#3-t5-架构的数学描述)
   - 3.1 [Encoder-Decoder 完整架构](#31-encoder-decoder-完整架构)
   - 3.2 [相对位置编码（Relative Position Bias）](#32-相对位置编码relative-position-bias)
   - 3.3 [交叉注意力机制](#33-交叉注意力机制)
   - 3.4 [T5 各规模配置](#34-t5-各规模配置)
4. [预训练目标：Span Corruption](#4-预训练目标span-corruption)
   - 4.1 [Span Corruption 数学定义](#41-span-corruption-数学定义)
   - 4.2 [与 BERT MLM 和 GPT-2 LM 的对比](#42-与-bert-mlm-和-gpt-2-lm-的对比)
   - 4.3 [损失函数与梯度分析](#43-损失函数与梯度分析)
5. [训练优化方法总结](#5-训练优化方法总结)
   - 5.1 [Adafactor 优化器](#51-adafactor-优化器)
   - 5.2 [学习率调度与训练策略](#52-学习率调度与训练策略)
   - 5.3 [多任务混合训练](#53-多任务混合训练)
6. [从数学到代码：完整实现](#6-从数学到代码完整实现)
   - 6.1 [NumPy 实现核心组件](#61-numpy-实现核心组件)
   - 6.2 [PyTorch 完整实现](#62-pytorch-完整实现)
7. [T5 的系统性消融实验](#7-t5-的系统性消融实验)
   - 7.1 [架构对比](#71-架构对比)
   - 7.2 [预训练目标对比](#72-预训练目标对比)
   - 7.3 [关键实验结论](#73-关键实验结论)
8. [与其他模型的关系](#8-与其他模型的关系)
   - 8.1 [BERT vs GPT-2 vs T5](#81-bert-vs-gpt-2-vs-t5)
   - 8.2 [T5 的后续影响](#82-t5-的后续影响)

[扩展阅读与实现](#扩展阅读与实现)

[参考资源](#参考资源)

附录：[符号表](#附录符号表)

---

## 1. 引言：为什么需要统一的 Text-to-Text 框架？

### 1.1 NLP 范式的碎片化问题

在 T5 之前，不同 NLP 任务需要不同的模型架构和输出形式：

| 任务类型 | 典型任务 | 输出形式 | 代表模型 |
|---------|---------|---------|---------|
| 分类 | 情感分析、NLI | 类别标签 | BERT + 分类头 |
| 序列标注 | NER、POS | 每 token 标签 | BERT + 序列头 |
| 生成 | 翻译、摘要 | 文本序列 | Seq2Seq |
| 回归 | STS | 连续分数 | BERT + 回归头 |

每种任务都需要设计特定的输出层和损失函数——这违背了迁移学习"一个模型解决所有问题"的理想。

### 1.2 T5 的核心洞察

T5 提出了一个优雅的统一：**将所有 NLP 任务都转化为 Text-to-Text 格式**。

$$
\boxed{\text{任何 NLP 任务} \to \text{输入文本} \xrightarrow{\text{T5}} \text{输出文本}}
$$

**示例**：

| 任务 | 输入（含前缀） | 输出 |
|------|--------------|------|
| 翻译 | "translate English to German: That is good" | "Das ist gut" |
| 情感分类 | "sst2 sentence: This movie is great" | "positive" |
| 摘要 | "summarize: [长文本]" | "[摘要文本]" |
| 相似度 | "stsb sentence1: A man is eating. sentence2: A man eats." | "4.2" |

**革命性意义**：

1. **统一架构**：所有任务共享同一个 Encoder-Decoder 模型
2. **统一损失**：所有任务都用交叉熵（预测目标 token 序列）
3. **统一接口**：输入/输出都是文本字符串
4. **系统性研究**：在统一框架下公平对比各种设计选择

### 1.3 本科数学知识映射表

| 数学概念 | T5 中的应用 | 代码对应 |
|---------|------------|---------|
| 条件概率 $P(Y \mid X)$ | Encoder-Decoder 建模 | `model(input_ids, decoder_ids)` |
| 交叉熵 $H(p, q)$ | 统一损失函数 | `F.cross_entropy()` |
| 相对距离函数 | 相对位置编码 | `relative_position_bias()` |
| 概率链式法则 | 自回归解码 | 逐 token 生成 |
| 注意力加权和 | 交叉注意力 | `cross_attention()` |

---

## 2. 核心思想：一切皆 Text-to-Text

### 2.1 统一任务格式

T5 的输入格式为 **"任务前缀: 实际输入"**，输出为**目标文本字符串**。

**形式化定义**：

给定任务 $\tau$、输入 $x$、输出 $y$，T5 将其统一为：

$$
\text{input}_\tau = \text{prefix}(\tau) \oplus x, \quad \text{target}_\tau = \text{verbalize}(y)
$$

其中 $\oplus$ 表示字符串拼接，$\text{verbalize}(\cdot)$ 将任意输出转为文本。

**分类任务的文本化**：

传统方法输出类别索引 $y \in \{0, 1, \ldots, C-1\}$，T5 输出类别名称：

$$
\text{传统: } f(x) \to \{0, 1\} \quad \xrightarrow{\text{T5}} \quad f(x) \to \{\text{"negative"}, \text{"positive"}\}
$$

**回归任务的文本化**：

$$
\text{STS-B: } f(x_1, x_2) \to 3.8 \quad \xrightarrow{\text{T5}} \quad f(x_1, x_2) \to \text{"3.8"}
$$

将浮点数四舍五入到最近的 0.2，转为字符串。

### 2.2 任务前缀的数学意义

任务前缀从数学上看，是对模型的**条件化**：

$$
P_\theta(y \mid x) \to P_\theta(y \mid \text{prefix}, x)
$$

前缀 token 经过编码器处理后，通过交叉注意力传递给解码器，引导模型进入正确的"任务模式"。

**与 GPT-2 Zero-shot 的区别**：

| 维度 | GPT-2 Zero-shot | T5 |
|------|:---------------:|:---:|
| 架构 | Decoder-only | Encoder-Decoder |
| 前缀 | 自然语言提示 | 固定任务前缀 |
| 训练 | 纯语言建模 | 多任务微调 |
| 效果 | 弱（无监督） | 强（有监督信号） |

### 2.3 C4 数据集与数据处理

T5 使用了精心构建的 **Colossal Clean Crawled Corpus (C4)**：

| 属性 | 值 |
|------|-----|
| 来源 | Common Crawl（2019年4月） |
| 清洗后规模 | ~750GB 文本 |
| Token 数 | ~1T（SentencePiece 编码后） |
| 语言 | 仅英文 |

**清洗步骤**：移除短页面（<5句）、脏词页面、JavaScript/lorem ipsum 页面，基于三句重叠去重。

**SentencePiece 分词**：

T5 使用 SentencePiece（Unigram 模型），词表大小 $|V| = 32{,}000$：

$$
\hat{x} = \arg\max_{x \in \mathcal{S}(X)} \sum_{i=1}^{|x|} \log P(x_i)
$$

其中 $\mathcal{S}(X)$ 是输入 $X$ 的所有可能分词结果。

---

## 3. T5 架构的数学描述

### 3.1 Encoder-Decoder 完整架构

T5 使用标准的 Transformer Encoder-Decoder 架构，但做了几处修改：

**编码器（处理输入文本）**：

$$
\boxed{
\begin{aligned}
\text{Encoder Layer } l: \quad
a^{(l)} &= x^{(l-1)} + \text{SelfAttn}\left(\text{RMSNorm}(x^{(l-1)})\right) \\
x^{(l)} &= a^{(l)} + \text{FFN}\left(\text{RMSNorm}(a^{(l)})\right)
\end{aligned}
}
$$

**解码器（生成输出文本）**：

$$
\boxed{
\begin{aligned}
\text{Decoder Layer } l: \quad
a^{(l)} &= y^{(l-1)} + \text{CausalSelfAttn}\left(\text{RMSNorm}(y^{(l-1)})\right) \\
b^{(l)} &= a^{(l)} + \text{CrossAttn}\left(\text{RMSNorm}(a^{(l)}), \; H^{\text{enc}}\right) \\
y^{(l)} &= b^{(l)} + \text{FFN}\left(\text{RMSNorm}(b^{(l)})\right)
\end{aligned}
}
$$

其中 $H^{\text{enc}} = x^{(L_{\text{enc}})}$ 是编码器的最终输出。

**与原始 Transformer 的关键区别**：

| 特性 | 原始 Transformer | T5 |
|------|:---------------:|:---:|
| 归一化 | Post-Norm (LayerNorm) | Pre-Norm (RMSNorm) |
| 位置编码 | 正弦/余弦（绝对） | 相对位置偏置 |
| 激活函数 | ReLU | ReLU（T5.1 后用 GEGLU） |
| 归一化位置 | 子层之后 | 子层之前 |

**RMSNorm**（简化的 LayerNorm，去掉均值中心化）：

$$
\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma, \quad \text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}
$$

> **Q:** 为什么用 RMSNorm 而不是 LayerNorm？
>
> **A:** RMSNorm 省去了均值计算和偏移参数 $\beta$，在保持效果的同时减少约 10% 的计算量。实验表明归一化的主要作用来自缩放（除以 RMS），而非中心化（减去均值）。

### 3.2 相对位置编码（Relative Position Bias）

T5 不使用绝对位置嵌入，而是在注意力分数中加入**相对位置偏置**：

$$
\boxed{\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + B\right) V}
$$

其中 $B \in \mathbb{R}^{n \times n}$ 是相对位置偏置矩阵：

$$
B_{ij} = r_{b}(\text{clip}(i - j))
$$

**相对距离到桶的映射**（对数分桶）：

T5 将相对距离映射到有限数量的桶（bucket），然后为每个桶学习一个偏置值：

$$
b(i, j) = \begin{cases}
|i - j| & \text{if } |i - j| \leq k_{\text{exact}} \\
k_{\text{exact}} + \left\lfloor \frac{\log(|i - j| / k_{\text{exact}})}{\log(n_{\max} / k_{\text{exact}})} \cdot (n_{\text{bucket}} / 2 - k_{\text{exact}}) \right\rfloor & \text{otherwise}
\end{cases}
$$

其中 $k_{\text{exact}} = 8$（精确距离范围），$n_{\text{bucket}} = 32$（总桶数），$n_{\max} = 128$（最大裁剪距离）。

**设计直觉**：

- 近距离（$|i-j| \leq 8$）：每个距离一个独立的偏置（精确建模局部关系）
- 远距离（$|i-j| > 8$）：对数分桶，多个距离共享偏置（远处精度要求低）

**关键特性**：
- 偏置仅在**第一层**计算，然后**共享给所有层**
- 每个注意力头有独立的偏置参数
- 编码器使用双向偏置，解码器使用单向偏置

### 3.3 交叉注意力机制

交叉注意力是 Encoder-Decoder 架构的核心连接：

$$
\text{CrossAttn}(Y, H^{\text{enc}}) = \text{softmax}\left(\frac{Q_Y K_{H}^\top}{\sqrt{d_k}}\right) V_{H}
$$

其中：
- $Q_Y = Y W_Q$：查询来自**解码器**当前层
- $K_H = H^{\text{enc}} W_K$：键来自**编码器**输出
- $V_H = H^{\text{enc}} W_V$：值来自**编码器**输出

**维度变化**：

$$
Q_Y \in \mathbb{R}^{m \times d_k}, \quad K_H, V_H \in \mathbb{R}^{n \times d_k}
$$

注意力权重矩阵 $A \in \mathbb{R}^{m \times n}$（$m$=解码器长度，$n$=编码器长度）。

> **直觉**：交叉注意力让解码器的每个位置"查看"编码器的所有位置，决定生成当前 token 时应关注输入的哪些部分。

### 3.4 T5 各规模配置

| 参数 | Small | Base | Large | 3B | 11B |
|------|:-----:|:----:|:-----:|:--:|:---:|
| 层数 $L$ (enc+dec) | 6+6 | 12+12 | 24+24 | 24+24 | 24+24 |
| 隐藏维度 $d$ | 512 | 768 | 1024 | 1024 | 1024 |
| FFN 维度 $d_{ff}$ | 2048 | 3072 | 4096 | 16384 | 65536 |
| 注意力头数 $A$ | 8 | 12 | 16 | 32 | 128 |
| 每头维度 $d_k$ | 64 | 64 | 64 | 128 | 128 |
| 参数量 | 60M | 220M | 770M | 3B | 11B |

**参数量估算**（T5-Base）：

编码器单层（Self-Attention + FFN）：

$$
P_{\text{enc\_layer}} = \underbrace{4d^2}_{\text{SelfAttn}} + \underbrace{2d \cdot d_{ff}}_{\text{FFN}} + \underbrace{2d}_{\text{RMSNorm}} \approx 7.1\text{M}
$$

解码器单层（Self-Attention + Cross-Attention + FFN）：

$$
P_{\text{dec\_layer}} = \underbrace{4d^2}_{\text{SelfAttn}} + \underbrace{4d^2}_{\text{CrossAttn}} + \underbrace{2d \cdot d_{ff}}_{\text{FFN}} + \underbrace{3d}_{\text{RMSNorm}} \approx 9.5\text{M}
$$

总计：

$$
P_{\text{total}} = |V| \cdot d + 12 \times P_{\text{enc}} + 12 \times P_{\text{dec}} \approx 220\text{M}
$$

---

## 4. 预训练目标：Span Corruption

### 4.1 Span Corruption 数学定义

Span Corruption 是 T5 的核心预训练任务——随机遮盖输入中的**连续片段（span）**，让模型恢复被遮盖的内容。

**形式化定义**：

给定原始序列 $x = (x_1, x_2, \ldots, x_n)$：

1. 随机选择 $k$ 个不重叠的 span，总共覆盖约 15% 的 token
2. 每个 span 替换为唯一的哨兵 token $\langle s_i \rangle$
3. 目标序列由哨兵 + 对应原始 token 组成

**示例**：

```
原始:  Thank you for inviting me to your party last week
遮盖:  Thank you <s1> me to your party <s2> week
目标:  <s1> for inviting <s2> last <eos>
```

**数学表述**：

设 $\mathcal{M} = \{(s_i, e_i)\}_{i=1}^k$ 为被遮盖的 span 集合（$s_i$, $e_i$ 分别是起止位置），则：

$$
\text{input} = \text{replace\_spans}(x, \mathcal{M})
$$

$$
\text{target} = \bigoplus_{i=1}^{k} \left[\langle s_i \rangle \oplus x_{s_i:e_i}\right] \oplus \langle \text{eos} \rangle
$$

### 4.2 与 BERT MLM 和 GPT-2 LM 的对比

$$
\boxed{
\begin{aligned}
&\text{BERT MLM: } P(x_{\text{masked}} \mid x_{\text{unmasked}}) \quad &\text{（独立预测每个 [MASK]）} \\
&\text{GPT-2 LM: } \prod_{t=1}^{n} P(x_t \mid x_{<t}) \quad &\text{（逐个预测所有 token）} \\
&\text{T5 Span: } \prod_{t=1}^{m} P(y_t \mid y_{<t}, x_{\text{corrupted}}) \quad &\text{（自回归恢复被遮盖 span）}
\end{aligned}
}
$$

| 特性 | BERT MLM | GPT-2 LM | T5 Span Corruption |
|------|:--------:|:--------:|:-------------------:|
| 编码器 | ✅ 双向 | ❌ 无 | ✅ 双向 |
| 解码器 | ❌ 无 | ✅ 因果 | ✅ 因果 |
| 输入可见 | 全部（含 [MASK]） | 左侧上文 | 未遮盖部分 |
| 预测目标 | 仅 [MASK] 位置 | 所有位置 | 仅遮盖的 span |
| 计算效率 | 15% 有用梯度 | 100% 有用梯度 | 15% 有用梯度但目标更短 |
| 输出依赖 | 独立 | 自回归 | 自回归 |

**T5 Span Corruption 的优势**：

1. **编码器双向**：比 GPT-2 的单向更好地理解输入
2. **自回归解码**：比 BERT 的独立预测更好地建模 token 间依赖
3. **目标序列短**：只需生成被遮盖部分，训练效率更高

### 4.3 损失函数与梯度分析

**损失函数**（标准 seq2seq 交叉熵）：

$$
\boxed{\mathcal{L}(\theta) = -\frac{1}{m} \sum_{t=1}^{m} \log P_\theta(y_t \mid y_{<t}, x_{\text{corrupted}})}
$$

其中 $m$ 是目标序列长度，$y_t$ 是目标 token。

**梯度流分析**：

解码器侧（与 GPT-2 类似的因果约束）：

$$
\frac{\partial \mathcal{L}}{\partial h_t^{\text{dec}}} = -\frac{1}{m}\left(\mathbb{1}_{y_t} - P(\cdot \mid y_{<t}, x_{\text{corrupted}})\right) \cdot W_e
$$

通过交叉注意力流向编码器：

$$
\frac{\partial \mathcal{L}}{\partial H^{\text{enc}}} = \sum_{t=1}^{m} \frac{\partial \mathcal{L}}{\partial h_t^{\text{dec}}} \cdot \frac{\partial h_t^{\text{dec}}}{\partial H^{\text{enc}}}
$$

**关键观察**：编码器通过交叉注意力接收来自**所有**解码步骤的梯度，这意味着即使只遮盖了 15% 的 token，编码器的每个位置都会得到梯度更新。

---

## 5. 训练优化方法总结

### 5.1 Adafactor 优化器

T5 使用 **Adafactor** 而非 Adam，主要目的是节省内存：

**Adam 的内存问题**：

Adam 需要为每个参数存储一阶矩 $m$ 和二阶矩 $v$，内存开销为参数量的 3 倍。对于 T5-11B 来说：

$$
\text{Adam 内存} = 3 \times 11\text{B} \times 4\text{bytes} = 132\text{GB}
$$

**Adafactor 的核心思想**：

对二维参数矩阵 $W \in \mathbb{R}^{n \times m}$，Adafactor 用行因子 $r \in \mathbb{R}^n$ 和列因子 $c \in \mathbb{R}^m$ 的外积近似二阶矩：

$$
\boxed{\hat{v}_{ij} = \frac{r_i \cdot c_j}{\text{mean}(r)}}
$$

其中：

$$
r_i = \beta_2 r_i + (1 - \beta_2) \cdot \text{mean}_j(g_{ij}^2)
$$

$$
c_j = \beta_2 c_j + (1 - \beta_2) \cdot \text{mean}_i(g_{ij}^2)
$$

**内存节省**：

$$
\text{Adam: } O(n \times m) \quad \xrightarrow{\text{Adafactor}} \quad O(n + m)
$$

对于 $d \times d$ 的矩阵，内存从 $O(d^2)$ 降至 $O(d)$。

### 5.2 学习率调度与训练策略

**逆平方根调度**（T5 默认）：

$$
\boxed{\eta(t) = \frac{1}{\sqrt{\max(t, k)}}}
$$

其中 $k = 10{,}000$ 是预热步数。

**训练超参数**：

| 超参数 | 值 |
|--------|-----|
| 预训练步数 | $2^{19} \approx 524\text{K}$ |
| Batch Size | 128 序列 |
| 输入长度 | 512 tokens |
| 目标长度 | 114 tokens |
| 学习率 | $0.01$（Adafactor 默认） |
| Dropout | 0.1 |
| Span 遮盖率 | 15% |
| 平均 Span 长度 | 3 |

### 5.3 多任务混合训练

T5 在预训练后进行多任务微调，关键问题是如何平衡各任务的数据量。

**混合策略**：

设有 $K$ 个任务，任务 $k$ 有 $N_k$ 个样本。混合比例为：

$$
p_k = \min\left(\frac{N_k}{\sum_{j} N_j}, \; \frac{K_{\max}}{K}\right)
$$

其中 $K_{\max}$ 是人为设定的上限，防止大数据集任务（如翻译）完全主导训练。

T5 论文中使用了等比例混合 + 人工上限的策略，每个任务的采样数被限制在最多 $2^{16} = 65{,}536$ 个样本。

---

## 6. 从数学到代码：完整实现

### 6.1 NumPy 实现核心组件

```python
import numpy as np


def softmax(x, axis=-1):
    """数值稳定的 Softmax"""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def rms_norm(x, gamma, eps=1e-6):
    """
    RMSNorm（T5 使用，无偏置项）

    数学公式:
        RMSNorm(x) = x / RMS(x) * γ
        RMS(x) = √(mean(x²) + ε)
    """
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * gamma


def relative_position_bucket(rel_pos, bidirectional=True,
                              num_buckets=32, max_distance=128):
    """
    T5 相对位置分桶

    近距离 (|d| <= 8): 每个距离一个桶
    远距离 (|d| > 8): 对数分桶

    参数:
        rel_pos: (seq_q, seq_k) 相对位置矩阵 (i - j)
        bidirectional: 编码器用True，解码器用False
        num_buckets: 桶数量（默认32）
        max_distance: 最大裁剪距离

    返回:
        buckets: (seq_q, seq_k) 桶索引
    """
    ret = np.zeros_like(rel_pos, dtype=np.int32)

    if bidirectional:
        num_buckets //= 2
        # 正方向桶 = num_buckets + bucket, 负方向桶 = bucket
        ret += (rel_pos > 0).astype(np.int32) * num_buckets
        n = np.abs(rel_pos)
    else:
        # 单向：只看过去（rel_pos <= 0）
        n = np.maximum(-rel_pos, 0)

    # 精确桶 vs 对数桶
    max_exact = num_buckets // 2
    is_small = n < max_exact

    # 对数分桶
    val_if_large = max_exact + (
        np.log(n.astype(np.float32) / max_exact + 1e-6)
        / np.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).astype(np.int32)
    val_if_large = np.minimum(val_if_large, num_buckets - 1)

    ret += np.where(is_small, n, val_if_large)
    return ret


def relative_position_bias_numpy(seq_q, seq_k, embed_table,
                                  bidirectional=True):
    """
    计算相对位置偏置矩阵

    参数:
        seq_q: 查询序列长度
        seq_k: 键序列长度
        embed_table: (num_buckets, num_heads) 偏置嵌入
        bidirectional: 是否双向

    返回:
        bias: (1, num_heads, seq_q, seq_k)
    """
    # 构建相对位置矩阵
    context_pos = np.arange(seq_q)[:, None]  # (seq_q, 1)
    memory_pos = np.arange(seq_k)[None, :]   # (1, seq_k)
    rel_pos = context_pos - memory_pos        # (seq_q, seq_k)

    # 映射到桶
    buckets = relative_position_bucket(rel_pos, bidirectional)

    # 查表得到偏置
    bias = embed_table[buckets]  # (seq_q, seq_k, num_heads)
    bias = bias.transpose(2, 0, 1)  # (num_heads, seq_q, seq_k)
    return bias[None, :, :, :]      # (1, num_heads, seq_q, seq_k)


def cross_attention_numpy(Q, K_enc, V_enc):
    """
    交叉注意力 (NumPy)

    Q 来自解码器，K/V 来自编码器

    参数:
        Q: (batch, heads, seq_dec, d_k)
        K_enc: (batch, heads, seq_enc, d_k)
        V_enc: (batch, heads, seq_enc, d_k)

    返回:
        output: (batch, heads, seq_dec, d_k)
        weights: (batch, heads, seq_dec, seq_enc)
    """
    d_k = Q.shape[-1]
    scores = np.matmul(Q, K_enc.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    weights = softmax(scores, axis=-1)
    output = np.matmul(weights, V_enc)
    return output, weights


def span_corruption_numpy(tokens, mask_ratio=0.15, mean_span_len=3,
                           sentinel_start=32000):
    """
    T5 Span Corruption 预处理

    参数:
        tokens: 原始 token 序列 (1D array)
        mask_ratio: 遮盖比例（默认15%）
        mean_span_len: 平均 span 长度
        sentinel_start: 哨兵 token 起始 ID

    返回:
        corrupted_input: 遮盖后的输入
        target: 目标序列
    """
    n = len(tokens)
    num_mask = max(1, int(n * mask_ratio))

    # 确定 span 数量和位置
    num_spans = max(1, int(num_mask / mean_span_len))
    span_lengths = np.random.poisson(mean_span_len, num_spans)
    span_lengths = np.maximum(span_lengths, 1)

    # 随机选择 span 起始位置（不重叠）
    total_needed = span_lengths.sum()
    if total_needed > n:
        span_lengths = span_lengths[:max(1, n // mean_span_len)]
        total_needed = span_lengths.sum()

    # 简化版：均匀分布起始位置
    available = n - total_needed
    if available <= 0:
        span_lengths = np.array([min(num_mask, n)])
        available = n - span_lengths.sum()
        num_spans = 1

    gaps = np.sort(np.random.choice(
        available + num_spans, num_spans, replace=False
    ))
    starts = gaps - np.arange(num_spans)
    for i in range(1, len(starts)):
        starts[i] = max(starts[i], starts[i-1] + span_lengths[i-1])

    # 构建 corrupted input 和 target
    corrupted = []
    target = []
    prev_end = 0

    for i in range(len(starts)):
        s = starts[i]
        e = min(s + span_lengths[i], n)
        sentinel = sentinel_start + i

        # 添加未遮盖部分
        corrupted.extend(tokens[prev_end:s].tolist())
        corrupted.append(sentinel)

        # 目标：哨兵 + 原始 token
        target.append(sentinel)
        target.extend(tokens[s:e].tolist())

        prev_end = e

    # 添加剩余部分
    corrupted.extend(tokens[prev_end:].tolist())
    target.append(1)  # EOS token

    return np.array(corrupted), np.array(target)


# ========== 测试 NumPy 实现 ==========
if __name__ == "__main__":
    np.random.seed(42)

    # 测试 RMSNorm
    x = np.random.randn(2, 8, 64)
    gamma = np.ones(64)
    normed = rms_norm(x, gamma)
    print(f"RMSNorm 输出均方: {np.mean(normed**2, axis=-1)[0, 0]:.4f}")

    # 测试相对位置分桶
    seq_len = 12
    pos_q = np.arange(seq_len)[:, None]
    pos_k = np.arange(seq_len)[None, :]
    rel = pos_q - pos_k
    buckets = relative_position_bucket(rel, bidirectional=True)
    print(f"相对位置桶 (双向):\n{buckets[:6, :6]}")

    # 测试相对位置偏置
    num_heads = 4
    embed_table = np.random.randn(32, num_heads) * 0.01
    bias = relative_position_bias_numpy(seq_len, seq_len, embed_table)
    print(f"位置偏置形状: {bias.shape}")

    # 测试交叉注意力
    batch, heads, seq_dec, seq_enc, d_k = 2, 4, 6, 10, 16
    Q = np.random.randn(batch, heads, seq_dec, d_k)
    K = np.random.randn(batch, heads, seq_enc, d_k)
    V = np.random.randn(batch, heads, seq_enc, d_k)
    out, w = cross_attention_numpy(Q, K, V)
    print(f"交叉注意力输出: {out.shape}, 权重: {w.shape}")
    print(f"权重行和 = {w[0, 0, 0].sum():.4f}")

    # 测试 Span Corruption
    tokens = np.arange(100, 120)
    corrupted, target = span_corruption_numpy(tokens)
    print(f"\n原始序列长度: {len(tokens)}")
    print(f"遮盖后输入长度: {len(corrupted)}")
    print(f"目标序列长度: {len(target)}")
    print(f"遮盖后输入: {corrupted[:15]}...")
    print(f"目标序列: {target[:10]}...")
```

### 6.2 PyTorch 完整实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class T5RMSNorm(nn.Module):
    """
    T5 RMSNorm（无偏置项）

    数学公式:
        RMSNorm(x) = x / √(mean(x²) + ε) * γ
    """
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class T5RelativePositionBias(nn.Module):
    """
    T5 相对位置偏置

    将相对距离映射到桶，每个桶对应一个可学习偏置值。
    近距离精确，远距离对数分桶。
    """
    def __init__(self, num_heads: int, num_buckets: int = 32,
                 max_distance: int = 128, bidirectional: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.bidirectional = bidirectional
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

    @staticmethod
    def _relative_position_bucket(rel_pos: torch.Tensor,
                                   bidirectional: bool = True,
                                   num_buckets: int = 32,
                                   max_distance: int = 128) -> torch.Tensor:
        """相对位置 → 桶索引"""
        ret = torch.zeros_like(rel_pos, dtype=torch.long)

        if bidirectional:
            num_buckets //= 2
            ret += (rel_pos > 0).long() * num_buckets
            n = rel_pos.abs()
        else:
            n = (-rel_pos).clamp(min=0)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact + 1e-6)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).long()
        val_if_large = val_if_large.clamp(max=num_buckets - 1)

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, seq_q: int, seq_k: int,
                device: torch.device) -> torch.Tensor:
        """
        返回: (1, num_heads, seq_q, seq_k)
        """
        ctx = torch.arange(seq_q, device=device)[:, None]
        mem = torch.arange(seq_k, device=device)[None, :]
        rel_pos = ctx - mem

        buckets = self._relative_position_bucket(
            rel_pos, self.bidirectional, self.num_buckets, self.max_distance
        )
        bias = self.relative_attention_bias(buckets)  # (seq_q, seq_k, heads)
        return bias.permute(2, 0, 1).unsqueeze(0)     # (1, heads, seq_q, seq_k)


class T5Attention(nn.Module):
    """
    T5 注意力层（支持自注意力和交叉注意力）

    数学公式:
        Attention(Q, K, V) = softmax(QK^T/√d_k + B) V
    """
    def __init__(self, d_model: int, num_heads: int,
                 is_cross: bool = False, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.is_cross = is_cross

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                kv_input: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None,
                position_bias: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        参数:
            x: (B, T_q, d) 查询输入
            kv_input: (B, T_k, d) 键值输入（交叉注意力用）
            mask: (B, 1, T_q, T_k) 注意力掩码
            position_bias: (1, H, T_q, T_k) 相对位置偏置
        """
        B, T_q, _ = x.size()
        kv = kv_input if kv_input is not None else x
        T_k = kv.size(1)

        Q = self.q_proj(x).view(B, T_q, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(kv).view(B, T_k, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(kv).view(B, T_k, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if position_bias is not None:
            scores = scores + position_bias

        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        out = torch.matmul(weights, V)
        out = out.transpose(1, 2).contiguous().view(B, T_q, -1)
        out = self.o_proj(out)

        return out, weights


class T5FFN(nn.Module):
    """
    T5 前馈网络

    数学公式:
        FFN(x) = ReLU(xW_1) W_2
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.relu(self.w1(x))))


class T5EncoderBlock(nn.Module):
    """
    T5 编码器块 (Pre-Norm)

    数学公式:
        a = x + SelfAttn(RMSNorm(x))
        output = a + FFN(RMSNorm(a))
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = T5RMSNorm(d_model)
        self.self_attn = T5Attention(d_model, num_heads, dropout=dropout)
        self.norm2 = T5RMSNorm(d_model)
        self.ffn = T5FFN(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor,
                position_bias: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        normed = self.norm1(x)
        attn_out, attn_w = self.self_attn(normed, position_bias=position_bias)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, attn_w


class T5DecoderBlock(nn.Module):
    """
    T5 解码器块 (Pre-Norm)

    数学公式:
        a = y + CausalSelfAttn(RMSNorm(y))
        b = a + CrossAttn(RMSNorm(a), H_enc)
        output = b + FFN(RMSNorm(b))
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = T5RMSNorm(d_model)
        self.self_attn = T5Attention(d_model, num_heads, dropout=dropout)
        self.norm2 = T5RMSNorm(d_model)
        self.cross_attn = T5Attention(d_model, num_heads, is_cross=True,
                                       dropout=dropout)
        self.norm3 = T5RMSNorm(d_model)
        self.ffn = T5FFN(d_model, d_ff, dropout)

    def forward(self, y: torch.Tensor, enc_output: torch.Tensor,
                causal_mask: Optional[torch.Tensor] = None,
                self_pos_bias: Optional[torch.Tensor] = None,
                cross_pos_bias: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normed = self.norm1(y)
        sa_out, sa_w = self.self_attn(
            normed, mask=causal_mask, position_bias=self_pos_bias
        )
        y = y + sa_out

        normed = self.norm2(y)
        ca_out, ca_w = self.cross_attn(
            normed, kv_input=enc_output, position_bias=cross_pos_bias
        )
        y = y + ca_out

        y = y + self.ffn(self.norm3(y))
        return y, sa_w, ca_w


class T5Model(nn.Module):
    """
    完整 T5 模型

    结构:
        Encoder: [Embedding] → [EncoderBlock x L] → [RMSNorm]
        Decoder: [Embedding] → [DecoderBlock x L] → [RMSNorm] → [LM Head]
    """
    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model

        # 共享嵌入
        self.shared_embedding = nn.Embedding(vocab_size, d_model)

        # 编码器
        self.encoder_blocks = nn.ModuleList([
            T5EncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.encoder_norm = T5RMSNorm(d_model)

        # 解码器
        self.decoder_blocks = nn.ModuleList([
            T5DecoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.decoder_norm = T5RMSNorm(d_model)

        # 相对位置偏置（仅第一层计算，共享给所有层）
        self.enc_pos_bias = T5RelativePositionBias(
            num_heads, bidirectional=True
        )
        self.dec_pos_bias = T5RelativePositionBias(
            num_heads, bidirectional=False
        )

        # LM Head（与嵌入共享权重）
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.shared_embedding.weight

        self._init_weights()

    def _init_weights(self):
        """T5 权重初始化: N(0, factor) where factor = 1.0"""
        factor = 1.0
        d = self.d_model
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0,
                                std=factor * (d ** -0.5))
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=1.0)
            elif isinstance(module, T5RMSNorm):
                nn.init.ones_(module.weight)

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        编码器前向传播

        参数:
            input_ids: (B, T_enc) 输入 token 索引
        返回:
            enc_output: (B, T_enc, d_model)
        """
        x = self.shared_embedding(input_ids)
        pos_bias = self.enc_pos_bias(
            x.size(1), x.size(1), x.device
        )

        for block in self.encoder_blocks:
            x, _ = block(x, position_bias=pos_bias)

        return self.encoder_norm(x)

    def decode(self, decoder_ids: torch.Tensor,
               enc_output: torch.Tensor) -> torch.Tensor:
        """
        解码器前向传播

        参数:
            decoder_ids: (B, T_dec) 解码器输入 token
            enc_output: (B, T_enc, d_model) 编码器输出
        返回:
            logits: (B, T_dec, vocab_size)
        """
        y = self.shared_embedding(decoder_ids)
        T_dec = y.size(1)

        # 因果掩码
        causal_mask = torch.triu(
            torch.ones(T_dec, T_dec, device=y.device), diagonal=1
        ).bool().unsqueeze(0).unsqueeze(0)

        # 位置偏置
        self_pos_bias = self.dec_pos_bias(T_dec, T_dec, y.device)

        for block in self.decoder_blocks:
            y, _, _ = block(y, enc_output,
                            causal_mask=causal_mask,
                            self_pos_bias=self_pos_bias)

        y = self.decoder_norm(y)
        logits = self.lm_head(y)
        return logits

    def forward(self, input_ids: torch.Tensor,
                decoder_ids: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> dict:
        """
        完整前向传播

        参数:
            input_ids: (B, T_enc) 编码器输入
            decoder_ids: (B, T_dec) 解码器输入
            labels: (B, T_dec) 目标 token（训练用）
        返回:
            logits, loss
        """
        enc_output = self.encode(input_ids)
        logits = self.decode(decoder_ids, enc_output)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )

        return {"logits": logits, "loss": loss}

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor,
                 max_new_tokens: int = 50,
                 eos_token_id: int = 1) -> torch.Tensor:
        """
        自回归生成（贪心解码）

        参数:
            input_ids: (B, T_enc) 编码器输入
            max_new_tokens: 最大生成长度
            eos_token_id: 结束符 ID
        """
        enc_output = self.encode(input_ids)
        batch_size = input_ids.size(0)

        # 以 decoder_start_token (pad=0) 开始
        decoder_ids = torch.zeros(batch_size, 1, dtype=torch.long,
                                   device=input_ids.device)

        for _ in range(max_new_tokens):
            logits = self.decode(decoder_ids, enc_output)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            decoder_ids = torch.cat([decoder_ids, next_token], dim=1)

            # 检查是否所有序列都生成了 EOS
            if (next_token == eos_token_id).all():
                break

        return decoder_ids


# ========== 完整测试 ==========
if __name__ == "__main__":
    # 缩小版超参数（T5-Tiny）
    vocab_size, d_model, num_heads = 1000, 64, 4
    num_layers, d_ff, max_len = 2, 256, 32
    batch_size = 4

    # 1. 创建模型
    model = T5Model(vocab_size, d_model, num_heads, num_layers,
                    d_ff, max_len)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {total_params:,}")

    # 2. 准备输入
    enc_len, dec_len = 20, 10
    input_ids = torch.randint(0, vocab_size, (batch_size, enc_len))
    decoder_ids = torch.randint(0, vocab_size, (batch_size, dec_len))
    labels = torch.randint(0, vocab_size, (batch_size, dec_len))

    # 3. 前向传播
    model.eval()
    with torch.no_grad():
        out = model(input_ids, decoder_ids, labels)
    print(f"Logits 形状: {out['logits'].shape}")
    print(f"Loss: {out['loss'].item():.4f}")

    # 4. 编码器输出验证
    with torch.no_grad():
        enc_out = model.encode(input_ids)
    print(f"编码器输出形状: {enc_out.shape}")

    # 5. 训练一步
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    out = model(input_ids, decoder_ids, labels)
    loss_before = out["loss"].item()
    out["loss"].backward()
    opt.step()
    opt.zero_grad()

    out2 = model(input_ids, decoder_ids, labels)
    loss_after = out2["loss"].item()
    print(f"\n训练前 Loss: {loss_before:.4f}")
    print(f"训练后 Loss: {loss_after:.4f}")
    print(f"Loss 下降: {loss_before > loss_after}")

    # 6. 生成测试
    model.eval()
    prompt = torch.randint(0, vocab_size, (1, 15))
    generated = model.generate(prompt, max_new_tokens=10)
    print(f"\n生成序列长度: {generated.shape[1]}")
    print(f"生成的 token IDs: {generated[0].tolist()}")

    # 7. 权重绑定验证
    emb_w = model.shared_embedding.weight
    head_w = model.lm_head.weight
    print(f"\n权重绑定验证: {torch.equal(emb_w, head_w)}")

    print("\n✅ T5 模型测试通过！")
```

---

## 7. T5 的系统性消融实验

T5 论文的一个重大贡献是在统一框架下进行了**大量系统性消融实验**，为后续研究提供了宝贵的参考。

### 7.1 架构对比

T5 论文对比了三种主要架构：

| 架构 | 编码器 | 解码器 | 代表模型 |
|------|:------:|:------:|---------|
| Encoder-only | ✅ 双向 | ❌ | BERT |
| Decoder-only | ❌ | ✅ 因果 | GPT-2 |
| Encoder-Decoder | ✅ 双向 | ✅ 因果 | T5 |

**实验结果**（在 GLUE/SuperGLUE 上的平均分数）：

$$
\boxed{\text{Encoder-Decoder} > \text{Decoder-only (with prefix)} > \text{Encoder-only}}
$$

**关键发现**：

1. **Encoder-Decoder 胜出**：编码器的双向注意力对理解任务至关重要
2. **参数公平对比**：Encoder-Decoder 的参数量约为 Decoder-only 的 2 倍（因为有两个 stack），但在相同 FLOPS 下仍然更优
3. **Decoder-only 的 prefix 变体**：让部分输入使用双向注意力，效果介于两者之间

### 7.2 预训练目标对比

T5 对比了多种预训练目标：

| 预训练目标 | 描述 | GLUE 平均 |
|-----------|------|:---------:|
| 语言模型 (LM) | GPT 风格，预测下一个 token | 基准 |
| BERT-style MLM | 独立预测 [MASK] | +1.2 |
| Deshuffling | 还原打乱顺序的句子 | -0.5 |
| **Span Corruption** | **恢复被遮盖的连续 span** | **+1.5** |
| Random Spans | 不同 span 长度 | +1.3 |

**Span 长度的影响**：

| 平均 Span 长度 | 遮盖率 | 效果 |
|:-------------:|:------:|:----:|
| 1（等同 BERT） | 15% | 基准 |
| 2 | 15% | +0.3 |
| **3** | **15%** | **最优** |
| 5 | 15% | -0.1 |
| 10 | 15% | -0.5 |

### 7.3 关键实验结论

T5 论文通过消融实验得出的核心结论：

$$
\boxed{
\begin{aligned}
&\text{1. Encoder-Decoder 架构 > Decoder-only} \\
&\text{2. Span Corruption > MLM > LM} \\
&\text{3. 平均 span 长度 3, 遮盖率 15\% 最优} \\
&\text{4. 更大模型 + 更多数据 = 更好效果} \\
&\text{5. 多任务预训练 + 微调效果最佳}
\end{aligned}
}
$$

**Scaling 实验**：

| 模型规模 | GLUE | SuperGLUE | SQuAD |
|---------|:----:|:---------:|:-----:|
| T5-Small (60M) | 77.4 | 64.0 | 79.2 |
| T5-Base (220M) | 82.7 | 73.5 | 85.4 |
| T5-Large (770M) | 86.4 | 79.2 | 88.1 |
| T5-3B | 88.5 | 84.7 | 90.2 |
| T5-11B | **90.3** | **88.9** | **91.3** |

模型规模每增加约 4 倍，性能稳定提升——这为后来的 Scaling Laws 研究提供了重要证据。

---

## 8. 与其他模型的关系

### 8.1 BERT vs GPT-2 vs T5

| 维度 | BERT | GPT-2 | T5 |
|------|:----:|:-----:|:---:|
| **架构** | Encoder-only | Decoder-only | Encoder-Decoder |
| **注意力** | 全双向 | 因果（单向） | 编码器双向 + 解码器因果 |
| **位置编码** | 可学习绝对 | 可学习绝对 | 相对位置偏置 |
| **预训练** | MLM + NSP | 自回归 LM | Span Corruption |
| **微调** | 任务特定头 | Zero-shot | Text-to-Text |
| **输出形式** | 向量 → 分类/标注 | Token 序列 | Token 序列 |
| **统一性** | 低（不同任务不同头） | 中（Zero-shot 但效果弱） | 高（统一 Text-to-Text） |

**T5 的统一优势**：

$$
\underbrace{\text{BERT}}_{\text{理解}} + \underbrace{\text{GPT-2}}_{\text{生成}} \longrightarrow \underbrace{\text{T5}}_{\text{理解 + 生成，统一框架}}
$$

T5 证明了一个模型可以同时擅长理解和生成任务，关键在于 Encoder-Decoder 架构和 Text-to-Text 统一格式。

### 8.2 T5 的后续影响

```
T5 (2019) ── 统一 Text-to-Text 框架
  ├── mT5 (2020) ── 多语言版本（101 种语言）
  ├── T5.1.1 (2020) ── 架构改进（GEGLU, 无 Dropout）
  ├── ExT5 (2021) ── 扩展预训练任务
  ├── Flan-T5 (2022) ── 指令微调
  ├── UL2 (2022) ── 统一语言学习器
  └── 影响后续模型:
       ├── GPT-3 ── In-context Learning（文本化思想）
       ├── BART ── 另一种 Seq2Seq 预训练
       └── PaLM ── Encoder-Decoder → Decoder-only 规模化
```

**T5 在历史中的定位**：

T5 的贡献不仅是一个模型，更是一个**研究方法论**：

1. **统一框架**：提出了 Text-to-Text 范式，统一了 NLP 任务的输入输出
2. **系统性消融**：在统一框架下公平对比了数十种设计选择
3. **C4 数据集**：提供了高质量、可复现的预训练数据集
4. **Scaling 证据**：提供了模型规模与性能关系的系统性证据

---

## 扩展阅读与实现

### 问题 1：为什么 Encoder-Decoder 比 Decoder-only 更适合理解任务？

**解答**：

Decoder-only 模型使用因果掩码，位置 $t$ 只能看到 $x_1, \ldots, x_t$：

$$
h_t^{\text{dec}} = f(x_1, \ldots, x_t) \quad \text{（单向上下文）}
$$

Encoder-Decoder 中的编码器使用全连接注意力：

$$
h_t^{\text{enc}} = f(x_1, \ldots, x_n) \quad \text{（双向上下文）}
$$

对于理解任务（如分类、NLI），双向上下文至关重要。例如在 "The movie was not [MASK] good" 中，理解 "not" 修饰 "good" 需要同时看到两边。

编码器的双向注意力可以充分利用输入的全部信息，而解码器的因果约束则确保输出的自回归生成是合理的。

### 问题 2：T5 的相对位置编码 vs Transformer 的正弦位置编码

**解答**：

**正弦位置编码**（原始 Transformer）：

$$
PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d}), \quad PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})
$$

**问题**：位置信息被加到输入嵌入中，经过多层变换后可能被稀释。

**T5 相对位置偏置**：

$$
\text{score}_{ij} = q_i^\top k_j / \sqrt{d_k} + B_{ij}
$$

偏置直接加在注意力分数上，不会被后续变换稀释。

**关键对比**：

| 特性 | 正弦编码 | T5 相对偏置 |
|------|:-------:|:---------:|
| 编码方式 | 加到嵌入 | 加到注意力分数 |
| 信息保留 | 可能被稀释 | 直接影响注意力 |
| 外推性 | 理论可外推 | 受最大距离限制 |
| 参数量 | 无参数 | 少量可学习参数 |

### 问题 3：Span Corruption 的 15% 遮盖率从何而来？

**解答**：

15% 的遮盖率源自 BERT，但 T5 通过实验验证了这个选择：

| 遮盖率 | GLUE 平均 | 目标序列长度 |
|:------:|:---------:|:----------:|
| 10% | 82.1 | 较短 |
| **15%** | **82.7** | **中等** |
| 25% | 82.4 | 较长 |
| 50% | 81.5 | 很长 |

**15% 是效果和效率的平衡点**：
- 太低：模型学到的信号太少
- 太高：任务太难，且目标序列变长导致训练变慢
- 15% 时目标序列约为输入的 15%（加上哨兵 token），计算开销适中

### 问题 4：T5 如何处理回归任务（如 STS-B）？

**解答**：

STS-B 要求预测两个句子的相似度分数（0.0-5.0）。T5 的处理方式：

1. 将分数四舍五入到最近的 0.2：$\text{round}(3.73, 0.2) = 3.8$
2. 转为字符串："3.8"
3. 模型输出文本 "3.8"，解码时转回浮点数

**数学上**：这将连续回归问题离散化为 26 个类别 $\{0.0, 0.2, 0.4, \ldots, 5.0\}$，用分类的交叉熵损失替代回归的 MSE 损失。

$$
P(\text{score} = 3.8 \mid x) = P(\text{"3"}) \cdot P(\text{"."} \mid \text{"3"}) \cdot P(\text{"8"} \mid \text{"3."})
$$

实验表明，这种文本化方法在 STS-B 上的效果与专门的回归头相当。

### 问题 5：T5.1.1 的 GEGLU 激活函数是什么？

**解答**：

T5.1.1 将 FFN 中的 ReLU 替换为 **GEGLU**（Gated GELU）：

$$
\text{GEGLU}(x, W_1, V, W_2) = \left(\text{GELU}(xW_1) \odot xV\right) W_2
$$

其中 $\odot$ 是逐元素乘法，$V \in \mathbb{R}^{d \times d_{ff}}$ 是额外的门控矩阵。

**与原始 FFN 的对比**：

$$
\text{FFN}_{\text{ReLU}}(x) = \text{ReLU}(xW_1)W_2
$$

$$
\text{FFN}_{\text{GEGLU}}(x) = \left(\text{GELU}(xW_1) \odot xV\right) W_2
$$

GEGLU 引入了门控机制，让模型可以选择性地传递信息。为保持参数量不变，$d_{ff}$ 从 $4d$ 缩小为 $\frac{8d}{3}$（因为多了矩阵 $V$）。

---

## 参考资源

### 经典论文

1. Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., ... & Liu, P. J. (2020). [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683). JMLR.
   - **贡献**：提出 T5 和 Text-to-Text 统一框架，系统性消融实验

2. Shazeer, N. (2020). [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202). arXiv.
   - **贡献**：提出 GEGLU 等门控激活函数变体，被 T5.1.1 采用

3. Zhang, B., & Sennrich, R. (2019). [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467). NeurIPS 2019.
   - **贡献**：提出 RMSNorm，被 T5 采用的简化归一化方法

4. Shazeer, N., & Stern, M. (2018). [Adafactor: Adaptive Learning Rates with Sublinear Memory Cost](https://arxiv.org/abs/1804.04235). ICML 2018.
   - **贡献**：提出 Adafactor 优化器，T5 训练的核心组件

### 教材与书籍

5. Jurafsky, D., & Martin, J. H. [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/). 3rd ed. (Draft).
   - **章节**：第 11 章讲解序列到序列模型与 Encoder-Decoder 架构

### 在线资源与教程

6. Google Research Blog. [Exploring Transfer Learning with T5](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html).
   - **内容**：T5 论文的官方博客解读

7. Hugging Face. [T5 Documentation](https://huggingface.co/docs/transformers/model_doc/t5).
   - **内容**：T5 的工业级实现和使用指南

8. Roberts, A. [T5 on GitHub](https://github.com/google-research/text-to-text-transfer-transformer).
   - **内容**：T5 官方代码库（基于 Mesh TensorFlow）

---

## 附录：符号表

| 符号 | 含义 | 维度/类型 |
|------|------|----------|
| $n$ | 编码器输入序列长度 | 标量 |
| $m$ | 解码器目标序列长度 | 标量 |
| $d$ | 隐藏维度（$d_{\text{model}}$） | 标量，T5-Base: 768 |
| $d_k$ | 每个注意力头的维度 | 标量，64 |
| $d_{ff}$ | FFN 隐藏层维度 | 标量，T5-Base: 3072 |
| $L$ | 编码器/解码器层数 | 标量，T5-Base: 12 |
| $A$ | 注意力头数 | 标量，T5-Base: 12 |
| $\|V\|$ | SentencePiece 词表大小 | 标量，32,000 |
| $x$ | 编码器输入 token 序列 | $(n,)$ |
| $y$ | 解码器目标 token 序列 | $(m,)$ |
| $H^{\text{enc}}$ | 编码器最终输出 | $(n, d)$ |
| $h_t^{\text{dec}}$ | 解码器第 $t$ 位置隐藏状态 | $(d,)$ |
| $B$ | 相对位置偏置矩阵 | $(n, n)$ 或 $(m, m)$ |
| $b(i,j)$ | 位置 $i, j$ 的偏置桶索引 | 标量，整数 |
| $\langle s_i \rangle$ | 第 $i$ 个哨兵 token | 特殊 token |
| $Q, K, V$ | 查询、键、值矩阵 | 注意力中间量 |
| $\mathcal{L}$ | Seq2Seq 交叉熵损失 | 标量 |
| $\ell(\cdot, \cdot)$ | 交叉熵损失函数 | 函数 |
| $r_i, c_j$ | Adafactor 行/列因子 | 向量 |
| $\gamma$ | RMSNorm 缩放参数 | $(d,)$ |

**典型维度示例（T5-Base）：**
- $d = 768$（隐藏维度）
- $d_k = 64$（每头维度）
- $d_{ff} = 3072$（FFN 维度）
- $|V| = 32{,}000$（词表大小）
- $L = 12$（编码器/解码器各 12 层）

---

最后更新：2026-03-19
