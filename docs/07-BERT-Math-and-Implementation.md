# BERT 数学原理与实现 —— 双向预训练的完整推导

> **前置知识**：Transformer 编码器、Attention 机制、交叉熵损失、Python 基础  
> **与前面内容的联系**：建议先学习 [Transformer-Math-and-Implementation](./06-Transformer-Math-and-Implementation.md)，理解自注意力和编码器架构  
> **与后续内容的联系**：BERT 是预训练+微调范式的开创者，GPT-2 采用相反的单向自回归路线

---

## 目录

1. [引言：为什么需要双向预训练？](#1-引言为什么需要双向预训练)
   - 1.1 [从特征提取到预训练：NLP 范式的演进](#11-从特征提取到预训练nlp-范式的演进)
   - 1.2 [单向语言模型的局限性](#12-单向语言模型的局限性)
   - 1.3 [本科数学知识映射表](#13-本科数学知识映射表)
2. [核心思想：双向编码与掩码语言模型](#2-核心思想双向编码与掩码语言模型)
   - 2.1 [双向上下文表示](#21-双向上下文表示)
   - 2.2 [MLM：完形填空式预训练](#22-mlm完形填空式预训练)
   - 2.3 [NSP：句对关系预训练](#23-nsp句对关系预训练)
3. [BERT 架构的数学描述](#3-bert-架构的数学描述)
   - 3.1 [输入表示：三种嵌入的叠加](#31-输入表示三种嵌入的叠加)
   - 3.2 [双向 Attention 掩码](#32-双向-attention-掩码)
   - 3.3 [Transformer 编码器层](#33-transformer-编码器层)
   - 3.4 [BERT-Base 与 BERT-Large 配置](#34-bert-base-与-bert-large-配置)
4. [预训练目标的数学推导](#4-预训练目标的数学推导)
   - 4.1 [MLM 损失函数](#41-mlm-损失函数)
   - 4.2 [NSP 损失函数](#42-nsp-损失函数)
   - 4.3 [联合损失与梯度分析](#43-联合损失与梯度分析)
5. [训练优化方法总结](#5-训练优化方法总结)
   - 5.1 [预训练策略](#51-预训练策略)
   - 5.2 [微调策略](#52-微调策略)
   - 5.3 [优化器与学习率调度](#53-优化器与学习率调度)
6. [从数学到代码：完整实现](#6-从数学到代码完整实现)
   - 6.1 [NumPy 实现核心组件](#61-numpy-实现核心组件)
   - 6.2 [PyTorch 完整实现](#62-pytorch-完整实现)
7. [实践技巧与可视化](#7-实践技巧与可视化)
   - 7.1 [注意力可视化](#71-注意力可视化)
   - 7.2 [MLM 掩码策略分析](#72-mlm-掩码策略分析)
   - 7.3 [微调实战技巧](#73-微调实战技巧)
8. [与其他模型的关系](#8-与其他模型的关系)
   - 8.1 [BERT vs GPT：双向 vs 单向](#81-bert-vs-gpt双向-vs-单向)
   - 8.2 [BERT 的后续发展](#82-bert-的后续发展)
   - 8.3 [预训练范式总结](#83-预训练范式总结)

[扩展阅读与实现](#扩展阅读与实现)

[参考资源](#参考资源)

附录：[符号表](#附录符号表)

---

## 1. 引言：为什么需要双向预训练？

### 1.1 从特征提取到预训练：NLP 范式的演进

在 BERT 出现之前，NLP 经历了三个范式阶段：

**范式 1：特征工程时代**（2000s 之前）

手工设计特征（TF-IDF、n-gram 等），然后用传统分类器：

$$
y = f_{\theta}(\phi(x))
$$

其中 $\phi(x)$ 是手工特征，$f_{\theta}$ 是分类器（SVM、逻辑回归等）。

**范式 2：词向量时代**（2013-2017）

使用 [Word2Vec](./03-Word2Vec-Math-and-Implementation.md) 或 [GloVe](./04-GloVe-Math-and-Implementation.md) 学习静态词向量：

$$
\text{embedding}(w) = E[w] \in \mathbb{R}^d
$$

**核心问题**：词向量是**静态**的。同一个词在不同上下文中具有相同的表示：

| 句子 | "bank" 的含义 | 词向量 |
|------|-------------|--------|
| "I went to the **bank** to deposit money" | 银行 | 相同 |
| "The river **bank** was covered with flowers" | 河岸 | 相同 |

**范式 3：上下文表示时代**（2018—）

ELMo（2018）首先提出用**双向 LSTM** 生成上下文相关的词表示：

$$
h_t^{\text{ELMo}} = \gamma \sum_{j=0}^L s_j \cdot h_{t,j}
$$

但 ELMo 有一个关键限制：前向和后向 LSTM 是**独立训练**的，然后简单拼接：

$$
h_t^{\text{ELMo}} = [\overrightarrow{h}_t; \overleftarrow{h}_t]
$$

这意味着前向 LSTM 看不到右边的上下文，后向 LSTM 看不到左边的上下文。两者虽然拼接在一起，但**从未在同一层中交互**。

**BERT 的革命**：

> **在同一个注意力层中，让每个词同时关注左右两边的所有词**

$$
\boxed{h_t^{\text{BERT}} = \text{TransformerEncoder}(x_1, x_2, \ldots, x_n)_t}
$$

每个位置 $t$ 的输出 $h_t$ 都融合了**整个序列**的信息。

### 1.2 单向语言模型的局限性

传统语言模型（包括 GPT）采用**单向**（从左到右）建模：

$$
P(x_1, x_2, \ldots, x_n) = \prod_{t=1}^n P(x_t \mid x_1, \ldots, x_{t-1})
$$

**问题**：预测 $x_t$ 时只能利用左侧上下文 $x_{<t}$。

**示例**：

> "I went to the [MASK] to deposit money."

- **单向模型**：只能根据 "I went to the" 猜测，信息不足
- **双向模型**：同时利用 "I went to the" 和 "to deposit money"，能更准确地推断是 "bank"

**数学上的区别**：

| 模型类型 | 条件分布 | 信息利用 |
|---------|---------|---------|
| 单向（GPT） | $P(x_t \mid x_{<t})$ | 仅左侧上下文 |
| 双向独立（ELMo） | $[\overrightarrow{P}(x_t \mid x_{<t}); \overleftarrow{P}(x_t \mid x_{>t})]$ | 左右独立 |
| 真正双向（BERT） | $P(x_t \mid x_{\backslash t})$ | 左右同时交互 |

其中 $x_{\backslash t}$ 表示除位置 $t$ 以外的所有 token。

### 1.3 本科数学知识映射表

| 数学概念 | BERT 中的应用 | 代码对应 |
|---------|-------------|---------|
| 交叉熵 $H(p, q)$ | MLM/NSP 损失函数 | `F.cross_entropy()` |
| Softmax 函数 | 注意力权重 + 分类头 | `F.softmax()` |
| 矩阵乘法 $AB$ | 自注意力计算 | `torch.matmul()` |
| 向量加法 | 三种嵌入叠加 | `token_emb + seg_emb + pos_emb` |
| 概率论（条件概率） | 掩码语言模型目标 | MLM 概率预测 |
| 二分类（Sigmoid） | 句对预测 | `torch.sigmoid()` |
| 随机采样 | 掩码策略（80/10/10） | `torch.bernoulli()` |

---

## 2. 核心思想：双向编码与掩码语言模型

### 2.1 双向上下文表示

BERT 的核心洞察：**不需要修改 Transformer 架构本身**，只需要改变**训练目标**。

标准 Transformer 编码器本身就是双向的：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

在编码器中，注意力矩阵 $A \in \mathbb{R}^{n \times n}$ 的每个元素 $A_{ij}$ 都是非零的（没有因果掩码），这意味着：

$$
A_{ij} = \frac{\exp(q_i^\top k_j / \sqrt{d_k})}{\sum_{m=1}^n \exp(q_i^\top k_m / \sqrt{d_k})} > 0, \quad \forall i, j
$$

> **关键区别**：
> - **GPT（解码器）**：使用因果掩码，$A_{ij} = 0$ 当 $j > i$（不能看到未来）
> - **BERT（编码器）**：不使用因果掩码，每个位置都能看到所有位置

$$
\boxed{
M^{\text{BERT}}_{ij} = 1 \quad \forall i, j \in \{1, \ldots, n\}
}
$$

$$
M^{\text{GPT}}_{ij} = \begin{cases} 1 & \text{if } j \leq i \\ 0 & \text{otherwise} \end{cases}
$$

### 2.2 MLM：完形填空式预训练

**问题**：既然编码器是双向的，就不能用传统语言模型目标（会导致信息泄露——要预测的词已经能看到自己）。

**BERT 的解决方案**：**Masked Language Model (MLM)**

随机选择输入序列中 15% 的 token 进行掩码，然后预测被掩码的 token。

**掩码策略**（80/10/10 规则）：

对于被选中的 token $x_t$：

| 操作 | 概率 | 替换为 | 目的 |
|------|------|--------|------|
| 替换为 `[MASK]` | 80% | `[MASK]` 特殊标记 | 主要学习信号 |
| 替换为随机词 | 10% | 词表中随机 token | 增强鲁棒性 |
| 保持不变 | 10% | 原始 token | 对齐预训练与微调 |

**数学表述**：

设 $\mathcal{M}$ 为被掩码位置的集合（约 15% 的位置），则 MLM 目标为：

$$
\mathcal{L}_{\text{MLM}} = -\sum_{t \in \mathcal{M}} \log P(x_t \mid \tilde{x})
$$

其中 $\tilde{x}$ 是经过掩码处理后的输入序列。

> **Q:** 为什么不 100% 使用 `[MASK]` 替换？
>
> **A:** 因为微调时输入中没有 `[MASK]` 标记。如果预训练时总是用 `[MASK]`，模型会学到"只在看到 `[MASK]` 时才需要预测"，导致预训练和微调之间的**分布不匹配（mismatch）**。10% 保持不变和 10% 随机替换缓解了这个问题。

### 2.3 NSP：句对关系预训练

**动机**：许多下游任务需要理解**句子之间的关系**（如问答、自然语言推理）。

**Next Sentence Prediction (NSP)**：

给定两个句子 $A$ 和 $B$，判断 $B$ 是否是 $A$ 在原文中的下一句。

**数据构造**：

| 类型 | 标签 | 比例 | 示例 |
|------|------|------|------|
| 正样本 | IsNext | 50% | A = "我喜欢猫"，B = "它们很可爱" |
| 负样本 | NotNext | 50% | A = "我喜欢猫"，B = "今天天气真好" |

**输入格式**：

$$
\text{Input} = [\texttt{[CLS]}] \; A_1 \; A_2 \; \ldots \; A_m \; [\texttt{[SEP]}] \; B_1 \; B_2 \; \ldots \; B_k \; [\texttt{[SEP]}]
$$

其中 `[CLS]` 是分类标记（其最终隐藏状态用于 NSP 分类），`[SEP]` 是分隔标记。

**NSP 目标**：

$$
\mathcal{L}_{\text{NSP}} = -\left[ y \log P(\text{IsNext}) + (1 - y) \log P(\text{NotNext}) \right]
$$

其中 $y \in \{0, 1\}$ 是真实标签，预测概率基于 `[CLS]` 位置的输出：

$$
P(\text{IsNext}) = \text{softmax}(W_{\text{NSP}} \cdot h_{\texttt{[CLS]}} + b_{\text{NSP}})
$$

> **注意**：后续研究（如 RoBERTa）发现 NSP 对下游任务的帮助有限，甚至可能有害。但作为 BERT 原始设计的一部分，理解 NSP 仍然很重要。

---

## 3. BERT 架构的数学描述

### 3.1 输入表示：三种嵌入的叠加

BERT 的输入表示由**三种嵌入向量相加**得到：

$$
\boxed{E_{\text{input}}(t) = E_{\text{token}}(x_t) + E_{\text{segment}}(s_t) + E_{\text{position}}(t)}
$$

**1. Token 嵌入** $E_{\text{token}} \in \mathbb{R}^{|V| \times d}$

将 WordPiece token 映射为 $d$ 维向量：

$$
E_{\text{token}}(x_t) = E_{\text{token}}[x_t] \in \mathbb{R}^d
$$

其中 $|V|$ 是 WordPiece 词表大小（BERT 使用 30,522）。

**2. 段落嵌入** $E_{\text{segment}} \in \mathbb{R}^{2 \times d}$

区分句子 A 和句子 B：

$$
s_t = \begin{cases} 0 & \text{if token } t \in \text{Sentence A} \\ 1 & \text{if token } t \in \text{Sentence B} \end{cases}
$$

$$
E_{\text{segment}}(s_t) = E_{\text{segment}}[s_t] \in \mathbb{R}^d
$$

**3. 位置嵌入** $E_{\text{position}} \in \mathbb{R}^{L_{\max} \times d}$

与原始 Transformer 使用正弦函数不同，BERT 使用**可学习的位置嵌入**：

$$
E_{\text{position}}(t) = E_{\text{position}}[t] \in \mathbb{R}^d
$$

其中 $L_{\max} = 512$ 是最大序列长度。

**完整输入示例**：

```
输入:    [CLS]  I    love  cats  [SEP]  They  are  cute  [SEP]
Token:   101    1045 2293  8870  102    2027  2024  10140 102
Segment: 0      0    0     0     0      1     1     1     1
Position:0      1    2     3     4      5     6     7     8
```

### 3.2 双向 Attention 掩码

BERT 编码器使用的注意力掩码非常简单——**全 1 矩阵**（忽略 padding）：

$$
M_{ij} = \begin{cases}
1 & \text{if position } j \text{ is not padding} \\
0 & \text{if position } j \text{ is padding}
\end{cases}
$$

**与 GPT 解码器掩码的对比**：

GPT 使用下三角因果掩码：
$$
M^{\text{causal}}_{ij} = \begin{cases}
1 & \text{if } j \leq i \\
0 & \text{if } j > i
\end{cases}
$$

**可视化对比**（$n = 5$，无 padding）：

```
BERT 掩码（全连接）:         GPT 掩码（因果/下三角）:
1 1 1 1 1                   1 0 0 0 0
1 1 1 1 1                   1 1 0 0 0
1 1 1 1 1                   1 1 1 0 0
1 1 1 1 1                   1 1 1 1 0
1 1 1 1 1                   1 1 1 1 1
```

注意力分数在掩码为 0 的位置被设为 $-\infty$，经 softmax 后变为 0：

$$
\text{score}_{ij} = \begin{cases}
\frac{q_i^\top k_j}{\sqrt{d_k}} & \text{if } M_{ij} = 1 \\
-\infty & \text{if } M_{ij} = 0
\end{cases}
$$

### 3.3 Transformer 编码器层

BERT 直接复用 Transformer 编码器结构，每层包含：

**1. 多头自注意力 + 残差 + LayerNorm**：

$$
\tilde{h}^{(l)} = \text{LayerNorm}\left(h^{(l-1)} + \text{MultiHeadAttn}(h^{(l-1)}, h^{(l-1)}, h^{(l-1)})\right)
$$

**2. 前馈网络 + 残差 + LayerNorm**：

$$
h^{(l)} = \text{LayerNorm}\left(\tilde{h}^{(l)} + \text{FFN}(\tilde{h}^{(l)})\right)
$$

其中前馈网络使用 **GELU** 激活函数（不同于原始 Transformer 的 ReLU）：

$$
\text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2
$$

**GELU 的定义**：

$$
\boxed{\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]}
$$

其中 $\Phi(x)$ 是标准正态分布的 CDF。GELU 的近似形式：

$$
\text{GELU}(x) \approx 0.5x\left(1 + \tanh\left[\sqrt{\frac{2}{\pi}}\left(x + 0.044715x^3\right)\right]\right)
$$

> **Q:** 为什么用 GELU 而不是 ReLU？
>
> **A:** GELU 是光滑的、处处可导的，不会像 ReLU 在 $x=0$ 处产生梯度不连续。对于预训练这样的大规模优化任务，更平滑的损失面有助于训练稳定性。

### 3.4 BERT-Base 与 BERT-Large 配置

| 参数 | BERT-Base | BERT-Large |
|------|-----------|------------|
| 层数 $L$ | 12 | 24 |
| 隐藏维度 $d$ | 768 | 1024 |
| 注意力头数 $A$ | 12 | 16 |
| 每头维度 $d_k$ | 64 | 64 |
| FFN 维度 $d_{ff}$ | 3072 | 4096 |
| 总参数量 | 110M | 340M |
| 最大序列长度 | 512 | 512 |
| 词表大小 | 30,522 | 30,522 |

**参数量估算**（BERT-Base）：

嵌入层：
$$
P_{\text{emb}} = |V| \cdot d + L_{\max} \cdot d + 2 \cdot d = 30522 \times 768 + 512 \times 768 + 2 \times 768 \approx 23.8\text{M}
$$

单层 Transformer：
$$
P_{\text{layer}} = \underbrace{4 \cdot d^2}_{\text{MultiHead}} + \underbrace{2 \cdot d \cdot d_{ff}}_{\text{FFN}} + \underbrace{4d + 2d_{ff}}_{\text{biases}} \approx 7.1\text{M}
$$

总计：
$$
P_{\text{total}} = P_{\text{emb}} + L \cdot P_{\text{layer}} + P_{\text{heads}} \approx 23.8\text{M} + 12 \times 7.1\text{M} + 1.5\text{M} \approx 110\text{M}
$$

---

## 4. 预训练目标的数学推导

### 4.1 MLM 损失函数

**符号定义**：

- $x = (x_1, x_2, \ldots, x_n)$：原始输入序列
- $\tilde{x}$：经掩码处理后的输入序列
- $\mathcal{M} \subseteq \{1, 2, \ldots, n\}$：被掩码位置的集合，$|\mathcal{M}| \approx 0.15n$
- $h_t \in \mathbb{R}^d$：位置 $t$ 的最终隐藏状态
- $W_{\text{MLM}} \in \mathbb{R}^{|V| \times d}$：MLM 输出投影矩阵

**前向计算**：

1. 获取隐藏状态：$H = \text{BERT}(\tilde{x}) \in \mathbb{R}^{n \times d}$
2. 对掩码位置预测 token：

$$
\text{logits}_t = W_{\text{MLM}} \cdot h_t + b_{\text{MLM}} \in \mathbb{R}^{|V|}, \quad t \in \mathcal{M}
$$

3. Softmax 得到概率分布：

$$
P(x_t = w \mid \tilde{x}) = \frac{\exp(\text{logits}_{t,w})}{\sum_{w'=1}^{|V|} \exp(\text{logits}_{t,w'})}, \quad w \in \{1, \ldots, |V|\}
$$

**MLM 损失**（交叉熵）：

$$
\boxed{\mathcal{L}_{\text{MLM}} = -\frac{1}{|\mathcal{M}|} \sum_{t \in \mathcal{M}} \log P(x_t \mid \tilde{x})}
$$

**梯度分析**：

对于 MLM 输出层权重 $W_{\text{MLM}}$：

$$
\frac{\partial \mathcal{L}_{\text{MLM}}}{\partial W_{\text{MLM}}} = -\frac{1}{|\mathcal{M}|} \sum_{t \in \mathcal{M}} \left(\mathbb{1}_{x_t} - P(\cdot \mid \tilde{x})\right) h_t^\top
$$

其中 $\mathbb{1}_{x_t} \in \mathbb{R}^{|V|}$ 是真实 token 的 one-hot 向量。

> **直觉**：梯度的方向使得正确 token 的概率增大（$\mathbb{1}_{x_t}$ 项），其他 token 的概率减小（$-P(\cdot \mid \tilde{x})$ 项）。

### 4.2 NSP 损失函数

**符号定义**：

- $h_{\texttt{[CLS]}} \in \mathbb{R}^d$：`[CLS]` 位置的最终隐藏状态
- $W_{\text{NSP}} \in \mathbb{R}^{2 \times d}$：NSP 分类矩阵
- $y \in \{0, 1\}$：真实标签（0 = IsNext，1 = NotNext）

**前向计算**：

$$
\text{logits}_{\text{NSP}} = W_{\text{NSP}} \cdot h_{\texttt{[CLS]}} + b_{\text{NSP}} \in \mathbb{R}^2
$$

$$
P(\text{class} \mid A, B) = \text{softmax}(\text{logits}_{\text{NSP}}) \in \mathbb{R}^2
$$

**NSP 损失**（二分类交叉熵）：

$$
\boxed{\mathcal{L}_{\text{NSP}} = -\log P(y \mid A, B)}
$$

展开为：

$$
\mathcal{L}_{\text{NSP}} = -\left[y \log P(\text{IsNext} \mid A, B) + (1 - y) \log P(\text{NotNext} \mid A, B)\right]
$$

### 4.3 联合损失与梯度分析

BERT 的总预训练损失为两个目标的简单相加：

$$
\boxed{\mathcal{L}_{\text{BERT}} = \mathcal{L}_{\text{MLM}} + \mathcal{L}_{\text{NSP}}}
$$

**梯度流分析**：

对于 Transformer 编码器参数 $\theta$：

$$
\frac{\partial \mathcal{L}_{\text{BERT}}}{\partial \theta} = \frac{\partial \mathcal{L}_{\text{MLM}}}{\partial \theta} + \frac{\partial \mathcal{L}_{\text{NSP}}}{\partial \theta}
$$

**关键观察**：

1. **MLM 梯度**：来自 $|\mathcal{M}|$ 个掩码位置，信号分布在整个序列上
2. **NSP 梯度**：仅来自 `[CLS]` 一个位置，但通过注意力机制影响所有参数
3. MLM 提供**token 级别**的监督信号，NSP 提供**句子级别**的监督信号

**两个损失的量级对比**：

$$
\mathcal{L}_{\text{MLM}} \approx -\frac{1}{|\mathcal{M}|} \sum_{t \in \mathcal{M}} \log \frac{1}{|V|} \approx \log |V| \approx 10.3 \quad (\text{初始化时})
$$

$$
\mathcal{L}_{\text{NSP}} \approx -\log \frac{1}{2} = \log 2 \approx 0.69 \quad (\text{初始化时})
$$

> **注意**：MLM 损失在数值上远大于 NSP 损失，这意味着 MLM 在训练初期主导梯度更新方向。

---

## 5. 训练优化方法总结

### 5.1 预训练策略

**数据规模**：

| 数据集 | 大小 | 说明 |
|--------|------|------|
| BooksCorpus | 800M 词 | 未出版书籍 |
| English Wikipedia | 2,500M 词 | 仅文本，去除表格和列表 |
| **合计** | ~3.3B 词 | 约 16GB 文本 |

**两阶段预训练**：

| 阶段 | 序列长度 | 批大小 | 步数 | 说明 |
|------|---------|--------|------|------|
| 阶段 1 | 128 | 256 | 900K | 90% 的训练步，短序列加速 |
| 阶段 2 | 512 | 256 | 100K | 10% 的训练步，学习长距离依赖 |

**总计算量**：

$$
\text{FLOPs} \approx 6 \times P \times D \approx 6 \times 110\text{M} \times 137\text{B} \approx 9 \times 10^{19}
$$

其中 $P$ 是参数量，$D$ 是总 token 数（序列长度 × 批大小 × 步数）。

### 5.2 微调策略

BERT 微调的核心思想：**在预训练模型顶部添加一个简单的任务特定层**。

**分类任务**（如情感分析、NLI）：

$$
P(y \mid x) = \text{softmax}(W_{\text{cls}} \cdot h_{\texttt{[CLS]}} + b_{\text{cls}})
$$

**序列标注任务**（如 NER）：

$$
P(y_t \mid x) = \text{softmax}(W_{\text{tag}} \cdot h_t + b_{\text{tag}}), \quad t = 1, \ldots, n
$$

**问答任务**（如 SQuAD）：

预测答案的起始和结束位置：

$$
P_{\text{start}}(t) = \frac{\exp(W_s \cdot h_t)}{\sum_{j=1}^n \exp(W_s \cdot h_j)}
$$

$$
P_{\text{end}}(t) = \frac{\exp(W_e \cdot h_t)}{\sum_{j=1}^n \exp(W_e \cdot h_j)}
$$

$$
\text{score}(i, j) = P_{\text{start}}(i) \cdot P_{\text{end}}(j), \quad i \leq j
$$

### 5.3 优化器与学习率调度

**Adam with Weight Decay (AdamW)**：

$$
\theta_{t+1} = \theta_t - \eta \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t\right)
$$

其中 $\lambda$ 是权重衰减系数（BERT 使用 $\lambda = 0.01$）。

**学习率调度**（线性预热 + 线性衰减）：

$$
\eta(t) = \begin{cases}
\eta_{\max} \cdot \frac{t}{t_{\text{warmup}}} & \text{if } t < t_{\text{warmup}} \\
\eta_{\max} \cdot \frac{T - t}{T - t_{\text{warmup}}} & \text{if } t \geq t_{\text{warmup}}
\end{cases}
$$

**超参数设置**：

| 超参数 | 预训练 | 微调 |
|--------|--------|------|
| 学习率 $\eta$ | $1 \times 10^{-4}$ | $2 \times 10^{-5}$ |
| Batch Size | 256 | 16/32 |
| Epochs | — | 3-4 |
| Warmup | 10K steps | 10% |
| Weight Decay | 0.01 | 0.01 |
| Adam $\beta_1$ | 0.9 | 0.9 |
| Adam $\beta_2$ | 0.999 | 0.999 |
| Dropout | 0.1 | 0.1 |

---

## 6. 从数学到代码：完整实现

### 6.1 NumPy 实现核心组件

```python
import numpy as np

def softmax(x, axis=-1):
    """
    数值稳定的 Softmax
    
    数学公式:
        softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def gelu(x):
    """
    GELU 激活函数
    
    数学公式:
        GELU(x) = x * Φ(x) ≈ 0.5x(1 + tanh[√(2/π)(x + 0.044715x³)])
    """
    return 0.5 * x * (1.0 + np.tanh(
        np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))
    ))


def layer_norm(x, gamma, beta, eps=1e-12):
    """
    层归一化
    
    数学公式:
        LayerNorm(x) = γ * (x - μ) / √(σ² + ε) + β
    
    参数:
        x: 输入，形状 (batch, seq_len, d_model)
        gamma: 缩放参数，形状 (d_model,)
        beta: 偏移参数，形状 (d_model,)
    """
    mean = np.mean(x, axis=-1, keepdims=True)       # (batch, seq_len, 1)
    var = np.var(x, axis=-1, keepdims=True)          # (batch, seq_len, 1)
    x_norm = (x - mean) / np.sqrt(var + eps)         # (batch, seq_len, d_model)
    return gamma * x_norm + beta


def scaled_dot_product_attention_numpy(Q, K, V, mask=None):
    """
    缩放点积注意力 (NumPy)
    
    数学公式:
        Attention(Q, K, V) = softmax(QK^T / √d_k) V
    
    参数:
        Q: 查询矩阵，形状 (batch, heads, seq_len, d_k)
        K: 键矩阵，形状 (batch, heads, seq_len, d_k)
        V: 值矩阵，形状 (batch, heads, seq_len, d_v)
        mask: 注意力掩码，形状 (batch, 1, 1, seq_len)
    
    返回:
        output: 注意力输出，形状 (batch, heads, seq_len, d_v)
        weights: 注意力权重，形状 (batch, heads, seq_len, seq_len)
    """
    d_k = Q.shape[-1]
    
    # 1. 计算注意力分数：QK^T / √d_k
    scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    # scores: (batch, heads, seq_len, seq_len)
    
    # 2. 应用掩码（BERT 中仅处理 padding）
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)
    
    # 3. Softmax 归一化
    weights = softmax(scores, axis=-1)
    # weights: (batch, heads, seq_len, seq_len)
    
    # 4. 加权求和
    output = np.matmul(weights, V)
    # output: (batch, heads, seq_len, d_v)
    
    return output, weights


def bert_embedding_numpy(token_ids, segment_ids, position_ids,
                         token_emb, segment_emb, position_emb,
                         ln_gamma, ln_beta):
    """
    BERT 输入嵌入 (NumPy)
    
    数学公式:
        E(t) = E_token(x_t) + E_segment(s_t) + E_position(t)
        output = LayerNorm(E(t))
    
    参数:
        token_ids: token 索引，形状 (batch, seq_len)
        segment_ids: 段落索引，形状 (batch, seq_len)
        position_ids: 位置索引，形状 (batch, seq_len)
        token_emb: token 嵌入矩阵，形状 (vocab_size, d_model)
        segment_emb: 段落嵌入矩阵，形状 (2, d_model)
        position_emb: 位置嵌入矩阵，形状 (max_len, d_model)
        ln_gamma, ln_beta: LayerNorm 参数
    """
    # 查表获取嵌入向量
    tok_emb = token_emb[token_ids]       # (batch, seq_len, d_model)
    seg_emb = segment_emb[segment_ids]   # (batch, seq_len, d_model)
    pos_emb = position_emb[position_ids] # (batch, seq_len, d_model)
    
    # 三种嵌入相加
    embeddings = tok_emb + seg_emb + pos_emb  # (batch, seq_len, d_model)
    
    # LayerNorm
    embeddings = layer_norm(embeddings, ln_gamma, ln_beta)
    
    return embeddings


def mlm_loss_numpy(hidden_states, masked_positions, true_labels,
                   mlm_weight, mlm_bias):
    """
    MLM 损失计算 (NumPy)
    
    数学公式:
        L_MLM = -1/|M| * Σ_{t∈M} log P(x_t | x̃)
    
    参数:
        hidden_states: BERT 输出，形状 (batch, seq_len, d_model)
        masked_positions: 掩码位置索引列表
        true_labels: 真实 token ID 列表
        mlm_weight: 输出投影矩阵，形状 (vocab_size, d_model)
        mlm_bias: 输出偏置，形状 (vocab_size,)
    """
    total_loss = 0.0
    num_masked = len(masked_positions)
    
    for i, (batch_idx, pos) in enumerate(masked_positions):
        # 取出掩码位置的隐藏状态
        h = hidden_states[batch_idx, pos]  # (d_model,)
        
        # 计算 logits
        logits = np.dot(mlm_weight, h) + mlm_bias  # (vocab_size,)
        
        # Softmax 概率
        probs = softmax(logits)  # (vocab_size,)
        
        # 交叉熵损失
        true_id = true_labels[i]
        total_loss -= np.log(probs[true_id] + 1e-10)
    
    return total_loss / max(num_masked, 1)


def nsp_loss_numpy(cls_hidden, nsp_weight, nsp_bias, true_label):
    """
    NSP 损失计算 (NumPy)
    
    数学公式:
        L_NSP = -log P(y | A, B)
        P(y | A, B) = softmax(W_NSP * h_[CLS] + b_NSP)
    
    参数:
        cls_hidden: [CLS] 位置的隐藏状态，形状 (d_model,)
        nsp_weight: NSP 分类权重，形状 (2, d_model)
        nsp_bias: NSP 分类偏置，形状 (2,)
        true_label: 真实标签 (0 或 1)
    """
    logits = np.dot(nsp_weight, cls_hidden) + nsp_bias  # (2,)
    probs = softmax(logits)  # (2,)
    loss = -np.log(probs[true_label] + 1e-10)
    return loss


# ========== 测试 NumPy 实现 ==========
if __name__ == "__main__":
    np.random.seed(42)
    batch_size, seq_len, d_model, vocab_size = 2, 8, 64, 100
    
    # 初始化参数
    token_emb = np.random.randn(vocab_size, d_model) * 0.02
    segment_emb = np.random.randn(2, d_model) * 0.02
    position_emb = np.random.randn(512, d_model) * 0.02
    ln_gamma, ln_beta = np.ones(d_model), np.zeros(d_model)
    
    # 创建输入并计算嵌入
    token_ids = np.random.randint(0, vocab_size, (batch_size, seq_len))
    segment_ids = np.array([[0,0,0,0,0,1,1,1], [0,0,0,1,1,1,1,1]])
    position_ids = np.tile(np.arange(seq_len), (batch_size, 1))
    
    embeddings = bert_embedding_numpy(
        token_ids, segment_ids, position_ids,
        token_emb, segment_emb, position_emb, ln_gamma, ln_beta
    )
    print(f"嵌入输出形状: {embeddings.shape}")  # (2, 8, 64)
    
    # 测试注意力
    Q = K = V = embeddings[:, np.newaxis, :, :]
    attn_out, attn_weights = scaled_dot_product_attention_numpy(Q, K, V)
    print(f"注意力权重行和: {attn_weights[0, 0, 0].sum():.4f}")  # 1.0
    
    # 测试 MLM 损失（初始化时约 log(100) ≈ 4.6）
    mlm_weight = np.random.randn(vocab_size, d_model) * 0.02
    mlm_loss = mlm_loss_numpy(
        embeddings, [(0, 2), (0, 5), (1, 3)], [42, 17, 88],
        mlm_weight, np.zeros(vocab_size)
    )
    print(f"MLM 损失: {mlm_loss:.4f}")
    
    # 测试 NSP 损失（初始化时约 log(2) ≈ 0.69）
    nsp_loss = nsp_loss_numpy(
        embeddings[0, 0], np.random.randn(2, d_model) * 0.02,
        np.zeros(2), true_label=0
    )
    print(f"NSP 损失: {nsp_loss:.4f}")
    print(f"GELU([-2,-1,0,1,2]) = {gelu(np.array([-2.,-1.,0.,1.,2.]))}")
```

### 6.2 PyTorch 完整实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class BERTEmbedding(nn.Module):
    """
    BERT 输入嵌入层
    
    数学公式:
        E(t) = E_token(x_t) + E_segment(s_t) + E_position(t)
        output = LayerNorm(Dropout(E(t)))
    
    参数:
        vocab_size: 词表大小 |V|
        d_model: 嵌入维度 d
        max_len: 最大序列长度
        dropout: dropout 概率
    """
    def __init__(self, vocab_size: int, d_model: int, max_len: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.segment_embedding = nn.Embedding(2, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, token_ids: torch.Tensor,
                segment_ids: torch.Tensor) -> torch.Tensor:
        """
        参数:
            token_ids: (batch, seq_len) token 索引
            segment_ids: (batch, seq_len) 段落索引 (0 或 1)
        返回:
            embeddings: (batch, seq_len, d_model) 输入嵌入
        """
        seq_len = token_ids.size(1)
        # 自动生成位置索引: [0, 1, 2, ..., seq_len-1]
        position_ids = torch.arange(seq_len, device=token_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(token_ids)
        
        # 三种嵌入相加
        embeddings = (
            self.token_embedding(token_ids) +
            self.segment_embedding(segment_ids) +
            self.position_embedding(position_ids)
        )
        
        # LayerNorm + Dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class MultiHeadSelfAttention(nn.Module):
    """
    多头自注意力 (BERT 版本)
    
    数学公式:
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
        head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
        Attention(Q, K, V) = softmax(QK^T / √d_k) V
    
    参数:
        d_model: 模型维度
        num_heads: 注意力头数
        dropout: dropout 概率
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        参数:
            x: (batch, seq_len, d_model) 输入
            mask: (batch, 1, 1, seq_len) 注意力掩码
        返回:
            output: (batch, seq_len, d_model) 注意力输出
            weights: (batch, heads, seq_len, seq_len) 注意力权重
        """
        batch_size, seq_len, _ = x.size()
        
        # 线性投影 + 分割多头
        Q = self.W_Q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # Q, K, V: (batch, heads, seq_len, d_k)
        
        # 缩放点积注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores: (batch, heads, seq_len, seq_len)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        
        # 加权求和
        context = torch.matmul(weights, V)
        # context: (batch, heads, seq_len, d_k)
        
        # 拼接多头
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # 输出投影
        output = self.W_O(context)
        
        return output, weights


class PositionwiseFeedForward(nn.Module):
    """
    位置前馈网络 (BERT 版本，使用 GELU)
    
    数学公式:
        FFN(x) = GELU(xW_1 + b_1)W_2 + b_2
    
    参数:
        d_model: 输入/输出维度
        d_ff: 隐藏层维度 (通常为 4 * d_model)
        dropout: dropout 概率
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = self.linear1(x)       # (batch, seq_len, d_ff)
        x = F.gelu(x)             # GELU 激活
        x = self.dropout(x)
        x = self.linear2(x)       # (batch, seq_len, d_model)
        return x


class BERTEncoderLayer(nn.Module):
    """
    BERT 编码器层
    
    结构:
        x → [Self-Attention] → [Add & Norm] → [FFN] → [Add & Norm] → output
    
    数学公式:
        h̃ = LayerNorm(x + MultiHeadAttn(x))
        h  = LayerNorm(h̃ + FFN(h̃))
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-12)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        参数:
            x: (batch, seq_len, d_model) 输入
            mask: (batch, 1, 1, seq_len) 注意力掩码
        返回:
            output: (batch, seq_len, d_model)
            attn_weights: (batch, heads, seq_len, seq_len)
        """
        # 1. 自注意力 + 残差 + LayerNorm
        attn_output, attn_weights = self.attention(x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # 2. FFN + 残差 + LayerNorm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x, attn_weights


class BERTEncoder(nn.Module):
    """
    BERT 编码器 (多层堆叠)
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            BERTEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None
               ) -> Tuple[torch.Tensor, list]:
        """
        参数:
            x: (batch, seq_len, d_model) 输入嵌入
            mask: (batch, 1, 1, seq_len) 注意力掩码
        返回:
            output: (batch, seq_len, d_model) 最终隐藏状态
            all_attn_weights: 各层注意力权重列表
        """
        all_attn_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            all_attn_weights.append(attn_weights)
        return x, all_attn_weights


class MLMHead(nn.Module):
    """
    Masked Language Model 预测头
    
    数学公式:
        logits = W_MLM * GELU(W_dense * h + b_dense) + b_MLM
    
    注意: BERT 的 MLM 头包含一个额外的 Dense + GELU + LayerNorm 层
    """
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-12)
        self.decoder = nn.Linear(d_model, vocab_size)
        # 注意：原始 BERT 共享 token embedding 权重
        # self.decoder.weight = embedding.token_embedding.weight
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        参数:
            hidden_states: (batch, seq_len, d_model) 或 (num_masked, d_model)
        返回:
            logits: (..., vocab_size) 预测 logits
        """
        x = self.dense(hidden_states)
        x = F.gelu(x)
        x = self.layer_norm(x)
        logits = self.decoder(x)
        return logits


class NSPHead(nn.Module):
    """
    Next Sentence Prediction 预测头
    
    数学公式:
        logits = W_NSP * h_[CLS] + b_NSP
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.classifier = nn.Linear(d_model, 2)
    
    def forward(self, cls_hidden: torch.Tensor) -> torch.Tensor:
        """
        参数:
            cls_hidden: (batch, d_model) [CLS] 位置的隐藏状态
        返回:
            logits: (batch, 2) NSP 预测 logits
        """
        return self.classifier(cls_hidden)


class BERTModel(nn.Module):
    """
    完整 BERT 模型（预训练版本）
    
    结构:
        Input → [Embedding] → [Encoder x L] → [MLM Head + NSP Head]
    """
    def __init__(
        self,
        vocab_size: int = 30522,
        d_model: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        d_ff: int = 3072,
        max_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        
        # 嵌入层
        self.embedding = BERTEmbedding(vocab_size, d_model, max_len, dropout)
        
        # Transformer 编码器
        self.encoder = BERTEncoder(d_model, num_heads, d_ff, num_layers, dropout)
        
        # 预训练头
        self.mlm_head = MLMHead(d_model, vocab_size)
        self.nsp_head = NSPHead(d_model)
        
        # 参数初始化
        self._init_weights()
    
    def _init_weights(self):
        """
        BERT 权重初始化：正态分布 N(0, 0.02)
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        token_ids: torch.Tensor,       # (batch, seq_len)
        segment_ids: torch.Tensor,     # (batch, seq_len)
        attention_mask: Optional[torch.Tensor] = None,  # (batch, seq_len)
        masked_positions: Optional[torch.Tensor] = None  # (batch, num_masked)
    ) -> dict:
        """
        参数:
            token_ids: (batch, seq_len) 输入 token 索引
            segment_ids: (batch, seq_len) 段落索引
            attention_mask: (batch, seq_len) 1=有效位置, 0=padding
            masked_positions: (batch, num_masked) 被掩码的位置索引
        
        返回:
            字典包含:
            - hidden_states: (batch, seq_len, d_model) 最终隐藏状态
            - mlm_logits: (batch, num_masked, vocab_size) MLM 预测
            - nsp_logits: (batch, 2) NSP 预测
        """
        # 1. 构造注意力掩码
        if attention_mask is not None:
            # (batch, seq_len) → (batch, 1, 1, seq_len) 用于广播
            extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        else:
            extended_mask = None
        
        # 2. 嵌入层
        embeddings = self.embedding(token_ids, segment_ids)
        # embeddings: (batch, seq_len, d_model)
        
        # 3. 编码器
        hidden_states, all_attn_weights = self.encoder(embeddings, extended_mask)
        # hidden_states: (batch, seq_len, d_model)
        
        # 4. MLM 预测
        if masked_positions is not None:
            batch_size = hidden_states.size(0)
            # 收集掩码位置的隐藏状态
            # masked_positions: (batch, num_masked) → 用于 gather
            num_masked = masked_positions.size(1)
            masked_positions_expanded = masked_positions.unsqueeze(-1).expand(
                -1, -1, self.d_model
            )  # (batch, num_masked, d_model)
            masked_hidden = torch.gather(
                hidden_states, 1, masked_positions_expanded
            )  # (batch, num_masked, d_model)
            mlm_logits = self.mlm_head(masked_hidden)
            # mlm_logits: (batch, num_masked, vocab_size)
        else:
            # 对所有位置预测（用于推理）
            mlm_logits = self.mlm_head(hidden_states)
        
        # 5. NSP 预测
        cls_hidden = hidden_states[:, 0, :]  # (batch, d_model)
        nsp_logits = self.nsp_head(cls_hidden)
        # nsp_logits: (batch, 2)
        
        return {
            "hidden_states": hidden_states,
            "mlm_logits": mlm_logits,
            "nsp_logits": nsp_logits,
            "attention_weights": all_attn_weights,
        }


class BERTPretrainingLoss(nn.Module):
    """
    BERT 预训练损失
    
    数学公式:
        L = L_MLM + L_NSP
        L_MLM = -1/|M| * Σ log P(x_t | x̃)
        L_NSP = -log P(y | A, B)
    """
    def __init__(self):
        super().__init__()
        self.mlm_loss_fn = nn.CrossEntropyLoss(reduction="mean")
        self.nsp_loss_fn = nn.CrossEntropyLoss(reduction="mean")
    
    def forward(
        self,
        mlm_logits: torch.Tensor,     # (batch, num_masked, vocab_size)
        mlm_labels: torch.Tensor,     # (batch, num_masked)
        nsp_logits: torch.Tensor,     # (batch, 2)
        nsp_labels: torch.Tensor      # (batch,)
    ) -> dict:
        """
        返回:
            字典包含:
            - total_loss: 总损失
            - mlm_loss: MLM 损失
            - nsp_loss: NSP 损失
        """
        # MLM 损失
        mlm_logits_flat = mlm_logits.view(-1, mlm_logits.size(-1))
        mlm_labels_flat = mlm_labels.view(-1)
        mlm_loss = self.mlm_loss_fn(mlm_logits_flat, mlm_labels_flat)
        
        # NSP 损失
        nsp_loss = self.nsp_loss_fn(nsp_logits, nsp_labels)
        
        # 总损失
        total_loss = mlm_loss + nsp_loss
        
        return {
            "total_loss": total_loss,
            "mlm_loss": mlm_loss,
            "nsp_loss": nsp_loss,
        }


def create_mlm_data(token_ids, vocab_size, mask_token_id=103,
                    mask_prob=0.15, seed=None):
    """
    创建 MLM 训练数据（80/10/10 掩码策略）
    
    参数:
        token_ids: (batch, seq_len) 原始 token 索引
        vocab_size: 词表大小
        mask_token_id: [MASK] 的 ID（默认 103）
        mask_prob: 掩码比例（默认 15%）
    返回:
        masked_token_ids, masked_positions, masked_labels
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    batch_size, seq_len = token_ids.size()
    special_tokens = {0, 101, 102}  # [PAD], [CLS], [SEP]
    
    all_positions, all_labels = [], []
    masked_token_ids = token_ids.clone()
    
    for b in range(batch_size):
        candidates = [i for i in range(seq_len)
                      if token_ids[b, i].item() not in special_tokens]
        num_to_mask = max(1, int(len(candidates) * mask_prob))
        indices = sorted(np.random.choice(candidates, num_to_mask, replace=False).tolist())
        
        pos, lab = [], []
        for idx in indices:
            pos.append(idx)
            lab.append(token_ids[b, idx].item())
            r = np.random.random()
            if r < 0.8:    masked_token_ids[b, idx] = mask_token_id    # 80%: [MASK]
            elif r < 0.9:  masked_token_ids[b, idx] = np.random.randint(0, vocab_size)  # 10%: 随机
            # 10%: 保持不变
        all_positions.append(pos)
        all_labels.append(lab)
    
    # 填充到相同长度
    max_m = max(len(p) for p in all_positions)
    padded_pos = torch.zeros(batch_size, max_m, dtype=torch.long)
    padded_lab = torch.full((batch_size, max_m), -100, dtype=torch.long)
    for b in range(batch_size):
        n = len(all_positions[b])
        padded_pos[b, :n] = torch.tensor(all_positions[b])
        padded_lab[b, :n] = torch.tensor(all_labels[b])
    
    return masked_token_ids, padded_pos, padded_lab


# ========== 完整测试 ==========
if __name__ == "__main__":
    import numpy as np
    
    # 缩小版超参数
    vocab_size, d_model, num_heads = 1000, 128, 4
    num_layers, d_ff, max_len = 2, 512, 64
    batch_size, seq_len = 4, 32
    
    # 1. 创建模型
    model = BERTModel(vocab_size, d_model, num_heads, num_layers, d_ff, max_len)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 2. 构造输入: [CLS] A... [SEP] B... [SEP]
    token_ids = torch.randint(3, vocab_size, (batch_size, seq_len))
    token_ids[:, 0], token_ids[:, 15], token_ids[:, -1] = 101, 102, 102
    segment_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    segment_ids[:, 16:] = 1
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    
    # 3. 创建 MLM 数据
    masked_ids, masked_pos, masked_labels = create_mlm_data(
        token_ids, vocab_size, mask_token_id=103, seed=42
    )
    
    # 4. 前向传播 + 损失计算
    model.eval()
    with torch.no_grad():
        out = model(masked_ids, segment_ids, attention_mask, masked_pos)
    
    loss_fn = BERTPretrainingLoss()
    nsp_labels = torch.randint(0, 2, (batch_size,))
    losses = loss_fn(out["mlm_logits"], masked_labels, out["nsp_logits"], nsp_labels)
    
    print(f"MLM Loss: {losses['mlm_loss'].item():.4f}")
    print(f"NSP Loss: {losses['nsp_loss'].item():.4f}")
    print(f"Total:    {losses['total_loss'].item():.4f}")
    
    # 5. 验证双向注意力
    attn = out["attention_weights"][0]  # 第 1 层
    print(f"注意力行和: {attn[0, 0, 0].sum().item():.4f}")
    print(f"非零比例:   {(attn[0, 0] > 1e-6).float().mean().item():.4f}")
    
    # 6. 训练一步
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    out = model(masked_ids, segment_ids, attention_mask, masked_pos)
    loss = loss_fn(out["mlm_logits"], masked_labels, out["nsp_logits"], nsp_labels)
    loss["total_loss"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    opt.step()
    print(f"\n✅ BERT 模型测试通过！训练后 Loss: {loss['total_loss'].item():.4f}")
```

---

## 7. 实践技巧与可视化

### 7.1 注意力可视化

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_attention(attention_weights, tokens, layer=0, head=0):
    """
    可视化注意力权重热力图
    
    参数:
        attention_weights: 各层注意力权重列表或单层张量
        tokens: token 字符串列表
        layer: 要可视化的层
        head: 要可视化的注意力头
    """
    if isinstance(attention_weights, list):
        attn = attention_weights[layer][0, head].detach().cpu().numpy()
    else:
        attn = attention_weights[0, head].detach().cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(attn, cmap="Blues", vmin=0, vmax=attn.max())
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(tokens, fontsize=9)
    ax.set_xlabel("Key (被关注的位置)")
    ax.set_ylabel("Query (发起关注的位置)")
    ax.set_title(f"BERT 注意力热力图 (Layer {layer}, Head {head})")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig("bert_attention_viz.png", dpi=150)
    plt.show()
```

### 7.2 MLM 掩码策略分析

**为什么 15% 而不是更高或更低？**

| 掩码比例 | 优点 | 缺点 |
|---------|------|------|
| 5% | 上下文信息丰富 | 训练信号太少，收敛慢 |
| **15%** | **平衡训练信号和上下文** | **原论文选择** |
| 30% | 训练信号多 | 上下文损失过多，预测困难 |
| 50% | — | 接近噪声，模型难以学习 |

**80/10/10 策略的数学分析**：

设 $p_{\text{mask}} = 0.8$，$p_{\text{random}} = 0.1$，$p_{\text{keep}} = 0.1$。

对于一个被选中掩码的位置 $t$，其输入分布为：

$$
\tilde{x}_t = \begin{cases}
\texttt{[MASK]} & \text{w.p. } 0.8 \\
x_{\text{random}} \sim \text{Uniform}(V) & \text{w.p. } 0.1 \\
x_t & \text{w.p. } 0.1
\end{cases}
$$

**期望信息增益**：

模型需要在所有三种情况下都正确预测 $x_t$，这迫使模型学习**真正的上下文理解**，而不仅仅是记住"当看到 `[MASK]` 时需要预测"。

### 7.3 微调实战技巧

**技巧 1：逐层学习率衰减 (Layer-wise Learning Rate Decay)**

越靠近输入的层，学习率越小：

$$
\eta_l = \eta_{\text{base}} \cdot \alpha^{L - l}
$$

其中 $L$ 是总层数，$l$ 是当前层（从 0 开始），$\alpha \in (0, 1)$ 是衰减因子（典型值 0.95）。

**技巧 2：渐进式解冻 (Gradual Unfreezing)**

1. 先冻结所有 BERT 层，只训练任务头（1-2 epochs）
2. 从顶层开始逐层解冻
3. 最后解冻所有层

**技巧 3：常见微调超参数**

| 任务类型 | 学习率 | Batch Size | Epochs | 说明 |
|---------|--------|------------|--------|------|
| 文本分类 | 2e-5 | 32 | 3 | GLUE 基准 |
| NER | 3e-5 | 16 | 5 | 序列标注 |
| QA | 3e-5 | 12 | 2 | SQuAD |
| 相似度 | 2e-5 | 16 | 3 | STS-B |

---

## 8. 与其他模型的关系

### 8.1 BERT vs GPT：双向 vs 单向

| 特性 | BERT | GPT |
|------|------|-----|
| **预训练方向** | 双向 (Bidirectional) | 单向 (Left-to-Right) |
| **架构** | Transformer Encoder | Transformer Decoder |
| **预训练目标** | MLM + NSP | 自回归语言模型 |
| **注意力掩码** | 全连接（无因果掩码） | 下三角（因果掩码） |
| **擅长任务** | NLU（理解类） | NLG（生成类） |
| **输入** | 完整句子 | 前缀序列 |
| **输出** | 上下文表示 | 下一个 token |

**数学本质对比**：

BERT 建模的是**联合概率的变体**：
$$
P_{\text{BERT}}(x_t \mid x_{\backslash t}) = \frac{P(x_1, \ldots, x_n)}{\sum_{x_t'} P(x_1, \ldots, x_t', \ldots, x_n)}
$$

GPT 建模的是**条件概率的链式分解**：
$$
P_{\text{GPT}}(x_1, \ldots, x_n) = \prod_{t=1}^n P(x_t \mid x_1, \ldots, x_{t-1})
$$

> **关键洞察**：BERT 不能直接用于文本生成（因为它不是自回归的），但在理解类任务上通常优于同等规模的 GPT。

### 8.2 BERT 的后续发展

```
BERT (2018)
  ├── RoBERTa (2019) ── 去掉 NSP，更多数据，更长训练
  ├── ALBERT (2019) ── 参数共享，嵌入分解，减少参数
  ├── DistilBERT (2019) ── 知识蒸馏，6层，保留97%性能
  ├── SpanBERT (2019) ── 掩码连续 span，去掉 NSP
  ├── ELECTRA (2020) ── 替换 token 检测（更高效的预训练）
  └── DeBERTa (2020) ── 解耦注意力，增强掩码解码器
```

**各变体的关键改进**：

| 模型 | 关键改进 | 效果 |
|------|---------|------|
| RoBERTa | 去掉 NSP，动态掩码，更大 batch | GLUE 上超过 BERT-Large |
| ALBERT | 跨层参数共享 | 参数量减少 18x |
| DistilBERT | 知识蒸馏到 6 层 | 速度快 60%，保留 97% 性能 |
| ELECTRA | 判别式预训练 | 相同计算量下性能更好 |

### 8.3 预训练范式总结

$$
\boxed{
\text{预训练 (大规模无标注数据)} \xrightarrow{\text{迁移}} \text{微调 (少量标注数据)}
}
$$

BERT 确立了 NLP 领域的 **"预训练 + 微调"** 范式：

1. **阶段 1**：在大规模无标注语料上预训练，学习通用语言知识
2. **阶段 2**：在特定任务的少量标注数据上微调，适应具体任务

这一范式的优势：

| 优势 | 说明 |
|------|------|
| **数据效率** | 微调只需少量标注数据 |
| **通用性** | 一个预训练模型适用于多种任务 |
| **性能** | 在 11 项 NLP 任务上创造 SOTA |
| **易用性** | 微调只需添加一个简单的分类层 |

---

## 扩展阅读与实现

### 问题 1：为什么 BERT 使用 WordPiece 而不是字级别或词级别 tokenization？

**解答**：

WordPiece 是一种**子词 (subword)** 分词方法，平衡了词级别和字级别的优缺点：

| 方法 | 词表大小 | OOV 问题 | 语义粒度 |
|------|---------|---------|---------|
| 字级别 | ~100 | 无 OOV | 太细，序列太长 |
| 词级别 | ~100K+ | 严重 OOV | 合适 |
| **WordPiece** | **~30K** | **极少 OOV** | **平衡** |

WordPiece 的核心思想：从字符开始，逐步合并最频繁的字符对：

```
"unhappiness" → ["un", "##happy", "##ness"]
"playing"     → ["play", "##ing"]
```

前缀 `##` 表示该子词不是一个词的开头。

### 问题 2：BERT 的 `[CLS]` token 为什么能用于分类？

**解答**：

`[CLS]` 是一个特殊的 token，放在每个输入序列的开头。由于 BERT 是双向的，经过 $L$ 层 Transformer 后，`[CLS]` 的隐藏状态聚合了整个序列的信息：

$$
h_{\texttt{[CLS]}}^{(L)} = f\left(\text{全序列信息通过 } L \text{ 层注意力聚合}\right)
$$

在预训练阶段，NSP 任务强迫 `[CLS]` 学习句子级别的语义表示。

在微调阶段，我们在 `[CLS]` 的输出上添加分类层：

$$
P(y \mid x) = \text{softmax}(W \cdot h_{\texttt{[CLS]}}^{(L)} + b)
$$

### 问题 3：RoBERTa 为什么去掉了 NSP？

**解答**：

RoBERTa (Liu et al., 2019) 通过消融实验发现：

1. **NSP 对大多数下游任务没有帮助**，甚至在某些任务上有害
2. NSP 的负样本（随机配对的句子）太容易区分，模型主要通过**主题差异**而非**语义连贯性**来判断
3. 去掉 NSP 后，可以用更长的文本段进行训练，增加上下文信息

RoBERTa 的改进总结：

| 改进 | BERT | RoBERTa |
|------|------|---------|
| NSP | ✅ 有 | ❌ 去掉 |
| 掩码策略 | 静态（固定掩码） | 动态（每 epoch 重新掩码） |
| 数据量 | 16GB | 160GB |
| Batch Size | 256 | 8K |
| 训练步数 | 1M | 500K |

### 问题 4：BERT 能否用于文本生成？

**解答**：

BERT 本身**不能直接用于自回归文本生成**，因为：

1. BERT 是双向的，生成时没有"未来"上下文可以参考
2. BERT 的训练目标是 MLM（填空），不是序列生成

但有一些变通方法：

- **迭代式生成**：将所有位置标记为 `[MASK]`，然后逐步替换（如 MaskGAN）
- **非自回归生成**：一次性预测所有位置（如 CMLM）

不过，对于生成任务，GPT 系列或 T5（编码器-解码器）架构更合适。

### 问题 5：BERT 的参数效率问题

**解答**：

BERT-Base 110M 参数中，嵌入层占 ~23.8M（21.6%），编码器占 ~85.1M（77.4%）。ALBERT 针对此提出：

1. **嵌入分解**：$|V| \times d \to |V| \times e + e \times d$（$e=128$），嵌入参数从 23.4M 降至 4.0M
2. **跨层参数共享**：编码器参数从 $L \times 7.1\text{M} = 85.2\text{M}$ 降至 $7.1\text{M}$

---

## 参考资源

### 经典论文

1. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805). NAACL 2019.
   - **贡献**：提出 MLM + NSP 双目标预训练，确立预训练+微调范式

2. Liu, Y., et al. (2019). [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692). arXiv.
   - **贡献**：证明去掉 NSP、增加数据和训练时间能显著提升性能

3. Lan, Z., et al. (2020). [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942). ICLR 2020.
   - **贡献**：参数共享和嵌入分解，大幅减少参数量

4. Clark, K., et al. (2020). [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555). ICLR 2020.
   - **贡献**：替换 token 检测目标，更高效的预训练

### 教材与书籍

5. Jurafsky, D., & Martin, J. H. [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/). 3rd ed. (Draft).
   - **章节**：第 11 章详细讲解 BERT 和 Transformer 预训练

### 在线资源与教程

6. Alammar, J. [The Illustrated BERT, ELMo, and co.](https://jalammar.github.io/illustrated-bert/).
   - **内容**：BERT 架构和预训练目标的直观图解

7. Hugging Face. [BERT Documentation](https://huggingface.co/docs/transformers/model_doc/bert).
   - **内容**：BERT 的工业级实现和使用指南

8. Google Research. [BERT GitHub Repository](https://github.com/google-research/bert).
   - **内容**：BERT 原始 TensorFlow 实现和预训练模型

---

## 附录：符号表

| 符号 | 含义 | 维度/类型 |
|------|------|----------|
| $n$ | 输入序列长度 | 标量 |
| $d$ | 隐藏维度（$d_{model}$） | 标量，BERT-Base: 768 |
| $d_k$ | 每个注意力头的维度 | 标量，64 |
| $d_{ff}$ | FFN 隐藏层维度 | 标量，3072 |
| $L$ | Transformer 层数 | 标量，BERT-Base: 12 |
| $A$ | 注意力头数 | 标量，BERT-Base: 12 |
| $\|V\|$ | WordPiece 词表大小 | 标量，30,522 |
| $L_{\max}$ | 最大序列长度 | 标量，512 |
| $x_t$ | 位置 $t$ 的原始 token | 整数索引 |
| $\tilde{x}$ | 掩码处理后的输入序列 | $(n,)$ |
| $\mathcal{M}$ | 被掩码位置的集合 | 集合，$\|\mathcal{M}\| \approx 0.15n$ |
| $h_t$ | 位置 $t$ 的隐藏状态 | $(d,)$ |
| $h_{\texttt{[CLS]}}$ | `[CLS]` 位置的隐藏状态 | $(d,)$ |
| $E_{\text{token}}$ | Token 嵌入矩阵 | $(\|V\|, d)$ |
| $E_{\text{segment}}$ | 段落嵌入矩阵 | $(2, d)$ |
| $E_{\text{position}}$ | 位置嵌入矩阵 | $(L_{\max}, d)$ |
| $W_{\text{MLM}}$ | MLM 输出投影矩阵 | $(\|V\|, d)$ |
| $W_{\text{NSP}}$ | NSP 分类矩阵 | $(2, d)$ |
| $Q, K, V$ | 查询、键、值矩阵 | $(n, d_k)$ |
| $M$ | 注意力掩码矩阵 | $(n, n)$ |
| $\mathcal{L}_{\text{MLM}}$ | MLM 损失值 | 标量 |
| $\mathcal{L}_{\text{NSP}}$ | NSP 损失值 | 标量 |
| $\mathcal{L}_{\text{BERT}}$ | BERT 总预训练损失 | 标量 |
| $\ell(\cdot, \cdot)$ | 交叉熵损失函数 | 函数 |
| $\Phi(x)$ | 标准正态分布 CDF | 函数 |

**典型维度示例（BERT-Base）：**
- $d = 768$（隐藏维度）
- $d_k = 64$（每头维度）
- $d_{ff} = 3072$（FFN 维度）
- $|V| = 30{,}522$（词表大小）
- $L_{\max} = 512$（最大序列长度）

---

最后更新：2026-03-18
