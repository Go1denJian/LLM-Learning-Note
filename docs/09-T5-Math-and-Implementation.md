# T5 数学原理与实现 —— 统一 Text-to-Text 框架的完整推导

> **前置知识**：Transformer Encoder-Decoder、自注意力机制、交叉熵损失、Python 基础  
> **与前面内容的联系**：建议先学习 [GPT2-Math-and-Implementation](./08-GPT2-Math-and-Implementation.md)，理解自回归生成  
> **与后续内容的联系**：T5 的统一框架思想直接影响了 GPT-3 的 In-context Learning 和后续大模型设计

---

## 目录

1. [引言：为什么需要统一的 Text-to-Text 框架？](#1-引言为什么需要统一的-text-to-text-框架)
   - 1.1 [NLP 任务的碎片化困境](#11-nlp-任务的碎片化困境)
   - 1.2 [Text-to-Text 的核心洞察](#12-text-to-text-的核心洞察)
   - 1.3 [本科数学知识映射表](#13-本科数学知识映射表)
2. [核心思想：一切皆为文本生成](#2-核心思想一切皆为文本生成)
   - 2.1 [统一任务格式](#21-统一任务格式)
   - 2.2 [任务前缀设计](#22-任务前缀设计)
   - 2.3 [Colossal Clean Crawled Corpus (C4)](#23-colossal-clean-crawled-corpus-c4)
3. [T5 Encoder-Decoder 架构的数学描述](#3-t5-encoder-decoder-架构的数学描述)
   - 3.1 [整体架构概览](#31-整体架构概览)
   - 3.2 [Encoder：双向自注意力](#32-encoder双向自注意力)
   - 3.3 [Decoder：因果自注意力 + 交叉注意力](#33-decoder因果自注意力--交叉注意力)
   - 3.4 [Relative Position Bias](#34-relative-position-bias)
   - 3.5 [T5 各规模配置](#35-t5-各规模配置)
4. [Span Corruption 预训练目标的数学推导](#4-span-corruption-预训练目标的数学推导)
   - 4.1 [Span Corruption 机制](#41-span-corruption-机制)
   - 4.2 [损失函数与 Teacher Forcing](#42-损失函数与-teacher-forcing)
   - 4.3 [与 MLM / 自回归目标的对比](#43-与-mlm--自回归目标的对比)
5. [训练优化方法总结](#5-训练优化方法总结)
   - 5.1 [预训练策略](#51-预训练策略)
   - 5.2 [优化器：Adafactor](#52-优化器adafactor)
   - 5.3 [学习率调度与训练稳定性](#53-学习率调度与训练稳定性)
6. [从数学到代码：完整实现](#6-从数学到代码完整实现)
   - 6.1 [NumPy 实现核心组件](#61-numpy-实现核心组件)
   - 6.2 [PyTorch 完整实现](#62-pytorch-完整实现)
7. [多任务微调与推理](#7-多任务微调与推理)
   - 7.1 [多任务训练策略](#71-多任务训练策略)
   - 7.2 [温度采样混合比例](#72-温度采样混合比例)
   - 7.3 [Beam Search 解码](#73-beam-search-解码)
8. [与其他模型的关系](#8-与其他模型的关系)
   - 8.1 [BERT vs GPT-2 vs T5：三种范式](#81-bert-vs-gpt-2-vs-t5三种范式)
   - 8.2 [T5 的系统性消融实验](#82-t5-的系统性消融实验)
   - 8.3 [T5 的后续影响](#83-t5-的后续影响)

[扩展阅读与实现](#扩展阅读与实现)

[参考资源](#参考资源)

附录：[符号表](#附录符号表)

---

## 1. 引言：为什么需要统一的 Text-to-Text 框架？

### 1.1 NLP 任务的碎片化困境

在 T5 之前，NLP 领域面临一个根本问题：**不同任务需要不同的模型架构和输出格式**。

| 任务类型 | 输出格式 | 典型模型 |
|---------|---------|---------|
| 分类 | 单个标签 $y \in \{0, 1, \ldots, C-1\}$ | BERT + 分类头 |
| 序列标注 | 每个 token 一个标签 | BERT + token 分类头 |
| 生成 | 文本序列 | GPT-2 (自回归) |
| 翻译 | 文本序列 | Transformer Enc-Dec |
| 回归 | 实数 $y \in \mathbb{R}$ | BERT + 回归头 |

这种碎片化带来了三个问题：

1. **架构不统一**：每种任务需要设计不同的输出头（classification head、span extraction head 等）
2. **无法共享知识**：分类任务学到的知识难以迁移到生成任务
3. **比较困难**：不同架构的消融实验无法公平对比

### 1.2 Text-to-Text 的核心洞察

T5 的革命性提案：**将所有 NLP 任务统一为 text-to-text 格式**。

$$
\boxed{f_\theta: \text{text} \to \text{text}}
$$

无论是分类、翻译、摘要还是问答，输入和输出都是**文本字符串**：

| 任务 | 输入文本 | 输出文本 |
|------|---------|---------|
| 翻译 | "translate English to German: That is good." | "Das ist gut." |
| 摘要 | "summarize: [长文本]" | "简短摘要" |
| 情感分类 | "sst2 sentence: This movie is great." | "positive" |
| 相似度 | "stsb sentence1: A man is singing. sentence2: A woman is singing." | "3.2" |
| 问答 | "question: What is AI? context: [段落]" | "Artificial Intelligence" |

**数学统一**：所有任务都转化为条件文本生成问题：

$$
\boxed{P(\mathbf{y} \mid \mathbf{x}; \theta) = \prod_{t=1}^{m} P(y_t \mid y_{<t}, \mathbf{x}; \theta)}
$$

其中 $\mathbf{x} = (x_1, \ldots, x_n)$ 是输入序列（含任务前缀），$\mathbf{y} = (y_1, \ldots, y_m)$ 是输出序列。

### 1.3 本科数学知识映射表

| 数学概念 | T5 中的应用 | 代码对应 |
|---------|------------|---------|
| 条件概率 $P(\mathbf{y} \mid \mathbf{x})$ | Encoder-Decoder 生成 | `model(input_ids, decoder_input_ids)` |
| 交叉熵 $H(p, q)$ | 序列到序列损失 | `F.cross_entropy()` |
| 相对位置偏置 $b(i-j)$ | Relative Position Bias | `position_bias[i, j]` |
| 掩码矩阵 | Span Corruption | `mask_spans()` |
| Teacher Forcing | 解码器训练 | 右移目标序列 |
| Beam Search | 推理解码 | `model.generate(num_beams=4)` |

---

## 2. 核心思想：一切皆为文本生成

### 2.1 统一任务格式

T5 的核心创新：**用任务前缀（task prefix）区分不同任务**，模型架构和损失函数完全相同。

**形式化定义**：

给定任务 $\tau$、输入 $\mathbf{x}_{\text{raw}}$，T5 构造输入序列：

$$
\mathbf{x} = \text{prefix}(\tau) \oplus \mathbf{x}_{\text{raw}}
$$

其中 $\oplus$ 表示字符串拼接，$\text{prefix}(\tau)$ 是任务特定的文本前缀。

**所有任务共享同一个目标函数**：

$$
\boxed{\mathcal{L}(\theta) = -\sum_{(\mathbf{x}, \mathbf{y}) \in \mathcal{D}} \sum_{t=1}^{|\mathbf{y}|} \log P(y_t \mid y_{<t}, \mathbf{x}; \theta)}
$$

> **Q:** 为什么分类任务也要生成文本标签（如 "positive"），而不是直接输出 logits？
>
> **A:** 统一性的代价是分类效率略有下降，但收益是巨大的：(1) 所有任务共享完全相同的模型架构、损失函数和解码策略；(2) 多任务训练变得自然——只需混合不同任务的 text-to-text 样本；(3) 新任务只需设计前缀，无需修改模型结构。

### 2.2 任务前缀设计

T5 为每类任务设计了简洁的自然语言前缀：

| 任务类别 | 前缀示例 | 输入 | 输出 |
|---------|---------|------|------|
| 翻译 | "translate English to German:" | "That is good." | "Das ist gut." |
| 摘要 | "summarize:" | "[长文本]" | "[摘要]" |
| 分类 (CoLA) | "cola sentence:" | "The cat sat." | "acceptable" |
| 分类 (SST-2) | "sst2 sentence:" | "Great movie!" | "positive" |
| 相似度 (STS-B) | "stsb sentence1: ... sentence2: ..." | 两个句子 | "3.8"（浮点数文本） |
| 推理 (MNLI) | "mnli hypothesis: ... premise: ..." | 假设+前提 | "entailment" |
| 问答 (SQuAD) | "question: ... context: ..." | 问题+上下文 | 答案文本 |

**前缀的作用**——在条件概率中充当任务指示器：

$$
P(\mathbf{y} \mid \mathbf{x}_{\text{raw}}, \tau; \theta) \approx P(\mathbf{y} \mid \text{prefix}(\tau) \oplus \mathbf{x}_{\text{raw}}; \theta)
$$

### 2.3 Colossal Clean Crawled Corpus (C4)

T5 的预训练数据是精心清洗的大规模语料库 **C4**：

| 属性 | 值 |
|------|-----|
| 来源 | Common Crawl（2019 年 4 月） |
| 原始规模 | ~20TB 原始 HTML |
| 清洗后规模 | ~750GB 纯文本 |
| Token 数 | ~1T tokens |
| 语言 | 英语 |

**清洗流程**：

1. **语言过滤**：保留 langdetect 检测为英语（概率 ≥ 0.99）的页面
2. **去重**：基于 3-sentence 的去重（移除重复段落）
3. **质量过滤**：
   - 移除含脏话/敏感词的页面
   - 移除 "{" 开头的行（代码/JSON）
   - 移除句子不以标点结尾的页面
   - 移除过短的页面（< 5 句）

**为什么 C4 如此重要？**

$$
\underbrace{\text{BookCorpus (5GB)}}_{\text{GPT-1}} \to \underbrace{\text{WebText (40GB)}}_{\text{GPT-2}} \to \underbrace{\text{C4 (750GB)}}_{\text{T5}}
$$

T5 论文的核心贡献之一就是证明了：**在固定模型架构下，更大更干净的预训练数据能持续提升性能**。

---

## 3. T5 Encoder-Decoder 架构的数学描述

### 3.1 整体架构概览

T5 采用标准的 **Encoder-Decoder** 架构，与原始 Transformer 类似，但有几个关键修改：

```
输入: "translate English to German: That is good."
                    ↓
              [Token 嵌入]  (无位置嵌入!)
                    ↓
         ┌─── Encoder ───┐
         │  Self-Attn     │ × L_enc 层
         │  + FFN         │ (双向, 无因果掩码)
         │  + Relative    │
         │    Pos Bias    │
         └───────┬────────┘
                 │ Memory: (n, d)
                 ↓
         ┌─── Decoder ───┐
         │  Causal        │ × L_dec 层
         │  Self-Attn     │
         │  + Cross-Attn  │ (attend to encoder output)
         │  + FFN         │
         │  + Relative    │
         │    Pos Bias    │
         └───────┬────────┘
                 ↓
           [LM Head]
                 ↓
输出: "Das ist gut."
```

**T5 与原始 Transformer 的关键区别**：

| 特性 | 原始 Transformer | T5 |
|------|:---------------:|:---:|
| 位置编码 | 正弦位置嵌入（加在输入上） | Relative Position Bias（加在注意力分数上） |
| LayerNorm | Post-Norm | Pre-Norm (RMSNorm) |
| 激活函数 | ReLU | GELU（后改为 GeGLU） |
| LayerNorm 偏置 | 有 | 无（仅缩放，无偏移） |
| Dropout | Sublayer + Attention | Sublayer + Attention + FF |

### 3.2 Encoder：双向自注意力

Encoder 处理输入序列 $\mathbf{x} = (x_1, \ldots, x_n)$，输出上下文表示 $\mathbf{H}^{\text{enc}} \in \mathbb{R}^{n \times d}$。

**第 $l$ 层 Encoder 的计算**（Pre-Norm）：

$$
\boxed{
\begin{aligned}
\tilde{h}^{(l)} &= h^{(l-1)} + \text{SelfAttn}\left(\text{RMSNorm}(h^{(l-1)})\right) \\
h^{(l)} &= \tilde{h}^{(l)} + \text{FFN}\left(\text{RMSNorm}(\tilde{h}^{(l)})\right)
\end{aligned}
}
$$

**RMSNorm**（Root Mean Square Layer Normalization）：

$$
\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma, \quad \text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2}
$$

与标准 LayerNorm 的区别：**没有减去均值（无偏移项 $\beta$）**，计算更高效。

**Encoder 自注意力**（双向，无掩码）：

$$
\text{SelfAttn}(X) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + B^{\text{rel}}\right) V
$$

其中 $B^{\text{rel}} \in \mathbb{R}^{n \times n}$ 是 Relative Position Bias（见 3.4 节）。

注意：Encoder 的自注意力是**完全双向**的（无因果掩码），每个位置可以看到所有其他位置。

### 3.3 Decoder：因果自注意力 + 交叉注意力

Decoder 自回归地生成输出序列 $\mathbf{y} = (y_1, \ldots, y_m)$。每层包含**三个子层**：

**第 $l$ 层 Decoder 的完整计算**：

$$
\boxed{
\begin{aligned}
\tilde{s}^{(l)} &= s^{(l-1)} + \text{CausalSelfAttn}\left(\text{RMSNorm}(s^{(l-1)})\right) \\
\hat{s}^{(l)} &= \tilde{s}^{(l)} + \text{CrossAttn}\left(\text{RMSNorm}(\tilde{s}^{(l)}), \mathbf{H}^{\text{enc}}\right) \\
s^{(l)} &= \hat{s}^{(l)} + \text{FFN}\left(\text{RMSNorm}(\hat{s}^{(l)})\right)
\end{aligned}
}
$$

**因果自注意力**（Decoder Self-Attention）：

$$
\text{CausalSelfAttn}(S) = \text{softmax}\left(\frac{Q_s K_s^\top}{\sqrt{d_k}} + M^{\text{causal}} + B^{\text{rel}}\right) V_s
$$

其中 $M^{\text{causal}}$ 是因果掩码（与 GPT-2 相同），$Q_s, K_s, V_s$ 来自 Decoder 隐藏状态。

**交叉注意力**（Cross-Attention）：

$$
\text{CrossAttn}(S, H^{\text{enc}}) = \text{softmax}\left(\frac{Q_s (K_e)^\top}{\sqrt{d_k}} + B^{\text{rel}}_{\text{cross}}\right) V_e
$$

其中：
- $Q_s = S \cdot W_Q^{\text{cross}} \in \mathbb{R}^{m \times d_k}$ —— 查询来自 **Decoder**
- $K_e = H^{\text{enc}} \cdot W_K^{\text{cross}} \in \mathbb{R}^{n \times d_k}$ —— 键来自 **Encoder**
- $V_e = H^{\text{enc}} \cdot W_V^{\text{cross}} \in \mathbb{R}^{n \times d_k}$ —— 值来自 **Encoder**

**交叉注意力的直觉**：

$$
\underbrace{Q_s}_{\text{Decoder 在问}} \cdot \underbrace{K_e^\top}_{\text{Encoder 中哪里有答案？}} \to \underbrace{V_e}_{\text{从 Encoder 中取信息}}
$$

> **Q:** 为什么 T5 用 Encoder-Decoder 而不是 GPT-2 的 Decoder-only？
>
> **A:** T5 论文系统地对比了三种架构：(1) Encoder-Decoder、(2) Decoder-only (language model)、(3) Prefix LM。在相同参数量下，**Encoder-Decoder 在大多数任务上表现最好**，因为 Encoder 的双向注意力能更好地理解输入，Decoder 的因果注意力专注于生成输出。

### 3.4 Relative Position Bias

T5 **不使用位置嵌入**，而是在注意力分数上直接添加**相对位置偏置**。

**核心思想**：位置信息不编码在输入表示中，而是作为注意力分数的偏置项：

$$
\boxed{A_{ij} = \frac{q_i^\top k_j}{\sqrt{d_k}} + b(i - j)}
$$

其中 $b: \mathbb{Z} \to \mathbb{R}$ 是从相对位置到偏置值的映射。

**分桶策略**（Bucketed Relative Position）：

T5 不为每个可能的相对距离学习独立参数，而是将相对距离映射到有限个桶中：

$$
b(i - j) = B[\text{bucket}(i - j)]
$$

其中 $B \in \mathbb{R}^{n_{\text{buckets}}}$ 是可学习的偏置参数（每个注意力头独立），$\text{bucket}(\cdot)$ 是分桶函数。

**分桶函数**（T5 默认 32 个桶）：

$$
\text{bucket}(\delta) = \begin{cases}
\delta & \text{if } 0 \leq \delta < n_{\text{exact}} \quad (\text{精确桶}) \\
n_{\text{exact}} + \left\lfloor \frac{\log(\delta / n_{\text{exact}})}{\log(n_{\text{max}} / n_{\text{exact}})} \cdot (n_{\text{buckets}}/2 - n_{\text{exact}}) \right\rfloor & \text{if } \delta \geq n_{\text{exact}} \quad (\text{对数桶})
\end{cases}
$$

其中 $n_{\text{exact}} = 8$（小距离精确编码），$n_{\text{max}} = 128$（最大相对距离），$n_{\text{buckets}} = 32$。

对于双向注意力（Encoder），正负方向分别使用 $n_{\text{buckets}}/2 = 16$ 个桶。

**可视化**（32 个桶，双向）：

```
相对距离:  ... -5 -4 -3 -2 -1  0  1  2  3  4  5 ...  128+
桶索引:    ... 21 20 19 18 17 16  0  1  2  3  4 ...   15
           |←── 对数桶 ──|← 精确 →|←── 精确 →|── 对数桶 ──→|
```

**Relative Position Bias 的优势**：

1. **泛化到更长序列**：对数分桶使模型在训练时只见过短序列也能处理长序列
2. **参数高效**：32 个桶远少于 $n_{\text{max}}$ 个独立位置嵌入
3. **层间共享**：同一层的所有头共享桶索引，但偏置参数独立

> **注意**：T5 中 Relative Position Bias **仅在第一层计算**，然后在所有层中共享（共享桶索引和偏置值）。这是一个重要的效率优化。

### 3.5 T5 各规模配置

| 参数 | Small | Base | Large | 3B | 11B |
|------|:-----:|:----:|:-----:|:--:|:---:|
| 参数量 | 60M | 220M | 770M | 3B | 11B |
| Encoder 层数 $L_{\text{enc}}$ | 6 | 12 | 24 | 24 | 24 |
| Decoder 层数 $L_{\text{dec}}$ | 6 | 12 | 24 | 24 | 24 |
| 隐藏维度 $d$ | 512 | 768 | 1024 | 1024 | 1024 |
| FFN 维度 $d_{ff}$ | 2048 | 3072 | 4096 | 16384 | 65536 |
| 注意力头数 $A$ | 8 | 12 | 16 | 32 | 128 |
| 每头维度 $d_k$ | 64 | 64 | 64 | 128 | 128 |
| 词表大小 $\|V\|$ | 32,128 | 32,128 | 32,128 | 32,128 | 32,128 |

**参数量估算**（T5-Base）：

嵌入层（无位置嵌入）：
$$
P_{\text{emb}} = |V| \cdot d = 32128 \times 768 \approx 24.7\text{M}
$$

单层 Encoder（Self-Attn + FFN + RMSNorm）：
$$
P_{\text{enc\_layer}} = \underbrace{4d^2}_{\text{Self-Attn}} + \underbrace{2 \cdot d \cdot d_{ff}}_{\text{FFN}} + \underbrace{2d}_{\text{RMSNorm}} \approx 7.1\text{M}
$$

单层 Decoder（Self-Attn + Cross-Attn + FFN + RMSNorm）：
$$
P_{\text{dec\_layer}} = \underbrace{4d^2}_{\text{Self-Attn}} + \underbrace{4d^2}_{\text{Cross-Attn}} + \underbrace{2 \cdot d \cdot d_{ff}}_{\text{FFN}} + \underbrace{3d}_{\text{RMSNorm}} \approx 9.4\text{M}
$$

总计：
$$
P_{\text{total}} \approx P_{\text{emb}} + L_{\text{enc}} \cdot P_{\text{enc\_layer}} + L_{\text{dec}} \cdot P_{\text{dec\_layer}} \approx 24.7 + 12 \times 7.1 + 12 \times 9.4 \approx 223\text{M}
$$

---

## 4. Span Corruption 预训练目标的数学推导

### 4.1 Span Corruption 机制

T5 的预训练目标不是 BERT 的 MLM（随机遮蔽单个 token），而是 **Span Corruption**——随机遮蔽连续的 token 片段（spans）。

**算法流程**：

给定原始序列 $\mathbf{x}_{\text{orig}} = (x_1, x_2, \ldots, x_n)$：

1. **采样遮蔽位置**：随机选择约 15% 的 token 进行遮蔽
2. **合并为 spans**：将连续的遮蔽 token 合并为一个 span，平均 span 长度为 3
3. **替换为哨兵 token**：每个 span 替换为唯一的哨兵 token $\langle\text{extra\_id\_}k\rangle$
4. **构造目标序列**：目标由哨兵 token + 被遮蔽内容组成

**示例**：

```
原始文本:  "Thank you for inviting me to your party last week"
遮蔽 spans: "Thank you [SPAN1] me to your [SPAN2] last week"

Encoder 输入: "Thank you <extra_id_0> me to your <extra_id_1> last week"
Decoder 目标: "<extra_id_0> for inviting <extra_id_1> party <extra_id_2>"
```

**形式化定义**：

设遮蔽函数 $\text{corrupt}: \mathbf{x}_{\text{orig}} \to (\mathbf{x}_{\text{input}}, \mathbf{y}_{\text{target}})$：

$$
\mathbf{x}_{\text{input}} = \text{replace\_spans}(\mathbf{x}_{\text{orig}}, \text{spans}, \text{sentinels})
$$

$$
\mathbf{y}_{\text{target}} = \bigoplus_{k} \left(\langle\text{extra\_id\_}k\rangle \oplus \text{span}_k\right) \oplus \langle\text{extra\_id\_}K\rangle
$$

其中 $K$ 是 span 总数，末尾的 $\langle\text{extra\_id\_}K\rangle$ 标志目标结束。

**Span 采样过程的数学描述**：

设噪声比例 $\rho = 0.15$，平均 span 长度 $\mu = 3$，则：

- 预期被遮蔽的 token 数：$n_{\text{mask}} = \lfloor \rho \cdot n \rfloor$
- 预期 span 数：$K = \lfloor n_{\text{mask}} / \mu \rfloor$
- Encoder 输入长度：$n_{\text{input}} \approx n - n_{\text{mask}} + K$
- Decoder 目标长度：$m_{\text{target}} \approx n_{\text{mask}} + K + 1$

$$
\boxed{n_{\text{input}} + m_{\text{target}} \approx n + 2K + 1 < 2n}
$$

这意味着 Span Corruption 的输入+输出总长度**小于原始序列的两倍**，比逐 token 预测更高效。

### 4.2 损失函数与 Teacher Forcing

**损失函数**：

T5 的预训练损失是标准的序列到序列交叉熵：

$$
\boxed{\mathcal{L}(\theta) = -\frac{1}{m} \sum_{t=1}^{m} \log P(y_t \mid y_{<t}, \mathbf{x}_{\text{input}}; \theta)}
$$

其中 $\mathbf{y} = (y_1, \ldots, y_m)$ 是 Decoder 目标序列。

**Teacher Forcing 训练**：

训练时，Decoder 的输入是**右移的目标序列**（ground truth），而非模型自身的预测：

$$
\text{Decoder 输入} = (\langle\text{bos}\rangle, y_1, y_2, \ldots, y_{m-1})
$$

$$
\text{Decoder 目标} = (y_1, y_2, \ldots, y_m)
$$

**为什么用 Teacher Forcing？**

自回归生成时，模型在位置 $t$ 需要前面的输出 $y_{<t}$。如果用模型自身的预测（可能有错），错误会累积（exposure bias）。Teacher Forcing 用真实标签替代模型预测：

$$
\underbrace{P(y_t \mid y_1^*, \ldots, y_{t-1}^*, \mathbf{x})}_{\text{Teacher Forcing: 用真实标签}} \quad \text{vs} \quad \underbrace{P(y_t \mid \hat{y}_1, \ldots, \hat{y}_{t-1}, \mathbf{x})}_{\text{Free Running: 用模型预测}}
$$

**Teacher Forcing 的梯度分析**：

对 Decoder 输出层参数 $W_{\text{lm}}$ 的梯度：

$$
\frac{\partial \mathcal{L}}{\partial W_{\text{lm}}} = -\frac{1}{m}\sum_{t=1}^{m} \left(\mathbb{1}_{y_t} - P(\cdot \mid y_{<t}^*, \mathbf{x})\right) \cdot s_t^\top
$$

其中 $s_t$ 是 Decoder 在位置 $t$ 的隐藏状态，$y_{<t}^*$ 是真实标签前缀。

由于使用真实标签，每个位置的梯度**独立且无偏**，训练更稳定。

### 4.3 与 MLM / 自回归目标的对比

| 特性 | BERT (MLM) | GPT-2 (自回归) | T5 (Span Corruption) |
|------|:----------:|:-------------:|:-------------------:|
| 遮蔽方式 | 随机 15% 单 token | 无遮蔽（预测下一个） | 随机 15%，连续 span |
| 输出长度 | 与输入等长 | 与输入等长 | **远短于输入** |
| 架构 | Encoder-only | Decoder-only | Encoder-Decoder |
| 计算效率 | 仅 15% 位置有损失 | 所有位置有损失 | Decoder 短序列 → 高效 |
| 上下文 | 双向 | 单向（因果） | Encoder 双向 + Decoder 因果 |

**效率对比**：

设原始序列长度 $n = 512$，遮蔽 15%，span 长度 3：

- **BERT MLM**：输入 512，输出 512（但仅 ~77 个位置有损失）
- **GPT-2**：输入 512，输出 512（所有位置有损失）
- **T5 Span Corruption**：
  - Encoder 输入：$512 - 77 + 26 \approx 461$ tokens
  - Decoder 输出：$77 + 26 + 1 \approx 104$ tokens

$$
\boxed{\text{T5 Decoder 处理的序列长度仅为 Encoder 的 } \sim 23\%}
$$

这是 Span Corruption 的关键效率优势：Decoder（计算最密集的部分）只需处理短序列。

---

## 5. 训练优化方法总结

### 5.1 预训练策略

**数据处理**：

| 属性 | 值 |
|------|-----|
| 分词方法 | SentencePiece (Unigram) |
| 词表大小 | 32,128 |
| 最大输入长度 | 512 tokens |
| 最大目标长度 | 114 tokens |
| 预训练步数 | $2^{19} = 524{,}288$ 步（T5-Base） |
| Batch Size | 128 序列 × 512 tokens = 65,536 tokens/batch |
| 总预训练 Token 数 | $\sim 34\text{B}$ tokens |

**SentencePiece 分词**：

T5 使用 SentencePiece 的 Unigram 模型（非 BPE），优势在于：

1. **语言无关**：不需要预分词（如空格分割），适用于多语言
2. **概率模型**：每个子词有概率分数，可以优化子词选择
3. **确定性分词**：给定词表，分词结果唯一

### 5.2 优化器：Adafactor

T5 使用 **Adafactor** 优化器，而非 Adam。Adafactor 的核心优势是**内存效率**。

**Adam 的内存问题**：

Adam 需要存储两个与参数同维度的状态：
- 一阶矩 $m \in \mathbb{R}^{p \times q}$
- 二阶矩 $v \in \mathbb{R}^{p \times q}$

总内存：$3 \times p \times q$（参数 + 两个状态）。

**Adafactor 的解决方案**：

对矩阵参数 $W \in \mathbb{R}^{p \times q}$，将二阶矩分解为行和列的外积近似：

$$
\hat{v}_{ij} = r_i \cdot c_j
$$

其中 $r \in \mathbb{R}^p$（行因子），$c \in \mathbb{R}^q$（列因子）。

**Adafactor 更新规则**：

$$
r_t = \rho_t \cdot r_{t-1} + (1 - \rho_t) \cdot \frac{1}{q}\sum_{j=1}^{q} g_{t,ij}^2
$$

$$
c_t = \rho_t \cdot c_{t-1} + (1 - \rho_t) \cdot \frac{1}{p}\sum_{i=1}^{p} g_{t,ij}^2
$$

$$
\hat{v}_{t,ij} = \frac{r_{t,i} \cdot c_{t,j}}{\frac{1}{p}\sum_{i'} r_{t,i'}}
$$

$$
\boxed{\theta_{t+1} = \theta_t - \eta_t \cdot \frac{g_t}{\sqrt{\hat{v}_t} + \epsilon}}
$$

**内存对比**：

| 优化器 | 参数 $(p \times q)$ 的额外内存 | 11B 模型总内存 |
|--------|:----------------------------:|:-------------:|
| Adam | $2pq$ | ~88 GB |
| Adafactor | $p + q$ | ~44 GB |

### 5.3 学习率调度与训练稳定性

**逆平方根学习率调度**：

$$
\boxed{\eta(t) = \frac{1}{\sqrt{\max(t, t_{\text{warmup}})}}}
$$

在预热期 $t \leq t_{\text{warmup}}$ 内学习率恒定为 $1/\sqrt{t_{\text{warmup}}}$，之后按 $1/\sqrt{t}$ 衰减。

**T5 预训练超参数**：

| 超参数 | 值 |
|--------|-----|
| 初始学习率 | $0.01$ |
| 学习率调度 | 逆平方根 |
| Warmup 步数 | $10{,}000$ |
| Dropout | $0.1$ |
| 权重衰减 | 无（Adafactor 自带） |
| 梯度裁剪 | 无 |

**训练稳定性技巧**：

1. **Pre-Norm (RMSNorm)**：归一化在子层之前，梯度流更稳定
2. **无位置嵌入**：Relative Position Bias 避免了位置嵌入的初始化敏感性
3. **Adafactor**：自适应学习率，减少超参数调节需求
4. **Span Corruption**：Decoder 处理短序列，减少内存峰值

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
    GELU 激活函数

    数学公式:
        GELU(x) ≈ 0.5x(1 + tanh[√(2/π)(x + 0.044715x³)])
    """
    return 0.5 * x * (1.0 + np.tanh(
        np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))
    ))


def rms_norm(x, gamma, eps=1e-6):
    """
    RMS Layer Normalization (T5 使用)

    数学公式:
        RMSNorm(x) = x / RMS(x) * γ
        RMS(x) = √(mean(x²))

    注意: 无偏移项 β，仅有缩放参数 γ
    """
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * gamma


def relative_position_bucket(relative_position, bidirectional=True,
                              num_buckets=32, max_distance=128):
    """
    T5 相对位置分桶函数

    将相对位置 (i-j) 映射到有限个桶索引

    参数:
        relative_position: 相对位置矩阵 (query_len, key_len)
        bidirectional: 是否双向 (Encoder=True, Decoder=False)
        num_buckets: 桶数量 (默认 32)
        max_distance: 最大相对距离 (默认 128)

    返回:
        buckets: 桶索引矩阵 (query_len, key_len)
    """
    ret = np.zeros_like(relative_position, dtype=np.int32)
    n = -relative_position

    if bidirectional:
        num_buckets //= 2
        # 正方向 (i > j) 和负方向 (i < j) 分别使用一半的桶
        ret += (n < 0).astype(np.int32) * num_buckets
        n = np.abs(n)
    else:
        n = np.maximum(n, 0)

    # 精确桶: 小距离直接映射
    max_exact = num_buckets // 2
    is_small = n < max_exact

    # 对数桶: 大距离用对数映射
    val_if_large = max_exact + (
        np.log(n.astype(np.float32) / max_exact + 1e-6)
        / np.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).astype(np.int32)
    val_if_large = np.minimum(val_if_large, num_buckets - 1)

    ret += np.where(is_small, n, val_if_large)
    return ret


def compute_relative_position_bias(query_len, key_len, bias_table,
                                    num_heads, bidirectional=True):
    """
    计算 Relative Position Bias

    参数:
        query_len: 查询序列长度
        key_len: 键序列长度
        bias_table: (num_buckets, num_heads) 可学习偏置参数
        num_heads: 注意力头数
        bidirectional: 是否双向

    返回:
        bias: (num_heads, query_len, key_len)
    """
    # 构建相对位置矩阵
    context_position = np.arange(query_len)[:, None]  # (q, 1)
    memory_position = np.arange(key_len)[None, :]      # (1, k)
    relative_position = memory_position - context_position  # (q, k)

    # 分桶
    buckets = relative_position_bucket(
        relative_position, bidirectional=bidirectional
    )  # (q, k)

    # 查表
    bias = bias_table[buckets]  # (q, k, num_heads)
    bias = bias.transpose(2, 0, 1)  # (num_heads, q, k)
    return bias


def causal_mask(seq_len):
    """
    生成因果掩码 (下三角矩阵)

    M[i,j] = 0 if j <= i else -inf
    """
    return np.triu(np.ones((seq_len, seq_len)) * (-1e9), k=1)


def multi_head_attention_numpy(Q, K, V, num_heads, mask=None,
                                position_bias=None):
    """
    多头注意力 (NumPy)

    数学公式:
        Attention(Q, K, V) = softmax(QK^T/√d_k + mask + bias) V

    参数:
        Q: (batch, seq_q, d_model)
        K: (batch, seq_k, d_model)
        V: (batch, seq_k, d_model)
        num_heads: 注意力头数
        mask: (seq_q, seq_k) 可选掩码
        position_bias: (num_heads, seq_q, seq_k) 可选位置偏置

    返回:
        output: (batch, seq_q, d_model)
        weights: (batch, num_heads, seq_q, seq_k)
    """
    batch, seq_q, d_model = Q.shape
    seq_k = K.shape[1]
    d_k = d_model // num_heads

    # 分割多头
    Q = Q.reshape(batch, seq_q, num_heads, d_k).transpose(0, 2, 1, 3)
    K = K.reshape(batch, seq_k, num_heads, d_k).transpose(0, 2, 1, 3)
    V = V.reshape(batch, seq_k, num_heads, d_k).transpose(0, 2, 1, 3)
    # (batch, heads, seq, d_k)

    # 注意力分数
    scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    # (batch, heads, seq_q, seq_k)

    # 添加因果掩码
    if mask is not None:
        scores = scores + mask  # 广播

    # 添加相对位置偏置
    if position_bias is not None:
        scores = scores + position_bias  # (num_heads, seq_q, seq_k) 广播

    weights = softmax(scores, axis=-1)
    output = np.matmul(weights, V)  # (batch, heads, seq_q, d_k)
    output = output.transpose(0, 2, 1, 3).reshape(batch, seq_q, d_model)

    return output, weights


def t5_encoder_block_numpy(x, params, num_heads, position_bias=None):
    """
    T5 Encoder 块 (Pre-Norm with RMSNorm)

    数学公式:
        a = x + SelfAttn(RMSNorm(x))
        output = a + FFN(RMSNorm(a))
    """
    batch, seq_len, d_model = x.shape

    # --- Self-Attention ---
    x_norm = rms_norm(x, params['ln1_g'])
    Q = np.dot(x_norm, params['Wq'])
    K = np.dot(x_norm, params['Wk'])
    V = np.dot(x_norm, params['Wv'])
    attn_out, attn_w = multi_head_attention_numpy(
        Q, K, V, num_heads, position_bias=position_bias
    )
    attn_out = np.dot(attn_out, params['Wo'])
    a = x + attn_out

    # --- FFN ---
    a_norm = rms_norm(a, params['ln2_g'])
    ffn_out = gelu(np.dot(a_norm, params['W1']))
    ffn_out = np.dot(ffn_out, params['W2'])
    output = a + ffn_out

    return output, attn_w


def t5_decoder_block_numpy(s, enc_output, params, num_heads,
                            causal_m, self_pos_bias=None,
                            cross_pos_bias=None):
    """
    T5 Decoder 块 (Pre-Norm with RMSNorm)

    数学公式:
        a = s + CausalSelfAttn(RMSNorm(s))
        b = a + CrossAttn(RMSNorm(a), enc_output)
        output = b + FFN(RMSNorm(b))
    """
    batch, seq_dec, d_model = s.shape

    # --- Causal Self-Attention ---
    s_norm = rms_norm(s, params['ln1_g'])
    Qs = np.dot(s_norm, params['Wqs'])
    Ks = np.dot(s_norm, params['Wks'])
    Vs = np.dot(s_norm, params['Wvs'])
    self_attn_out, self_attn_w = multi_head_attention_numpy(
        Qs, Ks, Vs, num_heads, mask=causal_m,
        position_bias=self_pos_bias
    )
    self_attn_out = np.dot(self_attn_out, params['Wos'])
    a = s + self_attn_out

    # --- Cross-Attention ---
    a_norm = rms_norm(a, params['ln2_g'])
    Qc = np.dot(a_norm, params['Wqc'])
    Kc = np.dot(enc_output, params['Wkc'])
    Vc = np.dot(enc_output, params['Wvc'])
    cross_attn_out, cross_attn_w = multi_head_attention_numpy(
        Qc, Kc, Vc, num_heads, position_bias=cross_pos_bias
    )
    cross_attn_out = np.dot(cross_attn_out, params['Woc'])
    b = a + cross_attn_out

    # --- FFN ---
    b_norm = rms_norm(b, params['ln3_g'])
    ffn_out = gelu(np.dot(b_norm, params['W1']))
    ffn_out = np.dot(ffn_out, params['W2'])
    output = b + ffn_out

    return output, self_attn_w, cross_attn_w


def span_corruption_numpy(tokens, noise_density=0.15, mean_span_length=3,
                           sentinel_start_id=32100):
    """
    Span Corruption 预训练目标 (NumPy)

    步骤:
        1. 随机选择 ~15% 的 token 进行遮蔽
        2. 合并连续遮蔽位置为 span
        3. 用哨兵 token 替换 span
        4. 构造 decoder 目标序列

    参数:
        tokens: (seq_len,) 原始 token 序列
        noise_density: 遮蔽比例 (默认 0.15)
        mean_span_length: 平均 span 长度 (默认 3)
        sentinel_start_id: 哨兵 token 起始 ID

    返回:
        encoder_input: 替换后的输入序列
        decoder_target: 解码器目标序列
    """
    n = len(tokens)
    # 每个位置独立以 noise_density 概率被遮蔽
    mask = np.random.random(n) < noise_density

    # 合并连续遮蔽位置为 span
    spans = []
    i = 0
    while i < n:
        if mask[i]:
            start = i
            while i < n and mask[i]:
                i += 1
            spans.append((start, i))  # [start, end)
        else:
            i += 1

    # 构造 encoder 输入: 用哨兵 token 替换 span
    encoder_input = []
    prev_end = 0
    for k, (start, end) in enumerate(spans):
        encoder_input.extend(tokens[prev_end:start].tolist())
        encoder_input.append(sentinel_start_id + k)
        prev_end = end
    encoder_input.extend(tokens[prev_end:].tolist())

    # 构造 decoder 目标: 哨兵 token + 被遮蔽内容
    decoder_target = []
    for k, (start, end) in enumerate(spans):
        decoder_target.append(sentinel_start_id + k)
        decoder_target.extend(tokens[start:end].tolist())
    decoder_target.append(sentinel_start_id + len(spans))  # 结束哨兵

    return np.array(encoder_input), np.array(decoder_target)


# ========== 测试 NumPy 实现 ==========
if __name__ == "__main__":
    np.random.seed(42)
    batch, seq_len, d_model, num_heads = 2, 16, 64, 4
    d_k = d_model // num_heads
    d_ff = d_model * 4

    # --- 测试 RMSNorm ---
    x = np.random.randn(batch, seq_len, d_model)
    gamma = np.ones(d_model)
    normed = rms_norm(x, gamma)
    rms_val = np.sqrt(np.mean(normed[0, 0] ** 2))
    print(f"RMSNorm 输出 RMS ≈ 1.0: {rms_val:.4f}")

    # --- 测试 Relative Position Bias ---
    bias_table = np.random.randn(32, num_heads) * 0.02
    bias = compute_relative_position_bias(
        seq_len, seq_len, bias_table, num_heads, bidirectional=True
    )
    print(f"Position Bias 形状: {bias.shape}")  # (heads, seq, seq)
    print(f"  bias[0,0,0] (距离0): {bias[0, 0, 0]:.4f}")
    print(f"  bias[0,0,5] (距离5): {bias[0, 0, 5]:.4f}")

    # --- 测试 Span Corruption ---
    tokens = np.arange(20) + 100  # 模拟 token IDs: 100-119
    enc_input, dec_target = span_corruption_numpy(tokens)
    print(f"\nSpan Corruption:")
    print(f"  原始长度: {len(tokens)}")
    print(f"  Encoder 输入: {enc_input}")
    print(f"  Decoder 目标: {dec_target}")

    # --- 测试 Encoder 块 ---
    enc_params = {
        'ln1_g': np.ones(d_model), 'ln2_g': np.ones(d_model),
        'Wq': np.random.randn(d_model, d_model) * 0.02,
        'Wk': np.random.randn(d_model, d_model) * 0.02,
        'Wv': np.random.randn(d_model, d_model) * 0.02,
        'Wo': np.random.randn(d_model, d_model) * 0.02,
        'W1': np.random.randn(d_model, d_ff) * 0.02,
        'W2': np.random.randn(d_ff, d_model) * 0.02,
    }
    enc_out, enc_attn = t5_encoder_block_numpy(
        x, enc_params, num_heads, position_bias=bias
    )
    print(f"\nEncoder 块输出形状: {enc_out.shape}")

    # --- 测试 Decoder 块 ---
    seq_dec = 8
    s = np.random.randn(batch, seq_dec, d_model) * 0.02
    cm = causal_mask(seq_dec)
    dec_self_bias = compute_relative_position_bias(
        seq_dec, seq_dec, bias_table, num_heads, bidirectional=False
    )
    dec_cross_bias = compute_relative_position_bias(
        seq_dec, seq_len, bias_table, num_heads, bidirectional=False
    )
    dec_params = {
        'ln1_g': np.ones(d_model), 'ln2_g': np.ones(d_model),
        'ln3_g': np.ones(d_model),
        'Wqs': np.random.randn(d_model, d_model) * 0.02,
        'Wks': np.random.randn(d_model, d_model) * 0.02,
        'Wvs': np.random.randn(d_model, d_model) * 0.02,
        'Wos': np.random.randn(d_model, d_model) * 0.02,
        'Wqc': np.random.randn(d_model, d_model) * 0.02,
        'Wkc': np.random.randn(d_model, d_model) * 0.02,
        'Wvc': np.random.randn(d_model, d_model) * 0.02,
        'Woc': np.random.randn(d_model, d_model) * 0.02,
        'W1': np.random.randn(d_model, d_ff) * 0.02,
        'W2': np.random.randn(d_ff, d_model) * 0.02,
    }
    dec_out, dec_self_w, dec_cross_w = t5_decoder_block_numpy(
        s, enc_out, dec_params, num_heads, cm,
        self_pos_bias=dec_self_bias, cross_pos_bias=dec_cross_bias
    )
    print(f"Decoder 块输出形状: {dec_out.shape}")
    print(f"Decoder 因果性验证 - self_attn[0,3] = 0: "
          f"{dec_self_w[0, 0, 0, 3] < 1e-6}")

    print("\n✅ T5 NumPy 核心组件测试通过！")
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
    T5 RMS Layer Normalization

    数学公式:
        RMSNorm(x) = x / RMS(x) * γ
        RMS(x) = √(mean(x²) + ε)

    注意: 无偏移项 β
    """
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class T5RelativePositionBias(nn.Module):
    """
    T5 Relative Position Bias

    数学公式:
        A_ij = q_i^T k_j / √d_k + b(i - j)

    使用分桶策略将相对距离映射到有限个桶
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
    def _relative_position_bucket(relative_position, bidirectional=True,
                                   num_buckets=32, max_distance=128):
        """将相对位置映射到桶索引"""
        ret = torch.zeros_like(relative_position, dtype=torch.long)
        n = -relative_position

        if bidirectional:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.clamp(n, min=0)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact + 1e-6)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.clamp(val_if_large, max=num_buckets - 1)

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, query_len: int, key_len: int,
                device: torch.device) -> torch.Tensor:
        """
        返回:
            bias: (1, num_heads, query_len, key_len)
        """
        context_pos = torch.arange(query_len, device=device)[:, None]
        memory_pos = torch.arange(key_len, device=device)[None, :]
        relative_pos = memory_pos - context_pos  # (q, k)

        buckets = self._relative_position_bucket(
            relative_pos, self.bidirectional,
            self.num_buckets, self.max_distance
        )
        bias = self.relative_attention_bias(buckets)  # (q, k, heads)
        bias = bias.permute(2, 0, 1).unsqueeze(0)     # (1, heads, q, k)
        return bias


class T5Attention(nn.Module):
    """
    T5 注意力层 (支持自注意力和交叉注意力)

    参数:
        d_model: 模型维度
        num_heads: 注意力头数
        is_cross_attention: 是否为交叉注意力
        dropout: dropout 概率
    """
    def __init__(self, d_model: int, num_heads: int,
                 is_cross_attention: bool = False,
                 dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.is_cross_attention = is_cross_attention

        # Q 投影 (总是来自当前层输入)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        # K, V 投影 (自注意力来自自身, 交叉注意力来自 Encoder)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor,
                key_value_states: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None,
                position_bias: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        参数:
            hidden_states: (batch, seq_q, d_model) 查询输入
            key_value_states: (batch, seq_k, d_model) 键值输入 (交叉注意力)
            mask: (seq_q, seq_k) 可选因果掩码
            position_bias: (1, heads, seq_q, seq_k) 相对位置偏置

        返回:
            output: (batch, seq_q, d_model)
            weights: (batch, heads, seq_q, seq_k)
        """
        B, T_q, _ = hidden_states.shape
        kv_input = key_value_states if self.is_cross_attention \
            else hidden_states

        Q = self.q_proj(hidden_states)
        K = self.k_proj(kv_input)
        V = self.v_proj(kv_input)

        T_k = K.size(1)

        # 分割多头
        Q = Q.view(B, T_q, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(B, T_k, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(B, T_k, self.num_heads, self.d_k).transpose(1, 2)

        # 注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 添加位置偏置
        if position_bias is not None:
            scores = scores + position_bias

        # 添加因果掩码
        if mask is not None:
            scores = scores + mask

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
        FFN(x) = GELU(xW_1) W_2
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.wi = nn.Linear(d_model, d_ff, bias=False)
        self.wo = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.wo(F.gelu(self.wi(x))))


class T5EncoderBlock(nn.Module):
    """
    T5 Encoder 块 (Pre-Norm)

    数学公式:
        a = x + SelfAttn(RMSNorm(x))
        output = a + FFN(RMSNorm(a))
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 dropout: float = 0.1):
        super().__init__()
        self.self_attn_norm = T5RMSNorm(d_model)
        self.self_attn = T5Attention(d_model, num_heads, dropout=dropout)
        self.ff_norm = T5RMSNorm(d_model)
        self.ff = T5FFN(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                position_bias: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-Attention
        normed = self.self_attn_norm(x)
        attn_out, attn_w = self.self_attn(normed,
                                           position_bias=position_bias)
        x = x + self.dropout(attn_out)

        # FFN
        normed = self.ff_norm(x)
        ff_out = self.ff(normed)
        x = x + self.dropout(ff_out)

        return x, attn_w


class T5DecoderBlock(nn.Module):
    """
    T5 Decoder 块 (Pre-Norm)

    数学公式:
        a = s + CausalSelfAttn(RMSNorm(s))
        b = a + CrossAttn(RMSNorm(a), enc_output)
        output = b + FFN(RMSNorm(b))
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 dropout: float = 0.1):
        super().__init__()
        self.self_attn_norm = T5RMSNorm(d_model)
        self.self_attn = T5Attention(d_model, num_heads, dropout=dropout)
        self.cross_attn_norm = T5RMSNorm(d_model)
        self.cross_attn = T5Attention(d_model, num_heads,
                                       is_cross_attention=True,
                                       dropout=dropout)
        self.ff_norm = T5RMSNorm(d_model)
        self.ff = T5FFN(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, s: torch.Tensor,
                enc_output: torch.Tensor,
                causal_mask: Optional[torch.Tensor] = None,
                self_pos_bias: Optional[torch.Tensor] = None,
                cross_pos_bias: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Causal Self-Attention
        normed = self.self_attn_norm(s)
        self_attn_out, self_attn_w = self.self_attn(
            normed, mask=causal_mask, position_bias=self_pos_bias
        )
        s = s + self.dropout(self_attn_out)

        # Cross-Attention
        normed = self.cross_attn_norm(s)
        cross_attn_out, cross_attn_w = self.cross_attn(
            normed, key_value_states=enc_output,
            position_bias=cross_pos_bias
        )
        s = s + self.dropout(cross_attn_out)

        # FFN
        normed = self.ff_norm(s)
        ff_out = self.ff(normed)
        s = s + self.dropout(ff_out)

        return s, self_attn_w, cross_attn_w


class T5Model(nn.Module):
    """
    完整 T5 模型 (Encoder-Decoder)

    结构:
        Input → [Embedding] → [Encoder × L] → [Decoder × L] → [LM Head]
    """
    def __init__(
        self,
        vocab_size: int = 32128,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model

        # 共享嵌入层 (无位置嵌入！)
        self.shared_embedding = nn.Embedding(vocab_size, d_model)

        # Encoder
        self.encoder_blocks = nn.ModuleList([
            T5EncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.encoder_final_norm = T5RMSNorm(d_model)

        # Encoder 相对位置偏置 (仅第一层计算，所有层共享)
        self.encoder_pos_bias = T5RelativePositionBias(
            num_heads, bidirectional=True
        )

        # Decoder
        self.decoder_blocks = nn.ModuleList([
            T5DecoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])
        self.decoder_final_norm = T5RMSNorm(d_model)

        # Decoder 相对位置偏置 (因果方向)
        self.decoder_self_pos_bias = T5RelativePositionBias(
            num_heads, bidirectional=False
        )
        self.decoder_cross_pos_bias = T5RelativePositionBias(
            num_heads, bidirectional=False
        )

        # LM Head (与嵌入共享权重)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.shared_embedding.weight

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        """T5 权重初始化: N(0, 1/√d)"""
        factor = 1.0 / math.sqrt(self.d_model)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=factor)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=1.0)
            elif isinstance(module, T5RMSNorm):
                nn.init.ones_(module.weight)

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Encoder 前向传播

        参数:
            input_ids: (batch, src_len) 输入 token 索引
        返回:
            enc_output: (batch, src_len, d_model)
        """
        x = self.dropout(self.shared_embedding(input_ids))

        # 计算位置偏置 (仅一次, 所有层共享)
        pos_bias = self.encoder_pos_bias(
            input_ids.size(1), input_ids.size(1), input_ids.device
        )

        for block in self.encoder_blocks:
            x, _ = block(x, position_bias=pos_bias)

        return self.encoder_final_norm(x)

    def decode(self, decoder_input_ids: torch.Tensor,
               enc_output: torch.Tensor) -> torch.Tensor:
        """
        Decoder 前向传播

        参数:
            decoder_input_ids: (batch, tgt_len) 目标 token 索引
            enc_output: (batch, src_len, d_model) Encoder 输出
        返回:
            dec_output: (batch, tgt_len, d_model)
        """
        tgt_len = decoder_input_ids.size(1)
        src_len = enc_output.size(1)
        device = decoder_input_ids.device

        s = self.dropout(self.shared_embedding(decoder_input_ids))

        # 因果掩码
        c_mask = torch.triu(
            torch.ones(tgt_len, tgt_len, device=device), diagonal=1
        ).bool()
        c_mask = c_mask.float().masked_fill(c_mask, float('-inf'))

        # 位置偏置
        self_bias = self.decoder_self_pos_bias(tgt_len, tgt_len, device)
        cross_bias = self.decoder_cross_pos_bias(tgt_len, src_len, device)

        for block in self.decoder_blocks:
            s, _, _ = block(s, enc_output, causal_mask=c_mask,
                            self_pos_bias=self_bias,
                            cross_pos_bias=cross_bias)

        return self.decoder_final_norm(s)

    def forward(self, input_ids: torch.Tensor,
                decoder_input_ids: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> dict:
        """
        完整前向传播

        参数:
            input_ids: (batch, src_len) Encoder 输入
            decoder_input_ids: (batch, tgt_len) Decoder 输入 (右移目标)
            labels: (batch, tgt_len) 目标标签
        返回:
            logits, loss
        """
        enc_output = self.encode(input_ids)
        dec_output = self.decode(decoder_input_ids, enc_output)
        logits = self.lm_head(dec_output)

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
                 num_beams: int = 1,
                 temperature: float = 1.0,
                 bos_token_id: int = 0,
                 eos_token_id: int = 1) -> torch.Tensor:
        """
        自回归生成 (贪心 / 采样)

        参数:
            input_ids: (batch, src_len) Encoder 输入
            max_new_tokens: 最大生成 token 数
            num_beams: beam 数 (1 = 贪心/采样)
            temperature: 温度参数
            bos_token_id: 起始 token ID
            eos_token_id: 结束 token ID
        """
        B = input_ids.size(0)
        device = input_ids.device

        enc_output = self.encode(input_ids)
        decoder_ids = torch.full((B, 1), bos_token_id,
                                  dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            dec_output = self.decode(decoder_ids, enc_output)
            logits = self.lm_head(dec_output[:, -1:, :]).squeeze(1)
            logits = logits / temperature

            if num_beams == 1:
                # 贪心或采样
                if temperature <= 0.01:
                    next_token = logits.argmax(dim=-1, keepdim=True)
                else:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            decoder_ids = torch.cat([decoder_ids, next_token], dim=1)

            if (next_token == eos_token_id).all():
                break

        return decoder_ids


# ========== 完整测试 ==========
if __name__ == "__main__":
    # 缩小版超参数
    vocab_size, d_model, num_heads = 1000, 128, 4
    num_layers, d_ff, max_len = 2, 512, 64
    batch_size, src_len, tgt_len = 4, 32, 16

    # 1. 创建模型
    model = T5Model(
        vocab_size=vocab_size, d_model=d_model, num_heads=num_heads,
        num_encoder_layers=num_layers, num_decoder_layers=num_layers,
        d_ff=d_ff, max_len=max_len
    )
    total_params = sum(p.numel() for p in model.parameters())
    unique_params = total_params - model.shared_embedding.weight.numel()
    print(f"总参数量: {total_params:,}")
    print(f"去重参数量 (共享嵌入): {unique_params:,}")

    # 2. 准备输入
    input_ids = torch.randint(0, vocab_size, (batch_size, src_len))
    # Teacher Forcing: decoder 输入 = 右移的目标
    target_ids = torch.randint(0, vocab_size, (batch_size, tgt_len))
    decoder_input_ids = torch.cat([
        torch.zeros(batch_size, 1, dtype=torch.long),  # BOS
        target_ids[:, :-1]
    ], dim=1)

    # 3. 前向传播 + 损失
    model.eval()
    with torch.no_grad():
        out = model(input_ids, decoder_input_ids, labels=target_ids)
    print(f"\nLogits 形状: {out['logits'].shape}")
    print(f"Loss: {out['loss'].item():.4f}")
    print(f"PPL:  {torch.exp(out['loss']).item():.2f}")

    # 4. 验证 Encoder 是双向的 (无因果掩码)
    enc_out = model.encode(input_ids)
    print(f"\nEncoder 输出形状: {enc_out.shape}")

    # 5. 训练一步
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    out = model(input_ids, decoder_input_ids, labels=target_ids)
    out["loss"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    opt.step()
    print(f"训练后 Loss: {out['loss'].item():.4f}")

    # 6. 文本生成测试
    model.eval()
    prompt = torch.randint(0, vocab_size, (1, 10))
    generated = model.generate(prompt, max_new_tokens=20, temperature=0.8)
    print(f"\n生成序列长度: {generated.shape[1]}")
    print(f"生成的 token IDs: {generated[0].tolist()}")

    # 7. 验证 Relative Position Bias
    bias_module = model.encoder_pos_bias
    bias = bias_module(8, 8, torch.device('cpu'))
    print(f"\nRelative Position Bias 形状: {bias.shape}")
    print(f"  对角线 (距离0): {bias[0, 0, 0, 0].item():.4f}")
    print(f"  相邻 (距离1):   {bias[0, 0, 0, 1].item():.4f}")
    print(f"  远距 (距离7):   {bias[0, 0, 0, 7].item():.4f}")

    print("\n✅ T5 模型测试通过！")
```

---

## 7. 多任务微调与推理

### 7.1 多任务训练策略

T5 在预训练后进行**多任务微调**，将所有下游任务混合在一起训练。

**混合策略**：

给定 $N$ 个任务 $\{\tau_1, \tau_2, \ldots, \tau_N\}$，每个任务有数据集 $\mathcal{D}_i$。混合训练的目标：

$$
\mathcal{L}_{\text{multi}}(\theta) = \sum_{i=1}^{N} \lambda_i \cdot \mathcal{L}_i(\theta)
$$

其中 $\lambda_i$ 是任务 $\tau_i$ 的采样权重。

**T5 的采样策略**：

每个 batch 从某个任务中采样，任务的采样概率按数据集大小的幂次加权：

$$
\boxed{P(\tau_i) = \frac{\min(|\mathcal{D}_i|, K)}{\sum_{j=1}^{N} \min(|\mathcal{D}_j|, K)}}
$$

其中 $K$ 是人工设定的上限（T5 使用 $K = 2^{19} = 524{,}288$），防止大数据集过度主导。

### 7.2 温度采样混合比例

T5 论文还探索了**温度采样**来控制任务混合比例：

$$
P(\tau_i) \propto |\mathcal{D}_i|^{1/T}
$$

| 温度 $T$ | 效果 |
|----------|------|
| $T = 1$ | 按数据集大小成比例采样（大任务主导） |
| $T \to \infty$ | 均匀采样（等概率） |
| $T \to 0$ | 只采样最大的数据集 |

T5 发现 $K$-截断策略（等价于某种温度设置）在实践中效果最好。

### 7.3 Beam Search 解码

T5 在推理时使用 **Beam Search** 进行解码：

**算法**：

维护 $B$ 个候选序列（beams），每步扩展所有候选：

$$
\text{score}(\mathbf{y}_{1:t}) = \sum_{i=1}^{t} \log P(y_i \mid y_{<i}, \mathbf{x}; \theta)
$$

**长度归一化**：

$$
\boxed{\text{score}_{\text{norm}}(\mathbf{y}) = \frac{1}{|\mathbf{y}|^\alpha} \sum_{t=1}^{|\mathbf{y}|} \log P(y_t \mid y_{<t}, \mathbf{x})}
$$

其中 $\alpha \in [0.6, 1.0]$ 是长度惩罚因子。$\alpha = 0$ 退化为标准 Beam Search，$\alpha = 1$ 等价于平均对数概率。

**T5 的默认解码配置**：

| 参数 | 值 |
|------|-----|
| Beam width $B$ | 4 |
| 长度惩罚 $\alpha$ | 0.6 |
| 最大生成长度 | 任务相关 |

> **Q:** 为什么 T5 用 Beam Search 而不是 Top-p 采样？
>
> **A:** T5 的大部分下游任务（翻译、摘要、分类）有**确定性的正确答案**，不需要多样性。Beam Search 更适合这类任务。对于开放式生成（如故事续写），Top-p 采样更好。

---

## 8. 与其他模型的关系

### 8.1 BERT vs GPT-2 vs T5：三种范式

| 维度 | BERT | GPT-2 | T5 |
|------|:----:|:-----:|:--:|
| **架构** | Encoder-only | Decoder-only | Encoder-Decoder |
| **注意力** | 双向 | 因果（单向） | Enc 双向 + Dec 因果 |
| **预训练** | MLM + NSP | 自回归 LM | Span Corruption |
| **位置编码** | 可学习绝对 | 可学习绝对 | Relative Position Bias |
| **LayerNorm** | Post-Norm | Pre-Norm | Pre-Norm (RMSNorm) |
| **下游任务** | 加任务头微调 | Zero-shot 提示 | Text-to-Text 微调 |
| **输出格式** | 任务相关 | 文本 | **统一文本** |

**三种范式的数学对比**：

$$
\text{BERT: } P(x_t \mid x_{\backslash t}) \quad \text{（完形填空，双向）}
$$

$$
\text{GPT-2: } P(x_t \mid x_{<t}) \quad \text{（预测下一个，单向）}
$$

$$
\text{T5: } P(\mathbf{y} \mid \mathbf{x}) = \prod_{t=1}^{m} P(y_t \mid y_{<t}, \mathbf{x}) \quad \text{（序列到序列，双向编码+因果解码）}
$$

**统一的演进方向**：

$$
\boxed{\underbrace{\text{BERT}}_{\text{理解}} + \underbrace{\text{GPT-2}}_{\text{生成}} \longrightarrow \underbrace{\text{T5}}_{\text{统一理解+生成}}}
$$

### 8.2 T5 的系统性消融实验

T5 论文的另一大贡献是**对迁移学习的系统性探索**。论文对比了以下维度：

**架构对比**（固定参数量）：

| 架构 | GLUE 平均 | 说明 |
|------|:---------:|------|
| Encoder-Decoder | **83.28** | T5 选择 |
| Decoder-only (LM) | 80.92 | GPT 风格 |
| Prefix LM | 82.19 | 混合风格 |

**预训练目标对比**：

| 目标 | GLUE 平均 | 说明 |
|------|:---------:|------|
| Span Corruption (15%) | **83.28** | T5 选择 |
| BERT-style MLM | 82.76 | 逐 token 遮蔽 |
| Language Model | 81.54 | 自回归 |
| Deshuffling | 80.12 | 打乱恢复 |

**Span 长度对比**：

| 平均 Span 长度 | GLUE 平均 | 效率 |
|:--------------:|:---------:|:----:|
| 2 | 82.95 | 中 |
| 3 | **83.28** | 高 |
| 5 | 83.10 | 最高 |
| 10 | 82.56 | 最高 |

**关键发现**：

$$
\boxed{\text{Encoder-Decoder + Span Corruption (15\%, span=3)} = \text{最优配置}}
$$

### 8.3 T5 的后续影响

```
T5 (2019) ── 统一 Text-to-Text 框架
  ├── mT5 (2020) ── 多语言 T5 (101 种语言)
  ├── T5 v1.1 (2020) ── 架构改进 (GeGLU, 无 Dropout 预训练)
  ├── FLAN-T5 (2022) ── 指令微调的 T5
  │    └── 证明指令微调大幅提升 Zero-shot 能力
  ├── UL2 (2022) ── 统一语言学习器
  │    └── 混合多种去噪目标
  └── 对后续模型的影响:
       ├── GPT-3: 沿用了 Text-to-Text 的统一思想
       ├── PaLM: 采用了 T5 的相对位置编码
       └── LLaMA: 采用了 RMSNorm + 无偏置设计
```

**T5 在历史中的定位**：

T5 的核心贡献不仅是一个模型，更是一个**实验框架**：

1. **统一框架**：证明所有 NLP 任务可以统一为 text-to-text
2. **系统性对比**：在同一框架下公平对比了几乎所有主流设计选择
3. **工程影响**：RMSNorm、Relative Position Bias、SentencePiece 成为后续模型的标配
4. **规模效应**：从 60M 到 11B 的规模对比，为 Scaling Laws 研究铺路

---

## 扩展阅读与实现

### 问题 1：为什么 T5 用 Encoder-Decoder 而不是 Decoder-only？

**解答**：

T5 论文通过实验发现，在**相同参数量**下：

$$
\text{Enc-Dec (2L layers total)} > \text{Dec-only (L layers)} \approx \text{Prefix LM (L layers)}
$$

**原因分析**：

1. **参数利用率**：Encoder-Decoder 中，Encoder 和 Decoder 各有 $L$ 层，总参数量约 $2L$ 层。但 Encoder 处理输入、Decoder 处理输出，参数"各司其职"。

2. **注意力模式**：Encoder 的**双向注意力**能更好地理解输入语义。Decoder-only 的因果掩码限制了对输入的理解能力。

3. **计算效率**：Span Corruption 使得 Decoder 只需处理短序列（~20% 输入长度），大部分计算在高效的 Encoder 中完成。

**但有一个重要注意**：当参数量非常大时（>100B），Decoder-only 架构（如 GPT-3、PaLM）逐渐追平甚至超过 Encoder-Decoder。这可能是因为超大模型的 Decoder 已经有足够容量同时处理理解和生成。

### 问题 2：Relative Position Bias vs 绝对位置嵌入 vs 旋转位置编码

**解答**：

| 方法 | 模型 | 长度泛化 | 参数量 | 计算开销 |
|------|------|:-------:|:------:|:-------:|
| 正弦位置编码 | Transformer | 中等 | 0 | 低 |
| 可学习绝对位置 | BERT, GPT-2 | 差 | $n \cdot d$ | 低 |
| Relative Position Bias | T5 | **好** | $n_{\text{buckets}} \cdot A$ | 中 |
| ALiBi | BLOOM | **好** | 0 | 低 |
| RoPE | LLaMA, GPT-NeoX | **最好** | 0 | 中 |

T5 的 Relative Position Bias 的关键优势：

1. **对数分桶**使模型能处理训练时未见过的长距离
2. **每头独立**的偏置参数让不同头学习不同的位置模式
3. **仅在第一层计算**，然后共享给所有层，计算效率较高

### 问题 3：Span Corruption 的 span 长度如何影响性能？

**解答**：

设原始序列长度 $n$，噪声密度 $\rho$，平均 span 长度 $\mu$：

- Span 数量：$K = \lfloor \rho \cdot n / \mu \rfloor$
- Decoder 目标长度：$m \approx \rho \cdot n + K$

当 $\mu$ 增大时：
- $K$ 减小 → Decoder 目标更短 → **计算更高效**
- 但每个 span 更长 → 任务更难（需要生成更多连续 token）
- 极端情况 $\mu \to n$：退化为自回归 LM（Decoder 处理整个序列）

T5 发现 $\mu = 3$ 是效率和性能的最佳平衡点：

$$
\boxed{\text{span length} = 3: \quad \text{性能最优，效率较高}}
$$

### 问题 4：Teacher Forcing 的 Exposure Bias 问题

**解答**：

Teacher Forcing 训练时，Decoder 在位置 $t$ 看到的是真实标签 $y_{<t}^*$。但推理时看到的是模型自身的预测 $\hat{y}_{<t}$。这种训练/推理不一致称为 **Exposure Bias**。

**影响**：

$$
\underbrace{P(y_t \mid y_{<t}^*, \mathbf{x})}_{\text{训练时：完美前缀}} \neq \underbrace{P(y_t \mid \hat{y}_{<t}, \mathbf{x})}_{\text{推理时：可能有错的前缀}}
$$

如果模型在推理时早期犯了一个错误，后续所有预测都基于这个错误的前缀，导致错误累积。

**T5 的缓解策略**：

1. **Beam Search**：维护多个候选，减少单次错误的影响
2. **大规模预训练**：模型在预训练中见过大量文本，对各种前缀都有较好的鲁棒性
3. **Span Corruption**：目标序列较短，Exposure Bias 的影响范围有限

### 问题 5：T5 的 SentencePiece vs GPT-2 的 Byte-level BPE

**解答**：

| 特性 | SentencePiece (T5) | Byte-level BPE (GPT-2) |
|------|:------------------:|:---------------------:|
| 基本单元 | Unicode 字符 | UTF-8 字节 |
| 分词方式 | Unigram LM | BPE 合并 |
| 预分词 | 不需要 | 需要正则分割 |
| 词表大小 | 32,128 | 50,257 |
| OOV 处理 | 回退到字符 | 回退到字节 |
| 多语言 | ✅ 原生支持 | ✅ 通过字节 |
| 确定性 | ✅ 唯一分词 | ✅ 唯一分词 |

SentencePiece 的 Unigram 模型为每个子词分配概率，选择**最大似然分词**：

$$
\mathbf{x}^* = \arg\max_{\mathbf{x} \in S(\text{input})} \sum_{i=1}^{|\mathbf{x}|} \log P(x_i)
$$

其中 $S(\text{input})$ 是所有可能的分词方式集合。

---

## 参考资源

### 经典论文

1. Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., & Liu, P. J. (2020). [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683). JMLR, 21(140), 1-67.
   - **贡献**：提出 T5 和统一 Text-to-Text 框架，系统性对比迁移学习设计选择

2. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). NeurIPS 2017.
   - **贡献**：提出 Transformer 架构，T5 的基础

3. Shazeer, N., & Stern, M. (2018). [Adafactor: Adaptive Learning Rates with Sublinear Memory Cost](https://arxiv.org/abs/1804.04235). ICML 2018.
   - **贡献**：提出 Adafactor 优化器，T5 预训练使用

4. Shaw, P., Uszkoreit, J., & Vaswani, A. (2018). [Self-Attention with Relative Position Representations](https://arxiv.org/abs/1803.02155). NAACL 2018.
   - **贡献**：提出相对位置编码，T5 的 Relative Position Bias 基于此改进

5. Zhang, B., & Sennrich, R. (2019). [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467). NeurIPS 2019.
   - **贡献**：提出 RMSNorm，T5 采用的归一化方法

### 教材与书籍

6. Jurafsky, D., & Martin, J. H. [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/). 3rd ed. (Draft).
   - **章节**：第 11 章详细讲解 Encoder-Decoder 模型与序列到序列学习

### 在线资源与教程

7. Google Research. [T5 GitHub Repository](https://github.com/google-research/text-to-text-transfer-transformer).
   - **内容**：T5 的官方实现（基于 Mesh TensorFlow）

8. Hugging Face. [T5 Documentation](https://huggingface.co/docs/transformers/model_doc/t5).
   - **内容**：T5 的 PyTorch 实现和使用指南

9. Alammar, J. [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/).
   - **内容**：Transformer Encoder-Decoder 架构的直观图解，有助于理解 T5 的基础架构

---

## 附录：符号表

| 符号 | 含义 | 维度/类型 |
|------|------|----------|
| $n$ | Encoder 输入序列长度 | 标量 |
| $m$ | Decoder 目标序列长度 | 标量 |
| $d$ | 隐藏维度（$d_{\text{model}}$） | 标量，T5-Base: 768 |
| $d_k$ | 每个注意力头的维度 | 标量，64 |
| $d_{ff}$ | FFN 隐藏层维度 | 标量，T5-Base: 3072 |
| $L_{\text{enc}}$ | Encoder 层数 | 标量，T5-Base: 12 |
| $L_{\text{dec}}$ | Decoder 层数 | 标量，T5-Base: 12 |
| $A$ | 注意力头数 | 标量，T5-Base: 12 |
| $\|V\|$ | SentencePiece 词表大小 | 标量，32,128 |
| $\mathbf{x}$ | Encoder 输入序列 | $(n,)$ token 索引 |
| $\mathbf{y}$ | Decoder 目标序列 | $(m,)$ token 索引 |
| $h^{(l)}$ | Encoder 第 $l$ 层隐藏状态 | $(n, d)$ |
| $s^{(l)}$ | Decoder 第 $l$ 层隐藏状态 | $(m, d)$ |
| $\mathbf{H}^{\text{enc}}$ | Encoder 最终输出 | $(n, d)$ |
| $Q, K, V$ | 查询、键、值矩阵 | 依上下文而定 |
| $Q_s, K_s, V_s$ | Decoder 自注意力的 Q/K/V | $(m, d_k)$ |
| $Q_s, K_e, V_e$ | 交叉注意力的 Q/K/V | $Q_s: (m, d_k)$, $K_e, V_e: (n, d_k)$ |
| $b(i-j)$ | 相对位置偏置函数 | $\mathbb{Z} \to \mathbb{R}$ |
| $B^{\text{rel}}$ | 相对位置偏置矩阵 | $(A, n, n)$ 或 $(A, m, m)$ |
| $M^{\text{causal}}$ | 因果注意力掩码 | $(m, m)$，上三角为 $-\infty$ |
| $\rho$ | Span Corruption 噪声密度 | 标量，0.15 |
| $\mu$ | 平均 span 长度 | 标量，3 |
| $K$ | span 数量 | 标量 |
| $\langle\text{extra\_id\_}k\rangle$ | 第 $k$ 个哨兵 token | 特殊 token |
| $\text{prefix}(\tau)$ | 任务 $\tau$ 的文本前缀 | 字符串 |
| $\mathcal{L}$ | 序列到序列损失（交叉熵） | 标量 |
| $\ell(\cdot, \cdot)$ | 交叉熵损失函数 | 函数 |
| $\alpha$ | Beam Search 长度惩罚 | 标量，0.6 |
| $B$ | Beam Search 的 beam width | 标量 |
| $\gamma$ | RMSNorm 缩放参数 | $(d,)$ |

**典型维度示例（T5-Base）：**
- $d = 768$（隐藏维度）
- $d_k = 64$（每头维度）
- $d_{ff} = 3072$（FFN 维度）
- $|V| = 32{,}128$（词表大小）
- $L_{\text{enc}} = L_{\text{dec}} = 12$（Encoder/Decoder 层数）

---

最后更新：2026-03-19
