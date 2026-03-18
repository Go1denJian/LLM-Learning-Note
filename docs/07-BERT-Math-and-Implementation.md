# BERT 数学原理与实现 —— 双向编码的预训练革命

> **前置知识**：Transformer Encoder 架构、注意力机制、交叉熵损失、Python 基础  
> **与前面内容的联系**：建议先学习 [Transformer-Math-and-Implementation](./06-Transformer-Math-and-Implementation.md)，理解自注意力与编码器架构  
> **与后续内容的联系**：BERT 开启了"预训练 + 微调"范式，是理解 GPT-2、T5 等后续模型的关键

---

## 目录

1. [引言：为什么需要 BERT？](#1-引言为什么需要-bert)
   - 1.1 [从单向到双向：语言模型的局限](#11-从单向到双向语言模型的局限)
   - 1.2 [预训练 + 微调范式的诞生](#12-预训练--微调范式的诞生)
   - 1.3 [本科数学知识映射表](#13-本科数学知识映射表)
2. [基础概念：BERT 的核心思想](#2-基础概念bert-的核心思想)
   - 2.1 [双向编码 vs 单向编码](#21-双向编码-vs-单向编码)
   - 2.2 [输入表示：三重嵌入](#22-输入表示三重嵌入)
   - 2.3 [特殊标记的作用](#23-特殊标记的作用)
3. [核心算法：双向 Attention 与预训练任务](#3-核心算法双向-attention-与预训练任务)
   - 3.1 [双向自注意力的掩码机制](#31-双向自注意力的掩码机制)
   - 3.2 [Masked Language Model (MLM)](#32-masked-language-model-mlm)
   - 3.3 [Next Sentence Prediction (NSP)](#33-next-sentence-prediction-nsp)
4. [梯度推导与参数更新](#4-梯度推导与参数更新)
   - 4.1 [MLM 损失的梯度推导](#41-mlm-损失的梯度推导)
   - 4.2 [NSP 损失的梯度推导](#42-nsp-损失的梯度推导)
   - 4.3 [联合损失与多任务学习](#43-联合损失与多任务学习)
5. [训练优化方法总结](#5-训练优化方法总结)
   - 5.1 [AdamW 优化器](#51-adamw-优化器)
   - 5.2 [学习率预热与线性衰减](#52-学习率预热与线性衰减)
   - 5.3 [大批量训练策略](#53-大批量训练策略)
6. [从数学到代码：完整实现](#6-从数学到代码完整实现)
   - 6.1 [NumPy 实现核心组件](#61-numpy-实现核心组件)
   - 6.2 [PyTorch 完整实现](#62-pytorch-完整实现)
7. [实践技巧与可视化](#7-实践技巧与可视化)
   - 7.1 [注意力可视化](#71-注意力可视化)
   - 7.2 [微调最佳实践](#72-微调最佳实践)
8. [与其他模型的关系](#8-与其他模型的关系)
   - 8.1 [BERT vs GPT：编码器 vs 解码器](#81-bert-vs-gpt编码器-vs-解码器)
   - 8.2 [BERT 的后继者们](#82-bert-的后继者们)
9. [扩展阅读与实现](#扩展阅读与实现)
   - 9.1 [Whole Word Masking 的数学分析](#91-whole-word-masking-的数学分析)
   - 9.2 [为什么 NSP 后来被质疑？](#92-为什么-nsp-后来被质疑)
   - 9.3 [BERT 的参数效率分析](#93-bert-的参数效率分析)
10. [参考资源](#参考资源)

[附录：符号表](#附录符号表)

---

## 1. 引言：为什么需要 BERT？

### 1.1 从单向到双向：语言模型的局限

在 BERT 之前，主流的语言模型（如 ELMo、GPT-1）都基于**单向**建模：

**单向语言模型（GPT 风格）**：

$$
P(w_1, w_2, \ldots, w_n) = \prod_{t=1}^n P(w_t \mid w_1, \ldots, w_{t-1})
$$

每个词只能看到**左侧上下文**。

**双向语言模型（ELMo 风格）**：

ELMo 使用两个独立的单向 LSTM 拼接：

$$
\overrightarrow{h}_t = \text{LSTM}_{\text{forward}}(w_t, \overrightarrow{h}_{t-1})
$$
$$
\overleftarrow{h}_t = \text{LSTM}_{\text{backward}}(w_t, \overleftarrow{h}_{t+1})
$$
$$
h_t^{\text{ELMo}} = [\overrightarrow{h}_t; \overleftarrow{h}_t]
$$

**ELMo 的问题**：左右方向是**独立训练**的，然后简单拼接——并非真正的双向交互。

**BERT 的突破**：

> **通过 Masked Language Model，让每个位置同时看到左右两侧的全部上下文**

$$
P(w_t \mid w_1, \ldots, w_{t-1}, w_{t+1}, \ldots, w_n) \quad \text{（真正的双向条件概率）}
$$

**对比总结**：

| 模型 | 方向 | 上下文利用 | 训练信号 |
|------|------|-----------|---------|
| GPT | 单向（左→右） | 仅左侧 | 自回归 LM |
| ELMo | 伪双向（拼接） | 左右独立 | 双向 LM |
| **BERT** | **真双向** | **左右交互** | **MLM + NSP** |

### 1.2 预训练 + 微调范式的诞生

BERT 正式确立了 NLP 的"两阶段"范式：

**阶段 1：预训练（Pre-training）**

在大规模无标注语料上学习通用语言表示：

$$
\theta^* = \arg\min_\theta \mathcal{L}_{\text{pretrain}}(\theta) = \arg\min_\theta \left(\mathcal{L}_{\text{MLM}} + \mathcal{L}_{\text{NSP}}\right)
$$

**阶段 2：微调（Fine-tuning）**

在下游任务的标注数据上微调所有参数：

$$
\theta^{**} = \arg\min_\theta \mathcal{L}_{\text{task}}(\theta; \theta^*)
$$

**关键洞察**：预训练的参数 $\theta^*$ 提供了良好的初始化，使得微调只需少量标注数据即可达到优异性能。

### 1.3 本科数学知识映射表

| 数学概念 | BERT 中的应用 | 代码对应 |
|---------|-------------|---------|
| 矩阵乘法 | 自注意力计算 | `torch.matmul(Q, K.T)` |
| Softmax | 注意力权重归一化 | `F.softmax(scores, dim=-1)` |
| 交叉熵 | MLM/NSP 损失函数 | `F.cross_entropy()` |
| 概率论（条件概率） | 双向语言建模 | MLM 预测被掩码的词 |
| 伯努利分布 | 掩码采样策略 | `torch.bernoulli()` |
| 二分类交叉熵 | NSP 损失 | `F.binary_cross_entropy()` |
| 向量拼接 | 三重嵌入求和 | `token_emb + seg_emb + pos_emb` |

---

## 2. 基础概念：BERT 的核心思想

### 2.1 双向编码 vs 单向编码

**数学定义**：

给定输入序列 $\mathbf{x} = (x_1, x_2, \ldots, x_n)$，BERT Encoder 的第 $l$ 层输出为：

$$
H^{(l)} = \text{TransformerBlock}(H^{(l-1)}) \in \mathbb{R}^{n \times d}
$$

其中每个 $h_t^{(l)}$（第 $t$ 个位置的隐藏向量）依赖于**所有位置**的输入：

$$
h_t^{(l)} = f(h_1^{(l-1)}, h_2^{(l-1)}, \ldots, h_n^{(l-1)})
$$

**与单向模型的关键区别**：

在 GPT（单向解码器）中，位置 $t$ 只能看到 $\{1, 2, \ldots, t\}$：

$$
h_t^{(l)} = f(h_1^{(l-1)}, h_2^{(l-1)}, \ldots, h_t^{(l-1)}) \quad \text{（GPT：因果掩码）}
$$

而在 BERT（双向编码器）中，位置 $t$ 可以看到**所有**位置 $\{1, 2, \ldots, n\}$：

$$
\boxed{h_t^{(l)} = f(h_1^{(l-1)}, h_2^{(l-1)}, \ldots, h_n^{(l-1)}) \quad \text{（BERT：无因果掩码）}}
$$

### 2.2 输入表示：三重嵌入

BERT 的输入由三个嵌入相加构成：

$$
\boxed{E_{\text{input}} = E_{\text{token}} + E_{\text{segment}} + E_{\text{position}}}
$$

**Token Embedding** $E_{\text{token}} \in \mathbb{R}^{|V| \times d}$：

将每个 WordPiece token 映射为 $d$ 维向量：
$$
e_{\text{token}}(x_t) = E_{\text{token}}[x_t] \in \mathbb{R}^d
$$

**Segment Embedding** $E_{\text{segment}} \in \mathbb{R}^{2 \times d}$：

区分句子 A 和句子 B（仅两个可学习向量）：
$$
e_{\text{segment}}(x_t) = \begin{cases}
E_{\text{segment}}[0] & \text{if } x_t \in \text{Sentence A} \\
E_{\text{segment}}[1] & \text{if } x_t \in \text{Sentence B}
\end{cases}
$$

**Position Embedding** $E_{\text{position}} \in \mathbb{R}^{L_{\max} \times d}$：

BERT 使用**可学习**的位置嵌入（不同于 Transformer 原文的正弦编码）：
$$
e_{\text{position}}(t) = E_{\text{position}}[t] \in \mathbb{R}^d, \quad t \in \{0, 1, \ldots, L_{\max}-1\}
$$

**完整输入**（位置 $t$）：
$$
h_t^{(0)} = e_{\text{token}}(x_t) + e_{\text{segment}}(x_t) + e_{\text{position}}(t)
$$

### 2.3 特殊标记的作用

BERT 引入两个特殊标记：

**[CLS]**（Classification Token）：
- 放在序列开头（位置 0）
- 经过多层 Transformer 后，其隐藏向量 $h_{\text{[CLS]}}^{(L)}$ 用作**句子级表示**
- 用于 NSP 分类和下游分类任务

**[SEP]**（Separator Token）：
- 放在每个句子末尾，分隔句子 A 和 B
- 帮助模型识别句子边界

**输入格式示例**：

```
[CLS] I love NLP [SEP] It is fascinating [SEP]
  0    1   2   3    4    5  6      7        8    ← 位置编号
  A    A   A   A    A    B  B      B        B    ← Segment ID
```

---

## 3. 核心算法：双向 Attention 与预训练任务

### 3.1 双向自注意力的掩码机制

**Transformer Encoder 的自注意力**：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V
$$

其中 $M \in \mathbb{R}^{n \times n}$ 是**注意力掩码矩阵**。

**BERT（双向）的掩码**：

$$
M_{ij}^{\text{BERT}} = \begin{cases}
0 & \text{if position } j \text{ is valid (non-padding)} \\
-\infty & \text{if position } j \text{ is padding}
\end{cases}
$$

**关键区别**：BERT 只需处理 padding 掩码，**不需要因果掩码**。

$$
\boxed{M^{\text{BERT}} = \begin{bmatrix}
0 & 0 & 0 & -\infty \\
0 & 0 & 0 & -\infty \\
0 & 0 & 0 & -\infty \\
0 & 0 & 0 & -\infty
\end{bmatrix} \quad \text{（最后一位是 padding）}}
$$

**对比 GPT（单向）的因果掩码**：

$$
M^{\text{GPT}} = \begin{bmatrix}
0 & -\infty & -\infty & -\infty \\
0 & 0 & -\infty & -\infty \\
0 & 0 & 0 & -\infty \\
0 & 0 & 0 & 0
\end{bmatrix} \quad \text{（上三角为 $-\infty$）}
$$

**Softmax 后的注意力权重**：

BERT 中，对于非 padding 位置：
$$
\alpha_{ij}^{\text{BERT}} = \frac{\exp(q_i^\top k_j / \sqrt{d_k})}{\sum_{k \in \text{valid}} \exp(q_i^\top k_k / \sqrt{d_k})} > 0 \quad \forall j \in \text{valid positions}
$$

每个位置都能关注到所有其他有效位置，实现了**真正的双向交互**。

### 3.2 Masked Language Model (MLM)

**动机**：直接使用双向编码器训练传统语言模型会导致"信息泄露"——每个词能通过多层注意力间接"看到自己"。

**解决方案**：随机掩码部分输入 token，让模型预测被掩码的原始 token。

#### 3.2.1 掩码策略

对于输入序列中的每个 token $x_t$，以概率 $p = 0.15$ 被选中进行掩码。选中后：

$$
\tilde{x}_t = \begin{cases}
\text{[MASK]} & \text{with probability } 0.80 \\
x_{\text{random}} & \text{with probability } 0.10 \\
x_t & \text{with probability } 0.10
\end{cases}
$$

**设计理由**：

| 策略 | 比例 | 目的 |
|------|------|------|
| 替换为 [MASK] | 80% | 主要学习信号 |
| 替换为随机词 | 10% | 增加鲁棒性，防止模型只在 [MASK] 位置预测 |
| 保持不变 | 10% | 减少预训练与微调的不匹配 |

#### 3.2.2 MLM 损失函数

设被掩码位置的集合为 $\mathcal{M}$，原始 token 为 $x_t$，BERT 在该位置的输出隐藏向量为 $h_t^{(L)}$。

**预测分布**：

$$
P(x_t = w \mid \tilde{\mathbf{x}}) = \text{softmax}(W_{\text{MLM}} h_t^{(L)} + b_{\text{MLM}})_w
$$

其中 $W_{\text{MLM}} \in \mathbb{R}^{|V| \times d}$，$b_{\text{MLM}} \in \mathbb{R}^{|V|}$。

> **注意**：在原始 BERT 实现中，$W_{\text{MLM}}$ 与 Token Embedding 矩阵 $E_{\text{token}}$ 共享权重（weight tying），并在 softmax 之前通过一个额外的线性变换 + GELU 激活 + LayerNorm：
>
> $$h_t' = \text{LayerNorm}(\text{GELU}(W_{\text{proj}} h_t^{(L)} + b_{\text{proj}}))$$
> $$P(x_t \mid \tilde{\mathbf{x}}) = \text{softmax}(E_{\text{token}} \cdot h_t' + b_{\text{MLM}})$$

**MLM 损失**（交叉熵）：

$$
\boxed{\mathcal{L}_{\text{MLM}} = -\frac{1}{|\mathcal{M}|} \sum_{t \in \mathcal{M}} \log P(x_t \mid \tilde{\mathbf{x}})}
$$

**展开**：

$$
\mathcal{L}_{\text{MLM}} = -\frac{1}{|\mathcal{M}|} \sum_{t \in \mathcal{M}} \left[ \log \frac{\exp(w_{x_t}^\top h_t^{(L)} + b_{x_t})}{\sum_{w=1}^{|V|} \exp(w_w^\top h_t^{(L)} + b_w)} \right]
$$

其中 $w_{x_t}$ 是 $W_{\text{MLM}}$ 的第 $x_t$ 行。

#### 3.2.3 掩码比例的数学分析

**为什么选择 15%？**

设序列长度为 $n$，掩码比例为 $p$：

- 每个样本的有效训练信号：$p \cdot n$ 个 token
- 过高的 $p$：破坏上下文信息，模型难以理解语义
- 过低的 $p$：训练信号稀疏，收敛缓慢

**收敛速度分析**：

每个 batch 的有效梯度更新量 $\propto p \cdot n \cdot B$（$B$ 为 batch size）。

相比自回归模型（每个 token 都有训练信号），MLM 需要约 $1/p \approx 6.67$ 倍的训练步数才能获得等量的训练信号。

> **Q:** 为什么不直接把所有 token 都掩码？
>
> **A:** 如果掩码 100%，模型无法获取任何上下文信息，退化为独立预测每个词——失去了双向建模的意义。15% 是实验中发现的最优平衡点。

### 3.3 Next Sentence Prediction (NSP)

#### 3.3.1 任务定义

给定句子对 $(A, B)$，判断 $B$ 是否是 $A$ 的**下一句**：

$$
P(\text{IsNext} \mid A, B) = \sigma(w_{\text{NSP}}^\top h_{\text{[CLS]}}^{(L)} + b_{\text{NSP}})
$$

其中 $\sigma$ 是 sigmoid 函数，$h_{\text{[CLS]}}^{(L)}$ 是 [CLS] 标记的最终隐藏向量。

**训练数据构造**：

| 类别 | 比例 | 构造方式 |
|------|------|---------|
| IsNext（正例） | 50% | $B$ 是 $A$ 的真实下一句 |
| NotNext（负例） | 50% | $B$ 是从语料库随机采样的句子 |

#### 3.3.2 NSP 损失函数

使用二分类交叉熵：

$$
\boxed{\mathcal{L}_{\text{NSP}} = -\left[y \log P(\text{IsNext}) + (1-y) \log(1 - P(\text{IsNext}))\right]}
$$

其中 $y = 1$ 表示 IsNext，$y = 0$ 表示 NotNext。

**等价形式**（使用双输出 softmax）：

原始 BERT 实现使用 2-class softmax 而非 sigmoid：

$$
P(\text{IsNext} \mid A, B) = \text{softmax}(W_{\text{NSP}} h_{\text{[CLS]}}^{(L)} + b_{\text{NSP}})
$$

其中 $W_{\text{NSP}} \in \mathbb{R}^{2 \times d}$，输出两个 logit。

$$
\mathcal{L}_{\text{NSP}} = -\log P(y \mid h_{\text{[CLS]}}^{(L)})
$$

#### 3.3.3 联合预训练损失

BERT 的总预训练损失是 MLM 和 NSP 的简单相加：

$$
\boxed{\mathcal{L}_{\text{pretrain}} = \mathcal{L}_{\text{MLM}} + \mathcal{L}_{\text{NSP}}}
$$

> **注意**：原始论文中两个损失没有加权系数，直接相加。后续研究（如 RoBERTa）发现去掉 NSP 反而可以提升性能。

---

## 4. 梯度推导与参数更新

### 4.1 MLM 损失的梯度推导

设 MLM 头的输出 logits 为：
$$
z_t = W_{\text{MLM}} h_t^{(L)} + b_{\text{MLM}} \in \mathbb{R}^{|V|}
$$

softmax 概率：
$$
p_{t,w} = \frac{\exp(z_{t,w})}{\sum_{w'=1}^{|V|} \exp(z_{t,w'})}
$$

交叉熵损失（对单个被掩码位置 $t$）：
$$
\mathcal{L}_t = -\log p_{t,x_t} = -z_{t,x_t} + \log \sum_{w'=1}^{|V|} \exp(z_{t,w'})
$$

**对 logits 的梯度**：

$$
\frac{\partial \mathcal{L}_t}{\partial z_{t,w}} = p_{t,w} - \mathbb{1}[w = x_t]
$$

$$
\boxed{\frac{\partial \mathcal{L}_t}{\partial z_t} = p_t - e_{x_t} \in \mathbb{R}^{|V|}}
$$

其中 $e_{x_t}$ 是第 $x_t$ 个位置为 1 的 one-hot 向量。

**对 MLM 权重矩阵的梯度**：

$$
\frac{\partial \mathcal{L}_t}{\partial W_{\text{MLM}}} = \frac{\partial \mathcal{L}_t}{\partial z_t} \cdot \frac{\partial z_t}{\partial W_{\text{MLM}}} = (p_t - e_{x_t}) (h_t^{(L)})^\top
$$

$$
\boxed{\frac{\partial \mathcal{L}_t}{\partial W_{\text{MLM}}} = (p_t - e_{x_t}) (h_t^{(L)})^\top \in \mathbb{R}^{|V| \times d}}
$$

**对隐藏向量的梯度**（用于反向传播到 Transformer 层）：

$$
\boxed{\frac{\partial \mathcal{L}_t}{\partial h_t^{(L)}} = W_{\text{MLM}}^\top (p_t - e_{x_t}) \in \mathbb{R}^d}
$$

### 4.2 NSP 损失的梯度推导

设 NSP 头的输出 logits 为：
$$
z_{\text{NSP}} = W_{\text{NSP}} h_{\text{[CLS]}}^{(L)} + b_{\text{NSP}} \in \mathbb{R}^2
$$

softmax 概率：
$$
p_{\text{NSP}} = \text{softmax}(z_{\text{NSP}}) = \begin{bmatrix} p_0 \\ p_1 \end{bmatrix}
$$

其中 $p_0 = P(\text{NotNext})$，$p_1 = P(\text{IsNext})$。

**对 logits 的梯度**：

$$
\boxed{\frac{\partial \mathcal{L}_{\text{NSP}}}{\partial z_{\text{NSP}}} = p_{\text{NSP}} - e_y \in \mathbb{R}^2}
$$

**对 [CLS] 隐藏向量的梯度**：

$$
\boxed{\frac{\partial \mathcal{L}_{\text{NSP}}}{\partial h_{\text{[CLS]}}^{(L)}} = W_{\text{NSP}}^\top (p_{\text{NSP}} - e_y) \in \mathbb{R}^d}
$$

### 4.3 联合损失与多任务学习

**联合梯度**：

对于 Transformer 参数 $\theta$，总梯度为两个任务梯度之和：

$$
\frac{\partial \mathcal{L}_{\text{pretrain}}}{\partial \theta} = \frac{\partial \mathcal{L}_{\text{MLM}}}{\partial \theta} + \frac{\partial \mathcal{L}_{\text{NSP}}}{\partial \theta}
$$

**梯度流分析**：

- MLM 梯度从被掩码位置的输出头回传到 Transformer 各层
- NSP 梯度从 [CLS] 位置的输出头回传到 Transformer 各层
- 两者在 Transformer 层共享参数，梯度相加

```
MLM Head: h_t^(L) → z_t → L_MLM → ∂L_MLM/∂θ
                                         ↘
                                          + → ∂L_pretrain/∂θ → θ_new = θ - η·∇θ
                                         ↗
NSP Head: h_[CLS]^(L) → z_NSP → L_NSP → ∂L_NSP/∂θ
```

> **Q:** MLM 和 NSP 的梯度是否会冲突？
>
> **A:** 理论上可能存在梯度方向冲突（多任务学习中的常见问题）。但实际中，MLM 提供的是 token 级别的信号，NSP 提供的是句子级别的信号，两者互补性较强。不过后续研究表明 NSP 的贡献有限，可能引入噪声。

---

## 5. 训练优化方法总结

### 5.1 AdamW 优化器

BERT 使用 AdamW（带权重衰减的 Adam）：

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
$$
$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$
$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

**AdamW 与 Adam + L2 的关键区别**：

$$
\boxed{\theta_{t+1} = \theta_t - \eta \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t\right) \quad \text{（AdamW）}}
$$

标准 Adam + L2 正则化：
$$
\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t + \lambda \theta_t}{\sqrt{\hat{v}_t} + \epsilon} \quad \text{（Adam + L2，权重衰减被自适应学习率缩放）}
$$

**BERT 的超参数**：
- $\beta_1 = 0.9$, $\beta_2 = 0.999$
- $\epsilon = 10^{-6}$
- $\lambda = 0.01$（权重衰减）

### 5.2 学习率预热与线性衰减

**学习率调度**：

$$
\text{lr}(t) = \begin{cases}
\text{lr}_{\text{peak}} \cdot \frac{t}{t_{\text{warmup}}} & \text{if } t \leq t_{\text{warmup}} \\[6pt]
\text{lr}_{\text{peak}} \cdot \frac{T - t}{T - t_{\text{warmup}}} & \text{if } t > t_{\text{warmup}}
\end{cases}
$$

其中 $T$ 是总训练步数，$t_{\text{warmup}}$ 是预热步数。

**BERT-Base 配置**：
- $\text{lr}_{\text{peak}} = 10^{-4}$
- $t_{\text{warmup}} = 10{,}000$ 步
- $T = 1{,}000{,}000$ 步

### 5.3 大批量训练策略

**BERT 的训练配置**：

| 参数 | BERT-Base | BERT-Large |
|------|-----------|------------|
| 层数 $L$ | 12 | 24 |
| 隐藏维度 $d$ | 768 | 1024 |
| 注意力头数 $h$ | 12 | 16 |
| 参数量 | 110M | 340M |
| Batch Size | 256 | 256 |
| 序列长度 | 512 | 512 |
| 训练步数 | 1M | 1M |
| 训练语料 | BooksCorpus + English Wikipedia | 同左 |

**大批量训练的数学意义**：

梯度估计的方差：
$$
\text{Var}(\hat{g}_B) = \frac{\text{Var}(g)}{B}
$$

Batch Size $B$ 越大，梯度估计越稳定，可以使用更大的学习率。

---

## 6. 从数学到代码：完整实现

### 6.1 NumPy 实现核心组件

```python
import numpy as np

def softmax(x, axis=-1):
    """
    数值稳定的 softmax
    
    数学公式:
        softmax(x_i) = exp(x_i - max(x)) / Σ exp(x_j - max(x))
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def gelu(x):
    """
    GELU 激活函数 (BERT 使用 GELU 而非 ReLU)
    
    数学公式:
        GELU(x) = x · Φ(x) ≈ 0.5x(1 + tanh[√(2/π)(x + 0.044715x³)])
    
    其中 Φ(x) 是标准正态分布的 CDF
    """
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

def layer_norm(x, gamma, beta, eps=1e-12):
    """
    层归一化
    
    数学公式:
        LayerNorm(x) = γ · (x - μ) / √(σ² + ε) + β
    
    参数:
        x: 输入 (batch, seq_len, d)
        gamma: 缩放参数 (d,)
        beta: 偏移参数 (d,)
    """
    mean = np.mean(x, axis=-1, keepdims=True)  # (batch, seq_len, 1)
    var = np.var(x, axis=-1, keepdims=True)     # (batch, seq_len, 1)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    缩放点积注意力（双向，无因果掩码）
    
    数学公式:
        Attention(Q, K, V) = softmax(QK^T / √d_k + M) V
    
    参数:
        Q: 查询矩阵 (batch, heads, seq_len, d_k)
        K: 键矩阵   (batch, heads, seq_len, d_k)
        V: 值矩阵   (batch, heads, seq_len, d_v)
        mask: padding 掩码 (batch, 1, 1, seq_len)
              1 = 有效位置, 0 = padding
    
    返回:
        output: (batch, heads, seq_len, d_v)
        weights: (batch, heads, seq_len, seq_len)
    """
    d_k = Q.shape[-1]
    
    # 步骤1: QK^T / √d_k
    scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    # scores: (batch, heads, seq_len, seq_len)
    
    # 步骤2: 应用 padding 掩码（BERT 不需要因果掩码）
    if mask is not None:
        scores = np.where(mask == 1, scores, -1e9)
    
    # 步骤3: softmax 归一化
    weights = softmax(scores, axis=-1)
    # weights: (batch, heads, seq_len, seq_len)
    
    # 步骤4: 加权求和
    output = np.matmul(weights, V)
    # output: (batch, heads, seq_len, d_v)
    
    return output, weights

def mlm_loss_numpy(logits, labels, mask_positions):
    """
    MLM 交叉熵损失（NumPy 实现）
    
    数学公式:
        L_MLM = -1/|M| Σ_{t∈M} log P(x_t | x̃)
    
    参数:
        logits: 模型输出 (batch, seq_len, vocab_size)
        labels: 原始 token ID (batch, seq_len)
        mask_positions: 布尔掩码 (batch, seq_len)
                        True = 被掩码的位置
    
    返回:
        loss: 标量损失值
    """
    # 仅计算被掩码位置的损失
    batch_size, seq_len, vocab_size = logits.shape
    
    total_loss = 0.0
    count = 0
    
    for b in range(batch_size):
        for t in range(seq_len):
            if mask_positions[b, t]:
                # softmax 概率
                probs = softmax(logits[b, t])
                # 交叉熵损失
                total_loss -= np.log(probs[labels[b, t]] + 1e-10)
                count += 1
    
    return total_loss / max(count, 1)

def nsp_loss_numpy(cls_logits, labels):
    """
    NSP 二分类交叉熵损失（NumPy 实现）
    
    数学公式:
        L_NSP = -[y log P(IsNext) + (1-y) log(1 - P(IsNext))]
    
    参数:
        cls_logits: [CLS] 位置的 logits (batch, 2)
        labels: NSP 标签 (batch,)  0=NotNext, 1=IsNext
    
    返回:
        loss: 标量损失值
    """
    probs = softmax(cls_logits, axis=-1)  # (batch, 2)
    batch_size = cls_logits.shape[0]
    
    total_loss = 0.0
    for b in range(batch_size):
        total_loss -= np.log(probs[b, labels[b]] + 1e-10)
    
    return total_loss / batch_size

# ========== 测试 NumPy 实现 ==========

def test_numpy_components():
    """测试 NumPy 实现的正确性"""
    np.random.seed(42)
    batch_size, seq_len, d_k, num_heads, vocab_size = 2, 8, 16, 4, 100
    
    # 测试 GELU
    x = np.array([-2, -1, 0, 1, 2], dtype=np.float64)
    print(f"GELU: {gelu(x)}")  # GELU(0)≈0, GELU(1)≈0.841
    
    # 测试 Attention（双向，无因果掩码，仅 padding 掩码）
    Q = np.random.randn(batch_size, num_heads, seq_len, d_k)
    K = np.random.randn(batch_size, num_heads, seq_len, d_k)
    V = np.random.randn(batch_size, num_heads, seq_len, d_k)
    mask = np.ones((batch_size, 1, 1, seq_len))
    mask[:, :, :, -2:] = 0  # 最后2位是 padding
    
    output, weights = scaled_dot_product_attention(Q, K, V, mask)
    print(f"Attention weights sum: {weights[0,0,0].sum():.4f}")  # 应为 1.0
    print(f"Attention to padding: {weights[0,0,0,-2:]}")  # 应接近 0
    
    # 测试 MLM/NSP Loss
    logits = np.random.randn(batch_size, seq_len, vocab_size)
    labels = np.random.randint(0, vocab_size, (batch_size, seq_len))
    mask_pos = np.zeros((batch_size, seq_len), dtype=bool)
    mask_pos[0, 2] = True; mask_pos[1, 5] = True
    print(f"MLM Loss: {mlm_loss_numpy(logits, labels, mask_pos):.4f} (baseline ~{np.log(vocab_size):.2f})")
    
    cls_logits = np.random.randn(batch_size, 2)
    print(f"NSP Loss: {nsp_loss_numpy(cls_logits, np.array([1,0])):.4f} (baseline ~{np.log(2):.2f})")

if __name__ == "__main__":
    test_numpy_components()
```

### 6.2 PyTorch 完整实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

# =============================================
# 6.2.1 BERT 嵌入层
# =============================================

class BERTEmbedding(nn.Module):
    """
    BERT 输入嵌入 = Token Embedding + Segment Embedding + Position Embedding
    
    数学公式:
        E_input = E_token(x) + E_segment(s) + E_position(t)
    
    参数:
        vocab_size: 词表大小 |V|
        d_model: 嵌入维度 d
        max_len: 最大序列长度
        dropout: dropout 概率
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        max_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        # 三重嵌入
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.segment_embedding = nn.Embedding(2, d_model)  # 只有 0 和 1 两个 segment
        self.position_embedding = nn.Embedding(max_len, d_model)  # 可学习位置编码
        
        # LayerNorm + Dropout
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        input_ids: torch.Tensor,       # (batch, seq_len)
        segment_ids: torch.Tensor,      # (batch, seq_len)
        position_ids: Optional[torch.Tensor] = None  # (batch, seq_len)
    ) -> torch.Tensor:
        seq_len = input_ids.size(1)
        
        # 自动生成位置 ID
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        # 三重嵌入相加
        token_emb = self.token_embedding(input_ids)      # (batch, seq_len, d)
        segment_emb = self.segment_embedding(segment_ids) # (batch, seq_len, d)
        position_emb = self.position_embedding(position_ids) # (batch, seq_len, d)
        
        embeddings = token_emb + segment_emb + position_emb
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings  # (batch, seq_len, d)

# =============================================
# 6.2.2 多头自注意力（双向）
# =============================================

class BERTSelfAttention(nn.Module):
    """
    BERT 双向多头自注意力（不使用因果掩码，仅 padding 掩码）
    
    数学公式:
        Attention(Q, K, V) = softmax(QK^T / √d_k + M_pad) V
    """
    def __init__(self, d_model: int = 768, num_heads: int = 12, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度
        
        # Q, K, V 投影矩阵
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        
        # 输出投影
        self.W_O = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,    # (batch, seq_len, d_model)
        attention_mask: Optional[torch.Tensor] = None  # (batch, 1, 1, seq_len)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape
        
        # 线性投影
        Q = self.W_Q(hidden_states)  # (batch, seq_len, d_model)
        K = self.W_K(hidden_states)
        V = self.W_V(hidden_states)
        
        # 分割多头: (batch, seq_len, d_model) → (batch, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 缩放点积注意力: QK^T / √d_k
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores: (batch, num_heads, seq_len, seq_len)
        
        # 应用 padding 掩码（BERT 的关键：无因果掩码）
        if attention_mask is not None:
            scores = scores + attention_mask  # mask 中 padding 位置为 -1e9
        
        # Softmax 归一化
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 加权求和
        context = torch.matmul(attention_weights, V)
        # context: (batch, num_heads, seq_len, d_k)
        
        # 拼接多头
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # 输出投影
        output = self.W_O(context)
        
        return output, attention_weights

# =============================================
# 6.2.3 BERT Transformer Block
# =============================================

class BERTBlock(nn.Module):
    """
    BERT Transformer Block (Post-LN): Self-Attention → Add&Norm → FFN(GELU) → Add&Norm
    """
    def __init__(
        self,
        d_model: int = 768,
        num_heads: int = 12,
        d_ff: int = 3072,
        dropout: float = 0.1
    ):
        super().__init__()
        # 自注意力
        self.self_attention = BERTSelfAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-12)
        
        # 前馈网络（GELU 激活）
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model, eps=1e-12)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. 自注意力 + 残差 + LayerNorm
        attn_output, attn_weights = self.self_attention(hidden_states, attention_mask)
        hidden_states = self.norm1(hidden_states + self.dropout(attn_output))
        
        # 2. FFN + 残差 + LayerNorm
        ffn_output = self.ffn(hidden_states)
        hidden_states = self.norm2(hidden_states + ffn_output)
        
        return hidden_states, attn_weights

# =============================================
# 6.2.4 BERT 主模型
# =============================================

class BERTModel(nn.Module):
    """
    BERT 主模型: Embedding → [BERTBlock × L] → sequence_output + pooled_output
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
        
        # Transformer 编码器层
        self.layers = nn.ModuleList([
            BERTBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # [CLS] 输出的池化层
        self.pooler = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh()
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,        # (batch, seq_len)
        segment_ids: torch.Tensor,       # (batch, seq_len)
        attention_mask: Optional[torch.Tensor] = None  # (batch, seq_len)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回:
            sequence_output: (batch, seq_len, d_model) 所有位置的隐藏向量
            pooled_output: (batch, d_model) [CLS] 位置的池化输出
        """
        # 构建 attention mask: (batch, seq_len) → (batch, 1, 1, seq_len)
        if attention_mask is not None:
            # 将 0/1 掩码转为 0/-1e9 的加性掩码
            extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
            extended_mask = (1.0 - extended_mask.float()) * (-1e9)
        else:
            extended_mask = None
        
        # 嵌入
        hidden_states = self.embedding(input_ids, segment_ids)
        # hidden_states: (batch, seq_len, d_model)
        
        # 逐层 Transformer
        all_attention_weights = []
        for layer in self.layers:
            hidden_states, attn_weights = layer(hidden_states, extended_mask)
            all_attention_weights.append(attn_weights)
        
        # 序列输出
        sequence_output = hidden_states  # (batch, seq_len, d_model)
        
        # 池化输出（取 [CLS] 位置）
        pooled_output = self.pooler(sequence_output[:, 0])  # (batch, d_model)
        
        return sequence_output, pooled_output

# =============================================
# 6.2.5 MLM 预训练头
# =============================================

class MLMHead(nn.Module):
    """MLM 预测头: h_t → Dense → GELU → LayerNorm → Linear(共享权重) → logits"""
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-12)
        
        # 输出投影（可与 token embedding 共享权重）
        self.decoder = nn.Linear(d_model, vocab_size, bias=True)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        参数:
            hidden_states: (batch, seq_len, d_model) 或 (num_masked, d_model)
        返回:
            logits: (..., vocab_size)
        """
        hidden = self.dense(hidden_states)
        hidden = self.activation(hidden)
        hidden = self.layer_norm(hidden)
        logits = self.decoder(hidden)
        return logits

# =============================================
# 6.2.6 NSP 预训练头
# =============================================

class NSPHead(nn.Module):
    """NSP 头: h_[CLS] → Linear(2) → logits"""
    def __init__(self, d_model: int):
        super().__init__()
        self.classifier = nn.Linear(d_model, 2)
    
    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        """
        参数:
            pooled_output: (batch, d_model) — [CLS] 的池化输出
        返回:
            logits: (batch, 2)
        """
        return self.classifier(pooled_output)

# =============================================
# 6.2.7 完整 BERT 预训练模型
# =============================================

class BERTForPreTraining(nn.Module):
    """完整 BERT 预训练模型 = BERT + MLM Head + NSP Head, L = L_MLM + L_NSP"""
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
        self.bert = BERTModel(vocab_size, d_model, num_heads, num_layers, d_ff, max_len, dropout)
        self.mlm_head = MLMHead(d_model, vocab_size)
        self.nsp_head = NSPHead(d_model)
        
        # 共享权重：MLM 输出层与 token embedding 共享
        self.mlm_head.decoder.weight = self.bert.embedding.token_embedding.weight
    
    def forward(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        mlm_labels: Optional[torch.Tensor] = None,   # (batch, seq_len), -100 表示不计算损失
        nsp_labels: Optional[torch.Tensor] = None     # (batch,), 0 或 1
    ) -> dict:
        # BERT 编码
        sequence_output, pooled_output = self.bert(input_ids, segment_ids, attention_mask)
        
        # MLM 预测
        mlm_logits = self.mlm_head(sequence_output)  # (batch, seq_len, vocab_size)
        
        # NSP 预测
        nsp_logits = self.nsp_head(pooled_output)  # (batch, 2)
        
        result = {
            "mlm_logits": mlm_logits,
            "nsp_logits": nsp_logits,
            "sequence_output": sequence_output,
            "pooled_output": pooled_output
        }
        
        # 计算损失
        if mlm_labels is not None:
            mlm_loss = F.cross_entropy(
                mlm_logits.view(-1, mlm_logits.size(-1)),  # (batch*seq_len, vocab_size)
                mlm_labels.view(-1),                        # (batch*seq_len,)
                ignore_index=-100                           # 忽略非掩码位置
            )
            result["mlm_loss"] = mlm_loss
        
        if nsp_labels is not None:
            nsp_loss = F.cross_entropy(nsp_logits, nsp_labels)
            result["nsp_loss"] = nsp_loss
        
        if mlm_labels is not None and nsp_labels is not None:
            result["total_loss"] = result["mlm_loss"] + result["nsp_loss"]
        
        return result

# =============================================
# 6.2.8 掩码生成工具
# =============================================

def create_mlm_masks(
    input_ids: torch.Tensor,
    vocab_size: int,
    mask_token_id: int = 103,  # [MASK] 的 token ID
    special_token_ids: set = None,
    mlm_probability: float = 0.15
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    创建 MLM 掩码和标签
    
    掩码策略:
        - 15% 的 token 被选中
        - 选中的 token: 80% → [MASK], 10% → 随机词, 10% → 保持不变
    
    参数:
        input_ids: (batch, seq_len) 输入 token ID
        vocab_size: 词表大小
        mask_token_id: [MASK] token 的 ID
        special_token_ids: 特殊 token ID 集合（不被掩码）
        mlm_probability: 掩码概率 (默认 0.15)
    
    返回:
        masked_input_ids: (batch, seq_len) 掩码后的输入
        mlm_labels: (batch, seq_len) MLM 标签，-100 表示不计算损失
    """
    if special_token_ids is None:
        special_token_ids = {0, 101, 102, 103}  # [PAD], [CLS], [SEP], [MASK]
    
    masked_input_ids = input_ids.clone()
    mlm_labels = torch.full_like(input_ids, -100)  # -100 = 忽略
    
    # 生成掩码概率矩阵
    probability_matrix = torch.full(input_ids.shape, mlm_probability)
    
    # 特殊 token 不掩码
    for token_id in special_token_ids:
        probability_matrix.masked_fill_(input_ids == token_id, value=0.0)
    
    # 采样掩码位置
    masked_indices = torch.bernoulli(probability_matrix).bool()
    
    # 设置 MLM 标签（仅被掩码位置有真实标签）
    mlm_labels[masked_indices] = input_ids[masked_indices]
    
    # 80% → [MASK]
    indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
    masked_input_ids[indices_replaced] = mask_token_id
    
    # 10% → 随机词（在剩余的掩码位置中，50% 随机替换）
    indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(vocab_size, input_ids.shape, dtype=input_ids.dtype)
    masked_input_ids[indices_random] = random_words[indices_random]
    
    # 剩余 10% 保持不变（不需要额外操作）
    
    return masked_input_ids, mlm_labels

# =============================================
# 6.2.9 微调分类模型
# =============================================

class BERTForSequenceClassification(nn.Module):
    """
    BERT 序列分类微调模型
    
    结构:
        Input → BERT → [CLS] pooled_output → Dropout → Linear → logits
    
    数学公式:
        logits = W_cls · Tanh(W_pool · h_[CLS] + b_pool) + b_cls
    """
    def __init__(self, bert_model: BERTModel, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(bert_model.d_model, num_classes)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> dict:
        _, pooled_output = self.bert(input_ids, segment_ids, attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # (batch, num_classes)
        
        result = {"logits": logits}
        
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            result["loss"] = loss
        
        return result

# =============================================
# 6.2.10 使用示例与测试
# =============================================

def test_bert():
    """测试 BERT 模型的前向传播"""
    vocab_size, d_model, num_heads = 1000, 128, 4
    num_layers, d_ff, max_len = 2, 512, 64
    batch_size, seq_len = 4, 32
    
    model = BERTForPreTraining(vocab_size, d_model, num_heads, num_layers, d_ff, max_len)
    
    # 模拟输入: [CLS] ... [SEP] ... [SEP] + padding
    input_ids = torch.randint(4, vocab_size, (batch_size, seq_len))
    input_ids[:, 0] = 101; input_ids[:, 15] = 102; input_ids[:, -1] = 102
    segment_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    segment_ids[:, 16:] = 1
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    attention_mask[:, -3:] = 0
    
    masked_input_ids, mlm_labels = create_mlm_masks(input_ids, vocab_size)
    nsp_labels = torch.randint(0, 2, (batch_size,))
    
    model.eval()
    with torch.no_grad():
        result = model(masked_input_ids, segment_ids, attention_mask, mlm_labels, nsp_labels)
    
    print(f"序列输出: {result['sequence_output'].shape}")  # (4, 32, 128)
    print(f"MLM Loss: {result['mlm_loss']:.4f}, NSP Loss: {result['nsp_loss']:.4f}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

if __name__ == "__main__":
    test_bert()
```

---

## 7. 实践技巧与可视化

### 7.1 注意力可视化

```python
import matplotlib.pyplot as plt

def visualize_attention(attn_weights, tokens, layer=0, head=0):
    """可视化指定层/头的注意力权重热力图"""
    attn = attn_weights[layer][0, head].detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(attn[:len(tokens), :len(tokens)], cmap='Blues')
    ax.set_xticks(range(len(tokens))); ax.set_xticklabels(tokens, rotation=90)
    ax.set_yticks(range(len(tokens))); ax.set_yticklabels(tokens)
    ax.set_xlabel('Key'); ax.set_ylabel('Query')
    plt.tight_layout(); plt.savefig('bert_attention_viz.png', dpi=150)
```

**BERT 注意力的典型模式**：

| 模式 | 描述 | 出现位置 |
|------|------|---------|
| 对角线模式 | 每个词主要关注自己 | 低层 |
| 垂直线模式 | 所有词关注 [CLS] 或 [SEP] | 中层 |
| 宽泛关注 | 注意力分散在所有词上 | 高层 |
| 语法模式 | 关注语法依赖的词 | 特定头 |

### 7.2 微调最佳实践

**学习率选择**：

| 任务类型 | 推荐学习率 | Batch Size |
|---------|-----------|------------|
| 文本分类 | 2e-5 ~ 5e-5 | 16 ~ 32 |
| NER | 3e-5 ~ 5e-5 | 16 ~ 32 |
| 问答 | 3e-5 | 12 ~ 32 |
| 句子相似度 | 2e-5 ~ 3e-5 | 16 ~ 32 |

**微调关键代码**：

```python
# AdamW: bias 和 LayerNorm 不加权重衰减
no_decay = ['bias', 'LayerNorm.weight']
optimizer = torch.optim.AdamW([
    {'params': [p for n, p in model.named_parameters()
               if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters()
               if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
], lr=2e-5)

# 训练循环（含 warmup + 线性衰减 + 梯度裁剪）
for epoch in range(3):
    for batch in dataloader:
        loss = model(**batch)['loss']
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
```

---

## 8. 与其他模型的关系

### 8.1 BERT vs GPT：编码器 vs 解码器

| 特性 | BERT | GPT |
|------|------|-----|
| **架构** | Transformer Encoder | Transformer Decoder |
| **注意力方向** | 双向（全局） | 单向（因果掩码） |
| **预训练目标** | MLM + NSP | 自回归 LM |
| **掩码** | Padding mask | Causal mask + Padding mask |
| **优势** | 理解任务（分类、NER） | 生成任务（文本生成） |
| **代表性下游任务** | GLUE、SQuAD | 文本续写、对话 |

**数学对比**：

BERT 的目标函数：
$$
\mathcal{L}_{\text{BERT}} = -\sum_{t \in \mathcal{M}} \log P(x_t \mid x_1, \ldots, x_{t-1}, x_{t+1}, \ldots, x_n)
$$

GPT 的目标函数：
$$
\mathcal{L}_{\text{GPT}} = -\sum_{t=1}^n \log P(x_t \mid x_1, \ldots, x_{t-1})
$$

**核心区别**：BERT 条件概率中包含**左右两侧**上下文，GPT 只包含**左侧**上下文。

### 8.2 BERT 的后继者们

```
BERT (2018)
  ├── RoBERTa (2019): 去掉 NSP + 更多数据 + 更长训练
  ├── ALBERT (2019): 参数共享 + 嵌入分解 → 更小模型
  ├── DistilBERT (2019): 知识蒸馏 → 更快推理
  ├── SpanBERT (2019): Span 掩码 → 更好的 span 预测
  ├── ELECTRA (2020): 替换检测 → 更高效训练
  └── DeBERTa (2020): 解耦注意力 → 更强性能
```

**关键改进总结**：

| 模型 | 核心改进 | 效果 |
|------|---------|------|
| RoBERTa | 去掉 NSP，动态掩码，更大 batch | 所有任务显著提升 |
| ALBERT | 跨层参数共享，嵌入矩阵分解 | 参数减少 80%，性能相当 |
| ELECTRA | 替换 token 检测（非 MLM） | 相同计算量下效果更好 |
| DeBERTa | 位置-内容解耦注意力 | SuperGLUE 超越人类 |

---

## 扩展阅读与实现

### 9.1 Whole Word Masking 的数学分析

**问题**：标准 MLM 掩码 WordPiece 子词可能导致泄露。

例如，"playing" 被分为 "play" + "##ing"，如果只掩码 "##ing"：
- 模型可以从 "play" 轻松推断出 "##ing"
- 训练信号变得过于简单

**Whole Word Masking**：如果一个词的某个子词被选中，该词的**所有子词**都被掩码。

$$
\text{如果 } x_t \in \text{word}_k \text{ 被选中掩码} \Rightarrow \forall x_j \in \text{word}_k, \; x_j \text{ 也被掩码}
$$

**效果**：迫使模型学习更深层的语义理解，而非简单的子词拼接。

> **Q:** Whole Word Masking 会改变掩码比例吗？
>
> **A:** 会略有变化。由于以整词为单位掩码，实际掩码的 token 比例可能略高于 15%（因为一个词可能包含多个子词）。实现中通常通过调整采样概率来保持总掩码比例接近 15%。

### 9.2 为什么 NSP 后来被质疑？

**RoBERTa 的实验发现**：

| 设置 | MNLI | SST-2 | SQuAD |
|------|------|-------|-------|
| BERT (MLM + NSP) | 84.6 | 93.0 | 88.5 |
| 去掉 NSP (仅 MLM) | **85.2** | **93.5** | **89.0** |

**可能的原因**：

1. **任务过于简单**：随机采样的负例与正例差异太大，模型可以通过主题匹配（而非逻辑推理）轻松区分
2. **损失冲突**：NSP 损失可能与 MLM 损失产生梯度冲突，干扰语言理解的学习
3. **数据构造偏差**：负例来自不同文档，模型学到的是"主题一致性"而非"句间逻辑"

**数学分析**：

理想的 NSP 应该学习：
$$
P(\text{IsNext} \mid A, B) \propto P(B \mid A) \quad \text{（条件概率：给定 A，B 的可能性）}
$$

但实际学到的可能是：
$$
P(\text{IsNext} \mid A, B) \propto \text{sim}(\text{topic}(A), \text{topic}(B)) \quad \text{（主题相似度）}
$$

### 9.3 BERT 的参数效率分析

**BERT-Base 参数分布**：

| 组件 | 参数量 | 占比 |
|------|--------|------|
| Token Embedding ($|V| \times d$) | $30522 \times 768 = 23.4\text{M}$ | 21.3% |
| Position Embedding ($512 \times d$) | $512 \times 768 = 0.4\text{M}$ | 0.4% |
| Segment Embedding ($2 \times d$) | $2 \times 768 = 0.002\text{M}$ | ~0% |
| Self-Attention ($4d^2 \times L$) | $4 \times 768^2 \times 12 = 28.3\text{M}$ | 25.7% |
| FFN ($2 \times d \times d_{ff} \times L$) | $2 \times 768 \times 3072 \times 12 = 56.6\text{M}$ | 51.5% |
| LayerNorm + bias | ~$0.1\text{M}$ | ~0.1% |
| Pooler | $768^2 = 0.6\text{M}$ | 0.5% |
| **总计** | **~110M** | **100%** |

> **关键发现**：FFN 占据了超过一半的参数，这也是后续 ALBERT 通过参数共享主要节省参数的来源。

---

## 参考资源

### 经典论文

1. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805). NAACL 2019.
   - **贡献**：提出 MLM + NSP 预训练方法，开创"预训练 + 微调"范式

2. Liu, Y., et al. (2019). [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692). arXiv.
   - **贡献**：证明去掉 NSP、增加训练数据和训练时间可显著提升 BERT

3. Lan, Z., et al. (2019). [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942). ICLR 2020.
   - **贡献**：跨层参数共享和嵌入矩阵分解，大幅减少参数量

4. Clark, K., et al. (2020). [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555). ICLR 2020.
   - **贡献**：用替换 token 检测替代 MLM，提升训练效率

### 教材与书籍

5. Jurafsky, D. & Martin, J.H. [Speech and Language Processing (3rd ed.)](https://web.stanford.edu/~jurafsky/slp3/). Stanford University.
   - **章节**：第 11 章详细讲解 BERT 及其变体

### 在线资源与教程

6. Alammar, J. [The Illustrated BERT](https://jalammar.github.io/illustrated-bert/).
   - **内容**：直观可视化 BERT 的架构和预训练过程

7. Hugging Face. [BERT Documentation](https://huggingface.co/docs/transformers/model_doc/bert).
   - **内容**：BERT 模型的工业级实现和使用教程

8. [Stanford CS224N: NLP with Deep Learning](https://web.stanford.edu/class/cs224n/). Lecture 14-15.
   - **内容**：BERT 及预训练模型的理论讲解

---

## 附录：符号表

| 符号 | 含义 | 维度/类型 |
|------|------|----------|
| $n$ | 输入序列长度 | 标量 |
| $d$ / $d_{\text{model}}$ | 隐藏维度 | 标量，BERT-Base: 768 |
| $d_k$ | 查询/键维度 | 标量，$d / h$ |
| $d_{ff}$ | FFN 隐藏层维度 | 标量，BERT-Base: 3072 |
| $h$ | 注意力头数 | 标量，BERT-Base: 12 |
| $L$ | Transformer 层数 | 标量，BERT-Base: 12 |
| $\|V\|$ | 词表大小 | 标量，BERT: 30522 |
| $L_{\max}$ | 最大序列长度 | 标量，BERT: 512 |
| $\mathcal{M}$ | 被掩码位置的集合 | 集合 |
| $p$ | MLM 掩码概率 | 标量，0.15 |
| $E_{\text{token}}$ | Token 嵌入矩阵 | $(\|V\|, d)$ |
| $E_{\text{segment}}$ | Segment 嵌入矩阵 | $(2, d)$ |
| $E_{\text{position}}$ | Position 嵌入矩阵 | $(L_{\max}, d)$ |
| $h_t^{(l)}$ | 第 $l$ 层第 $t$ 位置的隐藏向量 | $(d,)$ |
| $h_{\text{[CLS]}}^{(L)}$ | 最终层 [CLS] 的隐藏向量 | $(d,)$ |
| $W_{\text{MLM}}$ | MLM 输出权重矩阵 | $(\|V\|, d)$ |
| $W_{\text{NSP}}$ | NSP 分类权重矩阵 | $(2, d)$ |
| $Q, K, V$ | 查询/键/值矩阵 | $(n, d_k)$ |
| $M$ | 注意力掩码矩阵 | $(n, n)$ |
| $\mathcal{L}_{\text{MLM}}$ | MLM 损失值 | 标量 |
| $\mathcal{L}_{\text{NSP}}$ | NSP 损失值 | 标量 |
| $\mathcal{L}_{\text{pretrain}}$ | 预训练总损失值 | 标量 |
| $\ell(\cdot, \cdot)$ | 交叉熵损失函数 | 函数 |

**典型配置示例**：
- BERT-Base: $d = 768$, $h = 12$, $L = 12$, $d_{ff} = 3072$, 参数量 110M
- BERT-Large: $d = 1024$, $h = 16$, $L = 24$, $d_{ff} = 4096$, 参数量 340M

---

最后更新：2026-03-18
