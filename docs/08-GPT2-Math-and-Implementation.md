# GPT-2 数学原理与实现 —— 自回归语言模型与 Zero-shot 的完整推导

> **前置知识**：Transformer 解码器、自注意力机制、交叉熵损失、Python 基础  
> **与前面内容的联系**：建议先学习 [Transformer-Math-and-Implementation](./06-Transformer-Math-and-Implementation.md) 和 [BERT-Math-and-Implementation](./07-BERT-Math-and-Implementation.md)  
> **与后续内容的联系**：GPT-2 是 GPT 系列的关键节点，直接通向 GPT-3 的 Few-shot 范式和更大规模的语言模型

---

## 目录

1. [引言：为什么需要更大的自回归模型？](#1-引言为什么需要更大的自回归模型)
   - 1.1 [从 GPT 到 GPT-2：规模与零样本的突破](#11-从-gpt-到-gpt-2规模与零样本的突破)
   - 1.2 [BERT vs GPT-2：理解与生成的分水岭](#12-bert-vs-gpt-2理解与生成的分水岭)
   - 1.3 [本科数学知识映射表](#13-本科数学知识映射表)
2. [核心思想：自回归语言建模与 Zero-shot](#2-核心思想自回归语言建模与-zero-shot)
   - 2.1 [自回归分解：从联合概率到条件链](#21-自回归分解从联合概率到条件链)
   - 2.2 [Zero-shot 学习：语言模型即任务求解器](#22-zero-shot-学习语言模型即任务求解器)
   - 2.3 [任务条件化：$P(\text{output} \mid \text{input}, \text{task})$](#23-任务条件化poutput--input-task)
3. [GPT-2 架构的数学描述](#3-gpt-2-架构的数学描述)
   - 3.1 [输入表示：Token 嵌入 + 位置嵌入](#31-输入表示token-嵌入--位置嵌入)
   - 3.2 [因果注意力掩码 (Causal Mask)](#32-因果注意力掩码-causal-mask)
   - 3.3 [Pre-Norm Transformer 解码器层](#33-pre-norm-transformer-解码器层)
   - 3.4 [GPT-2 各版本配置](#34-gpt-2-各版本配置)
4. [自回归损失函数与梯度推导](#4-自回归损失函数与梯度推导)
   - 4.1 [语言模型交叉熵损失](#41-语言模型交叉熵损失)
   - 4.2 [困惑度 (Perplexity)](#42-困惑度-perplexity)
   - 4.3 [梯度分析与权重共享](#43-梯度分析与权重共享)
5. [训练优化方法总结](#5-训练优化方法总结)
   - 5.1 [预训练策略与数据集](#51-预训练策略与数据集)
   - 5.2 [优化器与学习率调度](#52-优化器与学习率调度)
   - 5.3 [训练稳定性技巧](#53-训练稳定性技巧)
6. [从数学到代码：完整实现](#6-从数学到代码完整实现)
   - 6.1 [NumPy 实现核心组件](#61-numpy-实现核心组件)
   - 6.2 [PyTorch 完整实现](#62-pytorch-完整实现)
7. [文本生成策略：采样方法的数学原理](#7-文本生成策略采样方法的数学原理)
   - 7.1 [贪心搜索与 Beam Search](#71-贪心搜索与-beam-search)
   - 7.2 [温度采样 (Temperature Sampling)](#72-温度采样-temperature-sampling)
   - 7.3 [Top-k 采样](#73-top-k-采样)
   - 7.4 [Top-p (Nucleus) 采样](#74-top-p-nucleus-采样)
   - 7.5 [采样策略可视化与对比](#75-采样策略可视化与对比)
8. [与其他模型的关系](#8-与其他模型的关系)
   - 8.1 [GPT 系列演进：GPT → GPT-2 → GPT-3](#81-gpt-系列演进gpt--gpt-2--gpt-3)
   - 8.2 [GPT-2 vs BERT：单向生成 vs 双向理解](#82-gpt-2-vs-bert单向生成-vs-双向理解)
   - 8.3 [预训练范式的转变](#83-预训练范式的转变)

[扩展阅读与实现](#扩展阅读与实现)

[参考资源](#参考资源)

附录：[符号表](#附录符号表)

---

## 1. 引言：为什么需要更大的自回归模型？

### 1.1 从 GPT 到 GPT-2：规模与零样本的突破

GPT (2018) 证明了**生成式预训练 + 判别式微调**的有效性。但它仍然需要为每个下游任务进行有监督微调。GPT-2 (2019) 提出了一个更大胆的假设：

> **足够大的语言模型不需要任何标注数据或微调，就能直接执行各种 NLP 任务。**

这就是 **Zero-shot** 学习的核心思想。

**GPT 到 GPT-2 的关键变化**：

| 维度 | GPT (2018) | GPT-2 (2019) |
|------|-----------|--------------|
| 参数量 | 117M | 1.5B（12.8x） |
| 训练数据 | BooksCorpus (5GB) | WebText (40GB) |
| 层数 | 12 | 48 |
| 隐藏维度 | 768 | 1600 |
| 上下文长度 | 512 | 1024 |
| 使用范式 | 预训练 + 微调 | **Zero-shot** |

**GPT-2 的核心论点**：

$$
\boxed{\text{Language Model} = \text{Unsupervised Multitask Learner}}
$$

当语言模型在足够多样的文本上训练到足够大时，它会隐式地学会执行各种任务——因为这些任务的示例已经以自然语言的形式出现在训练数据中。

### 1.2 BERT vs GPT-2：理解与生成的分水岭

在 [BERT](./07-BERT-Math-and-Implementation.md) 中，我们看到了双向编码的威力。GPT-2 走了一条完全不同的道路：

| 特性 | BERT (2018) | GPT-2 (2019) |
|------|------------|--------------|
| **方向** | 双向 (Bidirectional) | 单向 (Left-to-Right) |
| **架构** | Transformer Encoder | Transformer Decoder |
| **预训练目标** | MLM + NSP（填空） | 自回归 LM（续写） |
| **掩码类型** | 全连接（无因果掩码） | 因果掩码（下三角） |
| **核心能力** | 理解 (NLU) | 生成 (NLG) |
| **下游使用** | 需要微调 | Zero-shot / Few-shot |

**数学本质的区别**：

BERT 建模**掩码位置的条件概率**：
$$
P_{\text{BERT}}(x_t \mid x_{\backslash t})
$$

GPT-2 建模**自回归条件概率的乘积**：
$$
P_{\text{GPT-2}}(x_1, x_2, \ldots, x_n) = \prod_{t=1}^{n} P(x_t \mid x_1, \ldots, x_{t-1})
$$

> **关键区别**：BERT 擅长"理解已有文本"，GPT-2 擅长"生成新文本"。BERT 能看到全局，GPT-2 只能看到过去——但正因如此，GPT-2 能够**自回归地生成任意长度的文本**。

### 1.3 本科数学知识映射表

| 数学概念 | GPT-2 中的应用 | 代码对应 |
|---------|---------------|---------|
| 条件概率链式法则 | 自回归语言建模 | `for t in range(seq_len)` |
| 交叉熵 $H(p, q)$ | 语言模型损失函数 | `F.cross_entropy()` |
| Softmax 函数 | 注意力权重 + token 预测 | `F.softmax()` |
| 下三角矩阵 | 因果注意力掩码 | `torch.tril()` |
| 矩阵乘法 $AB$ | 自注意力计算 | `torch.matmul()` |
| 概率分布采样 | 温度/Top-k/Top-p 采样 | `torch.multinomial()` |
| 对数函数 $\log$ | 困惑度计算 | `torch.log()` |
| 信息熵 $H(p)$ | 困惑度 $= 2^{H(p)}$ | `torch.exp(loss)` |

---

## 2. 核心思想：自回归语言建模与 Zero-shot

### 2.1 自回归分解：从联合概率到条件链

语言建模的目标是学习文本序列的概率分布。给定一个 token 序列 $x = (x_1, x_2, \ldots, x_n)$，其联合概率可以通过**概率链式法则**分解为：

$$
\boxed{P(x_1, x_2, \ldots, x_n) = \prod_{t=1}^{n} P(x_t \mid x_1, x_2, \ldots, x_{t-1}) = \prod_{t=1}^{n} P(x_t \mid x_{<t})}
$$

其中 $x_{<t} = (x_1, x_2, \ldots, x_{t-1})$ 表示位置 $t$ 之前的所有 token。

**这个分解是精确的**（不涉及任何近似），因为概率链式法则本身就是恒等式。GPT-2 的工作是用一个参数化的神经网络 $f_\theta$ 来近似每个条件概率：

$$
P_\theta(x_t \mid x_{<t}) = \text{softmax}(f_\theta(x_{<t}))_{x_t}
$$

**自回归生成过程**：

```
给定前缀: "The cat sat on the"

步骤 1: P(x_6 | "The cat sat on the") → 采样得到 "mat"
步骤 2: P(x_7 | "The cat sat on the mat") → 采样得到 "and"
步骤 3: P(x_8 | "The cat sat on the mat and") → 采样得到 "purred"
...
```

每一步只生成一个 token，然后将其追加到输入序列中，继续生成下一个。

### 2.2 Zero-shot 学习：语言模型即任务求解器

GPT-2 论文的核心洞察是：**一个通用的语言模型可以隐式地学会执行特定的 NLP 任务**。

传统的监督学习范式：

$$
P(y \mid x; \theta) \quad \text{(需要标注数据 } (x, y) \text{ 来训练)}
$$

GPT-2 的 Zero-shot 范式：

$$
P(y \mid x, \text{task description}) \quad \text{(只需要自然语言描述)}
$$

**示例**：

| 任务 | 传统方式 | GPT-2 Zero-shot 方式 |
|------|---------|---------------------|
| 翻译 | 训练 Seq2Seq 模型 | 输入 "translate English to French: cheese =" |
| 摘要 | 训练摘要模型 | 在文章末尾加 "TL;DR:" |
| 问答 | 训练 QA 模型 | 输入 "Q: 问题 A:" |
| 情感分析 | 训练分类器 | 输入 "Review: ... Sentiment:" |

> **Q:** 为什么语言模型能做到 Zero-shot？
>
> **A:** 因为互联网文本中自然包含了各种任务的示例。例如，网页上有大量 "Translate X to Y: ..." 格式的文本，语言模型在学习预测下一个 token 时，隐式地学会了翻译。WebText 数据集的多样性是关键。

### 2.3 任务条件化：$P(\text{output} \mid \text{input}, \text{task})$

GPT-2 将所有任务统一为条件语言建模：

$$
\boxed{P(\text{output} \mid x_1, \ldots, x_n, \text{task\_token}_1, \ldots, \text{task\_token}_m)}
$$

具体来说，不同任务通过不同的**提示格式（prompt format）**来区分：

**翻译任务**：
$$
P(\text{"fromage"} \mid \text{"translate English to French: cheese ="})
$$

**摘要任务**：
$$
P(\text{summary tokens} \mid \text{article tokens}, \text{"TL;DR:"})
$$

**问答任务**：
$$
P(\text{answer} \mid \text{context}, \text{"Q:"}, \text{question}, \text{"A:"})
$$

**数学形式化**：

设输入为 $c = (c_1, \ldots, c_m)$（上下文 + 任务描述），则 GPT-2 生成输出 $y = (y_1, \ldots, y_k)$ 的概率为：

$$
P(y \mid c) = \prod_{t=1}^{k} P(y_t \mid c_1, \ldots, c_m, y_1, \ldots, y_{t-1})
$$

这与标准自回归语言模型完全一致——GPT-2 不需要任何架构修改就能处理不同任务。

---

## 3. GPT-2 架构的数学描述

### 3.1 输入表示：Token 嵌入 + 位置嵌入

与 BERT 使用三种嵌入不同，GPT-2 仅使用**两种嵌入**（没有段落嵌入）：

$$
\boxed{E_{\text{input}}(t) = E_{\text{token}}(x_t) + E_{\text{position}}(t)}
$$

**1. Token 嵌入** $E_{\text{token}} \in \mathbb{R}^{|V| \times d}$

GPT-2 使用 **Byte Pair Encoding (BPE)** 进行分词，词表大小 $|V| = 50{,}257$：

$$
E_{\text{token}}(x_t) = E_{\text{token}}[x_t] \in \mathbb{R}^d
$$

**BPE vs WordPiece**：

| 特性 | BPE (GPT-2) | WordPiece (BERT) |
|------|-------------|-----------------|
| 合并策略 | 最频繁的字节对 | 最大化语言模型似然 |
| 词表大小 | 50,257 | 30,522 |
| 基本单位 | 字节 (byte) | Unicode 字符 |
| OOV 处理 | 任意文本可编码 | 需要 `[UNK]` token |

GPT-2 使用**字节级 BPE**，这意味着它可以编码任意 UTF-8 文本，完全不会出现 OOV（词表外）问题。

**2. 位置嵌入** $E_{\text{position}} \in \mathbb{R}^{L_{\max} \times d}$

与 BERT 相同，GPT-2 使用**可学习的位置嵌入**：

$$
E_{\text{position}}(t) = E_{\text{position}}[t] \in \mathbb{R}^d
$$

其中 $L_{\max} = 1024$ 是最大上下文长度（GPT-1 为 512）。

### 3.2 因果注意力掩码 (Causal Mask)

GPT-2 的核心约束：**每个位置只能关注自身及之前的位置**。这通过因果掩码实现。

**因果掩码的定义**：

$$
\boxed{M^{\text{causal}}_{ij} = \begin{cases} 1 & \text{if } j \leq i \\ 0 & \text{if } j > i \end{cases}}
$$

即下三角矩阵（包含对角线）：

$$
M^{\text{causal}} = \begin{pmatrix}
1 & 0 & 0 & \cdots & 0 \\
1 & 1 & 0 & \cdots & 0 \\
1 & 1 & 1 & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & 1 & 1 & \cdots & 1
\end{pmatrix} \in \mathbb{R}^{n \times n}
$$

**掩码的应用方式**：

在计算注意力分数时，掩码为 0 的位置被设为 $-\infty$：

$$
\text{score}_{ij} = \begin{cases}
\frac{q_i^\top k_j}{\sqrt{d_k}} & \text{if } j \leq i \\
-\infty & \text{if } j > i
\end{cases}
$$

经过 Softmax 后，$-\infty$ 变为 0：

$$
\alpha_{ij} = \frac{\exp(\text{score}_{ij})}{\sum_{m=1}^{n} \exp(\text{score}_{im})} = \begin{cases}
> 0 & \text{if } j \leq i \\
0 & \text{if } j > i
\end{cases}
$$

**可视化对比**（$n = 5$）：

```
BERT (全连接):              GPT-2 (因果掩码):
1  1  1  1  1              1  0  0  0  0
1  1  1  1  1              1  1  0  0  0
1  1  1  1  1              1  1  1  0  0
1  1  1  1  1              1  1  1  1  0
1  1  1  1  1              1  1  1  1  1
位置 i 可以看到所有位置     位置 i 只能看到 j ≤ i
```

> **为什么需要因果掩码？**
>
> 在训练时，GPT-2 需要并行预测所有位置的下一个 token。如果位置 $i$ 能看到位置 $i+1$，就相当于"偷看答案"。因果掩码确保了每个位置只能利用过去的信息，与自回归生成时的情况一致。

**因果掩码的数学意义**：

因果掩码等价于对注意力矩阵施加一个约束：

$$
\alpha_{ij} = 0, \quad \forall j > i
$$

这意味着位置 $i$ 的输出仅是位置 $1, 2, \ldots, i$ 的值向量的加权和：

$$
\boxed{z_i = \sum_{j=1}^{i} \alpha_{ij} v_j \quad (\text{而非 } \sum_{j=1}^{n} \alpha_{ij} v_j)}
$$

### 3.3 Pre-Norm Transformer 解码器层

GPT-2 相比原始 Transformer 和 GPT-1 的一个重要改进是使用了 **Pre-Norm**（LayerNorm 前置）而非 Post-Norm：

**Post-Norm（原始 Transformer / GPT-1 / BERT）**：

$$
\tilde{h} = \text{LayerNorm}(x + \text{SubLayer}(x))
$$

**Pre-Norm（GPT-2）**：

$$
\boxed{\tilde{h} = x + \text{SubLayer}(\text{LayerNorm}(x))}
$$

**完整的 GPT-2 解码器层**：

$$
\begin{aligned}
a^{(l)} &= x^{(l-1)} + \text{CausalMultiHeadAttn}\left(\text{LayerNorm}(x^{(l-1)})\right) \\
x^{(l)} &= a^{(l)} + \text{FFN}\left(\text{LayerNorm}(a^{(l)})\right)
\end{aligned}
$$

其中 FFN 使用 GELU 激活函数：

$$
\text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2
$$

**最终输出**需要一个额外的 LayerNorm：

$$
h = \text{LayerNorm}(x^{(L)})
$$

> **Q:** Pre-Norm vs Post-Norm 有什么区别？
>
> **A:** Pre-Norm 在深层网络中训练更稳定。数学上，Post-Norm 的梯度需要通过 LayerNorm 传播，而 Pre-Norm 的残差连接提供了一条"干净"的梯度通路。对于 GPT-2 的 48 层网络，这一改进至关重要。

**Pre-Norm 的梯度优势**：

对于 Pre-Norm，从第 $L$ 层到第 $l$ 层的梯度：

$$
\frac{\partial x^{(L)}}{\partial x^{(l)}} = I + \sum_{k=l+1}^{L} \frac{\partial \text{SubLayer}_k}{\partial x^{(l)}}
$$

恒等矩阵 $I$ 确保梯度不会消失。

### 3.4 GPT-2 各版本配置

GPT-2 发布了 4 个版本，参数量从 117M 到 1.5B：

| 参数 | GPT-2 Small | GPT-2 Medium | GPT-2 Large | GPT-2 XL |
|------|------------|-------------|-------------|----------|
| 层数 $L$ | 12 | 24 | 36 | 48 |
| 隐藏维度 $d$ | 768 | 1024 | 1280 | 1600 |
| 注意力头数 $A$ | 12 | 16 | 20 | 25 |
| 每头维度 $d_k$ | 64 | 64 | 64 | 64 |
| FFN 维度 $d_{ff}$ | 3072 | 4096 | 5120 | 6400 |
| 上下文长度 | 1024 | 1024 | 1024 | 1024 |
| 词表大小 $|V|$ | 50,257 | 50,257 | 50,257 | 50,257 |
| 总参数量 | 117M | 345M | 774M | **1,558M** |

**参数量估算（GPT-2 XL）**：

嵌入层：
$$
P_{\text{emb}} = |V| \cdot d + L_{\max} \cdot d = 50257 \times 1600 + 1024 \times 1600 \approx 82.0\text{M}
$$

单层 Transformer（Pre-Norm 解码器）：
$$
P_{\text{layer}} = \underbrace{4 \cdot d^2}_{\text{MultiHead } (W^Q, W^K, W^V, W^O)} + \underbrace{2 \cdot d \cdot d_{ff}}_{\text{FFN}} + \underbrace{4d + 2d_{ff} + 2d}_{\text{biases + LN}} \approx 30.7\text{M}
$$

总计：
$$
P_{\text{total}} = P_{\text{emb}} + L \cdot P_{\text{layer}} + P_{\text{final\_LN}} \approx 82.0\text{M} + 48 \times 30.7\text{M} + 0.003\text{M} \approx 1{,}556\text{M}
$$

> **注意**：GPT-2 使用**权重共享 (weight tying)**——输出层的投影矩阵与 token 嵌入矩阵共享，因此不需要额外的 $|V| \times d$ 参数。

---

## 4. 自回归损失函数与梯度推导

### 4.1 语言模型交叉熵损失

**符号定义**：

- $x = (x_1, x_2, \ldots, x_n)$：输入 token 序列
- $h_t \in \mathbb{R}^d$：位置 $t$ 的最终隐藏状态
- $W_{\text{emb}} \in \mathbb{R}^{|V| \times d}$：token 嵌入矩阵（输出层共享）
- $\ell(\cdot, \cdot)$：交叉熵损失函数

**前向计算**：

1. 获取隐藏状态：$H = \text{GPT2}(x_1, \ldots, x_{n-1}) \in \mathbb{R}^{(n-1) \times d}$
2. 对每个位置 $t$ 预测下一个 token $x_{t+1}$：

$$
\text{logits}_t = W_{\text{emb}} \cdot h_t \in \mathbb{R}^{|V|}
$$

注意这里使用了权重共享：输出投影矩阵就是 token 嵌入矩阵 $W_{\text{emb}}$。

3. Softmax 得到概率分布：

$$
P(x_{t+1} = w \mid x_{\leq t}) = \frac{\exp(\text{logits}_{t,w})}{\sum_{w'=1}^{|V|} \exp(\text{logits}_{t,w'})}, \quad w \in \{1, \ldots, |V|\}
$$

**语言模型损失**（交叉熵）：

$$
\boxed{\mathcal{L}_{\text{LM}} = -\frac{1}{n-1} \sum_{t=1}^{n-1} \log P_\theta(x_{t+1} \mid x_1, \ldots, x_t)}
$$

展开为：

$$
\mathcal{L}_{\text{LM}} = -\frac{1}{n-1} \sum_{t=1}^{n-1} \left[ \text{logits}_{t, x_{t+1}} - \log \sum_{w=1}^{|V|} \exp(\text{logits}_{t, w}) \right]
$$

> **与 BERT MLM 的区别**：
> - BERT MLM：只在 $|\mathcal{M}| \approx 0.15n$ 个掩码位置计算损失
> - GPT-2 LM：在所有 $n-1$ 个位置计算损失
> - GPT-2 的训练信号密度更高（每个 token 都贡献损失）

### 4.2 困惑度 (Perplexity)

困惑度是评估语言模型质量的标准指标，直觉上表示"模型在每一步平均有多少个等概率的选择"：

$$
\boxed{\text{PPL} = \exp\left(-\frac{1}{n} \sum_{t=1}^{n} \log P_\theta(x_t \mid x_{<t})\right) = \exp(\mathcal{L}_{\text{LM}})}
$$

**直觉解释**：

| PPL | 含义 |
|-----|------|
| 1 | 完美预测，每个 token 都确定 |
| 10 | 平均每步在 10 个 token 间犹豫 |
| 100 | 几乎随机猜测 |
| $|V|$ | 均匀分布（完全不知道） |

**GPT-2 在不同数据集上的 PPL**：

| 数据集 | GPT-2 Small | GPT-2 XL | SOTA (当时) |
|--------|------------|----------|------------|
| Penn Treebank | 65.85 | 35.76 | — |
| WikiText-2 | 29.41 | **18.34** | 前 SOTA: 33.0 |
| WikiText-103 | 26.37 | **17.48** | 前 SOTA: 18.3 |
| 1 Billion Word | 43.31 | — | — |

**PPL 与交叉熵的关系**：

$$
\text{PPL} = 2^{H(p, q)} = \exp(H_{\text{nat}}(p, q))
$$

其中 $H_{\text{nat}}$ 是以自然对数计算的交叉熵（nats），$H$ 是以 2 为底的交叉熵（bits）。

### 4.3 梯度分析与权重共享

**权重共享 (Weight Tying)** 是 GPT-2 的一个重要设计：输入嵌入矩阵 $W_{\text{emb}}$ 同时用作输出层的投影矩阵。

**数学表达**：

输入：$e_t = W_{\text{emb}}[x_t] \in \mathbb{R}^d$

输出：$\text{logits}_t = W_{\text{emb}} \cdot h_t \in \mathbb{R}^{|V|}$

**权重共享下的梯度分析**：

$W_{\text{emb}}$ 同时接收来自输入和输出两个方向的梯度：

$$
\frac{\partial \mathcal{L}}{\partial W_{\text{emb}}} = \underbrace{\frac{\partial \mathcal{L}}{\partial W_{\text{emb}}}^{\text{(output)}}}_{\text{来自输出层}} + \underbrace{\frac{\partial \mathcal{L}}{\partial W_{\text{emb}}}^{\text{(input)}}}_{\text{来自嵌入查表}}
$$

**输出层梯度**：

对于位置 $t$，令 $p_t = \text{softmax}(\text{logits}_t)$，$y_t$ 为 one-hot 标签：

$$
\frac{\partial \mathcal{L}_t}{\partial W_{\text{emb}}} = (p_t - y_t) \cdot h_t^\top \in \mathbb{R}^{|V| \times d}
$$

汇总所有位置：

$$
\boxed{\frac{\partial \mathcal{L}}{\partial W_{\text{emb}}}^{\text{(output)}} = \frac{1}{n-1} \sum_{t=1}^{n-1} (p_t - y_t) \cdot h_t^\top}
$$

> **直觉**：梯度的方向使得正确 token 对应的嵌入向量更接近隐藏状态 $h_t$，错误 token 的嵌入向量远离 $h_t$。

**权重共享的好处**：

1. **参数效率**：节省 $|V| \times d$ 个参数（GPT-2 XL 中约 80M）
2. **语义一致性**：输入和输出空间共享同一个语义空间
3. **正则化效果**：防止输出层过拟合

---

## 5. 训练优化方法总结

### 5.1 预训练策略与数据集

**WebText 数据集**：

GPT-2 团队创建了一个全新的数据集 WebText：

| 属性 | 值 |
|------|-----|
| 来源 | Reddit 上获得 ≥3 karma 的外链 |
| 文档数 | ~8 million |
| 去重后大小 | ~40 GB 文本 |
| 目的 | 高质量、多样化的网页文本 |

**为什么用 Reddit karma 过滤？**

Reddit karma 是一种众包质量过滤——获得 ≥3 upvotes 的链接更可能指向有价值的内容。

**数据预处理**：

1. 使用 [Dragnet](https://github.com/dragnet-org/dragnet) 和 [Newspaper](https://github.com/codelucas/newspaper) 提取正文
2. 去除 Wikipedia 文档（避免与测试集重叠）
3. 字节级 BPE 分词

### 5.2 优化器与学习率调度

**Adam 优化器**：

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \\
\theta_{t+1} &= \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\end{aligned}
$$

**超参数设置**（GPT-2 XL）：

| 超参数 | 值 |
|--------|-----|
| 学习率 $\eta$ | $2.5 \times 10^{-4}$ |
| Batch Size | 512 |
| 序列长度 | 1024 |
| 总 token 数 | ~10B |
| Adam $\beta_1$ | 0.9 |
| Adam $\beta_2$ | 0.999 |
| $\epsilon$ | $1 \times 10^{-8}$ |
| 梯度裁剪 | 全局范数 1.0 |

**学习率调度**（余弦退火 + 线性预热）：

$$
\eta(t) = \begin{cases}
\eta_{\max} \cdot \frac{t}{t_{\text{warmup}}} & \text{if } t < t_{\text{warmup}} \\
\eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t - t_{\text{warmup}}}{T - t_{\text{warmup}}} \cdot \pi\right)\right) & \text{if } t \geq t_{\text{warmup}}
\end{cases}
$$

### 5.3 训练稳定性技巧

**1. 残差连接的初始化缩放**

GPT-2 对残差路径上的投影层进行特殊初始化：

$$
W \sim \mathcal{N}\left(0, \frac{0.02}{\sqrt{2L}}\right)
$$

其中 $L$ 是层数。因子 $\frac{1}{\sqrt{2L}}$ 确保经过 $L$ 层残差累加后，输出的方差不会爆炸。

**2. Pre-Norm 架构**

如 3.3 节所述，LayerNorm 放在子层之前而非之后，提高深层网络的训练稳定性。

**3. 最终 LayerNorm**

在最后一层 Transformer 输出后，额外添加一个 LayerNorm：

$$
h_{\text{final}} = \text{LayerNorm}(x^{(L)})
$$

这是 GPT-2 的特有设计，进一步稳定输出分布。

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
        GELU(x) = x · Φ(x) ≈ 0.5x(1 + tanh[√(2/π)(x + 0.044715x³)])
    """
    return 0.5 * x * (1.0 + np.tanh(
        np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))
    ))


def layer_norm(x, gamma, beta, eps=1e-5):
    """
    层归一化
    
    数学公式:
        LayerNorm(x) = γ · (x - μ) / √(σ² + ε) + β
    
    参数:
        x: 输入，形状 (..., d_model)
        gamma: 缩放参数，形状 (d_model,)
        beta: 偏移参数，形状 (d_model,)
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta


def causal_mask(seq_len):
    """
    生成因果注意力掩码（下三角矩阵）
    
    数学公式:
        M_ij = 1 if j ≤ i, else 0
    
    参数:
        seq_len: 序列长度 n
    
    返回:
        mask: (1, 1, seq_len, seq_len) 因果掩码
    """
    # 下三角矩阵，包含对角线
    mask = np.tril(np.ones((seq_len, seq_len), dtype=np.float32))
    # 扩展维度用于广播: (1, 1, n, n)
    return mask[np.newaxis, np.newaxis, :, :]


def causal_self_attention_numpy(Q, K, V, mask):
    """
    因果自注意力 (NumPy)
    
    数学公式:
        CausalAttention(Q, K, V) = softmax(QK^T / √d_k + M_causal) V
        其中 M_causal 中被掩码位置为 -∞
    
    参数:
        Q: 查询，形状 (batch, heads, seq_len, d_k)
        K: 键，形状 (batch, heads, seq_len, d_k)
        V: 值，形状 (batch, heads, seq_len, d_k)
        mask: 因果掩码，形状 (1, 1, seq_len, seq_len)
    
    返回:
        output: (batch, heads, seq_len, d_k)
        weights: (batch, heads, seq_len, seq_len)
    """
    d_k = Q.shape[-1]
    
    # 1. 计算注意力分数: QK^T / √d_k
    scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    # scores: (batch, heads, seq_len, seq_len)
    
    # 2. 应用因果掩码: 未来位置设为 -∞
    scores = np.where(mask == 0, -1e9, scores)
    
    # 3. Softmax 归一化
    weights = softmax(scores, axis=-1)
    # weights: (batch, heads, seq_len, seq_len)
    
    # 4. 加权求和
    output = np.matmul(weights, V)
    # output: (batch, heads, seq_len, d_k)
    
    return output, weights


def gpt2_embedding_numpy(token_ids, token_emb, position_emb):
    """
    GPT-2 输入嵌入 (NumPy)
    
    数学公式:
        E(t) = E_token(x_t) + E_position(t)
    
    参数:
        token_ids: token 索引，形状 (batch, seq_len)
        token_emb: token 嵌入矩阵，形状 (vocab_size, d_model)
        position_emb: 位置嵌入矩阵，形状 (max_len, d_model)
    """
    batch_size, seq_len = token_ids.shape
    
    # Token 嵌入
    tok_emb = token_emb[token_ids]  # (batch, seq_len, d_model)
    
    # 位置嵌入
    positions = np.arange(seq_len)
    pos_emb = position_emb[positions]  # (seq_len, d_model)
    
    # 两种嵌入相加
    embeddings = tok_emb + pos_emb[np.newaxis, :, :]  # (batch, seq_len, d_model)
    
    return embeddings


def gpt2_decoder_layer_numpy(x, W_Q, W_K, W_V, W_O, W_1, b_1, W_2, b_2,
                              ln1_gamma, ln1_beta, ln2_gamma, ln2_beta,
                              num_heads, mask):
    """
    GPT-2 解码器层 (Pre-Norm) (NumPy)
    
    数学公式:
        a = x + CausalMultiHeadAttn(LayerNorm(x))
        output = a + FFN(LayerNorm(a))
    
    参数:
        x: 输入，形状 (batch, seq_len, d_model)
        W_Q, W_K, W_V: 注意力投影，形状 (d_model, d_model)
        W_O: 输出投影，形状 (d_model, d_model)
        W_1: FFN 第一层，形状 (d_model, d_ff)
        b_1: FFN 第一层偏置，形状 (d_ff,)
        W_2: FFN 第二层，形状 (d_ff, d_model)
        b_2: FFN 第二层偏置，形状 (d_model,)
        ln1_gamma, ln1_beta: 第一个 LayerNorm 参数
        ln2_gamma, ln2_beta: 第二个 LayerNorm 参数
        num_heads: 注意力头数
        mask: 因果掩码
    """
    batch_size, seq_len, d_model = x.shape
    d_k = d_model // num_heads
    
    # === 1. Pre-Norm + 因果自注意力 ===
    # LayerNorm
    x_norm = layer_norm(x, ln1_gamma, ln1_beta)
    
    # 线性投影
    Q = np.dot(x_norm, W_Q)  # (batch, seq_len, d_model)
    K = np.dot(x_norm, W_K)
    V = np.dot(x_norm, W_V)
    
    # 分割多头: (batch, seq_len, d_model) → (batch, heads, seq_len, d_k)
    Q = Q.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    K = K.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    V = V.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    
    # 因果自注意力
    attn_out, attn_weights = causal_self_attention_numpy(Q, K, V, mask)
    
    # 拼接多头: (batch, heads, seq_len, d_k) → (batch, seq_len, d_model)
    attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
    
    # 输出投影
    attn_out = np.dot(attn_out, W_O)
    
    # 残差连接
    a = x + attn_out
    
    # === 2. Pre-Norm + FFN ===
    # LayerNorm
    a_norm = layer_norm(a, ln2_gamma, ln2_beta)
    
    # FFN: GELU(x W_1 + b_1) W_2 + b_2
    ffn_hidden = gelu(np.dot(a_norm, W_1) + b_1)  # (batch, seq_len, d_ff)
    ffn_out = np.dot(ffn_hidden, W_2) + b_2        # (batch, seq_len, d_model)
    
    # 残差连接
    output = a + ffn_out
    
    return output, attn_weights


def lm_loss_numpy(hidden_states, token_emb, targets):
    """
    语言模型损失计算 (NumPy) — 使用权重共享
    
    数学公式:
        L_LM = -1/(n-1) Σ_{t=1}^{n-1} log P(x_{t+1} | x_{≤t})
        logits_t = W_emb · h_t  (权重共享)
    
    参数:
        hidden_states: GPT-2 输出，形状 (batch, seq_len, d_model)
        token_emb: token 嵌入矩阵，形状 (vocab_size, d_model)，同时用作输出投影
        targets: 目标 token ID，形状 (batch, seq_len)
    """
    batch_size, seq_len, d_model = hidden_states.shape
    total_loss = 0.0
    count = 0
    
    for b in range(batch_size):
        for t in range(seq_len - 1):
            # 位置 t 预测位置 t+1
            h = hidden_states[b, t]  # (d_model,)
            
            # 权重共享: logits = W_emb · h
            logits = np.dot(token_emb, h)  # (vocab_size,)
            
            # Softmax
            probs = softmax(logits)
            
            # 交叉熵
            target_id = targets[b, t + 1]
            total_loss -= np.log(probs[target_id] + 1e-10)
            count += 1
    
    return total_loss / max(count, 1)


# ========== 温度采样与 Top-k/Top-p 采样 (NumPy) ==========

def temperature_sampling_numpy(logits, temperature=1.0):
    """
    温度采样
    
    数学公式:
        P(x_i) = exp(z_i / T) / Σ exp(z_j / T)
    
    参数:
        logits: 原始 logits，形状 (vocab_size,)
        temperature: 温度参数 T > 0
    """
    scaled_logits = logits / temperature
    probs = softmax(scaled_logits)
    return np.random.choice(len(probs), p=probs)


def top_k_sampling_numpy(logits, k=50, temperature=1.0):
    """
    Top-k 采样
    
    步骤:
        1. 取 logits 中最大的 k 个值
        2. 将其余位置设为 -∞
        3. 对保留的 k 个值进行温度缩放 + softmax + 采样
    
    参数:
        logits: 原始 logits，形状 (vocab_size,)
        k: 保留的 top-k 个 token
        temperature: 温度参数
    """
    # 找到第 k 大的值作为阈值
    top_k_indices = np.argsort(logits)[-k:]
    threshold = logits[top_k_indices[0]]
    
    # 过滤: 低于阈值的设为 -∞
    filtered_logits = np.where(logits >= threshold, logits, -1e9)
    
    # 温度缩放 + 采样
    return temperature_sampling_numpy(filtered_logits, temperature)


def top_p_sampling_numpy(logits, p=0.9, temperature=1.0):
    """
    Top-p (Nucleus) 采样
    
    步骤:
        1. 将 logits 转为概率并按降序排列
        2. 计算累积概率
        3. 保留累积概率 ≤ p 的 token（至少保留一个）
        4. 对保留的 token 进行重新归一化 + 采样
    
    参数:
        logits: 原始 logits，形状 (vocab_size,)
        p: 累积概率阈值 (nucleus)
        temperature: 温度参数
    """
    # 温度缩放
    scaled_logits = logits / temperature
    probs = softmax(scaled_logits)
    
    # 按概率降序排列
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    
    # 计算累积概率
    cumulative_probs = np.cumsum(sorted_probs)
    
    # 找到累积概率超过 p 的位置
    cutoff_index = np.searchsorted(cumulative_probs, p) + 1
    cutoff_index = min(cutoff_index, len(sorted_probs))
    
    # 保留 top-p 内的 token
    top_p_indices = sorted_indices[:cutoff_index]
    top_p_probs = probs[top_p_indices]
    
    # 重新归一化
    top_p_probs = top_p_probs / top_p_probs.sum()
    
    # 采样
    chosen = np.random.choice(top_p_indices, p=top_p_probs)
    return chosen


# ========== 测试 NumPy 实现 ==========
if __name__ == "__main__":
    np.random.seed(42)
    batch_size, seq_len, d_model, vocab_size = 2, 8, 64, 100
    num_heads = 4
    d_ff = 256
    d_k = d_model // num_heads
    
    # 初始化参数
    token_emb = np.random.randn(vocab_size, d_model) * 0.02
    position_emb = np.random.randn(1024, d_model) * 0.02
    
    # 创建输入
    token_ids = np.random.randint(0, vocab_size, (batch_size, seq_len))
    
    # 测试嵌入
    embeddings = gpt2_embedding_numpy(token_ids, token_emb, position_emb)
    print(f"嵌入输出形状: {embeddings.shape}")  # (2, 8, 64)
    
    # 测试因果掩码
    mask = causal_mask(seq_len)
    print(f"因果掩码形状: {mask.shape}")  # (1, 1, 8, 8)
    print(f"因果掩码 (seq_len=4):\n{causal_mask(4)[0, 0]}")
    
    # 测试因果自注意力
    Q = embeddings[:, np.newaxis, :, :d_k].repeat(num_heads, axis=1)
    K = Q.copy()
    V = Q.copy()
    attn_out, attn_weights = causal_self_attention_numpy(Q, K, V, mask)
    print(f"因果注意力权重行和: {attn_weights[0, 0, 0].sum():.4f}")  # 1.0
    
    # 验证因果性: 位置 0 不应该关注位置 1, 2, ...
    print(f"位置 0 对位置 1 的注意力: {attn_weights[0, 0, 0, 1]:.6f}")  # ≈ 0
    print(f"位置 2 对位置 0 的注意力: {attn_weights[0, 0, 2, 0]:.6f}")  # > 0
    
    # 测试语言模型损失
    lm_loss = lm_loss_numpy(embeddings, token_emb, token_ids)
    print(f"LM 损失: {lm_loss:.4f} (初始化时约 log(100) ≈ {np.log(100):.4f})")
    print(f"困惑度: {np.exp(lm_loss):.2f}")
    
    # 测试采样方法
    test_logits = np.array([2.0, 1.5, 0.5, -0.5, -1.0, -2.0])
    print(f"\n温度采样 (T=0.5): token={temperature_sampling_numpy(test_logits, 0.5)}")
    print(f"温度采样 (T=2.0): token={temperature_sampling_numpy(test_logits, 2.0)}")
    print(f"Top-k 采样 (k=3): token={top_k_sampling_numpy(test_logits, k=3)}")
    print(f"Top-p 采样 (p=0.9): token={top_p_sampling_numpy(test_logits, p=0.9)}")
    
    print(f"\nGELU([-2,-1,0,1,2]) = {gelu(np.array([-2.,-1.,0.,1.,2.]))}")
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
        E(t) = E_token(x_t) + E_position(t)
        (无段落嵌入，无 LayerNorm，因为 Pre-Norm 在每层内部)
    
    参数:
        vocab_size: 词表大小 |V| = 50257
        d_model: 嵌入维度 d
        max_len: 最大序列长度 = 1024
        dropout: dropout 概率
    """
    def __init__(self, vocab_size: int, d_model: int, max_len: int = 1024,
                 dropout: float = 0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        参数:
            token_ids: (batch, seq_len) token 索引
        返回:
            embeddings: (batch, seq_len, d_model) 输入嵌入
        """
        seq_len = token_ids.size(1)
        position_ids = torch.arange(seq_len, device=token_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(token_ids)
        
        # Token 嵌入 + 位置嵌入
        embeddings = (
            self.token_embedding(token_ids) +
            self.position_embedding(position_ids)
        )
        
        return self.dropout(embeddings)


class CausalMultiHeadAttention(nn.Module):
    """
    因果多头自注意力 (GPT-2 版本)
    
    数学公式:
        CausalMultiHead(x) = Concat(head_1, ..., head_h) W^O
        head_i = CausalAttention(xW_i^Q, xW_i^K, xW_i^V)
        CausalAttention(Q, K, V) = softmax(QK^T / √d_k + M_causal) V
    
    关键区别: 使用因果掩码，位置 i 只能关注 j ≤ i
    
    参数:
        d_model: 模型维度
        num_heads: 注意力头数
        max_len: 最大序列长度（用于预计算因果掩码）
        dropout: dropout 概率
    """
    def __init__(self, d_model: int, num_heads: int, max_len: int = 1024,
                 dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 合并 Q, K, V 投影为一个线性层 (效率更高)
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # 预计算因果掩码 (下三角矩阵)
        # 注册为 buffer，不参与梯度计算
        mask = torch.tril(torch.ones(max_len, max_len))
        self.register_buffer("causal_mask", mask.view(1, 1, max_len, max_len))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        参数:
            x: (batch, seq_len, d_model) 输入
        返回:
            output: (batch, seq_len, d_model) 注意力输出
            weights: (batch, heads, seq_len, seq_len) 注意力权重
        """
        batch_size, seq_len, _ = x.size()
        
        # 合并投影: (batch, seq_len, 3 * d_model)
        qkv = self.c_attn(x)
        Q, K, V = qkv.split(self.d_model, dim=-1)
        
        # 分割多头: (batch, seq_len, d_model) → (batch, heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 缩放点积注意力 + 因果掩码
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores: (batch, heads, seq_len, seq_len)
        
        # 应用因果掩码
        causal = self.causal_mask[:, :, :seq_len, :seq_len]
        scores = scores.masked_fill(causal == 0, float("-inf"))
        
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)
        
        # 加权求和
        context = torch.matmul(weights, V)
        # context: (batch, heads, seq_len, d_k)
        
        # 拼接多头
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # 输出投影 + 残差 dropout
        output = self.resid_dropout(self.c_proj(context))
        
        return output, weights


class GPT2FFN(nn.Module):
    """
    GPT-2 前馈网络 (使用 GELU)
    
    数学公式:
        FFN(x) = GELU(x W_1 + b_1) W_2 + b_2
    
    参数:
        d_model: 输入/输出维度
        d_ff: 隐藏层维度 (= 4 * d_model)
        dropout: dropout 概率
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.c_fc = nn.Linear(d_model, d_ff)
        self.c_proj = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)         # (batch, seq_len, d_ff)
        x = F.gelu(x)            # GELU 激活
        x = self.c_proj(x)       # (batch, seq_len, d_model)
        x = self.dropout(x)
        return x


class GPT2DecoderLayer(nn.Module):
    """
    GPT-2 解码器层 (Pre-Norm)
    
    结构 (Pre-Norm):
        x → LayerNorm → [CausalAttention] → + → LayerNorm → [FFN] → + → output
            └──────────── residual ────────┘    └────── residual ──┘
    
    数学公式:
        a = x + CausalMultiHeadAttn(LayerNorm(x))
        output = a + FFN(LayerNorm(a))
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 max_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model, eps=1e-5)
        self.attn = CausalMultiHeadAttention(d_model, num_heads, max_len, dropout)
        self.ln_2 = nn.LayerNorm(d_model, eps=1e-5)
        self.ffn = GPT2FFN(d_model, d_ff, dropout)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        参数:
            x: (batch, seq_len, d_model) 输入
        返回:
            output: (batch, seq_len, d_model)
            attn_weights: (batch, heads, seq_len, seq_len)
        """
        # 1. Pre-Norm + 因果自注意力 + 残差
        attn_output, attn_weights = self.attn(self.ln_1(x))
        x = x + attn_output
        
        # 2. Pre-Norm + FFN + 残差
        ffn_output = self.ffn(self.ln_2(x))
        x = x + ffn_output
        
        return x, attn_weights


class GPT2Model(nn.Module):
    """
    完整 GPT-2 模型
    
    结构:
        Input → [Embedding] → [Decoder x L] → [Final LayerNorm] → [LM Head]
    
    关键设计:
        - Pre-Norm (LayerNorm 在子层之前)
        - 因果注意力掩码 (下三角)
        - 权重共享 (输入嵌入 = 输出投影)
        - 最终 LayerNorm
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
        self.max_len = max_len
        
        # 嵌入层
        self.embedding = GPT2Embedding(vocab_size, d_model, max_len, dropout)
        
        # Transformer 解码器层
        self.layers = nn.ModuleList([
            GPT2DecoderLayer(d_model, num_heads, d_ff, max_len, dropout)
            for _ in range(num_layers)
        ])
        
        # 最终 LayerNorm (GPT-2 特有)
        self.ln_f = nn.LayerNorm(d_model, eps=1e-5)
        
        # 语言模型头 (权重共享)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # 权重共享: lm_head.weight = embedding.token_embedding.weight
        self.lm_head.weight = self.embedding.token_embedding.weight
        
        # 参数初始化
        self._init_weights()
    
    def _init_weights(self):
        """
        GPT-2 权重初始化
        
        - 线性层: N(0, 0.02)
        - 残差路径投影: N(0, 0.02 / √(2L))
        - 嵌入层: N(0, 0.02)
        - LayerNorm: γ=1, β=0
        """
        num_layers = len(self.layers)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                # 残差路径上的投影层使用缩放初始化
                if "c_proj" in name:
                    nn.init.normal_(
                        module.weight, mean=0.0,
                        std=0.02 / math.sqrt(2 * num_layers)
                    )
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        token_ids: torch.Tensor,        # (batch, seq_len)
        targets: Optional[torch.Tensor] = None  # (batch, seq_len)
    ) -> dict:
        """
        参数:
            token_ids: (batch, seq_len) 输入 token 索引
            targets: (batch, seq_len) 目标 token 索引 (训练时提供)
        
        返回:
            字典包含:
            - logits: (batch, seq_len, vocab_size) 预测 logits
            - loss: 标量，语言模型损失 (仅训练时)
            - hidden_states: (batch, seq_len, d_model) 最终隐藏状态
        """
        # 1. 嵌入层
        x = self.embedding(token_ids)
        # x: (batch, seq_len, d_model)
        
        # 2. Transformer 解码器层
        all_attn_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x)
            all_attn_weights.append(attn_weights)
        
        # 3. 最终 LayerNorm
        x = self.ln_f(x)
        # x: (batch, seq_len, d_model)
        
        # 4. 语言模型头 (权重共享)
        logits = self.lm_head(x)
        # logits: (batch, seq_len, vocab_size)
        
        # 5. 计算损失 (如果提供了目标)
        loss = None
        if targets is not None:
            # 将 logits 和 targets 对齐:
            # logits[:, t, :] 预测 targets[:, t+1]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = targets[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_targets.view(-1)
            )
        
        return {
            "logits": logits,
            "loss": loss,
            "hidden_states": x,
            "attention_weights": all_attn_weights,
        }
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,        # (1, prefix_len)
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        自回归文本生成
        
        数学公式:
            x_{t+1} ~ P(· | x_1, ..., x_t) (经温度/Top-k/Top-p 调整)
        
        参数:
            input_ids: (1, prefix_len) 前缀 token
            max_new_tokens: 最大生成 token 数
            temperature: 温度参数 T
            top_k: Top-k 采样的 k 值
            top_p: Top-p 采样的 p 值
        
        返回:
            generated: (1, prefix_len + max_new_tokens) 生成的完整序列
        """
        self.eval()
        generated = input_ids
        
        for _ in range(max_new_tokens):
            # 截断到最大上下文长度
            input_truncated = generated[:, -self.max_len:]
            
            # 前向传播
            outputs = self.forward(input_truncated)
            logits = outputs["logits"][:, -1, :]  # (1, vocab_size)
            
            # 温度缩放
            logits = logits / temperature
            
            # Top-k 过滤
            if top_k is not None:
                top_k_values, _ = torch.topk(logits, top_k)
                threshold = top_k_values[:, -1].unsqueeze(-1)
                logits = torch.where(
                    logits < threshold, torch.tensor(float("-inf")), logits
                )
            
            # Top-p (Nucleus) 过滤
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(
                    logits, descending=True
                )
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                # 移除累积概率超过 p 的 token
                sorted_mask = cumulative_probs - F.softmax(
                    sorted_logits, dim=-1
                ) >= top_p
                sorted_logits[sorted_mask] = float("-inf")
                # 恢复原始顺序
                logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)
            
            # 采样
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 追加到序列
            generated = torch.cat([generated, next_token], dim=-1)
        
        return generated


# ========== 完整测试 ==========
if __name__ == "__main__":
    # 超参数（缩小版，用于测试）
    vocab_size = 1000
    d_model = 128
    num_heads = 4
    num_layers = 4
    d_ff = 512
    max_len = 64
    batch_size = 4
    seq_len = 32
    
    print("=" * 60)
    print("GPT-2 模型测试")
    print("=" * 60)
    
    # 1. 创建模型
    model = GPT2Model(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_len=max_len,
        dropout=0.1
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    # 注意: 权重共享使得 lm_head 不额外计数
    unique_params = sum(
        p.numel() for p in set(model.parameters())
    )
    print(f"\n总参数量 (含共享): {total_params:,}")
    print(f"唯一参数量: {unique_params:,}")
    
    # 2. 创建输入数据
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # 3. 前向传播 (训练模式)
    model.train()
    outputs = model(token_ids=token_ids, targets=token_ids)
    
    print(f"\nLogits 形状: {outputs['logits'].shape}")
    print(f"隐藏状态形状: {outputs['hidden_states'].shape}")
    print(f"LM Loss: {outputs['loss'].item():.4f}")
    print(f"困惑度: {torch.exp(outputs['loss']).item():.2f}")
    
    # 4. 验证因果性
    attn_weights = outputs["attention_weights"][0]  # 第 1 层
    print(f"\n注意力权重形状 (第 1 层): {attn_weights.shape}")
    
    # 位置 0 对位置 1 的注意力应为 0 (因果掩码)
    attn_0_to_1 = attn_weights[0, 0, 0, 1].item()
    # 位置 2 对位置 0 的注意力应 > 0
    attn_2_to_0 = attn_weights[0, 0, 2, 0].item()
    print(f"位置 0→1 注意力: {attn_0_to_1:.6f} (应为 0，因果掩码)")
    print(f"位置 2→0 注意力: {attn_2_to_0:.6f} (应 > 0)")
    
    # 5. 验证权重共享
    emb_weight = model.embedding.token_embedding.weight
    head_weight = model.lm_head.weight
    print(f"\n权重共享验证: {torch.equal(emb_weight, head_weight)}")
    
    # 6. 训练一步测试
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-4)
    
    loss = outputs["loss"]
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()
    
    # 再次前向验证损失下降
    with torch.no_grad():
        outputs_after = model(token_ids=token_ids, targets=token_ids)
    print(f"\n训练前 Loss: {outputs['loss'].item():.4f}")
    print(f"训练后 Loss: {outputs_after['loss'].item():.4f}")
    
    # 7. 生成测试
    model.eval()
    prefix = torch.randint(0, vocab_size, (1, 5))
    generated = model.generate(
        input_ids=prefix,
        max_new_tokens=20,
        temperature=0.8,
        top_k=50,
    )
    print(f"\n生成测试:")
    print(f"  前缀长度: {prefix.size(1)}")
    print(f"  生成后长度: {generated.size(1)}")
    print(f"  生成的 token IDs: {generated[0].tolist()}")
    
    print("\n✅ GPT-2 模型测试通过！")
```

---

## 7. 文本生成策略：采样方法的数学原理

文本生成是 GPT-2 最核心的应用场景。不同的采样策略会产生截然不同的文本质量和多样性。

### 7.1 贪心搜索与 Beam Search

**贪心搜索 (Greedy Search)**：

每一步选择概率最高的 token：

$$
x_t = \arg\max_{w \in V} P(w \mid x_{<t})
$$

**优点**：简单、确定性  
**缺点**：容易陷入重复循环，生成单调的文本

**Beam Search**：

维护 $B$ 个候选序列（beam），每步扩展所有候选：

$$
\text{score}(y_1, \ldots, y_t) = \sum_{i=1}^{t} \log P(y_i \mid y_{<i})
$$

选择总分最高的 $B$ 个序列保留。

**长度归一化**：

$$
\text{score}_{\text{norm}}(y) = \frac{1}{|y|^\alpha} \sum_{i=1}^{|y|} \log P(y_i \mid y_{<i})
$$

其中 $\alpha \in [0, 1]$ 是长度惩罚参数。$\alpha = 0$ 时无惩罚，$\alpha = 1$ 时完全归一化。

> **注意**：Beam Search 在机器翻译中效果很好，但在开放式文本生成中往往产生**退化文本**（重复、通用、无聊）。GPT-2 论文推荐使用随机采样方法。

### 7.2 温度采样 (Temperature Sampling)

温度参数 $T$ 控制概率分布的"锐度"：

$$
\boxed{P_T(x_t = w) = \frac{\exp(z_w / T)}{\sum_{w'=1}^{|V|} \exp(z_{w'} / T)}}
$$

其中 $z_w$ 是原始 logit 值。

**温度的效果**：

| $T$ 值 | 效果 | 分布特性 |
|--------|------|---------|
| $T \to 0^+$ | 接近贪心 | 退化为 argmax（确定性） |
| $T = 1$ | 原始分布 | 模型学到的分布 |
| $T > 1$ | 更平坦 | 增加多样性和随机性 |
| $T \to \infty$ | 均匀分布 | 完全随机 |

**数学分析**：

当 $T \to 0^+$ 时：

$$
\lim_{T \to 0^+} P_T(w) = \begin{cases}
1 & \text{if } w = \arg\max_i z_i \\
0 & \text{otherwise}
\end{cases}
$$

当 $T \to \infty$ 时：

$$
\lim_{T \to \infty} P_T(w) = \frac{1}{|V|} \quad \forall w
$$

**直觉**：温度控制了模型的"自信程度"。低温度使模型更自信（倾向于选择概率最高的 token），高温度使模型更"开放"（更可能选择低概率的 token）。

### 7.3 Top-k 采样

Top-k 采样（Fan et al., 2018）只从概率最高的 $k$ 个 token 中采样：

$$
\boxed{P_{\text{top-k}}(w) = \begin{cases}
\frac{P(w)}{\sum_{w' \in V_k} P(w')} & \text{if } w \in V_k \\
0 & \text{otherwise}
\end{cases}}
$$

其中 $V_k$ 是概率最高的 $k$ 个 token 的集合。

**算法步骤**：

1. 计算所有 token 的 logits
2. 找到最大的 $k$ 个 logits
3. 将其余 logits 设为 $-\infty$
4. 对保留的 $k$ 个 logits 进行 softmax
5. 从归一化后的分布中采样

**$k$ 值的影响**：

| $k$ | 效果 |
|-----|------|
| $k = 1$ | 贪心搜索 |
| $k = 10$ | 较集中，质量高但多样性低 |
| $k = 50$ | 平衡质量和多样性 |
| $k = |V|$ | 无过滤（原始分布） |

> **Top-k 的问题**：固定的 $k$ 无法适应不同的概率分布。当分布非常集中时（如只有 2 个合理选择），$k=50$ 会引入大量噪声；当分布很平坦时（如有 100 个合理选择），$k=50$ 会截断太多好的选择。

### 7.4 Top-p (Nucleus) 采样

Top-p 采样（Holtzman et al., 2020）动态地选择"累积概率达到 $p$ 的最小 token 集合"：

$$
\boxed{V_p = \min\left\{S \subseteq V : \sum_{w \in S} P(w) \geq p\right\}}
$$

即选择**最小**的 token 子集，使得它们的概率之和 $\geq p$。

**算法步骤**：

1. 计算所有 token 的概率 $P(w)$
2. 按概率降序排列
3. 计算累积概率
4. 找到累积概率首次 $\geq p$ 的位置
5. 保留该位置及之前的所有 token
6. 重新归一化并采样

**数学表述**：

设 $\sigma$ 为按概率降序排列的排列函数，$\sigma(1)$ 是概率最高的 token：

$$
P_{\text{top-p}}(w) = \begin{cases}
\frac{P(w)}{\sum_{w' \in V_p} P(w')} & \text{if } w \in V_p \\
0 & \text{otherwise}
\end{cases}
$$

其中：
$$
V_p = \left\{ \sigma(1), \sigma(2), \ldots, \sigma(k^*) \right\}, \quad k^* = \min\left\{k : \sum_{i=1}^{k} P(\sigma(i)) \geq p\right\}
$$

**Top-p vs Top-k 的优势**：

Top-p **自适应地**调整候选集大小：

```
场景 1：分布集中
P = [0.9, 0.05, 0.03, 0.01, ...]
Top-k (k=10): 保留 10 个（含大量低概率噪声）
Top-p (p=0.9): 只保留 1 个 ✓

场景 2：分布平坦
P = [0.1, 0.09, 0.08, 0.08, 0.07, ...]
Top-k (k=3): 只保留 3 个（截断太多好选择）
Top-p (p=0.9): 保留约 12 个 ✓
```

### 7.5 采样策略可视化与对比

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_sampling_strategies():
    """
    可视化不同采样策略对概率分布的影响
    """
    # 模拟 logits (10 个 token)
    logits = np.array([3.0, 2.5, 2.0, 1.0, 0.5, 0.0, -0.5, -1.0, -2.0, -3.0])
    tokens = [f"tok_{i}" for i in range(len(logits))]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. 原始分布
    probs = softmax(logits)
    axes[0, 0].bar(tokens, probs, color='steelblue')
    axes[0, 0].set_title("原始分布 (T=1.0)")
    axes[0, 0].set_ylabel("概率")
    
    # 2. 低温度 (T=0.3)
    probs_low = softmax(logits / 0.3)
    axes[0, 1].bar(tokens, probs_low, color='darkred')
    axes[0, 1].set_title("低温度 (T=0.3)")
    
    # 3. 高温度 (T=2.0)
    probs_high = softmax(logits / 2.0)
    axes[0, 2].bar(tokens, probs_high, color='forestgreen')
    axes[0, 2].set_title("高温度 (T=2.0)")
    
    # 4. Top-k (k=3)
    top_k_probs = probs.copy()
    top_k_indices = np.argsort(probs)[-3:]
    mask_k = np.zeros_like(probs)
    mask_k[top_k_indices] = probs[top_k_indices]
    mask_k = mask_k / mask_k.sum()
    axes[1, 0].bar(tokens, mask_k, color='darkorange')
    axes[1, 0].set_title("Top-k (k=3)")
    axes[1, 0].set_ylabel("概率")
    
    # 5. Top-p (p=0.9)
    sorted_idx = np.argsort(probs)[::-1]
    cumsum = np.cumsum(probs[sorted_idx])
    cutoff = np.searchsorted(cumsum, 0.9) + 1
    top_p_set = set(sorted_idx[:cutoff])
    mask_p = np.array([probs[i] if i in top_p_set else 0 for i in range(len(probs))])
    mask_p = mask_p / mask_p.sum()
    axes[1, 1].bar(tokens, mask_p, color='purple')
    axes[1, 1].set_title("Top-p (p=0.9)")
    
    # 6. Top-k + Top-p + Temperature
    combined_logits = logits / 0.7
    combined_probs = softmax(combined_logits)
    sorted_idx_c = np.argsort(combined_probs)[::-1]
    cumsum_c = np.cumsum(combined_probs[sorted_idx_c])
    cutoff_c = np.searchsorted(cumsum_c, 0.9) + 1
    top_p_set_c = set(sorted_idx_c[:cutoff_c])
    top_k_set_c = set(np.argsort(combined_probs)[-5:])
    final_set = top_p_set_c & top_k_set_c
    mask_c = np.array([combined_probs[i] if i in final_set else 0 for i in range(len(probs))])
    if mask_c.sum() > 0:
        mask_c = mask_c / mask_c.sum()
    axes[1, 2].bar(tokens, mask_c, color='teal')
    axes[1, 2].set_title("Top-k=5 + Top-p=0.9 + T=0.7")
    
    for ax in axes.flat:
        ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig("sampling_strategies.png", dpi=150)
    plt.show()
```

**各策略的推荐使用场景**：

| 策略 | 推荐参数 | 适用场景 |
|------|---------|---------|
| 贪心搜索 | — | 确定性输出（代码补全） |
| Beam Search | $B=5$ | 翻译、摘要 |
| 温度采样 | $T=0.7\text{-}1.0$ | 创意写作 |
| Top-k | $k=40\text{-}100$ | 通用生成 |
| Top-p | $p=0.9\text{-}0.95$ | **推荐默认** |
| Top-k + Top-p + T | $k=50, p=0.95, T=0.8$ | 高质量多样生成 |

---

## 8. 与其他模型的关系

### 8.1 GPT 系列演进：GPT → GPT-2 → GPT-3

```
GPT (2018, 117M)
  │  创新: 生成式预训练 + 判别式微调
  │  局限: 仍需为每个任务微调
  │
GPT-2 (2019, 1.5B)
  │  创新: Zero-shot 多任务学习
  │  关键: 更大模型 + 更好数据 → 涌现能力
  │  架构改进: Pre-Norm, 更长上下文
  │
GPT-3 (2020, 175B)
  │  创新: Few-shot / In-context Learning
  │  关键: 100x 参数 → 无需梯度更新即可学习新任务
  │
GPT-4 (2023, ?B)
     创新: 多模态, 更强推理
     关键: RLHF 对齐, 安全性
```

**关键指标对比**：

| 模型 | 参数量 | 数据量 | 上下文长度 | 使用方式 |
|------|--------|--------|-----------|---------|
| GPT | 117M | ~5GB | 512 | 微调 |
| GPT-2 | 1.5B | ~40GB | 1024 | Zero-shot |
| GPT-3 | 175B | ~570GB | 2048 | Few-shot |

**Scaling Law 的启示**：

GPT-2 的实验表明，语言模型的性能随以下因素**对数线性**增长：

$$
\mathcal{L}(N, D) \approx \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{D_c}{D}\right)^{\alpha_D}
$$

其中 $N$ 是参数量，$D$ 是数据量，$\alpha_N \approx 0.076$，$\alpha_D \approx 0.095$。

### 8.2 GPT-2 vs BERT：单向生成 vs 双向理解

**架构对比**：

| 组件 | BERT | GPT-2 |
|------|------|-------|
| 基础块 | Transformer Encoder | Transformer Decoder |
| Norm 位置 | Post-Norm | **Pre-Norm** |
| 注意力掩码 | 全连接 | **因果掩码** |
| 位置嵌入 | 可学习 | 可学习 |
| 段落嵌入 | ✅ 有 | ❌ 无 |
| 激活函数 | GELU | GELU |
| 权重共享 | MLM head ↔ Emb | **LM head ↔ Emb** |
| 最终 LayerNorm | ❌ 无 | ✅ 有 |

**训练目标对比**：

$$
\mathcal{L}_{\text{BERT}} = \underbrace{-\frac{1}{|\mathcal{M}|} \sum_{t \in \mathcal{M}} \log P(x_t \mid \tilde{x})}_{\text{MLM: 填空}} + \underbrace{(-\log P(y \mid A, B))}_{\text{NSP: 句对判断}}
$$

$$
\mathcal{L}_{\text{GPT-2}} = \underbrace{-\frac{1}{n-1} \sum_{t=1}^{n-1} \log P(x_{t+1} \mid x_{\leq t})}_{\text{自回归 LM: 续写}}
$$

**信号密度对比**：

- BERT：每个序列只有 ~15% 的位置贡献损失
- GPT-2：每个位置都贡献损失（100%）
- GPT-2 的训练效率在每个 token 上更高

### 8.3 预训练范式的转变

GPT-2 标志着从"预训练+微调"向"预训练+提示"的范式转变：

$$
\boxed{
\underbrace{\text{BERT 范式}}_{\text{预训练} \to \text{微调}} \quad \longrightarrow \quad \underbrace{\text{GPT-2 范式}}_{\text{预训练} \to \text{Zero-shot 提示}}
}
$$

**范式演进时间线**：

| 时间 | 范式 | 代表模型 | 下游使用方式 |
|------|------|---------|-------------|
| 2013 | 特征提取 | Word2Vec | 固定词向量 + 分类器 |
| 2018 | 预训练 + 微调 | BERT, GPT | 标注数据微调所有参数 |
| 2019 | **Zero-shot** | **GPT-2** | **自然语言提示** |
| 2020 | Few-shot | GPT-3 | 上下文学习（几个示例） |
| 2022 | 指令微调 + RLHF | InstructGPT | 人类反馈对齐 |

GPT-2 的核心贡献不在于架构创新（变化很小），而在于**证明了规模带来的质变**：当模型足够大、数据足够多样时，语言模型会展现出令人惊讶的**涌现能力（emergent abilities）**。

---

## 扩展阅读与实现

### 问题 1：为什么 GPT-2 使用 Pre-Norm 而不是 Post-Norm？

**解答**：

Pre-Norm 和 Post-Norm 的数学区别：

**Post-Norm**（BERT/GPT-1）：
$$
x^{(l)} = \text{LayerNorm}(x^{(l-1)} + f(x^{(l-1)}))
$$

**Pre-Norm**（GPT-2）：
$$
x^{(l)} = x^{(l-1)} + f(\text{LayerNorm}(x^{(l-1)}))
$$

**梯度传播分析**：

对于 Post-Norm，第 $L$ 层到第 $l$ 层的梯度需要通过 $L - l$ 个 LayerNorm 层，每个 LayerNorm 都会重新缩放梯度，可能导致梯度不稳定。

对于 Pre-Norm，残差连接提供了一条"短路"梯度通路：

$$
\frac{\partial x^{(L)}}{\partial x^{(l)}} = I + \text{其他项}
$$

恒等矩阵 $I$ 保证了梯度至少有一条不衰减的通路，这对 GPT-2 的 48 层网络至关重要。

**实验证据**（Xiong et al., 2020）表明：
- Post-Norm 需要精细的学习率预热，否则训练不稳定
- Pre-Norm 可以在没有预热的情况下稳定训练
- 在深层网络（>12 层）中，Pre-Norm 的优势更加明显

### 问题 2：BPE 分词的数学原理是什么？

**解答**：

BPE（Byte Pair Encoding）是一种自底向上的子词分词算法。

**算法**：

1. 初始化词表为所有单个字节（256 个）
2. 统计所有相邻 token 对的频率
3. 合并最频繁的 token 对为新 token
4. 重复步骤 2-3 直到词表达到目标大小

**示例**：

```
初始词表: {'l', 'o', 'w', 'e', 'r', 'n', 's', 't'}
语料频率: "low" × 5, "lowest" × 2, "newer" × 6, "lower" × 3

第 1 轮: 最频繁对 ('e', 'r') → 合并为 'er'
第 2 轮: 最频繁对 ('er', '</w>') → 合并为 'er</w>'  
第 3 轮: 最频繁对 ('l', 'o') → 合并为 'lo'
...
```

**GPT-2 的字节级 BPE**：

GPT-2 使用字节（而非字符）作为基本单位，这意味着：
- 初始词表 = 256 个字节值
- 可以编码任意 UTF-8 文本
- 不需要 `[UNK]` token
- 最终词表大小 = 50,257（256 字节 + 50,000 合并 + 1 个结束标记）

### 问题 3：权重共享为什么有效？

**解答**：

权重共享（weight tying）让输入嵌入矩阵 $W_{\text{emb}} \in \mathbb{R}^{|V| \times d}$ 同时作为输出层的投影矩阵。

**直觉**：输入嵌入将 token 映射到语义空间，输出层将隐藏状态映射回 token 空间。如果这两个映射共享同一个矩阵，就意味着**输入和输出在同一个语义空间中**。

**数学分析**：

输出 logit 为：
$$
z_w = W_{\text{emb}}[w]^\top \cdot h_t = \langle e_w, h_t \rangle
$$

这是嵌入向量 $e_w$ 和隐藏状态 $h_t$ 的内积，即**余弦相似度**（如果归一化的话）。

模型学到的是：让 $h_t$ 在嵌入空间中接近正确的下一个 token 的嵌入向量。

**参数节省**：

$$
\Delta P = |V| \times d = 50{,}257 \times 1{,}600 \approx 80\text{M}
$$

对于 GPT-2 XL（1,558M 参数），这约占 5% 的参数。

### 问题 4：GPT-2 的 Zero-shot 能力从何而来？

**解答**：

GPT-2 的 Zero-shot 能力来自三个因素的结合：

**1. 数据多样性**：WebText 包含各种格式的文本

```
翻译示例: "In French, 'cat' is 'chat'"
摘要示例: 新闻文章后面跟着 "TL;DR: ..."
QA 示例: 论坛帖子 "Q: ... A: ..."
```

**2. 模型容量**：1.5B 参数足以"记住"这些模式

**3. 自回归目标**：预测下一个 token 的任务迫使模型理解文本的结构和含义

从信息论的角度：

$$
H(X_{t+1} \mid X_{\leq t}) \leq H(X_{t+1} \mid X_{\leq t}, \text{Task})
$$

一个好的语言模型必须隐式地理解"任务"才能降低困惑度。

### 问题 5：GPT-2 与 GPT-1 的架构差异细节

**解答**：

| 组件 | GPT-1 | GPT-2 |
|------|-------|-------|
| LayerNorm 位置 | Post-Norm | **Pre-Norm** |
| 最终 LayerNorm | 无 | **有** |
| 残差初始化缩放 | 标准 $\mathcal{N}(0, 0.02)$ | **$\mathcal{N}(0, 0.02/\sqrt{2L})$** |
| 上下文长度 | 512 | **1024** |
| 词表 | BPE (40,000) | **字节级 BPE (50,257)** |
| 分词 | 字符级 BPE | **字节级 BPE** |

这些改进看似微小，但对于训练一个 48 层、1.5B 参数的模型至关重要。特别是 Pre-Norm + 残差缩放初始化的组合，解决了深层网络的训练稳定性问题。

---

## 参考资源

### 经典论文

1. Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf). OpenAI Technical Report.
   - **贡献**：提出 GPT-2，证明大规模语言模型具有 Zero-shot 多任务学习能力

2. Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. (2018). [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf). OpenAI Technical Report.
   - **贡献**：GPT-1，首次提出"生成式预训练 + 判别式微调"范式

3. Holtzman, A., Buys, J., Du, L., Forbes, M., & Choi, Y. (2020). [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751). ICLR 2020.
   - **贡献**：提出 Top-p (Nucleus) 采样，分析了文本生成退化问题

4. Brown, T., et al. (2020). [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165). NeurIPS 2020.
   - **贡献**：GPT-3，将 GPT-2 的理念扩展到 175B 参数，展示 Few-shot 能力

5. Xiong, R., et al. (2020). [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745). ICML 2020.
   - **贡献**：理论分析 Pre-Norm vs Post-Norm 的训练动态差异

### 教材与书籍

6. Jurafsky, D., & Martin, J. H. [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/). 3rd ed. (Draft).
   - **章节**：第 10 章详细讲解语言模型和文本生成

### 在线资源与教程

7. Alammar, J. [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/).
   - **内容**：GPT-2 架构和生成过程的直观图解

8. Hugging Face. [GPT-2 Documentation](https://huggingface.co/docs/transformers/model_doc/gpt2).
   - **内容**：GPT-2 的工业级实现和使用指南

9. OpenAI. [GPT-2 GitHub Repository](https://github.com/openai/gpt-2).
   - **内容**：GPT-2 原始 TensorFlow 实现和预训练模型

---

## 附录：符号表

| 符号 | 含义 | 维度/类型 |
|------|------|----------|
| $n$ | 输入序列长度 | 标量 |
| $d$ | 隐藏维度（$d_{\text{model}}$） | 标量，GPT-2 XL: 1600 |
| $d_k$ | 每个注意力头的维度 | 标量，64 |
| $d_{ff}$ | FFN 隐藏层维度 | 标量，GPT-2 XL: 6400 |
| $L$ | Transformer 层数 | 标量，GPT-2 XL: 48 |
| $A$ | 注意力头数 | 标量，GPT-2 XL: 25 |
| $\|V\|$ | BPE 词表大小 | 标量，50,257 |
| $L_{\max}$ | 最大上下文长度 | 标量，1024 |
| $T$ | 温度参数 | 标量，$T > 0$ |
| $k$ | Top-k 采样参数 | 标量，正整数 |
| $p$ | Top-p 采样参数 | 标量，$p \in (0, 1]$ |
| $x_t$ | 位置 $t$ 的 token | 整数索引 |
| $x_{<t}$ | 位置 $t$ 之前的所有 token | $(t-1,)$ |
| $h_t$ | 位置 $t$ 的隐藏状态 | $(d,)$ |
| $z_w$ | token $w$ 的 logit 值 | 标量 |
| $E_{\text{token}}$ | Token 嵌入矩阵 | $(\|V\|, d)$ |
| $E_{\text{position}}$ | 位置嵌入矩阵 | $(L_{\max}, d)$ |
| $W_{\text{emb}}$ | 嵌入矩阵（输入+输出共享） | $(\|V\|, d)$ |
| $M^{\text{causal}}$ | 因果注意力掩码 | $(n, n)$，下三角矩阵 |
| $Q, K, V$ | 查询、键、值矩阵 | $(n, d_k)$ |
| $\alpha_{ij}$ | 注意力权重（位置 $i$ 对 $j$） | 标量，$\alpha_{ij} = 0$ 当 $j > i$ |
| $\mathcal{L}_{\text{LM}}$ | 语言模型损失值 | 标量 |
| $\text{PPL}$ | 困惑度 | 标量，$\text{PPL} = \exp(\mathcal{L}_{\text{LM}})$ |
| $\ell(\cdot, \cdot)$ | 交叉熵损失函数 | 函数 |
| $V_k$ | Top-k 候选 token 集合 | 集合，$|V_k| = k$ |
| $V_p$ | Top-p 候选 token 集合 | 集合，大小自适应 |

**典型维度示例（GPT-2 XL）：**
- $d = 1600$（隐藏维度）
- $d_k = 64$（每头维度）
- $d_{ff} = 6400$（FFN 维度）
- $|V| = 50{,}257$（词表大小）
- $L_{\max} = 1024$（最大上下文长度）
- $L = 48$（Transformer 层数）

---

最后更新：2026-03-19
