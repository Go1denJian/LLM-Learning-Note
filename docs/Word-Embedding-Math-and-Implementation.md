# Word Embedding 数学原理与实现 —— 从共现矩阵到词向量

> **前置知识**：矩阵分解、条件概率、梯度下降、Python 基础  
> **与 Transformer 的联系**：Word Embedding 是 Transformer 输入层的基础

---

## 目录

1. [引言：为什么需要词向量？](#1-引言为什么需要词向量)
2. [从 One-Hot 到 Dense Embedding](#2-从 one-hot 到 dense-embedding)
3. [共现矩阵与 PMI](#3-共现矩阵与-pmi)
4. [Word2Vec 的两种架构](#4-word2vec-的两种架构)
5. [负采样的数学本质](#5-负采样的数学本质)
6. [梯度推导与参数更新](#6-梯度推导与参数更新)
7. [从数学到代码：完整实现](#7-从数学到代码完整实现)
8. [实践技巧与可视化](#8-实践技巧与可视化)
9. [练习与思考题](#9-练习与思考题)

---

## 1. 引言：为什么需要词向量？

### 1.1 NLP 的核心问题

**问题**：如何让计算机理解自然语言中的词语？

**朴素方案：One-Hot 编码**

给定词表 $V = \{w_1, w_2, \ldots, w_{|V|}\}$，每个词表示为 $|V|$ 维向量：

$$
\mathbf{v}_{w_i} = [0, \ldots, 1, \ldots, 0]^\top \quad (\text{第 } i \text{ 位为 1})
$$

**问题**：
1. **维度灾难**：$|V|$ 可达百万级，向量稀疏
2. **语义鸿沟**：任意两个词正交（内积为 0），无法表达语义距离
   $$
   \mathbf{v}_{\text{king}}^\top \mathbf{v}_{\text{queen}} = 0, \quad \mathbf{v}_{\text{king}}^\top \mathbf{v}_{\text{apple}} = 0
   $$
   即：$\text{dist}(\mathbf{v}_{\text{king}}, \mathbf{v}_{\text{queen}}) = \text{dist}(\mathbf{v}_{\text{king}}, \mathbf{v}_{\text{apple}})$，无法区分语义相似度
3. **计算低效**：矩阵规模 $O(|V|^2)$

### 1.2 Word Embedding 的思想

**核心思想**：将词映射到低维、稠密的向量空间

$$
f: w \in V \mapsto \mathbf{v}_w \in \mathbb{R}^d \quad (d \ll |V|)
$$

**典型配置**：
- $|V| = 100,000$（词表大小）
- $d = 300$（嵌入维度）
- 压缩比：$300 / 100,000 = 0.3\%$

**优势**：
1. **语义保持**：相似词的向量距离近
   $$
   \text{dist}(\mathbf{v}_{\text{king}}, \mathbf{v}_{\text{queen}}) \ll \text{dist}(\mathbf{v}_{\text{king}}, \mathbf{v}_{\text{apple}})
   $$
   其中 $\text{dist}(\cdot, \cdot)$ 表示向量距离（如欧氏距离或余弦距离）
2. **计算高效**：矩阵规模 $O(|V| \cdot d)$
3. **可迁移性**：预训练词向量可用于多种下游任务

---

## 2. 从 One-Hot 到 Dense Embedding

### 2.1 计算复杂度分析

**One-Hot + Softmax 的计算成本**：

假设词表 $|V| = 100,000$，嵌入维度 $d = 300$。

**符号定义**：
- $W_{in} \in \mathbb{R}^{d \times |V|}$：输入嵌入矩阵（词表→隐藏层）
- $W_{out} \in \mathbb{R}^{d \times |V|}$：输出嵌入矩阵（隐藏层→词表）
- $\mathbf{x} \in \mathbb{R}^{|V|}$：输入 one-hot 向量

**前向传播**：
1. 输入层 → 隐藏层：$\mathbf{h} = W_{in}^\top \mathbf{x}$，其中 $\mathbf{x}$ 是 one-hot 向量
   $$
   O(d) \quad (\text{因为 } \mathbf{x} \text{ 只有一个 1})
   $$
2. 隐藏层 → 输出层：$\mathbf{z} = W_{out}^\top \mathbf{h}$
   $$
   O(|V| \cdot d) = 100,000 \times 300 = 3 \times 10^7
   $$
3. Softmax 归一化：
   $$
   O(|V|) = 100,000
   $$

**总复杂度**：$O(|V| \cdot d)$

**对比**：
- $d \times d = 300 \times 300 = 90,000$
- $d \times |V| = 300 \times 100,000 = 30,000,000$

**结论**：输出层 softmax 是主要瓶颈（占比 > 99%）。

> **注**：上述复杂度分析为后续 Word2Vec 训练奠定基础。由于 softmax 需要遍历整个词表进行归一化，当词表很大时（如 10 万词），每次参数更新都需要计算 3000 万次乘法，这成为训练效率的主要瓶颈。后续 Word2Vec 提出的优化方法（Hierarchical Softmax 和 Negative Sampling）正是为了解决这一问题，这些方法将在第 5 章详细介绍。

---

## 3. 共现矩阵与 PMI

### 3.1 共现矩阵的定义

**定义**：统计词 $i$ 和词 $j$ 在窗口大小 $m$ 内共同出现的次数。

$$
X_{ij} = \text{count}(w_i \text{ 与 } w_j \text{ 在窗口 } m \text{ 内共现})
$$

**示例**（窗口 $m=2$）：

```
句子："the cat sat on the mat"

共现对：
(the, cat), (the, sat), (cat, sat), (cat, on), (sat, on), (sat, the), ...
```

**矩阵规模**：$|V| \times |V|$（通常稀疏）

### 3.2 PMI（Pointwise Mutual Information）

**定义**：衡量两个词共现的关联程度。

$$
\text{PMI}(i, j) = \log \frac{P(i, j)}{P(i) P(j)}
$$

**概率定义详解**：

| 符号 | 中文含义 | 计算公式 | 直观理解 |
|-----|---------|---------|---------|
| $P(i, j)$ | **联合概率** | $P(i, j) = \frac{X_{ij}}{\sum_{i',j'} X_{i'j'}}$ | 词 $i$ 和词 $j$ 共同出现的概率（占所有共现对的比例） |
| $P(i)$ | **边缘概率** | $P(i) = \frac{\sum_j X_{ij}}{\sum_{i',j'} X_{i'j'}} = \sum_j P(i, j)$ | 词 $i$ 出现的概率（对 $j$ 求和，即不管 $j$ 是什么，$i$ 出现的总次数） |
| $P(j)$ | **边缘概率** | $P(j) = \frac{\sum_i X_{ij}}{\sum_{i',j'} X_{i'j'}} = \sum_i P(i, j)$ | 词 $j$ 出现的概率（对 $i$ 求和） |

**关键关系**：
$$
P(i) = \sum_j P(i, j) \quad \text{（边缘概率 = 联合概率对另一变量求和）}
$$

**示例说明**：
假设语料库中共有 1000 个共现对：
- "国王"和"女王"共现 10 次 → $X_{ij} = 10$
- "国王"和所有词共现 100 次 → $\sum_j X_{ij} = 100$
- 则 $P(i, j) = 10/1000 = 0.01$，$P(i) = 100/1000 = 0.1$

**PMI 的直观解释**：

$$
\text{PMI}(i, j) = \log \frac{P(i, j)}{P(i) P(j)} = \log \frac{\text{实际共现概率}}{\text{随机期望概率}}
$$

| PMI 值 | 含义 | 说明 |
|--------|------|------|
| $\text{PMI} > 0$ | **正相关** | 实际共现频率 **高于** 随机期望（两词有关联） |
| $\text{PMI} = 0$ | **独立** | 实际共现频率 **等于** 随机期望（两词无关） |
| $\text{PMI} < 0$ | **负相关** | 实际共现频率 **低于** 随机期望（两词互斥） |

> **注释：与事件独立性的关系**
> 
> 在概率论中，两个事件 $A$ 和 $B$ 独立的定义是 $P(A, B) = P(A)P(B)$。
> 
> 对比 PMI 公式：
> - 当 $P(i, j) = P(i)P(j)$ 时，$\text{PMI}(i, j) = \log 1 = 0$
> - 这说明 **PMI = 0 等价于统计独立性**
> - PMI 衡量的是"偏离独立性的程度"：
>   - PMI > 0：正相关（比独立时更常共现）
>   - PMI < 0：负相关（比独立时更少共现）
> 
> 因此，PMI 可以看作是**基于样本的统计独立性的一种量化度量**（注意：这里是从观测数据出发的经验度量，区别于抽象概率空间中的理论独立性）。

**问题**：PMI 对低频词敏感，可能产生很大的负值（例如：两个罕见词偶尔共现一次，PMI 会计算出很大的正值；但更多时候罕见词不共现，PMI 为负值且绝对值很大）。

### 3.3 PPMI（Positive PMI）

**背景问题**：

PMI 虽然理论上很好，但在实际应用中存在两个问题：

1. **负值难以解释**：PMI < 0 表示两词"负相关"，但在自然语言中，大多数词对只是"无关"而非"互斥"，负值没有明确的语义意义。

2. **对低频词敏感**：
   - 罕见词很少共现 → PMI 经常为负
   - 负值的绝对值可能很大（因为 $\log$ 函数在接近 0 时趋向 $-\infty$）
   - 这些大负值主要是噪声，而非有意义的信号

**PPMI 的定义**：

将 PMI 的负值截断为 0，只保留正相关：

$$
\text{PPMI}(i, j) = \max(\text{PMI}(i, j), 0) = 
\begin{cases}
\text{PMI}(i, j), & \text{if } \text{PMI}(i, j) > 0 \\
0, & \text{otherwise}
\end{cases}
$$

**为什么截断为 0 有效**：

1. **语义合理性**：
   - PMI > 0：两词确实有关联（如 "国王"-"女王"）
   - PMI = 0：两词独立（如 "苹果"-"桌子"）
   - PMI < 0：通常只是噪声，而非真正的"互斥"

2. **矩阵稀疏性**：
   - 语料库中大部分词对是无关的（PMI ≤ 0）
   - 截断后变为 0，可以用稀疏矩阵存储
   - 大幅减少存储空间和计算量

3. **实验验证**：
   - 下游任务（如词相似度、类比推理）中，PPMI 通常优于原始 PMI
   - 负值对词向量的质量没有贡献，甚至有害

**示例对比**：

| 词对 | 共现次数 | PMI | PPMI | 说明 |
|-----|---------|-----|------|------|
| (国王, 女王) | 100 | 2.3 | 2.3 | 强正相关，保留 |
| (苹果, 香蕉) | 50 | 1.8 | 1.8 | 正相关，保留 |
| (国王, 苹果) | 2 | -0.5 | 0 | 弱负相关（噪声），截断 |
| (罕见词A, 罕见词B) | 1 | -3.2 | 0 | 强负相关（偶然共现），截断 |

**优势总结**：
- 减少噪声（消除低频词的偶然共现带来的大负值）
- 矩阵更稀疏（便于存储和计算）
- 下游任务效果更好（实验验证）
- 语义更清晰（只关注"有关联"的词对）

### 3.4 从 PPMI 到词向量

**问题引入**：

我们有了 PPMI 矩阵，它包含了词与词之间的语义关联信息。但 PPMI 矩阵是 $|V| \times |V|$ 的（词表大小平方），当词表为 10 万时，矩阵有 100 亿个元素，无法直接用于下游任务。

**核心问题**：如何将 PPMI 矩阵"压缩"成低维稠密的词向量？

**关键思想**：矩阵分解

如果我们能找到两个矩阵 $W_{in}$ 和 $W_{out}$，使得：

$$
\text{PPMI} \approx W_{in}^\top W_{out}
$$

那么：
- $W_{in} \in \mathbb{R}^{d \times |V|}$ 就是输入词向量（第 $i$ 列为词 $i$ 的向量）
- $W_{out} \in \mathbb{R}^{d \times |V|}$ 就是输出词向量
- $d \ll |V|$（如 $d=300$，而 $|V|=100,000$）

**为什么这个分解是合理的？**

1. **信息压缩角度**：
   - PPMI 矩阵虽然大，但有效信息集中在较低维度的子空间中
   - 大部分词对是无关的（PPMI = 0），只有少数词对有强关联
   - 低秩分解可以捕捉这些主要关联，丢弃噪声

2. **理论基础**（Levy & Goldberg, 2014）：

   **原文定理**（Levy & Goldberg, 2014, Theorem 1）：
   > "Skip-gram with Negative Sampling (SGNS) is implicitly factorizing a word-context matrix, whose cells are the pointwise mutual information (PMI) of the respective word and context pairs, shifted by a global constant."
   > 
   > （SGNS 隐式地分解一个词-上下文矩阵，其单元格是相应词和上下文对的点互信息（PMI），并平移了一个全局常数。）

   **具体结论**：
   - SGNS 的目标函数等价于对 **PMI 矩阵** 进行 **加权 SVD 分解**
   - 在最优解处，词向量的内积满足：
     $$
     \mathbf{v}_w^\top \mathbf{u}_c = \text{PMI}(w, c) - \log k
     $$
     其中 $k$ 是负采样数
   - 这意味着：**词向量的内积应该等于 PMI 值（减去一个常数）**
   - 因此，我们寻找满足 $W_{in}^\top W_{out} \approx \text{PMI}$ 的分解是合理的

3. **几何解释**：
   - 如果两个词在语义上相似，它们在 PPMI 矩阵中会有相似的行/列
   - 低秩分解将这些相似的行/列映射到相近的向量
   - 例如："国王"和"女王"在 PPMI 矩阵中都有高值对应 "皇冠"、"宫殿"等词
   - 分解后，它们的词向量 $\mathbf{v}_{\text{国王}}$ 和 $\mathbf{v}_{\text{女王}}$ 也会很接近

**为什么是近似（$\approx$）？**

- PPMI 矩阵的秩可能高达 $|V|$（满秩）
- 但 $W_{in}^\top W_{out}$ 的秩最多为 $d$（远小于 $|V|$）
- 我们只能保留最重要的 $d$ 个维度，丢弃次要信息
- 这种"有损压缩"在实践中效果很好

**方法 1：SVD 分解（显式方法）**

**SVD（奇异值分解）定义**：

对于矩阵 $A \in \mathbb{R}^{m \times n}$，SVD 分解为：

$$
A = U \Sigma V^\top
$$

其中：
- $U \in \mathbb{R}^{m \times m}$：左奇异向量矩阵，列向量为标准正交基
- $\Sigma \in \mathbb{R}^{m \times n}$：对角矩阵，对角线元素 $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$ 为奇异值
- $V \in \mathbb{R}^{n \times n}$：右奇异向量矩阵，列向量为标准正交基

**低秩近似**：

取前 $d$ 个最大的奇异值及其对应的奇异向量：

$$
A \approx U_d \Sigma_d V_d^\top
$$

其中 $U_d \in \mathbb{R}^{m \times d}$，$\Sigma_d \in \mathbb{R}^{d \times d}$，$V_d \in \mathbb{R}^{n \times d}$。

**应用到 PPMI**：

$$
\text{PPMI} \approx U_d \Sigma_d V_d^\top = (U_d \sqrt{\Sigma_d})(\sqrt{\Sigma_d} V_d^\top) = W_{in}^\top W_{out}
$$

因此可以令：
- $W_{in} = \sqrt{\Sigma_d} V_d^\top$（或 $U_d \sqrt{\Sigma_d}$）
- $W_{out} = \sqrt{\Sigma_d} U_d^\top$（或 $\sqrt{\Sigma_d} V_d^\top$）

**方法 2：Word2Vec（隐式方法）**

Word2Vec 不直接计算 PPMI 矩阵，而是通过优化 Skip-gram 或 CBOW 的目标函数，**隐式地**学习到低秩分解的结果。

**两种方法的对比**：

| 方法 | 优点 | 缺点 |
|------|------|------|
| SVD 分解 | 数学上精确，可解释性强 | 需要存储和分解巨大的 PPMI 矩阵 |
| Word2Vec | 不需要显式构造 PPMI 矩阵，可扩展性好 | 结果是隐式的，不如 SVD 直观 |

**实践建议**：
- 小规模数据：可以使用 SVD 分解 PPMI 矩阵（如 GloVe 的思想）
- 大规模数据：使用 Word2Vec 的隐式分解（更高效）

---

## 4. Word2Vec 的两种架构

### 4.1 CBOW（Continuous Bag of Words）

**思想**：用上下文预测中心词。

```
上下文：[the, cat, sat, on]  →  中心词：the
```

**数学表述**：

给定上下文词 $\{w_{t-m}, \ldots, w_{t-1}, w_{t+1}, \ldots, w_{t+m}\}$，预测中心词 $w_t$。

**前向传播**：

1. 上下文词向量平均：
   $$
   \mathbf{h} = \frac{1}{2m} \sum_{-m \leq j \leq m, j \neq 0} \mathbf{v}_{w_{t+j}}
   $$
   其中 $\mathbf{v}_{w} \in \mathbb{R}^d$ 是词 $w$ 的输入向量（$W_{in} \in \mathbb{R}^{d \times |V|}$ 的第 $w$ 列）。

2. 输出层 softmax：
   $$
   P(w_t \mid \text{context}) = \frac{\exp(\mathbf{u}_{w_t}^\top \mathbf{h})}{\sum_{w \in V} \exp(\mathbf{u}_w^\top \mathbf{h})}
   $$
   其中 $\mathbf{u}_w \in \mathbb{R}^d$ 是词 $w$ 的输出向量（$W_{out} \in \mathbb{R}^{d \times |V|}$ 的第 $w$ 列）。

**损失函数**（负对数似然）：

**为什么使用负对数似然？**

CBOW 的目标是最大化正确预测中心词的概率 $P(w_t \mid \text{context})$。在机器学习中，我们通常将最大化问题转化为最小化问题，即最小化负对数似然（Negative Log-Likelihood, NLL）：

$$
\mathcal{L}_{CBOW} = -\log P(w_t \mid \text{context})
$$

**推导过程**：

1. 根据 softmax 概率定义：
   $$
   P(w_t \mid \text{context}) = \frac{\exp(\mathbf{u}_{w_t}^\top \mathbf{h})}{\sum_{w \in V} \exp(\mathbf{u}_w^\top \mathbf{h})}
   $$

2. 取对数：
   $$
   \log P(w_t \mid \text{context}) = \mathbf{u}_{w_t}^\top \mathbf{h} - \log \sum_{w \in V} \exp(\mathbf{u}_w^\top \mathbf{h})
   $$

3. 取负号得到损失函数：
   $$
   \mathcal{L}_{CBOW} = -\log P(w_t \mid \text{context}) = -\mathbf{u}_{w_t}^\top \mathbf{h} + \log \sum_{w \in V} \exp(\mathbf{u}_w^\top \mathbf{h})
   $$

**直观理解**：
- $-\mathbf{u}_{w_t}^\top \mathbf{h}$：增大正确词 $w_t$ 的得分
- $\log \sum_{w \in V} \exp(\mathbf{u}_w^\top \mathbf{h})$：压低所有词的得分（尤其是得分高的词）
- 整体效果：正确词的得分相对其他词越来越高

**与交叉熵的关系**：

负对数似然等价于 one-hot 标签与预测分布的交叉熵：

$$
\mathcal{L}_{CBOW} = -\sum_{w \in V} y_w \log P(w \mid \text{context}) = -\log P(w_t \mid \text{context})
$$

其中 $y_w = 1$ 当且仅当 $w = w_t$，否则 $y_w = 0$。

### 4.2 Skip-gram

**思想**：用中心词预测上下文。

```
中心词：cat  →  上下文：[the, sat, on, the]
```

**数学表述**：

给定中心词 $w_t$，预测上下文词 $\{w_{t-m}, \ldots, w_{t-1}, w_{t+1}, \ldots, w_{t+m}\}$。

**关键假设**：上下文词在给定中心词条件下独立。

$$
P(w_{t-m}, \ldots, w_{t+m} \mid w_t) \approx \prod_{-m \leq j \leq m, j \neq 0} P(w_{t+j} \mid w_t)
$$

**预测概率**：
$$
P(w_{t+j} \mid w_t) = \frac{\exp(\mathbf{u}_{w_{t+j}}^\top \mathbf{v}_{w_t})}{\sum_{w \in V} \exp(\mathbf{u}_w^\top \mathbf{v}_{w_t})}
$$

**损失函数**：
$$
\begin{aligned}
\mathcal{L}_{SkipGram} &= -\sum_{-m \leq j \leq m, j \neq 0} \log P(w_{t+j} \mid w_t) \\
&= -\sum_{j} \left( \mathbf{u}_{w_{t+j}}^\top \mathbf{v}_{w_t} - \log \sum_{w \in V} \exp(\mathbf{u}_w^\top \mathbf{v}_{w_t}) \right)
\end{aligned}
$$

### 4.3 CBOW vs Skip-gram 对比

| 特性 | CBOW | Skip-gram |
|-----|------|-----------|
| 输入 | 上下文词 | 中心词 |
| 输出 | 中心词 | 上下文词 |
| 训练速度 | 快（平滑） | 慢（多个样本） |
| 罕见词效果 | 较差 | 较好 |
| 适用场景 | 大规模语料 | 小规模语料 |

---

## 5. 负采样的数学本质

### 5.1 从多分类到二分类

**问题**：Softmax 需要归一化整个词表 $O(|V|)$。

**负采样思想**：将问题转化为二分类。

- **正样本**：真实共现的词对 $(w_c, w_o)$，标签 $D=1$
- **负样本**：随机采样的词对 $(w_c, w_i)$，标签 $D=0$

**符号说明**：
- $\mathbf{v}_{w} \in \mathbb{R}^d$：词 $w$ 的输入向量（$W_{in}$ 的第 $w$ 列）
- $\mathbf{u}_{w} \in \mathbb{R}^d$：词 $w$ 的输出向量（$W_{out}$ 的第 $w$ 列）

**二分类概率**：
$$
P(D=1 \mid w_c, w_o) = \sigma(\mathbf{u}_{w_o}^\top \mathbf{v}_{w_c}) = \frac{1}{1 + \exp(-\mathbf{u}_{w_o}^\top \mathbf{v}_{w_c})}
$$

### 5.2 损失函数推导

**Skip-gram with Negative Sampling (SGNS)**：

对于正样本 $(w_c, w_o)$ 和 $k$ 个负样本 $\{w_1, \ldots, w_k\}$：

$$
\mathcal{L} = -\log \sigma(\mathbf{u}_{w_o}^\top \mathbf{v}_{w_c}) - \sum_{i=1}^k \log \sigma(-\mathbf{u}_{w_i}^\top \mathbf{v}_{w_c})
$$

**推导**：

1. 正样本项：最大化 $P(D=1)$
   $$
   -\log \sigma(\mathbf{u}_{w_o}^\top \mathbf{v}_{w_c})
   $$

2. 负样本项：最大化 $P(D=0)$
   $$
   -\log P(D=0) = -\log(1 - \sigma(\mathbf{u}_{w_i}^\top \mathbf{v}_{w_c})) = -\log \sigma(-\mathbf{u}_{w_i}^\top \mathbf{v}_{w_c})
   $$

**关键性质**：$\sigma(-x) = 1 - \sigma(x)$

### 5.3 负采样分布

**经验分布**（Mikolov et al., 2013）：

$$
P(w_i) = \frac{f(w_i)^{3/4}}{\sum_{w} f(w)^{3/4}}
$$

其中 $f(w)$ 是词 $w$ 的词频。

**为什么用 $3/4$ 次方**：
- 平衡高频词和低频词
- 避免常见词（如 "the", "a"）被过度采样

### 5.4 SGNS 与 PMI 的关系

**定理**（Levy & Goldberg, 2014）：SGNS 隐式分解 PMI 矩阵。

在最优点，有：
$$
\mathbf{v}_w^\top \mathbf{u}_c \approx \text{PMI}(w, c) - \log k
$$

**推导**：

1. Bayes 最优分类器：
   $$
   \sigma(\mathbf{v}_w^\top \mathbf{u}_c) = \frac{P_{data}(w, c)}{P_{data}(w, c) + k \cdot P_{noise}(w, c)}
   $$

2. 假设 $P_{noise}(w, c) = P(w)P(c)$（独立性）：
   $$
   \mathbf{v}_w^\top \mathbf{u}_c = \log \frac{P(w, c)}{P(w)P(c)} - \log k = \text{PMI}(w, c) - \log k
   $$

**结论**：SGNS 等价于对移位 PMI 矩阵的低秩分解。

---

## 6. Word2Vec 训练优化方法总结

在前面的章节中，我们学习了 Word2Vec 的两种架构（CBOW 和 Skip-gram）以及负采样的数学原理。现在，让我们回顾并总结 Word2Vec 训练中用于解决 softmax 计算瓶颈的优化方法。

### 6.1 为什么要优化？

**回顾第 2.1 节的分析**：
- 标准 Softmax 的复杂度为 $O(|V| \cdot d)$
- 当词表 $|V| = 100,000$，维度 $d = 300$ 时，每次更新需要 3000 万次乘法
- 这是训练效率的主要瓶颈

### 6.2 三种优化方法对比

| 方法 | 时间复杂度 | 核心思想 | 优点 | 缺点 | 适用场景 |
|-----|-----------|---------|------|------|---------|
| **Standard Softmax** | $O(\|V\| \cdot d)$ | 计算所有词的概率并归一化 | 概率精确 | 计算量巨大 | 词表较小（&lt;1万） |
| **Hierarchical Softmax** | $O(d \cdot \log \|V\|)$ | 用二叉树（Huffman树）将多分类转为 $\log \|V\|$ 个二分类 | 概率精确，复杂度降低 | 实现复杂，树结构需要预计算 | 词表较大，需要精确概率 |
| **Negative Sampling** | $O(k \cdot d)$ | 只采样 $k$ 个负例，将问题转为二分类 | 实现简单，速度极快，效果相当 | 概率近似，非精确 | 词表很大，追求训练速度（最常用） |

### 6.3 方法选择建议

**实际应用中的选择**：

1. **Negative Sampling（推荐）**：
   - 这是目前最主流的做法
   - $k = 5 \sim 20$（负样本数）远小于词表大小
   - 加速比：$|V|/k \approx 5,000 \sim 20,000$ 倍
   - 在大多数任务上与 full softmax 效果相当，甚至更好

2. **Hierarchical Softmax**：
   - 适用于需要精确概率的场景
   - 对罕见词的建模效果更好
   - 但实现复杂，现在使用较少

3. **Standard Softmax**：
   - 仅用于教学和理解原理
   - 实际训练中几乎不用

### 6.4 负采样的关键细节

**为什么 Negative Sampling 效果好？**

1. **计算效率**：只更新 $k+1$ 个词（1个正例 + $k$ 个负例），而非整个词表
2. **噪声对比估计**：通过区分正例和负例来学习，简化了问题
3. **高频词降权**：负采样分布 $P(w) \propto f(w)^{3/4}$ 降低了高频词的采样概率，减少了 "the", "a" 等常见词的干扰

**超参数选择**：
- $k = 5 \sim 20$：负样本数，通常 5-10 就足够
- $power = 0.75$：频率幂次，经验值

---

## 7. 梯度推导与参数更新

### 7.1 CBOW 梯度推导

**设定**：
- 上下文词：$\{w_1, \ldots, w_C\}$（$C = 2m$）
- 目标词：$w_o$
- 隐藏层：$\mathbf{h} = \frac{1}{C} \sum_{i=1}^C \mathbf{v}_{w_i}$

**损失函数**：
$$
\mathcal{L} = -\mathbf{u}_{w_o}^\top \mathbf{h} + \log \sum_{j=1}^{|V|} \exp(\mathbf{u}_j^\top \mathbf{h})
$$

**步骤 1：对输出向量 $\mathbf{u}_j$ 的梯度**

定义 softmax 输出：
$$
\hat{y}_j = \frac{\exp(\mathbf{u}_j^\top \mathbf{h})}{\sum_k \exp(\mathbf{u}_k^\top \mathbf{h})}
$$

目标 one-hot 向量：
$$
y_j = \begin{cases} 1, & j = w_o \\ 0, & \text{otherwise} \end{cases}
$$

梯度：
$$
\frac{\partial \mathcal{L}}{\partial \mathbf{u}_j} = (\hat{y}_j - y_j) \mathbf{h}
$$

**矩阵形式**：
$$
\frac{\partial \mathcal{L}}{\partial W_{out}} = \mathbf{h} (\hat{\mathbf{y}} - \mathbf{y})^\top
$$

**步骤 2：对隐藏层 $\mathbf{h}$ 的梯度**

$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial \mathbf{h}} &= -\mathbf{u}_{w_o} + \sum_j \hat{y}_j \mathbf{u}_j \\
&= \sum_j (\hat{y}_j - y_j) \mathbf{u}_j \\
&= W_{out} (\hat{\mathbf{y}} - \mathbf{y})
\end{aligned}
$$

**步骤 3：对输入向量 $\mathbf{v}_{w_i}$ 的梯度**

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{v}_{w_i}} = \frac{1}{C} \frac{\partial \mathcal{L}}{\partial \mathbf{h}} = \frac{1}{C} W_{out} (\hat{\mathbf{y}} - \mathbf{y})
$$

### 7.2 负采样梯度

**正样本梯度**：

$$
\mathcal{L}_{pos} = -\log \sigma(\mathbf{u}_{w_o}^\top \mathbf{v}_{w_c})
$$

令 $x = \mathbf{u}_{w_o}^\top \mathbf{v}_{w_c}$：

$$
\frac{\partial \mathcal{L}_{pos}}{\partial x} = \sigma(x) - 1
$$

**负样本梯度**：

$$
\mathcal{L}_{neg} = -\log \sigma(-\mathbf{u}_{w_i}^\top \mathbf{v}_{w_c})
$$

$$
\frac{\partial \mathcal{L}_{neg}}{\partial x} = \sigma(x)
$$

**参数更新**：

对于正样本：
$$
\mathbf{u}_{w_o} \leftarrow \mathbf{u}_{w_o} - \eta (\sigma(\mathbf{u}_{w_o}^\top \mathbf{v}_{w_c}) - 1) \mathbf{v}_{w_c}
$$

对于负样本：
$$
\mathbf{u}_{w_i} \leftarrow \mathbf{u}_{w_i} - \eta \sigma(\mathbf{u}_{w_i}^\top \mathbf{v}_{w_c}) \mathbf{v}_{w_c}
$$

### 7.3 参数更新总结表

| 参数 | 梯度 | 更新规则 |
|-----|------|---------|
| $\mathbf{u}_j$（输出） | $(\hat{y}_j - y_j)\mathbf{h}$ | $\mathbf{u}_j \leftarrow \mathbf{u}_j - \eta (\hat{y}_j - y_j)\mathbf{h}$ |
| $\mathbf{v}_{w_i}$（输入） | $\frac{1}{C} \sum_j (\hat{y}_j - y_j)\mathbf{u}_j$ | $\mathbf{v}_{w_i} \leftarrow \mathbf{v}_{w_i} - \frac{\eta}{C} \sum_j (\hat{y}_j - y_j)\mathbf{u}_j$ |
| 负采样 $\mathbf{u}_{w_o}$ | $(\sigma(x)-1)\mathbf{v}_{w_c}$ | $\mathbf{u}_{w_o} \leftarrow \mathbf{u}_{w_o} - \eta (\sigma(x)-1)\mathbf{v}_{w_c}$ |
| 负采样 $\mathbf{u}_{w_i}$ | $\sigma(x)\mathbf{v}_{w_c}$ | $\mathbf{u}_{w_i} \leftarrow \mathbf{u}_{w_i} - \eta \sigma(x)\mathbf{v}_{w_c}$ |

---

## 7. 从数学到代码：完整实现

### 7.1 数据预处理

```python
import numpy as np
from collections import Counter, defaultdict
from typing import List, Tuple, Dict

class Vocabulary:
    """词表管理"""
    def __init__(self, min_freq: int = 5):
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.word_freq: Counter = Counter()
        self.min_freq = min_freq
    
    def build(self, sentences: List[List[str]]):
        """构建词表"""
        # 统计词频
        for sentence in sentences:
            self.word_freq.update(sentence)
        
        # 过滤低频词
        words = [w for w, f in self.word_freq.items() if f >= self.min_freq]
        
        # 构建映射
        for idx, word in enumerate(words):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
    
    def __len__(self):
        return len(self.word2idx)


def generate_skipgram_pairs(
    sentences: List[List[str]],
    vocab: Vocabulary,
    window_size: int = 2
) -> List[Tuple[int, int]]:
    """
    生成 Skip-gram 训练样本
    
    参数:
        sentences: 分词后的句子列表
        vocab: 词表
        window_size: 上下文窗口大小
    
    返回:
        pairs: (center_word, context_word) 索引对
    """
    pairs = []
    
    for sentence in sentences:
        # 转换为索引
        indices = [vocab.word2idx[w] for w in sentence if w in vocab.word2idx]
        
        # 生成 (center, context) 对
        for i, center_idx in enumerate(indices):
            # 上下文范围
            start = max(0, i - window_size)
            end = min(len(indices), i + window_size + 1)
            
            for j in range(start, end):
                if i != j:
                    context_idx = indices[j]
                    pairs.append((center_idx, context_idx))
    
    return pairs
```

### 7.2 负采样

```python
class NegativeSampler:
    """负采样器"""
    def __init__(self, vocab: Vocabulary, power: float = 0.75):
        self.vocab = vocab
        self.power = power
        
        # 计算采样分布 P(w) = f(w)^power / sum(f(w)^power)
        freqs = np.array([vocab.word_freq[w] ** power for w in vocab.word2idx])
        self.probs = freqs / freqs.sum()
    
    def sample(self, num_samples: int, exclude: int = None) -> List[int]:
        """
        采样负样本
        
        参数:
            num_samples: 采样数量
            exclude: 排除的词索引（通常是正样本）
        
        返回:
            negative_indices: 负样本索引列表
        """
        samples = []
        while len(samples) < num_samples:
            idx = np.random.choice(len(self.vocab), p=self.probs)
            if idx != exclude:
                samples.append(idx)
        return samples
```

### 7.3 Word2Vec 模型

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class Word2VecSkipGram(nn.Module):
    """
    Skip-gram with Negative Sampling
    
    数学公式:
        L = -log σ(u_wo^T v_wc) - sum_{i=1}^k log σ(-u_wi^T v_wc)
    
    参数:
        vocab_size: 词表大小
        embedding_dim: 嵌入维度
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 300):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # 输入词向量 W_in
        self.W_in = nn.Embedding(vocab_size, embedding_dim)
        # 输出词向量 W_out
        self.W_out = nn.Embedding(vocab_size, embedding_dim)
        
        # 初始化（Xavier）
        nn.init.xavier_uniform_(self.W_in.weight)
        nn.init.xavier_uniform_(self.W_out.weight)
    
    def forward(
        self,
        center_words: torch.Tensor,  # (batch_size,)
        context_words: torch.Tensor,  # (batch_size,)
        negative_words: torch.Tensor  # (batch_size, num_negatives)
    ) -> torch.Tensor:
        """
        前向传播计算损失
        
        参数:
            center_words: 中心词索引
            context_words: 上下文词索引（正样本）
            negative_words: 负样本索引
        
        返回:
            loss: 标量损失
        """
        batch_size = center_words.size(0)
        num_negatives = negative_words.size(1)
        
        # 1. 获取词向量
        v_c = self.W_in(center_words)  # (batch, dim)
        u_o = self.W_out(context_words)  # (batch, dim)
        u_neg = self.W_out(negative_words)  # (batch, num_neg, dim)
        
        # 2. 正样本得分：u_o^T v_c
        pos_score = torch.sum(u_o * v_c, dim=1)  # (batch,)
        pos_loss = -F.logsigmoid(pos_score)  # -log σ(u_o^T v_c)
        
        # 3. 负样本得分：u_neg^T v_c
        neg_score = torch.bmm(u_neg, v_c.unsqueeze(2)).squeeze(2)  # (batch, num_neg)
        neg_loss = -F.logsigmoid(-neg_score).sum(dim=1)  # -sum log σ(-u_neg^T v_c)
        
        # 4. 总损失
        loss = pos_loss + neg_loss
        
        return loss.mean()
```

### 7.4 训练循环

```python
def train_word2vec(
    model: Word2VecSkipGram,
    pairs: List[Tuple[int, int]],
    vocab: Vocabulary,
    num_negatives: int = 5,
    batch_size: int = 512,
    num_epochs: int = 10,
    learning_rate: float = 0.001
):
    """
    训练 Word2Vec 模型
    
    参数:
        model: Word2Vec 模型
        pairs: (center, context) 对列表
        vocab: 词表
        num_negatives: 负样本数量
        batch_size: 批次大小
        num_epochs: 训练轮数
        learning_rate: 学习率
    """
    # 创建负采样器
    neg_sampler = NegativeSampler(vocab)
    
    # 准备数据
    center_words = torch.tensor([p[0] for p in pairs], dtype=torch.long)
    context_words = torch.tensor([p[1] for p in pairs], dtype=torch.long)
    
    dataset = TensorDataset(center_words, context_words)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        
        for batch_center, batch_context in dataloader:
            # 采样负样本
            batch_negatives = torch.stack([
                torch.tensor(neg_sampler.sample(num_negatives, exclude=c.item()))
                for c in batch_context
            ])
            
            # 前向传播
            optimizer.zero_grad()
            loss = model(batch_center, batch_context, batch_negatives)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    return model
```

### 7.5 词向量可视化

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def visualize_embeddings(model: Word2VecSkipGram, vocab: Vocabulary, words: List[str]):
    """
    可视化词向量（t-SNE / PCA 降维）
    
    参数:
        model: 训练好的 Word2Vec 模型
        vocab: 词表
        words: 要可视化的词列表
    """
    # 获取词向量
    embeddings = []
    valid_words = []
    
    for word in words:
        if word in vocab.word2idx:
            idx = vocab.word2idx[word]
            vec = model.W_in.weight[idx].detach().numpy()
            embeddings.append(vec)
            valid_words.append(word)
    
    embeddings = np.array(embeddings)
    
    # PCA 降维到 2D
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    # 可视化
    plt.figure(figsize=(12, 10))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6)
    
    # 添加词标签
    for i, word in enumerate(valid_words):
        plt.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                    fontsize=9, alpha=0.8)
    
    plt.title('Word Embeddings Visualization (PCA)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True, alpha=0.3)
    plt.savefig('word_embeddings_viz.png', dpi=150, bbox_inches='tight')
    print("词向量可视化已保存：word_embeddings_viz.png")
```

### 7.6 词相似度计算

```python
from scipy.spatial.distance import cosine

def word_similarity(model: Word2VecSkipGram, vocab: Vocabulary, word1: str, word2: str):
    """
    计算两个词的余弦相似度
    
    参数:
        model: Word2Vec 模型
        vocab: 词表
        word1, word2: 要比较的词
    
    返回:
        similarity: 余弦相似度 (0-1)
    """
    if word1 not in vocab.word2idx or word2 not in vocab.word2idx:
        return None
    
    idx1 = vocab.word2idx[word1]
    idx2 = vocab.word2idx[word2]
    
    vec1 = model.W_in.weight[idx1].detach().numpy()
    vec2 = model.W_in.weight[idx2].detach().numpy()
    
    # 余弦相似度 = 1 - 余弦距离
    similarity = 1 - cosine(vec1, vec2)
    
    return similarity


def find_similar_words(
    model: Word2VecSkipGram,
    vocab: Vocabulary,
    word: str,
    top_k: int = 10
):
    """
    查找与给定词最相似的 top_k 个词
    
    参数:
        model: Word2Vec 模型
        vocab: 词表
        word: 查询词
        top_k: 返回数量
    """
    if word not in vocab.word2idx:
        print(f"词 '{word}' 不在词表中")
        return
    
    # 获取查询词向量
    idx = vocab.word2idx[word]
    query_vec = model.W_in.weight[idx].detach().numpy()
    
    # 计算所有词的余弦相似度
    similarities = []
    for i in range(len(vocab)):
        vec = model.W_in.weight[i].detach().numpy()
        sim = 1 - cosine(query_vec, vec)
        similarities.append((i, sim))
    
    # 排序
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # 输出 top_k
    print(f"\n与 '{word}' 最相似的 {top_k} 个词:")
    print("-" * 40)
    for idx, sim in similarities[:top_k]:
        word = vocab.idx2word[idx]
        print(f"{word:20s} {sim:.4f}")
```

---

## 8. 实践技巧与可视化

### 8.1 超参数选择

| 参数 | 典型值 | 说明 |
|-----|--------|------|
| embedding_dim | 100-300 | 词向量维度 |
| window_size | 2-10 | 上下文窗口 |
| num_negatives | 5-20 | 负样本数量 |
| learning_rate | 0.001-0.01 | 学习率 |
| min_freq | 5-50 | 最小词频 |
| power (负采样) | 0.75 | 频率幂次 |

### 8.2 训练技巧

1. **学习率衰减**：
   ```python
   scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
   ```

2. **词频截断**：
   - 过滤低频词（减少噪声）
   - 可选：过滤超高频词（如 "the", "a"）

3. **子词信息**（进阶）：
   - FastText：考虑词内字符 n-gram
   - 适合处理罕见词和 OOV 问题

### 8.3 可视化示例代码

```python
def plot_training_loss(losses: List[float]):
    """绘制训练损失曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Word2Vec Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('training_loss.png', dpi=150)
    print("训练损失曲线已保存：training_loss.png")


def plot_word_analogies(model, vocab):
    """
    可视化词向量类比关系
    
    经典示例：king - man + woman ≈ queen
    """
    # 计算向量运算
    king = model.W_in.weight[vocab.word2idx['king']].detach().numpy()
    man = model.W_in.weight[vocab.word2idx['man']].detach().numpy()
    woman = model.W_in.weight[vocab.word2idx['woman']].detach().numpy()
    
    # king - man + woman
    result_vec = king - man + woman
    
    # 查找最接近的词
    # ...（实现略）
```

---

## 9. 扩展阅读与实现

本章提供详细的数学推导、代码实现和深度分析，帮助巩固前面学习的知识。

### 9.1 CBOW 梯度完整推导

**问题**：给定损失函数
$$
\mathcal{L} = -\mathbf{u}_{w_o}^\top \mathbf{h} + \log \sum_{j=1}^{|V|} \exp(\mathbf{u}_j^\top \mathbf{h})
$$
其中 $\mathbf{h} = \frac{1}{C} \sum_{i=1}^C \mathbf{v}_{w_i}$，推导 $\frac{\partial \mathcal{L}}{\partial \mathbf{v}_{w_i}}$。

**完整推导**：

**步骤 1：定义中间变量**

令 $z_j = \mathbf{u}_j^\top \mathbf{h}$，则 softmax 输出为：
$$
\hat{y}_j = \frac{\exp(z_j)}{\sum_k \exp(z_k)} = \frac{\exp(\mathbf{u}_j^\top \mathbf{h})}{\sum_k \exp(\mathbf{u}_k^\top \mathbf{h})}
$$

**步骤 2：对 $\mathbf{h}$ 求梯度**

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{h}} = -\mathbf{u}_{w_o} + \frac{\partial}{\partial \mathbf{h}} \log \sum_j \exp(\mathbf{u}_j^\top \mathbf{h})
$$

第二项使用链式法则：
$$
\frac{\partial}{\partial \mathbf{h}} \log \sum_j \exp(\mathbf{u}_j^\top \mathbf{h}) = \frac{1}{\sum_j \exp(\mathbf{u}_j^\top \mathbf{h})} \cdot \sum_j \exp(\mathbf{u}_j^\top \mathbf{h}) \cdot \mathbf{u}_j = \sum_j \hat{y}_j \mathbf{u}_j
$$

因此：
$$
\frac{\partial \mathcal{L}}{\partial \mathbf{h}} = -\mathbf{u}_{w_o} + \sum_j \hat{y}_j \mathbf{u}_j = \sum_j (\hat{y}_j - y_j) \mathbf{u}_j = W_{out}(\hat{\mathbf{y}} - \mathbf{y})
$$

**步骤 3：对 $\mathbf{v}_{w_i}$ 求梯度**

由于 $\mathbf{h} = \frac{1}{C} \sum_{i=1}^C \mathbf{v}_{w_i}$，使用链式法则：

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{v}_{w_i}} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}} \cdot \frac{\partial \mathbf{h}}{\partial \mathbf{v}_{w_i}} = \frac{1}{C} W_{out}(\hat{\mathbf{y}} - \mathbf{y})
$$

**结论**：
$$
\boxed{\frac{\partial \mathcal{L}}{\partial \mathbf{v}_{w_i}} = \frac{1}{C} \sum_j (\hat{y}_j - y_j) \mathbf{u}_j}
$$

### 9.2 SGNS 与 PMI 关系的完整证明

**定理**（Levy & Goldberg, 2014）：在最优解处，
$$
\mathbf{v}_w^\top \mathbf{u}_c = \text{PMI}(w, c) - \log k
$$

**证明**：

**步骤 1：写出 SGNS 目标函数**

对于正样本 $(w, c)$ 和 $k$ 个负样本，损失函数为：
$$
\mathcal{L} = -\log \sigma(\mathbf{v}_w^\top \mathbf{u}_c) - \sum_{i=1}^k \mathbb{E}_{c_i \sim P_n} [\log \sigma(-\mathbf{v}_w^\top \mathbf{u}_{c_i})]
$$

**步骤 2：求最优条件**

对 $\mathbf{v}_w^\top \mathbf{u}_c$ 求导并令其为零：

$$
\frac{\partial \mathcal{L}}{\partial (\mathbf{v}_w^\top \mathbf{u}_c)} = -(1 - \sigma(\mathbf{v}_w^\top \mathbf{u}_c)) + k \cdot P_n(c) \cdot \sigma(\mathbf{v}_w^\top \mathbf{u}_c) = 0
$$

**步骤 3：求解**

令 $x = \mathbf{v}_w^\top \mathbf{u}_c$：

$$
-(1 - \sigma(x)) + k P_n(c) \sigma(x) = 0
$$

整理得：
$$
\sigma(x) + k P_n(c) \sigma(x) = 1
$$

$$
\sigma(x) = \frac{1}{1 + k P_n(c)}
$$

**步骤 4：代入 sigmoid 的定义**

由于 $\sigma(x) = \frac{1}{1 + \exp(-x)}$：

$$
\frac{1}{1 + \exp(-x)} = \frac{1}{1 + k P_n(c)}
$$

因此：
$$
\exp(-x) = k P_n(c)
$$

取对数：
$$
-x = \log k + \log P_n(c)
$$

**步骤 5：假设噪声分布**

假设负采样分布 $P_n(c) = P(c)$（即按词频采样），则：

$$
x = -\log k - \log P(c) = \log \frac{1}{P(c)} - \log k
$$

**步骤 6：联系 PMI**

从 Bayes 最优分类器的角度（见原文推导），实际上有：

$$
\sigma(x) = \frac{P(w, c)}{P(w, c) + k P(w)P(c)}
$$

解得：
$$
x = \log \frac{P(w, c)}{P(w)P(c)} - \log k = \text{PMI}(w, c) - \log k
$$

**结论**：
$$
\boxed{\mathbf{v}_w^\top \mathbf{u}_c = \text{PMI}(w, c) - \log k}
$$

**直观理解**：
- 词向量的内积 $\mathbf{v}_w^\top \mathbf{u}_c$ 等于 PMI 值减去一个常数 $\log k$
- 这说明 Word2Vec 学到的词向量**确实**在重构 PMI 矩阵
- 常数偏移 $\log k$ 不影响相对关系（相似度计算时会被抵消）

---

### 9.3 负采样分布的 $3/4$ 次方分析

**问题**：为什么使用 $P(w) \propto f(w)^{3/4}$ 而不是 $P(w) \propto f(w)$？

**完整分析**：

**原始分布**（按词频）：
$$
P_{freq}(w) = \frac{f(w)}{\sum_{w'} f(w')}
$$

**问题**：
- 高频词（如 "the", "a"）概率过高
- 负采样时会过度采样这些词
- 导致模型对高频词"过拟合"

**改进分布**（$3/4$ 次方）：
$$
P_{sample}(w) = \frac{f(w)^{3/4}}{\sum_{w'} f(w')^{3/4}}
$$

**效果对比**（假设词频分布服从 Zipf 定律）：

| 词频排名 | 原始频率 | $freq^{3/4}$ | 相对概率变化 |
|---------|---------|-------------|-------------|
| 第 1 名 | 10000 | 10000^{0.75} ≈ 1785 | 降低 5.6 倍 |
| 第 100 名 | 100 | 100^{0.75} ≈ 31.6 | 降低 3.2 倍 |
| 第 10000 名 | 1 | 1^{0.75} = 1 | 不变 |

**结论**：
- $3/4$ 次方"压平"了分布
- 高频词采样概率降低，低频词相对提升
- 平衡了学习信号，避免被高频词主导

---

### 9.4 CBOW 完整实现

**完整代码**：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Word2VecCBOW(nn.Module):
    """
    CBOW with Negative Sampling
    
    数学公式：
        h = (1/C) * sum(v_context)
        L = -log σ(u_target^T h) - sum_{i=1}^k log σ(-u_neg_i^T h)
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 300):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # 输入词向量（上下文词用）
        self.W_in = nn.Embedding(vocab_size, embedding_dim)
        # 输出词向量（目标词用）
        self.W_out = nn.Embedding(vocab_size, embedding_dim)
        
        # Xavier 初始化
        nn.init.xavier_uniform_(self.W_in.weight)
        nn.init.xavier_uniform_(self.W_out.weight)
    
    def forward(self, context_words: torch.Tensor, 
                target_word: torch.Tensor,
                negative_words: torch.Tensor) -> torch.Tensor:
        """
        参数:
            context_words: (batch_size, context_size) 上下文词索引
            target_word: (batch_size,) 目标词索引
            negative_words: (batch_size, num_negatives) 负样本索引
        """
        batch_size = context_words.size(0)
        
        # 1. 获取上下文词向量并平均
        context_embeds = self.W_in(context_words)  # (batch, context_size, dim)
        h = torch.mean(context_embeds, dim=1)  # (batch, dim)
        
        # 2. 正样本得分
        u_target = self.W_out(target_word)  # (batch, dim)
        pos_score = torch.sum(u_target * h, dim=1)  # (batch,)
        pos_loss = -F.logsigmoid(pos_score)
        
        # 3. 负样本得分
        u_neg = self.W_out(negative_words)  # (batch, num_neg, dim)
        # 扩展 h 以便广播: (batch, 1, dim)
        h_expanded = h.unsqueeze(1)
        neg_score = torch.sum(u_neg * h_expanded, dim=2)  # (batch, num_neg)
        neg_loss = -F.logsigmoid(-neg_score).sum(dim=1)
        
        # 4. 总损失
        loss = pos_loss + neg_loss
        return loss.mean()
```

**与 Skip-gram 的区别**：

| 特性 | CBOW | Skip-gram |
|------|------|-----------|
| 输入 | 多个上下文词 | 单个中心词 |
| 隐藏层 | 上下文向量平均 | 直接是中心词向量 |
| 输出 | 预测单个目标词 | 预测多个上下文词 |
| 训练速度 | 快 | 慢 |
| 罕见词效果 | 较差 | 较好 |

---

### 9.5 Hierarchical Softmax 原理与实现

**核心思想**：
- 用二叉树（Huffman 树）编码词表
- 从根到叶子的路径表示预测过程
- 每个内部节点做二分类（左/右）
- 复杂度从 $O(|V|)$ 降为 $O(\log |V|)$

**Huffman 树构建**：
- 按词频构建最优二叉树
- 高频词路径短，低频词路径长
- 符合信息论中的最优编码

**路径概率计算**：

对于词 $w$，设其路径为 $n_1, n_2, \ldots, n_L$（$L$ 为路径长度）：

$$
P(w \mid \text{context}) = \prod_{j=1}^{L-1} \sigma(\mathbf{v}_{n_j}^\top \mathbf{h})^{b_j} (1 - \sigma(\mathbf{v}_{n_j}^\top \mathbf{h}))^{1-b_j}
$$

其中 $b_j \in \{0, 1\}$ 表示走左/右分支。

**与 Negative Sampling 的对比**：

| 特性 | Hierarchical Softmax | Negative Sampling |
|------|---------------------|-------------------|
| 复杂度 | $O(\log |V|)$ | $O(k)$ |
| 概率精确性 | 精确 | 近似 |
| 实现难度 | 较复杂（需构建树） | 简单 |
| 对罕见词 | 更好（路径短） | 一般 |
| 当前使用 | 较少 | 主流 |

---

### 9.6 Word2Vec vs Transformer Embedding 深度对比

**问题**：Word2Vec 与 Transformer 的 Embedding 有什么区别？

**完整对比**：

| 维度 | Word2Vec | Transformer |
|------|----------|-------------|
| **训练方式** | 静态预训练，词向量固定 | 动态学习，随任务微调 |
| **上下文依赖** | 无，每个词只有一个向量 | 有，同一词在不同上下文表示不同 |
| **位置信息** | 无 | 有（Positional Encoding） |
| **语义理解** | 浅层语义（共现统计） | 深层语义（多层注意力） |
| **计算复杂度** | 低 | 高（$O(n^2)$ 注意力） |
| **适用场景** | 资源受限、快速部署 | 需要深度理解的复杂任务 |

**关键区别详解**：

**1. 静态 vs 动态**
- Word2Vec："bank" 只有一个向量，无法区分 "river bank" 和 "bank account"
- Transformer："bank" 的表示随上下文变化，自动消歧

**2. 位置信息**
- Word2Vec：完全忽略词序（"狗咬人"和"人咬狗"表示相同）
- Transformer：通过位置编码保留词序信息

**3. 训练目标**
- Word2Vec：预测局部上下文（窗口内）
- Transformer：预测全局上下文（整个句子）

**实践建议**：
- 简单任务/资源受限：Word2Vec 足够
- 复杂 NLP 任务：使用 Transformer（BERT/GPT 等）

---

### 9.7 词向量质量评估方法

**问题**：如何评估词向量的质量？（除了可视化）

**评估方法分类**：

**1. 内在评估（Intrinsic Evaluation）**

**词相似度任务**：
- 数据集：WordSim-353, SimLex-999, MEN
- 计算词向量的余弦相似度，与人工标注的相似度比较
- 指标：Spearman 相关系数

**词类比任务**：
- 经典例子：king - man + woman ≈ queen
- 数据集：Google Analogy Test Set
- 准确率：正确预测的比例

**实现代码**：

```python
def evaluate_analogy(model, vocab, analogies):
    """
    评估词类比任务
    analogies: [(a, b, c, expected), ...] 如 ("king", "man", "woman", "queen")
    """
    correct = 0
    total = 0
    
    for a, b, c, expected in analogies:
        if not all(w in vocab.word2idx for w in [a, b, c, expected]):
            continue
        
        # 计算：a - b + c
        vec_a = model.W_in.weight[vocab.word2idx[a]].detach()
        vec_b = model.W_in.weight[vocab.word2idx[b]].detach()
        vec_c = model.W_in.weight[vocab.word2idx[c]].detach()
        
        result_vec = vec_a - vec_b + vec_c
        
        # 找最接近的词（排除 a, b, c）
        similarities = []
        for word, idx in vocab.word2idx.items():
            if word in [a, b, c]:
                continue
            vec = model.W_in.weight[idx].detach()
            sim = F.cosine_similarity(result_vec.unsqueeze(0), 
                                       vec.unsqueeze(0))
            similarities.append((word, sim.item()))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        predicted = similarities[0][0]
        
        if predicted == expected:
            correct += 1
        total += 1
    
    return correct / total if total > 0 else 0
```

**2. 外在评估（Extrinsic Evaluation）**

**下游任务性能**：
- 文本分类：使用词向量作为特征，看分类准确率
- 命名实体识别（NER）
- 情感分析
- 机器翻译

**优点**：直接反映在实际任务中的效果
**缺点**：受下游模型影响，难以单独评估词向量

**3. 其他评估指标**

**词汇重叠度**：
- 与人工构建的同义词词典（如 WordNet）的重叠程度

**聚类质量**：
- 用 K-means 聚类，检查是否按语义聚类
- 指标： purity, NMI (Normalized Mutual Information)

**推荐评估流程**：
1. 先用词相似度/类比任务快速筛选
2. 在目标下游任务上验证最终效果
3. 可视化作为辅助检查

---

## 附录：符号表

| 符号 | 含义 | 典型值 |
|-----|------|--------|
| $V$ | 词表大小 | 100,000 |
| $d$ | 嵌入维度 | 300 |
| $m$ | 上下文窗口 | 2-5 |
| $k$ | 负样本数量 | 5-20 |
| $W_{in}$ | 输入词向量矩阵 | $\mathbb{R}^{d \times V}$ |
| $W_{out}$ | 输出词向量矩阵 | $\mathbb{R}^{d \times V}$ |
| $\mathbf{v}_w$ | 词 $w$ 的输入向量 | $\mathbb{R}^d$ |
| $\mathbf{u}_w$ | 词 $w$ 的输出向量 | $\mathbb{R}^d$ |
| $P(w)$ | 负采样分布 | $f(w)^{3/4} / Z$ |

---

## 参考文献

1. Mikolov, T., et al. (2013). [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781). ICLR 2013.
2. Mikolov, T., et al. (2013). [Distributed Representations of Words and Phrases](https://arxiv.org/abs/1310.4546). NeurIPS 2013.
3. Levy, O., & Goldberg, Y. (2014). [Neural Word Embedding as Implicit Matrix Factorization](https://arxiv.org/abs/1402.3722). NeurIPS 2014.
4. [Word2Vec 原始代码](https://code.google.com/archive/p/word2vec/)

---

最后更新：2026-03-11
