# Word Embedding 数学原理与实现 —— 从共现矩阵到词向量

> **教学对象**：具备本科数学基础（线性代数、概率论、微积分）的学习者  
> **教学目标**：理解 Word2Vec 的数学本质，掌握从共现矩阵到词向量的推导  
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
2. **语义鸿沟**：任意两个词的点积为 0，无法衡量相似度
   $$
   \mathbf{v}_{\text{king}}^\top \mathbf{v}_{\text{queen}} = 0, \quad \mathbf{v}_{\text{king}}^\top \mathbf{v}_{\text{apple}} = 0
   $$
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
   \text{cosine}(\mathbf{v}_{\text{king}}, \mathbf{v}_{\text{queen}}) \approx 0.8
   $$
2. **计算高效**：矩阵规模 $O(|V| \cdot d)$
3. **可迁移性**：预训练词向量可用于多种下游任务

### 1.3 本科数学知识映射表

| 数学概念 | Word Embedding 中的应用 | 代码对应 |
|---------|----------------------|---------|
| 条件概率 $P(w_2|w_1)$ | Skip-gram 预测上下文 | `softmax(W_out @ h)` |
| 矩阵分解 $X \approx UV$ | 词向量学习 | `W_in, W_out` |
| 梯度下降 $\theta - \eta \nabla_\theta \mathcal{L}$ | 参数更新 | `optimizer.step()` |
| Sigmoid 函数 $\sigma(x)$ | 负采样二分类 | `torch.sigmoid()` |
| PMI 点互信息 | 共现矩阵分析 | `np.log(P_ij / (P_i * P_j))` |

---

## 2. 从 One-Hot 到 Dense Embedding

### 2.1 计算复杂度分析

**One-Hot + Softmax 的计算成本**：

假设词表 $|V| = 100,000$，嵌入维度 $d = 300$。

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

### 2.2 优化策略

| 方法 | 复杂度 | 思想 |
|-----|--------|------|
| Standard Softmax | $O(|V| \cdot d)$ | 全词表归一化 |
| Hierarchical Softmax | $O(d \cdot \log |V|)$ | 二叉树路径 |
| Negative Sampling | $O(k \cdot d)$ | 采样 $k$ 个负例 |

**Negative Sampling 优势**：
- $k = 5 \sim 20$（典型值）
- 加速比：$|V| / k \approx 5,000 \sim 20,000$ 倍

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

其中：
- $P(i, j) = \frac{X_{ij}}{\sum_{i',j'} X_{i'j'}}$（联合概率）
- $P(i) = \frac{\sum_j X_{ij}}{\sum_{i',j'} X_{i'j'}}$（边缘概率）

**解释**：
- $\text{PMI} > 0$：正相关（共现频率高于随机期望）
- $\text{PMI} = 0$：独立
- $\text{PMI} < 0$：负相关（共现频率低于随机期望）

**问题**：PMI 对低频词敏感，可能产生很大的负值。

### 3.3 PPMI（Positive PMI）

**改进**：将负值截断为 0。

$$
\text{PPMI}(i, j) = \max(\text{PMI}(i, j), 0)
$$

**优势**：
- 减少噪声（低频词的偶然共现）
- 矩阵更稀疏，便于存储和计算

### 3.4 从 PPMI 到词向量

**思想**：对 PPMI 矩阵进行低秩分解。

$$
\text{PPMI} \approx W_{in}^\top W_{out}
$$

其中：
- $W_{in} \in \mathbb{R}^{d \times |V|}$（输入词向量）
- $W_{out} \in \mathbb{R}^{d \times |V|}$（输出词向量）

**方法**：
1. SVD 分解：$\text{PPMI} = U \Sigma V^\top$，取前 $d$ 个奇异值
2. Word2Vec：通过优化目标隐式分解

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
   其中 $\mathbf{v}_{w} = W_{in}[:, w]$。

2. 输出层 softmax：
   $$
   P(w_t \mid \text{context}) = \frac{\exp(\mathbf{u}_{w_t}^\top \mathbf{h})}{\sum_{w \in V} \exp(\mathbf{u}_w^\top \mathbf{h})}
   $$
   其中 $\mathbf{u}_w = W_{out}[:, w]$。

**损失函数**（负对数似然）：
$$
\mathcal{L}_{CBOW} = -\log P(w_t \mid \text{context}) = -\mathbf{u}_{w_t}^\top \mathbf{h} + \log \sum_{w \in V} \exp(\mathbf{u}_w^\top \mathbf{h})
$$

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

## 6. 梯度推导与参数更新

### 6.1 CBOW 梯度推导

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

### 6.2 负采样梯度

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

### 6.3 参数更新总结表

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

## 9. 练习与思考题

### 9.1 数学推导练习

**练习 1**：推导 CBOW 的梯度

给定损失函数：
$$
\mathcal{L} = -\mathbf{u}_{w_o}^\top \mathbf{h} + \log \sum_{j=1}^{|V|} \exp(\mathbf{u}_j^\top \mathbf{h})
$$

其中 $\mathbf{h} = \frac{1}{C} \sum_{i=1}^C \mathbf{v}_{w_i}$。

推导 $\frac{\partial \mathcal{L}}{\partial \mathbf{v}_{w_i}}$。

**练习 2**：证明 SGNS 与 PMI 的关系

证明在最优点：
$$
\mathbf{v}_w^\top \mathbf{u}_c \approx \text{PMI}(w, c) - \log k
$$

**练习 3**：分析负采样分布

为什么使用 $P(w) \propto f(w)^{3/4}$ 而不是 $P(w) \propto f(w)$？

### 9.2 代码实现练习

**练习 4**：实现 CBOW 模型

```python
class Word2VecCBOW(nn.Module):
    """实现 CBOW 模型"""
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        # TODO: 定义输入/输出嵌入层
        pass
    
    def forward(self, context_words, target_words, negative_words):
        # TODO: 实现前向传播
        # 1. 获取上下文词向量并平均
        # 2. 计算正样本损失
        # 3. 计算负样本损失
        # 4. 返回总损失
        pass
```

**练习 5**：添加 Hierarchical Softmax

实现基于二叉树的 Hierarchical Softmax，将复杂度从 $O(|V|)$ 降为 $O(\log |V|)$。

### 9.3 思考题

**思考 1**：Word2Vec 与 Transformer 的 Embedding 有什么区别？

**思考 2**：为什么 Skip-gram 对罕见词效果更好？

**思考 3**：如何评估词向量的质量？（除了可视化）

---

## 附录：符号表

| 符号 | 含义 | 典型值 |
|-----|------|--------|
| $|V|$ | 词表大小 | 100,000 |
| $d$ | 嵌入维度 | 300 |
| $m$ | 上下文窗口 | 2-5 |
| $k$ | 负样本数量 | 5-20 |
| $W_{in}$ | 输入词向量矩阵 | $\mathbb{R}^{d \times |V|}$ |
| $W_{out}$ | 输出词向量矩阵 | $\mathbb{R}^{d \times |V|}$ |
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

**最后更新**: 2026-03-11  
**作者**: OpenClaw Engineer (AI + Mathematics Professor)  
**许可证**: MIT
