# Word Embedding

## 1. 论文出处

Word2Vec 是 Google 团队在 2013 年提出的，用于学习单词向量表示的模型。核心论文有两篇：

- [Efficient Estimation of Word Representations in Vector Space (Mikolov et al., 2013)](https://arxiv.org/abs/1301.3781)  
- [Distributed Representations of Words and Phrases and their Compositionality (Mikolov et al., 2013)](https://arxiv.org/abs/1310.4546)  

本文笔记基于以上两篇 Word2Vec 原始论文，结合数学表达和训练流程，整理 word embedding 的核心思想与实现方式。

---

## 2. Word Embedding 的目标

**目标**：将词语 $w \in V$（ $V$ 为词表）映射为一个 $d$ 维向量

$$
f: w \mapsto \mathbf{v}_w \in \mathbb{R}^d
$$

使得 **语义相近的词向量距离更近**。

一般而言，最为简单的做法是给每一个词单独设置一个维度，把 $V$ 映射到一个高维的向量空间，最终得到的表示是一个 $V$ 维的 one-hot 向量。  
但是，这样的表示无法体现词与词之间的语义关系，同时计算过程中涉及到的矩阵规模是 $V \times V$，导致计算代价极高，并且矩阵大多是稀疏的。

Word2Vec 的设计目标是将词映射到一个低维、稠密的向量空间（embedding space），其维度设为 $H$。  
在这种情况下，输入层到隐藏层、隐藏层到输出层的计算复杂度可以近似写为：

$$
Q = H \times H + H \times V
$$

- $H \times H$ ：输入到隐藏层的计算复杂度  
- $H \times V$ ：隐藏层到输出层的 softmax 计算复杂度  

其中， $H \times V$ 的规模往往远大于 $H \times H$ ，成为训练的主要瓶颈。  
例如，当 $H=300$， $V=100,000$ 时， $H \times H = 90,000$ ，而 $H \times V = 30,000,000$ 。可以看出，绝大部分计算开销来自输出层的 softmax。

因此，Word2Vec 原始论文提出了两种优化方法来降低复杂度：  
- **Hierarchical Softmax**：将复杂度降为 $O(H \times \log V)$  
- **Negative Sampling**：将复杂度降为 $O(H \times k)$（其中 $k \ll V$）
---

## 3. 两种主要算法

### 3.1 Continuous Bag of Words (CBOW)

**思路**：用上下文（context words）预测目标词（center word）。

* **输入**：上下文词向量 ${w_{t-m}, ..., w_{t-1}, w_{t+1}, ..., w_{t+m}}$
* **输出**：中心词 $w_t$

#### 数学表达

1. **输入向量表示**：
   对上下文单词做平均：
   
$$
   \mathbf{h} = \frac{1}{2m}\sum_{-m \leq j \leq m, j \neq 0} \mathbf{v}*{w*{t+j}}
$$

3. **预测概率分布**（Softmax 层）：

$$
   P(w_t \mid context) = \frac{\exp(\mathbf{u}*{w_t}^\top \mathbf{h})}{\sum*{w \in V} \exp(\mathbf{u}_w^\top \mathbf{h})}
$$

其中：

* $\mathbf{v}_w \in \mathbb{R}^d$ 为输入嵌入
* $\mathbf{u}_w \in \mathbb{R}^d$ 为输出嵌入

3. **损失函数**：最大化正确词的概率

$$
   \mathcal{L}_{CBOW} = -\log P(w_t \mid context)
$$

---

### 3.2 Skip-gram

**思路**：用中心词预测上下文。

* **输入**：中心词 $w_t$
* **输出**：上下文词 ${w_{t-m}, ..., w_{t-1}, w_{t+1}, ..., w_{t+m}}$

#### 数学表达

1. **预测概率分布**：

$$
   P(w_{t+j} \mid w_t) = \frac{\exp(\mathbf{u}*{w*{t+j}}^\top \mathbf{v}*{w_t})}{\sum*{w \in V} \exp(\mathbf{u}*w^\top \mathbf{v}*{w_t})}
$$

2. **损失函数**：最大化上下文词的概率

$$
   \mathcal{L}*{SkipGram} = - \sum*{-m \leq j \leq m, j \neq 0} \log P(w_{t+j} \mid w_t)
$$

---

## 4. 实际训练例子

### 数据示例

语料：`the cat sat on the mat`
词表 $V = {\text{the, cat, sat, on, mat}}$

设窗口大小 $m=2$，词向量维度 $d=2$（方便展示）。

---

### 4.1 CBOW 例子

句子片段：`the cat sat`

* **上下文**：`the, sat`
* **目标词**：`cat`

1. **输入**：
   假设词向量初始化：
   
$$
   v_{the} = (0.2, 0.1), \quad v_{sat} = (0.4, -0.1)
$$

上下文平均：

$$
h = \frac{v_{the} + v_{sat}}{2} = (0.3, 0.0)
$$

2. **输出预测**（Softmax）：

$$
   P(w \mid h) = \frac{\exp(u_w^\top h)}{\sum_{w' \in V} \exp(u_{w'}^\top h)}
$$

若 $u_{cat}=(0.1,0.2)$，则

$$
u_{cat}^\top h = 0.03
$$

计算所有词的 softmax，得到预测分布。

3. **损失函数**：
   目标词是 "cat"，损失：
   
$$
   \mathcal{L} = -\log P(\text{cat} \mid the,sat)
$$

4. **参数更新**：
   对 $v_{the}, v_{sat}, u_{cat}$ 等做梯度下降。

---

### 4.2 Skip-gram 例子

句子片段：`the cat sat`

* **中心词**：`cat`
* **上下文**：`the, sat`

1. **输入**：
   假设 $v_{cat} = (0.5,0.2)$

2. **预测上下文词**：

$$
   P(the \mid cat) = \frac{\exp(u_{the}^\top v_{cat})}{\sum_{w \in V}\exp(u_w^\top v_{cat})}
$$

3. **损失函数**：

$$
   \mathcal{L} = - \big( \log P(the \mid cat) + \log P(sat \mid cat) \big)
$$

4. **参数更新**：
   调整 $v_{cat}$ 与对应的 $u_{the}, u_{sat}$。

---

## 5. 训练加速技巧

* **Negative Sampling**：不对整个词表做 softmax，而是对正样本 + 若干负样本做二分类。
* **Hierarchical Softmax**：使用霍夫曼树加速 softmax 计算。

---

## 6. 最终输出

训练完成后，每个词 $w \in V$ 得到一个向量 $\mathbf{v}_w \in \mathbb{R}^d$。

例如：

```
the  -> [0.21, -0.11, 0.05, ...]
cat  -> [0.12, 0.33, -0.22, ...]
sat  -> [-0.09, 0.44, 0.15, ...]
```

这些向量可以用于：

* 计算词语相似度（cosine similarity）
* 作为下游 NLP 任务的输入特征（分类、翻译等）

---


## 8. Negative Sampling （负采样）

### 背景

在 Word2Vec 的 CBOW 或 Skip-gram 中，预测目标词时需要对整个词表 $V$ 做 **softmax**，计算代价很大：
[
P(w \mid h) = \frac{\exp(\mathbf{u}*w^\top h)}{\sum*{w' \in V} \exp(\mathbf{u}_{w'}^\top h)}
]

当词表规模达到百万级时，训练效率极低。

**Negative Sampling** 提供了一种近似方法：将多分类问题转化为 **二分类问题**。

---

### 8.1 思想

对于一个正样本对 $(w_t, w_{context})$，目标是区分它和若干负样本 $(w_t, w_{neg})$。

* **正样本**：真实出现的中心词–上下文对，标签 $y=1$
* **负样本**：随机采样的词–上下文对，标签 $y=0$

最终目标：最大化正样本的预测概率，最小化负样本的预测概率。

---

### 8.2 数学公式

1. **二分类预测函数**
   采用 sigmoid 函数：
   
$$
   P(D=1 \mid w_c, w_o) = \sigma(\mathbf{u}*{w_o}^\top \mathbf{v}*{w_c}) = \frac{1}{1 + \exp(-\mathbf{u}*{w_o}^\top \mathbf{v}*{w_c})}
$$

* $w_c$：中心词
* $w_o$：上下文词（或负样本词）
* $\mathbf{v}, \mathbf{u}$：输入/输出嵌入向量

2. **损失函数**
   Skip-gram with Negative Sampling (SGNS) 的损失函数：

$$
   \mathcal{L} = - \log \sigma(\mathbf{u}*{w_o}^\top \mathbf{v}*{w_c}) - \sum_{i=1}^k \log \sigma(-\mathbf{u}*{w_i}^\top \mathbf{v}*{w_c})
$$

其中：

* $w_o$ = 正样本（真实上下文词）
* $w_i$ = $k$ 个负样本（随机词）

3. **目标**

* 第一个项：最大化正样本的相似度
* 第二个项：最小化负样本的相似度

---

### 8.3 训练例子

语料：`the cat sat on the mat`
窗口大小 $m=1$

训练样本（Skip-gram）：

* 中心词 `cat`，上下文词 `the`
* 正样本对：(`cat`, `the`)

负采样：随机从词表 $V={the, cat, sat, on, mat}$ 中挑选 $k=2$ 个负样本，例如 `on, mat`。

---

#### Step 1: 正样本预测

假设：

$$
v_{cat} = (0.5, 0.1), \quad u_{the} = (0.2, 0.4)
$$

计算：

$$
u_{the}^\top v_{cat} = 0.5 \times 0.2 + 0.1 \times 0.4 = 0.14
$$

$$
\sigma(0.14) \approx 0.535
$$

---

#### Step 2: 负样本预测

假设：

$$
u_{on} = (0.1, -0.3), \quad u_{mat} = (-0.2, 0.2)
$$

* 对 `on`：

$$
  u_{on}^\top v_{cat} = 0.5 \times 0.1 + 0.1 \times (-0.3) = 0.02
$$
$$
  \sigma(-0.02) \approx 0.495
$$

* 对 `mat`：

$$
  u_{mat}^\top v_{cat} = 0.5 \times (-0.2) + 0.1 \times 0.2 = -0.08
$$
$$
  \sigma(-(-0.08)) = \sigma(0.08) \approx 0.520
$$

---

#### Step 3: 损失函数

$$
\mathcal{L} = -\log 0.535 - \log 0.495 - \log 0.520
$$
$$
\approx 0.625 + 0.704 + 0.653 = 1.982
$$

训练时通过梯度下降更新 $v_{cat}, u_{the}, u_{on}, u_{mat}$。

---

### 8.4 总结

* **优势**：

  * 不需要计算整个词表的 softmax，训练效率大幅提升
  * 常用 $k=5\sim20$ 的负采样即可
* **劣势**：

  * 损失函数不再是最大似然估计，而是近似方法
  * 对低频词效果较差

---

## Word2Vec 三种训练方式对比

| 方法             | 输出目标           | 损失函数形式                                                                                                | 复杂度  | 
| -------------- | -------------- | ----------------------------------------------------------------------------------------------------- | ------ |
| CBOW           | 中心词            | $\mathcal{L} = -\log P(w_t \mid context)$                                                             | $O( \| V \| )$ |
| Skip-gram      | 上下文词集合         | $\mathcal{L} = - \sum \log P(w_{t+j} \mid w_t)$                                                       | $O( \| V \| )$ |
| Skip-gram + NS | 正样本 + $k$ 个负样本 | $\mathcal{L} = - \log \sigma(u_{w_o}^\top v_{w_c}) - \sum_{i=1}^k \log \sigma(-u_{w_i}^\top v_{w_c})$ | $O(k)$ | 

---


