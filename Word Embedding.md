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
f: w \mapsto \vec{v}_w \in \mathbb{R}^d
$$

使得 **语义相近的词向量距离更近**。

一般而言，最为简单的做法是给每一个词单独设置一个维度，把 $V$ 映射到一个高维的向量空间，最终得到的表示是一个 $V$ 维的 one-hot 向量。  
但是，这样的表示无法体现词与词之间的语义关系，同时计算过程中涉及到的矩阵规模是 $V \times V$，导致计算代价极高，并且矩阵大多是稀疏的。

Word2Vec 的设计目标是将词映射到一个低维、稠密的向量空间（embedding space），其维度设为 $d$。  
在这种情况下，输入层到隐藏层、隐藏层到输出层的计算复杂度可以近似写为：

$$
Q = d \times d + d \times V
$$

- $d \times d$ ：输入到隐藏层的计算复杂度  
- $d \times V$ ：隐藏层到输出层的 softmax 计算复杂度  

其中， $H \times V$ 的规模往往远大于 $H \times H$ ，成为训练的主要瓶颈。  
例如，当 $H=300$， $V=100,000$ 时， $H \times H = 90,000$ ，而 $H \times V = 30,000,000$ 。可以看出，绝大部分计算开销来自输出层的 softmax。

因此，Word2Vec 原始论文提出了两种优化方法来降低复杂度：  
- **Hierarchical Softmax**：将复杂度降为 $O(H \times \log V)$  
- **Negative Sampling**：将复杂度降为 $O(H \times k)$（其中 $k \ll V$）
---

## 3. 两种主要算法

### 3.1 Continuous Bag of Words (CBOW)

**思路**：用上下文（context words）预测目标词（center word）。

* **输入**：上下文词 ${w_{t-m}, ..., w_{t-1}, w_{t+1}, ..., w_{t+m}}$
* **输出**：中心词 $w_t$

#### 数学表达

1. **输入向量表示**：
   对上下文单词做平均：
   
$$
   \vec{h} = \frac{1}{2m}\sum_{-m \leq j \leq m, j \neq 0} \vec{v}*{w*{t+j}}
$$

3. **预测概率分布**（Softmax 层）：

$$
   P(w_t \mid context) = \frac{\exp(\vec{u}*{w_t}^\top \vec{h})}{\sum*{w \in V} \exp(\vec{u}_w^\top \vec{h})}
$$

其中：

* $\vec{v}_w \in \mathbb{R}^d$ 为输入嵌入
* $\vec{u}_w \in \mathbb{R}^d$ 为输出嵌入

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
   P(w_{t+j} \mid w_t) = \frac{\exp(\vec{u}*{w*{t+j}}^\top \vec{v}*{w_t})}{\sum*{w \in V} \exp(\vec{u}*w^\top \vec{v}*{w_t})}
$$

2. **损失函数**：最大化上下文词的概率

$$
   \mathcal{L}*{SkipGram} = - \sum*{-m \leq j \leq m, j \neq 0} \log P(w_{t+j} \mid w_t)
$$


## 4. 最终输出

训练完成后，每个词 $w \in V$ 得到一个向量 $\vec{v}_w \in \mathbb{R}^d$。

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


## 5. Negative Sampling （负采样）

### 背景

在 Word2Vec 的 CBOW 或 Skip-gram 中，预测目标词时需要对整个词表 $V$ 做 **softmax**，计算代价很大：

$$
P(w \mid h) = \frac{\exp(\vec{u}*w^\top h)}{\sum*{w' \in V} \exp(\vec{u}_{w'}^\top h)}
$$

当词表规模达到百万级时，训练效率极低。

**Negative Sampling** 提供了一种近似方法：将多分类问题转化为 **二分类问题**。

---

### 5.1 思想

对于一个正样本对 $(w_t, w_{context})$，目标是区分它和若干负样本 $(w_t, w_{neg})$。

* **正样本**：真实出现的中心词–上下文对，标签 $y=1$
* **负样本**：随机采样的词–上下文对，标签 $y=0$

最终目标：最大化正样本的预测概率，最小化负样本的预测概率。

---

### 5.2 数学公式

1. **二分类预测函数**
   采用 sigmoid 函数：
   
$$
   P(D=1 \mid w_c, w_o) = \sigma(\vec{u}*{w_o}^\top \vec{v}*{w_c}) = \frac{1}{1 + \exp(-\vec{u}*{w_o}^\top \vec{v}*{w_c})}
$$

* $w_c$：中心词
* $w_o$：上下文词（或负样本词）
* $\vec{v}, \vec{u}$：输入/输出嵌入向量

2. **损失函数**
   Skip-gram with Negative Sampling (SGNS) 的损失函数：

$$
   \mathcal{L} = - \log \sigma(\vec{u}*{w_o}^\top \vec{v}*{w_c}) - \sum_{i=1}^k \log \sigma(-\vec{u}*{w_i}^\top \vec{v}*{w_c})
$$

其中：

* $w_o$ = 正样本（真实上下文词）
* $w_i$ = $k$ 个负样本（随机词）

3. **目标**

* 第一个项：最大化正样本的相似度
* 第二个项：最小化负样本的相似度

---

### 5.3 训练例子

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

### 5.4 总结

* **优势**：

  * 不需要计算整个词表的 softmax，训练效率大幅提升
  * 常用 $k=5\sim20$ 的负采样即可
* **劣势**：

  * 损失函数不再是最大似然估计，而是近似方法
  * 对低频词效果较差

---

## 9.Word2Vec 三种训练方式对比

| 方法             | 输出目标           | 损失函数形式                                                                                                | 复杂度  | 
| -------------- | -------------- | ----------------------------------------------------------------------------------------------------- | ------ |
| CBOW           | 中心词            | $\mathcal{L} = -\log P(w_t \mid context)$                                                             | $O( \| V \| )$ |
| Skip-gram      | 上下文词集合         | $\mathcal{L} = - \sum \log P(w_{t+j} \mid w_t)$                                                       | $O( \| V \| )$ |
| Skip-gram + NS | 正样本 + $k$ 个负样本 | $\mathcal{L} = - \log \sigma(u_{w_o}^\top v_{w_c}) - \sum_{i=1}^k \log \sigma(-u_{w_i}^\top v_{w_c})$ | $O(k)$ | 

---


## 10.附录：CBOW 模型：损失函数与梯度更新公式推导

### 10.1 模型设定

* 词表大小： $V$ 
* 向量维度： $N$ 
* 输入（上下文词）： $w_1, w_2, \dots, w_C$ 
* 输出（目标词）： $w_o$ 

模型使用两套嵌入矩阵：

| 符号                 | 说明              | 维度             |
| ------------------ | --------------- | -------------- |
| $V_{\text{in}}$  | 输入嵌入矩阵（用于上下文词）  | $N \times V$ |
| $V_{\text{out}}$ | 输出嵌入矩阵（用于预测目标词） | $N \times V$ |

---

### 10.2 前向传播（Forward）

对上下文中每个词 ( w_i )，取其输入向量：
$$
\mathbf{v}*{w_i} = V*{\text{in}}[:, w_i]
$$

求它们的平均，得到隐藏层表示：
$$
\mathbf{h} = \frac{1}{C} \sum_{i=1}^C \mathbf{v}_{w_i}
$$

计算每个词的预测得分（logit）：
$$
z_j = \mathbf{u}_j^\top \mathbf{h}, \quad \text{其中 } \mathbf{u}*j = V*{\text{out}}[:, j]
$$

计算 softmax 概率：
$$
\hat{y}*j = \frac{\exp(z_j)}{\sum*{k=1}^V \exp(z_k)}
$$

---

### 10.3 损失函数（单样本）

CBOW 的目标是最大化预测正确中心词 ( w_o ) 的概率，对应的负对数似然为：
$$
\mathcal{L} = -\log P(w_o|\text{context})
= -\mathbf{u}_{w_o}^\top \mathbf{h}

* \log\sum_{j=1}^V \exp(\mathbf{u}_j^\top \mathbf{h})
  $$

---

### 10.4 梯度求导

#### （1）对输出向量 ( \mathbf{u}_j ) 的梯度：

softmax 输出：
$$
\hat{y}_j = \frac{\exp(\mathbf{u}_j^\top \mathbf{h})}{\sum_k \exp(\mathbf{u}_k^\top \mathbf{h})}
$$

one-hot 目标：
$$
y_j =
\begin{cases}
1, & j = w_o \
0, & \text{otherwise}
\end{cases}
$$

因此：
$$
\frac{\partial \mathcal{L}}{\partial \mathbf{u}_j}
= (\hat{y}_j - y_j),\mathbf{h}
$$

**矩阵形式：**
$$
\frac{\partial \mathcal{L}}{\partial V_{\text{out}}}
= \mathbf{h},(\hat{\mathbf{y}} - \mathbf{y})^\top
$$

---

#### （2）对隐藏层 ( \mathbf{h} ) 的梯度：

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{h}}
= \sum_{j=1}^V (\hat{y}_j - y_j),\mathbf{u}*j
= V*{\text{out}},(\hat{\mathbf{y}} - \mathbf{y})
$$

---

#### （3）对输入向量 ( \mathbf{v}_{w_i} ) 的梯度：

因为
$$
\mathbf{h} = \frac{1}{C} \sum_{i=1}^C \mathbf{v}*{w_i}
$$
所以：
$$
\frac{\partial \mathcal{L}}{\partial \mathbf{v}*{w_i}}
= \frac{1}{C} \frac{\partial \mathcal{L}}{\partial \mathbf{h}}
= \frac{1}{C} \sum_{j=1}^V (\hat{y}_j - y_j),\mathbf{u}_j
$$

**矩阵形式：**
$$
\frac{\partial \mathcal{L}}{\partial V_{\text{in}}[:, w_i]}
= \frac{1}{C},V_{\text{out}},(\hat{\mathbf{y}} - \mathbf{y})
$$

---

### 10.5 参数更新（梯度下降）

使用学习率 ( \eta )，按标准 SGD 更新：

#### （a）更新输出矩阵：

$$
V_{\text{out}}[:, j]
\leftarrow
V_{\text{out}}[:, j] - \eta,(\hat{y}_j - y_j),\mathbf{h}
$$

#### （b）更新输入矩阵（上下文词）：

$$
V_{\text{in}}[:, w_i]
\leftarrow
V_{\text{in}}[:, w_i] -
\eta \cdot \frac{1}{C} \sum_{j=1}^V (\hat{y}_j - y_j),\mathbf{u}_j
$$

在实现时：

* 对 softmax 版本，( \sum_j ) 是全词表求和；
* 对 **负采样** 或 **层次 softmax**，这一步会替换为更小范围的求和。

---

### 10.6梯度方向解释

* 输出层梯度 ((\hat{y}_j - y_j)\mathbf{h})：
  使目标词 ( w_o ) 的概率增大，非目标词的概率降低。

* 输入层梯度：
  把上下文向量朝着能更好预测目标词的方向调整。

这使得**相似上下文中的词共享相似的隐藏向量方向**。

---

 ### 10.最终总结表

| 项目    | 符号                                                         | 表达式                                                                               | 含义            |
| ----- | ---------------------------------------------------------- | --------------------------------------------------------------------------------- | ------------- |
| 损失函数  | ( \mathcal{L} )                                            | (-\mathbf{u}_{w_o}^\top \mathbf{h} + \log\sum_j e^{\mathbf{u}_j^\top \mathbf{h}}) | softmax 负对数似然 |
| 输出层梯度 | ( \frac{\partial \mathcal{L}}{\partial \mathbf{u}_j} )     | ((\hat{y}_j - y_j)\mathbf{h})                                                     | 每个词的输出向量更新方向  |
| 隐层梯度  | ( \frac{\partial \mathcal{L}}{\partial \mathbf{h}} )       | ( \sum_j (\hat{y}_j - y_j)\mathbf{u}_j )                                          | 用于反传给输入层      |
| 输入层梯度 | ( \frac{\partial \mathcal{L}}{\partial \mathbf{v}_{w_i}} ) | ( \frac{1}{C}\sum_j (\hat{y}_j - y_j)\mathbf{u}_j )                               | 平均分配给每个上下文词   |
| 更新规则  |                                                            | ( \theta \leftarrow \theta - \eta\nabla_\theta \mathcal{L} )                      | 标准 SGD        |

---

