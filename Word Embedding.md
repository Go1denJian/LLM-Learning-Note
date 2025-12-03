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
f: w \mapsto \vec{v}_w \in \mathbf{R}^d
$$

使得 **语义相近的词向量距离更近**。

一般而言，最为简单的做法是给每一个词单独设置一个维度，把 $V$ 映射到一个高维的向量空间，最终得到的表示是一个 $|V|$ 维的 one-hot 向量。  
但是，这样的表示无法体现词与词之间的语义关系，同时计算过程中涉及到的矩阵规模是 $|V| \times |V|$，导致计算代价极高，并且矩阵大多是稀疏的。

Word2Vec 的设计目标是将词映射到一个低维、稠密的向量空间（embedding space）。  
在这种情况下，输入层到隐藏层、隐藏层到输出层的计算复杂度可以近似写为：

$$
Q = d \times d + d \times |V|
$$

- $d \times d$ ：输入到隐藏层的计算复杂度  
- $d \times |V|$ ：隐藏层到输出层的 softmax 计算复杂度  

其中， $d \times |V|$ 的规模往往远大于 $d \times d$ ，成为训练的主要瓶颈。  
例如，当 $d=300$， $|V|=100,000$ 时， $d \times d = 90,000$ ，而 $d \times |V| = 30,000,000$ 。可以看出，绝大部分计算开销来自输出层的 softmax。

因此，Word2Vec 原始论文提出了两种优化方法来降低复杂度：  
- **Hierarchical Softmax**：将复杂度降为 $O(d \times \log |V|)$  
- **Negative Sampling**：将复杂度降为 $O(d \times k)$（其中 $k \ll |V|$）
总之word embeding 可以极大的减少后续模型训练中的计算复杂度。

---

## 3. 两种主要算法

### 3.1 Continuous Bag of Words (CBOW)

CBOW的目标是使用上下文（context words）预测目标词（center word），记窗口为 $m$

* **输入**：上下文词 ${w_{t-m}, ..., w_{t-1}, w_{t+1}, ..., w_{t+m}}$
* **输出**：中心词 $w_t$


1. **对上下文单词做平均：**
   
$$
   \mathbf{h} = \frac{1}{2m}\sum_{-m \leq j \leq m, j \neq 0} \mathbf{v}_{w_{t+j}}
$$

1. **目标词预测概率分布**（Softmax 层）：

$$
   P(w_t \mid w_{t+j}, -m \leq j \leq m, j \neq 0 ) = \frac{\exp(\mathbf{u}_{w_t}^\top \mathbf{h})}{\sum_{w \in V} \exp(\mathbf{u}_w^\top \mathbf{h})}
$$

其中：

* $\mathbf{v}_w \in \mathbf{R}^d$ 为输入嵌入
* $\mathbf{u}_w \in \mathbf{R}^d$ 为输出嵌入

3. **损失函数**：最大化正确词的概率，最小化负对数似然函数

$$
\begin{equation} \begin{aligned}
\mathcal{L}_{CBOW} &= -\log P(w_t \mid  w_{t+j}, -m \leq j \leq m, j \neq 0 ) \\
   &= -\mathbf{u}_{w_t}^\top \mathbf{h} + \log(\sum_{w \in V} \exp(\mathbf{u}_w^\top \mathbf{h}))
\end{aligned} \end{equation}
   
$$

---

### 3.2 Skip-gram

**思路**：用中心词预测上下文。

* **输入**：中心词 $w_t$
* **输出**：上下文词 ${w_{t-m}, ..., w_{t-1}, w_{t+1}, ..., w_{t+m}}$

#### 数学表达

1. **预测概率分布**：

$$
   P(w_{t+j} \mid w_t) = \frac{\exp(\mathbf{u}_{w_{t+j}}^\top \mathbf{v}_{w_t})}{\sum_{w \in V} \exp(\mathbf{u}_w^\top \mathbf{v}_{w_t})}
$$

2. **损失函数**：最大化上下文词的概率

$$
   \mathcal{L}_{SkipGram} = - \sum_{-m \leq j \leq m, j \neq 0} \log P(w_{t+j} \mid w_t)
$$


> [!NOTE] Skip-gram损失函数的推导说明
> 需要说明的是，这里做了一个很重要的假设，即在已知 $w_t$的条件下，$w_i,w_j，i \neq j$的预测是互相独立的，这是建模分析中的一个近似。z这样我们就把问题从预测一个上下文（句子）转化为预测每个位置的词，从而概率写成乘积的形式，即：
> $$
> \begin{equation} \begin{aligned}
  P(w_{t+j}, -m \leq j \leq m, j \neq 0 \mid w_t) \approx \prod_{-m \leq j \leq m, j \neq 0}P(w_{t+j}|w_t)
\end{aligned} \end{equation}
> $$
> 

## 4. 最终输出

训练完成后，每个词 $w \in V$ 得到一个向量 $\mathbf{v}_w \in \mathbf{R}^d$。这些向量可以用于 计算词语相似度（cosine similarity）或 作为下游 NLP 任务的输入特征（分类、翻译等）

---


## 5. Negative Sampling （负采样）

### 背景

在 Word2Vec 的 CBOW 或 Skip-gram 中，预测目标词时需要对整个词表 $V$ 做 **softmax**，计算代价很大：

$$
P(w \mid \mathbf{h}) = \frac{\exp(\mathbf{u}_w^\top \mathbf{h})}{\sum_{w' \in V} \exp(\mathbf{u}_{w'}^\top \mathbf{h})}
$$

当词表规模达到百万级时，训练效率极低。

**Negative Sampling** 提供了一种近似方法：将多分类问题转化为 **二分类问题**。

---

### 5.1 思想

对于一个正样本对 $(w_t, w_{context})$，目标是区分它和若干负样本 $(w_t, w_{neg})$。

* **正样本**：真实出现的中心词–上下文对，记事件（标签） $D=1$
* **负样本**：随机采样的词–上下文对，记事件（标签） $D=0$

最终目标：最大化正样本的预测概率，最小化负样本的预测概率。

---

### 5.2 数学公式

1. **二分类预测函数**
   采用 sigmoid 函数：
   
$$
   P(D=1 \mid w_c, w_o) = \sigma(\mathbf{u}_{w_o}^\top \mathbf{v}_{w_c}) = \frac{1}{1 + \exp(-\mathbf{u}_{w_o}^\top \mathbf{v}_{w_c})}
$$

* $w_c$：中心词
* $w_o$：上下文词（或负样本词）
* $\mathbf{u}, \mathbf{v}$：输入/输出嵌入向量

> [!NOTE] Sigmoid函数
> 这里使用Sigmoid函数是一个常用手法，即我们需要构造一个这样的映射：
> $$
\begin{equation} \begin{aligned}
 f: \mathbb{R}^d \rightarrow \mathcal{B} = \{ \text{Bernoulli}(p) \mid p \in [0, 1] \}
\end{aligned} \end{equation}
> $$
> 而上述的 $\sigma(\mathbf{u}_{w_o}^\top\mathbf{v}_{w_c})$ 就给出了一个到 $p$ 的映射


1. **损失函数**
   Skip-gram with Negative Sampling (SGNS) 的损失函数：

$$
   \mathcal{L} = - \log \sigma(\mathbf{u}_{w_o}^\top \mathbf{v}_{w_c}) - \sum_{i=1}^k \log \sigma(-\mathbf{u}_{w_i}^\top \mathbf{v}_{w_c})
$$

其中：

* $w_o$ = 正样本（真实上下文词）
* $w_i$ = $k$ 个负样本（随机词）

目标是 第一个项：最大化正样本的相似度 与 第二个项：最小化负样本的相似度


> [!NOTE] Skip-gram with Negative Sampling (SGNS) 的损失函数
> 这里的损失函数构造实际上使用了sigmoid函数的特性：$\sigma(-x) = 1-\sigma(x)$



---

### 5.3 总结

* **优势**：
  * 不需要计算整个词表的 softmax，训练效率大幅提升
  * 常用 $k=5\sim20$ 的负采样即可
* **劣势**：
  * 损失函数不再是最大似然估计，而是近似方法
  * 对低频词效果较差

---

## 6.Word2Vec 三种训练方式对比

| 方法             | 输出目标           | 损失函数形式                                                                                                | 复杂度  | 
| -------------- | -------------- | ----------------------------------------------------------------------------------------------------- | ------ |
| CBOW           | 中心词            | $\mathcal{L} = -\log P(w_t \mid context)$                                                             | $O( \| V \| )$ |
| Skip-gram      | 上下文词集合         | $\mathcal{L} = - \sum \log P(w_{t+j} \mid w_t)$                                                       | $O( \| V \| )$ |
| Skip-gram + NS | 正样本 + $k$ 个负样本 | $\mathcal{L} = - \log \sigma(u_{w_o}^\top v_{w_c}) - \sum_{i=1}^k \log \sigma(-u_{w_i}^\top v_{w_c})$ | $O(k)$ | 

---


## 7.附录：CBOW 模型：损失函数与梯度更新公式推导

### 7.1 模型设定

* 词表大小： $|V|$ 
* 向量维度： $d$ 
* 输入（上下文词）： $w_{t+j}, -m \leq j \leq m, j \neq 0$
* 输出（目标词）： $w_o$ 

模型使用两套嵌入矩阵：

| 符号               | 说明              | 维度                     |
| ---------------- | --------------- | ---------------------- |
| $V_{\text{in}}$  | 输入嵌入矩阵（用于上下文词）  | $d \times \mid V \mid$ |
| $V_{\text{out}}$ | 输出嵌入矩阵（用于预测目标词） | $d \times \mid V \mid$ |

---

### 7.2 前向传播（Forward）

对上下文中每个词 ( $w_i$ )，取其输入向量（即每列代表一个词向量）：
$$
\mathbf{v}_{w_i} = V_{\text{in}}[:, w_i]
$$

求上下文的平均，得到隐藏层表示：
$$
\mathbf{h} = \frac{1}{2m-1} \sum_{-m \leq j \leq m, j \neq 0} \mathbf{v}_{w_j}
$$

计算每个词的预测得分（logit）：
$$
z_j = \mathbf{u}_j^\top \mathbf{h}, \quad \text{其中 } \mathbf{u}_j = V_{\text{out}}[:, j]
$$

计算 softmax 概率：
$$
\hat{y}_j = \frac{\exp(z_j)}{\sum_{k = 1}^{|V|} \exp(z_k)}
$$

---

### 7.3 损失函数（单样本）

CBOW 的目标是最大化预测正确中心词 ( $w_o$ ) 的概率，对应的负对数似然为（对于该函数需要最小化，以达到概率最大化的目的）：
$$
\mathcal{L} = -\log P(w_o|\text{context})
= -\mathbf{u}_{w_o}^\top \mathbf{h}

+ \log\sum_{w_j \in V} \exp(\mathbf{u}_j^\top \mathbf{h})
  $$

---

### 7.4 梯度求导

#### （1）对输出向量 ( $\mathbf{u}_j$ ) 的梯度：

softmax 输出：
$$
\hat{y}_j = \frac{\exp(\mathbf{u}_j^\top \mathbf{h})}{\sum_k \exp(\mathbf{u}_k^\top \mathbf{h})}
$$

记one-hot 目标：
$$
\begin{equation} \begin{aligned}
y_j =
\begin{cases}
1, & j = w_o \\
0, & \text{otherwise}
\end{cases}
\end{aligned} \end{equation}
$$

因此：
$$
\frac{\partial \mathcal{L}}{\partial \mathbf{u}_j}
= (\hat{y}_j - y_j)\mathbf{h}
$$

**矩阵形式：**
$$
\frac{\partial \mathcal{L}}{\partial V_{\text{out}}}
= \mathbf{h}(\hat{\mathbf{y}} - \mathbf{y})^\top
$$

---

#### （2）对隐藏层 ( $\mathbf{h}$ ) 的梯度：

$$
\begin{equation} \begin{aligned}
\frac{\partial \mathcal{L}}{\partial \mathbf{h}}
&= -\mathbf{u}_{w_o} + \frac{\sum_j\exp(\mathbf{u}_j^\top \mathbf{h})\mathbf{u}_j}{\sum_k \exp(\mathbf{u}_k^\top \mathbf{h})}\\
&= -\sum_j y_j\mathbf{u}_j + \sum_j \hat y_j\mathbf{u}_j\\
&= \sum_{j=1}^{|V|} (\hat{y}_j - y_j)\mathbf{u}_j \\
&= V_{\text{out}}(\hat{\mathbf{y}} - \mathbf{y})
\end{aligned} \end{equation}

$$

---

#### （3）对输入向量 ( $\mathbf{v}_{w_i}$ ) 的梯度：

因为
$$
\mathbf{h} = \frac{1}{C} \sum_{i=1}^C \mathbf{v}*{w_i}
$$
所以：
$$
\frac{\partial \mathcal{L}}{\partial \mathbf{v}_{w_i}}
= \frac{1}{C} \frac{\partial \mathcal{L}}{\partial \mathbf{h}}
= \frac{1}{C} \sum_{j=1}^V (\hat{y}_j - y_j)\mathbf{u}_j
$$

**矩阵形式：**
$$
\frac{\partial \mathcal{L}}{\partial V_{\text{in}}[:, w_i]}
= \frac{1}{C}V_{\text{out}}(\hat{\mathbf{y}} - \mathbf{y})
$$

---

### 7.5 参数更新（梯度下降）

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

* 对 softmax 版本，( $\sum_j$ ) 是全词表求和；
* 对 **负采样** 或 **层次 softmax**，这一步会替换为更小范围的求和。

---

### 7.6梯度方向解释

* 输出层梯度 (($\hat{y}_j - y_j)\mathbf{h}$)：
  使目标词 ( $w_o$ ) 的概率增大，非目标词的概率降低。

* 输入层梯度：
  把上下文向量朝着能更好预测目标词的方向调整。

这使得**相似上下文中的词共享相似的隐藏向量方向**。

---

### 7.7.最终总结表

| 项目    | 符号                                                           | 表达式                                                                                 | 含义            |
| ----- | ------------------------------------------------------------ | ----------------------------------------------------------------------------------- | ------------- |
| 损失函数  | ( $\mathcal{L}$ )                                            | ($-\mathbf{u}_{w_o}^\top \mathbf{h} + \log\sum_j e^{\mathbf{u}_j^\top \mathbf{h}}$) | softmax 负对数似然 |
| 输出层梯度 | ( $\frac{\partial \mathcal{L}}{\partial \mathbf{u}_j}$)      | ($(\hat{y}_j - y_j)\mathbf{h})$                                                     | 每个词的输出向量更新方向  |
| 隐层梯度  | ( $\frac{\partial \mathcal{L}}{\partial \mathbf{h}}$ )       | ( $\sum_j (\hat{y}_j - y_j)\mathbf{u}_j$ )                                          | 用于反传给输入层      |
| 输入层梯度 | ( $\frac{\partial \mathcal{L}}{\partial \mathbf{v}_{w_i}}$ ) | ( $\frac{1}{C}\sum_j (\hat{y}_j - y_j)\mathbf{u}_j$ )                               | 平均分配给每个上下文词   |
| 更新规则  | $θ={V_{\text{in}}​,V_{\text{out}}​}$                         | ( $\theta \leftarrow \theta - \eta\nabla_\theta \mathcal{L}$ )                      | 标准 SGD        |

---

