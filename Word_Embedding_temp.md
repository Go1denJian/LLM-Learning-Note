
# Word Embedding 全指南（GitHub README 风格）

## 目录
- [引言](#引言)
- [Word Embedding 是什么？](#word-embedding-是什么)
- [Word2Vec：CBOW 与 Skip-gram](#word2veccbow-与-skip-gram)
- [Negative Sampling（负采样）](#negative-sampling负采样)
- [共现矩阵 X（PMI / PPMI）](#共现矩阵-xpmi--ppmi)
- [SGNS ≈ PMI 理论](#sgns--pmi-理论)
- [Word2Vec / GloVe / SVD 模型关系图](#word2vec--glove--svd-模型关系图)
- [为什么要理解共现矩阵？](#为什么要理解共现矩阵)
- [代码示例（tiny corpus）](#代码示例tiny-corpus)
- [如何使用 text8 训练 Word2Vec](#如何使用-text8-训练-word2vec)
- [总结](#总结)

---

## 引言
Word Embedding 是自然语言处理（NLP）中将离散词语映射到连续向量空间的核心技术。  
经典方法 Word2Vec 通过预测任务学习语义结构，而传统基于计数的方法（如 SVD、GloVe）通过分解共现矩阵得到向量。

---

## Word Embedding 是什么？
目标函数可表示为：

$$
f: w \mapsto \mathbf{v}_w \in \mathbb{R}^d
$$

通常要求：
- 语义相似的词在向量空间中相近  
- 可支持线性类比（king - man + woman = queen）

---

## Word2Vec：CBOW 与 Skip-gram

### CBOW（Continuous Bag of Words）
使用上下文预测中心词  
隐藏层表示：

$$
\mathbf{h} = \frac{1}{2m} \sum_{j
eq 0} \mathbf{v}_{w_{t+j}}
$$

预测概率：

$$
P(w_t|context)=\frac{e^{u_{w_t}^T h}}{\sum_{w\in V} e^{u_w^T h}}
$$

---

### Skip-gram
使用中心词预测上下文词：

$$
P(w_{t+j}|w_t)=\frac{e^{u_{w_{t+j}}^T v_{w_t}}}{\sum_{w\in V} e^{u_w^T v_{w_t}}}
$$

损失：

$$
\mathcal{L}=-\sum_{j
eq 0}\log P(w_{t+j}|w_t)
$$

---

## Negative Sampling（负采样）

Word2Vec 的 softmax 成本过高，所以使用负采样代替多分类：

正样本：
$$
\sigma(u_{w_o}^T v_{w_c})
$$

负样本：
$$
\sigma(-u_{w_i}^T v_{w_c})
$$

SGNS 损失：

$$
\mathcal{L}=-\log\sigma(u_{w_o}^T v_{w_c}) - \sum_{i=1}^k \log \sigma(-u_{w_i}^T v_{w_c})
$$

---

## 共现矩阵 X（PMI / PPMI）

定义：

$$
X_{ij}=	\text{词 i 与词 j 在窗口中共现的次数}
$$

PMI：
$$
PMI(i,j)=\log \frac{P(i,j)}{P(i)P(j)}
$$

PPMI：

$$
PPMI(i,j)=\max(PMI(i,j),0)
$$

许多传统方法（如 SVD / GloVe）直接基于该矩阵构建词向量。

---

## SGNS ≈ PMI 理论

重要结论（Levy & Goldberg 2014）：

$$
v_w^T u_c pprox PMI(w,c) - \log k
$$

这意味着：
- **Word2Vec 隐式地在分解 PMI 矩阵**
- 预测式模型与计数式模型统一在同一框架下

---

## Word2Vec / GloVe / SVD 模型关系图

```
                ┌──────────────────────────────┐
                │       Word Embedding          │
                └──────────────────────────────┘
                               │
        ┌──────────────────────┴────────────────────────┐
        │                                               │
┌──────────────────┐                         ┌────────────────────┐
│ Count-based 方法  │                         │ Predict-based 方法 │
└──────────────────┘                         └────────────────────┘
        │                                               │
  SVD / LSA       GloVe                        CBOW      Skip-gram
        │                                               │
         └─────────────── 都在逼近 PMI 结构 ───────────────┘
```

---

## 为什么要理解共现矩阵？

虽然训练 Word2Vec 时**不需要显式构建**共现矩阵，但它在理论中非常重要：

- 解释 Word2Vec 为什么有效  
- 展示词向量几何结构来源（类比能力）  
- 理解 GloVe/SVD 的必要基础  
- 理解 SGNS 与 PMI 的关系

**总结：理解必要，构建不必要。**

---

## 代码示例（tiny corpus）

示例脚本可用于教学版本的 SGNS：

```
from word2vec_toy import train_on_toy_corpus
train_on_toy_corpus()
```

完整文件已在工具输出中提供（word2vec_toy.py）。

---

## 如何使用 text8 训练 Word2Vec

推荐使用 `gensim`：

```
from gensim.models import Word2Vec
sentences = LineSentence("text8")
model = Word2Vec(sentences, vector_size=200, window=5, min_count=5,
                 workers=4, sg=1, negative=10, epochs=5)
model.save("text8_word2vec.bin")
```

---

## 总结

- Word2Vec 是预测式模型，但本质逼近 PMI  
- Count-based 与 Predict-based 方法在数学上是统一的  
- 共现统计决定了词向量空间的语义结构  
- SGNS 是简单但极其有效的向量学习方式  

