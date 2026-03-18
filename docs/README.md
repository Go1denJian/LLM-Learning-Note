# 文档中心

LLM Learning Notes 的完整笔记索引与导航。

---

## 学习路径

建议按时间顺序阅读：

```
基础篇 → RNN → LSTM → Word2Vec → GloVe → GRU/Seq2Seq → Transformer → BERT → GPT-2 → ...
```

---

## 核心笔记

### 00：信息论基础

| 文档 | 说明 | 难度 | 前置知识 |
|------|------|:----:|----------|
| [Entropy-CrossEntropy-KL-Explained](./00-Entropy-CrossEntropy-KL-Explained.md) | 信息论基础、交叉熵损失的本质 | ⭐⭐ | 概率论基础 |

---

### 01：RNN (Elman, 1990)

| 文档 | 说明 | 难度 | 前置知识 |
|------|------|:----:|----------|
| [RNN-Fundamentals](./01-RNN-Fundamentals.md) | 循环结构、BPTT、梯度问题 | ⭐⭐⭐ | 基础篇 |

---

### 02：LSTM (Hochreiter, 1997)

| 文档 | 说明 | 难度 | 前置知识 |
|------|------|:----:|----------|
| [LSTM-Deep-Dive](./02-LSTM-Deep-Dive.md) | 三门机制、细胞状态、梯度流 | ⭐⭐⭐⭐ | RNN基础 |

---

### 03：Word2Vec (Mikolov, 2013)

| 文档 | 说明 | 难度 | 前置知识 |
|------|------|:----:|----------|
| [Word2Vec-Math-and-Implementation](./03-Word2Vec-Math-and-Implementation.md) | 从共现矩阵到词向量的完整推导 | ⭐⭐⭐ | 线性代数、梯度下降 |

---

### 04：GloVe (Pennington, 2014) — 待写

---

### 05：GRU (Cho, 2014)

| 文档 | 说明 | 难度 | 前置知识 |
|------|------|:----:|----------|
| [GRU-and-Seq2Seq](./05-GRU-and-Seq2Seq.md) | 简化门控、编码器-解码器架构 | ⭐⭐⭐⭐ | LSTM基础 |

---

### 06：Transformer (Vaswani, 2017)

| 文档 | 说明 | 难度 | 前置知识 |
|------|------|:----:|----------|
| [Transformer-Math-and-Implementation](./06-Transformer-Math-and-Implementation.md) | 自注意力机制完整实现 | ⭐⭐⭐⭐⭐ | RNN/LSTM基础 |

---

### 07–15：待实现

| 编号 | 论文 | 年份 | 状态 |
|------|------|------|------|
| 07 | BERT (Devlin, 2018) | 2018 | 待写 |
| 08 | GPT-2 (Radford, 2019) | 2019 | 待写 |
| 09 | T5 (Raffel, 2019) | 2019 | 待写 |
| 10 | GPT-3 (Brown, 2020) | 2020 | 待写 |
| 11 | Switch Transformers (Fedus, 2021) | 2021 | 待写 |
| 12 | LoRA (Hu, 2021) | 2021 | 待写 |
| 13 | InstructGPT/RLHF (Ouyang, 2022) | 2022 | 待写 |
| 14 | LLaMA (Touvron, 2023) | 2023 | 待写 |
| 15 | DeepSeek-R1 (DeepSeek, 2024) | 2024 | 待写 |

---

## 代码文档

### src/word2vec/

```python
from src.word2vec import Vocabulary, Word2VecSkipGram, train_word2vec_skipgram
```

### src/transformer/

```python
from src.transformer import Encoder
```

---

## 示例脚本

| 脚本 | 说明 |
|------|------|
| `examples/test_word2vec.py` | Word2Vec 组件测试 |
| `examples/test_transformer.py` | Transformer 组件测试 |
| `examples/train_word2vec.py` | Word2Vec 完整训练 + 可视化 |

---

## 外部资源

- [Stanford CS224N](https://web.stanford.edu/class/cs224n/)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

---

最后更新：2026-03-18
