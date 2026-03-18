# 文档中心

LLM Learning Notes 的完整笔记索引与导航。

---

## 学习路径

建议按以下顺序阅读：

```
基础篇 → 词嵌入篇 → 序列模型篇 → Transformer篇
```

---

## 核心笔记

### 00-Foundation：基础篇

| 文档 | 说明 | 难度 | 前置知识 |
|------|------|:----:|----------|
| [Entropy-CrossEntropy-KL-Explained](./00-Entropy-CrossEntropy-KL-Explained.md) | 信息论基础、交叉熵损失的本质 | ⭐⭐ | 概率论基础 |

**内容概览**：
- 信息量与熵的定义
- 交叉熵与KL散度的关系
- 为什么深度学习使用交叉熵损失

---

### 01-Word-Embedding：词嵌入篇

| 文档 | 说明 | 难度 | 前置知识 |
|------|------|:----:|----------|
| [Word-Embedding-Math-and-Implementation](./01-Word-Embedding/Word-Embedding-Math-and-Implementation.md) | 从共现矩阵到词向量的完整推导 | ⭐⭐⭐ | 线性代数、梯度下降 |

**内容概览**：
1. 引言：为什么需要词向量
2. 从 One-Hot 到 Dense Embedding
3. 共现矩阵与 PMI
4. Word2Vec 的两种架构（CBOW/Skip-gram）
5. 负采样的数学本质
6. 梯度推导与参数更新
7. 从数学到代码：完整实现
8. 实践技巧与可视化
9. 扩展阅读与实现
10. 参考资源
附录：符号表

---

### 02-RNN：循环神经网络篇

| 文档 | 说明 | 难度 | 前置知识 |
|------|------|:----:|----------|
| [RNN-Fundamentals](./02-RNN/RNN-Fundamentals.md) | 循环结构、BPTT、梯度问题 | ⭐⭐⭐ | Word Embedding |

**内容概览**：
1. 从MLP到RNN：为什么需要循环结构？
2. RNN的数学表达
3. BPTT：随时间反向传播
4. 梯度推导与参数更新
5. 梯度消失与训练优化
6. 从数学到代码：RNN完整实现
7. 与其他模型的关系
8. 扩展阅读与实现
9. 参考资源
附录：符号表

---

### 03-LSTM：长短期记忆网络篇

| 文档 | 说明 | 难度 | 前置知识 |
|------|------|:----:|----------|
| [LSTM-Deep-Dive](./03-LSTM/LSTM-Deep-Dive.md) | 三门机制、细胞状态、梯度流 | ⭐⭐⭐⭐ | RNN基础 |

**内容概览**：
1. 引言：为什么需要 LSTM？
2. LSTM 的数学表达
3. 核心算法：三门机制
4. 梯度推导与参数更新
5. 训练优化方法总结
6. 从数学到代码：完整实现
7. 实践技巧与可视化
8. 扩展阅读与实现
9. 参考资源
附录：符号表

---

### 04-GRU：门控循环单元篇

| 文档 | 说明 | 难度 | 前置知识 |
|------|------|:----:|----------|
| [GRU-and-Seq2Seq](./04-GRU/GRU-and-Seq2Seq.md) | 简化门控、编码器-解码器架构 | ⭐⭐⭐⭐ | LSTM基础 |

**内容概览**：
1. 引言：为什么需要 GRU？
2. GRU 的数学表达
3. 核心算法：门控机制
4. 梯度推导与参数更新
5. 训练优化方法总结
6. 从数学到代码：完整实现
7. Seq2Seq 架构
8. 注意力机制初步
9. 扩展阅读与实现
10. 参考资源
附录：符号表

---

### 05-Transformer：注意力机制篇

| 文档 | 说明 | 难度 | 前置知识 |
|------|------|:----:|----------|
| [Transformer-Math-and-Implementation](./05-Transformer-Math-and-Implementation.md) | 自注意力机制完整实现 | ⭐⭐⭐⭐⭐ | RNN/LSTM基础 |

**内容概览**：
1. 引言：为什么 Transformer 需要数学
2. 核心思想：注意力作为矩阵运算
3. Scaled Dot-Product Attention 的数学推导
4. Multi-Head Attention 的线性代数解释
5. 位置编码的傅里叶视角
6. 从数学到代码：完整实现
7. 实践中的关键技巧
8. 扩展阅读与实现
9. 参考资源
附录：符号表

---

## 代码文档

### src/word2vec/

```python
"""
Word2Vec 实现模块

包含:
- Vocabulary: 词表管理
- NegativeSampler: 负采样器
- Word2VecSkipGram: Skip-gram 模型
- Word2VecCBOW: CBOW 模型
- 训练函数
- 可视化函数
"""
```

**使用示例**:
```python
from src.word2vec import Vocabulary, Word2VecSkipGram, train_word2vec_skipgram

# 构建词表
vocab = Vocabulary(min_freq=10)
vocab.build(sentences)

# 创建模型
model = Word2VecSkipGram(vocab_size=len(vocab), embedding_dim=300)

# 训练
model, losses = train_word2vec_skipgram(model, pairs, vocab)
```

### src/transformer/

```python
"""
Transformer 实现模块

包含:
- ScaledDotProductAttention: 缩放点积注意力
- MultiHeadAttention: 多头注意力
- PositionwiseFeedForward: 位置前馈网络
- PositionalEncoding: 位置编码
- EncoderLayer: 编码器层
- Encoder: 完整编码器
"""
```

**使用示例**:
```python
from src.transformer import Encoder

# 创建编码器
encoder = Encoder(
    vocab_size=10000,
    d_model=512,
    num_heads=8,
    num_layers=6
)

# 前向传播
output = encoder(input_ids)
```

---

## 示例脚本

### examples/test_word2vec.py
测试 Word2Vec 各组件功能。

```bash
python examples/test_word2vec.py
```

### examples/test_transformer.py
测试 Transformer 各组件功能。

```bash
python examples/test_transformer.py
```

### examples/train_word2vec.py
完整训练 Word2Vec 模型并可视化。

```bash
python examples/train_word2vec.py
```

---

## 可视化资源

运行示例脚本后，生成的可视化文件保存在项目根目录：

| 文件 | 说明 | 生成脚本 |
|------|------|---------|
| `word_embeddings_viz.png` | 词向量 PCA/t-SNE 可视化 | `train_word2vec.py` |
| `training_loss.png` | 训练损失曲线 | `train_word2vec.py` |
| `positional_encoding_viz.png` | 位置编码波形图 | `numpy_demo.py` |
| `attention_weights_viz.png` | 注意力权重热力图 | `numpy_demo.py` |

---

## 常见问题

### Q: 需要什么数学基础？
**A**: 本科水平的线性代数（矩阵运算、特征值）、概率论（条件概率、分布）、微积分（梯度、链式法则）。

### Q: 代码需要什么依赖？
**A**: PyTorch、NumPy、Matplotlib、scikit-learn。运行 `pip install -r requirements.txt` 安装。

### Q: 如何验证学习效果？
**A**: 完成每章的练习与思考题，能够独立推导公式并修改代码。

### Q: 笔记中的章节编号为什么不统一？
**A**: 这是学习笔记而非教材，章节结构根据内容需要灵活调整。每篇笔记内部结构完整，可以独立阅读。

---

## 外部资源

### 论文
- [Word2Vec (Mikolov et al., 2013)](https://arxiv.org/abs/1301.3781)
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [Neural Word Embedding as Implicit Matrix Factorization](https://arxiv.org/abs/1402.3722)

### 教程
- [The Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

### 课程
- [Stanford CS224N](https://web.stanford.edu/class/cs224n/)

---

最后更新：2026-03-18
6-03-18
