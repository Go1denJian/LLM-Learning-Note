# 文档中心

LLM Learning Notes 的完整文档与学习指南。

---

## 核心教案

### Word Embedding

| 文档 | 说明 | 难度 |
|------|------|------|
| [Word-Embedding-Math-and-Implementation.md](./Word-Embedding-Math-and-Implementation.md) | 从共现矩阵到词向量的完整推导 | 入门 |

**内容概览**：
1. 引言：为什么需要词向量
2. 从 One-Hot 到 Dense Embedding
3. 共现矩阵与 PMI
4. Word2Vec 的两种架构（CBOW/Skip-gram）
5. 负采样的数学本质
6. 梯度推导与参数更新
7. 从数学到代码：完整实现
8. 实践技巧与可视化
9. 练习与思考题

**前置知识**：线性代数、概率论、梯度下降

---

### Transformer

| 文档 | 说明 | 难度 |
|------|------|------|
| [Transformer-Math-and-Implementation.md](./Transformer-Math-and-Implementation.md) | 从线性代数到 Transformer 编码器 | 进阶 |

**内容概览**：
1. 引言：为什么 Transformer 需要数学
2. 核心思想：注意力作为矩阵运算
3. Scaled Dot-Product Attention 的数学推导
4. Multi-Head Attention 的线性代数解释
5. 位置编码的傅里叶视角
6. 从数学到代码：完整实现
7. 实践中的关键技巧
8. 练习与思考题

**前置知识**：矩阵运算、softmax 函数、Python 基础

---

## 学习指南

### 学习路线

```
Week 1: Word Embedding          Week 2: Transformer            Week 3: 综合实践
    ↓                                ↓                                ↓
教案阅读 → 代码验证 → 练习       教案阅读 → 代码验证 → 练习       训练模型 → 可视化分析
```

### 详细计划

#### 第 1 周：Word Embedding 基础

| 天数 | 任务 | 产出 |
|------|------|------|
| Day 1-2 | 阅读教案第 1-5 节 | 理解 PMI 和负采样 |
| Day 3 | 阅读教案第 6-7 节 | 理解梯度推导 |
| Day 4 | 运行 `examples/test_word2vec.py` | 代码验证 |
| Day 5 | 完成课后练习 | 巩固知识 |
| Day 6-7 | 扩展阅读 | PMI 矩阵分解论文 |

#### 第 2 周：Transformer 核心

| 天数 | 任务 | 产出 |
|------|------|------|
| Day 1-2 | 阅读教案第 1-4 节 | 理解注意力机制 |
| Day 3 | 阅读教案第 5-6 节 | 理解位置编码 |
| Day 4 | 运行 `examples/test_transformer.py` | 代码验证 |
| Day 5 | 运行 `src/transformer/numpy_demo.py` | 数学验证 |
| Day 6-7 | 完成课后练习 | 巩固知识 |

#### 第 3 周：综合实践

| 天数 | 任务 | 产出 |
|------|------|------|
| Day 1-2 | 运行 `examples/train_word2vec.py` | 训练词向量 |
| Day 3 | 可视化分析 | 词向量图 |
| Day 4-5 | 尝试修改超参数 | 对比实验 |
| Day 6-7 | 总结与复习 | 学习笔记 |

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

## 常见问题

### Q: 需要什么数学基础？
**A**: 本科水平的线性代数（矩阵运算、特征值）、概率论（条件概率、分布）、微积分（梯度、链式法则）。

### Q: 代码需要什么依赖？
**A**: PyTorch、NumPy、Matplotlib、scikit-learn。运行 `pip install -r requirements.txt` 安装。

### Q: 如何验证学习效果？
**A**: 完成每章的练习与思考题，能够独立推导公式并修改代码。

---

最后更新：2026-03-11
作者：OpenClaw Engineer
