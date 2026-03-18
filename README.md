# LLM Learning Notes

从数学原理到代码实现的深度学习与大模型学习笔记。

> **核心理念**：理解公式背后的直觉，掌握从数学到代码的映射

---

## 学习路径

```
┌─────────────────────────────────────────────────────────────────┐
│  基础篇                                                          │
│  └── [熵与KL散度](docs/00-Foundation/)                           │
│      信息论基础、交叉熵损失的本质                                 │
├─────────────────────────────────────────────────────────────────┤
│  词嵌入篇                                                        │
│  └── [Word Embedding](docs/01-Word-Embedding/)                   │
│      从共现矩阵到 Word2Vec 的完整推导                            │
├─────────────────────────────────────────────────────────────────┤
│  序列模型篇                                                      │
│  ├── [RNN 基础](docs/02-RNN/) → 循环结构与梯度问题               │
│  ├── [LSTM 深入](docs/03-LSTM/) → 三门机制与长期依赖             │
│  └── [GRU 与 Seq2Seq](docs/04-GRU/) → 简化设计与编码器-解码器    │
├─────────────────────────────────────────────────────────────────┤
│  注意力机制篇                                                    │
│  └── [Transformer](docs/05-Transformer/)                         │
│      自注意力、多头注意力、位置编码完整实现                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 核心内容

### 基础篇

| 文档 | 说明 | 难度 |
|------|------|:----:|
| [熵与KL散度](docs/00-Entropy-CrossEntropy-KL-Explained.md) | 信息论基础、交叉熵损失的本质 | ⭐⭐ |

### 词嵌入篇

| 文档 | 说明 | 难度 |
|------|------|:----:|
| [Word Embedding](docs/01-Word-Embedding/Word-Embedding-Math-and-Implementation.md) | 从共现矩阵到词向量的完整推导 | ⭐⭐⭐ |

**内容**：共现矩阵、PMI、Word2Vec、负采样、梯度推导

### 序列模型篇

| 文档 | 说明 | 难度 |
|------|------|:----:|
| [RNN Fundamentals](docs/02-RNN/RNN-Fundamentals.md) | 循环结构、BPTT、梯度消失/爆炸 | ⭐⭐⭐ |
| [LSTM Deep Dive](docs/03-LSTM/LSTM-Deep-Dive.md) | 三门机制、细胞状态、梯度流 | ⭐⭐⭐⭐ |
| [GRU & Seq2Seq](docs/04-GRU/GRU-and-Seq2Seq.md) | 简化门控、编码器-解码器架构 | ⭐⭐⭐⭐ |

**特色**：每篇都包含完整的 NumPy 实现 + PyTorch 验证

### Transformer 篇

| 文档 | 说明 | 难度 |
|------|------|:----:|
| [Transformer](docs/05-Transformer/Transformer-Math-and-Implementation.md) | 自注意力机制完整实现 | ⭐⭐⭐⭐⭐ |

---

## 项目结构

```
LLM-Learning-Note/
├── docs/                          # 学习笔记（按学习路径组织）
│   ├── 00-Foundation/             # 基础：熵、KL散度
│   ├── 01-Word-Embedding/         # 词嵌入
│   ├── 02-RNN/                    # 循环神经网络
│   ├── 03-LSTM/                   # 长短期记忆网络
│   ├── 04-GRU/                    # 门控循环单元 + Seq2Seq
│   ├── 05-Transformer/            # Transformer
│   └── 06-Advanced/               # 进阶（预留）
│
├── src/                           # 源代码实现
│   ├── word2vec/                  # Word2Vec (Skip-gram + CBOW)
│   ├── transformer/               # Transformer Encoder
│   └── ...                        # RNN/LSTM/GRU 实现（待补充）
│
├── examples/                      # 可运行示例
├── guides/                        # 学习指南
├── tests/                         # 单元测试
└── assets/                        # 可视化输出
```

---

## 快速开始

### 1. 安装依赖

```bash
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行示例

```bash
# Word2Vec 测试
python examples/test_word2vec.py

# Transformer 测试
python examples/test_transformer.py
```

### 3. 阅读文档

- [文档中心](docs/README.md) - 完整笔记索引与导航

---

## 数学公式速查

### RNN
$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$

### LSTM
$$\begin{aligned}
f_t &= \sigma(W_{xf} x_t + W_{hf} h_{t-1} + b_f) \\
i_t &= \sigma(W_{xi} x_t + W_{hi} h_{t-1} + b_i) \\
C_t &= f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \\
h_t &= o_t \odot \tanh(C_t)
\end{aligned}$$

### Transformer Attention
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

---

## 参考资源

- [Stanford CS224N](https://web.stanford.edu/class/cs224n/) - NLP 经典课程
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

---

最后更新：2026-03-18
