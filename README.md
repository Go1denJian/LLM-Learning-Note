# LLM Learning Notes

从数学原理到代码实现的深度学习与大模型学习笔记。

> 核心理念：理解公式背后的直觉，掌握从数学到代码的映射

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

# 完整训练
python examples/train_word2vec.py
```

### 3. 阅读文档

- [文档中心](docs/README.md) - 完整笔记索引
- [Word Embedding](docs/Word-Embedding-Math-and-Implementation.md)
- [Transformer](docs/Transformer-Math-and-Implementation.md)

---

## 项目结构

```
LLM-Learning-Note/
├── docs/                      # 学习笔记
│   ├── README.md
│   ├── Word-Embedding-Math-and-Implementation.md
│   ├── Transformer-Math-and-Implementation.md
│   └── ...
│
├── src/                       # 源代码
│   ├── word2vec/             # Word2Vec 实现
│   │   └── __init__.py
│   └── transformer/          # Transformer 实现
│       ├── __init__.py
│       └── numpy_demo.py
│
├── examples/                  # 示例脚本
│   ├── test_word2vec.py
│   ├── test_transformer.py
│   └── train_word2vec.py
│
├── assets/                    # 生成的资源
│   └── (可视化图片)
│
├── archive/                   # 归档内容
│
├── requirements.txt           # Python 依赖
├── .gitignore
└── README.md                  # 本文件
```

---

## 核心内容

### Word Embedding

| 资源 | 说明 |
|------|------|
| [笔记](docs/Word-Embedding-Math-and-Implementation.md) | 从共现矩阵到 Word2Vec |
| [源码](src/word2vec/__init__.py) | Skip-gram + CBOW 实现 |
| [示例](examples/train_word2vec.py) | 完整训练流程 |

数学内容：共现矩阵、PMI、负采样、梯度推导

### Transformer

| 资源 | 说明 |
|------|------|
| [笔记](docs/Transformer-Math-and-Implementation.md) | 从线性代数到 Transformer |
| [源码](src/transformer/__init__.py) | Encoder 完整实现 |
| [NumPy 验证](src/transformer/numpy_demo.py) | 纯 NumPy 数学验证 |

数学内容：注意力机制、多头注意力、位置编码

---

## 数学公式速查

### Word Embedding

**Skip-gram with Negative Sampling**:

$$
\mathcal{L} = -\log \sigma(\mathbf{u}_{w_o}^\top \mathbf{v}_{w_c}) - \sum_{i=1}^k \log \sigma(-\mathbf{u}_{w_i}^\top \mathbf{v}_{w_c})
$$

### Transformer

**Scaled Dot-Product Attention**:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

---

## 参考资源

- [Stanford CS224N](https://web.stanford.edu/class/cs224n/) - NLP 课程
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - 可视化教程
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer 论文

---

## 许可证

MIT License

---

最后更新：2026-03-11
