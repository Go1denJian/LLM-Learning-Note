# LLM Learning Notes

从数学原理到代码实现的深度学习与大模型学习笔记。

> **核心理念**：理解公式背后的直觉，掌握从数学到代码的映射

---

## 学习路径（按论文时间顺序）

```
00 信息论基础 → 01 RNN (1990) → 02 LSTM (1997) → 03 Word2Vec (2013)
→ 04 GloVe (2014) → 05 GRU/Seq2Seq (2014) → 06 Transformer (2017)
→ 07 BERT (2018) → 08 GPT-2 (2019) → 09 T5 (2019) → 10 GPT-3 (2020)
→ 11 Switch Transformer (2021) → ... → 15 DeepSeek-R1 (2024)
```

---

## 核心内容

| 编号 | 论文 | 文档 | 难度 | 状态 |
|------|------|------|:----:|:----:|
| 00 | 信息论基础 | [Entropy-CrossEntropy-KL](docs/00-Entropy-CrossEntropy-KL-Explained.md) | ⭐⭐ | ✅ |
| 01 | RNN (Elman, 1990) | [RNN-Fundamentals](docs/01-RNN-Fundamentals.md) | ⭐⭐⭐ | ✅ |
| 02 | LSTM (Hochreiter, 1997) | [LSTM-Deep-Dive](docs/02-LSTM-Deep-Dive.md) | ⭐⭐⭐⭐ | ✅ |
| 03 | Word2Vec (Mikolov, 2013) | [Word2Vec-Math-and-Implementation](docs/03-Word2Vec-Math-and-Implementation.md) | ⭐⭐⭐ | ✅ |
| 04 | GloVe (Pennington, 2014) | [GloVe-Math-and-Implementation](docs/04-GloVe-Math-and-Implementation.md) | ⭐⭐⭐ | ✅ |
| 05 | GRU (Cho, 2014) | [GRU-and-Seq2Seq](docs/05-GRU-and-Seq2Seq.md) | ⭐⭐⭐⭐ | ✅ |
| 06 | Transformer (Vaswani, 2017) | [Transformer-Math-and-Implementation](docs/06-Transformer-Math-and-Implementation.md) | ⭐⭐⭐⭐⭐ | ✅ |
| 07 | BERT (Devlin, 2018) | [BERT-Math-and-Implementation](docs/07-BERT-Math-and-Implementation.md) | ⭐⭐⭐⭐⭐ | ✅ |
| 08 | GPT-2 (Radford, 2019) | [GPT2-Math-and-Implementation](docs/08-GPT2-Math-and-Implementation.md) | ⭐⭐⭐⭐⭐ | ✅ |
| 09 | T5 (Raffel, 2019) | [T5-Math-and-Implementation](docs/09-T5-Math-and-Implementation.md) | ⭐⭐⭐⭐ | ✅ |
| 10 | GPT-3 (Brown, 2020) | [GPT3-Scaling-and-InContext](docs/10-GPT3-Scaling-and-InContext.md) | ⭐⭐⭐⭐ | ✅ |
| 11 | Switch Transformers (Fedus, 2021) | [Switch-Transformer-MoE](docs/11-Switch-Transformer-MoE.md) | ⭐⭐⭐⭐⭐ | ✅ |
| 12 | LoRA (Hu, 2021) | — | ⭐⭐⭐ | 待写 |
| 13 | InstructGPT/RLHF (Ouyang, 2022) | — | ⭐⭐⭐⭐⭐ | 待写 |
| 14 | LLaMA (Touvron, 2023) | — | ⭐⭐⭐⭐ | 待写 |
| 15 | DeepSeek-R1 (DeepSeek, 2024) | — | ⭐⭐⭐⭐⭐ | 待写 |

**特色**：每篇包含完整数学推导 + NumPy 从零实现 + PyTorch 验证

---

## 项目结构

```
LLM-Learning-Note/
├── docs/                          # 学习笔记（扁平结构，编号排序）
├── src/                           # 源代码实现
│   ├── word2vec/                  # Word2Vec (Skip-gram + CBOW)
│   └── transformer/               # Transformer Encoder
├── examples/                      # 可运行示例
├── guides/                        # 学习指南
├── GPT2/                          # GPT-2 扩展示例
└── requirements.txt
```

---

## 快速开始

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python examples/test_word2vec.py
python examples/test_transformer.py
```

详细导航：[文档中心](docs/README.md)

---

## 数学公式速查

**RNN**: $h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$

**LSTM**: $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$

**Attention**: $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$

---

## 参考资源

- [Stanford CS224N](https://web.stanford.edu/class/cs224n/)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

---

最后更新：2026-03-18
