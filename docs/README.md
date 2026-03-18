# 文档中心

LLM Learning Notes 的完整笔记索引与导航。

---

## 学习路径

建议按时间顺序阅读：

```
基础篇 → RNN → LSTM → Word2Vec → GloVe → GRU/Seq2Seq → Transformer → BERT → GPT-2 → T5 → ...
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

### 04：GloVe (Pennington, 2014)

| 文档 | 说明 | 难度 | 前置知识 |
|------|------|:----:|----------|
| [GloVe-Math-and-Implementation](./04-GloVe-Math-and-Implementation.md) | 共现概率比值、加权最小二乘目标 | ⭐⭐⭐ | Word2Vec |

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

### 07：BERT (Devlin, 2018)

| 文档 | 说明 | 难度 | 前置知识 |
|------|------|:----:|----------|
| [BERT-Math-and-Implementation](./07-BERT-Math-and-Implementation.md) | 双向编码、MLM、NSP预训练 | ⭐⭐⭐⭐⭐ | Transformer基础 |

---

### 08：GPT-2 (Radford, 2019)

| 文档 | 说明 | 难度 | 前置知识 |
|------|------|:----:|----------|
| [GPT2-Math-and-Implementation](./08-GPT2-Math-and-Implementation.md) | 因果Attention、Zero-shot生成、采样策略 | ⭐⭐⭐⭐⭐ | BERT/Transformer基础 |

---

### 09：T5 (Raffel, 2019)

| 文档 | 说明 | 难度 | 前置知识 |
|------|------|:----:|----------|
| [T5-Math-and-Implementation](./09-T5-Math-and-Implementation.md) | 统一Text-to-Text框架、Span Corruption、Encoder-Decoder | ⭐⭐⭐⭐ | BERT/GPT-2基础 |

---

### 10：GPT-3 (Brown, 2020)

| 文档 | 说明 | 难度 | 前置知识 |
|------|------|:----:|----------|
| [GPT3-Scaling-and-InContext](./10-GPT3-Scaling-and-InContext.md) | Scaling Laws、In-context Learning、稀疏注意力、大规模训练 | ⭐⭐⭐⭐ | GPT-2/T5基础 |

---

### 11：Switch Transformer (Fedus, 2021)

| 文档 | 说明 | 难度 | 前置知识 |
|------|------|:----:|----------|
| [Switch-Transformer-MoE](./11-Switch-Transformer-MoE.md) | MoE稀疏激活、Top-1路由、负载均衡损失、万亿参数训练 | ⭐⭐⭐⭐⭐ | Transformer/GPT-3基础 |

---

### 12：LoRA (Hu, 2021)

| 文档 | 说明 | 难度 | 前置知识 |
|------|------|:----:|----------|
| [LoRA-Math-and-Implementation](./12-LoRA-Math-and-Implementation.md) | 低秩适应、参数高效微调、权重合并推理 | ⭐⭐⭐⭐ | Transformer/GPT-3基础 |

---

### 13：InstructGPT/RLHF (Ouyang, 2022)

| 文档 | 说明 | 难度 | 前置知识 |
|------|------|:----:|----------|
| [RLHF-Math-and-Implementation](./13-RLHF-Math-and-Implementation.md) | RLHF三阶段训练、PPO算法、奖励模型、KL约束 | ⭐⭐⭐⭐⭐ | GPT-3/LoRA基础 |

---

### 14：LLaMA (Touvron, 2023)

| 文档 | 说明 | 难度 | 前置知识 |
|------|------|:----:|----------|
| [LLaMA-Architecture-and-Implementation](./14-LLaMA-Architecture-and-Implementation.md) | RMSNorm、SwiGLU激活、RoPE旋转位置编码、高效训练 | ⭐⭐⭐⭐ | Transformer/RLHF基础 |

---

### 15：待实现

| 编号 | 论文 | 年份 | 状态 |
|------|------|------|------|
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

最后更新：2026-03-19
