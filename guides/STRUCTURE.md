# 目录结构说明

---

## 完整目录树

```
LLM-Learning-Note/
├── docs/                                # 学习笔记（扁平，编号排序）
│   ├── README.md                          # 文档中心入口
│   ├── 00-Entropy-CrossEntropy-KL-Explained.md
│   ├── 01-RNN-Fundamentals.md
│   ├── 02-LSTM-Deep-Dive.md
│   ├── 03-Word2Vec-Math-and-Implementation.md
│   ├── 04-GloVe-*.md                     # 待写
│   ├── 05-GRU-and-Seq2Seq.md
│   ├── 06-Transformer-Math-and-Implementation.md
│   └── 07–15: BERT → DeepSeek-R1         # 待写
│
├── src/                                 # 源代码
│   ├── word2vec/__init__.py
│   └── transformer/__init__.py, numpy_demo.py
│
├── examples/                            # 示例脚本
│   ├── test_word2vec.py
│   ├── test_transformer.py
│   └── train_word2vec.py
│
├── guides/                              # 学习指南
│   ├── QUICKSTART.md
│   └── STRUCTURE.md
│
├── GPT2/                                # GPT-2 扩展示例
└── requirements.txt
```

---

## 设计理念

- `docs/` — 只放笔记，扁平结构，编号即顺序
- `src/` — 只放代码模块
- `examples/` — 只放可运行脚��
- `guides/` — 只放学习指南

---

## 命名规范

- **文档**：`NN-Pascal-Case.md`（如 `03-Word2Vec-Math-and-Implementation.md`）
- **代码**：`snake_case.py`
- **指南**：`UPPER-CASE.md`

---

最后更新：2026-03-18
