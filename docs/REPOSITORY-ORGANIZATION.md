# 仓库组织指南

## 目录结构

```
LLM-Learning-Note/
│
├── 核心笔记
│   ├── Word-Embedding-Math-and-Implementation.md    # Word2Vec 完整笔记（推荐）
│   ├── Transformer-Math-and-Implementation.md       # Transformer 完整笔记（推荐）
│   │
│   └── 存档笔记（参考用）
│       ├── Word Embedding.md                        # Word2Vec 原始笔记
│       └── Transformer.md                           # Transformer 架构参考
│
├── 代码实现
│   ├── word2vec_implementation.py                   # Word2Vec 完整实现（Skip-gram + CBOW）
│   ├── transformer_implementation.py                # Transformer 完整实现（Encoder）
│   ├── transformer_numpy_demo.py                    # Transformer 纯 NumPy 验证
│   └── example_train_word2vec.py                    # Word2Vec 训练示例
│
├── 文档
│   ├── README.md                                    # 总览与学习路线（入口）
│   ├── README-Transformer.md                        # Transformer 运行指南
│   └── REPOSITORY-ORGANIZATION.md                   # 本文件（仓库组织）
│
├── 子目录（扩展内容）
│   ├── word2vec/                                    # Word2Vec 相关扩展
│   │   └── wordEmbeddingBasic.py                    # 旧版实现（参考）
│   │
│   └── GPT2/                                        # GPT-2 相关扩展
│       ├── step1_infer.py                           # GPT-2 推理
│       ├── step2_lora_train.py                      # LoRA 微调
│       └── step3_infer_lora.py                      # LoRA 推理
│
└── 生成的可视化（运行后产生）
    ├── word_embeddings_viz.png                      # 词向量可视化
    ├── training_loss.png                            # 训练损失曲线
    ├── positional_encoding_viz.png                  # 位置编码波形
    └── attention_weights_viz.png                    # 注意力权重热力图
```

---

## 学习建议

### 入门（第 1 周）
```
Day 1-2: Word Embedding 笔记 + word2vec_implementation.py
Day 3-4: 运行 example_train_word2vec.py
Day 5-7: Transformer 笔记第 1-4 节
```

### 进阶（第 2 周）
```
Day 1-3: Transformer 笔记第 5-8 节
Day 4-5: 运行 transformer_implementation.py
Day 6-7: 完成练习与思考题
```

---

## 文件说明

### 核心笔记

| 文件 | 内容 | 适合人群 |
|------|------|---------|
| Word-Embedding-Math-and-Implementation.md | 从共现矩阵到 Word2Vec | NLP 入门 |
| Transformer-Math-and-Implementation.md | 从线性代数到 Transformer | 深度学习进阶 |

### 代码文件

| 文件 | 功能 | 依赖 |
|------|------|------|
| word2vec_implementation.py | Word2Vec 训练 | PyTorch |
| transformer_implementation.py | Transformer Encoder | PyTorch |
| transformer_numpy_demo.py | 数学验证 | NumPy + Matplotlib |
| example_train_word2vec.py | 完整训练示例 | PyTorch |

---

## 使用建议

### 1. 阅读顺序
1. 先看 README.md 了解整体结构
2. 选择感兴趣的笔记开始学习
3. 边读边运行对应代码

### 2. 代码运行
```bash
# Word2Vec 测试
python word2vec_implementation.py

# Word2Vec 完整训练
python example_train_word2vec.py

# Transformer 测试
python transformer_implementation.py

# Transformer NumPy 验证（无需 PyTorch）
python transformer_numpy_demo.py
```

### 3. 扩展学习
- `word2vec/` 和 `GPT2/` 目录包含扩展内容
- 可在核心笔记完成后参考

---

## 生成的可视化文件

运行代码后会自动生成以下文件：

| 文件 | 说明 | 生成脚本 |
|------|------|---------|
| word_embeddings_viz.png | 词向量 t-SNE/PCA 可视化 | word2vec_implementation.py |
| training_loss.png | 训练损失曲线 | word2vec_implementation.py |
| positional_encoding_viz.png | 位置编码波形图 | transformer_numpy_demo.py |
| attention_weights_viz.png | 注意力权重热力图 | transformer_numpy_demo.py |

---

## 版本历史

| 日期 | 更新内容 |
|------|---------|
| 2026-03-11 | 完成 Word Embedding 和 Transformer 核心笔记 |
| 2026-03-11 | 添加完整代码实现和训练示例 |
| 2026-03-11 | 优化仓库结构和文档 |

---

## 注意事项

1. **存档笔记**：`Word Embedding.md` 和 `Transformer.md` 是原始笔记，内容较简略，建议以新笔记为主
2. **代码依赖**：确保安装 PyTorch、NumPy、Matplotlib、scikit-learn
3. **GPU 加速**：如果有 GPU，修改 `device='cuda'` 可加速训练

---

最后更新：2026-03-11
