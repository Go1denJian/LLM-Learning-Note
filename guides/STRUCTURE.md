# 目录结构说明

本目录包含项目的辅助指南和说明文档。

---

## 文件说明

| 文件 | 说明 |
|------|------|
| [QUICKSTART.md](./QUICKSTART.md) | 5 分钟快速入门指南 |
| [STRUCTURE.md](./STRUCTURE.md) | 本文件（目录结构说明） |

---

## 完整目录树

```
LLM-Learning-Note/
│
├── 核心文档
│   ├── README.md                           # 项目总览（入口）
│   ├── requirements.txt                    # Python 依赖
│   └── .gitignore                          # Git 忽略规则
│
├── docs/                                # 学习笔记（扁平结构）
│   ├── README.md                          # 文档中心入口
│   ├── 00-Entropy-CrossEntropy-KL-Explained.md
│   ├── 01-Word-Embedding-Math-and-Implementation.md
│   ├── 02-RNN-Fundamentals.md
│   ├── 03-LSTM-Deep-Dive.md
│   ├── 04-GRU-and-Seq2Seq.md
│   └── 05-Transformer-Math-and-Implementation.md
│
├── src/                                 # 源代码
│   ├── word2vec/
│   │   └── __init__.py                    # Word2Vec 完整实现
│   └── transformer/
│       ├── __init__.py                    # Transformer 完整实现
│       └── numpy_demo.py                  # NumPy 数学验证
│
├── examples/                            # 示例脚本
│   ├── test_word2vec.py                   # Word2Vec 组件测试
│   ├── test_transformer.py                # Transformer 组件测试
│   └── train_word2vec.py                  # Word2Vec 训练示例
│
├── guides/                              # 学习指南
│   ├── QUICKSTART.md                      # 快速入门（5 分钟）
│   └── STRUCTURE.md                       # 目录结构说明
│
├── assets/                              # 资源文件（待添加）
│
├── images/                              # 生成的可视化图片
│   ├── word_embeddings_viz.png            # 词向量可视化
│   ├── training_loss.png                  # 训练损失曲线
│   ├── positional_encoding_viz.png        # 位置编码波形
│   └── attention_weights_viz.png          # 注意力权重热力图
│
├── archive/                             # 归档内容
│   ├── Word Embedding.md                  # 旧版 Word2Vec 笔记
│   ├── Transformer.md                     # 旧版 Transformer 笔记
│   └── wordEmbeddingBasic.py              # 旧版实现
│
└── GPT2/                               # GPT-2 扩展示例（独立）
    ├── step1_infer.py
    ├── step2_lora_train.py
    └── step3_infer_lora.py
```

---

## 设计理念

### 1. 分离关注点

- `docs/` - 只放笔记（Markdown）
- `src/` - 只放代码（Python 模块）
- `examples/` - 只放可运行脚本
- `guides/` - 只放学习指南

### 2. 渐进式学习

```
guides/QUICKSTART.md     → 5 分钟入门
    ↓
docs/README.md           → 完整学习路线
    ↓
docs/*.md                → 核心笔记
    ↓
src/                     → 代码实现
    ↓
examples/                → 实践练习
```

### 3. 可运行优先

所有代码示例都应该：
- 可直接运行
- 有清晰的输入输出
- 包含测试验证

---

## 使用规范

### 添加新内容

**新笔记** → 放入 `docs/`
```bash
docs/My-New-Topic-Math-and-Implementation.md
```

**新代码模块** → 放入 `src/`
```bash
src/my_module/__init__.py
```

**新示例** → 放入 `examples/`
```bash
examples/run_my_module.py
```

**新指南** → 放入 `guides/`
```bash
guides/MY-GUIDE.md
```

### 命名规范

- **文档**：`Pascal-Case.md`（如 `Word-Embedding-Math-and-Implementation.md`）
- **代码**：`snake_case.py`（如 `word2vec_implementation.py`）
- **指南**：`UPPER-CASE.md`（如 `QUICKSTART.md`）

---

## 归档策略

以下内容应移入 `archive/`：

- 旧版笔记（被新笔记替代）
- 过时的代码实现
- 历史版本文件

归档文件仅供参考，不再维护。

---

最后更新：2026-03-11
