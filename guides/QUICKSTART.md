# 快速入门指南

5 分钟快速开始 LLM Learning Notes 的学习之旅。

---

## 1. 安装（2 分钟）

```bash
# 克隆或进入项目目录
cd LLM-Learning-Note

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

**验证安装**：
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}')"
```

---

## 2. 运行测试（1 分钟）

### Word2Vec 测试
```bash
python examples/test_word2vec.py
```

**预期输出**：
```
============================================================
Word2Vec 组件测试套件
============================================================
测试 1: 词表构建
✓ 词表大小：xxx
...
测试结果：6 通过，0 失败
```

### Transformer 测试
```bash
python examples/test_transformer.py
```

**预期输出**：
```
============================================================
Transformer 组件测试套件
============================================================
测试 1: Scaled Dot-Product Attention
✓ 输入 Q 形状：torch.Size([2, 10, 64])
...
测试结果：7 通过，0 失败
```

---

## 3. 开始学习（2 分钟）

### 选项 A：Word Embedding（推荐入门）

1. **阅读教案**
   ```bash
   # 在线查看或下载 PDF
   cat docs/Word-Embedding-Math-and-Implementation.md
   ```

2. **运行训练示例**
   ```bash
   python examples/train_word2vec.py
   ```

3. **查看生成的可视化**
   - `word_embeddings_viz.png` - 词向量分布
   - `training_loss.png` - 训练曲线

### 选项 B：Transformer（进阶）

1. **阅读教案**
   ```bash
   cat docs/Transformer-Math-and-Implementation.md
   ```

2. **运行 NumPy 验证（无需 PyTorch）**
   ```bash
   python src/transformer/numpy_demo.py
   ```

3. **查看生成的可视化**
   - `positional_encoding_viz.png`
   - `attention_weights_viz.png`

---

## 下一步

### 第 1 周：Word Embedding
- 完成教案第 1-5 节阅读
- 推导负采样梯度公式
- 运行训练示例并调整超参数

### 第 2 周：Transformer
- 完成教案第 1-6 节阅读
- 理解多头注意力的子空间投影
- 修改 Encoder 层数并观察效果

### 第 3 周：综合实践
- 用自己的语料训练词向量
- 可视化分析词向量质量
- 尝试实现完整的 Transformer

---

## 遇到问题？

### 常见错误

**错误 1**: `ModuleNotFoundError: No module named 'torch'`
```bash
# 解决：确保虚拟环境已激活
source venv/bin/activate
pip install -r requirements.txt
```

**错误 2**: 测试失败
```bash
# 解决：检查 Python 版本（需要 3.8+）
python --version
```

**错误 3**: 内存不足
```bash
# 解决：减小 batch_size 或 embedding_dim
# 编辑 examples/train_word2vec.py
CONFIG = {
    'batch_size': 64,      # 改为 32
    'embedding_dim': 100,  # 改为 50
}
```

---

## 资源链接

- [完整文档](./docs/README.md)
- [学习路线](./docs/README.md#学习指南)
- [代码文档](./src/README.md)（待创建）

---

祝你学习愉快！

如有问题，欢迎查阅教案或运行测试脚本验证环境。
