# GPT-3 数学原理与实现 —— 大规模语言模型与 In-context Learning 的完整推导

> **前置知识**：Transformer Decoder、自回归语言模型、交叉熵损失、Python 基础  
> **与前面内容的联系**：建议先学习 [GPT2-Math-and-Implementation](./08-GPT2-Math-and-Implementation.md) 和 [T5-Math-and-Implementation](./09-T5-Math-and-Implementation.md)，理解自回归生成与统一框架思想  
> **与后续内容的联系**：GPT-3 的 In-context Learning 和 Scaling Laws 直接影响了后续 InstructGPT/RLHF 和 LLaMA 等大模型的设计

---

## 目录

1. [引言：为什么需要更大的语言模型？](#1-引言为什么需要更大的语言模型)
   - 1.1 [预训练-微调范式的局限](#11-预训练-微调范式的局限)
   - 1.2 [In-context Learning 的核心洞察](#12-in-context-learning-的核心洞察)
   - 1.3 [本科数学知识映射表](#13-本科数学知识映射表)
2. [Scaling Laws：规模的数学规律](#2-scaling-laws规模的数学规律)
   - 2.1 [幂律关系的基本形式](#21-幂律关系的基本形式)
   - 2.2 [参数、数据与计算的三维权衡](#22-参数数据与计算的三维权衡)
   - 2.3 [计算最优分配（Chinchilla 视角）](#23-计算最优分配chinchilla-视角)
3. [GPT-3 架构的数学描述](#3-gpt-3-架构的数学描述)
   - 3.1 [Decoder-only 架构总览](#31-decoder-only-架构总览)
   - 3.2 [稀疏注意力模式：Dense 与 Locally Banded 交替](#32-稀疏注意力模式dense-与-locally-banded-交替)
   - 3.3 [GPT-3 各规模配置与参数估算](#33-gpt-3-各规模配置与参数估算)
4. [In-context Learning 数学分析](#4-in-context-learning-数学分析)
   - 4.1 [Zero-shot / One-shot / Few-shot 形式化定义](#41-zero-shot--one-shot--few-shot-形式化定义)
   - 4.2 [In-context Learning 的条件概率视角](#42-in-context-learning-的条件概率视角)
   - 4.3 [为什么 In-context Learning 有效？——隐式贝叶斯推断](#43-为什么-in-context-learning-有效隐式贝叶斯推断)
   - 4.4 [示例数量与性能的关系](#44-示例数量与性能的关系)
5. [大规模训练优化方法总结](#5-大规模训练优化方法总结)
   - 5.1 [梯度累积](#51-梯度累积)
   - 5.2 [混合精度训练（FP16/BF16 + FP32）](#52-混合精度训练fp16bf16--fp32)
   - 5.3 [模型并行训练](#53-模型并行训练)
   - 5.4 [学习率调度与超参数](#54-学习率调度与超参数)
6. [从数学到代码：完整实现](#6-从数学到代码完整实现)
   - 6.1 [NumPy 实现核心组件](#61-numpy-实现核心组件)
   - 6.2 [PyTorch 完整实现](#62-pytorch-完整实现)
7. [Scaling Laws 可视化与实践技巧](#7-scaling-laws-可视化与实践技巧)
   - 7.1 [Scaling Laws 可视化](#71-scaling-laws-可视化)
   - 7.2 [实践调参建议](#72-实践调参建议)
8. [与其他模型的关系](#8-与其他模型的关系)
   - 8.1 [GPT-2 vs GPT-3：从 Zero-shot 到 Few-shot](#81-gpt-2-vs-gpt-3从-zero-shot-到-few-shot)
   - 8.2 [GPT-3 在大模型发展中的定位](#82-gpt-3-在大模型发展中的定位)
   - 8.3 [GPT-3 的后续发展](#83-gpt-3-的后续发展)

[扩展阅读与实现](#扩展阅读与实现)

[参考资源](#参考资源)

附录：[符号表](#附录符号表)

---

## 1. 引言：为什么需要更大的语言模型？

### 1.1 预训练-微调范式的局限

在 GPT-3 之前，NLP 的主流范式是**预训练 + 微调**（Pre-train + Fine-tune）：

| 步骤 | 操作 | 代表模型 |
|------|------|---------|
| 预训练 | 在大规模无标注语料上学习通用语言表示 | BERT, GPT-2, T5 |
| 微调 | 在特定任务的标注数据上调整全部参数 | BERT-QA, GPT-2-Summarize |

这一范式存在三个根本问题：

1. **数据依赖**：每个下游任务仍需大量标注数据
2. **泛化有限**：微调后的模型在分布外数据上表现下降
3. **不灵活**：无法动态切换任务，每个任务需要独立部署

### 1.2 In-context Learning 的核心洞察

GPT-3 提出了一种**无需梯度更新**的新范式——**In-context Learning**：

> **不修改模型参数，仅通过在输入中提供任务描述和少量示例，让模型"理解"并执行新任务。**

$$
\boxed{P_\theta(y \mid \text{task description}, \text{examples}, x) \quad \text{（参数 } \theta \text{ 固定不变）}}
$$

**三种学习方式**：

```
Zero-shot:  "Translate English to French: cheese →"         → "fromage"
One-shot:   "sea otter → loutre de mer, cheese →"           → "fromage"
Few-shot:   "sea otter → loutre de mer, peppermint → menthe poivrée, cheese →"  → "fromage"
```

**关键发现**：当模型足够大时（175B 参数），Few-shot 性能可以**逼近甚至超过**经过微调的小模型。

### 1.3 本科数学知识映射表

| 数学概念 | GPT-3 中的应用 | 代码对应 |
|---------|--------------|---------|
| 幂律 $y = ax^{-\alpha}$ | Scaling Laws | `loss = a * N**(-alpha)` |
| 条件概率 $P(y \mid x, \text{ctx})$ | In-context Learning | `model(context + query)` |
| 贝叶斯推断 | ICL 的隐式机制 | 注意力权重的动态选择 |
| 浮点数表示 (IEEE 754) | 混合精度训练 | `torch.float16` / `torch.bfloat16` |
| 矩阵分块乘法 | 模型并行 | 张量按列/行切分 |
| 梯度期望 $\mathbb{E}[\nabla \mathcal{L}]$ | 梯度累积 | 多步累加后更新 |
| 带状矩阵 | 稀疏注意力 | Locally banded attention |

---

## 2. Scaling Laws：规模的数学规律

### 2.1 幂律关系的基本形式

GPT-3 的核心理论基础是 **Scaling Laws**（Kaplan et al., 2020），它揭示了语言模型性能与规模之间的**幂律关系**：

$$
\boxed{\mathcal{L}(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad \mathcal{L}(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad \mathcal{L}(C) = \left(\frac{C_c}{C}\right)^{\alpha_C}}
$$

其中 $\alpha_N \approx 0.076$，$\alpha_D \approx 0.095$，$\alpha_C \approx 0.050$。

**对数空间的线性关系**：

$$
\log \mathcal{L}(N) = -\alpha_N \log N + \alpha_N \log N_c
$$

在双对数坐标中，损失与参数量呈**直线关系**——这是 Scaling Laws 最重要的实验观察。

### 2.2 参数、数据与计算的三维权衡

当三个变量同时变化时，损失可以用**联合幂律**描述：

$$
\boxed{\mathcal{L}(N, D) = \left[\left(\frac{N_c}{N}\right)^{\alpha_N / \alpha_D} + \frac{D_c}{D}\right]^{\alpha_D}}
$$

物理含义：$N$ 和 $D$ 需要同步增长，任何一方不足都会成为瓶颈。

**计算量约束**：对于 Transformer，总计算量为：

$$
\boxed{C \approx 6ND \quad \text{(FLOPs)}}
$$

其中因子 6 来自前向传播 $2ND$ 加反向传播 $4ND$。

Kaplan et al. (2020) 的最优分配：$N^* \propto C^{0.73}$，$D^* \propto C^{0.27}$。

### 2.3 计算最优分配（Chinchilla 视角）

Chinchilla 论文修正了 Kaplan 的分配策略：

$$
\boxed{N^* \propto C^{0.50}, \quad D^* \propto C^{0.50}, \quad D^* \approx 20N}
$$

| 模型 | 参数量 $N$ | 训练 Token $D$ | $D/N$ | Chinchilla 最优？ |
|------|-----------|---------------|:-----:|:----------------:|
| GPT-3 175B | 175B | 300B | 1.7 | ❌ 严重不足 |
| Chinchilla 70B | 70B | 1.4T | 20 | ✅ |
| LLaMA 65B | 65B | 1.4T | 21.5 | ✅ |

GPT-3 的训练**远未收敛**——这解释了为什么后来更小但训练更充分的模型能超过 GPT-3。

---

## 3. GPT-3 架构的数学描述

### 3.1 Decoder-only 架构总览

GPT-3 沿用 GPT-2 的 **Decoder-only Transformer** 架构：

```
输入: "Translate English to French: cheese →"
  ↓
[Token Embedding + Position Embedding]
  ↓
[Transformer Decoder Block × 96]  ← 交替使用 dense 和 sparse attention
  ↓
[Layer Norm] → [LM Head] → "fromage"
```

**单层解码器块（Pre-Norm）**：

$$
\boxed{
\begin{aligned}
a^{(l)} &= x^{(l-1)} + \text{CausalAttn}\left(\text{LayerNorm}(x^{(l-1)})\right) \\
x^{(l)} &= a^{(l)} + \text{FFN}\left(\text{LayerNorm}(a^{(l)})\right)
\end{aligned}
}
$$

**与 GPT-2 的差异**：

| 组件 | GPT-2 | GPT-3 |
|------|-------|-------|
| 注意力模式 | 全 Dense | Dense + Locally Banded 交替 |
| 最大参数量 | 1.5B | 175B |
| 上下文长度 | 1024 | 2048 |
| 初始化 | 标准 | 按深度缩放残差连接 |
| 训练精度 | FP32 | 混合精度 (FP16) |

### 3.2 稀疏注意力模式：Dense 与 Locally Banded 交替

**Dense Attention**（标准全因果注意力）：

$$
A^{\text{dense}}_{ij} = \begin{cases} q_i^\top k_j / \sqrt{d_k} & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}
$$

**Locally Banded Sparse Attention**（局部带状注意力）：

$$
\boxed{A^{\text{sparse}}_{ij} = \begin{cases} q_i^\top k_j / \sqrt{d_k} & \text{if } i - w \leq j \leq i \\ -\infty & \text{otherwise} \end{cases}}
$$

其中 $w$ 是窗口大小（通常 $w = 256$），计算复杂度从 $O(n^2 d)$ 降至 $O(nwd)$。

**交替策略**：偶数层使用 Dense，奇数层使用 Locally Banded：

- **Dense 层**：捕获全局依赖（主题一致性、长距离指代）
- **Sparse 层**：捕获局部结构（语法、短语边界）
- 任意两个 token 之间的信息**最多经过 2 层**即可传递

### 3.3 GPT-3 各规模配置与参数估算

| 模型 | 参数量 | 层数 $L$ | 隐藏维度 $d$ | 头数 $A$ | $d_k$ | $d_{ff}$ |
|------|:------:|:------:|:--------:|:------:|:----:|:------:|
| Small | 125M | 12 | 768 | 12 | 64 | 3072 |
| XL | 1.3B | 24 | 2048 | 24 | 128 | 8192 |
| 13B | 13B | 40 | 5140 | 40 | 128 | 20480 |
| **175B** | **175B** | **96** | **12288** | **96** | **128** | **49152** |

**参数量估算（175B）**：

$$
P_{\text{layer}} = \underbrace{4d^2}_{\text{Self-Attn}} + \underbrace{2d \cdot d_{ff}}_{\text{FFN}} + \text{bias} \approx 12d^2 \approx 1.81\text{B}
$$

$$
P_{\text{total}} = P_{\text{emb}} + L \times P_{\text{layer}} \approx 0.64\text{B} + 96 \times 1.81\text{B} \approx 174.5\text{B}
$$

**训练计算量**：

$$
\boxed{C_{\text{GPT-3}} = 6ND \approx 3.14 \times 10^{23} \text{ FLOPs} \approx 3{,}640 \text{ PF-days}}
$$

---

## 4. In-context Learning 数学分析

### 4.1 Zero-shot / One-shot / Few-shot 形式化定义

**Zero-shot**：仅任务描述 → $P_\theta(y \mid \text{task\_desc}, x)$

**One-shot**：一个示例 → $P_\theta(y \mid \text{task\_desc}, x_1, y_1, x)$

**Few-shot**（$K$ 个示例）：

$$
\boxed{P_\theta(y \mid \text{task\_desc}, \{(x_k, y_k)\}_{k=1}^K, x) \quad \text{（} \theta \text{ 固定）}}
$$

**与微调的根本区别**：

| 维度 | Fine-tuning | In-context Learning |
|------|-------------|-------------------|
| 参数更新 | ✅ 更新 $\theta$ | ❌ $\theta$ 固定 |
| 标注数据量 | 数百~数千 | 0~数十 |
| 任务切换 | 重新微调 | 更换 prompt |
| 灾难性遗忘 | ⚠️ 可能 | ❌ 不会 |

### 4.2 In-context Learning 的条件概率视角

ICL 本质上是条件概率链的延伸。给定上下文 $\mathbf{c}$（含任务描述和示例）：

$$
P_\theta(\mathbf{y} \mid \mathbf{c}) = \prod_{t=1}^{T} P_\theta(y_t \mid c_1, \ldots, c_M, y_1, \ldots, y_{t-1})
$$

注意力机制允许模型动态关注上下文中最相关的部分：

$$
\alpha_{t,j} = \frac{\exp(q_t^\top k_j / \sqrt{d_k})}{\sum_{j'} \exp(q_t^\top k_{j'} / \sqrt{d_k})}
$$

通过 $\alpha_{t,j}$，模型可以同时关注任务描述（理解操作）、示例对应关系（学习模式）和当前查询（执行任务）。

### 4.3 为什么 In-context Learning 有效？——隐式贝叶斯推断

Xie et al. (2022) 提出 ICL 的**隐式贝叶斯推断**解释：预训练数据由多种"概念"$\theta_c$ 生成，ICL 在隐式地推断当前概念：

$$
\boxed{P(\theta_c \mid \text{examples}) \propto P(\text{examples} \mid \theta_c) \cdot P(\theta_c)}
$$

具体过程：

$$
P(\theta_c \mid x_1, y_1, \ldots, x_K, y_K) = \frac{\prod_{k=1}^K P(y_k \mid x_k, \theta_c) \cdot P(\theta_c)}{\sum_{\theta_c'} \prod_{k=1}^K P(y_k \mid x_k, \theta_c') \cdot P(\theta_c')}
$$

- **Zero-shot**：使用先验 $P(\theta_c)$
- **One-shot**：一个示例使后验开始集中
- **Few-shot**：多个示例使后验更尖锐，概念推断更准确

### 4.4 示例数量与性能的关系

**关键发现：模型越大，few-shot 的提升越明显。**

$$
\text{Accuracy}(N, K) \approx f(N) + g(N) \cdot \log(1 + K)
$$

$$
\boxed{g(N_{\text{large}}) \gg g(N_{\text{small}})}
$$

| 模型规模 | Zero-shot | Few-shot (K=32) | Δ |
|---------|:---------:|:--------------:|:-:|
| 125M | 42.1% | 44.8% | +2.7% |
| 1.3B | 51.3% | 58.2% | +6.9% |
| 13B | 57.8% | 68.5% | +10.7% |
| 175B | 64.2% | 78.9% | +14.7% |

> 数据为示意性展示，说明规模与 ICL 能力的"涌现"趋势。

---

## 5. 大规模训练优化方法总结

### 5.1 梯度累积

当显存不足时，**梯度累积**在 $G$ 个 micro-batch 上累加梯度，等效 batch size $B = G \cdot b$：

$$
\boxed{g_{\text{acc}} = \frac{1}{G} \sum_{j=1}^{G} \left(\frac{1}{b} \sum_{i=1}^{b} \nabla_\theta \ell(x_{j,i}, \theta)\right) = \frac{1}{B} \sum_{i=1}^{B} \nabla_\theta \ell(x_i, \theta)}
$$

数学上与大 batch 训练**完全等价**。GPT-3 还使用动态 batch size：从 32K tokens 逐步增至 3.2M tokens。

### 5.2 混合精度训练（FP16/BF16 + FP32）

| 格式 | 符号位 | 指数位 | 尾数位 | 动态范围 |
|------|:------:|:------:|:------:|:-------:|
| FP32 | 1 | 8 | 23 | $\sim 10^{\pm 38}$ |
| FP16 | 1 | 5 | 10 | $\sim 10^{\pm 5}$ |
| BF16 | 1 | 8 | 7 | $\sim 10^{\pm 38}$ |

**混合精度训练流程**：

$$
\boxed{
\begin{aligned}
&\textbf{1.} \; W_{\text{fp16}} \leftarrow \text{cast}(W_{\text{fp32}}) \quad
\textbf{2.} \; \text{loss} = \text{Forward}(x, W_{\text{fp16}}) \\
&\textbf{3.} \; g_{\text{fp16}} = \nabla (\text{loss} \times s) \quad
\textbf{4.} \; g_{\text{fp32}} = g_{\text{fp16}} / s \\
&\textbf{5.} \; W_{\text{fp32}} \leftarrow W_{\text{fp32}} - \eta \cdot g_{\text{fp32}}
\end{aligned}
}
$$

**损失缩放**防止 FP16 梯度下溢：初始 $s = 2^{16}$，连续无溢出则 $s \leftarrow 2s$，溢出则 $s \leftarrow s/2$。

### 5.3 模型并行训练

175B 参��无法放入单张 GPU，必须使用**模型并行**。

**张量并行——列并行**：将 $W$ 按列切分到 $p$ 张 GPU：

$$
W = [W_1 \mid W_2 \mid \cdots \mid W_p], \quad Y_i = X W_i
$$

**张量并行——行并行**：将 $W$ 按行切分，结果 AllReduce 求和：

$$
Y = \sum_{i=1}^{p} X_i W_i
$$

**FFN 的高效并行**（仅需一次 AllReduce）：

$$
\boxed{Y_1^{(i)} = \text{GELU}(x W_1^{(i)}) \quad \xrightarrow{\text{行并行}} \quad Y_2 = \sum_{i=1}^p Y_1^{(i)} W_2^{(i)}}
$$

**自注意力**天然适合并行——每张 GPU 处理 $A/p$ 个注意力头。

### 5.4 学习率调度与超参数

GPT-3 使用**余弦退火**学习率调度：

$$
\boxed{\eta(t) = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t - t_w}{T - t_w} \cdot \pi\right)\right)}
$$

**关键超参数（175B）**：

| 参数 | 值 |
|------|---|
| 优化器 | Adam ($\beta_1 = 0.9, \beta_2 = 0.95$) |
| 学习率 | $0.6 \times 10^{-4}$ |
| 权重衰减 | 0.1 |
| 梯度裁剪 | 1.0 |
| Batch Size (最终) | 3.2M tokens |

**残差连接的深度缩放**：

$$
W_{\text{residual}}^{(l)} \sim \mathcal{N}\left(0, \frac{0.02}{\sqrt{2L}}\right)
$$

确保 96 层模型残差路径方差不随深度爆炸。经验规律：$\eta^* \propto N^{-0.2}$。

---

## 6. 从数学到代码：完整实现

### 6.1 NumPy 实现核心组件

```python
import numpy as np


def softmax(x, axis=-1):
    """数值稳定的 Softmax: softmax(x_i) = exp(x_i - max) / Σexp(x_j - max)"""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def layer_norm(x, gamma, beta, eps=1e-5):
    """LayerNorm(x) = γ * (x - μ) / √(σ² + ε) + β"""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


def gelu(x):
    """GELU(x) = x · Φ(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))"""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def dense_causal_attention(Q, K, V):
    """
    标准全因果注意力

    数学公式: Attention = softmax(QK^T/√d_k + M_causal) V

    参数: Q, K, V: (batch, heads, seq_len, d_k)
    返回: output, weights
    """
    d_k = Q.shape[-1]
    seq_len = Q.shape[2]
    scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    causal_mask = np.triu(np.ones((seq_len, seq_len)) * (-1e9), k=1)
    scores = scores + causal_mask
    weights = softmax(scores, axis=-1)
    return np.matmul(weights, V), weights


def locally_banded_attention(Q, K, V, window_size=256):
    """
    局部带状稀疏注意力

    数学公式: A_ij = q_i^T k_j / √d_k  (仅当 i - w ≤ j ≤ i)

    参数:
        Q, K, V: (batch, heads, seq_len, d_k)
        window_size: 局部窗口大小 w
    """
    d_k = Q.shape[-1]
    seq_len = Q.shape[2]
    scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)

    # 因果掩码 + 窗口掩码
    mask = np.triu(np.ones((seq_len, seq_len)) * (-1e9), k=1)
    for i in range(seq_len):
        for j in range(max(0, i - window_size)):
            mask[i, j] = -1e9

    scores = scores + mask
    weights = softmax(scores, axis=-1)
    return np.matmul(weights, V), weights


def gpt3_block_numpy(x, Wq, Wk, Wv, Wo, W1, b1, W2, b2,
                     gamma1, beta1, gamma2, beta2,
                     layer_idx, num_heads, window_size=256):
    """
    GPT-3 单层 Transformer 块 (Pre-Norm)

    偶数层 Dense Attention，奇数层 Locally Banded Attention

    数学公式:
        a = x + CausalAttn(LayerNorm(x))
        output = a + FFN(LayerNorm(a))
    """
    B, T, d = x.shape
    d_k = d // num_heads

    # Pre-Norm + Self-Attention
    normed = layer_norm(x, gamma1, beta1)
    Q = np.dot(normed, Wq).reshape(B, T, num_heads, d_k).transpose(0, 2, 1, 3)
    K = np.dot(normed, Wk).reshape(B, T, num_heads, d_k).transpose(0, 2, 1, 3)
    V = np.dot(normed, Wv).reshape(B, T, num_heads, d_k).transpose(0, 2, 1, 3)

    if layer_idx % 2 == 0:
        attn_out, _ = dense_causal_attention(Q, K, V)
    else:
        attn_out, _ = locally_banded_attention(Q, K, V, window_size)

    attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, T, d)
    a = x + np.dot(attn_out, Wo)

    # Pre-Norm + FFN (GELU)
    normed2 = layer_norm(a, gamma2, beta2)
    ffn_out = np.dot(gelu(np.dot(normed2, W1) + b1), W2) + b2
    return a + ffn_out


def scaling_law_loss(N, N_c=8.8e13, alpha=0.076):
    """Scaling Law: L(N) = (N_c / N)^α"""
    return (N_c / N) ** alpha


# ========== 测试 ==========
if __name__ == "__main__":
    np.random.seed(42)
    B, T, d, H = 2, 32, 64, 4
    d_k, d_ff = d // H, d * 4

    # 1. Dense vs Sparse 注意力
    Q = np.random.randn(B, H, T, d_k)
    K = np.random.randn(B, H, T, d_k)
    V = np.random.randn(B, H, T, d_k)

    dense_out, dense_w = dense_causal_attention(Q, K, V)
    sparse_out, sparse_w = locally_banded_attention(Q, K, V, window_size=8)

    print(f"Dense 注意力 - 因果性 A[0,5]≈0: {dense_w[0,0,0,5] < 1e-6}")
    print(f"Sparse 注意力 - 窗口外 A[20,0]≈0: {sparse_w[0,0,20,0] < 1e-6}")
    print(f"Sparse 注意力 - 窗口内 A[20,15]>0: {sparse_w[0,0,20,15] > 0}")

    # 2. GPT-3 单层块
    x = np.random.randn(B, T, d)
    params = {k: np.random.randn(*s) * 0.02 for k, s in [
        ('Wq', (d, d)), ('Wk', (d, d)), ('Wv', (d, d)), ('Wo', (d, d)),
        ('W1', (d, d_ff)), ('W2', (d_ff, d))]}
    params.update({k: v for k, v in [
        ('b1', np.zeros(d_ff)), ('b2', np.zeros(d)),
        ('gamma1', np.ones(d)), ('beta1', np.zeros(d)),
        ('gamma2', np.ones(d)), ('beta2', np.zeros(d))]})

    out = gpt3_block_numpy(x, **{k: params[k] for k in
        ['Wq','Wk','Wv','Wo','W1','b1','W2','b2',
         'gamma1','beta1','gamma2','beta2']},
        layer_idx=0, num_heads=H)
    print(f"\nGPT-3 块输出形状: {out.shape}, 残差有效: {np.abs(out - x).max() > 0}")

    # 3. Scaling Law
    for N in [125e6, 1.3e9, 13e9, 175e9]:
        print(f"  N={N/1e9:.1f}B → Loss={scaling_law_loss(N):.4f}")

    print("\n✅ GPT-3 NumPy 核心组件测试通过！")
```

### 6.2 PyTorch 完整实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class GPT3Attention(nn.Module):
    """
    GPT-3 因果注意力（支持 Dense 和 Locally Banded 两种模式）

    Dense:  A_ij = q_i^T k_j / √d_k  (j ≤ i)
    Sparse: A_ij = q_i^T k_j / √d_k  (i - w ≤ j ≤ i)
    """
    def __init__(self, d_model: int, num_heads: int,
                 use_sparse: bool = False, window_size: int = 256,
                 dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_sparse = use_sparse
        self.window_size = window_size

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        Q = self.q_proj(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 因果掩码
        causal = torch.triu(torch.ones(T, T, device=x.device) * float('-inf'), diagonal=1)
        scores = scores + causal.unsqueeze(0).unsqueeze(0)

        # 局部窗口掩码（仅 sparse 层）
        if self.use_sparse:
            row = torch.arange(T, device=x.device).unsqueeze(1)
            col = torch.arange(T, device=x.device).unsqueeze(0)
            band = torch.where(col < row - self.window_size,
                              torch.tensor(float('-inf'), device=x.device),
                              torch.tensor(0.0, device=x.device))
            scores = scores + band.unsqueeze(0).unsqueeze(0)

        weights = self.attn_dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(weights, V)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


class GPT3Block(nn.Module):
    """
    GPT-3 Transformer 块 (Pre-Norm): a = x + Attn(LN(x)), out = a + FFN(LN(a))
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 use_sparse: bool = False, window_size: int = 256,
                 dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = GPT3Attention(d_model, num_heads, use_sparse, window_size, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.ln1(x)))
        h = self.dropout(self.fc2(F.gelu(self.fc1(self.ln2(x)))))
        return x + h


class GPT3Model(nn.Module):
    """
    完整 GPT-3 模型 (Decoder-only)

    注意力模式: 偶数层 Dense, 奇数层 Locally Banded
    初始化: 残差路径按 1/√(2L) 缩放
    """
    def __init__(self, vocab_size: int = 50257, d_model: int = 768,
                 num_heads: int = 12, num_layers: int = 12,
                 d_ff: int = 3072, max_len: int = 2048,
                 window_size: int = 256, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            GPT3Block(d_model, num_heads, d_ff,
                      use_sparse=(i % 2 == 1),
                      window_size=window_size, dropout=dropout)
            for i in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # 权重共享
        self.drop = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        """GPT-3 初始化: 标准 N(0,0.02), 残差路径 N(0, 0.02/√(2L))"""
        std = 0.02
        res_std = std / math.sqrt(2 * self.num_layers)
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                if 'o_proj' in name or 'fc2' in name:
                    nn.init.normal_(p, mean=0, std=res_std)
                else:
                    nn.init.normal_(p, mean=0, std=std)
            elif 'bias' in name:
                nn.init.zeros_(p)

    def forward(self, input_ids: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> dict:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(pos))

        for block in self.blocks:
            x = block(x)

        logits = self.lm_head(self.final_norm(x))

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, self.vocab_size),
                labels[:, 1:].contiguous().view(-1), ignore_index=-100)

        return {"logits": logits, "loss": loss}

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 50,
                 temperature: float = 1.0, top_k: int = 0) -> torch.Tensor:
        """自回归生成（支持 top-k 采样）"""
        for _ in range(max_new_tokens):
            idx = input_ids if input_ids.size(1) <= 2048 else input_ids[:, -2048:]
            logits = self.forward(idx)["logits"][:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            next_tok = torch.multinomial(F.softmax(logits, dim=-1), 1)
            input_ids = torch.cat([input_ids, next_tok], dim=1)
        return input_ids


# ========== 测试 ==========
if __name__ == "__main__":
    V, d, H, L, d_ff = 1000, 128, 4, 4, 512
    B, T = 4, 64

    model = GPT3Model(V, d, H, L, d_ff, max_len=256, window_size=16)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 注意力模式检查
    for i, b in enumerate(model.blocks):
        print(f"  Layer {i}: {'Sparse' if b.attn.use_sparse else 'Dense'}")

    # 前向传播
    ids = torch.randint(0, V, (B, T))
    model.eval()
    with torch.no_grad():
        out = model(ids, ids)
    print(f"\nLogits: {out['logits'].shape}, Loss: {out['loss'].item():.4f}")

    # 因果性验证
    with torch.no_grad():
        full = model(ids)["logits"]
        part = model(ids[:, :32])["logits"]
    diff = (full[:, :32, :] - part).abs().max().item()
    print(f"因果性: 前32位差异={diff:.6f}")
    assert diff < 1e-4

    # 权重共享
    assert torch.equal(model.tok_emb.weight.data, model.lm_head.weight.data)
    print("权重共享 ✓")

    # 深度缩放验证
    res_std = 0.02 / math.sqrt(2 * L)
    print(f"深度缩放: 目标={res_std:.4f}, "
          f"o_proj={model.blocks[0].attn.o_proj.weight.std():.4f}, "
          f"fc2={model.blocks[0].fc2.weight.std():.4f}")

    # 训练一步
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1,
                            betas=(0.9, 0.95))
    out = model(ids, ids)
    out["loss"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    print(f"训练后 Loss: {out['loss'].item():.4f}")

    # 生成
    model.eval()
    gen = model.generate(torch.randint(0, V, (1, 5)), max_new_tokens=10, top_k=50)
    print(f"生成: {gen[0].tolist()}")

    print("\n✅ GPT-3 模型测试通过！")
```

---

## 7. Scaling Laws 可视化与实践技巧

### 7.1 Scaling Laws 可视化

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_scaling_laws():
    """可视化 Scaling Laws: 损失 vs 参数量/数据量/计算量"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (1) 损失 vs 参数量
    N = np.logspace(6, 12, 100)
    L_N = (8.8e13 / N) ** 0.076
    axes[0].loglog(N, L_N, 'b-', lw=2)
    for name, size in [('125M', 125e6), ('1.3B', 1.3e9), ('13B', 13e9), ('175B', 175e9)]:
        axes[0].plot(size, (8.8e13/size)**0.076, 'ro', ms=8)
        axes[0].annotate(name, (size, (8.8e13/size)**0.076),
                        textcoords="offset points", xytext=(5, 8), fontsize=9)
    axes[0].set(xlabel='Parameters (N)', ylabel='Test Loss', title='Loss vs Parameters')
    axes[0].grid(True, alpha=0.3)

    # (2) 损失 vs 数据量
    D = np.logspace(8, 13, 100)
    L_D = (5.4e13 / D) ** 0.095
    axes[1].loglog(D, L_D, 'g-', lw=2)
    for name, d in [('GPT-3 300B', 300e9), ('Chinchilla 1.4T', 1.4e12)]:
        axes[1].plot(d, (5.4e13/d)**0.095, 'rs', ms=8)
        axes[1].annotate(name, (d, (5.4e13/d)**0.095),
                        textcoords="offset points", xytext=(5, 8), fontsize=9)
    axes[1].set(xlabel='Training Tokens (D)', ylabel='Test Loss', title='Loss vs Data')
    axes[1].grid(True, alpha=0.3)

    # (3) 损失 vs 计算量
    C = np.logspace(17, 25, 100)
    L_C = (3.1e8 / C) ** 0.050
    axes[2].loglog(C, L_C, 'r-', lw=2)
    axes[2].plot(3.14e23, (3.1e8/3.14e23)**0.050, 'ro', ms=10)
    axes[2].annotate('GPT-3', (3.14e23, (3.1e8/3.14e23)**0.050),
                    textcoords="offset points", xytext=(10, 10), fontsize=10)
    axes[2].set(xlabel='Compute (FLOPs)', ylabel='Test Loss', title='Loss vs Compute')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("gpt3_scaling_laws.png", dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    plot_scaling_laws()
```

### 7.2 实践调参建议

**模型规模选择（基于 Chinchilla 法则）**：

| 计算预算 | 推荐规模 | 训练 Token 数 |
|---------|---------|-------------|
| 1 GPU 周 | ~125M | ~2.5B |
| 8 GPU 周 | ~1.3B | ~26B |
| 64 GPU 周 | ~6.7B | ~134B |
| 1000 GPU 周 | ~70B | ~1.4T |

**学习率经验公式**：

$$
\boxed{\eta^* \approx 6 \times 10^{-4} \times \left(\frac{N}{125 \times 10^6}\right)^{-0.2}}
$$

**In-context Learning 提示工程**：

| 技巧 | 效果 |
|------|------|
| 示例多样性（覆盖不同子类别） | 显著提升 |
| 示例顺序随机化 | 减少偏差 |
| 清晰的分隔符 | 稳定性提升 |
| 标签平衡 | 减少类别偏差 |

---

## 8. 与其他模型的关系

### 8.1 GPT-2 vs GPT-3：从 Zero-shot 到 Few-shot

| 维度 | GPT-2 | GPT-3 |
|------|-------|-------|
| **参数量** | 1.5B | 175B (117x) |
| **训练数据** | ~40GB (WebText) | ~570GB (混合) |
| **上下文长度** | 1024 | 2048 |
| **核心范式** | Zero-shot | Few-shot (ICL) |
| **注意力** | 全 Dense | Dense + Sparse 交替 |

$$
\boxed{\underbrace{\text{GPT-2 (1.5B)}}_{\text{Zero-shot 能力有限}} \xrightarrow{117 \times} \underbrace{\text{GPT-3 (175B)}}_{\text{涌现 Few-shot 能力}}}
$$

### 8.2 GPT-3 在大模型发展中的定位

**预训练范式的三次转变**：

```
Phase 1: Pre-train + Fine-tune  (BERT, 2018)     ← 每任务需标注数据
Phase 2: Pre-train + Prompt     (GPT-3, 2020)     ← 无需微调，通过 prompt 使用
Phase 3: Pre-train + Align      (InstructGPT, 2022) ← RLHF 对齐人类偏好
```

**GPT-3 的核心贡献**：

1. **证明了规模的力量**：首次展示超大模型的涌现能力
2. **开创了 In-context Learning 范式**：无需微调的任务执行
3. **建立了 Scaling Laws 的实践基础**：指导后续模型规模选择
4. **推动了 Prompt Engineering 的兴起**：新的人机交互方式

### 8.3 GPT-3 的后续发展

```
GPT-3 (2020) ── 175B, Few-shot Learning
  ├── Codex (2021) ── 代码生成
  ├── InstructGPT (2022) ── RLHF 对齐
  │    └── ChatGPT (2022) → GPT-4 (2023)
  ├── Chinchilla (2022) ── 修正 Scaling Laws
  │    └── LLaMA (2023) ── 开源高效大模型
  └── PaLM (2022) → Gemini (2023)
```

| GPT-3 遗留问题 | 后续解决方案 |
|------------|------------|
| 生成不可控 | InstructGPT/RLHF |
| 训练效率低 | Chinchilla Scaling Laws |
| 仅文本模态 | GPT-4, Gemini (多模态) |
| 参数量过大 | LoRA, QLoRA |
| 闭源不可复现 | LLaMA, Mistral |

---

## 扩展阅读与实现

### 问题 1：GPT-3 的训练数据构成及其影响

GPT-3 使用多数据源混合，高质量数据的采样比例**远高于**其在总量中的占比：

| 数据集 | 采样权重 | Token 数 | Epoch 数 |
|--------|:-------:|:-------:|:--------:|
| Common Crawl | 60% | 410B | 0.44 |
| WebText2 | 22% | 19B | 3.4 |
| Books1 | 8% | 12B | 2.0 |
| Books2 | 8% | 55B | 0.44 |
| Wikipedia | 3% | 3B | 3.0 |

WebText2 只占总量 ~4%，但采样权重 22%——体现了**数据质量 > 数据数量**。

### 问题 2：稀疏注意力的计算效率分析

对于 GPT-3 175B（$n=2048, d=12288, w=256$）：

$$
\frac{C_{\text{sparse}}}{C_{\text{dense}}} = \frac{w}{n} = \frac{256}{2048} = 12.5\%
$$

交替策略（48 层 Dense + 48 层 Sparse）的注意力计算节省约 43.75%。但 FFN 计算量更大，实际总节省约 15-20%。

### 问题 3：In-context Learning 的局限性

1. **上下文长度限制**：$K_{\max} = (n_{\text{ctx}} - |\text{desc}| - |\text{query}|) / |\text{example}|$，通常 $K \leq 100$
2. **对格式敏感**：换行符位置 ±5%，标签措辞 ±10-15%
3. **不能学习全新概念**：ICL 是**激活**预训练知识，非学习新映射
4. **推理效率**：$C_{\text{inference}} \propto (K \cdot |\text{example}| + |\text{query}|)^2$

### 问题 4：训练不稳定性及解决方案

- **损失尖峰**：梯度裁剪 $\|\nabla\| \leq 1.0$ + 动态损失缩放 + 检查点回滚
- **学习率敏感**：$\eta^* \propto N^{-0.2}$（175B 的 $\eta$ 比 125M 小 10 倍）
- **深度缩放**：$W_{\text{res}} \sim \mathcal{N}(0, 0.02/\sqrt{2L})$，保证方差与深度 $L$ 无关

---

## 参考资源

### 经典论文

1. Brown, T. B., Mann, B., Ryder, N., et al. (2020). [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165). NeurIPS 2020.
   - **贡献**：提出 GPT-3，首次展示大规模语言模型的 In-context Learning 能力

2. Kaplan, J., McCandlish, S., Henighan, T., et al. (2020). [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361). arXiv.
   - **贡献**：建立了语言模型损失与参数量/数据量/计算量的幂律关系

3. Hoffmann, J., Borgeaud, S., Mensch, A., et al. (2022). [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556). NeurIPS 2022.
   - **贡献**：Chinchilla 论文，修正了计算最优分配策略

4. Child, R., Gray, S., Radford, A., & Sutskever, I. (2019). [Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509). arXiv.
   - **贡献**：提出稀疏注意力模式，GPT-3 采用了其 locally banded 变体

5. Micikevicius, P., Narang, S., Alben, J., et al. (2018). [Mixed Precision Training](https://arxiv.org/abs/1710.03740). ICLR 2018.
   - **贡献**：提出混合精度训练方法

6. Shoeybi, M., Patwary, M., Puri, R., et al. (2019). [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053). arXiv.
   - **贡献**：提出张量并行策略

7. Xie, S. M., Raghunathan, A., Liang, P., & Ma, T. (2022). [An Explanation of In-context Learning as Implicit Bayesian Inference](https://arxiv.org/abs/2111.02080). ICLR 2022.
   - **贡献**：从贝叶斯推断角度解释 ICL

### 教材与书籍

8. Jurafsky, D., & Martin, J. H. [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/). 3rd ed. (Draft).
   - **章节**：第 10-12 章讲解大规模语言模型

### 在线资源与教程

9. Lilian Weng. [How to Train Really Large Models on Many GPUs](https://lilianweng.github.io/posts/2021-09-25-train-large/).
   - **内容**：大规模模型训练的并行策略详解

10. OpenAI. [GPT-3 API Documentation](https://platform.openai.com/docs/).
    - **内容**：GPT-3 的 API 使用和 prompt 设计最佳实践

---

## 附录：符号表

| 符号 | 含义 | 维度/类型 |
|------|------|----------|
| $n$ ($n_{\text{ctx}}$) | 上下文序列长度 | 标量，GPT-3: 2048 |
| $d$ ($d_{\text{model}}$) | 隐藏维度 | 标量，175B: 12288 |
| $d_k$ | 每个注意力头的维度 | 标量，128 |
| $d_{ff}$ | FFN 隐藏层维度 | 标量，$4d$ |
| $L$ | Transformer 层数 | 标量，175B: 96 |
| $A$ | 注意力头数 | 标量，175B: 96 |
| $N$ | 模型参数量 | 标量 |
| $D$ | 训练数据量 (token 数) | 标量 |
| $C$ | 训练计算量 (FLOPs) | 标量 |
| $\|V\|$ | BPE 词表大小 | 标量，50,257 |
| $w$ | 稀疏注意力窗口大小 | 标量，256 |
| $K$ | In-context 示例数量 | 标量 |
| $\mathcal{L}$ | 交叉熵损失 | 标量 |
| $\ell(\cdot, \cdot)$ | 交叉熵损失函数 | 函数 |
| $\mathcal{L}(N)$ | 参数量 $N$ 时的损失 | 标量（幂律） |
| $\alpha_N, \alpha_D, \alpha_C$ | Scaling Law 幂律指数 | 标量 |
| $N_c, D_c, C_c$ | Scaling Law 常数 | 标量 |
| $P_\theta(\cdot)$ | 参数为 $\theta$ 的语言模型 | 条件概率 |
| $\theta$ | 模型参数（ICL 中固定） | 参数向量 |
| $\theta_c$ | 隐式"概念"（贝叶斯推断） | 隐变量 |
| $Q, K, V$ | 查询、键、值矩阵 | $(n, d_k)$ |
| $s$ | 损失缩放因子 | 标量 |
| $\eta$ | 学习率 | 标量 |
| $G$ | 梯度累积步数 | 标量 |
| $B$ | Batch size (token 数) | 标量 |

**典型维度示例（GPT-3 175B）：**
- $d = 12{,}288$，$d_k = 128$，$d_{ff} = 49{,}152$
- $|V| = 50{,}257$，$L = 96$，$A = 96$，$n_{\text{ctx}} = 2{,}048$

---

最后更新：2026-03-19
