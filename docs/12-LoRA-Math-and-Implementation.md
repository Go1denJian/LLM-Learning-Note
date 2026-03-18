# LoRA 数学原理与实现 —— 低秩适应的完整推导

> **前置知识**：线性代数（矩阵分解、秩）、Transformer 架构、预训练-微调范式、Python 基础  
> **与前面内容的联系**：建议先学习 [Transformer-Math-and-Implementation](./06-Transformer-Math-and-Implementation.md) 和 [GPT3-Scaling-and-InContext](./10-GPT3-Scaling-and-InContext.md)，理解标准 Transformer 和大规模预训练  
> **与后续内容的联系**：LoRA 的参数高效微调思想直接影响了 QLoRA、DoRA、LoRA+ 等后续工作，也是 InstructGPT/RLHF 微调的关键基础设施

---

## 目录

1. [引言：为什么需要参数高效微调？](#1-引言为什么需要参数高效微调)
   - 1.1 [全参数微调的困境](#11-全参数微调的困境)
   - 1.2 [参数高效微调的核心思想](#12-参数高效微调的核心思想)
   - 1.3 [LoRA 的关键创新](#13-lora-的关键创新)
   - 1.4 [本科数学知识映射表](#14-本科数学知识映射表)
2. [低秩分解的数学基础](#2-低秩分解的数学基础)
   - 2.1 [矩阵秩回顾](#21-矩阵秩回顾)
   - 2.2 [低秩近似定理（Eckart-Young）](#22-低秩近似定理eckart-young)
   - 2.3 [预训练权重的内在低秩性](#23-预训练权重的内在低秩性)
   - 2.4 [低秩参数化：从 $\Delta W$ 到 $BA$](#24-低秩参数化从-delta-w-到-ba)
3. [LoRA 核心算法](#3-lora-核心算法)
   - 3.1 [标准微调 vs LoRA 微调](#31-标准微调-vs-lora-微调)
   - 3.2 [LoRA 的前向传播](#32-lora-的前向传播)
   - 3.3 [初始化策略的数学分析](#33-初始化策略的数学分析)
   - 3.4 [缩放因子 $\alpha / r$ 的作用](#34-缩放因子-alpha--r-的作用)
   - 3.5 [参数量与存储分析](#35-参数量与存储分析)
4. [梯度推导与参数更新](#4-梯度推导与参数更新)
   - 4.1 [LoRA 层的梯度计算](#41-lora-层的梯度计算)
   - 4.2 [梯度维度分析](#42-梯度维度分析)
   - 4.3 [冻结参数与可训练参数的分离](#43-冻结参数与可训练参数的分离)
   - 4.4 [权重合并：零额外推理延迟](#44-权重合并零额外推理延迟)
5. [训练优化方法总结](#5-训练优化方法总结)
   - 5.1 [秩 $r$ 的选择策略](#51-秩-r-的选择策略)
   - 5.2 [应用位置：哪些权重矩阵需要 LoRA？](#52-应用位置哪些权重矩阵需要-lora)
   - 5.3 [学习率与优化器选择](#53-学习率与优化器选择)
   - 5.4 [与其他参数高效方法的对比](#54-与其他参数高效方法的对比)
6. [从数学到代码：完整实现](#6-从数学到代码完整实现)
   - 6.1 [NumPy 实现核心组件](#61-numpy-实现核心组件)
   - 6.2 [PyTorch 完整实现](#62-pytorch-完整实现)
7. [实践技巧与可视化](#7-实践技巧与可视化)
   - 7.1 [低秩近似可视化](#71-低秩近似可视化)
   - 7.2 [实践调参建议](#72-实践调参建议)
8. [与其他模型的关系](#8-与其他模型的关系)
   - 8.1 [参数高效微调方法谱系](#81-参数高效微调方法谱系)
   - 8.2 [LoRA 在大模型发展中的定位](#82-lora-在大模型发展中的定位)
   - 8.3 [LoRA 的后续发展](#83-lora-的后续发展)

[扩展阅读与实现](#扩展阅读与实现)

[参考资源](#参考资源)

附录：[符号表](#附录符号表)

---

## 1. 引言：为什么需要参数高效微调？

### 1.1 全参数微调的困境

GPT-3 以来，预训练模型的参数量爆炸式增长：

| 模型 | 参数量 | 全量微调所需 GPU 显存 | 存储每个任务的检查点 |
|------|:------:|:-------------------:|:------------------:|
| BERT-Base | 110M | ~1 GB | ~440 MB |
| GPT-2 | 1.5B | ~12 GB | ~6 GB |
| GPT-3 | 175B | ~1.2 TB | ~700 GB |
| LLaMA-65B | 65B | ~520 GB | ~260 GB |

全参数微调要求更新**所有**模型参数：

$$
W_{\text{new}} = W_{\text{pretrained}} - \eta \frac{\partial \mathcal{L}_{\text{task}}}{\partial W}
$$

这带来三大困境：

1. **计算成本**：需要在所有参数上计算梯度并更新，显存占用巨大
2. **存储成本**：每个下游任务需要保存一份完整的模型副本
3. **部署困难**：多任务服务需要多份模型实例，资源浪费严重

$$
\boxed{\text{存储总成本} = N_{\text{tasks}} \times |W| \times \text{bytes\_per\_param}}
$$

对于 175B 参数模型和 100 个任务：$100 \times 175\text{B} \times 2\text{B} = 35\text{TB}$——不可接受。

### 1.2 参数高效微调的核心思想

参数高效微调（Parameter-Efficient Fine-Tuning, PEFT）的目标：

$$
\boxed{\text{只更新极少量参数} \quad |\Theta_{\text{trainable}}| \ll |\Theta_{\text{total}}| \quad \text{但保持微调效果}}
$$

现有方法分为三大类：

| 方法类别 | 代表工作 | 核心思想 |
|---------|---------|---------|
| **Adapter** | Adapter (Houlsby, 2019) | 在层间插入小型可训练模块 |
| **Prefix/Prompt** | Prefix-Tuning (Li, 2021) | 在输入序列前添加可训练向量 |
| **低秩适应** | **LoRA (Hu, 2021)** | 将权重更新约束为低秩矩阵 |

### 1.3 LoRA 的关键创新

LoRA（Low-Rank Adaptation）的核心假设：

> **预训练模型在微调过程中的权重变化 $\Delta W$ 具有很低的"内在秩"（Intrinsic Rank）。**

$$
\boxed{\Delta W = BA, \quad B \in \mathbb{R}^{d \times r}, \; A \in \mathbb{R}^{r \times k}, \; r \ll \min(d, k)}
$$

**三大优势**：

1. **参数效率**：可训练参数减少 $10{,}000\times$ 以上
2. **零额外推理延迟**：训练后可将 $BA$ 合并到原始权重
3. **任务切换**：只需交换小型 LoRA 模块（几 MB），共享基础模型

**与 Adapter 和 Prefix-Tuning 的关键区别**：

| 方法 | 推理延迟 | 可合并到基础模型 | 可训练参数占比 |
|------|:-------:|:--------------:|:------------:|
| Adapter | 增加（串行模块） | ❌ | ~3.6% |
| Prefix-Tuning | 增加（序列变长） | ❌ | ~0.1% |
| **LoRA** | **无增加** | **✅** | **~0.02%** |

### 1.4 本科数学知识映射表

| 数学概念 | LoRA 中的应用 | 代码对应 |
|---------|-------------|---------|
| 矩阵秩 $\text{rank}(M)$ | 衡量权重更新的复杂度 | `np.linalg.matrix_rank(W)` |
| SVD 分解 $M = U\Sigma V^\top$ | 低秩近似的理论基础 | `np.linalg.svd(W)` |
| Frobenius 范数 $\|M\|_F$ | 低秩近似的误差度量 | `np.linalg.norm(W, 'fro')` |
| 矩阵乘法 $BA$ | LoRA 的核心参数化 | `B @ A` |
| 高斯初始化 $\mathcal{N}(0, \sigma^2)$ | $A$ 矩阵的 Kaiming 初始化 | `nn.init.kaiming_uniform_` |
| 零矩阵 $\mathbf{0}$ | $B$ 矩阵的初始化 | `nn.init.zeros_` |
| 链式法则 | LoRA 梯度计算 | `loss.backward()` |
| 矩阵加法 $W + \Delta W$ | 权重合并推理 | `W_merged = W + (alpha/r) * B @ A` |

---

## 2. 低秩分解的数学基础

### 2.1 矩阵秩回顾

对于矩阵 $M \in \mathbb{R}^{m \times n}$，秩 $\text{rank}(M)$ 定义为 $M$ 的线性无关行（或列）的最大数量：

$$
\text{rank}(M) \leq \min(m, n)
$$

**几何直觉**：秩 $r$ 的矩阵将 $n$ 维输入映射到 $m$ 维空间中一个 $r$ 维子空间。

**低秩矩阵的等价表示**：如果 $\text{rank}(M) = r$，则存在分解：

$$
M = B A, \quad B \in \mathbb{R}^{m \times r}, \; A \in \mathbb{R}^{r \times n}
$$

存储量从 $mn$ 降为 $(m + n)r$。

### 2.2 低秩近似定理（Eckart-Young）

**奇异值分解（SVD）**：任意矩阵 $M \in \mathbb{R}^{m \times n}$ 可分解为：

$$
M = U \Sigma V^\top = \sum_{i=1}^{\min(m,n)} \sigma_i u_i v_i^\top
$$

其中 $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$。

**Eckart-Young 定理**：在 Frobenius 范数下，最优秩 $r$ 近似为：

$$
\boxed{M_r = \sum_{i=1}^{r} \sigma_i u_i v_i^\top = U_r \Sigma_r V_r^\top, \quad \|M - M_r\|_F = \sqrt{\sum_{i=r+1}^{\min(m,n)} \sigma_i^2}}
$$

如果奇异值快速衰减，则很小的 $r$ 就能捕获 $M$ 的大部分"信息"。

### 2.3 预训练权重的内在低秩性

Aghajanyan et al. (2020) 的关键发现：**预训练语言模型具有极低的"内在维度"（Intrinsic Dimensionality）**。

| 模型 | 参数量 | 内在维度 $d_{\text{int}}$ | 比例 $d_{\text{int}} / |\Theta|$ |
|------|:------:|:------------------------:|:-------------------------------:|
| BERT-Base | 110M | ~200 | $1.8 \times 10^{-6}$ |
| RoBERTa-Base | 125M | ~896 | $7.2 \times 10^{-6}$ |
| GPT-2 Medium | 354M | ~1,856 | $5.2 \times 10^{-6}$ |

这意味着微调过程中的有效参数空间远小于模型的总参数空间。定义 $\Delta W = W^* - W_0$，实验观察：$\Delta W$ 的奇异值**快速衰减**——

$$
\sigma_1(\Delta W) \gg \sigma_2(\Delta W) \gg \cdots \gg \sigma_r(\Delta W) \gg \sigma_{r+1}(\Delta W) \approx 0
$$

### 2.4 低秩参数化：从 $\Delta W$ 到 $BA$

LoRA 不直接学习 $\Delta W \in \mathbb{R}^{d \times k}$（参数量 $dk$），而是将其参数化为两个小矩阵的乘积：

$$
\boxed{\Delta W = B A, \quad B \in \mathbb{R}^{d \times r}, \; A \in \mathbb{R}^{r \times k}}
$$

**参数量对比**：

| 方法 | 参数量 | $d=4096, k=4096, r=8$ |
|------|:------:|:---------------------:|
| 全量 $\Delta W$ | $dk$ | $16{,}777{,}216$ |
| 低秩 $BA$ | $(d+k)r$ | $65{,}536$ |
| **压缩比** | $dk / [(d+k)r]$ | **$256\times$** |

**为什么用 $BA$ 乘积形式？** (1) 自然保证 $\text{rank}(\Delta W) \leq r$；(2) 无约束优化，标准梯度下降即可；(3) 计算高效，只需两次矩阵乘法。

---

## 3. LoRA 核心算法

### 3.1 标准微调 vs LoRA 微调

**标准全参数微调**：$h = (W_0 + \Delta W) x$，所有 $dk$ 个参数都是可训练的。

**LoRA 微调**：

$$
\boxed{h = W_0 x + \Delta W x = W_0 x + B A x}
$$

$W_0$ 被**冻结**，只有 $A, B$ 是可训练的——参数量从 $dk$ 降为 $(d+k)r$。

```
              ┌───────────────────┐
              │ W₀ (冻结, d × k) │
    x ──────▶│                   │──────▶ h = W₀x + BAx
    │         └───────────────────┘         ▲
    │                                       │
    │         ┌─────────┐  ┌─────────┐      │
    └────────▶│ A (r×k) │─▶│ B (d×r) │──────┘
              └─────────┘  └─────────┘
              (可训练)       (可训练)
```

### 3.2 LoRA 的前向传播

对于输入 $x \in \mathbb{R}^{k}$，完整前向传播：

**Step 1**：$h_0 = W_0 x$（原始输出）

**Step 2**：$\Delta h = \frac{\alpha}{r} B(Ax)$（低秩增量）

**Step 3**：$\boxed{h = h_0 + \Delta h = W_0 x + \frac{\alpha}{r} B A x}$

计算顺序：先算 $Ax$（$r \times k$ 乘 $k \times 1 = r \times 1$），再算 $B(Ax)$（$d \times r$ 乘 $r \times 1 = d \times 1$），总计算量为 $(d + k)r$ 次乘加——远小于 $dk$。

### 3.3 初始化策略的数学分析

LoRA 使用**非对称初始化**：

$$
\boxed{A \sim \mathcal{N}(0, \sigma_A^2), \quad B = \mathbf{0}}
$$

其中 $\sigma_A$ 通常采用 Kaiming 初始化：$\sigma_A = \sqrt{1/k}$。

**为什么 $B$ 初始化为零？** 训练开始时 $\Delta W = B A = \mathbf{0} \cdot A = \mathbf{0}$，保证 **LoRA 在训练初期等价于原始预训练模型**。

**为什么不两个都为零？** 如果 $A = B = \mathbf{0}$，梯度也为零，训练无法开始：

$$
\frac{\partial \mathcal{L}}{\partial A} = B^\top \frac{\partial \mathcal{L}}{\partial h} = \mathbf{0}^\top \cdot (\cdots) = \mathbf{0}
$$

> **Q:** 能否让 $A = \mathbf{0}$，$B \sim \mathcal{N}$？
>
> **A:** 数学上可以，效果等价。Hu et al. 选择 $B = \mathbf{0}$ 可能因为 $A$ 在"下投影"端（降维），非零初始化让第一步就产生有意义的低维表示。

### 3.4 缩放因子 $\alpha / r$ 的作用

$$
h = W_0 x + \frac{\alpha}{r} B A x
$$

当改变秩 $r$ 时，$\Delta W = BA$ 的规模会随 $r$ 变化。$\alpha / r$ 的缩放使得：

$$
\boxed{\text{改变 } r \text{ 时，} \alpha \text{ 保持不变，只需同比调整学习率}}
$$

**常见设置**：$\alpha = r$（无缩放）、$\alpha = 2r$（放大影响）、$\alpha = 16, r = 8$（论文默认）。

### 3.5 参数量与存储分析

**整个模型的 LoRA 参数量**（应用于 $L$ 层 Transformer 的 $W_Q, W_V$）：

$$
\boxed{P_{\text{total\_LoRA}} = N_{\text{layers}} \times N_{\text{matrices}} \times r(d + k)}
$$

**GPT-3 175B，$r = 8$，对 $W_Q, W_V$ 应用 LoRA**：

- 每层：$2 \times 8 \times 2 \times 12288 = 393{,}216$ 参数
- 96 层合计：$37.7\text{M}$，占比 $0.02\%$
- 存储：75 MB vs 全量微调 350 GB（$4{,}667\times$ 压缩）

---

## 4. 梯度推导与参数更新

### 4.1 LoRA 层的梯度计算

设 $s = \alpha / r$，$z = Ax$，则 $h = W_0 x + s \cdot Bz$。

**对 $B$ 的梯度**：

$$
\boxed{\frac{\partial \mathcal{L}}{\partial B} = \frac{\alpha}{r} \cdot \frac{\partial \mathcal{L}}{\partial h} (Ax)^\top \in \mathbb{R}^{d \times r}}
$$

**对 $A$ 的梯度**：

$$
\boxed{\frac{\partial \mathcal{L}}{\partial A} = \frac{\alpha}{r} \cdot B^\top \frac{\partial \mathcal{L}}{\partial h} \cdot x^\top \in \mathbb{R}^{r \times k}}
$$

**$W_0$ 的梯度存在但不使用**（冻结），节省优化器状态显存。

### 4.2 梯度维度分析

| 量 | 维度 | 说明 |
|----|------|------|
| $\frac{\partial \mathcal{L}}{\partial h}$ | $(d, 1)$ | 上游梯度 |
| $\frac{\partial \mathcal{L}}{\partial B}$ | $(d, r)$ | $B$ 的梯度 |
| $\frac{\partial \mathcal{L}}{\partial A}$ | $(r, k)$ | $A$ 的梯度 |

**梯度 FLOPs**：LoRA 为 $2dr + rk$，全量为 $dk$。当 $r \ll d, k$ 时，LoRA 梯度计算量远小于全量。

### 4.3 冻结参数与可训练参数的分离

**Adam 优化器显存分析**（参数 + 一阶矩 + 二阶矩）：

| 方法 | 可训练参数 | 优化器显存 (FP32) |
|------|:---------:|:----------------:|
| 全量微调 GPT-3 | 175B | $175\text{B} \times 12 = 2.1\text{TB}$ |
| LoRA $r=8$ | 37.7M | $37.7\text{M} \times 12 = 452\text{MB}$ |

### 4.4 权重合并：零额外推理延迟

训练完成后可**合并权重**：

$$
\boxed{W_{\text{merged}} = W_0 + \frac{\alpha}{r} BA}
$$

合并后前向传播 $h = W_{\text{merged}} x$ 与标准线性层完全相同——**零额外计算开销**。

**任务切换**：$W_{\text{task}_2} = W_{\text{task}_1} - \frac{\alpha}{r} B_1 A_1 + \frac{\alpha}{r} B_2 A_2$，只需交换几 MB 的 LoRA 模块。

---

## 5. 训练优化方法总结

### 5.1 秩 $r$ 的选择策略

**Hu et al. (2021) 的实验发现**：

| 秩 $r$ | GPT-3 175B WikiSQL Acc | GPT-3 175B MNLI Acc |
|:------:|:----------------------:|:-------------------:|
| 1 | 73.4 | 91.7 |
| 4 | **73.7** | 91.5 |
| 8 | 73.7 | **91.6** |
| 64 | 73.4 | 91.4 |

$$
\boxed{\text{极小的 } r \text{（如 } r = 1 \sim 8 \text{）就足以达到接近全量微调的效果}}
$$

| 场景 | 推荐 $r$ | 理由 |
|------|:-------:|------|
| 简单任务（情感分类） | 1~4 | 任务简单，低秩即可 |
| 中等任务（NLI、QA） | 4~16 | 平衡效果和效率 |
| 复杂任务（代码生成） | 16~64 | 需要更多适应能力 |
| 跨域适应 | 32~128 | 域差距大，需更多自由度 |

### 5.2 应用位置：哪些权重矩阵需要 LoRA？

| 应用位置 | 可训练参数 | WikiSQL Acc |
|---------|:---------:|:----------:|
| $W_Q$ only | 4.7M | 70.4 |
| $W_V$ only | 4.7M | 73.0 |
| $W_Q, W_V$ | 9.4M | **73.4** |
| $W_Q, W_K, W_V, W_O$ | 18.8M | 73.2 |

$$
\boxed{\text{推荐默认：对 } W_Q \text{ 和 } W_V \text{ 应用 LoRA（效果/参数最优比）}}
$$

后续研究发现对所有线性层（包括 FFN）都应用 LoRA 可以进一步提升效果，尤其在更大模型上。

### 5.3 学习率与优化器选择

LoRA 学习率通常比全参数微调**更大**：

| 方法 | 典型学习率 | 优化器 |
|------|:---------:|:------:|
| 全参数微调 | $1 \times 10^{-5}$ ~ $5 \times 10^{-5}$ | AdamW |
| LoRA | $1 \times 10^{-4}$ ~ $3 \times 10^{-4}$ | AdamW |

更大的学习率补偿了低秩约束带来的表达能力限制。

### 5.4 与其他参数高效方法的对比

| 方法 | 可训练参数 | 推理延迟 | 模型质量 | 可合并 |
|------|:---------:|:-------:|:-------:|:-----:|
| **全参数微调** | 100% | 无额外 | 最佳 | N/A |
| **Adapter** (2019) | ~3.6% | +增加 | 好 | ❌ |
| **Prefix-Tuning** (2021) | ~0.1% | +增加 | 良 | ❌ |
| **BitFit** (2021) | ~0.08% | 无额外 | 中 | ✅ |
| **LoRA** (2021) | ~0.02% | **无额外** | **好** | **✅** |

$$
\boxed{\text{LoRA} = \text{最少参数} + \text{零推理延迟} + \text{可合并} \quad \Rightarrow \quad \text{实践最优选择}}
$$

---

## 6. 从数学到代码：完整实现

### 6.1 NumPy 实现核心组件

```python
import numpy as np


def svd_rank_analysis(W, top_k=10):
    """
    分析矩阵的奇异值分布，验证低秩假设

    W = UΣV^T, 计算前 k 个奇异值捕获的能量比例
    """
    U, sigma, Vt = np.linalg.svd(W, full_matrices=False)
    total_energy = np.sum(sigma ** 2)
    cumulative_energy = np.cumsum(sigma ** 2) / total_energy
    return sigma, cumulative_energy[:top_k]


def low_rank_approx(W, r):
    """
    秩 r 最优近似 (Eckart-Young): W_r = U_r Σ_r V_r^T

    返回: (W_r, 相对 Frobenius 误差)
    """
    U, sigma, Vt = np.linalg.svd(W, full_matrices=False)
    W_r = (U[:, :r] * sigma[:r]) @ Vt[:r, :]
    error = np.sqrt(np.sum(sigma[r:] ** 2))
    total = np.sqrt(np.sum(sigma ** 2))
    return W_r, error / total if total > 0 else 0.0


class LoRALinearNumPy:
    """
    LoRA 线性层: h = W₀x + (α/r) · B · A · x
    初始化: A ~ N(0, 1/k), B = 0
    """
    def __init__(self, d_out, d_in, r, alpha=None):
        self.d_out, self.d_in, self.r = d_out, d_in, r
        self.alpha = alpha if alpha is not None else r
        self.scaling = self.alpha / self.r

        # 冻结权重 W₀
        self.W0 = np.random.randn(d_out, d_in) * 0.02
        # LoRA 参数: A ~ Kaiming, B = 0
        self.A = np.random.randn(r, d_in) * np.sqrt(1.0 / d_in)
        self.B = np.zeros((d_out, r))
        self.grad_A = self.grad_B = None
        self._cache = self._cache_z = None

    def forward(self, x):
        """前向: h = W₀x + (α/r) · B(Ax)"""
        self._cache = x
        h0 = x @ self.W0.T                       # 原始输出
        z = x @ self.A.T                          # (batch, r) 降维
        self._cache_z = z
        return h0 + self.scaling * (z @ self.B.T)  # 合并

    def backward(self, grad_h):
        """反向: ∂L/∂B = s·(∂L/∂h)^T·z, ∂L/∂A = s·B^T·(∂L/∂h)·x^T"""
        x, z = self._cache, self._cache_z
        self.grad_B = self.scaling * (grad_h.T @ z)          # (d_out, r)
        grad_z = self.scaling * (grad_h @ self.B)             # (batch, r)
        self.grad_A = grad_z.T @ x                            # (r, d_in)

    def update(self, lr):
        """SGD 更新（只更新 A, B）"""
        self.A -= lr * self.grad_A
        self.B -= lr * self.grad_B

    def merge(self):
        """权重合并: W_merged = W₀ + (α/r) · BA"""
        return self.W0 + self.scaling * (self.B @ self.A)


# ========== 测试 ==========
if __name__ == "__main__":
    np.random.seed(42)
    d_in, d_out, r, alpha = 512, 512, 8, 16

    # 1. LoRA 层测试
    lora = LoRALinearNumPy(d_out, d_in, r, alpha)
    frozen, trainable = lora.W0.size, lora.A.size + lora.B.size
    print(f"LoRA 层: {d_in}→{d_out}, r={r}, α={alpha}")
    print(f"  冻结: {frozen:,}, 可训练: {trainable:,}, 占比: {trainable/frozen:.4%}")

    # 2. 初始化验证: ΔW = 0
    assert np.allclose(lora.B @ lora.A, 0), "初始 ΔW 应为零"
    x = np.random.randn(4, d_in)
    h = lora.forward(x)
    assert np.allclose(h, x @ lora.W0.T), "初始输出应等于 W₀x"
    print(f"  初始化 ✓: ||ΔW|| = {np.linalg.norm(lora.B @ lora.A):.6f}")

    # 3. 训练一步
    target = np.random.randn(4, d_out)
    grad_h = 2 * (h - target) / 4
    lora.backward(grad_h)
    lora.update(lr=1e-3)
    delta_W = lora.B @ lora.A
    print(f"  训练后: ||ΔW|| = {np.linalg.norm(delta_W):.6f}, rank = {np.linalg.matrix_rank(delta_W)} (≤{r})")

    # 4. 权重合并
    W_merged = lora.merge()
    h_merged = x @ W_merged.T
    h_lora = lora.forward(x)
    assert np.allclose(h_merged, h_lora, atol=1e-6)
    print(f"  权重合并 ✓: 差异 = {np.abs(h_merged - h_lora).max():.2e}")

    # 5. 低秩分析
    print(f"\n低秩近似分析 (256×256 随机矩阵):")
    W = np.random.randn(256, 256)
    for rank in [1, 4, 8, 16, 32]:
        _, err = low_rank_approx(W, rank)
        print(f"  秩 {rank:2d}: 相对误差 = {err:.4f}")

    print("\n✅ NumPy LoRA 核心组件测试通过！")
```

### 6.2 PyTorch 完整实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, List, Tuple


class LoRALinear(nn.Module):
    """
    LoRA 线性层: h = W₀x + (α/r) · B · A · x

    W₀ 冻结，只训练 A 和 B。初始化: A ~ Kaiming, B = 0 → 初始 ΔW = 0
    """
    def __init__(self, in_features: int, out_features: int,
                 r: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.r, self.scaling = r, alpha / r
        self.merged = False

        # 冻结的原始权重
        self.linear = nn.Linear(in_features, out_features)
        self.linear.weight.requires_grad_(False)
        self.linear.bias.requires_grad_(False)

        # LoRA 参数
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, r))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def merge_weights(self):
        """合并: W = W₀ + (α/r)·BA"""
        if not self.merged:
            self.linear.weight.data += self.scaling * (self.lora_B @ self.lora_A)
            self.merged = True

    def unmerge_weights(self):
        """反合并: W = W - (α/r)·BA"""
        if self.merged:
            self.linear.weight.data -= self.scaling * (self.lora_B @ self.lora_A)
            self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.linear(x)
        if not self.merged:
            lora_out = F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)
            h = h + self.scaling * lora_out
        return h


class LoRAMultiHeadAttention(nn.Module):
    """带 LoRA 的多头注意力，可选对 W_Q/W_K/W_V/W_O 应用 LoRA"""
    def __init__(self, d_model: int, num_heads: int, r: int = 8,
                 alpha: float = 16.0, target: List[str] = None):
        super().__init__()
        self.d_model, self.num_heads = d_model, num_heads
        self.d_k = d_model // num_heads
        target = target or ['q', 'v']  # 论文默认

        def make(name):
            if name in target:
                return LoRALinear(d_model, d_model, r, alpha)
            lin = nn.Linear(d_model, d_model)
            lin.weight.requires_grad_(False)
            lin.bias.requires_grad_(False)
            return lin

        self.W_Q, self.W_K = make('q'), make('k')
        self.W_V, self.W_O = make('v'), make('o')
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        B, T, d = x.shape
        H, dk = self.num_heads, self.d_k
        reshape = lambda t: t.view(B, T, H, dk).transpose(1, 2)

        Q, K, V = reshape(self.W_Q(x)), reshape(self.W_K(x)), reshape(self.W_V(x))
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, d)
        return self.W_O(out)


class LoRATransformerBlock(nn.Module):
    """带 LoRA 的 Transformer 块 (Pre-Norm)"""
    def __init__(self, d_model, num_heads, d_ff, r=8, alpha=16.0,
                 target=None, lora_ffn=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = LoRAMultiHeadAttention(d_model, num_heads, r, alpha, target)
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(0.1)

        # FFN（可选 LoRA）
        if lora_ffn:
            self.ff1 = LoRALinear(d_model, d_ff, r, alpha)
            self.ff2 = LoRALinear(d_ff, d_model, r, alpha)
        else:
            self.ff1 = nn.Linear(d_model, d_ff)
            self.ff2 = nn.Linear(d_ff, d_model)
            for p in list(self.ff1.parameters()) + list(self.ff2.parameters()):
                p.requires_grad_(False)

    def forward(self, x, mask=None):
        x = x + self.drop(self.attn(self.ln1(x), mask))
        x = x + self.drop(self.ff2(F.gelu(self.ff1(self.ln2(x)))))
        return x


class LoRATransformer(nn.Module):
    """
    带 LoRA 微调的 Transformer 语言模型

    冻结所有基础参数，只训练 LoRA 的 A, B 矩阵。
    训练后可合并权重实现零额外推理延迟。
    """
    def __init__(self, vocab_size=32000, d_model=768, num_heads=12,
                 num_layers=12, d_ff=3072, max_len=512,
                 r=8, alpha=16.0, target_modules=None, lora_ffn=False):
        super().__init__()
        self.d_model, self.vocab_size = d_model, vocab_size

        # 嵌入层（冻结）
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.tok_emb.weight.requires_grad_(False)
        self.pos_emb.weight.requires_grad_(False)
        self.drop = nn.Dropout(0.1)

        # Transformer 块
        self.blocks = nn.ModuleList([
            LoRATransformerBlock(d_model, num_heads, d_ff, r, alpha,
                                target_modules, lora_ffn)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # 权重共享

    def forward(self, input_ids, labels=None):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(pos))
        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)

        for block in self.blocks:
            x = block(x, mask)

        logits = self.lm_head(self.final_norm(x))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, self.vocab_size),
                labels[:, 1:].contiguous().view(-1), ignore_index=-100)
        return {"logits": logits, "loss": loss}

    def get_trainable_params(self):
        """返回 (可训练参数数, 总参数数)"""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total

    def merge_lora(self):
        for m in self.modules():
            if isinstance(m, LoRALinear):
                m.merge_weights()

    def unmerge_lora(self):
        for m in self.modules():
            if isinstance(m, LoRALinear):
                m.unmerge_weights()

    def save_lora(self, path):
        """仅保存 LoRA 参数（几 MB）"""
        state = {k: v for k, v in self.state_dict().items() if 'lora_' in k}
        torch.save(state, path)
        return len(state)

    def load_lora(self, path):
        """加载 LoRA 参数"""
        self.load_state_dict(torch.load(path, weights_only=True), strict=False)


# ========== 测试 ==========
if __name__ == "__main__":
    torch.manual_seed(42)
    import numpy as np

    V, d, H, L, d_ff = 1000, 128, 4, 4, 512
    r, alpha, B_size, T = 8, 16, 4, 32

    # 1. LoRALinear 单元测试
    print("=" * 50)
    print("1. LoRALinear 测试")
    lora_layer = LoRALinear(d, d, r=r, alpha=alpha)
    x = torch.randn(B_size, T, d)

    # 初始 ΔW = 0
    h_lora = lora_layer(x)
    h_base = lora_layer.linear(x)
    assert torch.allclose(h_lora, h_base, atol=1e-6)
    print(f"  ✓ 初始 ΔW=0，输出等价")

    # 合并/卸载
    lora_layer.lora_B.data.normal_(0, 0.01)
    h_before = lora_layer(x)
    lora_layer.merge_weights()
    h_merged = lora_layer(x)
    assert torch.allclose(h_before, h_merged, atol=1e-5)
    lora_layer.unmerge_weights()
    print(f"  ✓ 合并/卸载: 差异 = {(h_before - h_merged).abs().max():.2e}")

    # 2. 完整模型
    print(f"\n{'=' * 50}")
    print("2. LoRA Transformer 模型")
    model = LoRATransformer(V, d, H, L, d_ff, max_len=256,
                            r=r, alpha=alpha, target_modules=['q', 'v'])
    trainable, total = model.get_trainable_params()
    print(f"  总参数: {total:,}, 可训练: {trainable:,} ({trainable/total:.2%})")

    ids = torch.randint(0, V, (B_size, T))
    out = model(ids, ids)
    print(f"  Loss: {out['loss'].item():.4f}")

    # 3. 训练验证
    print(f"\n{'=' * 50}")
    print("3. 训练验证")
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=3e-4, weight_decay=0.01)
    losses = []
    for step in range(5):
        out = model(ids, ids)
        out['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad()
        losses.append(out['loss'].item())
    assert losses[-1] < losses[0]
    print(f"  ✓ Loss: {losses[0]:.4f} → {losses[-1]:.4f}")

    # 4. 权重合并
    model.eval()
    with torch.no_grad():
        h1 = model(ids)['logits']
    model.merge_lora()
    with torch.no_grad():
        h2 = model(ids)['logits']
    diff = (h1 - h2).abs().max().item()
    print(f"  ✓ 合并前后差异: {diff:.2e}")
    model.unmerge_lora()

    # 5. LoRA 保存/加载
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        path = f.name
    n = model.save_lora(path)
    size_kb = os.path.getsize(path) / 1024
    print(f"  ✓ 保存: {n} 张量, {size_kb:.1f} KB")
    os.unlink(path)

    # 6. ΔW 秩验证
    for name, m in model.named_modules():
        if isinstance(m, LoRALinear):
            dW = (m.scaling * m.lora_B @ m.lora_A).detach().numpy()
            print(f"  ✓ {name}: rank(ΔW)={np.linalg.matrix_rank(dW)} (≤{r})")
            break

    print(f"\n✅ LoRA Transformer 测试通过！")
```

---

## 7. 实践技巧与可视化

### 7.1 低秩近似可视化

```python
import numpy as np
import matplotlib.pyplot as plt


def plot_lora_analysis():
    """LoRA 核心概念可视化"""
    np.random.seed(42)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (1) 奇异值衰减对比
    W_random = np.random.randn(256, 256)
    W_lowrank = np.random.randn(256, 8) @ np.random.randn(8, 256) + 0.1 * np.random.randn(256, 256)
    _, s1, _ = np.linalg.svd(W_random)
    _, s2, _ = np.linalg.svd(W_lowrank)
    axes[0].semilogy(s1[:50], 'b-o', ms=3, label='Random')
    axes[0].semilogy(s2[:50], 'r-s', ms=3, label='Low-rank + noise')
    axes[0].axvline(x=8, color='green', ls='--', alpha=0.7, label='True rank=8')
    axes[0].set(xlabel='Index', ylabel='$\\sigma_i$', title='Singular Value Decay')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # (2) 参数效率对比
    methods = ['Full FT', 'Adapter', 'Prefix', 'BitFit', 'LoRA']
    params = [100, 3.6, 0.1, 0.08, 0.02]
    perf = [100, 97, 93, 90, 97]
    colors = ['#2196F3', '#FF9800', '#9C27B0', '#4CAF50', '#F44336']
    for m, p, pf, c in zip(methods, params, perf, colors):
        axes[1].scatter(p, pf, s=300, c=c, alpha=0.8, edgecolors='k', lw=1.5, zorder=5)
        axes[1].annotate(m, (p, pf), textcoords="offset points",
                         xytext=(0, 12), ha='center', fontsize=9, fontweight='bold')
    axes[1].set_xscale('log')
    axes[1].set(xlabel='Trainable Params (%)', ylabel='Relative Perf (%)',
                title='Parameter Efficiency', xlim=(0.01, 200), ylim=(85, 105))
    axes[1].grid(True, alpha=0.3)

    # (3) 秩 vs 性能
    ranks = [1, 2, 4, 8, 16, 32, 64]
    acc = [73.4, 73.3, 73.7, 73.7, 73.6, 73.5, 73.4]
    axes[2].plot(ranks, acc, 'b-o', lw=2, ms=8)
    axes[2].axhspan(73.2, 73.8, color='blue', alpha=0.05)
    axes[2].set_xscale('log', base=2)
    axes[2].set(xlabel='Rank $r$', ylabel='WikiSQL Acc (%)',
                title='Rank vs Performance (GPT-3 175B)')
    axes[2].set_xticks(ranks)
    axes[2].set_xticklabels([str(r) for r in ranks])
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("lora_analysis.png", dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    plot_lora_analysis()
```

### 7.2 实践调参建议

**LoRA 超参数速查表**：

| 超参数 | 推荐值 | 说明 |
|--------|:------:|------|
| 秩 $r$ | 8~16 | 小任务 4，复杂任务 32+ |
| $\alpha$ | $2r$ | 常见 $\alpha = 16$ 配 $r = 8$ |
| Dropout | 0.05~0.1 | LoRA 路径的 Dropout |
| 学习率 | $1\text{e-}4$ ~ $3\text{e-}4$ | 比全量微调大 5~10 倍 |
| 应用层 | $W_Q, W_V$ | 基础配置；全层更优 |

**不同模型规模的推荐配置**：

$$
\boxed{
\begin{aligned}
\text{7B:} &\quad r = 8\text{-}16, \; \alpha = 16\text{-}32 \\
\text{13B:} &\quad r = 16\text{-}32, \; \alpha = 32\text{-}64 \\
\text{70B:} &\quad r = 16\text{-}64, \; \alpha = 32\text{-}128 \\
\text{175B:} &\quad r = 8, \; \alpha = 16
\end{aligned}
}
$$

**常见问题与解决方案**：

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| 效果差 | $r$ 太小 | 增大 $r$，应用到更多层 |
| 训练不稳定 | 学习率过大 | 降低 lr，增大 warmup |
| 过拟合 | 数据少 | 减小 $r$，增大 dropout |
| 忘记预训练知识 | $\alpha/r$ 太大 | 减小 $\alpha$ |
| 推理慢 | 未合并权重 | 调用 `merge_weights()` |

---

## 8. 与其他模型的关系

### 8.1 参数高效微调方法谱系

```
参数高效微调 (PEFT)
├── 选择性微调 (Selective)
│   ├── BitFit (2021) ── 只微调 bias
│   └── Diff Pruning (2021) ── 稀疏化 ΔW
├── 附加模块 (Additive)
│   ├── Adapter (2019) ── 层间插入小模块
│   ├── Prefix-Tuning (2021) ── 可学习前缀
│   └── Prompt Tuning (2021) ── 软提示
└── 重参数化 (Reparameterization)
    ├── LoRA (2021) ── ΔW = BA  ← 本篇
    ├── QLoRA (2023) ── 4-bit + LoRA
    ├── DoRA (2024) ── 分解方向和幅度
    └── LoRA+ (2024) ── 不等学习率
```

### 8.2 LoRA 在大模型发展中的定位

$$
\boxed{\underbrace{\text{全量微调}}_{\text{资源门槛高}} \xrightarrow{\text{LoRA}} \underbrace{\text{参数高效微调}}_{\text{单卡可行}} \xrightarrow{\text{QLoRA}} \underbrace{\text{消费级微调}}_{\text{人人可用}}}
$$

LoRA 的核心贡献：将大模型微调从"工业级基础设施"降维到"个人研究者可行"。

### 8.3 LoRA 的后续发展

```
LoRA (2021) ── 低秩适应，r=8 即可
  ├── AdaLoRA (2023) ── 自适应秩分配（不同层不同 r）
  ├── QLoRA (2023) ── 4-bit NF4 量化 + LoRA
  ├── LoRA-FA (2023) ── 冻结 A，只训练 B
  ├── DoRA (2024) ── 分解方向/幅度
  ├── LoRA+ (2024) ── A, B 不等学习率
  └── rsLoRA (2024) ── 缩放因子 α/√r
```

| LoRA 遗留问题 | 后续解决方案 |
|:------------:|:----------:|
| 显存需全精度基础模型 | QLoRA: 4-bit 量化 |
| 所有层相同秩 $r$ | AdaLoRA: 自适应秩 |
| A, B 相同学习率 | LoRA+: 不等学习率 |
| 与全量微调有差距 | DoRA: 方向/幅度分解 |
| 大 $r$ 效果退化 | rsLoRA: $\alpha/\sqrt{r}$ |

---

## 扩展阅读与实现

### 问题 1：LoRA 的低秩假设何时失效？

当目标任务与预训练分布**差距极大**时（如英文→中文、文本→代码），$\Delta W$ 的有效秩可能较高。解决方案：(1) 增大 $r$（64~128）；(2) 对所有线性层应用 LoRA；(3) LoRA + 部分层解冻的混合策略。实验发现即使 $r=64$，LoRA 仍远比全量微调高效。

### 问题 2：LoRA 与 Dropout 的交互

后续实践发现在 LoRA 路径上添加 Dropout 可缓解过拟合：$h = W_0 x + \frac{\alpha}{r} B \cdot A \cdot \text{Dropout}(x)$。Dropout 只作用于 LoRA 路径输入，不影响主路径 $W_0 x$，确保预训练知识不被扰动。

### 问题 3：多任务 LoRA 的组合

多个 LoRA 适配器可线性组合：$W_{\text{combined}} = W_0 + \sum_t \lambda_t \frac{\alpha_t}{r_t} B_t A_t$。支持：(1) **任务算术**：混合不同能力；(2) **LoRA 融合**：合并多个适配器；(3) **模型汤**：平均多个检查点。

### 问题 4：为什么 $BA$ 而不是 $AB$？

计算路径 $x \xrightarrow{A} \mathbb{R}^r \xrightarrow{B} \mathbb{R}^d$ 是**先降维再升维**的瓶颈结构（类似 Autoencoder）。$A$ 将输入压缩到 $r$ 维"适应空间"，$B$ 投影回模型维度。这让 LoRA 自动学习最重要的 $r$ 个适应方向。

### 问题 5：LoRA 与全量微调的梯度对比

全量 $\Delta W_{\text{full}}$ 无秩约束，LoRA 的 $\Delta W = BA$ 约束在秩 $r$ 子空间。可以证明 $\Delta W_{\text{LoRA}} \approx \text{Proj}_r(\Delta W_{\text{full}})$。当 $\Delta W_{\text{full}}$ 奇异值快速衰减时，LoRA 近乎无损。

---

## 参考资源

### 经典论文

1. Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). ICLR 2022.
   - **贡献**：提出低秩适应方法，可训练参数减少 10,000 倍，零推理延迟

2. Aghajanyan, A., Gupta, S., & Zettlemoyer, L. (2020). [Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning](https://arxiv.org/abs/2012.13255). ACL 2021.
   - **贡献**：证明预训练模型具有极低内在维度，为 LoRA 提供理论基础

3. Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). [QLoRA: Efficient Finetuning of Quantized Language Models](https://arxiv.org/abs/2305.14314). NeurIPS 2023.
   - **贡献**：4-bit 量化 + LoRA，65B 模型单卡可微调

4. Houlsby, N., Giurgiu, A., Jastrzebski, S., et al. (2019). [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751). ICML 2019.
   - **贡献**：提出 Adapter 方法，参数高效微调的先驱

5. Li, X. L., & Liang, P. (2021). [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190). ACL 2021.
   - **贡献**：可训练前缀实现参数高效微调

6. Liu, S., Wang, C., Yin, H., et al. (2024). [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353). ICML 2024.
   - **贡献**：分解权重方向和幅度，缩小 LoRA 与全量微调的差距

### 教材与书籍

7. Eckart, C. & Young, G. (1936). The approximation of one matrix by another of lower rank. Psychometrika.
   - **章节**：低秩近似最优性定理的原始证明

### 在线资源与教程

8. Hugging Face. [PEFT: Parameter-Efficient Fine-Tuning](https://huggingface.co/docs/peft).
   - **内容**：LoRA/QLoRA 的官方实现库和教程

9. Sebastian Raschka. [Practical Tips for Finetuning LLMs Using LoRA](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms).
   - **内容**：LoRA 超参数选择和实践经验的详细指南

---

## 附录：符号表

| 符号 | 含义 | 维度/类型 |
|------|------|----------|
| $d$ ($d_{\text{model}}$) | 模型隐藏维度 | 标量 |
| $k$ | 线性层输入维度 | 标量 |
| $d_{ff}$ | FFN 隐藏层维度 | 标量，通常 $4d$ |
| $d_k$ | 每个注意力头的维度 | 标量 |
| $n_h$ | 注意力头数 | 标量 |
| $L$ | Transformer 层数 | 标量 |
| $r$ | LoRA 秩 | 标量，通常 $1 \sim 64$ |
| $\alpha$ | LoRA 缩放超参数 | 标量，通常 $= r$ 或 $= 2r$ |
| $\|V\|$ | 词表大小 | 标量 |
| $x$ | 线性层输入 | $(k,)$ 或 $(B, T, k)$ |
| $h$ | 线性层输出 | $(d,)$ 或 $(B, T, d)$ |
| $W_0$ | 预训练冻结权重 | $(d, k)$ |
| $\Delta W$ | 权重更新量 | $(d, k)$ |
| $A$ | LoRA 下投影矩阵 | $(r, k)$ |
| $B$ | LoRA 上投影矩阵 | $(d, r)$ |
| $W_Q, W_K, W_V, W_O$ | 注意力投影矩阵 | $(d, d)$ |
| $\sigma_i$ | 矩阵第 $i$ 个奇异值 | 标量，$\sigma_1 \geq \sigma_2 \geq \cdots$ |
| $U, V$ | SVD 左/右奇异向量矩阵 | $(m, m)$, $(n, n)$ |
| $\mathcal{L}$ | 损失函数值 | 标量 |
| $\ell(\cdot, \cdot)$ | 损失函数 | 函数 |
| $\eta$ | 学习率 | 标量 |
| $\|\cdot\|_F$ | Frobenius 范数 | 标量 |
| $\text{rank}(\cdot)$ | 矩阵秩 | 标量 |

**典型维度示例（GPT-3 175B，LoRA $r=8$）：**
- $d = 12{,}288$，$d_{ff} = 49{,}152$，$d_k = 128$
- $L = 96$，$n_h = 96$，$r = 8$，$\alpha = 16$
- $|V| = 50{,}257$
- 可训练参数 $\approx 37.7\text{M}$（占比 $0.02\%$）
- LoRA 检查点 $\approx 75\text{MB}$（全量 $\approx 350\text{GB}$）

---

最后更新：2026-03-19
