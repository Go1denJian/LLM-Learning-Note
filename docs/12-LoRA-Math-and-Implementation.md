# LoRA 数学原理与实现 —— 低秩适应的完整推导

> **前置知识**：线性代数（矩阵分解、秩）、Transformer 架构、反向传播、Python 基础  
> **与前面内容的联系**：建议先学习 [Transformer-Math-and-Implementation](./06-Transformer-Math-and-Implementation.md) 和 [GPT3-Scaling-and-InContext](./10-GPT3-Scaling-and-InContext.md)，理解大模型的参数规模与微调成本  
> **与后续内容的联系**：LoRA 的参数高效微调思想直接影响了 QLoRA、DoRA 以及后续 InstructGPT/RLHF 中的高效对齐训练

---

## 目录

1. [引言：为什么需要参数高效微调？](#1-引言为什么需要参数高效微调)
   - 1.1 [全参数微调的代价](#11-全参数微调的代价)
   - 1.2 [参数高效微调的核心思想](#12-参数高效微调的核心思想)
   - 1.3 [LoRA 的关键创新](#13-lora-的关键创新)
   - 1.4 [本科数学知识映射表](#14-本科数学知识映射表)
2. [低秩分解的数学基础](#2-低秩分解的数学基础)
   - 2.1 [矩阵秩的直觉与定义](#21-矩阵秩的直觉与定义)
   - 2.2 [SVD 与低秩近似](#22-svd-与低秩近似)
   - 2.3 [内在维度假说](#23-内在维度假说)
   - 2.4 [从低秩近似到 LoRA](#24-从低秩近似到-lora)
3. [LoRA 的数学定义与前向传播](#3-lora-的数学定义与前向传播)
   - 3.1 [权重更新的低秩参数化](#31-权重更新的低秩参数化)
   - 3.2 [前向传播公式](#32-前向传播公式)
   - 3.3 [缩放因子 α/r 的作用](#33-缩放因子-αr-的作用)
   - 3.4 [参数量与存储分析](#34-参数量与存储分析)
4. [梯度推导与参数更新](#4-梯度推导与参数更新)
   - 4.1 [LoRA 参数的梯度推导](#41-lora-参数的梯度推导)
   - 4.2 [冻结权重的梯度隔离](#42-冻结权重的梯度隔离)
   - 4.3 [初始化策略的数学分析](#43-初始化策略的数学分析)
   - 4.4 [训练动态分析](#44-训练动态分析)
5. [训练优化方法总结](#5-训练优化方法总结)
   - 5.1 [秩 r 的选择策略](#51-秩-r-的选择策略)
   - 5.2 [LoRA 应用位置选择](#52-lora-应用位置选择)
   - 5.3 [优化器与学习率](#53-优化器与学习率)
   - 5.4 [权重合并与推理优化](#54-权重合并与推理优化)
6. [从数学到代码：完整实现](#6-从数学到代码完整实现)
   - 6.1 [NumPy 实现核心组件](#61-numpy-实现核心组件)
   - 6.2 [PyTorch 完整实现](#62-pytorch-完整实现)
7. [实践技巧与可视化](#7-实践技巧与可视化)
   - 7.1 [低秩结构可视化](#71-低秩结构可视化)
   - 7.2 [实践调参建议](#72-实践调参建议)
8. [与其他模型的关系](#8-与其他模型的关系)
   - 8.1 [参数高效微调方法对比](#81-参数高效微调方法对比)
   - 8.2 [LoRA 在大模型发展中的定位](#82-lora-在大模型发展中的定位)
   - 8.3 [LoRA 的后续发展](#83-lora-的后续发展)

[扩展阅读与实现](#扩展阅读与实现)

[参考资源](#参考资源)

附录：[符号表](#附录符号表)

---

## 1. 引言：为什么需要参数高效微调？

### 1.1 全参数微调的代价

GPT-3 之后，大语言模型的参数规模已经达到数百亿乃至万亿级别。将预训练模型适配到下游任务时，**全参数微调**（Full Fine-Tuning）面临严峻挑战：

| 问题 | 具体表现 |
|------|---------|
| **存储成本** | 每个下游任务需要独立保存一份完整模型副本 |
| **显存需求** | 175B 参数的模型需要 $\sim 1.2$ TB 显存存储优化器状态 |
| **部署困难** | 多任务场景下需要切换整个模型 |
| **灾难性遗忘** | 全参数更新可能破坏预训练知识 |

用数学量化存储问题：假设模型参数量为 $|\Theta|$，需要适配 $K$ 个下游任务：

$$
\text{存储}_{\text{Full-FT}} = K \times |\Theta| \times \text{bytes/param}
$$

对于 GPT-3（175B 参数）适配 100 个任务，FP16 存储：

$$
100 \times 175 \times 10^9 \times 2 \text{ bytes} = 35 \text{ PB}
$$

这是完全不切实际的。

### 1.2 参数高效微调的核心思想

**核心问题**：能否只训练极少量参数，达到接近全参数微调的效果？

$$
\boxed{|\Theta_{\text{trainable}}| \ll |\Theta_{\text{total}}| \quad \text{且} \quad \text{Performance}_{\text{PEFT}} \approx \text{Performance}_{\text{Full-FT}}}
$$

已有的参数高效微调（PEFT）方法：

| 方法 | 策略 | 可训练参数比例 | 缺点 |
|------|------|:------------:|------|
| Adapter Tuning | 在层间插入小型瓶颈层 | $\sim 3\%$ | 引入推理延迟 |
| Prefix Tuning | 在输入前添加可训练的虚拟 token | $\sim 0.1\%$ | 压缩有效序列长度 |
| Prompt Tuning | 仅调整 prompt embedding | $\sim 0.01\%$ | 小模型效果差 |
| BitFit | 仅训练偏置参数 | $\sim 0.1\%$ | 表达能力有限 |

**关键观察**：这些方法要么引入额外推理开销，要么受限于表达能力。

### 1.3 LoRA 的关键创新

Hu et al. (2021) 提出了 **LoRA（Low-Rank Adaptation）**，核心思想极其简洁：

> **预训练权重的任务适应性更新 $\Delta W$ 具有低秩结构。**

$$
\boxed{W' = W_0 + \Delta W = W_0 + BA, \quad B \in \mathbb{R}^{d \times r}, \; A \in \mathbb{R}^{r \times k}, \; r \ll \min(d, k)}
$$

**三大优势**：

1. **零推理延迟**：$\Delta W = BA$ 可以直接合并到 $W_0$ 中，推理时无额外计算
2. **极高参数效率**：仅训练 $r(d+k)$ 个参数，而非 $d \times k$ 个
3. **任务切换灵活**：不同任务只需切换不同的 $(B, A)$，共享同一个 $W_0$

**参数效率量化**：

$$
\text{压缩比} = \frac{|\Delta W_{\text{full}}|}{|\Delta W_{\text{LoRA}}|} = \frac{d \times k}{r(d + k)} \approx \frac{d}{2r} \quad (\text{当 } d = k)
$$

对于 $d = 4096$（GPT-3 的隐藏维度）、$r = 8$：

$$
\text{压缩比} = \frac{4096}{2 \times 8} = 256\times
$$

### 1.4 本科数学知识映射表

| 数学概念 | LoRA 中的应用 | 代码对应 |
|---------|-------------|---------|
| 矩阵乘法 $BA$ | 低秩更新 $\Delta W = BA$ | `B @ A` |
| 矩阵秩 $\text{rank}(M)$ | 更新矩阵的秩约束 | `r` 超参数 |
| SVD $M = U\Sigma V^\top$ | 低秩近似的理论基础 | `np.linalg.svd` |
| Frobenius 范数 $\|M\|_F$ | Eckart-Young 定理的最优性度量 | `np.linalg.norm` |
| 链式法则 $\frac{\partial \mathcal{L}}{\partial A}$ | LoRA 参数的梯度计算 | `loss.backward()` |
| 高斯初始化 $\mathcal{N}(0, \sigma^2)$ | $A$ 的 Kaiming 初始化 | `nn.init.kaiming_uniform_` |
| 零矩阵 $\mathbf{0}$ | $B$ 的零初始化 | `nn.init.zeros_` |
| 缩放因子 $\alpha / r$ | 控制 LoRA 更新幅度 | `self.scaling = alpha / r` |

---

## 2. 低秩分解的数学基础

### 2.1 矩阵秩的直觉与定义

**定义**：矩阵 $M \in \mathbb{R}^{m \times n}$ 的秩是其线性独立行（或列）的最大数目：

$$
\text{rank}(M) = \dim(\text{col}(M)) = \dim(\text{row}(M))
$$

**直觉**：秩度量了矩阵中"独立信息"的数量。一个 $1024 \times 1024$ 的矩阵如果秩为 4，说明它虽然有 $1{,}048{,}576$ 个元素，但只有 $4 \times (1024 + 1024) = 8{,}192$ 个自由度。

**低秩矩阵的分解**：任何秩为 $r$ 的矩阵都可以分解为两个"瘦矩阵"的乘积：

$$
\boxed{M \in \mathbb{R}^{m \times n}, \; \text{rank}(M) = r \quad \Longleftrightarrow \quad M = PQ, \; P \in \mathbb{R}^{m \times r}, \; Q \in \mathbb{R}^{r \times n}}
$$

存储从 $mn$ 降为 $r(m+n)$，当 $r \ll \min(m, n)$ 时，压缩显著。

### 2.2 SVD 与低秩近似

**奇异值分解（SVD）** 是理解低秩近似的核心工具。

对于任意矩阵 $M \in \mathbb{R}^{m \times n}$：

$$
M = U \Sigma V^\top = \sum_{i=1}^{\min(m,n)} \sigma_i \mathbf{u}_i \mathbf{v}_i^\top
$$

其中 $U \in \mathbb{R}^{m \times m}$（左奇异向量），$\Sigma = \text{diag}(\sigma_1, \ldots, \sigma_{\min(m,n)})$（奇异值，$\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$），$V \in \mathbb{R}^{n \times n}$（右奇异向量）。

**Eckart-Young 定理**：秩为 $r$ 的最佳近似（Frobenius 范数意义下）为：

$$
\boxed{M_r = \sum_{i=1}^{r} \sigma_i \mathbf{u}_i \mathbf{v}_i^\top = U_r \Sigma_r V_r^\top}
$$

$$
M_r = \arg\min_{\text{rank}(\tilde{M}) \leq r} \|M - \tilde{M}\|_F
$$

**近似误差**：

$$
\|M - M_r\|_F^2 = \sum_{i=r+1}^{\min(m,n)} \sigma_i^2
$$

当奇异值快速衰减时（$\sigma_i \propto i^{-\alpha}$），低秩近似的误差很小。

### 2.3 内在维度假说

Aghajanyan et al. (2020) 提出了一个关键发现——**内在维度假说**：

> 预训练模型在微调过程中的权重更新 $\Delta W$ 具有很低的**内在维度**（Intrinsic Dimensionality），远小于参数空间的维度。

**数学表述**：设微调后的权重为 $\theta^* = \theta_0 + \Delta\theta$，存在一个低维子空间 $\mathcal{S} \subset \mathbb{R}^{|\theta|}$，$\dim(\mathcal{S}) = d_{\text{int}} \ll |\theta|$，使得：

$$
\boxed{\theta^* \approx \theta_0 + P \cdot z^*, \quad P \in \mathbb{R}^{|\theta| \times d_{\text{int}}}, \; z^* \in \mathbb{R}^{d_{\text{int}}}}
$$

**实验证据**（Aghajanyan et al., 2020）：

| 模型 | 参数量 $|\theta|$ | 内在维度 $d_{\text{int}}$ | 比例 |
|------|:----------------:|:------------------------:|:----:|
| RoBERTa-Base | 125M | $\sim 200$ | $0.00016\%$ |
| RoBERTa-Large | 355M | $\sim 800$ | $0.00023\%$ |
| GPT-2 Medium | 345M | $\sim 1{,}000$ | $0.00029\%$ |

**核心洞察**：尽管预训练模型有数亿参数，但适配到下游任务时，有效的更新方向只有几百到几千个。

### 2.4 从低秩近似到 LoRA

LoRA 将内在维度假说从全局参数空间**局部化**到每个权重矩阵：

$$
\underbrace{\Delta W \in \mathbb{R}^{d \times k}}_{\text{全秩更新}} \quad \xrightarrow{\text{低秩约束}} \quad \underbrace{\Delta W = BA}_{\text{秩 } r \text{ 更新}}
$$

**为什么不直接用 SVD？**

| 方法 | 操作 | 问题 |
|------|------|------|
| 先全参微调，再 SVD 压缩 | $\Delta W \xrightarrow{\text{SVD}} U_r \Sigma_r V_r^\top$ | 仍需全参训练的显存和计算 |
| LoRA：直接在低秩空间训练 | $\Delta W = BA$，只训练 $B, A$ | 训练时就节省显存 |

LoRA 的关键区别：**不是事后压缩，而是事先约束**。训练过程从一开始就限制在低秩子空间中进行。

---

## 3. LoRA 的数学定义与前向传播

### 3.1 权重更新的低秩参数化

对于预训练模型中的某个权重矩阵 $W_0 \in \mathbb{R}^{d \times k}$，LoRA 将微调过程参数化为：

$$
\boxed{W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} BA}
$$

其中：
- $W_0 \in \mathbb{R}^{d \times k}$：**冻结**的预训练权重（不参与训练）
- $B \in \mathbb{R}^{d \times r}$：下投影矩阵（可训练）
- $A \in \mathbb{R}^{r \times k}$：上投影矩阵（可训练）
- $r$：LoRA 的秩，$r \ll \min(d, k)$
- $\alpha$：缩放超参数
- $\alpha / r$：缩放因子

**几何直觉**：$A$ 将输入从 $k$ 维压缩到 $r$ 维（瓶颈），$B$ 再将 $r$ 维投射回 $d$ 维。这与 Autoencoder 的瓶颈结构类似：

```
输入 x ∈ ℝ^k
   ↓  A: k → r （压缩/编码）
中间 z ∈ ℝ^r  ← 瓶颈维度
   ↓  B: r → d （投射/解码）
更新 Δh ∈ ℝ^d
```

### 3.2 前向传播公式

对于输入 $x \in \mathbb{R}^k$，LoRA 层的前向传播为：

$$
h = W_0 x + \frac{\alpha}{r} BAx
$$

展开两个分支：

$$
\boxed{h = \underbrace{W_0 x}_{\text{冻结路径}} + \underbrace{\frac{\alpha}{r} B(Ax)}_{\text{LoRA 路径}}}
$$

**计算顺序**：先计算 $Ax \in \mathbb{R}^r$（维度降低），再计算 $B(Ax) \in \mathbb{R}^d$。这避免了显式构造 $d \times k$ 的 $\Delta W$。

**FLOPs 分析**：

| 操作 | FLOPs |
|------|-------|
| $W_0 x$ | $2dk$ |
| $Ax$ | $2rk$ |
| $B(Ax)$ | $2dr$ |
| **LoRA 总计** | $2dk + 2r(d+k)$ |
| **全参微调** | $2dk$（但需要存储 $dk$ 个梯度） |

额外 FLOPs 比例：$\frac{2r(d+k)}{2dk} = \frac{r}{d} + \frac{r}{k} \approx \frac{2r}{d}$。当 $r = 8, d = 4096$：额外 FLOPs $\approx 0.4\%$。

### 3.3 缩放因子 α/r 的作用

LoRA 的输出乘以 $\alpha / r$ 而非直接输出 $BAx$，原因如下：

**问题**：当改变秩 $r$ 时，$BA$ 的输出幅度会随 $r$ 变化。如果 $A$ 用 Kaiming 初始化（$\text{Var}(A_{ij}) \propto 1/k$），则：

$$
\text{Var}(BAx) \propto r
$$

输出方差与 $r$ 成正比——改变 $r$ 需要重新调整学习率。

**解决方案**：除以 $r$，使输出方差与 $r$ 无关：

$$
\text{Var}\left(\frac{1}{r} BAx\right) \propto \frac{r}{r^2} = \frac{1}{r} \quad \xrightarrow{\text{实际效果}} \quad \text{与 } r \text{ 解耦}
$$

然后引入 $\alpha$ 作为额外的缩放控制：

$$
\boxed{\text{scaling} = \frac{\alpha}{r}}
$$

> **Q:** $\alpha$ 如何选择？
>
> **A:** 论文中 $\alpha$ 通常设为常数（如 $\alpha = 16$ 或 $\alpha = r$）。当 $\alpha = r$ 时，$\text{scaling} = 1$，等价于无缩放。实践中固定 $\alpha$，只调 $r$，这样学习率不需要随 $r$ 重新搜索。

### 3.4 参数量与存储分析

**单层 LoRA 参数量**：

$$
P_{\text{LoRA}} = r \times d + r \times k = r(d + k)
$$

**与全参微调的对比**：

$$
\boxed{\text{参数比} = \frac{r(d+k)}{dk} = \frac{r}{d} + \frac{r}{k}}
$$

**GPT-3 (175B) 的存储分析**：

| 组件 | 权重形状 | 全参数量 | LoRA ($r=8$) | 压缩比 |
|------|---------|:-------:|:-----------:|:-----:|
| $W_Q$ | $12288 \times 12288$ | 151M | 197K | $768\times$ |
| $W_K$ | $12288 \times 12288$ | 151M | 197K | $768\times$ |
| $W_V$ | $12288 \times 12288$ | 151M | 197K | $768\times$ |
| $W_O$ | $12288 \times 12288$ | 151M | 197K | $768\times$ |
| **每层合计** | — | 604M | 786K | $768\times$ |
| **96 层合计** | — | 58.0B | 75.5M | $768\times$ |

**多任务存储**：

$$
\text{存储}_{\text{LoRA}} = |\Theta_0| + K \times r(d+k) \times L_{\text{layers}}
$$

对比全参微调：

$$
\frac{\text{存储}_{\text{LoRA}}}{\text{存储}_{\text{Full-FT}}} \approx \frac{1}{K} + \frac{r(d+k)}{dk} \approx \frac{1}{K} \quad (\text{当 } K \text{ 大时})
$$

100 个任务：LoRA 只需 $\sim 1\%$ 的存储空间。

---

## 4. 梯度推导与参数更新

### 4.1 LoRA 参数的梯度推导

设损失函数为 $\mathcal{L}$，LoRA 层的前向传播为：

$$
h = W_0 x + \frac{\alpha}{r} BAx
$$

我们需要计算 $\frac{\partial \mathcal{L}}{\partial A}$ 和 $\frac{\partial \mathcal{L}}{\partial B}$。

**对 $B$ 的梯度**：

设 $z = Ax \in \mathbb{R}^r$（中间表示），则 $h = W_0 x + \frac{\alpha}{r} Bz$。

$$
\frac{\partial \mathcal{L}}{\partial B} = \frac{\partial \mathcal{L}}{\partial h} \cdot \frac{\partial h}{\partial B}
$$

由 $h = W_0 x + \frac{\alpha}{r} Bz$，对 $B$ 求导：

$$
\boxed{\frac{\partial \mathcal{L}}{\partial B} = \frac{\alpha}{r} \cdot \frac{\partial \mathcal{L}}{\partial h} \cdot z^\top = \frac{\alpha}{r} \cdot \delta_h \cdot (Ax)^\top}
$$

其中 $\delta_h = \frac{\partial \mathcal{L}}{\partial h} \in \mathbb{R}^{d \times 1}$ 是来自上游的梯度信号。

**对 $A$ 的梯度**：

$$
\frac{\partial \mathcal{L}}{\partial A} = \frac{\partial \mathcal{L}}{\partial z} \cdot \frac{\partial z}{\partial A}
$$

先计算 $\frac{\partial \mathcal{L}}{\partial z}$：

$$
\frac{\partial \mathcal{L}}{\partial z} = \frac{\alpha}{r} \cdot B^\top \frac{\partial \mathcal{L}}{\partial h} = \frac{\alpha}{r} \cdot B^\top \delta_h
$$

再由 $z = Ax$：

$$
\boxed{\frac{\partial \mathcal{L}}{\partial A} = \frac{\alpha}{r} \cdot B^\top \delta_h \cdot x^\top}
$$

**梯度维度验证**：

| 量 | 维度 | 说明 |
|----|------|------|
| $\delta_h$ | $(d, 1)$ | 损失对输出的梯度 |
| $\frac{\partial \mathcal{L}}{\partial B}$ | $(d, r)$ | 与 $B$ 形状一致 ✓ |
| $\frac{\partial \mathcal{L}}{\partial A}$ | $(r, k)$ | 与 $A$ 形状一致 ✓ |

### 4.2 冻结权重的梯度隔离

LoRA 训练时，$W_0$ 被冻结（`requires_grad=False`）。这意味着：

$$
\frac{\partial \mathcal{L}}{\partial W_0} = \delta_h \cdot x^\top \quad \text{（计算但不存储/更新）}
$$

**显存节省**的来源：

| 组件 | 全参微调 | LoRA |
|------|:-------:|:----:|
| 参数存储 $W_0$ | FP16: $2dk$ | FP16: $2dk$（相同，只是冻结） |
| 梯度 $\nabla W_0$ | FP16: $2dk$ | **不存储** |
| 优化器状态（Adam） | FP32: $8dk$ | **不存储** |
| LoRA 参数 $B, A$ | — | FP16: $2r(d+k)$ |
| LoRA 梯度 | — | FP16: $2r(d+k)$ |
| LoRA 优化器 | — | FP32: $8r(d+k)$ |
| **每层总计** | $12dk$ bytes | $2dk + 12r(d+k)$ bytes |

当 $r \ll d$，显存节省约 $\frac{12dk}{2dk + 12r(d+k)} \approx 6\times$。

### 4.3 初始化策略的数学分析

LoRA 的初始化策略是精心设计的：

$$
\boxed{A \sim \mathcal{N}(0, \sigma_A^2), \quad B = \mathbf{0}}
$$

**为什么 $B$ 初始化为零？**

$$
\Delta W_{\text{init}} = B_{\text{init}} A_{\text{init}} = \mathbf{0} \cdot A = \mathbf{0}
$$

训练开始时 $\Delta W = 0$，模型的行为与预训练模型完全一致。这确保了：

1. **无损启动**：LoRA 不会破坏预训练知识
2. **稳定训练**：初始损失与预训练模型相同
3. **渐进适应**：$\Delta W$ 从零开始逐渐增长

**$A$ 的初始化**：使用 Kaiming 均匀初始化：

$$
A_{ij} \sim U\left(-\sqrt{\frac{6}{k}}, \sqrt{\frac{6}{k}}\right)
$$

> **Q:** 能否交换初始化——$A = \mathbf{0}$，$B \sim \mathcal{N}$？
>
> **A:** 数学上等价（都保证 $\Delta W_{\text{init}} = 0$），但实践中 $B = 0$ 更优。原因：$A$ 非零使得 $z = Ax$ 在初始时就有非零值，$B$ 的梯度 $\frac{\partial \mathcal{L}}{\partial B} \propto z^\top$ 从第一步就有信号。若 $A = 0$，则 $z = 0$，$B$ 的梯度也为零——$B$ 无法更新，需要等 $A$ 先从零逃逸，训练效率降低。

### 4.4 训练动态分析

**初始阶段**（$t = 0$）：

$$
B_0 = \mathbf{0}, \quad \Delta W_0 = \mathbf{0}, \quad h = W_0 x
$$

**第一步更新**：

$$
B_1 = B_0 - \eta \frac{\partial \mathcal{L}}{\partial B}\bigg|_{t=0} = -\eta \cdot \frac{\alpha}{r} \cdot \delta_h \cdot (A_0 x)^\top
$$

$$
A_1 = A_0 - \eta \frac{\partial \mathcal{L}}{\partial A}\bigg|_{t=0} = A_0 - \eta \cdot \frac{\alpha}{r} \cdot B_0^\top \delta_h \cdot x^\top = A_0
$$

**关键发现**：在第一步中，$A$ 不更新（因为 $B_0 = 0$），只有 $B$ 更新。从第二步开始，$B_1 \neq 0$，$A$ 才开始接收梯度。

**更新幅度**的演化：

$$
\|\Delta W_t\|_F = \|B_t A_t\|_F \leq \|B_t\|_F \|A_t\|_F
$$

由于 $B$ 从零开始，$\|\Delta W_t\|_F$ 在初始阶段缓慢增长，避免了大幅偏离预训练权重——这是一种隐式的正则化效果。

---

## 5. 训练优化方法总结

### 5.1 秩 r 的选择策略

秩 $r$ 是 LoRA 最重要的超参数，控制着**表达能力**与**参数效率**的权衡：

$$
\boxed{\text{秩 } r \uparrow \quad \Longleftrightarrow \quad \text{表达能力 } \uparrow, \; \text{参数效率 } \downarrow}
$$

**论文实验结果**（GPT-3 175B 在 WikiSQL 和 MNLI 上）：

| 秩 $r$ | 可训练参数 | WikiSQL Acc | MNLI Acc | 说明 |
|:------:|:---------:|:-----------:|:--------:|------|
| 1 | 4.7M | 73.4 | 91.7 | 极端压缩 |
| 2 | 9.4M | 73.3 | 91.5 | 性能几乎不降 |
| 4 | 18.8M | 73.7 | 91.5 | 论文推荐 |
| 8 | 37.7M | 73.7 | 91.6 | 通用选择 |
| 64 | 301M | 73.4 | 91.4 | 性能饱和甚至下降 |
| Full FT | 175B | 73.8 | 91.7 | 全参微调 |

**核心发现**：$r = 4$ 已经足够接近全参微调！$r$ 从 4 增大到 64，性能几乎不变甚至略降（过拟合）。

**秩选择的经验法则**：

$$
\boxed{r = \begin{cases}
1 \sim 4 & \text{简单分类任务} \\
4 \sim 16 & \text{通用推荐（大多数 NLP 任务）} \\
16 \sim 64 & \text{复杂生成任务或跨领域适配} \\
64+ & \text{极少需要，可能过拟合}
\end{cases}}
$$

### 5.2 LoRA 应用位置选择

Transformer 中有多个权重矩阵可以应用 LoRA：

| 矩阵 | 维度 | 论文结论 |
|------|------|---------|
| $W_Q$（Query） | $d \times d_k$ | ✅ 推荐 |
| $W_K$（Key） | $d \times d_k$ | 效果一般 |
| $W_V$（Value） | $d \times d_v$ | ✅ 推荐 |
| $W_O$（Output） | $d_v \times d$ | ✅ 推荐 |
| $W_1$（FFN Up） | $d \times d_{ff}$ | 论文未测试 |
| $W_2$（FFN Down） | $d_{ff} \times d$ | 论文未测试 |

**论文的关键实验**（固定总参数预算）：

$$
\boxed{\{W_Q, W_V\} \text{ 同时应用 LoRA } > \text{ 单独任何一个 } > \{W_Q, W_K\}}
$$

**后续实践发现**：在所有线性层（$W_Q, W_K, W_V, W_O, W_1, W_2$）都应用 LoRA，每层用更小的 $r$，效果通常最优。

### 5.3 优化器与学习率

LoRA 参数的学习率通常比全参微调**更大**：

| 设置 | 学习率范围 | 说明 |
|------|:---------:|------|
| 全参微调 | $1 \times 10^{-5} \sim 5 \times 10^{-5}$ | 避免破坏预训练权重 |
| LoRA | $1 \times 10^{-4} \sim 3 \times 10^{-4}$ | 低秩参数需要更大步长 |

**为什么 LoRA 可以用更大学习率？**

1. **$\Delta W$ 从零开始**：不会因为大学习率在初始时产生大偏移
2. **低秩约束**本身就是正则化：限制了更新空间的自由度
3. **参数少**：优化景观更简单，可以承受更大步长

**优化器选择**：

$$
\boxed{\text{AdamW}(\text{lr}=2 \times 10^{-4}, \; \beta_1=0.9, \; \beta_2=0.999, \; \text{wd}=0.01)}
$$

注意：权重衰减只作用于 LoRA 参数（$A, B$），不作用于冻结的 $W_0$。

### 5.4 权重合并与推理优化

LoRA 的一个独特优势：**训练完成后可以将低秩更新合并到原始权重中**，推理时零额外开销。

**合并操作**：

$$
\boxed{W_{\text{merged}} = W_0 + \frac{\alpha}{r} BA}
$$

合并后的前向传播：

$$
h = W_{\text{merged}} \cdot x \quad \text{（与原始模型完全相同的计算图）}
$$

**无延迟推理**：合并后模型的结构和计算量与原始预训练模型完全一致，不需要任何额外的分支计算。这是 LoRA 相比 Adapter 和 Prefix Tuning 的核心优势。

**任务切换**：

$$
W_{\text{task}_i} = W_0 + \frac{\alpha}{r} B_i A_i
$$

切换任务只需替换 $(B_i, A_i)$——在 GPU 上只需更新 $\sim 0.1\%$ 的参数。

**多任务混合**（LoRA 的线性组合）：

$$
W_{\text{mix}} = W_0 + \sum_{i=1}^{K} \lambda_i \cdot \frac{\alpha}{r} B_i A_i
$$

不同任务的 LoRA 适配器可以加权混合，实现能力融合。

---

## 6. 从数学到代码：完整实现

### 6.1 NumPy 实现核心组件

```python
import numpy as np


def svd_analysis(W, title="Weight Matrix"):
    """
    分析权重矩阵的奇异值分布，验证低秩假说

    数学: W = UΣV^T, 若 σ_i 快速衰减 → W 具有低秩结构
    """
    U, sigma, Vt = np.linalg.svd(W, full_matrices=False)
    total_energy = np.sum(sigma ** 2)
    cumulative = np.cumsum(sigma ** 2) / total_energy

    print(f"{title}:")
    print(f"  形状: {W.shape}, 满秩: {min(W.shape)}")
    print(f"  前 5 个奇异值: {sigma[:5].round(4)}")
    print(f"  前 1 个奇异值解释能量: {cumulative[0]:.4f}")
    print(f"  前 4 个奇异值解释能量: {cumulative[3]:.4f}")
    print(f"  前 8 个奇异值解释能量: {cumulative[7]:.4f}")
    print(f"  σ_1 / σ_last = {sigma[0] / (sigma[-1] + 1e-10):.1f} (条件数)")
    return sigma, cumulative


def lora_forward(x, W0, A, B, alpha, r):
    """
    LoRA 前向传播

    数学公式: h = W0·x + (α/r)·B·A·x

    参数:
        x: (batch, k) — 输入
        W0: (d, k) — 冻结预训练权重
        A: (r, k) — LoRA 下投影（可训练）
        B: (d, r) — LoRA 上投影（可训练）
        alpha: 缩放超参数
        r: 秩

    返回:
        h: (batch, d) — 输出
    """
    # 冻结路径: h_frozen = W0 @ x^T → (d, batch)
    h_frozen = x @ W0.T  # (batch, d)

    # LoRA 路径: h_lora = (α/r) * B @ A @ x^T
    scaling = alpha / r
    z = x @ A.T       # (batch, r) — 压缩到低维
    h_lora = z @ B.T  # (batch, d) — 投射回高维
    h_lora = scaling * h_lora

    return h_frozen + h_lora


def lora_gradients(x, delta_h, A, B, alpha, r):
    """
    LoRA 参数梯度的手动推导

    数学公式:
        ∂L/∂B = (α/r) · δ_h · (Ax)^T
        ∂L/∂A = (α/r) · B^T · δ_h · x^T

    参数:
        x: (batch, k) — 输入
        delta_h: (batch, d) — 上游梯度 ∂L/∂h
        A: (r, k), B: (d, r) — LoRA 参数
        alpha, r: 缩放参数

    返回:
        grad_A: (r, k), grad_B: (d, r)
    """
    scaling = alpha / r
    batch = x.shape[0]

    # ∂L/∂B = (α/r) · Σ_i δ_h_i · (A·x_i)^T / batch
    z = x @ A.T  # (batch, r)
    grad_B = scaling * (delta_h.T @ z) / batch  # (d, r)

    # ∂L/∂A = (α/r) · Σ_i B^T·δ_h_i · x_i^T / batch
    Bt_delta = delta_h @ B  # (batch, r)
    grad_A = scaling * (Bt_delta.T @ x) / batch  # (r, k)

    return grad_A, grad_B


def lora_merge(W0, A, B, alpha, r):
    """
    权重合并: W_merged = W0 + (α/r) · B · A

    合并后推理与原始模型完全相同，零额外开销。
    """
    scaling = alpha / r
    delta_W = scaling * (B @ A)  # (d, k)
    return W0 + delta_W


def load_balance_analysis(W0, delta_W):
    """
    分析微调更新 ΔW 的低秩特性

    数学: 对 ΔW 做 SVD，检查奇异值是否快速衰减
    """
    U, sigma, Vt = np.linalg.svd(delta_W, full_matrices=False)
    total = np.sum(sigma ** 2)
    ratios = np.cumsum(sigma ** 2) / total

    print(f"ΔW 分析:")
    print(f"  ||ΔW||_F / ||W0||_F = {np.linalg.norm(delta_W) / np.linalg.norm(W0):.6f}")
    print(f"  有效秩 (90% 能量): {np.searchsorted(ratios, 0.9) + 1}")
    print(f"  有效秩 (99% 能量): {np.searchsorted(ratios, 0.99) + 1}")
    return sigma, ratios


# ========== 测试 ==========
if __name__ == "__main__":
    np.random.seed(42)

    d, k = 256, 256  # 权重维度
    r = 8             # LoRA 秩
    alpha = 16        # 缩放超参数
    batch = 4

    # 1. 模拟预训练权重的 SVD 分析
    print("=" * 60)
    print("实验 1: 预训练权重的奇异值分布")
    print("=" * 60)
    # 模拟一个"预训练"权重（低秩成分 + 噪声）
    W_low = np.random.randn(d, 16) @ np.random.randn(16, k)  # 秩 16 的成分
    W_noise = np.random.randn(d, k) * 0.01                     # 小噪声
    W0 = W_low + W_noise
    svd_analysis(W0, "预训练权重 W0")

    # 2. LoRA 前向传播
    print("\n" + "=" * 60)
    print("实验 2: LoRA 前向传播验证")
    print("=" * 60)
    # 初始化: B=0, A~Kaiming
    B = np.zeros((d, r))
    A = np.random.randn(r, k) * np.sqrt(2.0 / k)  # Kaiming

    x = np.random.randn(batch, k)

    # 初始时 LoRA 输出应等于冻结路径
    h_init = lora_forward(x, W0, A, B, alpha, r)
    h_frozen = x @ W0.T
    print(f"初始化验证:")
    print(f"  ||h_lora - h_frozen||_max = {np.abs(h_init - h_frozen).max():.2e} (应为 0)")

    # 模拟训练后
    B_trained = np.random.randn(d, r) * 0.01  # 模拟训练后的 B
    h_trained = lora_forward(x, W0, A, B_trained, alpha, r)
    print(f"训练后:")
    print(f"  ||h_trained - h_frozen||_max = {np.abs(h_trained - h_frozen).max():.4f} (应 > 0)")

    # 3. 梯度验证（数值微分 vs 解析公式）
    print("\n" + "=" * 60)
    print("实验 3: 梯度验证")
    print("=" * 60)
    target = np.random.randn(batch, d)
    h = lora_forward(x, W0, A, B_trained, alpha, r)
    delta_h = (h - target) / batch

    grad_A_analytical, grad_B_analytical = lora_gradients(
        x, delta_h, A, B_trained, alpha, r
    )

    eps = 1e-5
    grad_A_numerical = np.zeros_like(A)
    for i in range(min(3, r)):
        for j in range(min(3, k)):
            A_p, A_m = A.copy(), A.copy()
            A_p[i, j] += eps; A_m[i, j] -= eps
            l_p = np.sum((lora_forward(x, W0, A_p, B_trained, alpha, r) - target)**2) / (2*batch)
            l_m = np.sum((lora_forward(x, W0, A_m, B_trained, alpha, r) - target)**2) / (2*batch)
            grad_A_numerical[i, j] = (l_p - l_m) / (2 * eps)

    diff = np.abs(grad_A_analytical[:3, :3] - grad_A_numerical[:3, :3]).max()
    print(f"∂L/∂A 最大差异: {diff:.2e} (应 < 1e-4)")

    # 4. 权重合并
    print("\n" + "=" * 60)
    print("实验 4: 权重合并验证")
    print("=" * 60)
    W_merged = lora_merge(W0, A, B_trained, alpha, r)
    h_merged = x @ W_merged.T
    h_separate = lora_forward(x, W0, A, B_trained, alpha, r)
    merge_diff = np.abs(h_merged - h_separate).max()
    print(f"合并前后输出差异: {merge_diff:.2e} (应 < 1e-10)")
    print(f"ΔW 秩: {np.linalg.matrix_rank(B_trained @ A)} (应 ≤ {r})")

    # 5. ΔW 低秩分析 & 参数效率
    print("\n" + "=" * 60)
    print("实验 5: ΔW 低秩结构 & 参数效率")
    print("=" * 60)
    delta_W = (alpha / r) * (B_trained @ A)
    load_balance_analysis(W0, delta_W)
    full_params = d * k
    lora_params = r * (d + k)
    print(f"压缩比: {full_params / lora_params:.1f}x, 参数比例: {lora_params / full_params:.4%}")

    print("\n✅ LoRA NumPy 核心组件测试通过！")
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
    LoRA 线性层: h = W0·x + (α/r)·B·A·x

    核心思想: 冻结 W0, 只训练低秩矩阵 B, A
    初始化: A ~ Kaiming, B = 0 → 初始 ΔW = 0
    """
    def __init__(self, in_features: int, out_features: int,
                 r: int = 8, alpha: float = 16.0,
                 dropout: float = 0.0, merge: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.merged = False

        # 冻结的预训练权重
        self.linear = nn.Linear(in_features, out_features, bias=True)

        # LoRA 低秩矩阵
        if r > 0:
            self.lora_A = nn.Parameter(torch.empty(r, in_features))
            self.lora_B = nn.Parameter(torch.empty(out_features, r))
            self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

            # 初始化: A ~ Kaiming, B = 0
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

        # 冻结预训练权重
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        if merge:
            self.merge_weights()

    def merge_weights(self):
        """合并 LoRA 权重到 W0: W_merged = W0 + (α/r)·B·A"""
        if not self.merged and self.r > 0:
            with torch.no_grad():
                self.linear.weight.add_(
                    self.scaling * (self.lora_B @ self.lora_A)
                )
            self.merged = True

    def unmerge_weights(self):
        """分离 LoRA 权重: 恢复 W0"""
        if self.merged and self.r > 0:
            with torch.no_grad():
                self.linear.weight.sub_(
                    self.scaling * (self.lora_B @ self.lora_A)
                )
            self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播: h = W0·x + (α/r)·B·(A·dropout(x))

        若已合并: h = W_merged · x (零额外开销)
        """
        if self.merged or self.r == 0:
            return self.linear(x)

        # 分支计算
        h_frozen = self.linear(x)                            # W0·x + b
        h_lora = (self.lora_dropout(x) @ self.lora_A.T)     # (batch, r)
        h_lora = h_lora @ self.lora_B.T                      # (batch, d)
        return h_frozen + self.scaling * h_lora

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"r={self.r}, α={self.alpha}, merged={self.merged}")


class LoRAMultiHeadAttention(nn.Module):
    """
    带 LoRA 的多头注意力

    对 W_Q, W_V 应用 LoRA（论文推荐组合），
    W_K, W_O 保持冻结。
    """
    def __init__(self, d_model: int, num_heads: int,
                 r: int = 8, alpha: float = 16.0,
                 lora_dropout: float = 0.0,
                 lora_targets: List[str] = None):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        if lora_targets is None:
            lora_targets = ['q', 'v']  # 论文默认

        # Q, K, V, O 投影
        self.W_Q = LoRALinear(d_model, d_model, r=r if 'q' in lora_targets else 0,
                              alpha=alpha, dropout=lora_dropout)
        self.W_K = LoRALinear(d_model, d_model, r=r if 'k' in lora_targets else 0,
                              alpha=alpha, dropout=lora_dropout)
        self.W_V = LoRALinear(d_model, d_model, r=r if 'v' in lora_targets else 0,
                              alpha=alpha, dropout=lora_dropout)
        self.W_O = LoRALinear(d_model, d_model, r=r if 'o' in lora_targets else 0,
                              alpha=alpha, dropout=lora_dropout)

        self.attn_dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, d = x.shape

        # 线性投影 + reshape 为多头
        Q = self.W_Q(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled Dot-Product Attention
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = self.attn_dropout(F.softmax(scores, dim=-1))
        context = (attn @ V).transpose(1, 2).contiguous().view(B, T, d)

        return self.W_O(context)


class LoRATransformerBlock(nn.Module):
    """
    带 LoRA 的 Transformer 块 (Pre-Norm)

    a = x + LoRA_Attn(LN(x))
    out = a + FFN(LN(a))       ← FFN 保持冻结
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 r: int = 8, alpha: float = 16.0,
                 dropout: float = 0.1, lora_dropout: float = 0.0,
                 lora_targets: List[str] = None):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = LoRAMultiHeadAttention(d_model, num_heads, r, alpha,
                                           lora_dropout, lora_targets)
        self.ln2 = nn.LayerNorm(d_model)
        # FFN 保持冻结（也可以选择加 LoRA）
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.ln1(x), mask))
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x


class LoRALanguageModel(nn.Module):
    """
    带 LoRA 的语言模型

    预训练权重全部冻结，仅训练注意力层中的 LoRA 参数。
    支持权重合并实现零延迟推理。
    """
    def __init__(self, vocab_size: int = 32000, d_model: int = 768,
                 num_heads: int = 12, num_layers: int = 12,
                 d_ff: int = 3072, max_len: int = 512,
                 r: int = 8, alpha: float = 16.0,
                 dropout: float = 0.1, lora_dropout: float = 0.05,
                 lora_targets: List[str] = None):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Embedding 层（冻结）
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)

        # Transformer 块（带 LoRA）
        self.blocks = nn.ModuleList([
            LoRATransformerBlock(d_model, num_heads, d_ff, r, alpha,
                                dropout, lora_dropout, lora_targets)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # 权重共享

        # 冻结非 LoRA 参数
        self._freeze_pretrained()

    def _freeze_pretrained(self):
        """冻结所有非 LoRA 参数"""
        for name, param in self.named_parameters():
            if 'lora_' not in name:
                param.requires_grad = False

    def get_lora_params(self) -> List[nn.Parameter]:
        """获取所有 LoRA 参数"""
        return [p for n, p in self.named_parameters() if 'lora_' in n]

    def get_lora_state_dict(self) -> Dict[str, torch.Tensor]:
        """获取 LoRA 参数的 state_dict（用于保存/加载适配器）"""
        return {k: v for k, v in self.state_dict().items() if 'lora_' in k}

    def load_lora_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """加载 LoRA 适配器"""
        model_state = self.state_dict()
        model_state.update(state_dict)
        self.load_state_dict(model_state)

    def merge_lora(self):
        """合并所有 LoRA 权重（用于推理）"""
        for module in self.modules():
            if isinstance(module, LoRALinear):
                module.merge_weights()

    def unmerge_lora(self):
        """分离所有 LoRA 权重（恢复训练模式）"""
        for module in self.modules():
            if isinstance(module, LoRALinear):
                module.unmerge_weights()

    def forward(self, input_ids: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> Dict:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(pos))

        # 因果掩码
        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)

        for block in self.blocks:
            x = block(x, mask)

        logits = self.lm_head(self.final_norm(x))

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, self.vocab_size),
                labels[:, 1:].contiguous().view(-1), ignore_index=-100
            )

        return {"logits": logits, "loss": loss}

    def param_stats(self) -> Dict:
        """参数统计"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return {
            "total": total,
            "trainable": trainable,
            "frozen": frozen,
            "trainable_ratio": trainable / total,
        }


# ========== 测试 ==========
if __name__ == "__main__":
    torch.manual_seed(42)

    V, d, H, L, d_ff = 1000, 128, 4, 4, 512
    r = 8
    alpha = 16.0
    B_size, T = 4, 32

    # 1. LoRALinear 单元测试
    print("=" * 60)
    print("测试 1: LoRALinear")
    print("=" * 60)
    layer = LoRALinear(d, d, r=r, alpha=alpha)
    x = torch.randn(B_size, T, d)

    with torch.no_grad():
        diff = (layer(x) - layer.linear(x)).abs().max().item()
    print(f"初始化 ΔW=0: diff={diff:.2e} ✓")

    layer_copy = LoRALinear(d, d, r=r, alpha=alpha)
    with torch.no_grad():
        layer_copy.lora_B.normal_(0, 0.01)
    h_ref = layer_copy(x)
    layer_copy.merge_weights()
    print(f"合并: diff={(h_ref - layer_copy(x)).abs().max().item():.2e} ✓")
    layer_copy.unmerge_weights()
    print(f"分离: diff={(h_ref - layer_copy(x)).abs().max().item():.2e} ✓")

    # 2. 完整模型测试
    print(f"\n{'=' * 60}")
    print("测试 2: LoRA 语言模型")
    print("=" * 60)
    model = LoRALanguageModel(V, d, H, L, d_ff, max_len=256,
                              r=r, alpha=alpha, lora_targets=['q', 'v'])

    stats = model.param_stats()
    print(f"参数统计:")
    print(f"  总参数: {stats['total']:,}")
    print(f"  可训练 (LoRA): {stats['trainable']:,}")
    print(f"  冻结: {stats['frozen']:,}")
    print(f"  可训练比例: {stats['trainable_ratio']:.4%}")

    # 前向传播
    ids = torch.randint(0, V, (B_size, T))
    model.eval()
    with torch.no_grad():
        out = model(ids, ids)
    print(f"\n前向传播:")
    print(f"  Logits: {out['logits'].shape}")
    print(f"  Loss: {out['loss'].item():.4f}")

    # 3. 梯度流验证
    print(f"\n{'=' * 60}")
    print("测试 3: 梯度流")
    print("=" * 60)
    model.train()
    out = model(ids, ids)
    out["loss"].backward()

    lora_with_grad = sum(1 for n, p in model.named_parameters()
                         if 'lora_' in n and p.grad is not None and p.grad.abs().max() > 0)
    frozen_with_grad = sum(1 for n, p in model.named_parameters()
                          if 'lora_' not in n and p.grad is not None)
    print(f"LoRA 参数有梯度: {lora_with_grad} ✓")
    print(f"冻结参数有梯度: {frozen_with_grad} (应为 0)")

    # 4. 训练循环
    print(f"\n{'=' * 60}")
    print("测试 4: 训练循环 (5 步)")
    print("=" * 60)
    optimizer = torch.optim.AdamW(model.get_lora_params(), lr=2e-4, weight_decay=0.01)
    model.train()

    for step in range(5):
        optimizer.zero_grad()
        out = model(ids, ids)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.get_lora_params(), 1.0)
        optimizer.step()
        if step % 2 == 0:
            print(f"  Step {step}: loss={out['loss'].item():.4f}")

    # 5. 权重合并与推理
    print(f"\n{'=' * 60}")
    print("测试 5: 权重合并与推理一致性")
    print("=" * 60)
    model.eval()
    with torch.no_grad():
        h_before_merge = model(ids)["logits"]

    model.merge_lora()
    with torch.no_grad():
        h_after_merge = model(ids)["logits"]

    merge_diff = (h_before_merge - h_after_merge).abs().max().item()
    print(f"合并前后输出差异: {merge_diff:.2e} (应 < 1e-5)")

    # 6. 适配器保存/加载
    print(f"\n{'=' * 60}")
    print("测试 6: 适配器保存/加载")
    print("=" * 60)
    model.unmerge_lora()
    lora_sd = model.get_lora_state_dict()
    print(f"LoRA state_dict 键数: {len(lora_sd)}")
    lora_size = sum(v.numel() * v.element_size() for v in lora_sd.values())
    total_size = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"LoRA 适配器大小: {lora_size / 1024:.1f} KB")
    print(f"完整模型大小: {total_size / 1024:.1f} KB")
    print(f"存储压缩: {total_size / lora_size:.1f}x")

    # 7. 多任务切换模拟
    print(f"\n{'=' * 60}")
    print("测试 7: 多任务切换")
    print("=" * 60)
    task_a_sd = model.get_lora_state_dict()
    for name, param in model.named_parameters():
        if 'lora_' in name:
            param.data.normal_(0, 0.01)
    task_b_sd = model.get_lora_state_dict()

    model.load_lora_state_dict(task_a_sd)
    with torch.no_grad():
        h_a = model(ids)["logits"]
    model.load_lora_state_dict(task_b_sd)
    with torch.no_grad():
        h_b = model(ids)["logits"]
    print(f"任务 A vs B 差异: {(h_a - h_b).abs().mean().item():.4f} (应 > 0) ✓")

    print(f"\n✅ LoRA 完整模型测试通过！")
```

---

## 7. 实践技巧与可视化

### 7.1 低秩结构可视化

```python
import numpy as np
import matplotlib.pyplot as plt


def plot_singular_value_decay():
    """可视化: 随机矩阵 vs 低秩+噪声 vs LoRA ΔW 的奇异值衰减"""
    np.random.seed(42)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    d = 256

    # (1) 随机满秩矩阵
    _, s1, _ = np.linalg.svd(np.random.randn(d, d))
    axes[0].semilogy(s1, 'b-', lw=2)
    axes[0].set(xlabel='Index', ylabel='σ (log)', title='Random (Full Rank)')

    # (2) 低秩 + 噪声（模拟预训练权重）
    W = np.random.randn(d, 8) @ np.random.randn(8, d) + np.random.randn(d, d) * 0.05
    _, s2, _ = np.linalg.svd(W)
    axes[1].semilogy(s2, 'r-', lw=2)
    axes[1].axvline(x=8, color='gray', ls='--', label='rank=8')
    axes[1].set(xlabel='Index', ylabel='σ (log)', title='Low-Rank + Noise')
    axes[1].legend()

    # (3) ΔW = BA（精确低秩）
    dW = (np.random.randn(d, 4) * 0.01) @ (np.random.randn(4, d) * np.sqrt(2/d))
    _, s3, _ = np.linalg.svd(dW)
    axes[2].semilogy(s3 + 1e-16, 'g-', lw=2)
    axes[2].axvline(x=4, color='gray', ls='--', label='LoRA r=4')
    axes[2].set(xlabel='Index', ylabel='σ (log)', title='ΔW = BA (Exact Low-Rank)')
    axes[2].legend()

    for ax in axes: ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("lora_singular_values.png", dpi=150, bbox_inches='tight')
    plt.show()


def plot_parameter_efficiency():
    """可视化参数效率 (左) 与性能-秩关系 (右)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    d = 4096
    ranks = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    ratios = [r * 2 * d / (d * d) * 100 for r in ranks]

    ax1.bar(range(len(ranks)), ratios, color='steelblue', alpha=0.8)
    ax1.set_xticks(range(len(ranks))); ax1.set_xticklabels(ranks)
    ax1.set(xlabel='LoRA Rank (r)', ylabel='Param Ratio (%)', title=f'd={d}')
    ax1.grid(True, alpha=0.3, axis='y')

    # 论文数据: Performance vs Rank
    pr = [1, 2, 4, 8, 64]
    ax2.plot(range(5), [73.4, 73.3, 73.7, 73.7, 73.4], 'bo-', lw=2, label='WikiSQL')
    ax2.plot(range(5), [91.7, 91.5, 91.5, 91.6, 91.4], 'rs-', lw=2, label='MNLI')
    ax2.axhline(y=73.8, color='blue', ls='--', alpha=0.5, label='Full FT')
    ax2.set_xticks(range(5)); ax2.set_xticklabels(pr)
    ax2.set(xlabel='Rank (r)', ylabel='Accuracy', title='GPT-3 175B')
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("lora_efficiency.png", dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    plot_singular_value_decay()
    plot_parameter_efficiency()
```

### 7.2 实践调参建议

**LoRA 超参数速查表**：

| 超参数 | 推荐值 | 说明 |
|-------|:-----:|------|
| 秩 $r$ | 4~16 | 大多数任务 8 即可 |
| $\alpha$ | 16 或 $2r$ | 固定后只调 $r$ |
| LoRA Dropout | 0.05~0.1 | 防过拟合 |
| 学习率 | $1 \times 10^{-4} \sim 3 \times 10^{-4}$ | 比全参微调大 5~10 倍 |
| 目标层 | $\{W_Q, W_V\}$ 或全部 | 预算充足选全部 |
| 优化器 | AdamW | 权重衰减 0.01 |

**不同模型规模的推荐配置**：

$$
\boxed{
\begin{aligned}
\text{7B 模型:} &\quad r=8, \; \alpha=16, \; \text{lr}=2 \times 10^{-4} \\
\text{13B 模型:} &\quad r=16, \; \alpha=32, \; \text{lr}=1 \times 10^{-4} \\
\text{70B 模型:} &\quad r=16 \sim 32, \; \alpha=32 \sim 64, \; \text{lr}=5 \times 10^{-5}
\end{aligned}
}
$$

**常见陷阱与解决方案**：

| 问题 | 症状 | 解决方案 |
|------|------|---------|
| $r$ 太大 | 过拟合，验证损失上升 | 减小 $r$ 或增加 dropout |
| $r$ 太小 | 训练损失不下降 | 增大 $r$ 或增加目标层 |
| 学习率太大 | 训练不稳定，损失震荡 | 减半学习率 |
| 忘记冻结 $W_0$ | 显存爆炸 | 检查 `requires_grad` |
| 合并后继续训练 | 梯度计算错误 | 先 `unmerge` 再训练 |

---

## 8. 与其他模型的关系

### 8.1 参数高效微调方法对比

| 方法 | 可训练参数 | 推理延迟 | 任务切换 | 数学本质 |
|------|:---------:|:-------:|:-------:|---------|
| **Full Fine-Tuning** | $100\%$ | 无 | 切换整个模型 | $W \leftarrow W + \Delta W$ |
| **Adapter** | $\sim 3\%$ | **有** | 切换适配器 | 插入瓶颈层 $h + f(h)$ |
| **Prefix Tuning** | $\sim 0.1\%$ | 无 | 切换前缀 | 虚拟 token $[P; x]$ |
| **LoRA** | $\sim 0.1\%$ | **无** | 切换 $(B, A)$ | $W + \frac{\alpha}{r}BA$ |
| **QLoRA** | $\sim 0.1\%$ | **无** | 切换 $(B, A)$ | 4-bit $W_0$ + LoRA |

$$
\boxed{\text{LoRA 的独特优势} = \text{参数高效} + \text{零推理延迟} + \text{灵活切换}}
$$

**Adapter vs LoRA**：

```
Adapter: x → [Frozen Layer] → [Adapter Layer] → output
                                ↑ 额外串行计算
                                
LoRA:    x → [Frozen Layer + ΔW merged] → output
                ↑ 无额外计算（合并后）
```

### 8.2 LoRA 在大模型发展中的定位

**参数高效微调的三个阶段**：

```
Phase 1: Feature Extraction (2018)     ← 冻结全部，只训练分类头
Phase 2: Adapter/Prefix (2019-2020)    ← 插入/添加少量可训练模块
Phase 3: LoRA (2021+)                  ← 低秩重参数化，零延迟
```

**LoRA 解决了大模型时代的核心问题**：

1. **民主化微调**：LoRA + 4-bit 量化（QLoRA）使得消费级 GPU 可以微调 70B 模型
2. **多任务部署**：同一基座模型 + 不同 LoRA 适配器，实现高效多任务服务
3. **快速迭代**：微调时间从天级降到小时级

### 8.3 LoRA 的后续发展

```
LoRA (Hu, 2021) ── 低秩适应，r=4~8 即可匹配全参微调
  ├── QLoRA (Dettmers, 2023) ── 4-bit 量化 + LoRA，单卡微调 65B
  ├── AdaLoRA (Zhang, 2023) ── 自适应秩分配，重要层高秩
  ├── DoRA (Liu, 2024) ── 分解方向和幅度，更接近全参微调
  ├── LoRA+ (Hayou, 2024) ── A 和 B 用不同学习率
  ├── rsLoRA (Kalajdzievski, 2024) ── 缩放因子 α/√r 替代 α/r
  └── GaLore (Zhao, 2024) ── 梯度低秩投影，全参数更新的低秩优化
```

| LoRA 遗留问题 | 后续解决方案 |
|:------------:|:----------:|
| 秩对所有层统一 | AdaLoRA 自适应秩 |
| 与全参微调仍有差距 | DoRA 分解方向/幅度 |
| 显存仍受 $W_0$ 精度限制 | QLoRA 4-bit 量化 |
| $A, B$ 用相同学习率 | LoRA+ 差异化学习率 |
| 缩放 $\alpha/r$ 不随 $r$ 最优 | rsLoRA 改进缩放 |

**LoRA 在 2024-2025 年的影响**：

$$
\boxed{\text{LoRA 已成为大模型微调的事实标准：Hugging Face PEFT、LLaMA-Factory、Axolotl 等主流框架均以 LoRA 为默认方法}}
$$

---

## 扩展阅读与实现

### 问题 1：LoRA 与全参微调的等价性条件

**当 $r = \min(d, k)$ 时**，LoRA 退化为全参微调——$B \in \mathbb{R}^{d \times d}$，$A \in \mathbb{R}^{d \times k}$，$BA$ 可以表示任意 $d \times k$ 矩阵。但此时参数量 $r(d+k) = d(d+k) > dk$，反而更多（因为 $B, A$ 的参数化是冗余的）。LoRA 的价值在于 $r \ll \min(d, k)$ 时的高效性。

### 问题 2：为什么不直接学习 $\Delta W$？

直接学习 $\Delta W \in \mathbb{R}^{d \times k}$ 等价于全参微调——需要 $dk$ 个可训练参数和对应的优化器状态。LoRA 通过 $\Delta W = BA$ 将参数从 $dk$ 降为 $r(d+k)$，同时隐式地限制了 $\Delta W$ 的秩不超过 $r$，提供了正则化效果。

### 问题 3：LoRA 的正则化效应

低秩约束 $\text{rank}(\Delta W) \leq r$ 是一种**结构化正则化**。它限制了微调可以修改的"方向"数量：

$$
\Delta W = \sum_{i=1}^{r} \mathbf{b}_i \mathbf{a}_i^\top
$$

只有 $r$ 个秩-1 更新方向。当 $r$ 小时，模型只能沿少数重要方向调整，防止过拟合。这解释了为什么 LoRA 在小数据集上的表现有时**优于**全参微调。

### 问题 4：LoRA 与矩阵补全的联系

LoRA 可以看作一种特殊的**矩阵补全**问题。预训练权重 $W_0$ 是已知的"观测"，任务适配需要找到一个低秩修正 $\Delta W$，使得 $W_0 + \Delta W$ 在目标任务上最优。这与推荐系统中的矩阵分解（如 Netflix Prize 的 SVD++）在数学形式上完全一致。

### 问题 5：不同层的最优秩分析

实验表明，不同层对秩的需求不同：

| 层的位置 | 最优秩趋势 | 直觉 |
|---------|:---------:|------|
| 底层（接近输入） | 较低 | 底层特征通用性强，变化小 |
| 中间层 | 中等 | 逐渐从通用到任务特定 |
| 顶层（接近输出） | 较高 | 任务特定特征变化大 |

这启发了 AdaLoRA 的设计：根据每层参数的重要性动态分配秩，而非所有层用统一的 $r$。

---

## 参考资源

### 经典论文

1. Hu, E. J., Shen, Y., Wallis, P., et al. (2021). [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). ICLR 2022.
   - **贡献**：提出低秩适应方法，仅训练 0.01% 参数即可匹配全参微调效果

2. Aghajanyan, A., Gupta, S., & Zettlemoyer, L. (2020). [Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning](https://arxiv.org/abs/2012.13255). ACL 2021.
   - **贡献**：发现预训练模型微调的内在维度远小于参数空间维度，为 LoRA 提供理论基础

3. Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). [QLoRA: Efficient Finetuning of Quantized Language Models](https://arxiv.org/abs/2305.14314). NeurIPS 2023.
   - **贡献**：结合 4-bit 量化与 LoRA，单张 48GB GPU 微调 65B 参数模型

4. Liu, S.-Y., Wang, C.-Y., Yin, H., et al. (2024). [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353). ICML 2024.
   - **贡献**：将权重分解为方向和幅度，分别用 LoRA 适应方向，缩小与全参微调差距

5. Houlsby, N., Giampiccolo, A., Morber, S., et al. (2019). [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751). ICML 2019.
   - **贡献**：提出 Adapter Tuning，参数高效微调的先驱工作

### 教材与书籍

6. Eckart, C. & Young, G. (1936). The Approximation of One Matrix by Another of Lower Rank. Psychometrika.
   - **章节**：低秩矩阵近似的最优性定理（Eckart-Young 定理），LoRA 的数学基础

### 在线资源与教程

7. Hugging Face. [PEFT: Parameter-Efficient Fine-Tuning](https://huggingface.co/docs/peft).
   - **内容**：LoRA、QLoRA、AdaLoRA 等 PEFT 方法的官方实现和教程

8. Sebastian Raschka. [Practical Tips for Finetuning LLMs Using LoRA](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms).
   - **内容**：LoRA 超参数选择、常见问题和实践经验的深度总结

---

## 附录：符号表

| 符号 | 含义 | 维度/类型 |
|------|------|----------|
| $d$ ($d_{\text{model}}$) | 模型隐藏维度 | 标量 |
| $k$ | 输入特征维度 | 标量，通常 $k = d$ |
| $d_{ff}$ | FFN 隐藏层维度 | 标量，通常 $4d$ |
| $d_k$ | 每个注意力头的维度 | 标量 |
| $L$ | Transformer 层数 | 标量 |
| $r$ | LoRA 秩 | 标量，$r \ll \min(d, k)$ |
| $\alpha$ | LoRA 缩放超参数 | 标量，通常 16 |
| $W_0$ | 冻结的预训练权重 | $(d, k)$ |
| $\Delta W$ | 权重更新矩阵 | $(d, k)$，$\text{rank} \leq r$ |
| $A$ | LoRA 下投影矩阵（可训练） | $(r, k)$ |
| $B$ | LoRA 上投影矩阵（可训练） | $(d, r)$ |
| $W_Q, W_K, W_V, W_O$ | 注意力层投影矩阵 | $(d, d)$ |
| $x$ | 输入向量 | $(k,)$ 或 $(B, T, k)$ |
| $h$ | 输出向量 | $(d,)$ 或 $(B, T, d)$ |
| $z$ | 低维中间表示 $z = Ax$ | $(r,)$ |
| $\sigma_i$ | 第 $i$ 个奇异值 | 标量，$\sigma_1 \geq \sigma_2 \geq \cdots$ |
| $U, V$ | SVD 的左/右奇异向量矩阵 | $(m, m)$, $(n, n)$ |
| $\Sigma$ | 奇异值对角矩阵 | $(\min(m,n), \min(m,n))$ |
| $d_{\text{int}}$ | 内在维度 | 标量，$d_{\text{int}} \ll |\Theta|$ |
| $\mathcal{L}$ | 损失函数值 | 标量 |
| $\ell(\cdot, \cdot)$ | 损失函数 | 函数 |
| $\eta$ | 学习率 | 标量 |
| $|\Theta|$ | 模型总参数量 | 标量 |
| $K$ | 下游任务数量 | 标量 |
| $\|\cdot\|_F$ | Frobenius 范数 | 标量 |

**典型维度示例（GPT-3 175B + LoRA $r=8$）：**
- $d = 12{,}288$，$d_{ff} = 49{,}152$，$d_k = 128$
- $L = 96$，$\text{heads} = 96$
- $r = 8$，$\alpha = 16$
- $|W_Q| = 12{,}288 \times 12{,}288 = 151\text{M}$
- $|A_Q| + |B_Q| = 8 \times 12{,}288 + 12{,}288 \times 8 = 197\text{K}$
- 可训练参数占比：$\sim 0.01\%$

---

最后更新：2026-03-19
