# RLHF 数学原理与实现 —— 从人类反馈到强化学习的完整推导

> **前置知识**：强化学习基础（策略梯度、价值函数）、概率论（KL 散度）、Transformer 架构、预训练-微调范式、Python 基础  
> **与前面内容的联系**：建议先学习 [GPT3-Scaling-and-InContext](./10-GPT3-Scaling-and-InContext.md) 理解大规模预训练，以及 [LoRA-Math-and-Implementation](./12-LoRA-Math-and-Implementation.md) 理解参数高效微调  
> **与后续内容的联系**：RLHF 是 ChatGPT 的核心训练范式，直接影响了 LLaMA 系列的对齐训练和 DeepSeek-R1 的 GRPO 方法

---

## 目录

1. [引言：为什么需要人类反馈对齐？](#1-引言为什么需要人类反馈对齐)
   - 1.1 [预训练模型的对齐困境](#11-预训练模型的对齐困境)
   - 1.2 [从指令微调到人类偏好学习](#12-从指令微调到人类偏好学习)
   - 1.3 [RLHF 的三阶段训练范式](#13-rlhf-的三阶段训练范式)
   - 1.4 [本科数学知识映射表](#14-本科数学知识映射表)
2. [强化学习基础：语言模型视角](#2-强化学习基础语言模型视角)
   - 2.1 [马尔可夫决策过程（MDP）回顾](#21-马尔可夫决策过程mdp回顾)
   - 2.2 [语言生成作为序列决策问题](#22-语言生成作为序列决策问题)
   - 2.3 [策略梯度定理](#23-策略梯度定理)
   - 2.4 [KL 散度约束的数学基础](#24-kl-散度约束的数学基础)
3. [阶段一：监督微调（SFT）](#3-阶段一监督微调sft)
   - 3.1 [SFT 的目标函数](#31-sft-的目标函数)
   - 3.2 [指令数据的构造](#32-指令数据的构造)
   - 3.3 [SFT 的梯度计算](#33-sft-的梯度计算)
   - 3.4 [SFT 与预训练的数学区别](#34-sft-与预训练的数学区别)
4. [阶段二：奖励模型训练（RM）](#4-阶段二奖励模型训练rm)
   - 4.1 [Bradley-Terry 偏好模型](#41-bradley-terry-偏好模型)
   - 4.2 [奖励模型的架构设计](#42-奖励模型的架构设计)
   - 4.3 [排序损失函数推导](#43-排序损失函数推导)
   - 4.4 [奖励模型的梯度分析](#44-奖励模型的梯度分析)
   - 4.5 [奖励模型的标定与归一化](#45-奖励模型的标定与归一化)
5. [阶段三：PPO 强化学习优化](#5-阶段三ppo-强化学习优化)
   - 5.1 [RLHF 的优化目标](#51-rlhf-的优化目标)
   - 5.2 [PPO 算法的核心思想](#52-ppo-算法的核心思想)
   - 5.3 [PPO-Clip 目标函数推导](#53-ppo-clip-目标函数推导)
   - 5.4 [广义优势估计（GAE）](#54-广义优势估计gae)
   - 5.5 [KL 惩罚与约束的数学分析](#55-kl-惩罚与约束的数学分析)
   - 5.6 [Value Function 的训练](#56-value-function-的训练)
6. [从数学到代码：完整实现](#6-从数学到代码完整实现)
   - 6.1 [NumPy 实现核心组件](#61-numpy-实现核心组件)
   - 6.2 [PyTorch 完整实现](#62-pytorch-完整实现)
7. [实践技巧与可视化](#7-实践技巧与可视化)
   - 7.1 [奖励分布可视化](#71-奖励分布可视化)
   - 7.2 [PPO 训练动态监控](#72-ppo-训练动态监控)
   - 7.3 [调参建议与常见陷阱](#73-调参建议与常见陷阱)
8. [与其他模型的关系](#8-与其他模型的关系)
   - 8.1 [从 GPT-3 到 ChatGPT 的演进](#81-从-gpt-3-到-chatgpt-的演进)
   - 8.2 [RLHF 与 DPO 的理论联系](#82-rlhf-与-dpo-的理论联系)
   - 8.3 [对齐方法谱系](#83-对齐方法谱系)

[扩展阅读与实现](#扩展阅读与实现)

[参考资源](#参考资源)

附录：[符号表](#附录符号表)

---

## 1. 引言：为什么需要人类反馈对齐？

### 1.1 预训练模型的对齐困境

GPT-3 展示了令人惊叹的 In-context Learning 能力，但预训练语言模型存在根本性的**对齐问题（Alignment Problem）**：

| 问题类型 | 表现 | 数学根源 |
|----------|------|----------|
| **不忠实** | 生成看似合理但错误的内容 | 最大似然训练不惩罚"流畅的错误" |
| **不安全** | 生成有害、有偏见的内容 | 训练数据中包含有害信息 |
| **不遵循指令** | 无法准确理解用户意图 | 预训练目标 ≠ 指令跟随目标 |
| **冗长/空洞** | 生成不必要的冗长回答 | 最大似然鼓励高概率但低信息量的词 |

根本矛盾在于预训练目标与人类期望之间的**目标失配（Objective Mismatch）**：

$$
\underbrace{\max_\theta \sum_{t} \log p_\theta(x_t | x_{<t})}_{\text{预训练目标：预测下一个 token}} \neq \underbrace{\max_\theta \mathbb{E}[\text{人类满意度}]}_{\text{真实目标：有用、安全、忠实}}
$$

> **关键洞察**：预训练模型学会了"什么文本最可能出现"，而非"什么回答最有帮助"。RLHF 的核心就是弥合这一鸿沟。

### 1.2 从指令微调到人类偏好学习

对齐技术的发展经历了三个阶段：

**阶段 A：纯预训练**（GPT-2/3）
$$
\mathcal{L}_{\text{PT}} = -\sum_{t=1}^T \log p_\theta(x_t | x_{<t})
$$
- 无法控制输出方向，仅"续写"

**阶段 B：监督微调（SFT）**
$$
\mathcal{L}_{\text{SFT}} = -\sum_{t=1}^T \log p_\theta(y_t | x, y_{<t})
$$
- 在 $(x, y)$（指令，理想回答）对上微调
- 局限：需要大量高质量标注，且无法区分"好"与"更好"

**阶段 C：RLHF**
$$
\max_\theta \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)} \left[ R_\phi(x, y) \right] - \beta \, \text{KL}\left[\pi_\theta \| \pi_{\text{ref}}\right]
$$
- 从人类偏好排序中学习奖励函数
- 使用强化学习优化模型输出
- 通过 KL 约束防止偏离过远

### 1.3 RLHF 的三阶段训练范式

InstructGPT 论文（Ouyang et al., 2022）提出的经典三阶段流程：

```
┌─────────────────────────────────────────────────────────────────────┐
│                    RLHF 三阶段训练流程                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  阶段 1：监督微调（SFT）                                             │
│  ┌──────────────────────────────────────────┐                       │
│  │  预训练模型 + 人工标注数据                   │                       │
│  │  (prompt, ideal_response) pairs           │                       │
│  │  → SFT 模型 π_SFT                        │                       │
│  └──────────────────────────────────────────┘                       │
│                          ↓                                          │
│  阶段 2：奖励模型训练（RM）                                           │
│  ┌──────────────────────────────────────────┐                       │
│  │  SFT 模型生成多个回答                       │                       │
│  │  人类标注偏好排序 y_w ≻ y_l                │                       │
│  │  训练奖励模型 R_φ(x, y)                   │                       │
│  └──────────────────────────────────────────┘                       │
│                          ↓                                          │
│  阶段 3：PPO 强化学习优化                                             │
│  ┌──────────────────────────────────────────┐                       │
│  │  策略模型 π_θ 生成回答                      │                       │
│  │  奖励模型打分 + KL 惩罚                     │                       │
│  │  PPO 更新策略参数                           │                       │
│  └──────────────────────────────────────────┘                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

InstructGPT 的数据规模：

| 阶段 | 数据量 | 标注者 | 成本 |
|------|:------:|:------:|:----:|
| SFT | ~13,000 条 | 40 人 | 中等 |
| RM | ~33,000 条比较 | 40 人 | 较高 |
| PPO | ~31,000 条 prompt | — | 计算密集 |

### 1.4 本科数学知识映射表

| RLHF 概念 | 对应数学 | 本科课程 |
|-----------|----------|----------|
| 策略梯度 | 对数导数技巧（REINFORCE） | 概率论、优化 |
| Bradley-Terry 模型 | Logistic 回归、极大似然 | 统计学 |
| KL 散度 | 信息论、概率分布距离 | 信息论 |
| PPO-Clip | 信赖域优化、截断 | 最优化理论 |
| GAE | 时序差分、指数加权平均 | 强化学习 |
| 奖励塑形 | 势函数、辅助奖励 | 强化学习 |

---

## 2. 强化学习基础：语言模型视角

### 2.1 马尔可夫决策过程（MDP）回顾

标准 MDP 由五元组定义：

$$
\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)
$$

| 元素 | 含义 | 语言模型对应 |
|------|------|-------------|
| $\mathcal{S}$ | 状态空间 | 已生成的 token 序列 $(x, y_{<t})$ |
| $\mathcal{A}$ | 动作空间 | 词表 $\mathcal{V}$（选择下一个 token） |
| $P(s' \mid s, a)$ | 转移概率 | 确定性：拼接 $a$ 到序列末尾 |
| $R(s, a)$ | 奖励函数 | 由奖励模型 $R_\phi$ 提供 |
| $\gamma$ | 折扣因子 | 通常取 $1$（完整回答后统一评分） |

> **注意**：语言生成的 MDP 有一个重要特点——**转移是确定性的**（选了哪个 token，状态就确定了），随机性完全来自策略 $\pi_\theta(a_t \mid s_t)$。

### 2.2 语言生成作为序列决策问题

将语言模型 $\pi_\theta$ 视为策略：

$$
\pi_\theta(y_t | x, y_{<t}) = \text{softmax}\left(\frac{z_t}{\tau}\right)_{y_t}
$$

其中 $z_t = f_\theta(x, y_{<t})$ 是模型的 logits 输出。

**完整回答的概率**：

$$
\pi_\theta(y | x) = \prod_{t=1}^{|y|} \pi_\theta(y_t | x, y_{<t})
$$

**对数概率**（更常用）：

$$
\log \pi_\theta(y | x) = \sum_{t=1}^{|y|} \log \pi_\theta(y_t | x, y_{<t})
$$

**轨迹回报**：在 RLHF 中，通常只在生成结束时给出奖励（episode reward）：

$$
R_{\text{total}}(x, y) = R_\phi(x, y) - \beta \sum_{t=1}^{|y|} \log \frac{\pi_\theta(y_t | x, y_{<t})}{\pi_{\text{ref}}(y_t | x, y_{<t})}
$$

### 2.3 策略梯度定理

**REINFORCE 算法**的核心是对数导数技巧（log-derivative trick）：

$$
\nabla_\theta \mathbb{E}_{y \sim \pi_\theta} [R(y)] = \mathbb{E}_{y \sim \pi_\theta} \left[ R(y) \nabla_\theta \log \pi_\theta(y) \right]
$$

**推导过程**：

$$
\nabla_\theta \mathbb{E}_{y \sim \pi_\theta} [R(y)] = \nabla_\theta \sum_y \pi_\theta(y) R(y)
$$

$$
= \sum_y R(y) \nabla_\theta \pi_\theta(y)
$$

$$
= \sum_y R(y) \pi_\theta(y) \frac{\nabla_\theta \pi_\theta(y)}{\pi_\theta(y)}
$$

$$
\boxed{= \mathbb{E}_{y \sim \pi_\theta} \left[ R(y) \nabla_\theta \log \pi_\theta(y) \right]}
$$

> **直觉理解**：梯度方向 = 奖励 × 概率变化方向。高奖励的动作被增强，低奖励的动作被抑制。

**减小方差：基线（Baseline）**

直接使用 REINFORCE 方差很大，引入基线 $b$：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{y \sim \pi_\theta} \left[ (R(y) - b) \nabla_\theta \log \pi_\theta(y) \right]
$$

常用基线：$b = V_\psi(s)$（价值函数估计），此时 $R(y) - V_\psi(s)$ 称为**优势函数（Advantage）** $A(s, a)$。

### 2.4 KL 散度约束的数学基础

**KL 散度定义**：

$$
\text{KL}[\pi_\theta \| \pi_{\text{ref}}] = \mathbb{E}_{y \sim \pi_\theta} \left[ \log \frac{\pi_\theta(y | x)}{\pi_{\text{ref}}(y | x)} \right]
$$

**关键性质**：

1. **非负性**：$\text{KL}[p \| q] \geq 0$，等号当且仅当 $p = q$
2. **非对称性**：$\text{KL}[p \| q] \neq \text{KL}[q \| p]$
3. **模式覆盖 vs 模式寻找**：
   - $\text{KL}[\pi_\theta \| \pi_{\text{ref}}]$（前向 KL）：$\pi_\theta$ 覆盖 $\pi_{\text{ref}}$ 的所有模式
   - $\text{KL}[\pi_{\text{ref}} \| \pi_\theta]$（反向 KL）：$\pi_\theta$ 集中在 $\pi_{\text{ref}}$ 的某些模式

**在 RLHF 中的 token 级分解**：

$$
\text{KL}[\pi_\theta \| \pi_{\text{ref}}] = \sum_{t=1}^{|y|} \mathbb{E}_{y_{<t} \sim \pi_\theta} \left[ \text{KL}\left[\pi_\theta(\cdot | x, y_{<t}) \| \pi_{\text{ref}}(\cdot | x, y_{<t})\right] \right]
$$

> **为什么需要 KL 约束？** 没有约束，策略优化会找到奖励模型的"漏洞"——生成奖励模型给高分但人类认为质量差的文本（reward hacking）。

---

## 3. 阶段一：监督微调（SFT）

### 3.1 SFT 的目标函数

给定指令数据集 $\mathcal{D}_{\text{SFT}} = \{(x^{(i)}, y^{(i)})\}_{i=1}^N$，SFT 最小化负对数似然：

$$
\boxed{\mathcal{L}_{\text{SFT}}(\theta) = -\frac{1}{N} \sum_{i=1}^N \sum_{t=1}^{|y^{(i)}|} \log \pi_\theta(y_t^{(i)} | x^{(i)}, y_{<t}^{(i)})}
$$

注意：**只在回答部分 $y$ 计算损失**，不在 prompt $x$ 部分计算。

### 3.2 指令数据的构造

InstructGPT 使用三种数据来源：

| 数据类型 | 数量 | 构造方式 | 特点 |
|----------|:----:|----------|------|
| **Plain** | ~13K | 标注者自由编写 prompt + 理想回答 | 多样性高 |
| **Few-shot** | — | 从 API 用户请求中采样 | 真实分布 |
| **User-based** | — | 用户提交的 prompt | 反映真实需求 |

Prompt 分布覆盖：

```
生成类 (45.6%) → 开放式问答 (12.4%) → 头脑风暴 (11.2%)
                → 聊天 (8.4%) → 改写 (6.6%) → 摘要 (4.2%)
                → 分类 (3.5%) → 其他 (8.1%)
```

### 3.3 SFT 的梯度计算

对于单个样本 $(x, y)$，SFT 损失的梯度：

$$
\nabla_\theta \mathcal{L}_{\text{SFT}} = -\sum_{t=1}^{|y|} \nabla_\theta \log \pi_\theta(y_t | x, y_{<t})
$$

展开 softmax 层的梯度：

$$
\nabla_\theta \log \pi_\theta(y_t | x, y_{<t}) = \nabla_\theta z_{y_t} - \mathbb{E}_{a \sim \pi_\theta(\cdot | x, y_{<t})} [\nabla_\theta z_a]
$$

其中 $z_a$ 是 token $a$ 的 logit。

$$
\boxed{\nabla_\theta \log \pi_\theta(y_t | s_t) = \nabla_\theta z_{y_t} - \sum_{a \in \mathcal{V}} \pi_\theta(a | s_t) \nabla_\theta z_a}
$$

> **直觉**：增大正确 token $y_t$ 的 logit，同时按概率加权减小其他 token 的 logit。

### 3.4 SFT 与预训练的数学区别

| 维度 | 预训练 | SFT |
|------|--------|-----|
| 数据 | 大规模无标注语料 | 少量高质量 (prompt, response) |
| 目标 | 预测下一个 token | 在给定 prompt 下生成目标回答 |
| 损失范围 | 全序列 | 仅回答部分 |
| 学习率 | 较大 (1e-4 ~ 3e-4) | 较小 (1e-5 ~ 5e-5) |
| Epoch | 1 (大数据) | 2~5 (小数据，防过拟合) |

**损失掩码**的实现：

$$
\mathcal{L}_{\text{SFT}} = -\frac{1}{\sum_t m_t} \sum_{t=1}^{T} m_t \log \pi_\theta(x_t | x_{<t})
$$

其中掩码 $m_t = \begin{cases} 1 & \text{if } t \in \text{response tokens} \\ 0 & \text{if } t \in \text{prompt tokens} \end{cases}$

---

## 4. 阶段二：奖励模型训练（RM）

### 4.1 Bradley-Terry 偏好模型

人类偏好判断"回答 A 比回答 B 好"可以用 **Bradley-Terry 模型**形式化：

$$
\boxed{P(y_w \succ y_l | x) = \sigma\left(R_\phi(x, y_w) - R_\phi(x, y_l)\right)}
$$

其中：
- $y_w$ 是人类偏好的（winning）回答
- $y_l$ 是较差的（losing）回答
- $\sigma(z) = \frac{1}{1+e^{-z}}$ 是 sigmoid 函数
- $R_\phi(x, y)$ 是奖励模型的输出标量

> **数学直觉**：奖励差值越大，偏好概率越高。当 $R_\phi(x, y_w) - R_\phi(x, y_l) \to +\infty$ 时，$P \to 1$。

**Bradley-Terry 模型的由来**：

假设每个回答有一个潜在"质量分数" $R$，偏好概率正比于质量比：

$$
P(y_w \succ y_l) = \frac{e^{R(y_w)}}{e^{R(y_w)} + e^{R(y_l)}} = \sigma(R(y_w) - R(y_l))
$$

这与 **Logistic 回归**的形式完全一致。

### 4.2 奖励模型的架构设计

InstructGPT 的奖励模型基于 GPT-3 架构：

```
┌─────────────────────────────────────────────┐
│              奖励模型架构                      │
├─────────────────────────────────────────────┤
│                                             │
│  输入: [prompt] + [response]                │
│          ↓                                  │
│  Transformer Decoder (GPT-3 架构)           │
│  - 移除语言模型头                              │
│  - 保留所有 Transformer 层                   │
│          ↓                                  │
│  最后一个 token 的隐藏状态 h_T               │
│          ↓                                  │
│  线性投影层: W_r ∈ R^{1×d}                  │
│          ↓                                  │
│  标量奖励: R_φ(x, y) = W_r · h_T + b_r    │
│                                             │
└─────────────────────────────────────────────┘
```

$$
R_\phi(x, y) = W_r^\top h_T(x, y) + b_r
$$

其中 $h_T(x, y) \in \mathbb{R}^d$ 是输入序列 $[x; y]$ 最后一个 token 的隐藏表示。

**设计选择**：

| 选择 | InstructGPT 方案 | 原因 |
|------|-----------------|------|
| 初始化 | 从 SFT 模型初始化 | 已理解指令格式 |
| 模型大小 | 6B (vs 175B policy) | RM 不需要生成能力 |
| 输出头 | 单标量线性层 | 偏好是一维排序 |
| 训练数据 | 33K 人类比较 | 排序比打分容易标注 |

### 4.3 排序损失函数推导

给定偏好数据集 $\mathcal{D}_{\text{RM}} = \{(x^{(i)}, y_w^{(i)}, y_l^{(i)})\}_{i=1}^N$，奖励模型的训练目标是最大化正确排序的对数似然：

$$
\boxed{\mathcal{L}_{\text{RM}}(\phi) = -\frac{1}{N} \sum_{i=1}^N \log \sigma\left(R_\phi(x^{(i)}, y_w^{(i)}) - R_\phi(x^{(i)}, y_l^{(i)})\right)}
$$

**推导过程**：

从极大似然出发：

$$
\max_\phi \prod_{i=1}^N P(y_w^{(i)} \succ y_l^{(i)} | x^{(i)})
$$

取对数：

$$
\max_\phi \sum_{i=1}^N \log \sigma\left(R_\phi(x^{(i)}, y_w^{(i)}) - R_\phi(x^{(i)}, y_l^{(i)})\right)
$$

转为最小化：

$$
\min_\phi -\frac{1}{N} \sum_{i=1}^N \log \sigma\left(R_\phi(x^{(i)}, y_w^{(i)}) - R_\phi(x^{(i)}, y_l^{(i)})\right)
$$

**扩展到 K 路排序**：

InstructGPT 实际上每次让标注者对 $K$ 个回答进行排序（通常 $K = 4 \sim 9$），然后转化为 $\binom{K}{2}$ 个配对比较：

$$
\mathcal{L}_{\text{RM}}(\phi) = -\frac{1}{\binom{K}{2}} \sum_{(w,l) \in \text{pairs}} \log \sigma\left(R_\phi(x, y_w) - R_\phi(x, y_l)\right)
$$

> **效率优势**：标注 $K$ 个回答的排序产生 $O(K^2)$ 个训练样本，但只需 1 次前向传播（所有回答共享同一 prompt 的编码）。

### 4.4 奖励模型的梯度分析

对单个比较 $(x, y_w, y_l)$，损失的梯度：

$$
\nabla_\phi \mathcal{L} = -\left(1 - \sigma(\Delta R)\right) \left(\nabla_\phi R_\phi(x, y_w) - \nabla_\phi R_\phi(x, y_l)\right)
$$

其中 $\Delta R = R_\phi(x, y_w) - R_\phi(x, y_l)$。

$$
\boxed{\nabla_\phi \mathcal{L} = -\sigma(-\Delta R) \cdot \nabla_\phi \Delta R}
$$

> **直觉**：
> - 当模型已经正确排序（$\Delta R \gg 0$）时，$\sigma(-\Delta R) \approx 0$，梯度很小 → 不再过度更新
> - 当模型排序错误（$\Delta R < 0$）时，$\sigma(-\Delta R) \approx 1$，梯度最大 → 强力纠正
> - 这与 Logistic 回归的梯度性质完全一致

### 4.5 奖励模型的标定与归一化

奖励模型输出的绝对值没有固定含义，重要的是**相对排序**。InstructGPT 使用以下技巧：

**1. 奖励归一化**：

$$
R_{\text{norm}}(x, y) = \frac{R_\phi(x, y) - \mu_R}{\sigma_R}
$$

其中 $\mu_R$ 和 $\sigma_R$ 是训练集上奖励的均值和标准差。

**2. 奖励裁剪**：

$$
R_{\text{clip}}(x, y) = \text{clip}(R_\phi(x, y), -c, c)
$$

防止极端奖励值导致 PPO 训练不稳定。

**3. 标定检验**：

好的奖励模型应满足：

$$
\mathbb{E}[R_\phi(x, y_w)] > \mathbb{E}[R_\phi(x, y_l)] \quad \text{（排序一致性）}
$$

$$
\text{Acc}_{\text{pair}} = P(\Delta R > 0) \approx 0.65 \sim 0.75 \quad \text{（人类一致率上限 ~0.73）}
$$

---

## 5. 阶段三：PPO 强化学习优化

### 5.1 RLHF 的优化目标

RLHF 阶段的目标函数结合了奖励最大化和 KL 约束：

$$
\boxed{\max_\theta \; \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot|x)} \left[ R_\phi(x, y) - \beta \, \text{KL}\left[\pi_\theta(\cdot|x) \| \pi_{\text{ref}}(\cdot|x)\right] \right]}
$$

等价形式（token 级展开）：

$$
\max_\theta \; \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta} \left[ R_\phi(x, y) - \beta \sum_{t=1}^{|y|} \log \frac{\pi_\theta(y_t | x, y_{<t})}{\pi_{\text{ref}}(y_t | x, y_{<t})} \right]
$$

**闭式最优解**（理论分析）：

在无参数约束下，KL 正则化目标的最优策略为：

$$
\pi^*(y | x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y | x) \exp\left(\frac{R_\phi(x, y)}{\beta}\right)
$$

其中 $Z(x) = \sum_y \pi_{\text{ref}}(y | x) \exp\left(\frac{R_\phi(x, y)}{\beta}\right)$ 是配分函数。

> **重要推论**：这个闭式解正是 DPO（Direct Preference Optimization）的理论基础——绕过显式的奖励模型和 RL 训练。

### 5.2 PPO 算法的核心思想

PPO（Proximal Policy Optimization）是一种**近端策略优化**算法，通过限制每次更新的步长来保证训练稳定性。

**从 TRPO 到 PPO 的演进**：

TRPO 使用硬约束：
$$
\max_\theta \; \mathbb{E}\left[\frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} A^{\pi_{\theta_{\text{old}}}}(s,a)\right] \quad \text{s.t.} \quad \text{KL}[\pi_{\theta_{\text{old}}} \| \pi_\theta] \leq \delta
$$

PPO 用截断（clip）代替硬约束，更简单高效：

$$
\boxed{L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]}
$$

其中概率比 $r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}$。

### 5.3 PPO-Clip 目标函数推导

**概率比（Importance Sampling Ratio）**：

$$
r_t(\theta) = \frac{\pi_\theta(y_t | x, y_{<t})}{\pi_{\theta_{\text{old}}}(y_t | x, y_{<t})} = \exp\left(\log \pi_\theta(y_t | x, y_{<t}) - \log \pi_{\theta_{\text{old}}}(y_t | x, y_{<t})\right)
$$

**截断机制的分析**：

当优势 $\hat{A}_t > 0$（好的动作）时：

$$
L^{\text{CLIP}} = \min\left( r_t \hat{A}_t, \; \text{clip}(r_t, 1-\epsilon, 1+\epsilon) \hat{A}_t \right)
$$

- 如果 $r_t > 1 + \epsilon$：被截断为 $(1+\epsilon) \hat{A}_t$，防止过度增大好动作的概率
- 如果 $r_t < 1 - \epsilon$：$\min$ 选择 $r_t \hat{A}_t$，允许减小

当优势 $\hat{A}_t < 0$（差的动作）时：

- 如果 $r_t < 1 - \epsilon$：被截断为 $(1-\epsilon) \hat{A}_t$，防止过度减小差动作的概率
- 如果 $r_t > 1 + \epsilon$：$\min$ 选择 $r_t \hat{A}_t$，允许增大

**RLHF 中 PPO 的完整目标**：

$$
\boxed{\mathcal{L}_{\text{PPO}}(\theta) = \mathbb{E}_{(x,y) \sim \pi_{\theta_{\text{old}}}} \left[ \sum_{t=1}^{|y|} \min\left( r_t \hat{A}_t, \; \text{clip}(r_t, 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]}
$$

典型超参数：$\epsilon = 0.2$。

### 5.4 广义优势估计（GAE）

**时序差分（TD）误差**：

$$
\delta_t = r_t + \gamma V_\psi(s_{t+1}) - V_\psi(s_t)
$$

在 RLHF 中，token 级奖励为：

$$
r_t^{\text{token}} = \begin{cases} -\beta \log \frac{\pi_\theta(y_t | x, y_{<t})}{\pi_{\text{ref}}(y_t | x, y_{<t})} & t < |y| \\ R_\phi(x, y) - \beta \log \frac{\pi_\theta(y_t | x, y_{<t})}{\pi_{\text{ref}}(y_t | x, y_{<t})} & t = |y| \end{cases}
$$

**GAE 公式**：

$$
\boxed{\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{|y|-t-1} (\gamma \lambda)^l \delta_{t+l}}
$$

展开：

$$
\hat{A}_t = \delta_t + (\gamma\lambda)\delta_{t+1} + (\gamma\lambda)^2 \delta_{t+2} + \cdots
$$

| $\lambda$ | 效果 | 偏差-方差 |
|:---------:|------|----------|
| $0$ | $\hat{A}_t = \delta_t$（单步 TD） | 低方差，高偏差 |
| $1$ | $\hat{A}_t = \sum_{l \geq 0} \gamma^l r_{t+l} - V(s_t)$（MC） | 高方差，低偏差 |
| $0.95$ | 折中（InstructGPT 常用） | 平衡 |

### 5.5 KL 惩罚与约束的数学分析

InstructGPT 使用**自适应 KL 惩罚**：

$$
R_{\text{total}}(x, y) = R_\phi(x, y) - \beta \, \text{KL}[\pi_\theta(\cdot|x) \| \pi_{\text{ref}}(\cdot|x)]
$$

**自适应 $\beta$ 更新规则**：

设定目标 KL 值 $\text{KL}_{\text{target}}$，根据实际 KL 动态调整 $\beta$：

$$
\boxed{\beta \leftarrow \begin{cases} \beta / (1 + \alpha) & \text{if } \text{KL}_{\text{actual}} < \text{KL}_{\text{target}} / 1.5 \\ \beta \times (1 + \alpha) & \text{if } \text{KL}_{\text{actual}} > \text{KL}_{\text{target}} \times 1.5 \\ \beta & \text{otherwise} \end{cases}}
$$

其中 $\alpha$ 是调整步长（如 $\alpha = 0.1$）。

> **Q:** 为什么不直接固定 $\beta$？
>
> **A:** 训练过程中策略变化，固定 $\beta$ 可能导致：
> - $\beta$ 过小 → KL 爆炸，策略崩溃（reward hacking）
> - $\beta$ 过大 → 策略几乎不更新，浪费计算

**两种实现方式的对比**：

| 方式 | 目标函数 | 特点 |
|------|----------|------|
| **KL 惩罚** | $R - \beta \cdot \text{KL}$ | 软约束，简单实现 |
| **KL 约束** | $\max R \; \text{s.t.} \; \text{KL} \leq \delta$ | 硬约束，更严格 |

InstructGPT 使用 KL 惩罚 + 自适应 $\beta$，兼顾灵活性和稳定性。

### 5.6 Value Function 的训练

价值网络 $V_\psi(s_t)$ 估计从状态 $s_t$ 开始的期望回报：

$$
V_\psi(s_t) \approx \mathbb{E}\left[\sum_{l=0}^{|y|-t} \gamma^l r_{t+l}^{\text{token}}\right]
$$

**价值函数损失**（带截断）：

$$
\boxed{\mathcal{L}_{\text{VF}}(\psi) = \frac{1}{2} \mathbb{E}_t \left[ \max\left( (V_\psi(s_t) - V_t^{\text{target}})^2, \; (V_{\text{clip}}(s_t) - V_t^{\text{target}})^2 \right) \right]}
$$

其中：

$$
V_{\text{clip}}(s_t) = V_{\psi_{\text{old}}}(s_t) + \text{clip}\left(V_\psi(s_t) - V_{\psi_{\text{old}}}(s_t), -\epsilon_v, \epsilon_v\right)
$$

**PPO 总损失**：

$$
\mathcal{L}_{\text{total}} = -\mathcal{L}_{\text{PPO}} + c_1 \mathcal{L}_{\text{VF}} - c_2 H[\pi_\theta]
$$

其中 $H[\pi_\theta]$ 是策略熵（鼓励探索），$c_1 = 0.5$，$c_2 = 0.01$ 为典型值。

---

## 6. 从数学到代码：完整实现

### 6.1 NumPy 实现核心组件

#### 6.1.1 Bradley-Terry 奖励模型损失

```python
import numpy as np

def sigmoid(x):
    """数值稳定的 sigmoid 函数"""
    # 避免 exp 溢出
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x))
    )

def reward_model_loss(r_w, r_l):
    """
    Bradley-Terry 排序损失
    
    参数:
        r_w: shape (N,) — 偏好回答的奖励分数
        r_l: shape (N,) — 较差回答的奖励分数
    返回:
        loss: 标量 — 平均排序损失
        grad_r_w: shape (N,) — 对 r_w 的梯度
        grad_r_l: shape (N,) — 对 r_l 的梯度
    
    数学:
        L = -mean(log σ(r_w - r_l))
        ∂L/∂r_w = -mean(σ(-(r_w - r_l)))  (推动 r_w 增大)
        ∂L/∂r_l = +mean(σ(-(r_w - r_l)))  (推动 r_l 减小)
    """
    delta = r_w - r_l                    # (N,) 奖励差
    prob = sigmoid(delta)                # (N,) σ(r_w - r_l)
    loss = -np.mean(np.log(prob + 1e-8)) # 标量
    
    # 梯度: ∂L/∂Δr = -(1 - σ(Δr)) = -σ(-Δr)
    grad_delta = -(1.0 - prob) / len(r_w) # (N,)
    grad_r_w = grad_delta                  # ∂L/∂r_w
    grad_r_l = -grad_delta                 # ∂L/∂r_l
    
    return loss, grad_r_w, grad_r_l

# ===== 验证 =====
np.random.seed(42)
N = 100
r_w = np.random.randn(N) + 1.0  # 偏好回答的奖励 (均值偏高)
r_l = np.random.randn(N) - 0.5  # 较差回答的奖励 (均值偏低)

loss, grad_w, grad_l = reward_model_loss(r_w, r_l)
print(f"奖励模型损失: {loss:.4f}")
print(f"排序准确率: {np.mean(r_w > r_l):.2%}")
print(f"梯度均值 (r_w): {grad_w.mean():.6f} (应为负, 推动 r_w 增大)")
print(f"梯度均值 (r_l): {grad_l.mean():.6f} (应为正, 推动 r_l 减小)")

# 数值梯度验证
eps = 1e-5
r_w_plus = r_w.copy(); r_w_plus[0] += eps
r_w_minus = r_w.copy(); r_w_minus[0] -= eps
num_grad = (reward_model_loss(r_w_plus, r_l)[0] - reward_model_loss(r_w_minus, r_l)[0]) / (2 * eps)
print(f"\n数值梯度验证 (第0个样本):")
print(f"  解析梯度: {grad_w[0]:.8f}")
print(f"  数值梯度: {num_grad:.8f}")
print(f"  相对误差: {abs(grad_w[0] - num_grad) / (abs(num_grad) + 1e-10):.2e}")
```

#### 6.1.2 KL 散度计算

```python
def kl_divergence_token(log_probs_policy, log_probs_ref):
    """
    Token 级 KL 散度: KL[π_θ || π_ref]
    
    参数:
        log_probs_policy: shape (B, T) — 策略模型的 token 对数概率
        log_probs_ref:    shape (B, T) — 参考模型的 token 对数概率
    返回:
        kl_per_token: shape (B, T) — 每个 token 的 KL 贡献
        kl_per_seq:   shape (B,) — 每个序列的总 KL
    
    数学:
        KL_t = log π_θ(y_t|...) - log π_ref(y_t|...)
        KL_seq = Σ_t KL_t  (近似, 仅计算采样 token)
    """
    kl_per_token = log_probs_policy - log_probs_ref  # (B, T)
    kl_per_seq = np.sum(kl_per_token, axis=1)        # (B,)
    return kl_per_token, kl_per_seq

# ===== 验证 =====
B, T = 4, 20  # batch=4, seq_len=20
log_probs_policy = np.random.randn(B, T) * 0.1 - 2.0  # 模拟对数概率
log_probs_ref = log_probs_policy + np.random.randn(B, T) * 0.05  # 参考模型略有不同

kl_token, kl_seq = kl_divergence_token(log_probs_policy, log_probs_ref)
print(f"平均 token 级 KL: {kl_token.mean():.4f}")
print(f"平均序列级 KL: {kl_seq.mean():.4f}")
print(f"KL 范围: [{kl_seq.min():.4f}, {kl_seq.max():.4f}]")
```

#### 6.1.3 PPO-Clip 目标函数

```python
def ppo_clip_objective(log_probs, log_probs_old, advantages, epsilon=0.2):
    """
    PPO-Clip 目标函数
    
    参数:
        log_probs:     shape (B, T) — 当前策略的 token 对数概率
        log_probs_old: shape (B, T) — 旧策略的 token 对数概率
        advantages:    shape (B, T) — 优势估计
        epsilon:       截断参数 (默认 0.2)
    返回:
        loss:  标量 — PPO-Clip 损失 (取负用于梯度下降)
        stats: dict — 训练统计信息
    
    数学:
        r_t = exp(log π_θ - log π_old)
        L = min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)
    """
    # 概率比
    log_ratio = log_probs - log_probs_old          # (B, T)
    ratio = np.exp(log_ratio)                       # (B, T)
    
    # 截断概率比
    ratio_clipped = np.clip(ratio, 1.0 - epsilon, 1.0 + epsilon)  # (B, T)
    
    # PPO-Clip 目标: min(r*A, clip(r)*A)
    surr1 = ratio * advantages                      # (B, T)
    surr2 = ratio_clipped * advantages              # (B, T)
    ppo_objective = np.minimum(surr1, surr2)        # (B, T)
    
    # 平均目标 (取负用于梯度下降)
    loss = -np.mean(ppo_objective)
    
    # 统计信息
    clip_frac = np.mean(np.abs(ratio - 1.0) > epsilon)
    approx_kl = np.mean((ratio - 1.0) - log_ratio)  # 近似 KL
    
    stats = {
        "loss": loss,
        "clip_fraction": clip_frac,
        "approx_kl": approx_kl,
        "ratio_mean": ratio.mean(),
        "ratio_std": ratio.std(),
        "advantage_mean": advantages.mean(),
    }
    
    return loss, stats

# ===== 验证 =====
B, T = 8, 30
log_probs = np.random.randn(B, T) * 0.1 - 3.0
log_probs_old = log_probs + np.random.randn(B, T) * 0.02  # 策略略有变化
advantages = np.random.randn(B, T)  # 标准化后的优势

loss, stats = ppo_clip_objective(log_probs, log_probs_old, advantages)
print(f"PPO-Clip 损失: {loss:.4f}")
for k, v in stats.items():
    print(f"  {k}: {v:.4f}")
```

#### 6.1.4 GAE 计算

```python
def compute_gae(rewards, values, gamma=1.0, lam=0.95):
    """
    广义优势估计 (Generalized Advantage Estimation)
    
    参数:
        rewards: shape (T,) — token 级奖励
        values:  shape (T+1,) — 价值估计 (包含 V(s_{T+1})=0)
        gamma:   折扣因子 (RLHF 中通常为 1.0)
        lam:     GAE λ 参数
    返回:
        advantages: shape (T,) — GAE 优势估计
        returns:    shape (T,) — GAE 回报 (advantages + values[:T])
    
    数学:
        δ_t = r_t + γ V(s_{t+1}) - V(s_t)
        A_t = Σ_{l=0}^{T-t-1} (γλ)^l δ_{t+l}
    """
    T = len(rewards)
    advantages = np.zeros(T)
    last_gae = 0.0
    
    # 反向递推计算 GAE
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        advantages[t] = delta + gamma * lam * last_gae
        last_gae = advantages[t]
    
    returns = advantages + values[:T]
    return advantages, returns

# ===== 验证 =====
T = 20
rewards = np.zeros(T)
rewards[-1] = 3.5  # 仅在最后一个 token 给出奖励 (episode reward)

# 添加 token 级 KL 惩罚
beta = 0.1
kl_penalty = np.random.exponential(0.05, T)  # 模拟 KL 惩罚
rewards = rewards - beta * kl_penalty

values = np.concatenate([np.linspace(0.5, 3.0, T), [0.0]])  # 模拟价值估计

advantages, returns = compute_gae(rewards, values, gamma=1.0, lam=0.95)

# 优势标准化
advantages_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

print(f"原始优势: mean={advantages.mean():.4f}, std={advantages.std():.4f}")
print(f"标准化优势: mean={advantages_norm.mean():.4f}, std={advantages_norm.std():.4f}")
print(f"回报范围: [{returns.min():.4f}, {returns.max():.4f}]")
```

#### 6.1.5 自适应 KL 系数

```python
def adaptive_kl_controller(beta, kl_actual, kl_target, alpha=0.1):
    """
    自适应 KL 惩罚系数
    
    参数:
        beta:       当前 KL 系数
        kl_actual:  实际 KL 散度
        kl_target:  目标 KL 散度
        alpha:      调整步长
    返回:
        new_beta: 更新后的 KL 系数
    
    数学:
        if KL < target/1.5: β ← β/(1+α)  (放松约束)
        if KL > target*1.5: β ← β×(1+α)  (加强约束)
    """
    if kl_actual < kl_target / 1.5:
        new_beta = beta / (1.0 + alpha)
    elif kl_actual > kl_target * 1.5:
        new_beta = beta * (1.0 + alpha)
    else:
        new_beta = beta
    return new_beta

# ===== 模拟自适应过程 =====
beta = 0.1
kl_target = 6.0
print(f"初始 β = {beta:.4f}, KL 目标 = {kl_target:.1f}")
print("-" * 50)

kl_values = [2.0, 3.0, 5.0, 8.0, 12.0, 10.0, 7.0, 5.5, 6.0]
for step, kl in enumerate(kl_values):
    beta = adaptive_kl_controller(beta, kl, kl_target)
    status = "↓放松" if kl < kl_target / 1.5 else ("↑加强" if kl > kl_target * 1.5 else "→保持")
    print(f"Step {step}: KL={kl:.1f}, β={beta:.4f} {status}")
```

### 6.2 PyTorch 完整实现

#### 6.2.1 奖励模型

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class RewardModel(nn.Module):
    """
    基于 Transformer 的奖励模型
    
    架构: Transformer Decoder → 最后 token 隐藏状态 → 线性投影 → 标量奖励
    
    数学:
        R_φ(x, y) = W_r^T h_T([x; y]) + b_r
    """
    
    def __init__(self, vocab_size, d_model=256, n_heads=4, n_layers=4, 
                 max_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Token + 位置嵌入
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer 层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 奖励投影头: h_T → 标量
        self.reward_head = nn.Linear(d_model, 1, bias=True)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        nn.init.normal_(self.reward_head.weight, std=0.02)
        nn.init.zeros_(self.reward_head.bias)
    
    def forward(self, input_ids, attention_mask=None):
        """
        参数:
            input_ids: (B, T) — token ids
            attention_mask: (B, T) — 1 表示有效, 0 表示 padding
        返回:
            rewards: (B,) — 标量奖励
        """
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        
        # 嵌入
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.dropout(x)
        
        # 因果掩码
        causal_mask = torch.triu(
            torch.ones(T, T, device=input_ids.device), diagonal=1
        ).bool()
        
        # Transformer 编码
        x = self.transformer(x, mask=causal_mask)  # (B, T, d_model)
        
        # 获取最后一个有效 token 的隐藏状态
        if attention_mask is not None:
            # 找到每个序列最后一个有效 token 的位置
            seq_lens = attention_mask.sum(dim=1).long() - 1  # (B,)
            h_last = x[torch.arange(B, device=x.device), seq_lens]  # (B, d_model)
        else:
            h_last = x[:, -1, :]  # (B, d_model)
        
        # 投影到标量奖励
        rewards = self.reward_head(h_last).squeeze(-1)  # (B,)
        return rewards

class RewardModelLoss(nn.Module):
    """
    Bradley-Terry 排序损失
    
    数学:
        L = -log σ(R(x, y_w) - R(x, y_l))
    """
    
    def forward(self, rewards_w, rewards_l):
        """
        参数:
            rewards_w: (B,) — 偏好回答的奖励
            rewards_l: (B,) — 较差回答的奖励
        返回:
            loss: 标量
            acc:  排序准确率
        """
        loss = -F.logsigmoid(rewards_w - rewards_l).mean()
        acc = (rewards_w > rewards_l).float().mean()
        return loss, acc

# ===== 验证 =====
torch.manual_seed(42)

vocab_size = 1000
model = RewardModel(vocab_size, d_model=128, n_heads=4, n_layers=2)
criterion = RewardModelLoss()

# 模拟输入
B = 8
T = 32
input_w = torch.randint(0, vocab_size, (B, T))
input_l = torch.randint(0, vocab_size, (B, T))

rewards_w = model(input_w)
rewards_l = model(input_l)
loss, acc = criterion(rewards_w, rewards_l)

print(f"奖励模型参数量: {sum(p.numel() for p in model.parameters()):,}")
print(f"偏好回答奖励: mean={rewards_w.mean():.4f}, std={rewards_w.std():.4f}")
print(f"较差回答奖励: mean={rewards_l.mean():.4f}, std={rewards_l.std():.4f}")
print(f"排序损失: {loss.item():.4f}")
print(f"排序准确率: {acc.item():.2%}")
```

#### 6.2.2 PPO 训练器

```python
class PPOTrainer:
    """
    RLHF PPO 训练器
    
    包含:
        - 策略模型 (Actor): π_θ
        - 价值模型 (Critic): V_ψ
        - 参考模型: π_ref (冻结)
        - 奖励模型: R_φ (冻结)
    """
    
    def __init__(self, policy_model, value_model, ref_model, reward_model,
                 lr=1e-5, clip_eps=0.2, vf_coef=0.5, entropy_coef=0.01,
                 kl_coef=0.1, kl_target=6.0, gamma=1.0, lam=0.95,
                 max_grad_norm=0.5):
        
        self.policy = policy_model
        self.value = value_model
        self.ref = ref_model
        self.reward = reward_model
        
        # 冻结参考模型和奖励模型
        for p in self.ref.parameters():
            p.requires_grad = False
        for p in self.reward.parameters():
            p.requires_grad = False
        
        # 超参数
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.entropy_coef = entropy_coef
        self.kl_coef = kl_coef
        self.kl_target = kl_target
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()),
            lr=lr,
        )
    
    @torch.no_grad()
    def compute_rewards(self, input_ids, attention_mask, log_probs, ref_log_probs):
        """
        计算 token 级奖励 (奖励模型分数 + KL 惩罚)
        
        数学:
            r_t = -β log(π_θ(y_t|...) / π_ref(y_t|...))  (t < T)
            r_T = R_φ(x, y) - β log(π_θ(y_T|...) / π_ref(y_T|...))
        """
        B, T = input_ids.shape
        
        # 奖励模型打分 (整个序列)
        rm_scores = self.reward(input_ids, attention_mask)  # (B,)
        
        # Token 级 KL 惩罚
        kl_penalty = self.kl_coef * (log_probs - ref_log_probs)  # (B, T)
        
        # 构造 token 级奖励
        rewards = -kl_penalty  # (B, T) — 每个 token 的 KL 惩罚
        
        # 最后一个 token 加上奖励模型分数
        if attention_mask is not None:
            last_idx = attention_mask.sum(dim=1).long() - 1  # (B,)
            for i in range(B):
                rewards[i, last_idx[i]] += rm_scores[i]
        else:
            rewards[:, -1] += rm_scores
        
        return rewards, rm_scores, kl_penalty
    
    @torch.no_grad()
    def compute_advantages(self, rewards, values):
        """
        GAE 优势计算
        
        参数:
            rewards: (B, T)
            values:  (B, T)
        返回:
            advantages: (B, T) — 标准化后的优势
            returns:    (B, T) — 目标回报
        """
        B, T = rewards.shape
        advantages = torch.zeros_like(rewards)
        last_gae = torch.zeros(B, device=rewards.device)
        
        # 末尾 value = 0 (episode 结束)
        next_value = torch.zeros(B, device=rewards.device)
        
        for t in reversed(range(T)):
            delta = rewards[:, t] + self.gamma * next_value - values[:, t]
            advantages[:, t] = delta + self.gamma * self.lam * last_gae
            last_gae = advantages[:, t]
            next_value = values[:, t]
        
        returns = advantages + values
        
        # 优势标准化
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def ppo_step(self, input_ids, attention_mask, old_log_probs, 
                 advantages, returns, old_values):
        """
        PPO 更新步
        
        数学:
            L_policy = -min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)
            L_value  = max((V-V_target)², (V_clip-V_target)²)
            L_total  = L_policy + c1 * L_value - c2 * H[π]
        """
        # 当前策略的 log_probs 和 values
        # (实际实现中需要 policy 和 value 的 forward)
        # 这里用简化版本演示核心逻辑
        
        current_log_probs = self.policy(input_ids)   # (B, T) — 简化
        current_values = self.value(input_ids)        # (B, T) — 简化
        
        # === 策略损失 (PPO-Clip) ===
        log_ratio = current_log_probs - old_log_probs  # (B, T)
        ratio = torch.exp(log_ratio)
        
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # === 价值损失 (Clipped) ===
        values_clipped = old_values + torch.clamp(
            current_values - old_values, -self.clip_eps, self.clip_eps
        )
        vf_loss1 = (current_values - returns) ** 2
        vf_loss2 = (values_clipped - returns) ** 2
        value_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()
        
        # === 熵奖励 ===
        # entropy = -Σ π(a) log π(a) (鼓励探索)
        entropy = -(torch.exp(current_log_probs) * current_log_probs).mean()
        
        # === 总损失 ===
        total_loss = policy_loss + self.vf_coef * value_loss - self.entropy_coef * entropy
        
        # === 梯度更新 ===
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.policy.parameters()) + list(self.value.parameters()),
            self.max_grad_norm,
        )
        self.optimizer.step()
        
        # === 统计信息 ===
        clip_frac = (torch.abs(ratio - 1.0) > self.clip_eps).float().mean()
        approx_kl = ((ratio - 1.0) - log_ratio).mean()
        
        stats = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "total_loss": total_loss.item(),
            "clip_fraction": clip_frac.item(),
            "approx_kl": approx_kl.item(),
        }
        
        return stats
    
    def update_kl_coef(self, kl_actual):
        """更新自适应 KL 系数"""
        if kl_actual < self.kl_target / 1.5:
            self.kl_coef /= 1.1
        elif kl_actual > self.kl_target * 1.5:
            self.kl_coef *= 1.1

print("PPOTrainer 类定义完成")
print(f"关键超参数: clip_eps=0.2, kl_target=6.0, gamma=1.0, lam=0.95")
```

#### 6.2.3 完整 RLHF 训练循环

```python
def rlhf_training_loop(policy, value, ref, reward_model, 
                        prompts, tokenizer, n_epochs=3, 
                        batch_size=8, ppo_epochs=4):
    """
    完整的 RLHF 训练循环 (伪代码结构)
    
    流程:
        1. 采样: 用 policy 对 prompt 生成回答
        2. 打分: 用 reward_model 和 ref_model 计算奖励
        3. 估值: 用 value_model 估计优势
        4. 更新: PPO 多轮更新
        5. 调节: 更新 KL 系数
    """
    trainer = PPOTrainer(policy, value, ref, reward_model)
    
    for epoch in range(n_epochs):
        epoch_stats = []
        
        for batch_idx in range(0, len(prompts), batch_size):
            batch_prompts = prompts[batch_idx:batch_idx + batch_size]
            
            # ========== 阶段 1: 采样 ==========
            with torch.no_grad():
                # 用当前策略生成回答
                # generated = policy.generate(batch_prompts, ...)
                # input_ids = tokenize(batch_prompts + generated)
                
                # 计算 log probs (旧策略)
                # old_log_probs = policy.log_prob(input_ids)
                # ref_log_probs = ref.log_prob(input_ids)
                # old_values = value(input_ids)
                pass  # 实际实现需要模型的 generate 方法
            
            # ========== 阶段 2: 计算奖励 ==========
            # rewards, rm_scores, kl_penalty = trainer.compute_rewards(
            #     input_ids, attention_mask, old_log_probs, ref_log_probs
            # )
            
            # ========== 阶段 3: 计算优势 ==========
            # advantages, returns = trainer.compute_advantages(rewards, old_values)
            
            # ========== 阶段 4: PPO 多轮更新 ==========
            for ppo_epoch in range(ppo_epochs):
                pass  # stats = trainer.ppo_step(...)
            
            # ========== 阶段 5: 更新 KL 系数 ==========
            # kl_actual = kl_penalty.mean().item()
            # trainer.update_kl_coef(kl_actual)
        
        # 打印 epoch 统计
        print(f"Epoch {epoch + 1}/{n_epochs} 完成")

print("RLHF 训练循环定义完成")
print("注: 完整运行需要实际的语言模型 (如 GPT-2) 和 tokenizer")
```

#### 6.2.4 奖励模型训练完整示例

```python
def train_reward_model(model, train_data, val_data, 
                       epochs=3, lr=1e-5, batch_size=16):
    """
    训练奖励模型
    
    参数:
        model: RewardModel 实例
        train_data: list of (input_w, input_l) 对
        val_data: 验证集
        epochs: 训练轮数
        lr: 学习率
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = RewardModelLoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss, total_acc = 0.0, 0.0
        n_batches = 0
        
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            input_w = torch.stack([b[0] for b in batch])
            input_l = torch.stack([b[1] for b in batch])
            
            rewards_w = model(input_w)
            rewards_l = model(input_l)
            loss, acc = criterion(rewards_w, rewards_l)
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_acc += acc.item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        avg_acc = total_acc / n_batches
        
        # 验证
        model.eval()
        val_loss, val_acc, val_n = 0.0, 0.0, 0
        with torch.no_grad():
            for i in range(0, len(val_data), batch_size):
                batch = val_data[i:i+batch_size]
                input_w = torch.stack([b[0] for b in batch])
                input_l = torch.stack([b[1] for b in batch])
                
                rewards_w = model(input_w)
                rewards_l = model(input_l)
                loss, acc = criterion(rewards_w, rewards_l)
                
                val_loss += loss.item()
                val_acc += acc.item()
                val_n += 1
        
        val_loss /= max(val_n, 1)
        val_acc /= max(val_n, 1)
        
        print(f"Epoch {epoch+1}: train_loss={avg_loss:.4f}, train_acc={avg_acc:.2%}, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.2%}")

# ===== 验证 =====
torch.manual_seed(42)
vocab_size = 500
T = 24

model = RewardModel(vocab_size, d_model=64, n_heads=2, n_layers=2)

# 生成模拟数据: 偏好回答用较低 token id (模拟 "更好" 的回答)
train_data = []
for _ in range(200):
    input_w = torch.randint(0, vocab_size // 2, (T,))   # "好" 回答
    input_l = torch.randint(vocab_size // 2, vocab_size, (T,))  # "差" 回答
    train_data.append((input_w, input_l))

val_data = []
for _ in range(50):
    input_w = torch.randint(0, vocab_size // 2, (T,))
    input_l = torch.randint(vocab_size // 2, vocab_size, (T,))
    val_data.append((input_w, input_l))

train_reward_model(model, train_data, val_data, epochs=3, lr=3e-4)
```

---

## 7. 实践技巧与可视化

### 7.1 奖励分布可视化

```python
import numpy as np

def visualize_reward_distribution():
    """可视化奖励模型的分数分布"""
    np.random.seed(42)
    
    # 模拟不同训练阶段的奖励分布
    # SFT 模型: 中等奖励，方差较大
    sft_rewards = np.random.normal(0.0, 1.5, 1000)
    # RLHF 初期: 奖励略微提升
    rlhf_early = np.random.normal(0.5, 1.2, 1000)
    # RLHF 后期: 奖励显著提升，方差减小
    rlhf_late = np.random.normal(1.5, 0.8, 1000)
    
    print("=" * 60)
    print("奖励分布统计")
    print("=" * 60)
    print(f"{'阶段':<15} {'均值':>8} {'标准差':>8} {'中位数':>8} {'>0比例':>8}")
    print("-" * 60)
    
    for name, data in [("SFT基线", sft_rewards), 
                        ("RLHF初期", rlhf_early),
                        ("RLHF后期", rlhf_late)]:
        print(f"{name:<15} {data.mean():>8.3f} {data.std():>8.3f} "
              f"{np.median(data):>8.3f} {(data > 0).mean():>8.1%}")
    
    print("\n奖励提升 (vs SFT):")
    print(f"  RLHF 初期: +{rlhf_early.mean() - sft_rewards.mean():.3f}")
    print(f"  RLHF 后期: +{rlhf_late.mean() - sft_rewards.mean():.3f}")

visualize_reward_distribution()
```

### 7.2 PPO 训练动态监控

关键监控指标：

```python
def monitor_ppo_training():
    """模拟 PPO 训练过程的关键指标"""
    np.random.seed(42)
    n_steps = 50
    
    # 模拟训练曲线
    steps = np.arange(n_steps)
    
    # 奖励: 逐步上升后趋于稳定
    reward = 0.5 * np.tanh(0.1 * steps) + np.random.randn(n_steps) * 0.05 + 0.3
    
    # KL 散度: 先上升后被约束稳定
    kl = 3.0 + 4.0 * (1 - np.exp(-0.05 * steps)) + np.random.randn(n_steps) * 0.5
    
    # Clip 比例: 应在 0.1~0.3 之间
    clip_frac = 0.15 + 0.05 * np.sin(0.2 * steps) + np.random.randn(n_steps) * 0.02
    clip_frac = np.clip(clip_frac, 0.0, 1.0)
    
    # 策略熵: 缓慢下降 (策略变得更确定)
    entropy = 4.0 - 0.02 * steps + np.random.randn(n_steps) * 0.1
    
    print("=" * 80)
    print("PPO 训练监控面板")
    print("=" * 80)
    print(f"{'Step':>5} {'Reward':>8} {'KL':>8} {'ClipFrac':>10} {'Entropy':>8} {'状态':>8}")
    print("-" * 80)
    
    for i in range(0, n_steps, 5):
        status = "✅" if 2.0 < kl[i] < 10.0 else ("⚠️" if kl[i] > 10.0 else "📉")
        print(f"{i:>5d} {reward[i]:>8.3f} {kl[i]:>8.2f} {clip_frac[i]:>10.3f} "
              f"{entropy[i]:>8.3f} {status:>8}")
    
    print("\n" + "=" * 80)
    print("训练健康检查:")
    print(f"  ✅ 平均奖励趋势: {np.polyfit(steps, reward, 1)[0]:.4f}/step "
          f"({'上升' if np.polyfit(steps, reward, 1)[0] > 0 else '下降'})")
    print(f"  {'✅' if np.mean(kl) < 12 else '⚠️'} 平均 KL: {np.mean(kl):.2f} "
          f"(目标: 6.0)")
    print(f"  {'✅' if 0.05 < np.mean(clip_frac) < 0.3 else '⚠️'} 平均 Clip 比例: "
          f"{np.mean(clip_frac):.3f} (健康范围: 0.05~0.30)")
    print(f"  {'✅' if np.mean(entropy) > 1.0 else '⚠️'} 平均熵: {np.mean(entropy):.3f} "
          f"(应 > 1.0)")

monitor_ppo_training()
```

### 7.3 调参建议与常见陷阱

#### 关键超参数

| 超参数 | 推荐范围 | 说明 |
|--------|----------|------|
| **学习率** (PPO) | $1 \times 10^{-6}$ ~ $5 \times 10^{-6}$ | 过大导致崩溃 |
| **KL 系数** $\beta$ | $0.01$ ~ $0.2$ | 控制偏离程度 |
| **KL 目标** | $4.0$ ~ $8.0$ | InstructGPT 用 $6.0$ |
| **Clip $\epsilon$** | $0.1$ ~ $0.3$ | 通常 $0.2$ |
| **GAE $\lambda$** | $0.9$ ~ $0.99$ | 偏差-方差权衡 |
| **PPO epochs** | $2$ ~ $4$ | 每批数据的更新次数 |
| **Mini-batch size** | $32$ ~ $256$ | 影响梯度估计 |

#### 常见训练陷阱

**1. Reward Hacking（奖励作弊）**

```
症状: 奖励持续上升，但生成质量下降
原因: 策略找到奖励模型的漏洞
解决: 增大 KL 系数，改进奖励模型
```

$$
\text{Reward Hacking}: \quad R_\phi(x, y^*) \gg 0 \quad \text{但} \quad \text{Quality}(y^*) \ll \text{Quality}(y_{\text{SFT}})
$$

**2. KL 爆炸**

```
症状: KL 散度快速增长到 > 20
原因: 学习率过大 或 KL 系数过小
解决: 降低学习率，增大 KL 系数，使用自适应 β
```

**3. 奖励模型过拟合**

```
症状: RM 训练准确率 > 80% 但验证准确率下降
原因: 数据量不足或分布不均
解决: 增加数据多样性，早停，正则化
```

**4. 策略崩溃（Mode Collapse）**

```
症状: 生成回答趋于单一模式，熵快速下降
原因: 过度优化某类高奖励回答
解决: 增大熵系数 c₂，多样化 prompt 分布
```

---

## 8. 与其他模型的关系

### 8.1 从 GPT-3 到 ChatGPT 的演进

```
GPT-3 (2020)                    InstructGPT (2022)              ChatGPT (2022.11)
│                                │                               │
│ 纯预训练                        │ RLHF 三阶段                    │ InstructGPT 改进
│ 175B 参数                      │ SFT → RM → PPO               │ + 对话优化
│ Few-shot 能力                  │ 1.3B 超越 175B                 │ + 多轮交互
│ 无法遵循指令                    │ 人类偏好对齐                    │ + 安全过滤
│                                │                               │
└─── 目标失配 ──→               └─── 规模化 ──→                 └─── 产品化
```

InstructGPT 的关键发现：

$$
\boxed{\text{1.3B InstructGPT} \succ_{\text{human}} \text{175B GPT-3}}
$$

> **惊人结论**：经过 RLHF 对齐的 1.3B 模型，在人类评估中优于 100 倍大的未对齐 GPT-3。

### 8.2 RLHF 与 DPO 的理论联系

**DPO（Direct Preference Optimization）**直接从偏好数据优化策略，跳过奖励模型：

从 RLHF 的最优策略出发：

$$
\pi^*(y | x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y | x) \exp\left(\frac{R(x, y)}{\beta}\right)
$$

解出隐式奖励：

$$
R(x, y) = \beta \log \frac{\pi^*(y | x)}{\pi_{\text{ref}}(y | x)} + \beta \log Z(x)
$$

代入 Bradley-Terry 模型，$Z(x)$ 消掉：

$$
\boxed{\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]}
$$

| 方法 | 优势 | 劣势 |
|------|------|------|
| **RLHF (PPO)** | 灵活、可组合奖励 | 训练复杂、不稳定 |
| **DPO** | 简单、稳定 | 依赖偏好数据质量 |
| **GRPO** (DeepSeek) | 无需 Critic | 需要可验证奖励 |

### 8.3 对齐方法谱系

```
对齐方法谱系
│
├── 基于人类反馈
│   ├── RLHF (PPO) ← InstructGPT, ChatGPT
│   ├── DPO ← 无需 RM, 直接优化
│   ├── IPO ← 正则化 DPO
│   └── KTO ← 仅需好/坏标签
│
├── 基于 AI 反馈
│   ├── RLAIF ← Constitutional AI (Anthropic)
│   └── Self-Play ← 自我对弈改进
│
├── 基于规则奖励
│   ├── GRPO ← DeepSeek-R1
│   └── ReST ← 自我训练 + 过滤
│
└── 其他
    ├── SPIN ← 自对弈
    └── Rejection Sampling ← Best-of-N
```

**RLHF 在大模型发展中的定位**：

| 时间线 | 模型 | 关键技术 | RLHF 角色 |
|--------|------|----------|-----------|
| 2020 | GPT-3 | Scaling Laws | 无 |
| 2021 | LoRA | 参数高效微调 | 基础设施 |
| 2022 | InstructGPT | RLHF 三阶段 | **核心** |
| 2022 | ChatGPT | 对话优化 | **核心** |
| 2023 | LLaMA | 开源大模型 | SFT + RLHF |
| 2024 | DeepSeek-R1 | GRPO | RLHF 变体 |

---

## 扩展阅读与实现

### Q1: 为什么 RLHF 中使用前向 KL 而非反向 KL？

> **Q:** KL 散度是非对称的，RLHF 中为什么用 $\text{KL}[\pi_\theta \| \pi_{\text{ref}}]$ 而不是 $\text{KL}[\pi_{\text{ref}} \| \pi_\theta]$？
>
> **A:** 两种方向有不同的行为特征：
>
> - **前向 KL** $\text{KL}[\pi_\theta \| \pi_{\text{ref}}]$：
>   - 当 $\pi_{\text{ref}}(y) > 0$ 但 $\pi_\theta(y) \approx 0$ 时，惩罚小
>   - 允许 $\pi_\theta$ 忽略 $\pi_{\text{ref}}$ 的某些模式（**模式寻找**）
>   - 适合 RLHF：允许策略集中在高奖励区域
>
> - **反向 KL** $\text{KL}[\pi_{\text{ref}} \| \pi_\theta]$：
>   - 当 $\pi_\theta(y) \approx 0$ 但 $\pi_{\text{ref}}(y) > 0$ 时，惩罚巨大
>   - 强制 $\pi_\theta$ 覆盖 $\pi_{\text{ref}}$ 的所有模式（**模式覆盖**）
>   - 过于保守，限制策略改进空间

### Q2: PPO-Clip 与 PPO-Penalty 的区别？

> **Q:** PPO 有两个变体，哪个更常用？
>
> **A:**
>
> **PPO-Clip**（更常用）：
> $$L = \min(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t)$$
> - 直接截断概率比，实现简单
> - InstructGPT、大多数开源实现使用此变体
>
> **PPO-Penalty**：
> $$L = r_t A_t - \beta \text{KL}[\pi_\theta \| \pi_{\theta_{\text{old}}}]$$
> - 用 KL 惩罚代替截断
> - 理论上更接近 TRPO
>
> 实际中 PPO-Clip 效果更好且超参数更少。

### Q3: 奖励模型的大小应该如何选择？

> **Q:** InstructGPT 用 6B RM 训练 175B 策略模型，这个比例有什么依据？
>
> **A:** 经验发现：
>
> 1. **RM 不需要和策略模型一样大**：RM 只需要判断质量，不需要生成能力
> 2. **太小的 RM 容易被利用**：策略模型可能找到小 RM 的"盲点"
> 3. **经验比例**：RM 大小约为策略模型的 $\frac{1}{10}$ ~ $\frac{1}{3}$
> 4. **重要的是数据质量**：标注一致性 > 模型大小

### Q4: 如何检测 Reward Hacking？

> **Q:** 如何判断训练是否出现了奖励作弊？
>
> **A:** 关键监控指标：
>
> 1. **奖励-KL 曲线**：正常情况下 KL 增加伴随奖励缓慢上升；异常时奖励突然飙升
> 2. **人工抽检**：定期抽样检查生成质量
> 3. **Gold RM**：用独立的（更大的）奖励模型交叉验证
> 4. **多样性指标**：生成回答的 distinct-n 突然下降表明 mode collapse
>
> $$\text{Reward Hacking 信号}: \quad \frac{d R_\phi}{d \text{KL}} \gg \text{正常速率}$$

### Q5: RLHF 训练需要多少人类标注数据？

> **Q:** InstructGPT 只用了约 33K 比较数据，这够吗？
>
> **A:** 关键不在数量而在质量：
>
> | 数据属性 | 重要性 | InstructGPT 做法 |
> |----------|--------|-----------------|
> | 标注者一致性 | ⭐⭐⭐⭐⭐ | 选拔 + 培训 + 一致性检验 |
> | Prompt 多样性 | ⭐⭐⭐⭐ | 覆盖多种任务类型 |
> | 排序而非打分 | ⭐⭐⭐⭐ | K 路排序 → $O(K^2)$ 配对 |
> | 数据量 | ⭐⭐⭐ | 33K 比较（$K=4~9$） |
>
> 33K 排序在 $K=4$ 时产生 ~198K 配对比较，足够训练 6B RM。

---

## 参考资源

### 经典论文

1. Ouyang et al. (2022). [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155). NeurIPS 2022.
   - **贡献**：提出 InstructGPT 三阶段 RLHF 训练流程，奠定 ChatGPT 的技术基础

2. Christiano et al. (2017). [Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741). NeurIPS 2017.
   - **贡献**：首次提出从人类偏好中学习奖励函数用于 RL 训练

3. Schulman et al. (2017). [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347). arXiv.
   - **贡献**：提出 PPO 算法，成为 RLHF 的核心优化器

4. Stiennon et al. (2020). [Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325). NeurIPS 2020.
   - **贡献**：在文本摘要任务上验证 RLHF 有效性，InstructGPT 的直接前驱

5. Rafailov et al. (2023). [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290). NeurIPS 2023.
   - **贡献**：提出 DPO，证明可以跳过奖励模型直接从偏好数据优化策略

6. Schulman et al. (2015). [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438). ICLR 2016.
   - **贡献**：提出 GAE，解决策略梯度方法中偏差-方差权衡问题

### 教材与书籍

7. Sutton & Barto. [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html). MIT Press, 2018.
   - **章节**：第 13 章策略梯度方法，理解 REINFORCE 和基线

### 在线资源与教程

8. Hugging Face. [RLHF 完整教程](https://huggingface.co/blog/rlhf).
   - **内容**：RLHF 三阶段流程的直观解释和代码示例

9. Chip Huyen. [RLHF: Reinforcement Learning from Human Feedback](https://huyenchip.com/2023/05/02/rlhf.html).
   - **内容**：RLHF 的工程实践、挑战和开源生态系统分析

10. CarperAI. [trlX: Transformer Reinforcement Learning X](https://github.com/CarperAI/trlx).
    - **内容**：开源 RLHF 训练框架，支持 PPO 和 ILQL

11. Hugging Face. [TRL: Transformer Reinforcement Learning](https://github.com/huggingface/trl).
    - **内容**：Hugging Face 官方 RLHF/DPO 训练库

---

## 附录：符号表

| 符号 | 含义 | 维度/类型 |
|------|------|----------|
| $x$ | 输入 prompt | token 序列 |
| $y$ | 生成的回答 | token 序列 |
| $y_w$ | 人类偏好的回答（winning） | token 序列 |
| $y_l$ | 较差的回答（losing） | token 序列 |
| $y_t$ | 第 $t$ 个生成 token | 标量（token id） |
| $\pi_\theta(y \mid x)$ | 策略模型（语言模型） | 概率分布 |
| $\pi_{\text{ref}}(y \mid x)$ | 参考模型（冻结的 SFT 模型） | 概率分布 |
| $R_\phi(x, y)$ | 奖励模型输出 | 标量 |
| $V_\psi(s)$ | 价值函数估计 | 标量 |
| $\hat{A}_t$ | 优势估计 | 标量 |
| $\delta_t$ | TD 误差 | 标量 |
| $r_t(\theta)$ | 概率比（importance sampling ratio） | 标量 |
| $\beta$ | KL 惩罚系数 | 标量 |
| $\epsilon$ | PPO 截断参数 | 标量，通常 $0.2$ |
| $\gamma$ | 折扣因子 | 标量，RLHF 中通常 $1.0$ |
| $\lambda$ | GAE 参数 | 标量，通常 $0.95$ |
| $\sigma(\cdot)$ | Sigmoid 函数 | 函数 |
| $\text{KL}[\cdot \| \cdot]$ | KL 散度 | 标量 |
| $\mathcal{L}_{\text{SFT}}$ | SFT 损失值 | 标量 |
| $\mathcal{L}_{\text{RM}}$ | 奖励模型损失值 | 标量 |
| $\mathcal{L}_{\text{PPO}}$ | PPO 目标值 | 标量 |
| $\ell(\cdot, \cdot)$ | 损失函数 | 函数 |
| $\mathcal{D}$ | Prompt 数据集 | 集合 |
| $\mathcal{V}$ | 词表 | 集合，$\|\mathcal{V}\|$ 为词表大小 |
| $d$ ($d_{\text{model}}$) | 模型隐藏维度 | 标量 |
| $W_r$ | 奖励投影权重 | $(1, d)$ |
| $h_T$ | 最后一个 token 的隐藏状态 | $(d,)$ |
| $Z(x)$ | 配分函数 | 标量 |

**典型维度示例（InstructGPT 175B）：**
- $d = 12{,}288$，$L = 96$，$n_h = 96$
- $|\mathcal{V}| = 50{,}257$
- SFT 数据：~13K 条 (prompt, response) 对
- RM 数据：~33K 条人类比较（$K = 4 \sim 9$ 路排序）
- PPO 数据：~31K 条 prompt
- RM 模型：6B 参数
- 策略模型：1.3B / 6B / 175B 参数

---

最后更新：2026-03-19
