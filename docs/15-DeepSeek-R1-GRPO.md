# DeepSeek-R1 与 GRPO —— 通过强化学习激发大模型推理能力的完整数学推导

> **前置知识**：Transformer 架构、PPO/RLHF 基础（奖励模型、策略梯度）、语言模型预训练、Python 基础  
> **与前面内容的联系**：建议先学习 [RLHF-Math-and-Implementation](./13-RLHF-Math-and-Implementation.md) 理解 PPO 与奖励模型训练，以及 [LLaMA-Architecture-and-Implementation](./14-LLaMA-Architecture-and-Implementation.md) 理解开源大模型架构  
> **本笔记的定位**：本系列最终篇（07-BERT → 15-DeepSeek-R1），从对齐训练走向推理能力的前沿

---

## 目录

1. [引言：为什么需要推理模型？](#1-引言为什么需要推理模型)
   - 1.1 [大模型的推理短板](#11-大模型的推理短板)
   - 1.2 [从"对齐"到"推理"：范式转变](#12-从对齐到推理范式转变)
   - 1.3 [DeepSeek-R1 的核心创新概览](#13-deepseek-r1-的核心创新概览)
   - 1.4 [本科数学知识映射表](#14-本科数学知识映射表)
2. [基础概念：从 RLHF 到推理强化学习](#2-基础概念从-rlhf-到推理强化学习)
   - 2.1 [RLHF 的回顾与局限](#21-rlhf-的回顾与局限)
   - 2.2 [PPO 的价值模型瓶颈](#22-ppo-的价值模型瓶颈)
   - 2.3 [推理任务的特殊性：可验证奖励](#23-推理任务的特殊性可验证奖励)
   - 2.4 [DeepSeek-R1 的训练流水线总览](#24-deepseek-r1-的训练流水线总览)
3. [核心算法：GRPO（Group Relative Policy Optimization）](#3-核心算法grpogroup-relative-policy-optimization)
   - 3.1 [策略梯度回顾：REINFORCE 与 PPO](#31-策略梯度回顾reinforce-与-ppo)
   - 3.2 [GRPO 的核心思想：组内相对优势](#32-grpo-的核心思想组内相对优势)
   - 3.3 [GRPO 目标函数的完整推导](#33-grpo-目标函数的完整推导)
   - 3.4 [GRPO 与 PPO 的数学对比](#34-grpo-与-ppo-的数学对比)
4. [梯度推导与参数更新](#4-梯度推导与参数更新)
   - 4.1 [GRPO 策略梯度的完整推导](#41-grpo-策略梯度的完整推导)
   - 4.2 [KL 散度惩罚项的梯度](#42-kl-散度惩罚项的梯度)
   - 4.3 [Clip 机制的梯度分析](#43-clip-机制的梯度分析)
   - 4.4 [完整参数更新规则](#44-完整参数更新规则)
5. [训练优化方法总结](#5-训练优化方法总结)
   - 5.1 [冷启动数据构建](#51-冷启动数据构建)
   - 5.2 [自验证机制（Self-Verification）](#52-自验证机制self-verification)
   - 5.3 [多阶段训练策略](#53-多阶段训练策略)
   - 5.4 [奖励函数设计](#54-奖励函数设计)
   - 5.5 [拒绝采样与数据蒸馏](#55-拒绝采样与数据蒸馏)
6. [从数学到代码：完整实现](#6-从数学到代码完整实现)
   - 6.1 [NumPy 实现 GRPO 核心组件](#61-numpy-实现-grpo-核心组件)
   - 6.2 [PyTorch 完整 GRPO 训练器](#62-pytorch-完整-grpo-训练器)
7. [实践技巧与可视化](#7-实践技巧与可视化)
   - 7.1 [GRPO 训练动态可视化](#71-grpo-训练动态可视化)
   - 7.2 [组大小与方差分析](#72-组大小与方差分析)
   - 7.3 [推理 Token 长度与性能关系](#73-推理-token-长度与性能关系)
8. [与其他模型的关系](#8-与其他模型的关系)
   - 8.1 [从 RLHF 到推理 RL 的演进](#81-从-rlhf-到推理-rl-的演进)
   - 8.2 [DeepSeek-R1 vs OpenAI o1](#82-deepseek-r1-vs-openai-o1)
   - 8.3 [推理模型谱系](#83-推理模型谱系)

[扩展阅读与实现](#扩展阅读与实现)

[参考资源](#参考资源)

附录：[符号表](#附录符号表)

---

## 1. 引言：为什么需要推理模型？

### 1.1 大模型的推理短板

尽管 GPT-4、Claude 等大模型在语言理解和生成上表现优异，它们在**多步推理**任务中仍然存在系统性短板：

| 任务类型 | 示例 | GPT-4 (2023) | 人类专家 |
|----------|------|:------------:|:--------:|
| 数学竞赛 (AIME) | 15 道题 | 4/15 (26.7%) | 12/15 (80%) |
| 代码竞赛 (Codeforces) | Rating | ~1200 | ~2000+ |
| 形式逻辑推理 | 多步演绎 | ~60% | ~95% |
| 科学计算 | 物理/化学 | ~55% | ~85% |

> **核心问题**：标准语言模型是"一步到位"的——它在每个 token 位置用固定计算量生成输出，**无法动态分配更多计算给更难的问题**。

这与人类推理的方式形成鲜明对比：

$$
\boxed{\text{人类推理} = \text{思考}(\text{可变时间}) + \text{验证} + \text{回溯修正}}
$$

$$
\text{标准 LLM} = \text{逐 token 生成}(\text{固定计算量/token})
$$

### 1.2 从"对齐"到"推理"：范式转变

RLHF（InstructGPT, 2022）的核心目标是**对齐**——让模型输出符合人类偏好。而 DeepSeek-R1 的目标是**推理**——让模型学会思考过程本身。

```
对齐训练 (RLHF)                    推理训练 (DeepSeek-R1)
│                                   │
│ 目标: 输出质量↑                    │ 目标: 推理能力↑
│ 奖励: 人类偏好模型                 │ 奖励: 答案正确性 (可验证)
│ 算法: PPO (需要价值模型)           │ 算法: GRPO (无需价值模型)
│ 数据: 人类标注排序                 │ 数据: 数学/代码 (自动验证)
│ 输出: 直接回答                     │ 输出: 思维链 (CoT) + 回答
│                                   │
└── 让模型"说人话"                  └── 让模型"会思考"
```

### 1.3 DeepSeek-R1 的核心创新概览

DeepSeek-R1（DeepSeek, 2025）提出了一条从纯 RL 激发推理能力的技术路线：

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DeepSeek-R1 核心创新                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. GRPO: Group Relative Policy Optimization                       │
│  ┌──────────────────────────────────────────┐                       │
│  │  去除价值模型 → 节省 50%+ 训练资源         │                       │
│  │  组内相对优势估计 → 低方差梯度             │                       │
│  └──────────────────────────────────────────┘                       │
│                                                                     │
│  2. 冷启动数据构建                                                   │
│  ┌──────────────────────────────────────────┐                       │
│  │  少量高质量 CoT 数据 → 稳定 RL 起点        │                       │
│  │  多读一遍、反思模板 → 引导推理格式          │                       │
│  └──────────────────────────────────────────┘                       │
│                                                                     │
│  3. 自验证与推理涌现                                                  │
│  ┌──────────────────────────────────────────┐                       │
│  │  RL 过程中自发出现反思/验证行为             │                       │
│  │  "aha moment" → 模型自我纠正              │                       │
│  └──────────────────────────────────────────┘                       │
│                                                                     │
│  4. 蒸馏：大模型推理能力 → 小模型                                     │
│  ┌──────────────────────────────────────────┐                       │
│  │  DeepSeek-R1 (671B) → Qwen-1.5B/7B/32B  │                       │
│  │  蒸馏后小模型超越同规模 RL 训练            │                       │
│  └──────────────────────────────────────────┘                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.4 本科数学知识映射表

| DeepSeek-R1 概念 | 对应数学 | 本科课程 |
|------------------|----------|----------|
| GRPO 目标函数 | 期望、方差、优势函数 | 概率论、统计学 |
| 策略梯度 | 对数求导技巧、链式法则 | 高等数学 |
| KL 散度 | 相对熵、信息论 | 概率论、信息论 |
| Clip 机制 | 分段函数、min/max | 数学分析 |
| 组内归一化 | 均值、标准差、z-score | 统计学 |
| 蒙特卡洛估计 | 大数定律、中心极限定理 | 概率论 |
| 奖励函数设计 | 函数建模、正则化 | 优化理论 |

---

## 2. 基础概念：从 RLHF 到推理强化学习

### 2.1 RLHF 的回顾与局限

InstructGPT/ChatGPT 的 RLHF 训练三阶段（参见 [13-RLHF](./13-RLHF-Math-and-Implementation.md)）：

1. **SFT**（监督微调）：在人类标注数据上微调
2. **RM**（奖励模型训练）：学习人类偏好排序
3. **PPO**（近端策略优化）：用奖励模型指导策略优化

PPO 的优化目标：

$$
\mathcal{J}_{\text{PPO}}(\theta) = \mathbb{E}_{(s,a) \sim \pi_{\theta_{\text{old}}}} \left[ \min\left( r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
$$

其中 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ 是重要性比率，$\hat{A}_t$ 是优势函数估计。

**RLHF 的局限性**：

| 问题 | 具体表现 | 影响 |
|------|----------|------|
| **价值模型开销** | 需要与策略模型同等规模的 Critic | 训练内存/计算翻倍 |
| **奖励模型偏差** | RM 是人类偏好的近似，有系统偏差 | 奖励 hacking |
| **推理能力不足** | 优化"听起来对"而非"真的对" | 推理幻觉 |
| **标注成本高** | 需要大量人类偏好数据 | 扩展困难 |

### 2.2 PPO 的价值模型瓶颈

PPO 需要一个**价值模型**（Critic）来估计优势函数 $\hat{A}_t$：

$$
\hat{A}_t^{\text{GAE}} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

其中 $V(s)$ 是价值函数，需要训练一个与策略模型同等规模的神经网络来拟合。

**价值模型的问题**：

1. **内存开销**：对于 67B 参数的策略模型，价值模型也需要 ~67B 参数
2. **训练不稳定**：价值函数的估计误差直接影响策略梯度质量
3. **奖励尺度敏感**：不同任务的奖励范围差异大，需要精细调节

$$
\boxed{\text{PPO 总资源} = \underbrace{\text{策略模型}}_{\sim N \text{ 参数}} + \underbrace{\text{价值模型}}_{\sim N \text{ 参数}} + \underbrace{\text{奖励模型}}_{\sim N \text{ 参数}} + \underbrace{\text{参考模型}}_{\sim N \text{ 参数}} \approx 4N}
$$

### 2.3 推理任务的特殊性：可验证奖励

推理任务（数学、编程、逻辑）具有一个独特优势：**答案可以自动验证**。

$$
r(y, y^*) = \begin{cases}
1 & \text{如果答案正确（如 } \texttt{extract\_answer}(y) = y^* \text{）} \\
0 & \text{否则}
\end{cases}
$$

这意味着：
- **不需要奖励模型**：正确性可以通过规则判断
- **无奖励偏差**：验证结果是精确的，没有近似误差
- **可无限扩展**：可以自动生成数学题/编程题并验证

> **关键洞察**：对于推理任务，奖励信号是"免费"的——不需要训练奖励模型，只需要一个验证器（verifier）。

### 2.4 DeepSeek-R1 的训练流水线总览

DeepSeek-R1 的完整训练流程：

```
阶段 1: 冷启动 (Cold Start)
    基础模型 (DeepSeek-V3-Base)
         │
         ↓ SFT on 数千条高质量 CoT 数据
         │
    DeepSeek-R1-Zero-SFT (RL 起点)

阶段 2: 推理 RL (Reasoning RL)
    DeepSeek-R1-Zero-SFT
         │
         ↓ GRPO + 规则奖励 (数学/代码正确性)
         │ 多轮 RL 训练，推理能力涌现
         │
    DeepSeek-R1-RL-Checkpoint

阶段 3: 拒绝采样 + SFT
    DeepSeek-R1-RL-Checkpoint
         │
         ↓ 大量采样 → 过滤正确答案 → 收集推理数据
         ↓ 加入通用 SFT 数据 (写作、问答等)
         │
    DeepSeek-R1-SFT

阶段 4: 最终 RL 对齐
    DeepSeek-R1-SFT
         │
         ↓ GRPO (推理 + 通用对齐)
         │
    DeepSeek-R1 (最终模型)
```

---

## 3. 核心算法：GRPO（Group Relative Policy Optimization）

### 3.1 策略梯度回顾：REINFORCE 与 PPO

**REINFORCE**（最基础的策略梯度）：

$$
\nabla_\theta \mathcal{J}(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot R(\tau) \right]
$$

- 优点：无偏估计
- 缺点：**方差极高**（$R(\tau)$ 的绝对值直接影响梯度方向和大小）

**带基线的 REINFORCE**：

$$
\nabla_\theta \mathcal{J}(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot (R(\tau) - b) \right]
$$

基线 $b$ 不影响梯度的期望（因为 $\mathbb{E}[\nabla_\theta \log \pi \cdot b] = 0$），但降低方差。

**PPO** 使用 GAE 优势估计 + clip：

$$
\hat{A}_t^{\text{GAE}} = \sum_{l=0}^{\infty}(\gamma\lambda)^l (r_{t+l} + \gamma V(s_{t+l+1}) - V(s_{t+l}))
$$

需要训练价值函数 $V(s)$——这就是 GRPO 想要去除的部分。

### 3.2 GRPO 的核心思想：组内相对优势

GRPO（Shao et al., 2024）的核心洞察：

> **不需要价值模型来估计优势——对同一问题采样多个回答，用组内相对排名来估计优势。**

**直觉**：给定一个问题 $q$，让模型生成 $G$ 个回答 $\{o_1, o_2, \ldots, o_G\}$，每个回答获得奖励 $\{r_1, r_2, \ldots, r_G\}$。

与其用价值模型估计每个回答的优势，不如直接用**组内 z-score 归一化**：

$$
\boxed{\hat{A}_i = \frac{r_i - \text{mean}(\{r_1, \ldots, r_G\})}{\text{std}(\{r_1, \ldots, r_G\})}}
$$

**为什么这样做有效？**

1. **自动基线**：组内均值自然成为基线，无需额外训练
2. **尺度不变**：z-score 归一化消除了奖励绝对值的影响
3. **无需价值模型**：省去了一个与策略模型同等大小的网络

```
PPO 优势估计:                     GRPO 优势估计:
┌────────────┐                   ┌────────────────────────────┐
│ 策略模型    │ → 生成回答        │ 策略模型 → 对同一问题生成 G 个回答 │
│ 价值模型    │ → V(s)           │ 奖励评估 → r_1, r_2, ..., r_G   │
│ GAE 计算    │ → Â_t            │ z-score  → Â_i = (r_i - μ) / σ  │
│ (需要额外N参数) │               │ (不需要额外参数!)                 │
└────────────┘                   └────────────────────────────┘
```

### 3.3 GRPO 目标函数的完整推导

**设定**：
- $\pi_\theta$：当前策略（待优化）
- $\pi_{\theta_{\text{old}}}$：旧策略（用于采样）
- $\pi_{\text{ref}}$：参考策略（SFT 模型，用于 KL 约束）
- $q$：输入问题
- $o_i = (o_{i,1}, o_{i,2}, \ldots, o_{i,|o_i|})$：第 $i$ 个回答的 token 序列
- $G$：组大小（每个问题的采样数量）

**GRPO 目标函数**：

$$
\boxed{
\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{q \sim \mathcal{D}, \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(\cdot|q)} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left( \min\left( \rho_{i,t} \hat{A}_i, \; \text{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon) \hat{A}_i \right) - \beta \, D_{\text{KL}}^{(t)} \right) \right]
}
$$

其中：

**重要性比率**（token 级别）：

$$
\rho_{i,t} = \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})}
$$

**组内优势估计**：

$$
\hat{A}_i = \frac{r_i - \mu_r}{\sigma_r + \epsilon_{\text{norm}}}, \quad \mu_r = \frac{1}{G}\sum_{j=1}^G r_j, \quad \sigma_r = \sqrt{\frac{1}{G}\sum_{j=1}^G (r_j - \mu_r)^2}
$$

**KL 散度惩罚**（token 级别）：

$$
D_{\text{KL}}^{(t)} = \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\text{ref}}(o_{i,t} \mid q, o_{i,<t})} - \log \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\text{ref}}(o_{i,t} \mid q, o_{i,<t})} - 1
$$

> **注意**：这里使用的是 **KL 散度的近似形式**（非对称 KL 的一阶展开），而非完整的 KL 散度。令 $u = \frac{\pi_\theta}{\pi_{\text{ref}}}$，则 $D_{\text{KL}}^{(t)} = u - \log u - 1 \geq 0$，当且仅当 $u=1$（即 $\pi_\theta = \pi_{\text{ref}}$）时等号成立。

**目标函数各项的含义**：

| 项 | 公式 | 作用 |
|----|------|------|
| 策略比率 | $\rho_{i,t}$ | 重要性采样修正 |
| Clip | $\text{clip}(\rho_{i,t}, 1\!-\!\epsilon, 1\!+\!\epsilon)$ | 限制策略更新幅度 |
| 组内优势 | $\hat{A}_i$ | 无需价值模型的优势估计 |
| KL 惩罚 | $\beta D_{\text{KL}}^{(t)}$ | 防止策略偏离参考模型过远 |

### 3.4 GRPO 与 PPO 的数学对比

| 维度 | PPO (RLHF) | GRPO (DeepSeek-R1) |
|------|------------|---------------------|
| **优势估计** | $\hat{A}_t^{\text{GAE}} = \sum_l (\gamma\lambda)^l \delta_{t+l}$ | $\hat{A}_i = \frac{r_i - \mu_r}{\sigma_r}$ |
| **价值模型** | ✅ 需要（$\sim N$ 参数） | ❌ 不需要 |
| **优势粒度** | Token 级别 | 回答级别（整个回答共享同一优势） |
| **基线** | $V(s_t)$（学习得到） | $\mu_r$（组内均值） |
| **方差控制** | GAE ($\lambda$ 参数) | 组大小 $G$ + z-score 归一化 |
| **KL 约束** | 奖励中加 $-\beta \log \frac{\pi}{\pi_{\text{ref}}}$ | 目标函数中显式 KL 惩罚项 |
| **训练资源** | 4 个模型（策略+价值+奖励+参考） | 2-3 个模型（策略+参考+可选奖励） |

**关键数学差异**：

PPO 的优势是 **token 级别**的——每个 token 有不同的优势值：

$$
\hat{A}_t^{\text{PPO}} = f(r_t, V(s_t), V(s_{t+1}), \ldots)
$$

GRPO 的优势是 **回答级别**的——同一回答中所有 token 共享相同的优势值：

$$
\hat{A}_{i,t}^{\text{GRPO}} = \hat{A}_i \quad \forall t \in \{1, \ldots, |o_i|\}
$$

> **直觉**：GRPO 更像是"这个回答整体好不好"，而 PPO 是"每一步决策好不好"。对于推理任务，最终答案的正确性是最重要的信号，回答级别的优势足够有效。

---

## 4. 梯度推导与参数更新

### 4.1 GRPO 策略梯度的完整推导

为简化记号，考虑单个问题 $q$ 和单个回答 $o_i$ 中第 $t$ 个 token 的贡献。

**不考虑 clip 的基本梯度**：

$$
\nabla_\theta \left[ \rho_{i,t} \hat{A}_i \right] = \hat{A}_i \cdot \nabla_\theta \rho_{i,t}
$$

其中：

$$
\rho_{i,t} = \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})}
$$

由于 $\pi_{\theta_{\text{old}}}$ 不依赖 $\theta$（采样时固定）：

$$
\nabla_\theta \rho_{i,t} = \frac{\nabla_\theta \pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})}
$$

利用对数导数技巧 $\nabla_\theta \pi = \pi \cdot \nabla_\theta \log \pi$：

$$
\nabla_\theta \rho_{i,t} = \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})} \cdot \nabla_\theta \log \pi_\theta(o_{i,t} \mid q, o_{i,<t}) = \rho_{i,t} \cdot \nabla_\theta \log \pi_\theta(o_{i,t} \mid q, o_{i,<t})
$$

因此：

$$
\boxed{\nabla_\theta \left[ \rho_{i,t} \hat{A}_i \right] = \rho_{i,t} \cdot \hat{A}_i \cdot \nabla_\theta \log \pi_\theta(o_{i,t} \mid q, o_{i,<t})}
$$

**加入 clip 后**：

定义 clip 后的目标为：

$$
L_{i,t}^{\text{clip}} = \min\left( \rho_{i,t} \hat{A}_i, \; \text{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon) \hat{A}_i \right)
$$

其梯度为：

$$
\nabla_\theta L_{i,t}^{\text{clip}} = \begin{cases}
\rho_{i,t} \cdot \hat{A}_i \cdot \nabla_\theta \log \pi_\theta & \text{如果 } \rho_{i,t} \hat{A}_i \leq \text{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon) \hat{A}_i \\
0 & \text{否则（被 clip 阻止更新）}
\end{cases}
$$

> **Clip 的直觉**：当 $\hat{A}_i > 0$（好回答）且 $\rho_{i,t} > 1+\epsilon$（已经增大了太多概率），clip 阻止进一步增大。当 $\hat{A}_i < 0$（差回答）且 $\rho_{i,t} < 1-\epsilon$（已经减小了太多概率），clip 阻止进一步减小。

### 4.2 KL 散度惩罚项的梯度

KL 惩罚项：

$$
D_{\text{KL}}^{(t)} = u_{i,t} - \log u_{i,t} - 1, \quad u_{i,t} = \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\text{ref}}(o_{i,t} \mid q, o_{i,<t})}
$$

对 $\theta$ 求导（$\pi_{\text{ref}}$ 不依赖 $\theta$）：

$$
\frac{\partial D_{\text{KL}}^{(t)}}{\partial u_{i,t}} = 1 - \frac{1}{u_{i,t}} = 1 - \frac{\pi_{\text{ref}}}{\pi_\theta}
$$

$$
\frac{\partial u_{i,t}}{\partial \theta} = \frac{1}{\pi_{\text{ref}}} \cdot \nabla_\theta \pi_\theta = \frac{\pi_\theta}{\pi_{\text{ref}}} \cdot \nabla_\theta \log \pi_\theta = u_{i,t} \cdot \nabla_\theta \log \pi_\theta
$$

链式法则：

$$
\boxed{\nabla_\theta D_{\text{KL}}^{(t)} = \left(1 - \frac{\pi_{\text{ref}}(o_{i,t} \mid q, o_{i,<t})}{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}\right) \cdot \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\text{ref}}(o_{i,t} \mid q, o_{i,<t})} \cdot \nabla_\theta \log \pi_\theta(o_{i,t} \mid q, o_{i,<t})}
$$

简化：

$$
\nabla_\theta D_{\text{KL}}^{(t)} = \left(u_{i,t} - 1\right) \cdot \nabla_\theta \log \pi_\theta(o_{i,t} \mid q, o_{i,<t})
$$

> **直觉**：当 $\pi_\theta \gg \pi_{\text{ref}}$（$u \gg 1$）时，KL 梯度很大，强烈阻止偏离；当 $\pi_\theta \approx \pi_{\text{ref}}$（$u \approx 1$）时，KL 梯度接近零。

### 4.3 Clip 机制的梯度分析

Clip 机制的行为可以通过分情况讨论来理解：

**情况 1：$\hat{A}_i > 0$（好回答，应该增大概率）**

$$
L_{i,t}^{\text{clip}} = \min(\rho_{i,t}, 1+\epsilon) \cdot \hat{A}_i
$$

- 如果 $\rho_{i,t} \leq 1+\epsilon$：梯度正常传播，增大 $\pi_\theta$
- 如果 $\rho_{i,t} > 1+\epsilon$：梯度为零，**阻止过度增大**

**情况 2：$\hat{A}_i < 0$（差回答，应该减小概率）**

$$
L_{i,t}^{\text{clip}} = \max(\rho_{i,t}, 1-\epsilon) \cdot \hat{A}_i
$$

- 如果 $\rho_{i,t} \geq 1-\epsilon$：梯度正常传播，减小 $\pi_\theta$
- 如果 $\rho_{i,t} < 1-\epsilon$：梯度为零，**阻止过度减小**

$$
\boxed{\text{Clip 有效梯度} = \begin{cases}
\rho_{i,t} \hat{A}_i \nabla_\theta \log\pi_\theta & \text{if } \hat{A}_i > 0 \text{ and } \rho_{i,t} \leq 1+\epsilon \\
\rho_{i,t} \hat{A}_i \nabla_\theta \log\pi_\theta & \text{if } \hat{A}_i < 0 \text{ and } \rho_{i,t} \geq 1-\epsilon \\
0 & \text{otherwise}
\end{cases}}
$$

### 4.4 完整参数更新规则

综合策略梯度和 KL 惩罚，GRPO 的单步参数更新：

$$
\boxed{\theta \leftarrow \theta + \alpha \cdot \frac{1}{|\mathcal{B}|} \sum_{q \in \mathcal{B}} \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left( \nabla_\theta L_{i,t}^{\text{clip}} - \beta \nabla_\theta D_{\text{KL}}^{(t)} \right)}
$$

其中 $|\mathcal{B}|$ 是 mini-batch 中问题的数量，$\alpha$ 是学习率。

**完整的 GRPO 训练算法**：

```
输入: 策略模型 π_θ, 参考模型 π_ref, 数据集 D, 组大小 G
      学习率 α, clip 参数 ε, KL 系数 β

重复直到收敛:
    1. 从 D 采样一批问题 {q_1, ..., q_B}
    2. 对每个 q_j:
       a. 用 π_θ_old 生成 G 个回答: {o_1, ..., o_G} ~ π_θ_old(·|q_j)
       b. 计算每个回答的奖励: r_i = R(q_j, o_i)
       c. 计算组内优势: Â_i = (r_i - mean(r)) / (std(r) + ε_norm)
    3. 对 {(q_j, o_i, Â_i)} 执行多个 epoch 的梯度更新:
       a. 计算 ρ_i,t = π_θ(o_i,t|q, o_i,<t) / π_θ_old(o_i,t|q, o_i,<t)
       b. 计算 clip 目标 L^clip
       c. 计算 KL 惩罚 D_KL
       d. 更新: θ ← θ + α · ∇_θ(L^clip - β·D_KL)
    4. 更新旧策略: θ_old ← θ
```

---

## 5. 训练优化方法总结

### 5.1 冷启动数据构建

**问题**：直接对基础模型做 RL 训练，模型可能产生格式混乱、无法解析的输出（DeepSeek-R1-Zero 的问题）。

**解决方案**：用少量高质量数据做冷启动 SFT。

**冷启动数据来源**：

| 来源 | 数量 | 方法 |
|------|:----:|------|
| 人工标注 | ~数百条 | 数学/编程专家撰写详细推理过程 |
| Few-shot 生成 | ~数千条 | 用基础模型在 few-shot 提示下生成，人工过滤 |
| 格式模板 | — | 定义 `<think>...</think>` 推理格式 |

**冷启动数据的格式要求**：

```
问题: [数学/编程题目]
<think>
[推理过程]
- 分析条件
- 建立方程
- 逐步求解
- 验证答案
</think>
最终答案: [结果]
```

> **关键原则**：冷启动数据不需要很多，但质量必须高——它定义了 RL 训练的起点格式。

$$
\boxed{\text{冷启动} = \text{少量高质量 CoT 数据} + \text{SFT} \rightarrow \text{稳定的 RL 起点}}
$$

### 5.2 自验证机制（Self-Verification）

DeepSeek-R1 训练过程中观察到的一个重要涌现行为——**自验证**：

模型在 RL 训练中自发学会了以下行为模式：

1. **回溯检查**（Backtracking）：
   ```
   <think>
   让我解这个方程... x = 5
   等一下，让我验证一下：代入原方程 2(5) + 3 = 13 ≠ 15
   所以 x = 5 是错的，重新计算...
   x = 6, 验证: 2(6) + 3 = 15 ✓
   </think>
   ```

2. **多路径探索**：
   ```
   <think>
   方法一: 直接求解... 得到 x = 7
   方法二: 用另一种方法验证... 也得到 x = 7
   两种方法一致，答案可靠
   </think>
   ```

3. **"Aha Moment"**（顿悟时刻）：
   模型在 RL 过程中出现了类似人类"恍然大悟"的行为——先给出错误答案，然后自我纠正。

> **数学解释**：自验证行为被 RL 奖励信号强化。当模型验证后得到正确答案（$r=1$），包含验证步骤的推理路径获得正优势 $\hat{A}_i > 0$，其概率被增大。

$$
\text{自验证涌现} = \underbrace{\text{RL 奖励选择}}_{\text{正确答案 } r=1} + \underbrace{\text{策略梯度强化}}_{\hat{A}_i > 0 \Rightarrow \uparrow P(\text{验证路径})}
$$

### 5.3 多阶段训练策略

DeepSeek-R1 的四阶段训练细节：

| 阶段 | 输入模型 | 训练方法 | 数据规模 | 输出模型 | 目标 |
|:----:|----------|----------|:--------:|----------|------|
| 1 | V3-Base | SFT | ~数千条 | R1-Zero-SFT | 格式初始化 |
| 2 | R1-Zero-SFT | GRPO | ~数十万题 | R1-RL | 推理能力涌现 |
| 3 | R1-RL | SFT | ~80万条 | R1-SFT | 推理+通用能力 |
| 4 | R1-SFT | GRPO | 混合数据 | R1 (最终) | 综合对齐 |

**阶段 3 的数据构成**：

$$
\mathcal{D}_{\text{SFT}} = \underbrace{\mathcal{D}_{\text{reasoning}}}_{\text{~60万条推理数据}} \cup \underbrace{\mathcal{D}_{\text{general}}}_{\text{~20万条通用数据}}
$$

其中推理数据通过**拒绝采样**（Rejection Sampling）获得：

$$
\mathcal{D}_{\text{reasoning}} = \{(q, o^*) \mid o^* = \arg\max_{o \in \{o_1, \ldots, o_K\}} r(o, y_q), \; r(o^*, y_q) = 1\}
$$

即对每个问题 $q$ 采样 $K$ 个回答，只保留正确的。

### 5.4 奖励函数设计

DeepSeek-R1 使用**基于规则的奖励**（Rule-based Reward），而非学习的奖励模型：

**数学题奖励**：

$$
r_{\text{math}}(o, y^*) = \begin{cases}
1 & \text{if } \texttt{extract\_answer}(o) = y^* \\
0 & \text{otherwise}
\end{cases}
$$

**编程题奖励**：

$$
r_{\text{code}}(o, \mathcal{T}) = \frac{|\{t \in \mathcal{T} : \texttt{pass}(o, t)\}|}{|\mathcal{T}|}
$$

其中 $\mathcal{T}$ 是测试用例集合。

**格式奖励**（鼓励使用推理格式）：

$$
r_{\text{format}}(o) = \begin{cases}
r_{\text{bonus}} & \text{if } o \text{ 包含正确的 } \texttt{<think>...</think>} \text{ 格式} \\
0 & \text{otherwise}
\end{cases}
$$

**综合奖励**：

$$
\boxed{r(o, y^*) = r_{\text{accuracy}}(o, y^*) + \lambda_f \cdot r_{\text{format}}(o)}
$$

> **为什么不用学习的奖励模型？** 对于推理任务，规则奖励是精确的——答案要么对要么错。学习的奖励模型反而会引入偏差（reward hacking）。

### 5.5 拒绝采样与数据蒸馏

**拒绝采样（Rejection Sampling）**是连接 RL 阶段和 SFT 阶段的桥梁：

$$
\text{Rejection Sampling}: \quad o^* \sim \pi_\theta(\cdot | q) \text{ s.t. } r(o^*, y_q) = 1
$$

实践中，对每个问题采样 $K$ 次（如 $K=64$），保留正确的回答。

**蒸馏（Distillation）**——将大模型的推理能力转移到小模型：

$$
\mathcal{L}_{\text{distill}} = \mathbb{E}_{q \sim \mathcal{D}} \left[ -\sum_{t=1}^{|o^*|} \log \pi_{\theta_{\text{small}}}(o^*_t \mid q, o^*_{<t}) \right]
$$

其中 $o^*$ 是大模型（DeepSeek-R1-671B）生成的正确推理过程。

**蒸馏效果**：

| 小模型 | 直接 RL | 蒸馏 (from R1-671B) | 提升 |
|--------|:-------:|:-------------------:|:----:|
| Qwen-1.5B | 15.6% (AIME) | 28.9% | +85% |
| Qwen-7B | 55.5% (MATH-500) | 83.9% | +51% |
| Qwen-32B | 72.6% (MATH-500) | 94.3% | +30% |

$$
\boxed{\text{蒸馏} > \text{同规模 RL 训练} \quad \text{（在当前数据规模下）}}
$$

---

## 6. 从数学到代码：完整实现

### 6.1 NumPy 实现 GRPO 核心组件

#### 6.1.1 组内优势估计

```python
import numpy as np

def compute_group_advantages(rewards, eps=1e-8):
    """
    GRPO 核心: 组内相对优势估计
    
    数学:
        Â_i = (r_i - mean(r)) / (std(r) + ε)
    
    参数:
        rewards: shape (B, G) — B 个问题, 每个 G 个回答的奖励
        eps:     数值稳定常数
    返回:
        advantages: shape (B, G) — 归一化后的优势值
    """
    # 组内均值和标准差
    mu = np.mean(rewards, axis=-1, keepdims=True)      # (B, 1)
    sigma = np.std(rewards, axis=-1, keepdims=True)     # (B, 1)
    
    # z-score 归一化
    advantages = (rewards - mu) / (sigma + eps)          # (B, G)
    
    return advantages

# ===== 验证 =====
np.random.seed(42)
B, G = 4, 8  # 4 个问题, 每个 8 个回答

# 模拟奖励: 0/1 二值 (数学题正确性)
rewards = np.random.choice([0.0, 1.0], size=(B, G), p=[0.6, 0.4])
advantages = compute_group_advantages(rewards)

print("GRPO 组内优势估计:")
print(f"奖励 (前2个问题):\n{rewards[:2]}")
print(f"优势 (前2个问题):\n{advantages[:2]}")
print(f"\n验证: 每组优势均值 ≈ 0: {np.mean(advantages, axis=-1)}")
print(f"验证: 每组优势标准差 ≈ 1: {np.std(advantages, axis=-1)}")
```

#### 6.1.2 重要性比率与 Clip

```python
def compute_importance_ratio(log_probs_new, log_probs_old):
    """
    计算重要性采样比率
    
    数学:
        ρ_{i,t} = π_θ(o_{i,t}|q, o_{i,<t}) / π_{θ_old}(o_{i,t}|q, o_{i,<t})
               = exp(log π_θ - log π_{θ_old})
    
    参数:
        log_probs_new: shape (...,) — 新策略的 log 概率
        log_probs_old: shape (...,) — 旧策略的 log 概率
    返回:
        ratio: shape (...,) — 重要性比率
    """
    return np.exp(log_probs_new - log_probs_old)

def compute_clipped_objective(ratio, advantages, epsilon=0.2):
    """
    PPO/GRPO 的 clip 目标
    
    数学:
        L^clip = min(ρ·Â, clip(ρ, 1-ε, 1+ε)·Â)
    
    参数:
        ratio:      shape (B, G, T) — 重要性比率
        advantages: shape (B, G) 或 (B, G, 1) — 优势值
        epsilon:    clip 范围
    返回:
        objective:  shape (B, G, T) — clip 后的目标值
    """
    # 确保 advantages 可广播
    if advantages.ndim < ratio.ndim:
        advantages = np.expand_dims(advantages, axis=-1)  # (B, G, 1)
    
    # 未 clip 的目标
    unclipped = ratio * advantages
    
    # clip 后的目标
    clipped_ratio = np.clip(ratio, 1.0 - epsilon, 1.0 + epsilon)
    clipped = clipped_ratio * advantages
    
    # 取较小值 (悲观估计)
    objective = np.minimum(unclipped, clipped)
    
    return objective

# ===== 验证 clip 行为 =====
print("\nClip 机制行为验证:")
ratios = np.array([0.5, 0.8, 1.0, 1.2, 1.5, 2.0])
eps = 0.2

for A in [1.0, -1.0]:
    print(f"\n优势 Â = {A}:")
    print(f"{'ratio':>8} {'unclip':>10} {'clip':>10} {'result':>10} {'梯度':>8}")
    print("-" * 50)
    for r in ratios:
        unclip = r * A
        clip_r = np.clip(r, 1-eps, 1+eps)
        clip_val = clip_r * A
        result = min(unclip, clip_val)
        grad = "有" if result == unclip else "零"
        print(f"{r:>8.2f} {unclip:>10.2f} {clip_val:>10.2f} {result:>10.2f} {grad:>8}")
```

#### 6.1.3 KL 散度惩罚

```python
def compute_kl_penalty(log_probs_policy, log_probs_ref):
    """
    GRPO 的 KL 散度惩罚 (近似形式)
    
    数学:
        D_KL^(t) = u - log(u) - 1
        其中 u = π_θ / π_ref = exp(log π_θ - log π_ref)
    
    性质:
        - D_KL ≥ 0 (非负)
        - D_KL = 0 当且仅当 π_θ = π_ref
    
    参数:
        log_probs_policy: shape (...,) — 策略模型的 log 概率
        log_probs_ref:    shape (...,) — 参考模型的 log 概率
    返回:
        kl: shape (...,) — KL 散度值
    """
    log_ratio = log_probs_policy - log_probs_ref  # log(u)
    u = np.exp(log_ratio)                          # u = π_θ / π_ref
    kl = u - log_ratio - 1.0                       # u - log(u) - 1
    return kl

# ===== 验证 KL 散度性质 =====
print("\nKL 散度惩罚验证:")
log_ratios = np.linspace(-2, 2, 9)
print(f"{'log(π/π_ref)':>14} {'u=π/π_ref':>10} {'KL':>10}")
print("-" * 38)
for lr in log_ratios:
    u = np.exp(lr)
    kl = u - lr - 1.0
    print(f"{lr:>14.2f} {u:>10.4f} {kl:>10.4f}")

print(f"\n验证: KL ≥ 0 (最小值在 u=1, 即 log_ratio=0)")
print(f"最小 KL = {np.exp(0) - 0 - 1:.6f} (应为 0)")
```

#### 6.1.4 完整 GRPO 损失计算

```python
def grpo_loss_numpy(log_probs_new, log_probs_old, log_probs_ref,
                    rewards, mask=None, epsilon=0.2, beta=0.01, 
                    eps_norm=1e-8):
    """
    GRPO 完整损失函数 (NumPy 实现)
    
    数学:
        L = 1/(B·G) Σ_q Σ_i 1/|o_i| Σ_t [min(ρ·Â, clip(ρ)·Â) - β·D_KL]
    
    参数:
        log_probs_new: (B, G, T) — 新策略的 token log 概率
        log_probs_old: (B, G, T) — 旧策略的 token log 概率
        log_probs_ref: (B, G, T) — 参考模型的 token log 概率
        rewards:       (B, G)    — 每个回答的奖励
        mask:          (B, G, T) — padding mask (1=有效, 0=padding)
        epsilon:       clip 范围
        beta:          KL 系数
        eps_norm:      归一化稳定常数
    返回:
        loss: 标量 — 需要最大化的目标 (取负为损失)
        info: 字典 — 调试信息
    """
    B, G, T = log_probs_new.shape
    
    if mask is None:
        mask = np.ones((B, G, T))
    
    # 1. 组内优势估计
    advantages = compute_group_advantages(rewards, eps=eps_norm)  # (B, G)
    
    # 2. 重要性比率
    ratio = compute_importance_ratio(log_probs_new, log_probs_old)  # (B, G, T)
    
    # 3. Clip 目标
    clip_obj = compute_clipped_objective(ratio, advantages, epsilon)  # (B, G, T)
    
    # 4. KL 惩罚
    kl = compute_kl_penalty(log_probs_new, log_probs_ref)  # (B, G, T)
    
    # 5. 逐 token 目标
    per_token_obj = clip_obj - beta * kl  # (B, G, T)
    
    # 6. 应用 mask 并按回答长度平均
    per_token_obj = per_token_obj * mask
    seq_lengths = np.sum(mask, axis=-1, keepdims=True).clip(min=1)  # (B, G, 1)
    per_response_obj = np.sum(per_token_obj, axis=-1) / seq_lengths.squeeze(-1)  # (B, G)
    
    # 7. 全局平均
    objective = np.mean(per_response_obj)
    loss = -objective  # 最大化目标 = 最小化负目标
    
    info = {
        'objective': objective,
        'mean_advantage': np.mean(advantages),
        'mean_ratio': np.mean(ratio * mask) / np.mean(mask),
        'mean_kl': np.mean(kl * mask) / np.mean(mask),
        'clip_fraction': np.mean((np.abs(ratio - 1.0) > epsilon) * mask) / np.mean(mask),
    }
    
    return loss, info

# ===== 验证 =====
np.random.seed(42)
B, G, T = 4, 8, 16

# 模拟数据
log_probs_new = np.random.randn(B, G, T) * 0.1 - 3.0  # log 概率 (负值)
log_probs_old = log_probs_new + np.random.randn(B, G, T) * 0.05
log_probs_ref = log_probs_new + np.random.randn(B, G, T) * 0.1
rewards = np.random.choice([0.0, 1.0], size=(B, G), p=[0.5, 0.5])
mask = np.ones((B, G, T))
mask[:, :, -3:] = 0  # 模拟 padding

loss, info = grpo_loss_numpy(log_probs_new, log_probs_old, log_probs_ref,
                              rewards, mask)

print("GRPO 损失计算结果:")
for k, v in info.items():
    print(f"  {k}: {v:.6f}")
print(f"  loss: {loss:.6f}")
```

### 6.2 PyTorch 完整 GRPO 训练器

#### 6.2.1 GRPO 损失模块

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GRPOLoss(nn.Module):
    """
    GRPO (Group Relative Policy Optimization) 损失函数
    
    数学:
        L = -E[1/G Σ_i 1/|o_i| Σ_t (min(ρ·Â, clip(ρ)·Â) - β·D_KL)]
        
        ρ = π_θ / π_θ_old
        Â_i = (r_i - mean(r)) / (std(r) + ε)
        D_KL = π_θ/π_ref - log(π_θ/π_ref) - 1
    """
    
    def __init__(self, epsilon: float = 0.2, beta: float = 0.01,
                 eps_norm: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon      # clip 范围
        self.beta = beta            # KL 系数
        self.eps_norm = eps_norm    # 归一化稳定常数
    
    def compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        组内优势估计
        
        参数: rewards — (B, G)
        返回: advantages — (B, G)
        """
        mean_r = rewards.mean(dim=-1, keepdim=True)
        std_r = rewards.std(dim=-1, keepdim=True)
        return (rewards - mean_r) / (std_r + self.eps_norm)
    
    def forward(self, log_probs: torch.Tensor, 
                log_probs_old: torch.Tensor,
                log_probs_ref: torch.Tensor,
                rewards: torch.Tensor,
                mask: torch.Tensor) -> dict:
        """
        参数:
            log_probs:     (B, G, T) — 当前策略的 token log 概率
            log_probs_old: (B, G, T) — 旧策略的 token log 概率
            log_probs_ref: (B, G, T) — 参考模型的 token log 概率
            rewards:       (B, G)    — 每个回答的奖励
            mask:          (B, G, T) — 有效 token 掩码
        返回:
            字典: loss, objective, kl, clip_fraction, advantages
        """
        B, G, T = log_probs.shape
        
        # 1. 组内优势
        advantages = self.compute_advantages(rewards)  # (B, G)
        adv = advantages.unsqueeze(-1)                  # (B, G, 1)
        
        # 2. 重要性比率
        ratio = torch.exp(log_probs - log_probs_old)   # (B, G, T)
        
        # 3. Clip 目标
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1.0 - self.epsilon, 
                              1.0 + self.epsilon) * adv
        policy_obj = torch.min(unclipped, clipped)      # (B, G, T)
        
        # 4. KL 散度惩罚
        log_ratio_ref = log_probs - log_probs_ref
        u = torch.exp(log_ratio_ref)
        kl = u - log_ratio_ref - 1.0                    # (B, G, T)
        
        # 5. 逐 token 总目标
        per_token = policy_obj - self.beta * kl          # (B, G, T)
        
        # 6. 按回答长度平均
        per_token = per_token * mask
        seq_len = mask.sum(dim=-1).clamp(min=1)          # (B, G)
        per_response = per_token.sum(dim=-1) / seq_len   # (B, G)
        
        # 7. 全局平均
        objective = per_response.mean()
        loss = -objective  # 最大化目标
        
        # 统计信息
        with torch.no_grad():
            mean_kl = (kl * mask).sum() / mask.sum()
            clip_frac = ((torch.abs(ratio - 1.0) > self.epsilon).float() 
                         * mask).sum() / mask.sum()
        
        return {
            'loss': loss,
            'objective': objective.detach(),
            'mean_kl': mean_kl,
            'clip_fraction': clip_frac,
            'mean_advantage': advantages.mean().detach(),
            'mean_reward': rewards.mean().detach(),
        }
```

#### 6.2.2 GRPO 训练器

```python
class GRPOTrainer:
    """
    GRPO 训练器 — 完整的训练循环实现
    
    训练流程:
        1. 对每个问题, 用旧策略采样 G 个回答
        2. 计算每个回答的奖励 (规则验证)
        3. 计算组内优势
        4. 执行多个 epoch 的梯度更新
        5. 更新旧策略
    """
    
    def __init__(self, policy_model, ref_model, tokenizer,
                 reward_fn, group_size=8, lr=1e-6,
                 epsilon=0.2, beta=0.01, max_gen_len=512,
                 num_epochs_per_update=1):
        """
        参数:
            policy_model: 策略模型 (nn.Module)
            ref_model:    参考模型 (frozen)
            tokenizer:    分词器
            reward_fn:    奖励函数 reward_fn(question, response) -> float
            group_size:   每个问题的采样数量 G
            lr:           学习率
            epsilon:      clip 范围
            beta:         KL 系数
            max_gen_len:  最大生成长度
            num_epochs_per_update: 每次采样后的梯度更新次数
        """
        self.policy = policy_model
        self.ref = ref_model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.G = group_size
        self.max_gen_len = max_gen_len
        self.num_epochs = num_epochs_per_update
        
        # 冻结参考模型
        for p in self.ref.parameters():
            p.requires_grad = False
        
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=lr, weight_decay=0.01
        )
        self.grpo_loss = GRPOLoss(epsilon=epsilon, beta=beta)
        
        # 训练统计
        self.stats = []
    
    @torch.no_grad()
    def generate_responses(self, questions, device='cpu'):
        """
        对每个问题生成 G 个回答
        
        参数:
            questions: list[str] — B 个问题
        返回:
            responses: list[list[str]] — B × G 个回答文本
            all_token_ids: (B, G, T) — token ids (padded)
            mask: (B, G, T) — 有效 token mask
        """
        self.policy.eval()
        B = len(questions)
        all_responses = []
        
        for q in questions:
            q_responses = []
            for _ in range(self.G):
                # 编码问题
                input_ids = self.tokenizer.encode(q, return_tensors='pt').to(device)
                
                # 生成回答 (采样)
                with torch.no_grad():
                    output = self.policy.generate(
                        input_ids,
                        max_new_tokens=self.max_gen_len,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                    )
                
                # 解码
                response = self.tokenizer.decode(
                    output[0][input_ids.shape[1]:], 
                    skip_special_tokens=True
                )
                q_responses.append(response)
            all_responses.append(q_responses)
        
        return all_responses
    
    def compute_log_probs(self, model, input_ids, attention_mask):
        """
        计算模型在给定 token 序列上的 log 概率
        
        参数:
            model: 语言模型
            input_ids:      (B*G, T)
            attention_mask:  (B*G, T)
        返回:
            log_probs: (B*G, T-1) — 每个 token 的 log 概率
        """
        with torch.set_grad_enabled(model.training):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # (B*G, T, V)
        
        # 计算 log P(token_t | token_{<t})
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)  # (B*G, T-1, V)
        
        # 选择实际 token 的 log 概率
        target_ids = input_ids[:, 1:]  # (B*G, T-1)
        token_log_probs = log_probs.gather(
            dim=-1, index=target_ids.unsqueeze(-1)
        ).squeeze(-1)  # (B*G, T-1)
        
        return token_log_probs
    
    def train_step(self, questions, ground_truths, device='cpu'):
        """
        GRPO 单步训练
        
        参数:
            questions:     list[str] — B 个问题
            ground_truths: list[str] — B 个正确答案
        返回:
            stats: dict — 训练统计
        """
        B = len(questions)
        
        # ===== 阶段 1: 采样 =====
        self.policy.eval()
        responses = self.generate_responses(questions, device)
        
        # ===== 阶段 2: 计算奖励 =====
        rewards = torch.zeros(B, self.G)
        for i in range(B):
            for j in range(self.G):
                rewards[i, j] = self.reward_fn(
                    questions[i], responses[i][j], ground_truths[i]
                )
        rewards = rewards.to(device)
        
        # ===== 阶段 3: 编码所有回答 =====
        # (此处简化 — 实际需要 tokenize + pad 所有 (question, response) 对)
        # 假设已经得到:
        #   all_input_ids:    (B*G, T)
        #   all_attention_mask: (B*G, T)
        #   response_mask:    (B, G, T_resp) — 只计算回答部分的 loss
        
        # ===== 阶段 4: 计算旧策略和参考模型的 log 概率 =====
        # (采样时的策略就是旧策略)
        # log_probs_old = self.compute_log_probs(self.policy, ...)  # 采样时记录
        # log_probs_ref = self.compute_log_probs(self.ref, ...)
        
        # ===== 阶段 5: 梯度更新 =====
        self.policy.train()
        step_stats = {}
        
        for epoch in range(self.num_epochs):
            # 计算当前策略的 log 概率
            # log_probs_new = self.compute_log_probs(self.policy, ...)
            
            # GRPO 损失
            # result = self.grpo_loss(log_probs_new, log_probs_old,
            #                         log_probs_ref, rewards, response_mask)
            
            # 梯度更新
            # self.optimizer.zero_grad()
            # result['loss'].backward()
            # torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            # self.optimizer.step()
            
            # step_stats = {k: v.item() for k, v in result.items()}
            pass
        
        # 记录统计
        step_stats['mean_reward'] = rewards.mean().item()
        step_stats['reward_std'] = rewards.std().item()
        self.stats.append(step_stats)
        
        return step_stats
```

#### 6.2.3 奖励函数示例

```python
import re

class MathRewardFunction:
    """
    数学题奖励函数 — 基于规则的答案验证
    
    数学:
        r(o, y*) = 1 if extract_answer(o) == y* else 0
    """
    
    @staticmethod
    def extract_answer(response: str) -> str:
        """
        从回答中提取最终答案
        支持格式:
            - "答案是 42"
            - "\\boxed{42}"
            - "最终答案: 42"
            - "The answer is 42"
        """
        patterns = [
            r'\\boxed\{([^}]+)\}',
            r'(?:答案|answer|Answer)[是is:：]\s*(.+?)(?:\s|$|。)',
            r'(?:最终|final)(?:答案|answer)[是is:：]\s*(.+?)(?:\s|$|。)',
            r'=\s*(\d+(?:\.\d+)?)\s*$',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip()
        
        # 尝试提取最后一个数字
        numbers = re.findall(r'-?\d+(?:\.\d+)?', response)
        if numbers:
            return numbers[-1]
        
        return ""
    
    @staticmethod
    def normalize_answer(answer: str) -> str:
        """归一化答案格式 (去空格, 统一分数等)"""
        answer = answer.strip().lower()
        # 去掉多余符号
        answer = re.sub(r'[,\s]', '', answer)
        # 尝试转为浮点数比较
        try:
            return str(float(answer))
        except ValueError:
            return answer
    
    def __call__(self, question: str, response: str, 
                 ground_truth: str) -> float:
        """
        计算奖励
        
        参数:
            question:     问题文本
            response:     模型回答
            ground_truth: 正确答案
        返回:
            reward: 0.0 或 1.0
        """
        extracted = self.extract_answer(response)
        norm_extracted = self.normalize_answer(extracted)
        norm_truth = self.normalize_answer(ground_truth)
        
        return 1.0 if norm_extracted == norm_truth else 0.0

# ===== 验证 =====
reward_fn = MathRewardFunction()

test_cases = [
    ("2+3=?", "让我计算: 2+3 = 5, 所以答案是 5", "5"),
    ("x²=9, x=?", "解方程: x = \\boxed{3}", "3"),
    ("100/4=?", "100除以4等于24", "25"),
    ("sin(π/2)=?", "The answer is 1.0", "1"),
]

print("奖励函数验证:")
for q, r, gt in test_cases:
    reward = reward_fn(q, r, gt)
    extracted = reward_fn.extract_answer(r)
    print(f"  问题: {q}")
    print(f"  回答: {r}")
    print(f"  提取: '{extracted}', 正确: '{gt}', 奖励: {reward}")
    print()
```

#### 6.2.4 简化版 GRPO 训练循环（可运行）

```python
class SimpleGRPODemo:
    """
    简化版 GRPO 演示 — 用简单策略网络展示核心训练逻辑
    
    设置: 策略网络学习为不同"问题"(整数)选择正确"答案"(整数)
    """
    
    def __init__(self, num_questions=10, num_answers=10, 
                 group_size=8, hidden_dim=32):
        self.num_q = num_questions
        self.num_a = num_answers
        self.G = group_size
        
        # 简单策略网络: 问题 embedding → 答案分布
        self.policy = nn.Sequential(
            nn.Embedding(num_questions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_answers),
        )
        
        # 参考策略 (初始策略的副本)
        self.ref = nn.Sequential(
            nn.Embedding(num_questions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_answers),
        )
        self.ref.load_state_dict(self.policy.state_dict())
        for p in self.ref.parameters():
            p.requires_grad = False
        
        # 正确答案映射
        torch.manual_seed(0)
        self.correct_answers = torch.randint(0, num_answers, (num_questions,))
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-2)
        self.grpo_loss_fn = GRPOLoss(epsilon=0.2, beta=0.04)
    
    def reward(self, question_ids, answer_ids):
        """奖励: 答案正确 → 1, 否则 → 0"""
        correct = self.correct_answers[question_ids]
        return (answer_ids == correct).float()
    
    def train_step(self):
        """单步 GRPO 训练"""
        B = self.num_q  # 使用所有问题
        q_ids = torch.arange(B)
        
        # 1. 旧策略采样 G 个回答
        with torch.no_grad():
            old_logits = self.policy[0](q_ids)
            old_logits = self.policy[2](F.relu(old_logits))  # (B, num_a)
            old_probs = F.softmax(old_logits, dim=-1)        # (B, num_a)
            
            # 对每个问题采样 G 个回答
            samples = torch.multinomial(
                old_probs.unsqueeze(1).expand(B, self.G, -1).reshape(B*self.G, -1),
                num_samples=1
            ).view(B, self.G)  # (B, G)
            
            # 记录旧策略 log 概率
            old_log_probs = F.log_softmax(old_logits, dim=-1)  # (B, num_a)
            old_log_probs_selected = old_log_probs.gather(
                1, samples.view(B, self.G)
            )  # (B, G)
            
            # 参考策略 log 概率
            ref_logits = self.ref[0](q_ids)
            ref_logits = self.ref[2](F.relu(ref_logits))
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            ref_log_probs_selected = ref_log_probs.gather(1, samples)  # (B, G)
        
        # 2. 计算奖励
        rewards = self.reward(
            q_ids.unsqueeze(1).expand(B, self.G),
            samples
        )  # (B, G)
        
        # 3. 当前策略 log 概率
        new_logits = self.policy[0](q_ids)
        new_logits = self.policy[2](F.relu(new_logits))
        new_log_probs = F.log_softmax(new_logits, dim=-1)
        new_log_probs_selected = new_log_probs.gather(1, samples)  # (B, G)
        
        # 4. GRPO 损失 (这里每个"回答"只有 1 个 token, 所以 T=1)
        result = self.grpo_loss_fn(
            new_log_probs_selected.unsqueeze(-1),   # (B, G, 1)
            old_log_probs_selected.unsqueeze(-1),
            ref_log_probs_selected.unsqueeze(-1),
            rewards,
            torch.ones(B, self.G, 1),  # mask: 全1
        )
        
        # 5. 梯度更新
        self.optimizer.zero_grad()
        result['loss'].backward()
        self.optimizer.step()
        
        return {k: v.item() if isinstance(v, torch.Tensor) else v 
                for k, v in result.items()}
    
    def evaluate(self):
        """评估: 每个问题选最大概率的答案"""
        with torch.no_grad():
            q_ids = torch.arange(self.num_q)
            logits = self.policy[0](q_ids)
            logits = self.policy[2](F.relu(logits))
            predictions = logits.argmax(dim=-1)
            accuracy = (predictions == self.correct_answers).float().mean()
        return accuracy.item()

# ===== 运行 GRPO 训练演示 =====
torch.manual_seed(42)
demo = SimpleGRPODemo(num_questions=20, num_answers=10, group_size=16)

print("GRPO 训练演示 (简化版):")
print(f"任务: {demo.num_q} 个问题, 每个有 {demo.num_a} 个候选答案")
print(f"组大小 G = {demo.G}")
print(f"初始准确率: {demo.evaluate():.1%}")
print()

print(f"{'步骤':>4} {'损失':>8} {'奖励':>8} {'KL':>8} {'Clip%':>8} {'准确率':>8}")
print("-" * 52)

for step in range(200):
    stats = demo.train_step()
    if (step + 1) % 20 == 0:
        acc = demo.evaluate()
        print(f"{step+1:>4} {stats['loss']:.4f} "
              f"{stats['mean_reward']:.4f} "
              f"{stats['mean_kl']:.4f} "
              f"{stats['clip_fraction']:.4f} "
              f"{acc:.1%}")

print(f"\n最终准确率: {demo.evaluate():.1%}")
```

#### 6.2.5 GRPO 与 PPO 对比实验

```python
class SimplePPODemo:
    """
    简化版 PPO (带价值模型) 作为 GRPO 的对比
    """
    
    def __init__(self, num_questions=10, num_answers=10, hidden_dim=32):
        self.num_q = num_questions
        self.num_a = num_answers
        
        # 策略网络
        self.policy = nn.Sequential(
            nn.Embedding(num_questions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_answers),
        )
        
        # 价值网络 (PPO 额外需要!)
        self.value_net = nn.Sequential(
            nn.Embedding(num_questions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # 参考策略
        self.ref = nn.Sequential(
            nn.Embedding(num_questions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_answers),
        )
        self.ref.load_state_dict(self.policy.state_dict())
        for p in self.ref.parameters():
            p.requires_grad = False
        
        self.correct_answers = torch.randint(0, num_answers, (num_questions,))
        
        # PPO 需要两个优化器
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=1e-2
        )
        self.value_optimizer = torch.optim.Adam(
            self.value_net.parameters(), lr=1e-2
        )
    
    def count_parameters(self):
        """统计参数量 (包含价值模型)"""
        policy_params = sum(p.numel() for p in self.policy.parameters())
        value_params = sum(p.numel() for p in self.value_net.parameters())
        return {'policy': policy_params, 'value': value_params, 
                'total': policy_params + value_params}

def compare_grpo_ppo():
    """对比 GRPO 和 PPO 的参数效率"""
    num_q, num_a, hidden = 20, 10, 32
    
    grpo = SimpleGRPODemo(num_q, num_a, group_size=16, hidden_dim=hidden)
    ppo = SimplePPODemo(num_q, num_a, hidden_dim=hidden)
    
    grpo_params = sum(p.numel() for p in grpo.policy.parameters())
    ppo_params = ppo.count_parameters()
    
    print("GRPO vs PPO 参数量对比:")
    print(f"  GRPO 策略参数: {grpo_params:,}")
    print(f"  PPO  策略参数: {ppo_params['policy']:,}")
    print(f"  PPO  价值参数: {ppo_params['value']:,}")
    print(f"  PPO  总参数:   {ppo_params['total']:,}")
    print(f"  GRPO 节省:     {ppo_params['value']:,} 参数 "
          f"({ppo_params['value']/ppo_params['total']:.0%})")
    
    print(f"\n对于 LLM 规模 (如 67B):")
    print(f"  GRPO: ~67B (策略) + ~67B (参考) = ~134B")
    print(f"  PPO:  ~67B (策略) + ~67B (价值) + ~67B (奖励) + ~67B (参考) = ~268B")
    print(f"  GRPO 节省: ~50% 训练资源")

compare_grpo_ppo()
```

---

## 7. 实践技巧与可视化

### 7.1 GRPO 训练动态可视化

```python
import numpy as np

def visualize_grpo_training_dynamics():
    """可视化 GRPO 训练过程中的关键指标变化"""
    
    # 模拟训练数据
    np.random.seed(42)
    steps = 100
    
    # 模拟指标
    rewards = np.clip(0.1 + 0.7 * np.log1p(np.arange(steps)) / np.log(steps+1) 
                      + np.random.randn(steps) * 0.05, 0, 1)
    kl = 0.001 + 0.05 * np.log1p(np.arange(steps)) / np.log(steps+1) \
         + np.random.randn(steps) * 0.005
    kl = np.abs(kl)
    clip_frac = np.clip(0.05 + 0.15 * np.arange(steps) / steps 
                        + np.random.randn(steps) * 0.02, 0, 0.5)
    response_len = 200 + 300 * np.log1p(np.arange(steps)) / np.log(steps+1) \
                   + np.random.randn(steps) * 20
    
    print("GRPO 训练动态 (模拟):")
    print(f"{'步骤':>6} {'平均奖励':>10} {'KL散度':>10} {'Clip比例':>10} {'回答长度':>10}")
    print("-" * 50)
    for i in [0, 10, 25, 50, 75, 99]:
        print(f"{i+1:>6} {rewards[i]:>10.4f} {kl[i]:>10.4f} "
              f"{clip_frac[i]:>10.4f} {response_len[i]:>10.0f}")
    
    print(f"\n关键观察:")
    print(f"  1. 奖励从 {rewards[0]:.2f} 上升到 {rewards[-1]:.2f} (推理能力提升)")
    print(f"  2. KL 散度从 {kl[0]:.4f} 增长到 {kl[-1]:.4f} (策略偏离参考)")
    print(f"  3. 回答长度从 {response_len[0]:.0f} 增长到 {response_len[-1]:.0f} "
          f"(更详细的推理)")

visualize_grpo_training_dynamics()
```

### 7.2 组大小与方差分析

```python
def analyze_group_size_variance():
    """
    分析组大小 G 对优势估计方差的影响
    
    数学:
        Var[Â_i] = Var[(r_i - μ_r) / σ_r]
        随着 G 增大, μ_r 和 σ_r 的估计更准确, 优势方差降低
    """
    np.random.seed(42)
    
    # 真实奖励分布参数 (二项分布, 正确率 p)
    p_correct = 0.4  # 40% 正确率
    n_trials = 10000
    
    group_sizes = [2, 4, 8, 16, 32, 64]
    
    print("组大小 G 对优势估计的影响:")
    print(f"设定: 正确率 p={p_correct}, 奖励 ∈ {{0, 1}}")
    print(f"{'G':>4} {'Â 均值':>10} {'Â 标准差':>10} {'Â 偏度':>10} {'有效梯度比':>12}")
    print("-" * 50)
    
    for G in group_sizes:
        all_advantages = []
        effective_grad_ratios = []
        
        for _ in range(n_trials):
            # 采样 G 个奖励
            rewards = np.random.choice([0.0, 1.0], size=G, p=[1-p_correct, p_correct])
            
            # 计算优势
            mu = rewards.mean()
            sigma = rewards.std()
            
            if sigma > 1e-8:
                advantages = (rewards - mu) / sigma
                all_advantages.extend(advantages.tolist())
                # 有效梯度: 优势不为 0 的比例
                effective_grad_ratios.append(np.mean(np.abs(advantages) > 0.1))
            else:
                # 所有奖励相同 → 优势为 0 → 无梯度
                all_advantages.extend([0.0] * G)
                effective_grad_ratios.append(0.0)
        
        advantages_arr = np.array(all_advantages)
        print(f"{G:>4} {advantages_arr.mean():>10.4f} {advantages_arr.std():>10.4f} "
              f"{float(np.mean(advantages_arr**3) / max(advantages_arr.std()**3, 1e-8)):>10.4f} "
              f"{np.mean(effective_grad_ratios):>12.2%}")
    
    print(f"\n结论:")
    print(f"  - G 太小 (2-4): 优势估计方差大, 训练不稳定")
    print(f"  - G 太大 (64+): 采样开销大, 收益递减")
    print(f"  - 推荐 G = 8~16: 方差与效率的最佳平衡")

analyze_group_size_variance()
```

### 7.3 推理 Token 长度与性能关系

DeepSeek-R1 的一个重要发现：RL 训练过程中，模型的推理长度（`<think>` 内的 token 数）会自然增长。

```python
def analyze_reasoning_length():
    """
    分析推理长度与正确率的关系
    
    DeepSeek-R1 观察: 
        - RL 训练使推理长度自然增长
        - 更长的推理 ≈ 更多的"思考时间" ≈ 更高的正确率
        - 但存在最优长度, 过长会引入噪声
    """
    np.random.seed(42)
    
    # 模拟不同推理长度的正确率
    lengths = np.array([50, 100, 200, 400, 800, 1200, 1600, 2000])
    
    # 正确率: 对数增长后趋于饱和
    accuracy = 0.15 + 0.65 * (1 - np.exp(-lengths / 400))
    # 加入少量噪声和过长时的轻微下降
    accuracy = accuracy - 0.02 * np.maximum(0, (lengths - 1200) / 1000)
    accuracy = np.clip(accuracy + np.random.randn(len(lengths)) * 0.02, 0, 1)
    
    # 推理时间成本 (与长度成正比)
    time_cost = lengths / 100  # 相对时间
    
    print("推理长度 vs 正确率 (AIME 数学题模拟):")
    print(f"{'推理 tokens':>12} {'正确率':>10} {'时间成本':>10} {'效率':>10}")
    print("-" * 46)
    for l, a, t in zip(lengths, accuracy, time_cost):
        efficiency = a / t
        print(f"{l:>12} {a:>10.1%} {t:>10.1f}x {efficiency:>10.4f}")
    
    best_idx = np.argmax(accuracy)
    best_eff_idx = np.argmax(accuracy / time_cost)
    
    print(f"\n最高正确率: {lengths[best_idx]} tokens → {accuracy[best_idx]:.1%}")
    print(f"最高效率:   {lengths[best_eff_idx]} tokens → "
          f"{accuracy[best_eff_idx]:.1%} (效率 {accuracy[best_eff_idx]/time_cost[best_eff_idx]:.4f})")
    print(f"\n关键发现:")
    print(f"  - 推理长度 200→800: 正确率快速提升")
    print(f"  - 推理长度 >1200: 收益递减, 可能引入噪声")
    print(f"  - DeepSeek-R1 的 RL 训练自动找到了最优推理长度")

analyze_reasoning_length()
```

**DeepSeek-R1 的实际基准性能**：

| 基准 | GPT-4o | Claude 3.5 Sonnet | OpenAI o1 | **DeepSeek-R1** |
|------|:------:|:-----------------:|:---------:|:---------------:|
| AIME 2024 (数学竞赛) | 9.3% | 16.0% | 79.2% | **79.8%** |
| MATH-500 | 74.6% | 78.3% | 96.4% | **97.3%** |
| Codeforces Rating | 808 | 1033 | 1891 | **2029** |
| GPQA Diamond (科学) | 49.9% | 65.0% | 75.7% | **71.5%** |
| MMLU | 87.2% | 88.7% | 91.8% | **90.8%** |

> **关键发现**：DeepSeek-R1 在数学和编程上达到了与 OpenAI o1 comparable 的水平，某些基准甚至超越。

---

## 8. 与其他模型的关系

### 8.1 从 RLHF 到推理 RL 的演进

```
RLHF (2022)                    推理 RL (2024-2025)
│                               │
│ InstructGPT/ChatGPT           │ OpenAI o1 / DeepSeek-R1
│ │                             │ │
│ ├─ 奖励: 人类偏好 RM          │ ├─ 奖励: 规则验证 (精确)
│ ├─ 算法: PPO (需要 Critic)    │ ├─ 算法: GRPO (无需 Critic)
│ ├─ 输出: 直接回答             │ ├─ 输出: 思维链 + 回答
│ ├─ 目标: 对齐人类偏好         │ ├─ 目标: 提升推理能力
│ └─ 数据: 人类标注 (昂贵)      │ └─ 数据: 自动生成+验证 (可扩展)
│                               │
└── 让模型更安全、更有用         └── 让模型更聪明
```

**演进的数学本质**：

RLHF 优化的是**人类偏好的代理目标**：

$$
\max_\theta \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta} [r_\phi(x, y)] - \beta D_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}]
$$

其中 $r_\phi$ 是**学习得到的**奖励模型——是人类偏好的**近似**。

DeepSeek-R1 优化的是**精确的推理目标**：

$$
\max_\theta \mathbb{E}_{q \sim \mathcal{D}, o \sim \pi_\theta} [\mathbb{1}[\texttt{verify}(o) = \text{correct}]] - \beta D_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}]
$$

奖励信号是**精确的**——答案正确就是正确。

### 8.2 DeepSeek-R1 vs OpenAI o1

| 维度 | OpenAI o1 | DeepSeek-R1 |
|------|-----------|-------------|
| **发布** | 2024.09 | 2025.01 |
| **开源** | ❌ 闭源 | ✅ 开源 (MIT) |
| **基础模型** | 未知 (GPT-4 级) | DeepSeek-V3 (671B MoE) |
| **训练方法** | 未公开 (推测 RL) | GRPO + 多阶段训练 |
| **推理格式** | 隐藏思维链 | 显式 `<think>` 标签 |
| **数学 (AIME)** | 79.2% | **79.8%** |
| **编程 (CF)** | 1891 | **2029** |
| **蒸馏** | ❌ | ✅ (1.5B~70B 多规模) |

> **DeepSeek-R1 的历史意义**：证明了开源社区可以复现闭源推理模型的能力，并提供了完整的技术路线图。

### 8.3 推理模型谱系

```
推理模型发展谱系
│
├── 思维链 (Chain-of-Thought) 方向
│   ├── CoT Prompting (Wei et al., 2022) ← 手动提示
│   ├── Self-Consistency (Wang et al., 2023) ← 多路径投票
│   ├── Tree-of-Thought (Yao et al., 2023) ← 树搜索
│   └── STaR (Zelikman et al., 2022) ← 自我改进推理
│
├── RL 推理方向
│   ├── OpenAI o1 (2024.09) ← 闭源, 推测用 RL
│   ├── DeepSeek-R1-Zero (2025.01) ← 纯 RL, 无 SFT
│   ├── DeepSeek-R1 (2025.01) ← GRPO + 多阶段 ← 开源
│   ├── Qwen-QwQ (2025) ← 阿里推理模型
│   └── Kimi k1.5 (2025) ← Moonshot 推理模型
│
├── 过程奖励方向
│   ├── PRM (Process Reward Model) ← 过程级奖励
│   ├── Math-Shepherd (Wang et al., 2024) ← 数学过程奖励
│   └── OmegaPRM (Luo et al., 2024) ← 自动过程标注
│
└── 蒸馏方向
    ├── DeepSeek-R1-Distill-Qwen-1.5B ← 超小规模推理
    ├── DeepSeek-R1-Distill-Qwen-7B   ← 消费级推理
    ├── DeepSeek-R1-Distill-Qwen-32B  ← 高性能推理
    └── DeepSeek-R1-Distill-Llama-70B ← 最大蒸馏模型
```

**本系列学习路径中的位置**：

| 编号 | 模型 | 年份 | 关键贡献 | 与 R1 的关系 |
|------|------|------|----------|-------------|
| 06 | Transformer | 2017 | 自注意力 | 基础架构 |
| 07 | BERT | 2018 | 双向预训练 | 编码器分支 |
| 08 | GPT-2 | 2019 | 因果 LM | 生成范式 |
| 10 | GPT-3 | 2020 | In-context Learning | 规模验证 |
| 11 | Switch Transformer | 2021 | MoE | DeepSeek-V3 用 MoE |
| 12 | LoRA | 2021 | 参数高效微调 | 微调方法 |
| 13 | RLHF | 2022 | 人类对齐 | PPO 前身 |
| 14 | LLaMA | 2023 | 开源架构 | 架构基础 |
| **15** | **DeepSeek-R1** | **2025** | **推理 RL + GRPO** | **本篇** |

$$
\boxed{\text{系列总结: 从 Transformer 到 DeepSeek-R1} = \text{架构} + \text{规模} + \text{对齐} + \text{推理}}
$$

---

## 扩展阅读与实现

### Q1: GRPO 的组内优势为什么是无偏的？

> **Q:** 用组内均值作基线，得到的优势估计是无偏的吗？
>
> **A:** 严格来说，GRPO 的优势估计存在**有限样本偏差**，但随着组大小 $G$ 增大，偏差趋于零。
>
> 设真实优势为 $A_i = r_i - V^*(q)$（$V^*(q)$ 是问题 $q$ 的真实价值函数）。GRPO 用 $\mu_r = \frac{1}{G}\sum_j r_j$ 代替 $V^*(q)$：
>
> $$\hat{A}_i^{\text{GRPO}} \propto r_i - \mu_r = r_i - \frac{1}{G}\sum_{j=1}^G r_j$$
>
> 而 $\mathbb{E}[\mu_r] = V^*(q)$（因为 $r_j$ 是从 $\pi_{\theta_{\text{old}}}(\cdot|q)$ 采样的奖励）。
>
> 因此：
> $$\mathbb{E}[\hat{A}_i^{\text{GRPO}}] \propto \mathbb{E}[r_i] - \mathbb{E}[\mu_r] = V^*(q) - V^*(q) = 0 \quad \text{(对随机选取的 } i \text{)}$$
>
> 但对于**特定的** $i$，$\mu_r$ 中包含了 $r_i$ 本身，导致：
> $$\mathbb{E}[\hat{A}_i | r_i] \neq r_i - V^*(q) \quad \text{(有限样本偏差)}$$
>
> 当 $G \to \infty$ 时，$r_i$ 在 $\mu_r$ 中的贡献趋于零，偏差消失。实践中 $G=8\sim16$ 已经足够。

### Q2: DeepSeek-R1-Zero 有哪些有趣的涌现行为？

> **Q:** 不经过任何 SFT，直接对基础模型做 RL，会出现什么？
>
> **A:** DeepSeek-R1-Zero 展示了多个惊人的**涌现行为**：
>
> 1. **自发产生思维链**：模型在 RL 过程中自己学会了"先思考再回答"的模式
> 2. **"Aha Moment"**：模型出现了类似人类顿悟的行为——先给错误答案，然后说"等等，让我重新想想"并纠正
> 3. **推理长度自增长**：随着 RL 训练进行，模型的推理过程越来越长（从几十 tokens 到上千 tokens）
> 4. **多语言混用**：模型有时会在推理过程中混用英文和中文（"language mixing"问题）
>
> **但也有问题**：
> - 输出格式不稳定（有时推理过程没有清晰结构）
> - 可读性差（没有 `<think>` 标签，推理过程与答案混在一起）
> - 这就是为什么需要冷启动 SFT 的原因

### Q3: KL 系数 $\beta$ 如何选择？

> **Q:** $\beta$ 太大或太小会怎样？
>
> **A:** $\beta$ 控制策略偏离参考模型的程度：
>
> - **$\beta$ 太小**（如 $\beta = 0.001$）：策略可以自由偏离，可能导致**模式坍塌**（只生成一种类型的回答）或**奖励 hacking**
> - **$\beta$ 太大**（如 $\beta = 1.0$）：策略几乎无法更新，训练极慢
> - **推荐范围**：$\beta \in [0.01, 0.1]$，具体取决于任务
>
> DeepSeek-R1 的实践：
> $$\beta \approx 0.01 \sim 0.04 \quad \text{(推理 RL 阶段)}$$
>
> 也可以使用**自适应 KL**：监控 $D_{\text{KL}}$，如果超过目标值则增大 $\beta$，低于目标则减小。

### Q4: 为什么蒸馏比直接 RL 训练更有效？

> **Q:** 小模型直接做 RL 不如从大模型蒸馏，这是为什么？
>
> **A:** 有几个可能的原因：
>
> 1. **探索能力**：小模型的策略空间有限，RL 训练中很难通过随机探索找到正确的推理路径；而大模型已经找到了这些路径，蒸馏只需"模仿"
>
> 2. **信号密度**：RL 的奖励是稀疏的（只有最终答案对/错），而蒸馏提供了逐 token 的密集监督信号
>
> $$\underbrace{\text{RL 信号}}_{\text{1 bit/回答}} \quad \text{vs} \quad \underbrace{\text{蒸馏信号}}_{\text{完整分布/token}}$$
>
> 3. **数据效率**：蒸馏数据中包含了大模型的"推理模式"，小模型可以直接学习这些模式，而非从头发现
>
> 4. **容量瓶颈**：即使给足 RL 训练时间，小模型的容量也可能不足以独立发现复杂推理策略

### Q5: GRPO 可以用在推理任务之外吗？

> **Q:** GRPO 只适用于有明确正确答案的任务吗？
>
> **A:** GRPO 的框架是通用的，但其优势在可验证奖励的场景下最为明显。
>
> **适用场景**：
> - ✅ 数学题（答案可验证）
> - ✅ 编程题（测试用例可验证）
> - ✅ 逻辑推理（结论可形式化验证）
> - ⚠️ 开放问答（需要学习的奖励模型，退化为类似 RLHF）
> - ⚠️ 创意写作（奖励主观，GRPO 优势减弱）
>
> **扩展**：可以将 GRPO 与学习的奖励模型结合，用于通用对齐任务。DeepSeek-R1 在阶段 4 就是这样做的——用 GRPO + 混合奖励（规则+RM）进行最终对齐。

---

## 参考资源

### 经典论文

1. DeepSeek-AI (2025). [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948). arXiv.
   - **贡献**：提出 GRPO 训练推理模型的完整技术路线，开源 671B 推理模型及多规模蒸馏版本

2. Shao et al. (2024). [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300). arXiv.
   - **贡献**：首次提出 GRPO 算法，在数学推理任务上验证了无价值模型 RL 的可行性

3. DeepSeek-AI (2024). [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437). arXiv.
   - **贡献**：DeepSeek-R1 的基础模型，671B MoE 架构，MLA 注意力机制

4. Schulman et al. (2017). [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347). arXiv.
   - **贡献**：提出 PPO 算法，GRPO 的 clip 机制和策略更新框架的基础

5. Ouyang et al. (2022). [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155). NeurIPS 2022.
   - **贡献**：提出 InstructGPT/RLHF 训练范式，DeepSeek-R1 对齐训练的前身

6. Wei et al. (2022). [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903). NeurIPS 2022.
   - **贡献**：提出思维链提示，证明 LLM 可以通过中间推理步骤提升表现

7. Zelikman et al. (2022). [STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465). NeurIPS 2022.
   - **贡献**：用自我改进的方式训练推理能力，DeepSeek-R1 拒绝采样的思想来源之一

### 教材与书籍

8. Sutton & Barto. [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html). MIT Press.
   - **章节**：第 13 章策略梯度方法，理解 GRPO 的 RL 基础

### 在线资源与教程

9. DeepSeek-AI. [DeepSeek-R1 官方代码](https://github.com/deepseek-ai/DeepSeek-R1).
   - **内容**：模型权重、推理代码、蒸馏版本

10. Hugging Face. [Open-R1 项目](https://github.com/huggingface/open-r1).
    - **内容**：开源社区复现 DeepSeek-R1 训练流程的尝试

11. Hugging Face TRL. [GRPO Trainer](https://huggingface.co/docs/trl/grpo_trainer).
    - **内容**：Hugging Face TRL 库中的 GRPO 训练器实现

12. Lilian Weng. [Reward Hacking in Reinforcement Learning](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/).
    - **内容**：奖励 hacking 问题的综述，理解为什么规则奖励比学习的 RM 更适合推理任务

---

## 附录：符号表

| 符号 | 含义 | 维度/类型 |
|------|------|----------|
| $\pi_\theta$ | 当前策略（待优化的语言模型） | 函数 |
| $\pi_{\theta_{\text{old}}}$ | 旧策略（用于采样） | 函数 |
| $\pi_{\text{ref}}$ | 参考策略（SFT 模型） | 函数 |
| $q$ | 输入问题 | 文本 |
| $o_i$ | 第 $i$ 个回答 | token 序列 |
| $o_{i,t}$ | 第 $i$ 个回答的第 $t$ 个 token | 标量（token id） |
| $o_{i,<t}$ | 第 $i$ 个回答中 $t$ 之前的所有 token | token 序列 |
| $G$ | 组大小（每个问题的采样数量） | 标量，通常 $8 \sim 64$ |
| $r_i$ | 第 $i$ 个回答的奖励 | 标量 |
| $\hat{A}_i$ | 第 $i$ 个回答的组内优势估计 | 标量 |
| $\mu_r$ | 组内奖励均值 | 标量 |
| $\sigma_r$ | 组内奖励标准差 | 标量 |
| $\rho_{i,t}$ | 重要性采样比率 $\pi_\theta / \pi_{\theta_{\text{old}}}$ | 标量 |
| $\epsilon$ | Clip 范围参数 | 标量，通常 $0.1 \sim 0.2$ |
| $\beta$ | KL 散度惩罚系数 | 标量，通常 $0.01 \sim 0.1$ |
| $D_{\text{KL}}^{(t)}$ | Token 级 KL 散度近似 | 标量 |
| $\mathcal{J}_{\text{GRPO}}$ | GRPO 目标函数值 | 标量 |
| $\mathcal{L}$ | 损失函数值（$= -\mathcal{J}$） | 标量 |
| $\alpha$ | 学习率 | 标量 |
| $\mathcal{D}$ | 训练数据集 | 集合 |
| $\mathcal{B}$ | Mini-batch | 集合 |
| $B$ | Batch 中问题的数量 | 标量 |
| $T$ | 回答的最大 token 长度 | 标量 |
| $\|o_i\|$ | 第 $i$ 个回答的实际长度 | 标量 |
| $V(s)$ | 价值函数（PPO 需要，GRPO 不需要） | 函数 |
| $\hat{A}_t^{\text{GAE}}$ | GAE 优势估计（PPO 使用） | 标量 |
| $\gamma$ | RL 折扣因子 | 标量 |
| $\lambda$ | GAE 参数 | 标量 |
| $R(\tau)$ | 轨迹总回报 | 标量 |
| $\mathbb{1}[\cdot]$ | 指示函数 | $\{0, 1\}$ |
| $\otimes$ | 逐元素乘法（Hadamard 积） | 运算符 |

**典型超参数（DeepSeek-R1）：**
- 基础模型：DeepSeek-V3-Base（671B MoE，激活 37B）
- 组大小：$G = 8 \sim 64$
- Clip 范围：$\epsilon = 0.2$
- KL 系数：$\beta \approx 0.01 \sim 0.04$
- 学习率：$\alpha \approx 10^{-6}$
- 冷启动数据：数千条高质量 CoT
- 拒绝采样：$K = 64$（每题采样次数）
- 蒸馏规模：1.5B / 7B / 8B / 14B / 32B / 70B

---

最后更新：2026-03-19
