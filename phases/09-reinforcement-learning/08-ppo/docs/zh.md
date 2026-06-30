# 近端策略优化（PPO）

> A2C 在每次更新后丢弃每个展开。PPO 将策略梯度包裹在裁剪的重要性比率中，所以你可以在相同数据上做 10+ epochs 而不让策略爆炸。Schulman 等人（2017）。2026 年仍然是默认策略梯度算法。

**类型：** 构建
**语言：** Python
**前置条件：** 第九阶段 · 06（REINFORCE），第九阶段 · 07（Actor-Critic）
**时间：** 约 75 分钟

## 问题

A2C（第 07 课）是在策略的：梯度 `E_{π_θ}[A · ∇ log π_θ]` 要求从*当前* `π_θ` 采样的数据。进行一次更新，`π_θ` 就改变了；你使用的数据现在是离策略的。重用它，你的梯度就有偏。

展开是昂贵的。在 Atari 上，跨 8 个 env × 128 步的一次展开 = 1024 次转移和十几秒的环境时间。在一个梯度步后丢弃那是浪费的。

信任域策略优化（TRPO，Schulman 2015）是第一个修复：约束每次更新使得旧策略和新策略之间的 KL 散度保持在 `δ` 以下。理论上干净，但需要每次更新的共轭梯度求解。2026 年没人运行 TRPO。

PPO（Schulman 等人 2017）用简单的裁剪目标替换硬信任域约束。额外一行代码。每个展开十个 epoch。无共轭梯度。足够好的理论保证。九年后它仍然是从 MuJoCo 到 RLHF 的一切默认策略梯度算法。

## 概念

![PPO 裁剪代理目标：比率裁剪在 1 ± ε](../assets/ppo.svg)

**重要性比率。**

`r_t(θ) = π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t)`

这是新策略与收集数据的策略的似然比。`r_t = 1` 意味着无变化。`r_t = 2` 意味着新政策采取 `a_t` 的可能性是旧的两倍。

**裁剪代理。**

`L^{CLIP}(θ) = E_t [ min( r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t ) ]`

两项：

- 如果优势 `A_t > 0` 且比率试图增长超过 `1 + ε`，裁剪压平梯度——不要将一个好动作推到旧概率 `+ε` 以上。
- 如果优势 `A_t < 0` 且比率试图增长超过 `1 - ε`（意味着我们使得一个坏动作更可能，相较于其裁剪的减少），裁剪限制梯度——不要将一个坏动作推到 `-ε` 以下。

`min` 处理另一个方向：如果比率已经向*有利*方向移动，你仍然获得梯度（对会伤害你的那一侧不裁剪）。

典型 `ε = 0.2`。将目标绘制为 `r_t` 的函数：在"好侧"有平坦屋顶和在"坏侧"有平坦地板的分段线性函数。

**完整 PPO 损失。**

`L(θ, φ) = L^{CLIP}(θ) - c_v · (V_φ(s_t) - V_t^{target})² + c_e · H(π_θ(·|s_t))`

与 A2C 相同的 actor-critic 结构。三个系数，通常 `c_v = 0.5`、`c_e = 0.01`、`ε = 0.2`。

**训练循环。**

1. 在 `N` 个并行 env 上收集 `N × T` 转移，每个 `T` 步。
2. 计算优势（GAE），冻结它们为常数。
3. 冻结 `π_{θ_old}` 为当前 `π_θ` 的快照。
4. 对于 `K` 个 epoch，对 `(s, a, A, V_target, log π_old(a|s))` 的每个小批量：
   - 计算 `r_t(θ) = exp(log π_θ(a|s) - log π_old(a|s))`。
   - 应用 `L^{CLIP}` + 价值损失 + 熵。
   - 梯度步。
5. 丢弃展开。返回步骤 1。

`K = 10` 和 64 的小批量是标准超参数集。PPO 是鲁棒的：精确数字在 ±50% 内很少重要。

**KL 惩罚变体。** 原始论文提出了使用自适应 KL 惩罚的替代方案：`L = L^{PG} - β · KL(π_θ || π_old)` 其中 `β` 根据观察到的 KL 调整。裁剪版本成为主导；KL 变体在 RLHF 中存活（其中到参考策略的 KL 是你无论如何都总想要的单独约束）。

## 动手构建

### 步骤 1：在展开时捕获 `log π_old(a | s)`

```python
for step in range(T):
    probs = softmax(logits(theta, state_features(s)))
    a = sample(probs, rng)
    s_next, r, done = env.step(s, a)
    buffer.append({
        "s": s, "a": a, "r": r, "done": done,
        "v_old": value(w, state_features(s)),
        "log_pi_old": log(probs[a] + 1e-12),
    })
    s = s_next
```

快照在展开时获取一次。它在更新 epochs 期间不改变。

### 步骤 2：计算 GAE 优势（第 07 课）

与 A2C 相同。跨批次归一化。

### 步骤 3：裁剪代理更新

```python
for _ in range(K_EPOCHS):
    for mb in minibatches(buffer, size=64):
        for rec in mb:
            x = state_features(rec["s"])
            probs = softmax(logits(theta, x))
            logp = log(probs[rec["a"]] + 1e-12)
            ratio = exp(logp - rec["log_pi_old"])
            adv = rec["advantage"]
            surrogate = min(
                ratio * adv,
                clamp(ratio, 1 - EPS, 1 + EPS) * adv,
            )
            # 反向传播 -surrogate，添加价值损失，减去熵
            grad_logpi = onehot(rec["a"]) - probs
            if (adv > 0 and ratio >= 1 + EPS) or (adv < 0 and ratio <= 1 - EPS):
                pg_grad = 0.0  # 被裁剪
            else:
                pg_grad = ratio * adv
            for i in range(N_ACTIONS):
                for j in range(N_FEAT):
                    theta[i][j] += LR * pg_grad * grad_logpi[i] * x[j]
```

"被裁剪 → 零梯度"的模式是 PPO 的核心。如果新策略已经在有利方向上漂移太远，更新停止。

### 步骤 4：价值和熵

对评论家目标添加标准 MSE，对演员添加熵奖励，与 A2C 相同。

### 步骤 5：诊断

每次更新要观察三件事：

- **平均 KL** `E[log π_old - log π_θ]`。应该保持在 `[0, 0.02]`。如果超过 `0.1`，减少 `K_EPOCHS` 或 `LR`。
- **裁剪比例**——比率落在 `[1-ε, 1+ε]` 之外的样本比例。应该是 `~0.1-0.3`。如果 `~0`，裁剪从未触发 → 提高 `LR` 或 `K_EPOCHS`。如果 `~0.5+`，你在过拟合展开 → 降低它们。
- **解释方差** `1 - Var(V_target - V_pred) / Var(V_target)`。评论家质量指标。应该随着评论家学习上升到 1。

## 缺陷陷阱

- **裁剪系数调错。** `ε = 0.2` 是事实标准。到 `0.1` 使更新太胆小；`0.3+` 引发不稳定。
- **太多 epoch。** `K > 20` 常规性破坏稳定，因为策略从 `π_old` 漂移太远。限制 epoch，尤其对大型网络。
- **无奖励归一化。** 大的奖励尺度侵蚀到裁剪范围中。在计算优势之前归一化奖励（运行标准差）。
- **忘记优势归一化。** 每批次零均值/单位标准差归一化是标准做法。跳过它破坏 PPO 在大多数基准上。
- **学习率不衰减。** PPO 受益于线性 LR 衰减到零。常数 LR 通常更差。
- **重要性比率数学错误。** 为数值稳定性始终使用 `exp(log_new - log_old)`，而不是 `new / old`。
- **错误的梯度符号。** 最大化代理 = *最小化* `-L^{CLIP}`。翻转的符号是最常见的 PPO 错误。

## 使用它

PPO 是 2026 年跨令人惊讶多数域的默认 RL 算法：

| 用例 | PPO 变体 |
|----------|-------------|
| MuJoCo / 机器人学控制 | 带有高斯策略的 PPO，GAE(0.95) |
| Atari / 离散游戏 | 带类别策略的 PPO，滚动 128 步展开 |
| LLM 的 RLHF | 带有对参考模型的 KL 惩罚的 PPO，在响应末尾来自 RM 的奖励 |
| 大规模游戏代理 | IMPALA + PPO（AlphaStar、OpenAI Five） |
| 推理 LLM | GRPO（第 12 课）——无评论家的 PPO 变体 |
| 仅偏好数据 | DPO——PPO+KL 的闭式折叠，无在线采样 |

PPO *损失形状*——裁剪代理 + 价值 + 熵——是 DPO、GRPO 和几乎每个 RLHF 流水线的脚手架。

## 交付成果

保存为 `outputs/skill-ppo-trainer.md`：

```markdown
---
name: ppo-trainer
description: 为给定环境产生 PPO 训练配置和诊断计划。
version: 1.0.0
phase: 9
lesson: 8
tags: [rl, ppo, policy-gradient]
---

给定环境和训练预算，输出：

1. 展开大小。`N` envs × `T` 步。
2. 更新调度。`K` epochs，小批量大小，LR 调度。
3. 代理参数。`ε`（裁剪）、`c_v`、`c_e`，优势归一化开启。
4. 优势。GAE(`λ`) 带显式 `γ` 和 `λ`。
5. 诊断计划。KL、裁剪比例、解释方差阈值带警报。

拒绝 `K > 30` 或 `ε > 0.3`（不安全信任域）。拒绝任何无优势归一化或 KL/裁剪监控的 PPO 运行。标记持续超过 0.4 的裁剪比例为漂移。
```

## 练习

1. **简单。** 在 4×4 GridWorld 上以 `ε=0.2, K=4` 运行 PPO。在匹配的环境步上比较与 A2C（每展开一个 epoch）的样本效率。
2. **中等。** 扫描 `K ∈ {1, 4, 10, 30}`。绘制回报 vs 环境步并跟踪每次更新的平均 KL。在此任务上什么 `K` 时 KL 爆炸？
3. **困难。** 用自适应 KL 惩罚替换裁剪代理（如果 `KL > 2·target` 加倍 `β`，如果 `KL < target/2` 减半）。比较最终回报、稳定性和无裁剪性。

## 关键术语

| 术语 | 人们怎么说 | 它实际上的含义 |
|------|-----------------|-----------------------|
| 重要性比率 | "r_t(θ)" | `π_θ(a|s) / π_old(a|s)`；与收集数据的策略的偏差。 |
| 裁剪代理 | "PPO 的主要技巧" | `min(r·A, clip(r, 1-ε, 1+ε)·A)`；在裁剪之后有益侧梯度平坦。 |
| 信任域 | "TRPO / PPO 意图" | 限制每次更新的 KL 以保证单调改进。 |
| KL 惩罚 | "软信任域" | 替代 PPO：`L - β · KL(π_θ || π_old)`。自适应 `β`。 |
| 裁剪比例 | "裁剪多常触发" | 诊断——应该 0.1-0.3；除此之外意味着调错。 |
| 多 epoch 训练 | "数据重用" | 在每个展开上 K epochs；用样本效率交易的方差成本。 |
| 近在策略 | "主要在策略" | PPO 名义上在策略但 K>1 epochs 安全使用略离策略数据。 |
| PPO-KL | "另一个 PPO" | KL 惩罚变体；在 RLHF 中使用，其中 KL 到参考已经是约束。 |

## 延伸阅读

- [Schulman 等人 (2017). Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — 论文。
- [Schulman 等人 (2015). Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) — TRPO，PPO 的前身。
- [Andrychowicz 等人 (2021). What Matters In On-Policy RL? A Large-Scale Empirical Study](https://arxiv.org/abs/2006.05990) — 每个 PPO 超参数消融。
- [Ouyang 等人 (2022). Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — InstructGPT；PPO 在 RLHF 中的配方。
- [OpenAI Spinning Up — PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) — 干净的带 PyTorch 的现代阐述。
- [CleanRL PPO implementation](https://github.com/vwxyzjn/cleanrl) — 许多论文使用的参考单文件 PPO。
- [Hugging Face TRL — PPOTrainer](https://huggingface.co/docs/trl/main/en/ppo_trainer) — 在语言模型上 PPO 的生产配方；与第 09 课（RLHF）一起阅读。
- [Engstrom 等人 (2020). Implementation Matters in Deep Policy Gradients](https://arxiv.org/abs/2005.12729) — "37 代码级优化"论文；哪些 PPO 技巧是承重的，哪些是民间传说。
