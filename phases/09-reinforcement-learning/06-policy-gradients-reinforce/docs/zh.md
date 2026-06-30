# 策略梯度 — 从零开始的 REINFORCE

> 停止估计价值。直接参数化策略，计算预期回报的梯度，向山上迈步。Williams（1992）在一个定理中写下了它。这就是 PPO、GRPO 和每个 LLM RL 循环存在的原因。

**类型：** 构建
**语言：** Python
**前置条件：** 第三阶段 · 03（反向传播），第九阶段 · 03（蒙特卡洛），第九阶段 · 04（TD 学习）
**时间：** 约 75 分钟

## 问题

Q-learning 和 DQN 参数化*价值*函数。你通过 `argmax Q` 选择动作。这对离散动作和离散状态很好。当动作是连续的（对 10 维力矩的哪个 `argmax`？）或当你想要随机策略（`argmax` 按结构是确定性的）时，它失效了。

策略梯度改为参数化*策略*。`π_θ(a | s)` 是输出动作上分布的神经网络。从中采样以行动。计算相对于 `θ` 的预期回报梯度。向山上迈步。没有 `argmax`。没有 Bellman 递推。仅仅在 `J(θ) = E_{π_θ}[G]` 上的梯度上升。

REINFORCE 定理（Williams 1992）告诉你这个梯度是可计算的：`∇J(θ) = E_π[ G · ∇_θ log π_θ(a | s) ]`。运行一个回合。计算回报。乘以每步的 `∇ log π_θ(a | s)`。取平均。梯度上升。完成。

2026 年的每个 LLM-RL 算法——PPO、DPO、GRPO——都是 REINFORCE 的改良。用你的手指理解它是本阶段其余部分，以及第十阶段 · 07（RLHF 实现）和第十阶段 · 08（DPO）的先决条件。

## 概念

![策略梯度：softmax 策略、log-π 梯度、回报加权更新](../assets/policy-gradient.svg)

**策略梯度定理。** 对于由 `θ` 参数化的任何策略 `π_θ`：

`∇J(θ) = E_{τ ~ π_θ}[ Σ_{t=0}^{T} G_t · ∇_θ log π_θ(a_t | s_t) ]`

其中 `G_t = Σ_{k=t}^{T} γ^{k-t} r_{k+1}` 是从步 `t` 开始的折扣回报。期望是对从 `π_θ` 采样的完整轨迹 `τ`。

**证明简短。** 在期望下微分 `J(θ) = Σ_τ P(τ; θ) G(τ)`。使用 `∇P(τ; θ) = P(τ; θ) ∇ log P(τ; θ)`（对数导数技巧）。因式分解 `log P(τ; θ) = Σ log π_θ(a_t | s_t) + 不依赖于 θ 的环境项`。环境项消失。两行代数给出定理。

**方差减少技巧。** 标准 REINFORCE 有残忍的方差——回报嘈杂，`∇ log π` 嘈杂，它们的乘积非常嘈杂。两种标准修复：

1. **基线减法。** 将 `G_t` 替换为 `G_t - b(s_t)` 对于任何不依赖于 `a_t` 的基线 `b(s_t)`。无偏因为 `E[b(s_t) · ∇ log π(a_t | s_t)] = 0`。典型选择：`b(s_t) = V̂(s_t)` 由评论家学习 → actor-critic（第 07 课）。
2. **回报至终点。** 将 `Σ_t G_t · ∇ log π_θ(a_t | s_t)` 替换为 `Σ_t G_t^{from t} · ∇ log π_θ(a_t | s_t)`。只有未来回报对给定动作重要——过去回报贡献零均值噪声。

结合，你得到：

`∇J ≈ (1/N) Σ_{i=1}^{N} Σ_{t=0}^{T_i} [ G_t^{(i)} - V̂(s_t^{(i)}) ] · ∇_θ log π_θ(a_t^{(i)} | s_t^{(i)})`

即带有基线的 REINFORCE——A2C（第 07 课）和 PPO（第 08 课）的直接祖先。

**Softmax 策略参数化。** 对于离散动作，标准选择：

`π_θ(a | s) = exp(f_θ(s, a)) / Σ_{a'} exp(f_θ(s, a'))`

其中 `f_θ` 是任何输出每个动作分数的神经网络。梯度具有简洁形式：

`∇_θ log π_θ(a | s) = ∇_θ f_θ(s, a) - Σ_{a'} π_θ(a' | s) ∇_θ f_θ(s, a')`

即，所取动作的分数减去其在策略下的期望值。

**连续动作的高斯策略。** `π_θ(a | s) = N(μ_θ(s), σ_θ(s))`。`∇ log N(a; μ, σ)` 有闭式。这就是第 9 阶段 · 07 的 SAC 需要的一切。

```figure
policy-gradient-landscape
```

## 动手构建

### 步骤 1：softmax 策略网络

```python
def policy_logits(theta, state_features):
    return [dot(theta[a], state_features) for a in range(N_ACTIONS)]

def softmax(logits):
    m = max(logits)
    exps = [exp(l - m) for l in logits]
    Z = sum(exps)
    return [e / Z for e in exps]
```

对表格环境使用线性策略（每个动作一个权重向量）。对 Atari，换成 CNN 并保留 softmax 头。

### 步骤 2：采样和对数概率

```python
def sample_action(probs, rng):
    x = rng.random()
    cum = 0
    for a, p in enumerate(probs):
        cum += p
        if x <= cum:
            return a
    return len(probs) - 1

def log_prob(probs, a):
    return log(probs[a] + 1e-12)
```

### 步骤 3：捕获对数概率的展开

```python
def rollout(theta, env, rng, gamma):
    trajectory = []
    s = env.reset()
    while not done:
        logits = policy_logits(theta, s)
        probs = softmax(logits)
        a = sample_action(probs, rng)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r, probs))
        s = s_next
    return trajectory
```

### 步骤 4：REINFORCE 更新

```python
def reinforce_step(theta, trajectory, gamma, lr, baseline=0.0):
    returns = compute_returns(trajectory, gamma)
    for (s, a, _, probs), G in zip(trajectory, returns):
        advantage = G - baseline
        grad_log_pi_a = [-p for p in probs]
        grad_log_pi_a[a] += 1.0
        for i in range(N_ACTIONS):
            for j in range(len(s)):
                theta[i][j] += lr * advantage * grad_log_pi_a[i] * s[j]
```

梯度 `∇ log π(a|s) = e_a - π(·|s)`（`a` 的 onehot 减去概率）是 softmax 策略梯度的核心。将其烧入肌肉记忆。

### 步骤 5：基线

最近回合中 `G` 的运行均值是足够的方差减少，让 4×4 GridWorld 能够运行；大约 500 回合收敛。将基线升级为学习到的 `V̂(s)` 你就得到了 actor-critic。

## 缺陷陷阱

- **梯度爆炸。** 回报可能巨大。始终在乘以 `∇ log π` 之前在批次中归一化 `G` 到 `~N(0, 1)`。
- **熵崩溃。** 策略过早收敛到近确定性动作，停止探索，卡住。修复：添加熵奖励 `β · H(π(·|s))` 到目标。
- **高方差。** 标准 REINFORCE 需要数千回合。评论家基线（第 07 课）或 TRPO/PPO 的信任域（第 08 课）是标准修复。
- **样本低效。** 在策略意味着你在一次更新后丢弃每个转移。通过重要性采样的离策略校正带回数据，以方差为代价（PPO 的比率是裁剪的 IS 权重）。
- **非平稳梯度。** 来自 100 回合前的相同梯度使用旧的 `π`。在策略方法因此每几次展开更新一次。
- **信用分配。** 没有回报至终点，过去奖励贡献噪声。始终使用回报至终点。

## 使用它

在 2026 年，REINFORCE 很少直接运行，但其梯度公式无处不在：

| 用例 | 派生方法 |
|----------|---------------|
| 连续控制 | 带有高斯策略的 PPO / SAC |
| LLM RLHF | 带有 KL 惩罚的 PPO，在 token 级策略上运行 |
| LLM 推理（DeepSeek） | GRPO——带有群相对基线的 REINFORCE，无评论家 |
| 多代理 | 集中式评论家 REINFORCE（MADDPG、COMA） |
| 离散动作机器人学 | A2C、A3C、PPO |
| 仅偏好设置 | DPO——重写为偏好似然损失的 REINFORCE，无采样 |

当你在 2026 年训练脚本中读到 `loss = -advantage * log_prob` 时，那就是带有基线的 REINFORCE。整篇论文（DPO、GRPO、RLOO）都是在这一个行之上的方差减少技巧。

## 交付成果

保存为 `outputs/skill-policy-gradient-trainer.md`：

```markdown
---
name: policy-gradient-trainer
description: 为给定任务产生 REINFORCE / actor-critic / PPO 训练配置，并诊断方差问题。
version: 1.0.0
phase: 9
lesson: 6
tags: [rl, policy-gradient, reinforce]
---

给定环境（离散 / 连续动作、视野、奖励统计），输出：

1. 策略头。Softmax（离散）或高斯（连续），带有参数计数。
2. 基线。无（标准）、运行均值、学习到的 `V̂(s)`、或 A2C 评论家。
3. 方差控制。默认开启回报至终点、回报归一化、梯度裁剪值。
4. 熵奖励。系数 β 和衰减调度。
5. 批量大小。每次更新的回合数；在策略数据新鲜度契约。

拒绝在 > 500 步视野上使用 REINFORCE 无基线。拒绝使用 softmax 头的连续动作控制。标记任何 `β = 0` 且观察到策略熵 < 0.1 的运行已熵崩溃。
```

## 练习

1. **简单。** 在 4×4 GridWorld 上用线性 softmax 策略实现 REINFORCE。在无基线下训练 1,000 回合。绘制学习曲线；测量方差（回报的标准差）。
2. **中等。** 添加运行均值基线。再次训练。将样本效率和方差与标准运行进行比较。基线减少了多少收敛步数？
3. **困难。** 添加熵奖励 `β · H(π)`。扫描 `β ∈ {0, 0.01, 0.1, 1.0}`。绘制最终回报和策略熵。此任务的甜蜜点在哪里？

## 关键术语

| 术语 | 人们怎么说 | 它实际上的含义 |
|------|-----------------|-----------------------|
| 策略梯度 | "直接训练策略" | `∇J(θ) = E[G · ∇ log π_θ(a|s)]`；从对数导数技巧导出。 |
| REINFORCE | "原始 PG 算法" | Williams (1992)；乘以对数策略梯度的蒙特卡洛回报。 |
| 对数导数技巧 | "分数函数估计器" | `∇P(τ;θ) = P(τ;θ) · ∇ log P(τ;θ)`；使期望的梯度可处理。 |
| 基线 | "方差减少" | 从 `G` 减去的任何 `b(s)`；无偏因为 `E[b · ∇ log π] = 0`。 |
| 回报至终点 | "只有未来回报计数" | `G_t^{from t}` 而不是完整的 `G_0`；正确且更低方差。 |
| 熵奖励 | "鼓励探索" | `+β · H(π(·|s))` 项保持策略不崩溃。 |
| 在策略 | "在你刚看到的东西上训练" | 梯度期望关于当前策略——不能直接重用旧数据。 |
| 优势 | "比平均好多少" | `A(s, a) = G(s, a) - V(s)`；REINFORCE 带基线乘以此签名量。 |

## 延伸阅读

- [Williams (1992). Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://link.springer.com/article/10.1007/BF00992696) — 原始 REINFORCE 论文。
- [Sutton 等人 (2000). Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html) — 带有函数近似的现代策略梯度定理。
- [Sutton 和 Barto (2018). 第 13 章 — Policy Gradient Methods](http://incompleteideas.net/book/RLbook2020.pdf) — 教科书呈现。
- [OpenAI Spinning Up — VPG / REINFORCE](https://spinningup.openai.com/en/latest/algorithms/vpg.html) — 清晰的带有 PyTorch 代码的教学阐述。
- [Peters 和 Schaal (2008). Reinforcement Learning of Motor Skills with Policy Gradients](https://homes.cs.washington.edu/~todorov/courses/amath579/reading/PolicyGradient.pdf) — 将 REINFORCE 与信任域家族（TRPO、PPO）连接的方差减少和自然梯度视角。
