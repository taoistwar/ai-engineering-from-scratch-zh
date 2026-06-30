# Actor-Critic — A2C 与 A3C

> REINFORCE 嘈杂。添加一个学习 `V̂(s)` 的评论家，从回报中减去它，你得到了具有相同期望但方差远低的优势。这就是 actor-critic。A2C 同步运行它；A3C 跨线程运行它。两者都是每个现代深度 RL 方法的心理模型。

**类型：** 构建
**语言：** Python
**前置条件：** 第九阶段 · 04（TD 学习），第九阶段 · 06（REINFORCE）
**时间：** 约 75 分钟

## 问题

标准 REINFORCE 有效，但其方差可怕。蒙特卡洛回报 `G_t` 在回合之间可以摆动超过 10 倍。将该噪声乘以 `∇ log π` 并平均产生一个梯度估计器，需要数千回合才能移动策略，而你用少得多的 DQN 更新就能移动相同的距离。

方差来自使用原始回报。如果你减去一个基线 `b(s_t)`——状态的任何函数，包括学习到的价值——期望不变且方差下降。最佳可处理基线是 `V̂(s_t)`。现在乘以 `∇ log π` 的量是*优势*：

`A(s, a) = G - V̂(s)`

如果一个动作产生了高于平均的回报，它是好的；低于则是坏的。带有学习评论家的 REINFORCE 就是 *actor-critic*。评论家给予演员低方差的教师。这是 2015 年后的每个深度策略方法（A2C、A3C、PPO、SAC、IMPALA）。

## 概念

![Actor-critic：策略网络加价值网络，TD 残差作为优势](../assets/actor-critic.svg)

**两个网络，一个共享损失：**

- **演员** `π_θ(a | s)`：策略。采样以行动。用策略梯度训练。
- **评论家** `V_φ(s)`：估计来自状态的预期回报。训练最小化 `(V_φ(s) - target)²`。

**优势。** 两种标准形式：

- *MC 优势：* `A_t = G_t - V_φ(s_t)`。无偏，方差更高。
- *TD 优势：* `A_t = r_{t+1} + γ V_φ(s_{t+1}) - V_φ(s_t)`。有偏（使用 `V_φ`），方差低得多。也称为 *TD 残差* `δ_t`。

**n 步优势。** 在两者之间插值：

`A_t^{(n)} = r_{t+1} + γ r_{t+2} + … + γ^{n-1} r_{t+n} + γ^n V_φ(s_{t+n}) - V_φ(s_t)`

`n = 1` 是纯 TD。`n = ∞` 是 MC。大多数实现对 Atari 使用 `n = 5`，对 MuJoCo 上的 PPO 使用 `n = 2048`。

**广义优势估计（GAE）。** Schulman 等人（2016）提出对所有 n 步优势的指数加权平均：

`A_t^{GAE} = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}`

其中 `λ ∈ [0, 1]`。`λ = 0` 是 TD（低方差，高偏置）。`λ = 1` 是 MC（高方差，无偏）。`λ = 0.95` 是 2026 年默认值——调整直到偏置/方差旋钮在你想要的位置。

**A2C：同步优势 actor-critic。** 在 `N` 个并行环境中收集 `T` 步。为每步计算优势。在组合批次上更新演员和评论家。重复。A3C 的更简单、更具扩展性的兄弟。

**A3C：异步优势 actor-critic。** Mnih 等人（2016）。生成 `N` 个工作线程，每个运行一个环境。每个工作线程在其自己的展开上本地计算梯度，然后异步将它们应用到共享参数服务器。不需要重放缓冲区——工作线程通过运行不同轨迹来去相关。A3C 证明你可以在 CPU 上大规模训练。在 2026 年，基于 GPU 的 A2C（批量并行环境）占主导，因为 GPU 想要大批量。

**组合损失。**

`L(θ, φ) = -E[ A_t · log π_θ(a_t | s_t) ]  +  c_v · E[(V_φ(s_t) - G_t)²]  -  c_e · E[H(π_θ(·|s_t))]`

三项：策略梯度损失、价值回归、熵奖励。`c_v ~ 0.5`、`c_e ~ 0.01` 是规范起点。

## 动手构建

### 步骤 1：评论家

线性评论家 `V_φ(s) = w · features(s)` 用 MSE 更新：

```python
def critic_update(w, x, target, lr):
    v_hat = dot(w, x)
    err = target - v_hat
    for j in range(len(w)):
        w[j] += lr * err * x[j]
    return v_hat
```

在表格环境中评论家在几百回合内收敛。在 Atari 上，用共享 CNN 主干 + 价值头替换线性评论家。

### 步骤 2：n 步优势

给定长度 `T` 的展开和自举的最终 `V(s_T)`：

```python
def compute_advantages(rewards, values, gamma=0.99, lam=0.95, last_value=0.0):
    advantages = [0.0] * len(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_v = values[t + 1] if t + 1 < len(values) else last_value
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns
```

`returns` 是评论家目标。`advantages` 是乘以 `∇ log π` 的东西。

### 步骤 3：组合更新

```python
for step_i, (x, a, _r, probs) in enumerate(traj):
    adv = advantages[step_i]
    target_v = returns[step_i]

    # 评论家
    critic_update(w, x, target_v, lr_v)

    # 演员
    for i in range(N_ACTIONS):
        grad_logpi = (1.0 if i == a else 0.0) - probs[i]
        for j in range(N_FEAT):
            theta[i][j] += lr_a * adv * grad_logpi * x[j]
```

在策略，每次展开一次更新，演员和评论家分别的学习率。

### 步骤 4：并行化（A3C vs A2C）

- **A3C：** 启动 `N` 个线程。每个运行自己的环境和自己前向传播。周期性地将梯度更新推送到共享主控。主控没有锁——竞争是可以的，它们只是添加噪声。
- **A2C：** 在单个进程中运行 `N` 个环境实例，将观测堆叠到 `[N, obs_dim]` 批次，批量前向传播，批量反向传播。更高的 GPU 利用率，确定性的，更容易推理。2026 年的默认方案。

我们的玩具代码为清晰起见是单线程的；重写为批量 A2C 只需三行 numpy。

## 缺陷陷阱

- **演员梯度前的评论家偏置。** 如果评论家是随机的，其基线是无信息的，你在纯噪声上训练。在打开策略梯度之前预热评论家几百步，或使用慢速演员学习率。
- **优势归一化。** 每批次将优势归一化为零均值/单位标准差。以近乎零的成本极大地稳定了训练。
- **共享主干。** 在图像输入上为演员和评论家使用共享特征提取器。分离头。共享特征免费搭载两个损失。
- **在策略契约。** A2C 确恰好一次更新重用数据。更多则你的梯度有偏（重要性采样校正是 PPO 添加的）。
- **熵崩溃。** 没有 `c_e > 0`，策略在几百次更新内变成近确定性的并停止探索。
- **奖励尺度。** 优势幅度取决于奖励尺度。归一化奖励（例如运行标准差除法）以在任务之间获得一致的梯度幅度。

## 使用它

A2C/A3C 在 2026 年很少是最终选择，但它们是后来一切改良的架构：

| 方法 | 与 A2C 的关系 |
|--------|----------------|
| PPO | A2C + 裁剪的重要性比率用于多 epoch 更新 |
| IMPALA | A3C + V-trace 离策略校正 |
| SAC（第九阶段 · 07） | 带有软价值评论家的离策略 A2C（下一课） |
| GRPO（第九阶段 · 12） | 无评论家的 A2C——群相对优势 |
| DPO | 折叠成偏好排名损失的 A2C，无采样 |
| AlphaStar / OpenAI Five | 带有联赛训练 + 模仿预训练的 A2C |

如果你在 2026 年论文中看到"优势"，想想 actor-critic。

## 交付成果

保存为 `outputs/skill-actor-critic-trainer.md`：

```markdown
---
name: actor-critic-trainer
description: 为给定环境产生 A2C / A3C / GAE 配置，指定优势估计和损失权重。
version: 1.0.0
phase: 9
lesson: 7
tags: [rl, actor-critic, gae]
---

给定环境和计算预算，输出：

1. 并行性。A2C（GPU 批量）vs A3C（CPU 异步）和工作线程数。
2. 展开长度 T。每次更新每个 env 的步数。
3. 优势估计器。n 步或 GAE(λ)；指定 λ。
4. 损失权重。`c_v`（价值）、`c_e`（熵）、梯度裁剪。
5. 学习率。演员和评论家（如果使用则分开）。

拒绝在视野 > 1000 的环境上使用单工作线程 A2C（太在策略，太慢）。拒绝发布没有优势归一化的。标记任何 `c_e = 0` 且观察到熵 < 0.1 的运行已熵崩溃。
```

## 练习

1. **简单。** 在 4×4 GridWorld 上用 MC 优势（`G_t - V(s_t)`）训练 actor-critic。将样本效率与第 06 课的 REINFORCE 带运行均值基线比较。
2. **中等。** 切换到 TD 残差优势（`r + γ V(s') - V(s)`）。测量优势批次的方差。下降了多少？
3. **困难。** 实现 GAE(λ)。扫描 `λ ∈ {0, 0.5, 0.9, 0.95, 1.0}`。绘制最终回报 vs 样本效率。此任务的偏置/方差甜蜜点在哪里？

## 关键术语

| 术语 | 人们怎么说 | 它实际上的含义 |
|------|-----------------|-----------------------|
| 演员 | "策略网络" | `π_θ(a|s)`，由策略梯度更新。 |
| 评论家 | "价值网络" | `V_φ(s)`，由回归到回报 / TD 目标的 MSE 更新。 |
| 优势 | "比平均好多少" | `A(s, a) = Q(s, a) - V(s)` 或其估计器。`∇ log π` 的乘数。 |
| TD 残差 | "δ" | `δ_t = r + γ V(s') - V(s)`；一步优势估计。 |
| GAE | "插值旋钮" | n 步优势的指数加权和，由 `λ` 参数化。 |
| A2C | "同步 actor-critic" | 跨环境批量；每次展开一个梯度步。 |
| A3C | "异步 actor-critic" | 工作线程将梯度推送到共享参数服务器。原始论文；在 2026 年较少见。 |
| 自举 | "在视野处使用 V" | 截断展开，添加 `γ^n V(s_{t+n})` 来闭合和。 |

## 延伸阅读

- [Mnih 等人 (2016). Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) — A3C，原始异步 actor-critic 论文。
- [Schulman 等人 (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) — GAE。
- [Sutton 和 Barto (2018). 第 13 章 — Actor-Critic Methods](http://incompleteideas.net/book/RLbook2020.pdf) — 基础知识；当评论家是神经网络时，将这章与第 9 章的函数近似配对。
- [Espeholt 等人 (2018). IMPALA](https://arxiv.org/abs/1802.01561) — 带有 V-trace 离策略校正的可扩展分布式 actor-critic。
- [OpenAI Baselines / Stable-Baselines3](https://stable-baselines3.readthedocs.io/) — 值得阅读的生产 A2C/PPO 实现。
- [Konda 和 Tsitsiklis (2000). Actor-Critic Algorithms](https://papers.nips.cc/paper/1786-actor-critic-algorithms) — 两时间尺度 actor-critic 分解的基础收敛结果。
