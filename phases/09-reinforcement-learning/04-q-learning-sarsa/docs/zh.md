# 时间差分 — Q-Learning 与 SARSA

> 蒙特卡洛等到回合结束。TD 通过自举下一个值估计在每一步后更新。Q-learning 是离策略且乐观的；SARSA 是在策略且谨慎的。两者都是一行代码。两者都支撑本阶段每个深度 RL 方法。

**类型：** 构建
**语言：** Python
**前置条件：** 第九阶段 · 01（MDP），第九阶段 · 02（动态规划），第九阶段 · 03（蒙特卡洛）
**时间：** 约 75 分钟

## 问题

蒙特卡洛有效，但它有两个昂贵的要求。它需要会终止的回合，且只在最终回报收到后才更新。如果你的回合是 1,000 步，MC 等待 1,000 步更新任何东西。它是高方差、低偏见的，且在实践中慢。

动态规划有相反的概况——零方差自举备份——但需要已知模型。

时间差分（TD）学习分割差异。从单个转移 `(s, a, r, s')`，形成一个一步目标 `r + γ V(s')` 并将 `V(s)` 推向它。无模型。不需要完整回合。在 RHS 上使用近似 `V` 引入偏见，但方差比 MC 大幅降低，且从第一步起在线更新。

这是所有现代 RL——DQN、A2C、PPO、SAC——所依赖的支点。第 9 阶段的其余部分是在你将在本课中编写的一步 TD 更新之上构建的函数近似和技巧层。

## 概念

![Q-learning vs SARSA：离策略 max vs 在策略 Q(s', a')](../assets/td.svg)

**TD(0) 对 V 的更新：**

`V(s) ← V(s) + α [r + γ V(s') - V(s)]`

括号中的量是 TD 误差 `δ = r + γ V(s') - V(s)`。它是 MC 中 `G_t - V(s_t)` 的在线类比。收敛要求 `α` 满足 Robbins-Monro（`Σ α = ∞`，`Σ α² < ∞`）且所有状态被无限频繁访问。

**Q-learning。** 用于控制的离策略 TD 方法：

`Q(s, a) ← Q(s, a) + α [r + γ max_{a'} Q(s', a') - Q(s, a)]`

`max` 假设*贪心*策略将从 `s'` 开始被遵循，无论代理实际采取什么动作。这种解耦使得 Q-learning 在代理通过 ε-greedy 探索的同时学习 `Q*`。Mnih 等人（2015）将其转换为 Atari 上的深度 Q-learning（第 05 课）。

**SARSA。** 一个在策略 TD 方法：

`Q(s, a) ← Q(s, a) + α [r + γ Q(s', a') - Q(s, a)]`

名字是元组 `(s, a, r, s', a')`。SARSA 使用代理*实际*采取的下一个动作 `a'`，而非贪心 `argmax`。对正在运行的任何 ε-greedy `π` 收敛到 `Q^π`，在极限 `ε → 0` 时变为 `Q*`。

**悬崖行走差异。** 在经典悬崖行走任务上（从悬崖坠落 = 奖励 -100），Q-learning 学习沿悬崖边缘的最优路径，但在探索期间偶尔受到惩罚。SARSA 学习离悬崖一步的安全路径，因为它将探索噪声纳入其 Q 值。随着训练，在 `ε → 0` 时两者都达到最优。在实践中很重要：当在部署时探索实际发生时，SARSA 的行为更保守。

**预期 SARSA。** 用其在 `π` 下的期望值替换 `Q(s', a')`：

`Q(s, a) ← Q(s, a) + α [r + γ Σ_{a'} π(a'|s') Q(s', a') - Q(s, a)]`

比 SARSA 更低方差（无 `a'` 的采样），相同的在策略目标。在现代教科书中通常是默认选择。

**n 步 TD 和 TD(λ)。** 通过在自举之前等待 `n` 步在 TD(0) 和 MC 之间插值。`n=1` 是 TD，`n=∞` 是 MC。TD(λ) 以几何权重 `(1-λ)λ^{n-1}` 对所有 `n` 进行平均。大多数深度 RL 使用介于 3 和 20 之间的 `n`。

```figure
qlearning-gridworld
```

## 动手构建

### 步骤 1：在 ε-greedy 策略上的 SARSA

```python
def sarsa(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})

    def choose(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        s = env.reset()
        a = choose(s)
        while True:
            s_next, r, done = env.step(s, a)
            a_next = choose(s_next) if not done else None
            target = r + (gamma * Q[s_next][a_next] if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s, a = s_next, a_next
    return Q
```

八行。与 Q-learning 的*唯一*区别是目标行。

### 步骤 2：Q-learning

```python
def q_learning(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    for _ in range(episodes):
        s = env.reset()
        while True:
            a = choose(s, Q, epsilon)
            s_next, r, done = env.step(s, a)
            target = r + (gamma * max(Q[s_next].values()) if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s = s_next
    return Q
```

`max` 将目标与行为解耦。那一符号是在策略和离策略之间的区别。

### 步骤 3：学习曲线

跟踪每 100 回合的平均回报。Q-learning 在简单确定性 GridWorld 上收敛更快；SARSA 在悬崖行走上更保守。在 `code/main.py` 的 4×4 GridWorld 上，两者在约 2,000 回合后以 `α=0.1, ε=0.1` 接近最优。

### 步骤 4：与 DP 真实值比较

运行值迭代（第 02 课）以获得 `Q*`。检查 `max_{s,a} |Q_learned(s,a) - Q*(s,a)|`。一个健康的表格 TD 代理在 10,000 回合后在 4×4 GridWorld 上落在 `~0.5` 以内。

## 缺陷陷阱

- **初始 Q 值重要。** 乐观初始化（`Q = 0` 用于负奖励任务）鼓励探索。悲观初始化可能永远困住贪心策略。
- **α 调度。** 常数 `α` 对非平稳问题没问题。衰减 `α_n = 1/n` 在理论上给出收敛，但在实践中太慢——将 `α` 固定在 `[0.05, 0.3]` 并监控学习曲线。
- **ε 调度。** 从高开始（`ε=1.0`），衰减到 `ε=0.05`。"GLIE"（在极限中贪心并具有无限探索）是收敛条件。
- **Q-learning 中的 max 偏置。** 当 `Q` 嘈杂时，`max` 操作偏高。导致高估——Hasselt 的双 Q-learning（第 05 课中 DDQN 使用的）用两个 Q 表修复此。
- **非终止回合。** TD 可以在没有终端的情况下学习，但你需要要么限制步数，要么在限制处正确处理自举。标准做法：将限制视为非终端，继续自举。
- **状态哈希。** 如果状态是元组/张量，使用可哈希的键（tuple，不是 list；四舍五入的浮点元组，不是原始值）。

## 使用它

2026 年 TD 版图：

| 任务 | 方法 | 原因 |
|------|--------|--------|
| 小型表格环境 | Q-learning | 直接学习最优策略。 |
| 在策略安全关键 | SARSA / Expected SARSA | 在探索期间保守。 |
| 高维状态 | DQN（第九阶段 · 05） | 具有重放和目标网络的神经网络 Q 函数。 |
| 连续动作 | SAC / TD3（第九阶段 · 07） | 在 Q 网络上的 TD 更新；策略网络发出动作。 |
| LLM RL（基于奖励模型） | PPO / GRPO（第九阶段 · 08、12） | 通过 GAE 具有 TD 风格优势的 actor-critic。 |
| 离线 RL | CQL / IQL（第九阶段 · 08） | 带有保守正则化的 Q-learning。 |

你在 2026 年论文中读到的"RL"的百分之九十是 Q-learning 或 SARSA 的某种展开。在深入之前用手指理解表格更新。

## 交付成果

保存为 `outputs/skill-td-agent.md`：

```markdown
---
name: td-agent
description: 在 Q-learning、SARSA、Expected SARSA 之间为表格或小特征 RL 任务选择。
version: 1.0.0
phase: 9
lesson: 4
tags: [rl, td-learning, q-learning, sarsa]
---

给定表格或小特征环境，输出：

1. 算法。Q-learning / SARSA / Expected SARSA / n 步变体。与在策略 vs 离策略和方差相关的一句原因。
2. 超参数。α、γ、ε、衰减调度。
3. 初始化。Q_0 值（乐观 vs 零）及证明。
4. 收敛诊断。目标学习曲线，如果 DP 可用则检查 `|Q - Q*|`。
5. 部署注意事项。推理时探索将如何表现？是否需要 SARSA 的保守性？

拒绝将表格 TD 应用于 > 10⁶ 的状态空间。拒绝发布不带 max 偏置注意事项的 Q-learning 代理。标记任何 ε 全程保持 1.0 训练的代理（无利用阶段）。
```

## 练习

1. **简单。** 在 4×4 GridWorld 上实现 Q-learning 和 SARSA。为 2,000 回合绘制学习曲线（每 100 回合的平均回报）。谁收敛更快？
2. **中等。** 构建悬崖行走环境（4×12，最后一行是奖励 -100 并重置到起点的悬崖）。比较 Q-learning 和 SARSA 的最终策略。截屏打印每条路径。哪个更靠近悬崖？
3. **困难。** 实现双 Q-learning。在嘈杂奖励 GridWorld（每步奖励添加高斯噪声 σ=5）上，展示 Q-learning 以有意义的量高估 `V*(0,0)`，而双 Q-learning 不高估。

## 关键术语

| 术语 | 人们怎么说 | 它实际上的含义 |
|------|-----------------|-----------------------|
| TD 误差 | "更新信号" | `δ = r + γ V(s') - V(s)`，自举残差。 |
| TD(0) | "一步 TD" | 每步之后仅使用下一状态的估计进行更新。 |
| Q-learning | "离策略 RL 101" | 对下一状态动作使用 `max` 的 TD 更新；无论行为策略如何学习 `Q*`。 |
| SARSA | "在策略 Q-learning" | 使用实际下一个动作的 TD 更新；为当前 ε-greedy π 学习 `Q^π`。 |
| 预期 SARSA | "低方差 SARSA" | 用其在 π 下的期望替换采样得到的 `a'`。 |
| GLIE | "正确的探索调度" | 在极限中贪心并具有无限探索；Q-learning 收敛需要的。 |
| 自举 | "在目标中使用当前估计" | 区分 TD 与 MC 的特征。偏见的来源但大幅方差减少。 |
| 最大化偏置 | "Q-learning 高估" | 对嘈杂估计的 `max` 是向上偏置的；由双 Q-learning 修复。 |

## 延伸阅读

- [Watkins 和 Dayan (1992). Q-learning](https://link.springer.com/article/10.1007/BF00992698) — 原始论文和收敛证明。
- [Sutton 和 Barto (2018). 第 6 章 — Temporal-Difference Learning](http://incompleteideas.net/book/RLbook2020.pdf) — TD(0)、SARSA、Q-learning、Expected SARSA。
- [Hasselt (2010). Double Q-learning](https://papers.nips.cc/paper_files/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html) — 最大化偏置的修复。
- [Seijen、Hasselt、Whiteson、Wiering (2009). A Theoretical and Empirical Analysis of Expected SARSA](https://ieeexplore.ieee.org/document/4927542) — 预期 SARSA 的动机。
- [Rummery 和 Niranjan (1994). On-line Q-learning using connectionist systems](https://www.researchgate.net/publication/2500611_On-Line_Q-Learning_Using_Connectionist_Systems) — 创造了 SARSA 的论文（当时称为"modified connectionist Q-learning"）。
- [Sutton 和 Barto (2018). 第 7 章 — n-step Bootstrapping](http://incompleteideas.net/book/RLbook2020.pdf) — 将 TD(0) 推广到 TD(n)，从 Q-learning 到资格迹和后来 PPO 中 GAE 的路径。
