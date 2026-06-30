# 蒙特卡洛方法 — 从完整回合中学习

> 动态规划需要模型。蒙特卡洛只需要回合。运行策略，观察回报，取平均。RL 中最简单的思想——也是解锁下游一切的思想。

**类型：** 构建
**语言：** Python
**前置条件：** 第九阶段 · 01（MDP），第九阶段 · 02（动态规划）
**时间：** 约 75 分钟

## 问题

动态规划优雅，但它假设你可以对每个状态和动作查询 `P(s' | s, a)`。现实世界中几乎没有东西以那种方式工作。机器人不能解析地计算关节力矩后相机像素上的分布。定价算法不能对每个可能的客户反应进行积分。LLM 不能在 token 之后枚举所有可能的续写。

你需要一种只需要*采样*环境能力的方法。运行策略。获得轨迹 `s_0, a_0, r_1, s_1, a_1, r_2, …, s_T`。用它来估计价值。这就是蒙特卡洛。

从 DP 到 MC 的转变在哲学上是重要的：我们从*已知模型 + 精确备份*移动到*采样展开 + 平均回报*。方差跃升，但适用性爆炸。本课之后的每个 RL 算法——TD、Q-learning、REINFORCE、PPO、GRPO——本质上都是蒙特卡洛估计器，有时上面叠加了自举。

## 概念

![蒙特卡洛：展开、计算回报、取平均；首次访问 vs 每次访问](../assets/monte-carlo.svg)

**核心思想，一行：** `V^π(s) = E_π[G_t | s_t = s] ≈ (1/N) Σ_i G^{(i)}(s)` 其中 `G^{(i)}(s)` 是在策略 `π` 下访问 `s` 后观察到的回报。

**首次访问 vs 每次访问 MC。** 给定一个多次访问状态 `s` 的回合，首次访问 MC 只计数首次访问的回报；每次访问 MC 计数所有访问。两者在极限中都是无偏的。首次访问更简单分析（iid 样本）。每次访问使用更多每个回合的数据且在实践中通常收敛更快。

**增量均值。** 与其存储所有回报，不如更新运行平均：

`V_n(s) = V_{n-1}(s) + (1/n) [G_n - V_{n-1}(s)]`

重排：`V_new = V_old + α · (target - V_old)` 其中 `α = 1/n`。将 `1/n` 替换为常数步长 `α ∈ (0, 1)`，你就得到了跟踪 `π` 变化的非平稳 MC 估计器。这一移动是从 MC 到 TD 到每个现代 RL 算法的整个跳跃。

**探索现在是一个问题。** DP 通过枚举触及每个状态。MC 只看到策略访问的状态。如果 `π` 是确定性的，状态空间的整个区域永远不会被采样，它们的价值估计永远停留在零。三种修复，按历史顺序：

1. **探索起始。** 从随机 (s, a) 对开始每个回合。保证覆盖；在实践中不现实（你不能将机器人"重置"到任意状态）。
2. **ε-greedy。** 关于当前 Q 贪心行动，但以概率 `ε` 选择随机动作。所有状态-动作对渐近地被采样。
3. **离策略 MC。** 在行为策略 `μ` 下收集数据，通过重要性采样了解目标策略 `π`。方差高，但它是通向像 DQN 这样的重放缓冲区方法的桥梁。

**蒙特卡洛控制。** 评估 → 改进 → 评估，就像策略迭代，但评估是基于采样的：

1. 运行 `π`，获得一个回合。
2. 从观察到的回报更新 `Q(s, a)`。
3. 使 `π` 关于 `Q` ε-greedy。
4. 重复。

在温和条件下以概率 1 收敛到 `Q*` 和 `π*`（每对被无限频繁访问，`α` 满足 Robbins-Monro）。

```figure
epsilon-greedy
```

## 动手构建

### 步骤 1：展开 → (s, a, r) 列表

```python
def rollout(env, policy, max_steps=200):
    trajectory = []
    s = env.reset()
    for _ in range(max_steps):
        a = policy(s)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r))
        s = s_next
        if done:
            break
    return trajectory
```

无模型，只有 `env.reset()` 和 `env.step(s, a)`。与 gym 环境相同的接口，但已精简。

### 步骤 2：计算回报（反向扫描）

```python
def returns_from(trajectory, gamma):
    returns = []
    G = 0.0
    for _, _, r in reversed(trajectory):
        G = r + gamma * G
        returns.append(G)
    return list(reversed(returns))
```

一次传递，`O(T)`。反向递推 `G_t = r_{t+1} + γ G_{t+1}` 避免重新求和。

### 步骤 3：首次访问 MC 评估

```python
def mc_policy_evaluation(env, policy, episodes, gamma=0.99):
    V = defaultdict(float)
    counts = defaultdict(int)
    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for t, ((s, _, _), G) in enumerate(zip(trajectory, returns)):
            if s in seen:
                continue
            seen.add(s)
            counts[s] += 1
            V[s] += (G - V[s]) / counts[s]
    return V
```

三行代码完成工作：首次访问时标记状态为已见，增加计数，更新运行均值。

### 步骤 4：ε-greedy MC 控制（在策略）

```python
def mc_control(env, episodes, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    counts = defaultdict(lambda: {a: 0 for a in ACTIONS})

    def policy(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for (s, a, _), G in zip(trajectory, returns):
            if (s, a) in seen:
                continue
            seen.add((s, a))
            counts[s][a] += 1
            Q[s][a] += (G - Q[s][a]) / counts[s][a]
    return Q, policy
```

### 步骤 5：与 DP 黄金标准比较

随着回合数 → ∞，你对 `V^π` 的 MC 估计应该与第 02 课的 DP 结果一致。在实践中：在 4×4 GridWorld 上 50,000 个回合让你到达 DP 答案的 `~0.1` 以内。

## 缺陷陷阱

- **无限回合。** MC 要求回合有*终点*。如果你的策略可以永远循环，限制 `max_steps` 并将限制视为隐式失败。具有随机策略的 GridWorld 常规地超时——这是正常的，只需确保正确计数。
- **方差。** MC 使用完整回报。在长回合上，方差巨大——末尾一次不走运的奖励以相同量移动 `V(s_0)`。TD 方法（第 04 课）通过自举削减此。
- **状态覆盖。** 在新鲜 Q 上带平局的贪心 MC 将只尝试一个动作。你*必须*探索（ε-greedy、探索起始、UCB）。
- **非平稳策略。** 如果 `π` 变化（如 MC 控制中），旧回报来自不同的策略。常数 α MC 处理此；样本平均 MC 不处理。
- **离策略重要性采样。** 权重 `π(a|s)/μ(a|s)` 在轨迹上相乘。方差随视野爆炸。用每决定加权 IS 限制或切换到 TD。

## 使用它

2026 年蒙特卡洛方法的角色：

| 用例 | 为什么 MC |
|----------|--------|
| 短视野游戏（21 点、扑克） | 回合自然结束；回报干净。 |
| 日志策略的离线评估 | 在存储的轨迹上平均折扣回报。 |
| 蒙特卡洛树搜索（AlphaZero） | 来自树叶的 MC 展开指导选择。 |
| LLM RL 评估 | 为给定策略在采样的补全上计算平均奖励。 |
| PPO 中的基线估计 | 优势目标 `A_t = G_t - V(s_t)` 使用 MC `G_t`。 |
| 教学 RL | 实际有效的最简单算法——去除自举以看到核心。 |

现代深度 RL 算法（PPO、SAC）通过 `n`-步回报或 GAE 在纯 MC（完整回报）和纯 TD（一步自举）之间插值。两个端点都是同一估计器的实例。

## 交付成果

保存为 `outputs/skill-mc-evaluator.md`：

```markdown
---
name: mc-evaluator
description: 通过蒙特卡洛展开评估策略并产生收敛报告，如果可用则附带 DP 比较。
version: 1.0.0
phase: 9
lesson: 3
tags: [rl, monte-carlo, evaluation]
---

给定环境（有回合，具有 reset+step API）和策略，输出：

1. 方法。首次访问 vs 每次访问 MC。原因。
2. 回合预算。目标数量，方差诊断，预期标准误差。
3. 探索计划。ε 调度（如果需要）或探索起始。
4. 黄金标准比较。如果表格则 DP 最优 V*；否则来自 Q-learning / PPO 基线的界限。
5. 终止检查。最大步限制，超时，非终止轨迹的处理。

拒绝在非回合任务上无有限视野限制地运行 MC。拒绝在表格任务上每个状态少于 100 回合报告 V^π 估计。标记任何具有零方差动作的策略为探索风险。
```

## 练习

1. **简单。** 在 4×4 GridWorld 上实现均匀随机策略的首次访问 MC 评估。运行 10,000 个回合。将 `V(0,0)` 绘制为回合计数的函数，对照 DP 答案。
2. **中等。** 实现 ε-greedy MC 控制，`ε ∈ {0.01, 0.1, 0.3}`。在 20,000 回合后比较平均回报。曲线看起来怎样？偏差-方差权衡在哪里？
3. **困难。** 实现带重要性采样的*离策略* MC：在均匀随机策略 `μ` 下收集数据，为确定性最优策略 `π` 估计 `V^π`。比较纯 IS vs 每决定 IS vs 加权 IS。哪个方差最低？

## 关键术语

| 术语 | 人们怎么说 | 它实际上的含义 |
|------|-----------------|-----------------------|
| 蒙特卡洛 | "随机采样" | 通过对来自分布的 iid 样本取平均来估计期望。 |
| 回报 `G_t` | "未来奖励" | 从步 `t` 到回合结束的折扣奖励和：`Σ_{k≥0} γ^k r_{t+k+1}`。 |
| 首次访问 MC | "每个状态计数一次" | 只有回合中的首次访问贡献价值估计。 |
| 每次访问 MC | "使用所有访问" | 每次访问都贡献；略有偏但更样本高效。 |
| ε-greedy | "探索噪声" | 以概率 `1-ε` 选择贪心动作；以概率 `ε` 选择随机动作。 |
| 重要性采样 | "校正从错误分布中采样" | 通过 `π(a|s)/μ(a|s)` 乘积重新加权回报以从 `μ` 数据估计 `V^π`。 |
| 在策略 | "从我自己的数据中学习" | 目标策略 = 行为策略。标准 MC、PPO、SARSA。 |
| 离策略 | "从别人的数据中学习" | 目标策略 ≠ 行为策略。重要性采样 MC、Q-learning、DQN。 |

## 延伸阅读

- [Sutton 和 Barto (2018). 第 5 章 — Monte Carlo Methods](http://incompleteideas.net/book/RLbook2020.pdf) — 规范处理。
- [Singh 和 Sutton (1996). Reinforcement Learning with Replacing Eligibility Traces](https://link.springer.com/article/10.1007/BF00114726) — 首次访问 vs 每次访问分析。
- [Precup、Sutton、Singh (2000). Eligibility Traces for Off-Policy Policy Evaluation](http://incompleteideas.net/papers/PSS-00.pdf) — 离策略 MC 和方差控制。
- [Mahmood 等人 (2014). Weighted Importance Sampling for Off-Policy Learning](https://arxiv.org/abs/1404.6362) — 现代低方差 IS 估计器。
- [Tesauro (1995). TD-Gammon, A Self-Teaching Backgammon Program](https://dl.acm.org/doi/10.1145/203330.203343) — MC/TD 自我对弈收敛到超人类水平的首次大规模实证演示；本阶段后半部分每一课的概念先驱。
