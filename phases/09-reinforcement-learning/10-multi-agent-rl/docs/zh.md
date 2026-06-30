# 多代理 RL

> 单代理 RL 假设环境是平稳的。把两个学习代理放在同一个世界中，这个假设就被打破了：每个代理都是另一个代理环境的一部分，且两者都在变化。多代理 RL 是当马尔可夫假设不再成立时使学习收敛的一组技巧。

**类型：** 构建
**语言：** Python
**前置条件：** 第九阶段 · 04（Q-learning），第九阶段 · 06（REINFORCE），第九阶段 · 07（Actor-Critic）
**时间：** 约 45 分钟

## 问题

学习在房间中导航的机器人是单代理 RL 问题。足球队不是。AlphaStar 对抗星际争霸对手不是。投标代理的市场不是。两辆车协商四向停车的不是。多对多的现实世界问题不是。

在每个多代理环境中，从任何一个代理的角度看，其他代理*是*环境的一部分。随着它们学习并改变行为，环境变得非平稳。马尔可夫性质——"下一状态仅取决于当前状态和我的动作"——被违反，因为下一状态还取决于*其他*代理的选择，而它们的策略是移动目标。

这破坏了表格收敛证明（Q-learning 的保证假设平稳环境）。它也破坏了朴素的深度 RL：代理在循环中互相追逐，永远不收敛到稳定策略。你需要多代理特定的技巧：集中训练/分散执行、反事实基线、联赛游戏、自我对弈。

2026 年应用：机器人群、交通路由、自动驾驶车队、市场模拟器、多代理 LLM 系统（第十六阶段），以及任何具有多个智能参与者的游戏。

## 概念

![四个 MARL 体制：独立、集中评论家、自我对弈、联赛](../assets/marl.svg)

**形式化：马尔可夫博弈。** MDP 的推广：状态 `S`，联合动作 `a = (a_1, …, a_n)`，转移 `P(s' | s, a)`，以及每代理奖励 `R_i(s, a, s')`。每个代理 `i` 在其自己的策略 `π_i` 下最大化其自己的回报。如果奖励相同，是**完全合作的**。如果零和，是**对抗的**。如果混合，是**一般和的**。

**核心挑战：**

- **非平稳性。** 从代理 `i` 视角的 `P(s' | s, a_i)` 取决于 `π_{-i}`，它在变化。
- **信用分配。** 使用共享奖励，哪个代理导致了这个？
- **探索协调。** 代理必须探索互补策略，而非冗余探索相同状态。
- **可扩展性。** 联合动作空间随 `n` 指数增长。
- **部分可观察性。** 每个代理只看到自己的观测；全局状态是隐藏的。

**四种主导体制：**

**1. 独立 Q-learning / 独立 PPO（IQL、IPPO）。** 每个代理学习自己的 Q 或策略，将其他代理视为环境的一部分。简单，有时有效（尤其经验重放扮演平滑代理建模技巧）。理论收敛：无。实践：对松散耦合任务还好，对紧密耦合任务差。

**2. 集中训练、分散执行（CTDE）。** 最常见的现代范式。每个代理有自己的*策略* `π_i`，以本地观测 `o_i` 为条件——在部署时标准分散执行。在*训练*期间，集中式评论家 `Q(s, a_1, …, a_n)` 以完整全局状态和联合动作为条件。示例：
- **MADDPG**（Lowe 等人 2017）：每代理带有集中评论家的 DDPG。
- **COMA**（Foerster 等人 2017）：反事实基线——问"如果我做了动作 `a'` 我的奖励会是多少？"——隔离我的贡献。
- **MAPPO** / **IPPO** 带有共享评论家（Yu 等人 2022）：带有集中值函数的 PPO。2026 年合作 MARL 中占优势。
- **QMIX**（Rashid 等人 2018）：值分解——`Q_tot(s, a) = f(Q_1(s, a_1), …, Q_n(s, a_n))` 具有单调混合。

**3. 自我对弈。** 相同代理的两个副本互相博弈。对手的策略*是*我过去快照的策略。AlphaGo / AlphaZero / MuZero。OpenAI Five。对零和博弈效果最好；训练信号是对称的。

**4. 联赛游戏。** 自我对弈到一般和/对抗环境的扩展：保持过去和当前策略的种群，从联赛中采样对手，对抗它们训练。添加利用者（专精击败当前最佳的）和主要利用者（专精击败利用者的）。AlphaStar（星际争霸 II）。当博弈承认"剪刀石头布"策略循环时需要。

**通信。** 允许代理互相发送学习到的消息 `m_i`。在合作环境下有效。Foerster 等人（2016）表明可微的代理间通信可以端到端训练。今天的基于 LLM 的多代理系统（第十六阶段）本质上以自然语言通信。

## 动手构建

本课使用具有两个合作代理的 6×6 GridWorld。它们从对角开始，必须到达共享目标。共享奖励：当任一代理仍在移动时 -1，当两者都到达时 +10。参见 `code/main.py`。

### 步骤 1：多代理环境

```python
class CoopGridWorld:
    def __init__(self):
        self.size = 6
        self.goal = (5, 5)

    def reset(self):
        return ((0, 0), (5, 0))  # 两个代理

    def step(self, state, actions):
        a1, a2 = state
        new1 = move(a1, actions[0])
        new2 = move(a2, actions[1])
        done = (new1 == self.goal) and (new2 == self.goal)
        reward = 10.0 if done else -1.0
        return (new1, new2), reward, done
```

*联合*动作空间是 `|A|² = 16`。全局状态是两个位置。

### 步骤 2：独立 Q-learning

每个代理运行自己的 Q 表，以联合状态为键。每步：两者选择 ε-greedy 动作，收集联合转移，每个用自己的 Q 以共享奖励更新。

```python
def independent_q(env, episodes, alpha, gamma, epsilon):
    Q1, Q2 = defaultdict(default_q), defaultdict(default_q)
    for _ in range(episodes):
        s = env.reset()
        while not done:
            a1 = epsilon_greedy(Q1, s, epsilon)
            a2 = epsilon_greedy(Q2, s, epsilon)
            s_next, r, done = env.step(s, (a1, a2))
            target1 = r + gamma * max(Q1[s_next].values())
            target2 = r + gamma * max(Q2[s_next].values())
            Q1[s][a1] += alpha * (target1 - Q1[s][a1])
            Q2[s][a2] += alpha * (target2 - Q2[s][a2])
            s = s_next
```

在这个任务上工作因为奖励密集且对齐。在紧密耦合任务上失败（例如，一个代理必须*等待*另一个）。

### 步骤 3：带有分解值更新的集中 Q

使用联合动作上的一个 Q `Q(s, a_1, a_2)`。从共享奖励更新。在执行时通过边际化分散：`π_i(s) = argmax_{a_i} max_{a_{-i}} Q(s, a_1, a_2)`。用指数联合动作空间交换*正确*的全局视图。

### 步骤 4：简单自我对弈（对抗 2 代理）

相同代理，两个角色。训练代理 A 对抗代理 B；`K` 回合后，复制 A 的权重到 B。对称训练，一致的进步。AlphaZero 配方的缩影。

## 缺陷陷阱

- **非平稳重放。** 与独立代理的经验重放比单代理更差，因为旧转移是由现在过时的对手生成的。修复：重新标记或按新近度加权。
- **信用分配歧义。** 长回合后的共享奖励；没有清晰的方法说哪个代理贡献了。修复：反事实基线（COMA），或每代理奖励塑造。
- **策略漂移/追逐。** 每个代理的最优响应随着对方的每次更新而改变。修复：集中评论家，慢速学习率，或一次冻结一个。
- **通过协调的奖励黑客。** 代理找到设计者未预料的协调利用。拍卖代理收敛到零投标。修复：仔细的奖励设计，行为约束。
- **探索冗余。** 两个代理探索相同的状态-动作对。修复：每代理熵奖励，或角色条件化。
- **联赛循环。** 纯自我对弈可能陷入统治循环。修复：具有多样化对手的联赛游戏。
- **样本爆炸。** `n` 代理 × 状态空间 × 联合动作。用函数近似近似；因式分解的动作空间（每代理一个策略输出头）。

## 使用它

2026 年 MARL 应用地图：

| 领域 | 方法 | 备注 |
|--------|--------|-------|
| 合作导航 / 操作 | MAPPO / QMIX | CTDE；共享评论家 + 分散演员。 |
| 双人游戏（象棋、围棋、扑克） | 带有 MCTS 的自我对弈（AlphaZero） | 零和；对称训练。 |
| 复杂多人（Dota、星际争霸） | 联赛游戏 + 模仿预训练 | OpenAI Five、AlphaStar。 |
| 自动驾驶车队 | 带有注意力的 CTDE MAPPO / PPO | 部分观测；可变团队规模。 |
| 拍卖市场 | 博弈论均衡 + RL | 当 `n` → ∞ 时的平均场 RL。 |
| LLM 多代理系统（第十六阶段） | 自然语言通信 + 角色条件化 | 在代理规划层的 RL 循环。 |

在 2026 年，MARL 最大的增长领域是基于 LLM 的：语言模型代理的群体协商、辩论、构建软件。RL 在*轨迹级*输出上显示为偏好优化，而非 token 级（第十六阶段 · 03）。

## 交付成果

保存为 `outputs/skill-marl-architect.md`：

```markdown
---
name: marl-architect
description: 为给定任务选择正确的多代理 RL 体制（IPPO、CTDE、自我对弈、联赛）。
version: 1.0.0
phase: 9
lesson: 10
tags: [rl, multi-agent, marl, self-play]
---

给定具有 `n` 个代理的任务，输出：

1. 体制分类。合作 / 对抗 / 一般和。证明。
2. 算法。IPPO / MAPPO / QMIX / 自我对弈 / 联赛。与耦合紧密性和奖励结构相关的原因。
3. 信息访问。集中训练（什么全局信息进入评论家）？分散执行？
4. 信用分配。反事实基线、值分解或奖励塑造。
5. 探索计划。每代理熵、基于种群的训练或联赛。

拒绝在紧密耦合合作任务上使用独立 Q-learning。拒绝为有循环风险的一般和推荐自我对弈。标记任何无固定对手评估的 MARL 流水线（精心挑选的自我对弈数字很常见）。
```

## 练习

1. **简单。** 在 2 代理合作 GridWorld 上训练独立 Q-learning。多少回合后均值回报 > 0？绘制联合学习曲线。
2. **中等。** 添加"协调"任务：只有当两个代理在同一回合同时踏入目标时才到达。独立 Q 仍然收敛吗？什么被破坏了？
3. **困难。** 为 MAPPO 风格训练实现集中评论家并比较在协调任务上与独立 PPO 的收敛速度。

## 关键术语

| 术语 | 人们怎么说 | 它实际上的含义 |
|------|-----------------|-----------------------|
| 马尔可夫博弈 | "多代理 MDP" | `(S, A_1, …, A_n, P, R_1, …, R_n)`；每个代理有自己的奖励。 |
| CTDE | "集中训练、分散执行" | 训练时联合评论家；每个代理的策略仅使用本地观测。 |
| IPPO | "独立 PPO" | 每个代理分别运行 PPO。简单基线；经常被低估。 |
| MAPPO | "多代理 PPO" | 以全局状态为条件的带有集中值函数的 PPO。 |
| QMIX | "单调值分解" | `Q_tot = f_monotone(Q_1, …, Q_n)` 允许分散 argmax。 |
| COMA | "反事实多代理" | 优势 = 我的 Q 减去边际化我动作的期望 Q。 |
| 自我对弈 | "代理 vs 过去的自己" | 单代理、两个角色；零和博弈的标准做法。 |
| 联赛游戏 | "种群训练" | 缓存过去策略，从池中采样对手；处理策略循环。 |

## 延伸阅读

- [Lowe 等人 (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG)](https://arxiv.org/abs/1706.02275) — 带有集中评论家的 CTDE。
- [Foerster 等人 (2017). Counterfactual Multi-Agent Policy Gradients (COMA)](https://arxiv.org/abs/1705.08926) — 信用分配的反事实基线。
- [Rashid 等人 (2018). QMIX: Monotonic Value Function Factorisation](https://arxiv.org/abs/1803.11485) — 具有单调性的值分解。
- [Yu 等人 (2022). The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games (MAPPO)](https://arxiv.org/abs/2103.01955) — PPO 对 MARL 出乎意料地强。
- [Vinyals 等人 (2019). Grandmaster level in StarCraft II using multi-agent reinforcement learning (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z) — 大规模联赛游戏。
- [Silver 等人 (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270) — 零和博弈中的纯自我对弈。
- [Sutton 和 Barto (2018). 第 15 章 — Neuroscience & 第 17 章 — Frontiers](http://incompleteideas.net/book/RLbook2020.pdf) — 包括教科书对多代理设置和 CTDE 旨在解决的非平稳性问题的简短处理。
- [Zhang、Yang 和 Başar (2021). Multi-Agent Reinforcement Learning: A Selective Overview](https://arxiv.org/abs/1911.10635) — 涵盖合作、竞争和混合 MARL 的综述，附收敛结果。
