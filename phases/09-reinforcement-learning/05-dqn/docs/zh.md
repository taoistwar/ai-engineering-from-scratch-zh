# 深度 Q 网络（DQN）

> 2013：Mnih 在原始像素上训练一个 Q-learning 网络，在七个 Atari 游戏中击败每个经典 RL 代理。2015：扩展到 49 个游戏，在 Nature 上发表，点燃了深度 RL 时代。DQN 是 Q-learning 加上三个使函数近似稳定的技巧。

**类型：** 构建
**语言：** Python
**前置条件：** 第三阶段 · 03（反向传播），第九阶段 · 04（Q-learning、SARSA）
**时间：** 约 75 分钟

## 问题

表格 Q-learning 需要为每个（状态，动作）对提供单独的 Q 值。国际象棋棋盘约有 10⁴³ 个状态。Atari 帧是 210×160×3 = 100,800 个特征。表格 RL 在数千个状态时就死了，更不用说数十亿。

事后看来修复是显而易见的：用神经网络 `Q(s, a; θ)` 替换 Q 表。但事后显而易见花了几十年。朴素的带函数近似的 Q-learning 在"致命三元组"下发散——函数近似 + 自举 + 离策略学习。Mnih 等人（2013、2015）识别了三个稳定学习的工程技巧：

1. **经验重放** 去相关转移。
2. **目标网络** 冻结自举目标。
3. **奖励裁剪** 归一化梯度幅度。

Atari 上的 DQN 是第一次单个架构用单个超参数集从原始像素中解决了数十个控制问题。此后构建的一切"深度 RL"——DDQN、Rainbow、Dueling、Distributional、R2D2、Agent57——都堆叠在这个三技巧基础上。

## 概念

![DQN 训练循环：env、重放缓冲区、在线网络、目标网络、Bellman TD 损失](../assets/dqn.svg)

**目标。** DQN 在神经 Q 函数上最小化一步 TD 损失：

`L(θ) = E_{(s,a,r,s')~D} [ (r + γ max_{a'} Q(s', a'; θ^-) - Q(s, a; θ))² ]`

`θ` = 在线网络，每步通过梯度下降更新。`θ^-` = 目标网络，周期性地从 `θ` 复制（每约 10,000 步）。`D` = 过去转移的重放缓冲区。

**三个技巧，按重要性排序：**

**经验重放。** 约 10⁶ 个转移的环形缓冲区。每个训练步以均匀随机采样一个小批量。这打破了时间相关性（连续帧几乎相同），让网络从稀有奖励转移中学习多次，并去相关连续的梯度更新。没有它，带有神经常规的在线策略 TD 在 Atari 上发散。

**目标网络。** 在 Bellman 方程的两边使用相同的网络 `Q(·; θ)` 使目标在每次更新时都移动——"追逐自己的尾巴"。修复：保留第二个网络 `Q(·; θ^-)` 带有冻结权重。每 `C` 步，复制 `θ → θ^-`。这稳定了回归目标数千个梯度步。软更新 `θ^- ← τ θ + (1-τ) θ^-`（在 DDPG、SAC 中使用）是一个更平滑的变体。

**奖励裁剪。** Atari 奖励幅度从 1 变化到 1000+。裁剪到 `{-1, 0, +1}` 阻止任何单个游戏主导梯度。当奖励幅度重要时是错的；对 Atari 来说只有符号重要就够了。

**双 DQN。** Hasselt（2016）修复最大化偏置：使用在线网络*选择*动作，目标网络来*评估*它。

`target = r + γ Q(s', argmax_{a'} Q(s', a'; θ); θ^-)`

即插即用替换，始终更好。默认使用它。

**其他改进（Rainbow，2017）：** 优先重放（更频繁采样高 TD 误差转移）、决斗架构（分离 `V(s)` 和优势头）、噪声网络（学习到的探索）、n 步回报、分布 Q（C51/QR-DQN）、多步自举。每个添加几个百分点；收益大致累加。

## 动手构建

这里的代码是仅含标准库、无 numpy 的——我们在微型连续 GridWorld 上使用手动单隐藏层 MLP，所以每个训练步以微秒运行。算法与规模化的 Atari DQN 相同。

### 步骤 1：重放缓冲区

```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = []
        self.capacity = capacity
    def push(self, s, a, r, s_next, done):
        if len(self.buf) == self.capacity:
            self.buf.pop(0)
        self.buf.append((s, a, r, s_next, done))
    def sample(self, batch, rng):
        return rng.sample(self.buf, batch)
```

Atari 约 50,000 容量；5,000 对我们的玩具环境足够。

### 步骤 2：微型 Q 网络（手动 MLP）

```python
class QNet:
    def __init__(self, n_in, n_hidden, n_actions, rng):
        self.W1 = [[rng.gauss(0, 0.3) for _ in range(n_in)] for _ in range(n_hidden)]
        self.b1 = [0.0] * n_hidden
        self.W2 = [[rng.gauss(0, 0.3) for _ in range(n_hidden)] for _ in range(n_actions)]
        self.b2 = [0.0] * n_actions
    def forward(self, x):
        h = [max(0.0, sum(w * xi for w, xi in zip(row, x)) + b) for row, b in zip(self.W1, self.b1)]
        q = [sum(w * hi for w, hi in zip(row, h)) + b for row, b in zip(self.W2, self.b2)]
        return q, h
```

前向传播：线性 → ReLU → 线性。这就是整个网络。

### 步骤 3：DQN 更新

```python
def train_step(online, target, batch, gamma, lr):
    grads = zeros_like(online)
    for s, a, r, s_next, done in batch:
        q, h = online.forward(s)
        if done:
            y = r
        else:
            q_next, _ = target.forward(s_next)
            y = r + gamma * max(q_next)
        td_error = q[a] - y
        accumulate_grads(grads, online, s, h, a, td_error)
    apply_sgd(online, grads, lr / len(batch))
```

形状是第 04 课中的 Q-learning，有两个区别：(a) 我们通过可微的 `Q(·; θ)` 反向传播，而不是索引一个表，(b) 目标使用 `Q(·; θ^-)`。

### 步骤 4：外循环

对于每个回合，关于 `Q(·; θ)` ε-greedy 行动，将转移推入缓冲区，采样小批量，进行梯度步，周期性地同步 `θ^- ← θ`。模式：

```python
for episode in range(N):
    s = env.reset()
    while not done:
        a = epsilon_greedy(online, s, epsilon)
        s_next, r, done = env.step(s, a)
        buffer.push(s, a, r, s_next, done)
        if len(buffer) >= batch:
            train_step(online, target, buffer.sample(batch), gamma, lr)
        if steps % sync_every == 0:
            target = copy(online)
        s = s_next
```

在具有 16 维 one-hot 状态的微型 GridWorld 上，代理在约 500 回合内学会近最优策略。在 Atari 上，将其扩展到 2 亿帧并添加 CNN 特征提取器。

## 缺陷陷阱

- **致命三元组。** 函数近似 + 离策略 + 自举可以发散。DQN 用目标网络 + 重放缓解；不要移除任何一个。
- **探索。** ε 必须衰减，通常在前约 10% 训练中从 1.0 衰减到 0.01。没有足够的早期探索，Q 网络收敛到局部盆地。
- **高估。** 在嘈杂 Q 上的 `max` 是向上偏置的。在生产中始终使用双 DQN。
- **奖励尺度。** 裁剪或归一化奖励；梯度幅度与奖励幅度成正比。
- **重放缓冲区冷启动。** 在缓冲区有几千个转移之前不要训练。约 20 个样本上的早期梯度过拟合。
- **目标同步频率。** 太频繁 ≈ 无目标网络；太不频繁 ≈ 过时目标。Atari DQN 使用 10,000 环境步。经验法则：每约训练视野的 1/100 同步一次。
- **观测预处理。** Atari DQN 堆叠 4 帧使状态成为马尔可夫的。任何具有速度信息的环境都需要帧堆叠或递归状态。

## 使用它

2026 年，DQN 很少是 SOTA，但仍然是参考离策略算法：

| 任务 | 选择的方法 | 为什么不 DQN？ |
|------|------------------|--------------|
| 离散动作类 Atari | Rainbow DQN 或 Muesli | 相同框架，更多技巧。 |
| 连续控制 | SAC / TD3（第九阶段 · 07） | DQN 无策略网络。 |
| 在策略 / 高吞吐量 | PPO（第九阶段 · 08） | 无重放缓冲区；更易于扩展。 |
| 离线 RL | CQL / IQL / Decision Transformer | 保守 Q 目标，无自举爆炸。 |
| 大离散动作空间（推荐器） | 带动作嵌入的 DQN，或 IMPALA | 可以；装饰重要。 |
| LLM RL | PPO / GRPO | 序列级而非步级；不同的损失。 |

教训仍在传播。重放和目标网络出现在 SAC、TD3、DDPG、SAC-X、AlphaZero 的自我对弈缓冲区和每个离线 RL 方法中。奖励裁剪在 PPO 中作为优势归一化生存。架构是蓝图。

## 交付成果

保存为 `outputs/skill-dqn-trainer.md`：

```markdown
---
name: dqn-trainer
description: 为离散动作 RL 任务产生 DQN 训练配置（缓冲区、目标同步、ε 调度、奖励裁剪）。
version: 1.0.0
phase: 9
lesson: 5
tags: [rl, dqn, deep-rl]
---

给定离散动作环境（观测形状、动作计数、视野、奖励尺度），输出：

1. 网络。架构（MLP / CNN / Transformer）、特征维度、深度。
2. 重放缓冲区。容量、小批量大小、预热大小。
3. 目标网络。同步策略（每 C 步硬同步或软 τ）。
4. 探索。ε 起始 / 结束 / 调度长度。
5. 损失。Huber vs MSE、梯度裁剪值、奖励裁剪规则。
6. 双 DQN。默认开启，除非有明确理由禁用。

拒绝发布无目标网络、无重放缓冲区或 ε 保持 1 的 DQN。拒绝连续动作任务（路由到 SAC / TD3）。标记任何奖励范围 > 每步均值 10× 的需要裁剪或尺度归一化。
```

## 练习

1. **简单。** 运行 `code/main.py`。绘制每回合回报曲线。多少回合后运行均值超过 -10？
2. **中等。** 禁用目标网络（在 Bellman 目标的两边使用在线网络）。测量训练不稳定性——回报振荡还是发散？
3. **困难。** 添加双 DQN：使用在线网络选择 `argmax a'`，目标网络评估。在嘈杂奖励 GridWorld 上比较 1,000 回合后带与不带双 DQN 的 `Q(s_0, best_a)` 偏差 vs 真实 `V*(s_0)`。

## 关键术语

| 术语 | 人们怎么说 | 它实际上的含义 |
|------|-----------------|-----------------------|
| DQN | "深度 Q-learning" | 带有神经 Q 函数、重放缓冲区和目标网络的 Q-learning。 |
| 经验重放 | "洗牌后的转移" | 每个梯度步均匀采样的环形缓冲区；去相关数据。 |
| 目标网络 | "冻结自举" | 在 Bellman 目标中使用的 Q 的定期复制；稳定训练。 |
| 致命三元组 | "为什么 RL 发散" | 函数近似 + 自举 + 离策略 = 无收敛保证。 |
| 双 DQN | "最大化偏置的修复" | 在线网络选择动作，目标网络评估它。 |
| 决斗 DQN | "V 和 A 头" | 分解 Q = V + A - mean(A)；相同输出，更好的梯度流。 |
| Rainbow | "所有技巧" | DDQN + PER + dueling + n 步 + noisy + distributional 合一。 |
| PER | "优先重放" | 与 TD 误差大小成比例地采样转移。 |

## 延伸阅读

- [Mnih 等人 (2013). Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) — 引领了深度 RL 的 2013 年 NeurIPS 研讨会论文。
- [Mnih 等人 (2015). Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) — Nature 论文，49 游戏 DQN。
- [Hasselt、Guez、Silver (2016). Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) — DDQN。
- [Wang 等人 (2016). Dueling Network Architectures](https://arxiv.org/abs/1511.06581) — 决斗 DQN。
- [Hessel 等人 (2018). Rainbow: Combining Improvements in Deep RL](https://arxiv.org/abs/1710.02298) — 堆叠技巧论文。
- [OpenAI Spinning Up — DQN](https://spinningup.openai.com/en/latest/algorithms/dqn.html) — 清晰的现代阐述。
- [Sutton 和 Barto (2018). 第 9 章 — On-policy Prediction with Approximation](http://incompleteideas.net/book/RLbook2020.pdf) — DQN 的目标网络和重放缓冲区旨在驯服的"致命三元组"的教科书处理。
- [CleanRL DQN implementation](https://docs.cleanrl.dev/rl-algorithms/dqn/) — 在消融研究中使用的参考单文件 DQN；与本课从头开始的版本一起阅读很好。
