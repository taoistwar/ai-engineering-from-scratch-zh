# RL 用于游戏 — AlphaZero、MuZero 和 LLM 推理时代

> 1992：TD-Gammon 用纯 TD 在西洋双陆棋上击败人类冠军。2016：AlphaGo 击败李世石。2017：AlphaZero 从头开始统治国际象棋、将棋和围棋。2024：DeepSeek-R1 证明相同的配方，用 GRPO 取代 PPO，在推理上有效。游戏是驱动本阶段每个突破的基准。

**类型：** 构建
**语言：** Python
**前置条件：** 第九阶段 · 05（DQN），第九阶段 · 08（PPO），第九阶段 · 09（RLHF），第九阶段 · 10（MARL）
**时间：** 约 120 分钟

## 问题

游戏具有 RL 想要的一切。干净的奖励（赢/输）。无限回合（自我对弈重置）。完美的模拟（游戏*是*模拟器）。离散或小型连续动作空间。强制对抗鲁棒性的多代理结构。

并且游戏是每个重大 RL 突破被测试的方式。TD-Gammon（西洋双陆棋，1992）。Atari-DQN（2013）。AlphaGo（2016）。AlphaZero（2017）。OpenAI Five（Dota 2，2019）。AlphaStar（星际争霸 II，2019）。MuZero（学习的模型，2019）。AlphaTensor（矩阵乘法，2022）。AlphaDev（排序算法，2023）。DeepSeek-R1（数学推理，2025）——游戏 RL 技术在文本上有效的最新演示。

本毕业项目通过一个统一的镜头——**自我对弈 + 搜索 + 策略改进**——调查三个标志性架构：AlphaZero、MuZero 和 GRPO。每个推广前一个；GRPO 尤是应用于 LLM 推理的 AlphaZero 配方，以 token 作为动作，数学验证作为胜利信号。

## 概念

![AlphaZero ↔ MuZero ↔ GRPO：相同循环，不同环境](../assets/rl-games.svg)

**统一循环。**

```
while True:
    trajectory = self_play(current_policy, search)     # 与自己对弈
    policy_target = search.improved_policy(trajectory) # 搜索改进原始策略
    policy_net.update(policy_target, value_target)     # 在搜索输出上监督
```

**AlphaZero（2017）。** Silver 等人。给定一个具有已知规则的游戏（国际象棋、将棋、围棋）：

- 策略-价值网络：一个塔 `f_θ(s) → (p, v)`。`p` 是合法走法的先验。`v` 是预期对局结果。
- 蒙特卡洛树搜索（MCTS）：每步走法，展开可能续写的树。使用 `(p, v)` 作为先验 + 自举。通过 UCB（PUCT）选择节点：`a* = argmax Q(s, a) + c · p(a|s) · √N(s) / (1 + N(s, a))`。
- 自我对弈：进行代理 vs 代理对局。在第 `t` 步走法，MCTS 访问分布 `π_t` 成为策略训练目标。
- 损失：`L = (v - z)² - π · log p + c · ||θ||²`。`z` 是对局结果（+1 / 0 / -1）。

零人类知识。零手工启发式。单一配方在数千万局自我对弈后精通国际象棋、将棋和围棋。

**MuZero（2019）。** Schrittwieser 等人。移除规则已知的要求。

- 代替固定环境，学习*潜在动力学模型* `(h, g, f)`：
  - `h(s)`：将观测编码为潜在状态。
  - `g(s_latent, a)`：预测下一潜在状态 + 奖励。
  - `f(s_latent)`：预测策略先验 + 价值。
- MCTS 在*学习的潜在空间*中运行。相同搜索，相同训练循环。
- 在围棋、国际象棋、将棋*和* Atari 上工作——一个算法，无规则知识。

**随机 MuZero（2022）。** 添加随机动力学和机会节点；扩展到西洋双陆棋类游戏。

**Muesli、Gumbel MuZero（2022-2024）。** 样本效率和确定性搜索上的改进。

**GRPO（2024-2025）。** DeepSeek-R1 配方。相同的 AlphaZero 形状循环，应用于语言模型推理：

- "游戏"：回答数学 / 编码 / 推理问题。"赢" = 验证器（测试用例通过，数值答案匹配）返回 1。
- 策略：LLM。动作：token。状态：提示 + 迄今为止的响应。
- 无评论家（PPO 风格 V_φ）。而是，对每个提示，从策略中采样 `G` 个补全。为每个计算奖励。使用**群相对优势** `A_i = (r_i - mean_r) / std_r` 作为 REINFORCE 风格更新的信号。
- 对参考策略的 KL 惩罚以防止漂移（像 RLHF）。
- 完整损失：

  `L_GRPO(θ) = -E_{q, {o_i}} [ (1/G) Σ_i A_i · log π_θ(o_i | q) ] + β · KL(π_θ || π_ref)`

无奖励模型，无评论家，无 MCTS。群相对基线替换所有三个。在平衡 RLHF 质量下，在推理基准上匹配或超过 PPO-RLHF，计算量仅为其一小部分。

**完整的 R1 配方。** DeepSeek-R1（DeepSeek 2025）是一篇论文中的两个模型：

- **R1-Zero。** 从 DeepSeek-V3 基座模型开始。无 SFT。直接用两个奖励成分应用 GRPO：*准确率奖励*（基于规则——最终答案是否解析为正确数字/代码是否通过单元测试）和*格式奖励*（补全是否在 `<think>…</think>` 标签中包装其思维链）。经过数千步，平均响应长度从约 100 增长到约 10,000 token，数学基准分数攀升至接近 o1-preview 水平。模型从零开始学习推理。缺点：其思维链通常不可读，混合语言且缺乏风格润色。
- **R1。** 用四阶段流水线修复 R1-Zero 的可读性问题：
  1. **冷启动 SFT。** 收集几千个具有干净格式的长思维链演示。在其上监督微调基座模型。这给出可读的起点。
  2. **面向推理的 GRPO。** 应用 GRPO，带有准确率+格式奖励加上*语言一致性*奖励以防止代码切换。
  3. **拒绝采样 + SFT 第 2 轮。** 从 RL 检查点采样约 600K 推理轨迹，仅保留具有正确最终答案和可读思维链的，并结合约 200K 非推理 SFT 示例（写作、QA、自我认知）。再次微调基座。
  4. **全频谱 GRPO。** 涵盖推理（基于规则的奖励）和一般对齐（有帮助性/无害性基于偏好的奖励）的最后一轮 RL。

结果以开放权重在 AIME 和 MATH-500 上匹敌 o1，且足够小可蒸馏。同一篇论文还通过在 R1 的推理轨迹上进行 SFT 发布了六个蒸馏密集模型（Qwen-1.5B 到 Llama-70B）——学生端不做 RL。强 RL 教师的蒸馏在学生规模上始终击败从头 RL。

**为什么 GRPO 而非 PPO 用于推理。** DeepSeekMath 论文（2024 年 2 月）中的三个原因：(1) 无需训练价值网络，内存减半；(2) 群基线自然地处理推理任务产生的稀疏轨迹末端奖励；(3) 每提示归一化使优势在难度差异极大的问题上具有可比性，这是 PPO 的单一评论家无法做到的。

**无搜索 vs 基于搜索。** 游戏已分流：

- *具有长视野的完美信息游戏*（围棋、国际象棋）：仍然基于搜索。AlphaZero / MuZero 占主导。
- *LLM 推理*：生产中尚无 MCTS；完整展开上的 GRPO，best-of-N 用于推理计算。过程奖励模型（PRM）暗示步级搜索正被添加回来。

## 动手构建

`code/main.py` 中的代码实现**微型 GRPO**——具有多组样本的老虎机。算法与在 LLM 上相同；只有策略和环境更简单。它教授*损失*和*群相对优势*，即 2025 年的创新。

### 步骤 1：微型验证器环境

```python
QUESTIONS = [
    {"prompt": "q1", "correct": 3},
    {"prompt": "q2", "correct": 1},
]

def verify(prompt_idx, answer_token):
    return 1.0 if answer_token == QUESTIONS[prompt_idx]["correct"] else 0.0
```

在真实 GRPO 中验证器运行单元测试或检查数学等式。

### 步骤 2：策略：每个提示 K 个答案 token 上的 softmax

```python
def policy_probs(theta, p_idx):
    return softmax(theta[p_idx])
```

等效于以提示为条件的 LLM 的最终层输出。

### 步骤 3：群采样和群相对优势

```python
def grpo_step(theta, p_idx, G=8, beta=0.01, lr=0.1, rng=None):
    probs = policy_probs(theta, p_idx)
    samples = [sample(probs, rng) for _ in range(G)]
    rewards = [verify(p_idx, s) for s in samples]
    mean_r = sum(rewards) / G
    std_r = stddev(rewards) + 1e-8
    advs = [(r - mean_r) / std_r for r in rewards]

    for a, A in zip(samples, advs):
        grad = onehot(a) - probs
        for i in range(len(probs)):
            theta[p_idx][i] += lr * A * grad[i]
    # KL 惩罚：将 theta 拉向参考
    for i in range(len(probs)):
        theta[p_idx][i] -= beta * (theta[p_idx][i] - reference[p_idx][i])
```

群相对优势是 2024 年 DeepSeek 技巧。不需要评论家。"基线"是群均值，归一化使用群标准差。

### 步骤 4：与 REINFORCE 基线（无价值）比较

相同设置、相同计算量、普通 REINFORCE。GRPO 收敛更快更稳定。

### 步骤 5：观察熵和 KL

与 RLHF 相同的诊断：对参考的均值 KL、策略熵、奖励随时间变化。一旦这些稳定，训练完成。

## 缺陷陷阱

- **通过验证器博弈的奖励黑客。** GRPO 继承 RLHF 的风险：如果验证器错误或可被利用，LLM 会找到利用方式。鲁棒的验证器（多个测试用例、形式化证明）很重要。
- **群太小。** 群基线的方差随 `1/√G` 变化。低于 `G = 4`，优势信号嘈杂；标准选择是 `G = 8` 到 `64`。
- **长度偏置。** 不同长度的 LLM 补全有不同的对数概率。按 token 计数归一化，或使用序列级对数概率，或截断到最大长度。
- **纯自我对弈循环。** AlphaZero 风格训练可能陷入一般和博弈的统治循环。通过多样化对手池（联赛游戏，第 10 课）缓解。
- **搜索-策略不匹配。** AlphaZero 训练策略模仿搜索输出。如果策略网络太小无法表示搜索的分布，训练停滞。
- **计算门槛。** MuZero / AlphaZero 需要海量计算。单次消融通常数百 GPU 小时。存在用于学习的微型演示（例如 Connect Four 上的 AlphaZero）。
- **验证器覆盖。** 通过单元测试的错误解决方案强化缺陷。设计捕获边缘情况的验证器。

## 使用它

2026 年游戏 RL 版图，按领域：

| 领域 | 主导方法 |
|--------|-----------------|
| 双人零和棋盘游戏（围棋、国际象棋、将棋） | AlphaZero / MuZero / KataGo |
| 不完全信息纸牌游戏（扑克） | CFR + 深度学习（DeepStack、Libratus、Pluribus） |
| Atari / 像素游戏 | Muesli / MuZero / IMPALA-PPO |
| 大型多人策略（Dota、星际争霸） | PPO + 自我对弈 + 联赛（OpenAI Five、AlphaStar） |
| LLM 数学/代码推理 | GRPO（DeepSeek-R1、Qwen-RL、开源复现） |
| LLM 对齐 | DPO / RLHF-PPO（非 GRPO；验证器是偏好而非可验证） |
| 机器人学 | PPO + DR（非游戏 RL，但使用相同的策略梯度工具） |
| 组合问题 | AlphaZero 变体（AlphaTensor、AlphaDev） |

*配方*——自我对弈、搜索增强改进、策略蒸馏——跨越文本、像素和物理控制。GRPO 是最年轻的实例；更多即将到来。

## 交付成果

保存为 `outputs/skill-game-rl-designer.md`：

```markdown
---
name: game-rl-designer
description: 为给定领域设计游戏-RL 或推理-RL 训练流水线（AlphaZero / MuZero / GRPO）。
version: 1.0.0
phase: 9
lesson: 12
tags: [rl, alphazero, muzero, grpo, self-play]
---

给定目标（完美信息博弈 / 不完全信息 / Atari / LLM 推理 / 组合），输出：

1. 环境匹配。已知规则？马尔可夫？随机？多代理？告知 AlphaZero vs MuZero vs GRPO。
2. 搜索策略。MCTS（带有学习先验的 PUCT）、Gumbel 采样、best-of-N 或无。
3. 自我对弈计划。对称自我对弈 / 联赛 / 离线数据 / 验证器生成。
4. 目标信号。对局结果 / 验证器奖励 / 偏好 / 学习到的模型。包括鲁棒性计划。
5. 诊断。对基线的胜率、ELO 曲线、验证器通过率、KL 到参考。

拒绝在不完全信息博弈上使用 AlphaZero（路由到 CFR）。拒绝无受信任验证器的 GRPO。拒绝任何无固定基线对手集的游戏 RL 流水线（否则自我对弈 ELO 是无校准的）。
```

## 练习

1. **简单。** 在 `code/main.py` 中实现 GRPO 老虎机。在 2 个提示 × 每个 4 个答案 token 上训练。在 < 1,000 次更新内以 `G=8` 收敛。
2. **中等。** 插入 PPO（裁剪）和标准 REINFORCE。在相同老虎机上比较样本效率和奖励方差与 GRPO。
3. **困难。** 扩展到 length-2 "推理链"：代理发出两个 token，验证器奖励这对。测量 GRPO 如何处理两步序列上的信用分配。（提示：计算每*完整序列*的群优势，传播到两个 token 位置。）

## 关键术语

| 术语 | 人们怎么说 | 它实际上的含义 |
|------|-----------------|-----------------------|
| MCTS | "带学习网络的树搜索" | 蒙特卡洛树搜索；带有学习 `(p, v)` 先验的 UCB1/PUCT 选择。 |
| AlphaZero | "自我对弈 + MCTS" | 训练匹配 MCTS 访问和对局结果的策略-价值网络。 |
| MuZero | "带学习模型的 AlphaZero" | 相同循环但在潜在空间中通过学习的动力学。 |
| GRPO | "无评论家 PPO" | 群相对策略优化；REINFORCE 带群均值基线 + KL。 |
| PUCT | "AlphaZero 的 UCB" | `Q + c · p · √N / (1 + N_a)`——平衡价值估计和先验。 |
| 自我对弈 | "代理 vs 过去的自己" | 零和的标准做法；对称训练信号。 |
| 联赛游戏 | "基于种群的自我对弈" | 过去 + 当前 + 利用者采样作为对手。 |
| 验证器奖励 | "可验证 RL" | 奖励来自确定性检查器（测试通过、答案匹配）。 |
| 过程奖励 | "PRM" | 评分每个推理步骤，而非仅最终答案。 |

## 延伸阅读

- [Silver 等人 (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270).
- [Silver 等人 (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play (AlphaZero)](https://www.science.org/doi/10.1126/science.aar6404).
- [Schrittwieser 等人 (2020). Mastering Atari, Go, chess and shogi by planning with a learned model (MuZero)](https://www.nature.com/articles/s41586-020-03051-4).
- [Vinyals 等人 (2019). Grandmaster level in StarCraft II (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z).
- [DeepSeek-AI (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO)](https://arxiv.org/abs/2402.03300) — 引入 GRPO 和群相对基线的论文。
- [DeepSeek-AI (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) — 完整的四阶段 R1 配方加上 R1-Zero 消融。
- [Brown 等人 (2019). Superhuman AI for multiplayer poker (Pluribus)](https://www.science.org/doi/10.1126/science.aay2400) — 大规模 CFR + 深度学习。
- [Tesauro (1995). Temporal Difference Learning and TD-Gammon](https://dl.acm.org/doi/10.1145/203330.203343) — 开启一切的论文。
- [Hugging Face TRL — GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer) — 以自定义奖励函数应用 GRPO 的生产参考。
- [Qwen Team (2024). Qwen2.5-Math — GRPO replication](https://github.com/QwenLM/Qwen2.5-Math) — R1 配方的多规模开源复现。
- [Sutton 和 Barto (2018). 第 17 章 — Frontiers of Reinforcement Learning](http://incompleteideas.net/book/RLbook2020.pdf) — R1 在 LLM 规模上实例化的自我对弈、搜索和"设计的奖励"的教科书框架。
