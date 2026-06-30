# Tree of Thoughts 与 LATS：审慎搜索

> 单条 Chain-of-Thought 轨迹没有回退的空间。ToT（Yao et al., 2023）将推理变成一棵在每个节点上进行自我评估的树。LATS（Zhou et al., 2024）在蒙特卡洛树搜索下统一了 ToT、ReAct 和 Reflexion。Game of 24 从 4%（CoT）提升到 74%（ToT）；LATS 在 HumanEval 上达到 92.7% pass@1。

**类型:** Build
**语言:** Python（标准库）
**前置要求:** Phase 14 · 01（Agent Loop）, Phase 14 · 03（Reflexion）
**时间:** ~75 分钟

## 学习目标

- 将推理框架为搜索：节点是"思考"，边是"扩展"，价值是"有多大前景"。
- 实现一个标准库 ToT 风格的 BFS 树搜索，带有自我评估评分。
- 扩展到一个玩具 LATS MCTS 循环，包含 select / expand / simulate / backpropagate。
- 决定何时搜索值得 token 乘数（Game of 24、代码生成），何时单条轨迹足够（简单问答）。

## 问题

Chain-of-Thought 是一次线性行走。如果第一步就错了，每个后续步骤都在一个坏前提上工作。在 Game of 24（使用四个数字通过 + − × ÷ 得到 24）上，GPT-4 CoT 达到 4% 的准确率。模型早期选择了错误的子表达式，无法恢复。

推理所需要的是提出多个候选方案、评估它们、选择有前景的、并在出现死胡同时回退的能力。这就是搜索。Tree of Thoughts 和 LATS 是两个经典公式化表述。

## 概念

### Tree of Thoughts（Yao et al., NeurIPS 2023）

每个节点是一个连贯的中间步骤（"一个思考"）。每个节点可以扩展到 K 个子思考。LLM 通过评分提示自我评估每个节点。搜索探索这棵树——BFS、DFS 或束搜索。

```
                     (root: "从 4 6 4 1 得到 24")
                    /               |            \
           ("6 - 4 = 2")    ("4 + 1 = 5")    ("4 * 6 = 24")  <- 评分: 高
              /   \              |                  |
          ...    ...          ...                完成
```

自我评估是承重的部分。论文展示了三种变体：`sure / likely / impossible` 分类、`1..10` 数值评分、以及在候选之间投票。三种都在 Game of 24 上大幅超越 CoT（GPT-4 从 4% 到 74%）。

### LATS（Zhou et al., ICML 2024）

LATS 在 MCTS 下统一了 ToT、ReAct 和 Reflexion。LLM 扮演三个角色：

- **策略（Policy）**: 提出候选的下一步行动（ReAct 风格）。
- **价值函数（Value function）**: 对部分轨迹打分（ToT 风格自我评估）。
- **自我反思器（Self-reflector）**: 失败时，写一段自然语言反思（Reflexion 风格）并用它来为未来的 rollout 重新播种。

环境反馈（观察）混入价值函数中，使搜索受到真实工具结果而非仅模型意见的影响。论文时期的结果：HumanEval pass@1 使用 GPT-4 达到 92.7%（SOTA），WebShop 使用 GPT-3.5 平均分 75.9（接近基于梯度的微调）。

### MCTS，最简形式

每次迭代四个阶段：

1. **Select（选择）** — 使用 UCT（树的上置信界）从根走到叶子。
2. **Expand（扩展）** — 通过策略生成 K 个子节点。
3. **Simulate（模拟）** — 使用策略从子节点进行 rollout，用价值函数（或环境奖励）对叶子评分。
4. **Backpropagate（回传）** — 沿路径更新访问次数和价值估计。

UCT 公式: `Q(s, a) + c * sqrt(ln N(s) / N(s, a))`。第一项是利用；第二项是探索。按任务调优 `c`。

### 成本现实

搜索会爆炸 token。ToT 在 Game of 24 上使用 CoT 的 100–1000 倍的 token。LATS 类似。这并非免费；保留搜索用于：

- 单条轨迹明显不够的任务（Game of 24、复杂代码）。
- 墙钟时间不如正确性重要的任务。
- 具有廉价、可靠价值函数的任务（代码的单元测试、数学的明确目标）。

如果你的任务有单一正确答案但评估器有噪声，搜索常常使情况更差——它找到一个"评分高"的错误答案。

### 2026 年的定位

大多数生产级 agent 不运行 LATS。它们运行带有工具锚定验证的 ReAct（CRITIC，第 05 课）。搜索出现在专业领域：

- 编码 agent 运行测试作为价值函数（HumanEval 风格）。
- 深度研究 agent 探索多个查询路径。
- LangGraph 子图中规划密集的工作流。

AlphaEvolve（第 11 课）是 2025 年的极端：对代码的进化搜索，机器可检查的适应度，前沿收益（56 年来首次 4x4 矩阵乘法改进）。

## Build It

`code/main.py` 实现：

- 一个微型 ToT BFS，在风格化的"选择算术操作"任务上。
- 在同一任务上的玩具 LATS MCTS 循环（Select / Expand / Simulate / Backpropagate），使用 UCT 选择。
- 一个结合了符号分数和自我评估分数的价值函数。

运行它：

```
python3 code/main.py
```

追踪显示 ToT 在每个节点上通过 BFS 扩展三个候选，对比 LATS 通过 MCTS 收敛到最佳 rollout。两者都打印 token 计数。

## Use It

LangGraph 将 ToT 风格探索作为子图模式提供；LangChain 团队关于 LATS 的博客（2024 年 5 月）是参考教程。LlamaIndex 提供一个 `TreeOfThoughts` agent。对于大多数 2026 年的生产级 agent，此模式存在于 `if task_complexity > threshold: use_search()` 门控后面——参见第 05 课的 evaluator-optimizer 模式。

## Ship It

`outputs/skill-search-policy.md` 根据任务形状、预算和评估器保真度在 ReAct 线性、ToT、LATS 和进化搜索之间进行选择。

## 练习

1. 用 UCT c=0.1 对比 c=2.0 运行玩具 LATS。追踪中有什么变化？
2. 将价值函数替换为噪声更大的评分器（添加随机抖动）。MCTS 是否仍能找到最佳叶子？它能容忍的最小信噪比是多少？
3. 实现束搜索 ToT（每层保留 top-k）并与 BFS 比较。在紧凑的 token 预算上哪个更好？
4. 阅读 LATS 第 5.1 节。复现 HumanEval 轨迹数量：需要多少次 rollout 才能达到报告的 pass@1？
5. 阅读 LATS 论文关于"LATS 在什么情况下帮助较小"的讨论。写一段将任务形状映射到搜索策略的决策规则。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| Tree of Thoughts | "分支 CoT" | Yao et al. — 带有自我评估的思考节点树 |
| LATS | "用于 LLM 的 MCTS" | Zhou et al. — 在 MCTS 下统一 ToT + ReAct + Reflexion |
| UCT | "上置信界" | 平衡利用（Q）和探索（ln N / n）的选择公式 |
| 价值函数 | "这个状态有多好" | 提示的 LLM 评分或环境奖励；反馈给回传 |
| 策略 | "动作提议者" | ReAct 风格生成器；发出候选的下一个思考/动作 |
| Rollout | "模拟轨迹" | 使用策略从节点走到叶子，用价值函数评分 |
| 回传 | "更新祖先" | 将叶子的奖励推送到路径上，更新访问次数和 Q |
| 搜索成本 | "Token 爆炸" | 在 Game of 24 上是 CoT 的 100-1000 倍；在采用前做好预算 |

## 进一步阅读

- [Yao et al., Tree of Thoughts (arXiv:2305.10601)](https://arxiv.org/abs/2305.10601) — 经典论文
- [Zhou et al., LATS (arXiv:2310.04406)](https://arxiv.org/abs/2310.04406) — 带有 Reflexion 反馈的 MCTS
- [LangGraph 概览](https://docs.langchain.com/oss/python/langgraph/overview) — 用于搜索的子图模式
- [AlphaEvolve (arXiv:2506.13131)](https://arxiv.org/abs/2506.13131) — 带有程序化评估器的进化搜索
