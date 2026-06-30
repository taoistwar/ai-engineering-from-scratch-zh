# Reflexion：语言强化学习

> 基于梯度的 RL 需要数千次试验和一个 GPU 集群来修复一个失败模式。Reflexion（Shinn et al., NeurIPS 2023）用自然语言做到：每次失败试验后，agent 写一段反思，存储在情景记忆中，并用该记忆调整下一次试验。这就是 Letta 的睡眠时计算、Claude Code 的 CLAUDE.md 学习以及 pro-workflow 的 learn-rule 背后的模式。

**类型:** Build
**语言:** Python（标准库）
**前置要求:** Phase 14 · 01（Agent Loop）, Phase 14 · 02（ReWOO）
**时间:** ~60 分钟

## 学习目标

- 说出 Reflexion 的三个组件（Actor、Evaluator、Self-Reflector）以及情景记忆的作用。
- 实现一个标准库 Reflexion 循环，包含二元评估器、反思缓冲区和全新的重试。
- 在给定任务中，在标量、启发式和自我评估反馈源之间做出选择。
- 解释为什么语言强化能捕获基于梯度的 RL 需要数千次试验才能修复的错误。

## 问题

一个 agent 执行任务失败。在标准 RL 中，你会运行数千次更多试验，计算梯度，更新权重。昂贵、缓慢，而且大多数生产级 agent 没有为每次失败准备训练预算。

Reflexion（Shinn et al., arXiv:2303.11366）提出一个不同的问题：如果 agent 只是想一想它为什么失败，然后带着那个思考在其提示中重试呢？没有权重更新。没有梯度。只是在试验之间存储的自然语言。

结果：在 ALFWorld 上它击败了 ReAct 和其他未微调的基线。在 HotpotQA 上它比 ReAct 有所改进。在代码生成（HumanEval/MBPP）上它在当时达到了最先进水平。所有这些都没有一个梯度步骤。

## 概念

### 三个组件

```
Actor         : 生成一个轨迹（ReAct 风格循环）
Evaluator     : 对轨迹打分——二元、启发式或自我评估
Self-Reflector: 对失败写一段自然语言反思
```

加上一个数据结构：

```
情景记忆: 先前反思的列表，被前置到下一次试验的提示中
```

一次试验运行 Actor。Evaluator 对其进行评分。如果分数低，Self-Reflector 产生一段反思（"我选错了工具，因为我误读问题是在问 X，而它实际在问 Y"）。反思进入情景记忆。下一次试验全新开始但看到该反思。

### 三种评估器类型

1. **标量（Scalar）** — 一个外部二元信号。ALFWorld 成功或失败。HumanEval 测试通过或失败。最简单、信号最强。
2. **启发式（Heuristic）** — 预定义的失败特征。"如果 agent 连续两次产生相同的动作，标记为卡住。""如果轨迹超过 50 步，标记为低效。"
3. **自我评估（Self-evaluated）** — LLM 对自己的轨迹打分。在没有 ground truth 时使用。信号较弱；与工具锚定验证配合良好（第 05 课——CRITIC）。

2026 年的默认做法是混合：有标量时用标量，没有时用自我评估，启发式作为安全护栏。

### 为什么这是通用的

Reflexion 与其说是一个新算法，不如说是一个命名的模式。几乎每个生产级的"自愈" agent 都运行某种变体：

- Letta 的睡眠时计算（第 08 课）：一个单独的 agent 反思过去的对话并写入内存块。
- Claude Code 的 `CLAUDE.md` / "保存记忆" 模式：反思作为学习被捕获，前置到未来的会话中。
- pro-workflow 的 `/learn-rule` 命令：修正作为显式规则被捕获。
- LangGraph 的反思节点：一个节点对输出评分，如果需改进则路由到精炼。

全部源自同一个洞见：自然语言是足够丰富的媒介，可以在运行之间传递"我从失败中学到了什么"。

### 何时有效，何时无效

Reflexion 在以下情况有效：

- 存在明确的失败信号（测试失败、工具错误、错误答案）。
- 任务类别是可复现的（同样类型的问题可以再次被提问）。
- 反思有改进轨迹的空间（足够的行动预算）。

Reflexion 在以下情况无效：

- agent 第一次尝试就成功了。
- 失败是外部原因（网络宕机、工具损坏）——对"网络宕机了"的反思不会帮助未来的运行。
- 反思变成迷信——存储关于一次性偶然运行的叙述。

2026 年的陷阱：记忆腐化。反思积累；有些过时或错误；随着情景缓冲区增长，重运行变得更慢。缓解：定期压缩（第 06 课）、反思的 TTL、或一个单独的睡眠时清理 agent（Letta）。

```figure
react-trace
```

## Build It

`code/main.py` 在一个玩具谜题上实现 Reflexion：生成一个和为目标的 3 元素列表。Actor 发出候选列表；Evaluator 检查和；Self-Reflector 写一行关于哪里出错的说明。反思进入情景记忆用于下一次试验。

组件：

- `Actor` — 一个在看到反思时会改进的脚本化策略。
- `Evaluator.binary()` — 在目标和上的通过/失败。
- `SelfReflector` — 生成对失败的一行诊断。
- `EpisodicMemory` — 一个带有 TTL 语义的有界列表。

运行它：

```
python3 code/main.py
```

追踪显示三次试验。试验 1 失败，存储反思，试验 2 看到反思并改进但仍失败，试验 3 成功。与基线运行比较（没有反思）——它会卡在试验 1 的答案上。

## Use It

LangGraph 将反思作为一个节点模式提供。Claude Code 的 `/memory` 命令和 pro-workflow 的 `/learn-rule` 将情景缓冲区外化为一个 markdown 文件。Letta 的睡眠时计算在空闲时运行 Self-Reflector，这样主 agent 保持低延迟。OpenAI Agents SDK 不直接提供 Reflexion；你通过一个自定义 Guardrail 拒绝按分数不达标的轨迹，以及一个跨运行持续的 memory `Session` 来构建它。

## Ship It

`outputs/skill-reflexion-buffer.md` 创建和维护一个带有反思捕获、TTL 和去重的情景缓冲区。给定一个任务类别和一个失败，它产出一个真正能帮助下一次试验的反思（而不是泛泛的"更小心些"）。

## 练习

1. 从二元评估器切换到返回距离度量的标量评估器（离目标多远）。它收敛更快吗？
2. 为反思添加 10 次试验的 TTL。在那之后更旧的反思是有害还是有益？
3. 实现启发式评估器：如果相同的动作重复，标记试验为卡住。这如何与 Self-Reflector 交互？
4. 使用一个忽略反思的对抗性 Actor 运行 Reflexion。什么是最小反思提示工程，迫使 Actor 注意到它们？
5. 阅读 Reflexion 论文第 4 节关于 ALFWorld 的部分。从概念上复现 130% 的成功率提升：与标准 ReAct 的关键差异是什么？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| Reflexion | "自我修正" | Shinn et al. 2023 — Actor、Evaluator、Self-Reflector 加上情景记忆 |
| 语言强化 | "无需梯度的学习" | 自然语言反思被前置到下一次试验的提示中 |
| 情景记忆 | "每个任务的反思" | 一个任务类别的先前反思的有界缓冲区 |
| 标量评估器 | "二元成功信号" | 来自 ground truth 的通过/失败或数值评分 |
| 启发式评估器 | "基于模式的检测器" | 预定义的失败特征（如卡住循环、步数过多） |
| 自我评估器 | "LLM 作为自己轨迹的评判" | 低信号的回退方案，当没有 ground truth 时——配合工具锚定验证 |
| 记忆腐化 | "过时的反思" | 情景缓冲区填满陈旧条目；通过压缩/TTL 修复 |
| 睡眠时反思 | "异步自我反思" | 在热路径之外运行 Self-Reflector，使主 agent 保持快速 |

## 进一步阅读

- [Shinn et al., Reflexion: Language Agents with Verbal Reinforcement Learning (arXiv:2303.11366)](https://arxiv.org/abs/2303.11366) — 经典论文
- [Letta, Sleep-time Compute](https://www.letta.com/blog/sleep-time-compute) — 生产中的异步反思
- [Anthropic, Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — 将情景缓冲区作为上下文的一部分来管理
- [LangGraph 概览](https://docs.langchain.com/oss/python/langgraph/overview) — 反思节点模式
