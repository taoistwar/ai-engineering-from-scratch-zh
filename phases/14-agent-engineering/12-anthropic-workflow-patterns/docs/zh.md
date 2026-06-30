# Anthropic 的工作流模式：简单胜于复杂

> Schluntz 与 Zhang（Anthropic，2024 年 12 月）区分工作流（预定义路径）与 agent（动态工具使用）。五种工作流模式覆盖了大多数情况。从直接 API 调用开始。仅在步骤不可预测时才添加 agent。

**类型:** Learn + Build
**语言:** Python（标准库）
**前置要求:** Phase 14 · 01（Agent Loop）
**时间:** ~60 分钟

## 学习目标

- 说出 Anthropic 的五种工作流模式：prompt chaining、routing、parallelization、orchestrator-workers、evaluator-optimizer。
- 解释 agent-vs-workflow 的区别以及各自的工程成本。
- 识别何时选择工作流而非 agent（以及反之）。
- 在标准库中针对脚本化 LLM 实现全部五种模式。

## 问题

团队为可以通过单一函数调用解决的问题引入了多 agent 框架。成本是真实的：框架增加了层，模糊了提示，隐藏了控制流，并招致过早的复杂性。Schluntz 和 Zhang 于 2024 年 12 月的文章是被引用最多的行业反击：从简单开始，仅在复杂性值得其成本时才添加。

## 概念

### 工作流 vs agent

- **工作流。** 通过预定义代码路径编排的 LLM 和工具。工程师拥有图。
- **Agent。** LLM 动态指导自己的工具并采取自己的步骤。模型拥有图。

两者各有用处。工作流更便宜、更快、更容易调试。Agent 解锁开放式问题，但使失败模式更难以推理。

### 增强型 LLM

所有五种模式的基础：一个 LLM 内置了三种能力 —— 搜索（检索）、工具（动作）、记忆（持久化）。任何 API 调用都可以使用这些。

### 五种模式

1. **Prompt chaining。** 调用 1 的输出是调用 2 的输入。当任务有干净的线性分解时使用。步骤之间有可选的程序化门控。

2. **Routing。** 一个分类器 LLM 选择调用哪个下游 LLM 或工具。当不同类别的输入需要不同处理时使用（一级支持 vs 退款 vs bug vs 销售）。

3. **Parallelization。** 并发运行 N 个 LLM 调用，聚合结果。两种形状：分段（不同的块）和投票（相同提示，N 次运行，多数/综合）。

4. **Orchestrator-workers。** 一个编排器 LLM 动态决定运行哪些 worker（也是 LLM）并综合其输出。类似 agent 循环，但编排器不会无限循环。

5. **Evaluator-optimizer。** 一个 LLM 提议答案，另一个 LLM 评估它。迭代直到评估器通过。这是 Self-Refine（第 05 课）的泛化。

### 工作流优于 agent 的地方

- **可预测的任务。** 如果你能枚举步骤，你就应该枚举。
- **成本受限的任务。** 工作流有界的步骤计数；agent 可能螺旋。
- **合规受限的任务。** 审计者想读图，而不是从轨迹中推断它。

### Agent 优于工作流的地方

- **开放式研究。** 当下一步取决于上一步返回了什么。
- **可变长度任务。** 步骤数未知的几分钟到几小时的工作。
- **新领域。** 当你还不知道正确的工作流时——先探索，后编码。

### 上下文工程的伴侣

"Effective context engineering for AI agents"（Anthropic 2025）形式化了相邻学科：200k 窗口是预算，不是容器。包括什么、何时压缩、何时让上下文增长。在 Phase 14 关于上下文压缩的课程中详细讨论（本课程重新编号前的 Phase 14 第 06 课）。

## Build It

`code/main.py` 针对一个 `ScriptedLLM` 实现全部五种工作流模式：

- `prompt_chain(input, steps)` — 顺序执行。
- `route(input, classifier, handlers)` — 分类 + 分发。
- `parallel_vote(prompt, n, aggregator)` — N 次运行，聚合。
- `orchestrator_workers(task, workers)` — 编排器选择 worker。
- `evaluator_optimizer(task, proposer, evaluator, max_iter)` — 循环直到通过。

运行它：

```
python3 code/main.py
```

每种模式打印其追踪。每个模式的代码总行数约为 10-15 行；框架的成本以千行计。

## Use It

- 对大多数任务使用直接 API 调用。
- 仅在模式真正需要持久状态（LangGraph）、actor-model 并发（AutoGen v0.4）或角色模板化（CrewAI）时才使用框架。
- 当你想要 Claude Code harness 形状而又不想重建它时，使用 Claude Agent SDK。

## Ship It

`outputs/skill-workflow-picker.md` 为给定的任务描述选择正确的模式，包括决策理由和如果工作流不足时的 agent 重构路径。

## 练习

1. 实现带有置信度阈值的 routing。低于阈值 -> 升级到人工。对于一级支持用例，阈值应设在何处？
2. 为 `parallel_vote` 添加超时。当一个调用挂起时会发生什么？如何在缺少票数的情况下聚合？
3. 将 `evaluator_optimizer` 变成 bandit：在迭代之间保留 top-2 输出，这样后期好的结果不会被后期坏的结果覆盖。
4. 结合 prompt chaining 与 routing：路由器选择三条链中的一条。测量 token 成本与单大提示替代方案的对比。
5. 选择你的一个生产功能。画工作流图。数步数。在这里 agent 真的会更好吗？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| 工作流 | "预定义流程" | 工程师拥有的 LLM 和工具调用图 |
| Agent | "自主 AI" | 模型拥有的图；动态工具指导 |
| 增强型 LLM | "带工具的 LLM" | LLM + 搜索 + 工具 + 记忆；原子单元 |
| Prompt chaining | "顺序调用" | 调用 N 的输出是调用 N+1 的输入 |
| Routing | "分类器分发" | 选择哪条链/哪个模型处理输入 |
| Parallelization | "扇出" | N 个并发调用；通过分段或投票聚合 |
| Orchestrator-workers | "分发器 agent" | 编排器 LLM 动态选择专家 LLM |
| Evaluator-optimizer | "提议者 + 评判者" | 迭代直到评估器通过；Self-Refine 的泛化 |

## 进一步阅读

- [Anthropic, Building Effective Agents (2024 年 12 月)](https://www.anthropic.com/research/building-effective-agents) — 五种工作流模式
- [Anthropic, Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — 伴侣学科
- [LangGraph 概览](https://docs.langchain.com/oss/python/langgraph/overview) — 当有状态图值得其成本时
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — orchestrator-workers 模式的产品化
