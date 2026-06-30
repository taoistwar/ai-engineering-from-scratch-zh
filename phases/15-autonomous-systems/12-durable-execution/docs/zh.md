# 长周期后台智能体：持久执行

> 生产环境中的长周期智能体不以 `while True` 运行。每次LLM调用变成一个带有检查点、重试和重放的活动。Temporal的OpenAI Agents SDK集成于2026年3月正式发布。Claude Code例程（Anthropic）运行计划的Claude Code调用，无需持久本地进程。会话在人工输入时暂停，在部署间存活，并从按 `thread_id` 键值的最新检查点恢复。在新的便利性背后是一个旧模式——工作流编排——带有一个新的输入：LLM调用作为必须能在恢复时确定性重放的非确定性活动。

**Type:** Learn
**Languages:** Python (stdlib, minimal durable-execution state machine)
**Prerequisites:** Phase 15 · 10 (Permission modes), Phase 15 · 01 (Long-horizon agents)
**Time:** ~60 minutes

## 问题

考虑一个运行四小时的智能体。它调用三个工具，两次提示用户，并进行四十次LLM调用。运行到一半时，它运行的主机重启了。会发生什么？

- 在一个天真的 `while True` 循环中：一切尽失。运行从头重新开始。三个工具调用（具有真实副作用）再次执行。用户再次被提示他们已经批准的事情。四十次LLM调用被重新计费。
- 使用持久执行：运行从最近的检查点恢复。已完成的活动不会重新执行；它们的结果从持久日志中重放。用户不会重新批准他们已经批准的事情。已完成的LLM调用不会被重新计费。

这是工作流引擎已交付十年的相同模式（Temporal、Cadence、Uber的Cherami）。新的是LLM调用现在是一种活动——非确定性的、昂贵的、具有副作用的——并且它们干净地契合此模式。

本课贯穿始终的主题：长周期可靠性衰减（METR观察到"35分钟衰减"——成功率随周期大致二次方下降）。持久执行使得运行时间超过可靠性配置文件所支持的范围成为可能，如果设计正确，这是一种安全失败的新方式；如果设计错误，则是不安全的。

## 概念

### 活动、工作流和重放

- **工作流**：确定性的编排代码。定义活动的序列、分支、等待。必须是确定性的，以便可以从事件日志重放而不会产生意外的分歧。
- **活动**：一个非确定性的、可能失败的工作单元。LLM调用、工具调用、文件写入、HTTP请求。每个活动记录其输入和（完成后）输出。
- **事件日志**：持久后备存储。记录每个活动的开始、完成、失败、重试，以及每个工作流决策。
- **重放**：在恢复时，工作流代码从头重新运行；每个已完成的活动返回其记录的输出而不重新执行。只有未完成的活动实际运行。

这与React对虚拟DOM重新渲染，或Git从提交重建工作树的形状相同。编排器中的确定性使持久性变得廉价。

### 为什么LLM调用契合此模式

LLM调用具有以下特性：
- 非确定性的（temperature > 0；即使temperature为0也会在模型版本间漂移）。
- 昂贵的（金钱和延迟）。
- 可能失败的（速率限制、超时）。
- 具有副作用的（如果它们调用工具）。

这正是活动的配置文件。将每次LLM调用包装为活动，即可获得指数退避重试、跨重启的检查点以及可重放的可调试跟踪。

### 按 `thread_id` 键值的检查点

LangGraph、Microsoft Agent Framework、Cloudflare Durable Objects和Claude Code例程都收敛于相同的API形状：一个 `thread_id`（或等价物）标识会话；每次状态转换持久化到后端（默认PostgreSQL，开发用SQLite，缓存用Redis）；恢复读取最新检查点。

后端选择很重要：

- **PostgreSQL**：持久、可查询、在部署间存活。LangGraph的默认选择。
- **SQLite**：仅本地开发；跨主机丢失数据。
- **Redis**：快速但短暂，除非配置AOF/快照。
- **Cloudflare Durable Objects**：透明分布；由唯一键限定范围；存活数小时到数周。

### 人工输入作为一等状态

提议然后提交（第15课）需要一个持久的"等待人类"状态。工作流暂停，外部队列持有待处理请求，批准从该确切点恢复。没有持久性，这是尽力而为的；有了它，隔夜的批准到达，工作流在早上继续。

### 35分钟衰减

METR观察到，每个测量的智能体类别在持续运行约35分钟后都表现出可靠性衰减。将任务持续时间翻倍大致使失败率翻四倍。持久执行不能修复此问题；它让你运行超过可靠性配置文件支持的时间。安全模式是将持久性与要求在重新进入时需要新HITL的检查点结合，并与限制总计算的预算熔断开关（第13课）结合，而不管墙钟时间。

### 持久执行何时是错误的答案

- 运行短于几分钟且无人工输入。开销 > 收益。
- 严格的只读信息检索。
- 需要在一个上下文窗口内端到端正确性的任务（某些推理任务；某些一次性生成）。

```figure
memory-consolidation
```

## 运用

`code/main.py` 在标准库Python中实现一个极简的持久执行引擎。它支持：

- `@activity` 装饰器，将输入和输出记录到JSON事件日志。
- 对活动进行排序的工作流函数。
- `run_or_replay(workflow, event_log)` 函数，重放已完成的活动而不重新执行它们。

驱动程序模拟一个三活动工作流，在中途崩溃，展示(a)天真重试重新执行一切 vs (b)重放仅运行缺失的活动。

## 交付物

`outputs/skill-durable-execution-review.md` 审查一个提议的长周期智能体部署，确认正确的持久执行形状：活动、确定性、检查点后端、人工输入状态和恢复时的HITL策略。

## 练习

1. 运行 `code/main.py`。观察天真重试和重放之间活动执行次数的差异。更改崩溃点，展示重放次数相应变化。

2. 将玩具引擎改为显式使用 `thread_id`。模拟两个并发会话共享引擎，确认它们的事件日志不冲突。

3. 在玩具引擎中取一个活动。引入非确定性（工作流决策中的墙钟时间戳）。展示重放时的分歧。解释真实引擎如何处理此问题（副作用注册、`Workflow.now()` API）。

4. 阅读LangChain的"生产深度智能体背后的运行时"文章。列出运行时持久化的每种状态，指出每种覆盖了什么失败模式。

5. 为6小时自主编程任务设计检查点策略。你在哪里设置检查点？崩溃恢复是什么样子？什么需要新的HITL？

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|---|---|---|
| 工作流 | "智能体的脚本" | 确定性编排代码；可从事件日志重放 |
| 活动 | "一个步骤" | 非确定性单元（LLM调用、工具调用）；执行前后都记录 |
| 事件日志 | "后备存储" | 每次状态转换的持久记录 |
| 重放 | "恢复" | 重新运行工作流；已完成的活动返回记录结果而不重新执行 |
| 检查点 | "保存点" | 按thread_id键值的持久化状态；恢复时取最新 |
| thread_id | "会话键" | 限定持久状态范围的标识符 |
| 35分钟衰减 | "可靠性衰减" | METR：成功率随周期大致二次方下降 |
| 非确定性 | "重放时漂移" | 墙钟、随机、LLM输出；必须注册为副作用 |

## 进一步阅读

- [Anthropic — Claude Code Agent SDK: agent loop](https://code.claude.com/docs/en/agent-sdk/agent-loop) — 预算、轮次和恢复语义。
- [Microsoft — Agent Framework: human-in-the-loop and checkpointing](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — RequestInfoEvent形状。
- [LangChain — The Runtime Behind Production Deep Agents](https://www.langchain.com/conceptual-guides/runtime-behind-production-deep-agents) — 具体运行时要求。
- [OpenAI Agents SDK + Temporal integration (Trigger.dev announcement)](https://trigger.dev) — LLM调用的活动形状。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 35分钟衰减参考。
