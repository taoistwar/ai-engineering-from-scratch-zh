# AutoGen v0.4：Actor 模型与 Agent 框架

> AutoGen v0.4（Microsoft Research，2025 年 1 月）围绕 actor 模型重新设计了 agent 编排。异步消息交换、事件驱动 agent、故障隔离、原生并发。该框架现已进入维护模式，Microsoft Agent Framework（2025 年 10 月公开预览）成为其继任者。

**类型：** 学习 + 构建
**语言：** Python（标准库）
**前置条件：** 第 14 阶段 · 01（Agent 循环），第 14 阶段 · 12（工作流模式）
**时间：** ~75 分钟

## 学习目标

- 描述 actor 模型：agent 即 actor，消息是唯一的进程间通信方式，每个 actor 独立故障隔离。
- 列举 AutoGen v0.4 的三个 API 层——Core、AgentChat、Extensions——以及各自的用途。
- 解释为什么将消息传递与处理解耦能提供故障隔离和原生并发。
- 在 Python 中实现一个标准库 actor 运行时，并在其上移植一个双 agent 代码审查流程。

## 问题

大多数 agent 框架是同步的：一个 agent 生产，一个 agent 消费，在一个调用栈中。故障会使整个栈崩溃。并发是后来附加的。分布式需要重写。

AutoGen v0.4 的答案：actor 模型。每个 agent 是一个拥有私有收件箱的 actor。消息是唯一的交互方式。运行时将传递与处理解耦。故障隔离到单个 actor。并发是原生的。分布式只是不同的传输层。

## 概念

### Actor

一个 actor 拥有：

- 私有状态（外部永远不能直接触碰）。
- 一个收件箱（消息队列）。
- 一个处理器：`receive(message) -> effects`，其中 effects 可以是"回复"、"发送给其他 actor"、"生成新 actor"、"更新状态"、"停止自身"。

两个 actor 不能共享内存。它们只能发送消息。

### AutoGen v0.4 的三个 API 层

1. **Core。** 底层 actor 框架。`AgentRuntime`、`Agent`、`Message`、`Topic`。异步消息交换，事件驱动。
2. **AgentChat。** 任务驱动的高级 API（替代 v0.2 的 ConversableAgent）。`AssistantAgent`、`UserProxyAgent`、`RoundRobinGroupChat`、`SelectorGroupChat`。
3. **Extensions。** 集成——OpenAI、Anthropic、Azure、工具、记忆。

### 为什么解耦很重要

在 v0.2 模型中，调用 `agent_a.chat(agent_b)` 会同步阻塞 agent_a 直到 agent_b 返回。在 v0.4 中，`send(agent_b, msg)` 将消息放入 agent_b 的收件箱并返回。运行时稍后传递。三个后果：

- **故障隔离。** Agent B 崩溃不会导致 Agent A 崩溃——运行时在 B 的处理器中捕获故障并决定如何处理（记录、重试、死信）。
- **原生并发。** 同时有多条消息在飞行；actor 并发处理其收件箱。
- **分布式就绪。** 收件箱 + 传输层是相同的抽象，无论 actor 是在进程内还是在另一台主机上。

### 拓扑

- **RoundRobinGroupChat。** Agent 按固定轮换顺序轮流发言。
- **SelectorGroupChat。** 一个选择器 agent 根据对话上下文选择下一个发言人。
- **Magentic-One。** 用于网页浏览、代码执行、文件处理的参考多 agent 团队。基于 AgentChat 构建。

### 可观测性

内置 OpenTelemetry 支持。每条消息发出一个 span；工具调用携带符合 2026 年 OTel GenAI 语义约定（第 23 课）的 `gen_ai.*` 属性。

### 状态：维护模式

2026 年初：AutoGen v0.7.x 对研究和原型设计是稳定的。Microsoft 已将活跃开发转移到 Microsoft Agent Framework（2025 年 10 月 1 日公开预览；1.0 GA 目标为 2026 年 Q1 末）。AutoGen 的模式可以清晰地前向移植——actor 模型是持久的思想。

## 构建它

`code/main.py` 实现了一个标准库 actor 运行时：

- `Message` — 带有 `sender`、`recipient`、`topic`、`body` 的类型化负载。
- `Actor` — 抽象类，带有 `receive(message, runtime)`。
- `Runtime` — 具有共享队列、传递、故障隔离的事件循环。
- 双 actor 演示：`ReviewerAgent` 审查代码，`ChecklistAgent` 执行检查清单；它们交换消息直到达成共识。

运行它：

```
python3 code/main.py
```

跟踪显示消息传递、一个 actor 中模拟的故障不会导致另一个崩溃，以及收敛到共享裁决。

## 使用它

- **AutoGen v0.4/v0.7**（维护中）— 对研究、原型设计、多 agent 模式保持稳定。
- **Microsoft Agent Framework**（公开预览）— 前向路径；相同的 actor 模型思想，使用更新的 API。
- **LangGraph swarm 拓扑**（第 13 课）— 通过共享工具交接的类似模式。
- **自定义 actor 运行时** — 当你需要特定传输层时（NATS、RabbitMQ、gRPC）。

## 交付它

`outputs/skill-actor-runtime.md` 生成一个最小 actor 运行时，加上一个针对给定多 agent 任务的团队模板（RoundRobin 或 Selector）。

## 练习

1. 添加一个死信队列：当处理器抛出异常时，将失败的消息搁置以供人类检查。在你的玩具中，DLQ 被命中的频率如何？
2. 实现 `SelectorGroupChat`：一个选择器 actor 根据对话状态选择下一个处理消息的人。
3. 添加分布式传输：将进程内队列替换为 JSON-over-HTTP 服务器，使 actor 可以在独立进程中运行。
4. 为每条消息接入一个 OTel span（或无操作占位符）。按照第 23 课发出 `gen_ai.agent.name`、`gen_ai.operation.name`。
5. 阅读 AutoGen v0.4 的架构文章。将你的玩具移植到真正的 `autogen_core` API。你跳过了哪些在生产中重要的部分？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| Actor | "Agent" | 私有状态 + 收件箱 + 处理器；不共享内存 |
| 消息（Message） | "事件" | 类型化负载；actor 之间唯一的交互方式 |
| 收件箱（Inbox） | "邮箱" | 每个 actor 的待处理消息队列 |
| 运行时（Runtime） | "Agent 宿主" | 路由消息并隔离故障的事件循环 |
| 主题（Topic） | "频道" | actor 之间的命名发布-订阅路由 |
| 故障隔离（Fault isolation） | "让它崩溃" | 一个 actor 故障不会导致其他 actor 崩溃 |
| RoundRobinGroupChat | "固定轮换团队" | Agent 按顺序轮流发言 |
| SelectorGroupChat | "上下文路由团队" | 选择器决定下一个发言人 |
| Magentic-One | "参考团队" | 用于网页 + 代码 + 文件的多 agent 小队 |

## 进一步阅读

- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — 重新设计文章
- [LangGraph 概览](https://docs.langchain.com/oss/python/langgraph/overview) — 图形化替代方案
- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — AutoGen 默认发出的 span
