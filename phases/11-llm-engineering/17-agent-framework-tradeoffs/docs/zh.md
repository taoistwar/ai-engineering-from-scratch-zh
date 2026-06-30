# Agent Framework Tradeoffs — LangGraph vs CrewAI vs AutoGen vs Agno（智能体框架权衡）

> 每个框架卖的都是同一个演示（研究智能体构建报告）并隐藏同一个 bug（状态 schema 与编排层冲突）。选择核心抽象与你的问题形状匹配的框架；其他一切都是你写两遍的胶水代码。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 11 · 09 (Function Calling), Phase 11 · 16 (LangGraph)
**Time:** ~45 minutes

## 问题

你有一个需要多于一次 LLM 调用的任务。可能是研究工作流（计划、搜索、摘要、引用）。可能是代码审查流水线（解析 diff、批评、打补丁、验证）。可能是一个预订航班、写邮件和提交费用报告的多轮助手。你选择了一个框架。

三天后，你发现框架的抽象在泄漏。CrewAI 给你角色，但当 "researcher" 需要将结构化计划交给 "writer" 时它会与你对抗。AutoGen 给你智能体之间的聊天，但没有一等状态，所以你的检查点只是对话日志的 pickle。LangGraph 给你一个状态图，但强迫你在知道智能体将做什么之前命名每一个转移。Agno 给你一个单智能体抽象，当你尝试扇出到三个并发工作者时会出错。

解决办法不是"选最好的框架"，而是将框架的核心抽象与你的问题的形状匹配。本课绘制了这张地图。

## 概念

![Agent framework matrix: core abstraction vs problem shape](../assets/framework-matrix.svg)

四个框架主导着 2026 年的格局。它们的核心抽象并不相同。

| Framework | Core abstraction | Best fit | Worst fit |
|-----------|------------------|----------|-----------|
| **LangGraph** | `StateGraph` — typed state, nodes, conditional edges, checkpointer. | Workflows with explicit state and human-in-the-loop interrupts; production agents needing time-travel debugging. | Loose, role-driven brainstorming where the topology is unknown. |
| **CrewAI** | `Crew` — roles (goal, backstory), tasks, process (sequential or hierarchical). | Role-playing or persona-driven workflows with a short linear/hierarchical plan. | Anything stateful beyond the crew's turn history; complex branching. |
| **AutoGen** | `ConversableAgent` pair — two or more agents that speak in turns until an exit condition. | Multi-agent *dialogue* (teacher-student, proposer-critic, actor-reviewer) where the thinking emerges from the chat. | Deterministic workflows with a known DAG; anything needing durable state across restarts. |
| **Agno** | `Agent` — a single LLM + tools + memory, composable into teams. | Fast-to-build single agents and lightweight teams; strong multi-modality and built-in storage drivers. | Deep, explicitly-branched graphs with custom reducers. |

### "抽象"实际意味着什么

框架的核心抽象是你在白板上推销架构时画的东西。

- **LangGraph** → 你画一个图。节点是步骤，边是转移，每一点的状态对象都是类型化的。心智模型是状态机。
- **CrewAI** → 你画一个组织图。每个角色有职位描述，一个管理者路由任务。心智模型是一个专家小团队。
- **AutoGen** → 你画一个 Slack DM。两个智能体互相发消息；如果你需要主持人，第三个加入。心智模型是聊天。
- **Agno** → 你画一个盒子，上面挂着工具。将盒子并排放置形成一个团队。心智模型是"自带电池的智能体"。

### 状态问题

状态是大多数框架选择在线上崩溃的地方。

- **LangGraph。** 类型化状态（`TypedDict` 或 Pydantic 模型），逐字段归约器，一等检查点（SQLite/Postgres/Redis）。恢复、中断和时光旅行是免费的。*（参见 Phase 11 · 16。）*
- **CrewAI。** 状态通过 `context` 字段在任务之间作为字符串流动，或通过 `output_pydantic` 结构化。开箱即用没有持久的 crew 级存储；如果 crew 必须在重启后存活，你需要自行螺栓。
- **AutoGen。** 状态是聊天历史和任何用户定义的 `context`。对话记录持久化；任意工作流状态不持久化，除非你编写适配器。
- **Agno。** 内置存储驱动（SQLite、Postgres、Mongo、Redis、DynamoDB），通过 `storage=` 附加到 `Agent`——对话会话和用户记忆自动持久化。不是完整的图检查点；是一个会话存储。

### 分支问题

每个非平凡的智能体都会分支。谁决定分支很重要。

- **LangGraph** — 你通过条件边决定。路由是一个带命名分支的 Python 函数。分支在编译的图中是一等的；检查点记录走了哪个分支。
- **CrewAI** — 在层级模式中由管理者决定；在顺序模式中你在构建时决定。路由隐式存在于任务列表中；在管理者提示词之外没有一等"if"。
- **AutoGen** — 智能体通过聊天决定。分支从谁下一个发言而涌现。`GroupChatManager` 选择下一个发言者；你可以手写 `speaker_selection_method`，但默认是 LLM 驱动的。
- **Agno** — 智能体通过调用哪个下一个工具来决定。团队有 coordinator/router/collaborator 模式；更进一步的分支是开发者的责任。

### 可观察性问题

- **LangGraph** — 通过 LangSmith 或任何 OTel 导出器的 OpenTelemetry。每个节点转移是一个追踪 span；检查点兼作可重放的追踪。LangSmith 是一方选项；Langfuse/Phoenix 也有适配器。
- **CrewAI** — 自 2025 年底起一等 OpenTelemetry；与 Langfuse、Phoenix、Opik、AgentOps 集成。
- **AutoGen** — 通过 `autogen-core` 集成的 OpenTelemetry；AgentOps 和 Opik 有连接器。追踪粒度是每次智能体消息，而非每个节点。
- **Agno** — 内置 `monitoring=True` 标志加 OpenTelemetry 导出器；与 Langfuse 紧密集成用于会话追踪。

### 成本和延迟

所有四个框架都增加每次调用的开销（框架逻辑、验证、序列化）。开销递增的大致顺序：Agno ≈ LangGraph < CrewAI ≈ AutoGen。差异主要由框架完成的额外 LLM 路由数量决定。CrewAI 的层级管理者花费 token 决定谁下一个做；AutoGen 的 `GroupChatManager` 同样。LangGraph 只在你写 `llm.invoke` 的地方花费 token。Agno 的单智能体路径很薄。

当每次运行成本重要时，优先选择显式路由（LangGraph edges、AutoGen `speaker_selection_method`）而非 LLM 选择的路由。

### 互操作性

- **LangGraph** ↔ **LangChain** 工具、检索器、LLM。一等 MCP 适配器（工具作为 MCP 服务器导入）。
- **CrewAI** ↔ 工具继承自 `BaseTool`；LangChain 工具、LlamaIndex 工具和 MCP 工具都适配进来。通过 `allow_delegation=True` 进行 crew 到 crew 的委托。
- **AutoGen** → `FunctionTool` 包装任何 Python 可调用对象；MCP 适配器可用。与 AG2 生态系统的智能体到智能体模式紧密耦合。
- **Agno** → `@tool` 装饰器或 BaseTool 子类；MCP 适配器；工具可在智能体和团队间共享。

## 经验法则

> 你可以用一句话解释为什么给定框架适合给定智能体问题。

构建前检查清单：

1. **画出形状。** 这是一个图（类型化状态，命名转移）？角色扮演（专家交接工作）？聊天（智能体交谈直到完成）？带工具的单个智能体？
2. **决定谁分支。** 开发者决定的分支 → LangGraph。管理者智能体决定 → CrewAI 层级模式。聊天涌现 → AutoGen。工具调用决定 → Agno。
3. **检查状态预算。** 你需要从检查点恢复吗？时光旅行？运行中的人类中断？如果是，LangGraph 是默认选择；Agno 会话覆盖对话范围的状态。
4. **检查成本预算。** LLM 选择的路由每轮花费额外 token。如果智能体每天运行数千次，优先显式路由。
5. **预算框架开销。** 每个框架是一个额外的依赖。如果任务只是两次 LLM 调用和一个工具，写 30 行纯 Python；没有框架是最便宜的框架。

拒绝在你能画出图、组织图、聊天或智能体盒子之前伸手拿框架。拒绝选择一个迫使你与它的状态模型斗争以得到你实际需要的东西的框架。

## 决策矩阵

| Problem shape | Preferred framework | Why |
|---------------|---------------------|-----|
| Workflow DAG with typed state, human approvals, long-running | LangGraph | First-class state, checkpointer, interrupts, time-travel. |
| Research / writing pipeline with distinct roles | CrewAI (sequential) or LangGraph subgraphs | Role-per-task is cheap to express in CrewAI; scale up with LangGraph when branching gets complex. |
| Proposer-critic or teacher-student dialogue | AutoGen | Two-agent chat is its native shape. |
| Single agent with tools, sessions, memory | Agno | Thinnest setup, built-in storage and memory. |
| Thousands of parallel fanouts with reducers | LangGraph + `Send` | The only one with a first-class parallel-dispatch API. |
| Quick prototype, no framework commitment | Plain Python + provider SDK | No framework is the fastest framework. |

## 练习

1. **简单.** 用同样的任务——"research Anthropic's headquarters, write a 200-word brief, cite sources"——在 LangGraph（四个节点：plan、search、write、cite）和 CrewAI（三个角色：researcher、writer、editor）中实现。报告每次运行的 token 成本和代码行数。
2. **中等.** 在 AutoGen（researcher ↔ writer 聊天，editor 通过 `GroupChat` 加入）和 Agno（单个智能体带 `search_tools` 和 `write_tools`，再加一个会话存储）中构建同一任务。在 (a) 每次运行成本、(b) 崩溃后恢复的能力、(c) 在写入步骤之前注入人类审批的能力上对四种实现进行排名。
3. **困难.** 构建一个决策树脚本 `pick_framework.py`，接受简短的问题描述（JSON：`{has_typed_state, has_roles, has_dialogue, has_parallel_fanout, needs_resume}`）并返回一个带一句话理由的推荐。在你自行设计的六个案例上验证它。

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Orchestration | "How the agents coordinate" | The layer that decides which node/role/agent runs next. |
| Durable state | "Resume after a restart" | State that survives process death, attached to a checkpoint or session store. |
| LLM-selected routing | "Let the model decide" | A planner LLM picks the next step each turn; flexible but pays tokens on every decision. |
| Explicit routing | "Developer decides" | A Python function or static edge picks the next step; cheap and auditable. |
| Crew | "A CrewAI team" | Roles + tasks + process (sequential or hierarchical) bound into a single runnable. |
| GroupChat | "AutoGen's multi-agent chat" | A managed conversation between N agents with a speaker selector. |
| Team (Agno) | "Multi-agent Agno" | Route / coordinate / collaborate mode over a set of agents. |
| StateGraph | "LangGraph's graph" | Typed-state, node, conditional-edge, checkpointer abstraction. |

## Further Reading

同英文版。
