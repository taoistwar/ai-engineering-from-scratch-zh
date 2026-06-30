# LangGraph：有状态图与持久执行

> LangGraph 是 2026 年底层有状态编排的参考标准。Agent 是状态机；节点是函数；边是转换；状态是不可变的，并在每一步后创建检查点。从任何失败点精确恢复到离开时的状态。

**类型：** 学习 + 构建
**语言：** Python（标准库）
**前置条件：** 第 14 阶段 · 01（Agent 循环），第 14 阶段 · 12（工作流模式）
**时间：** ~75 分钟

## 学习目标

- 描述 LangGraph 的核心模型：具有不可变状态的状态机、函数节点、条件边和步骤后检查点。
- 列举文档强调的四大能力：持久执行、流式传输、人在回路、综合记忆。
- 解释 LangGraph 支持的三种编排拓扑：监督者、点对点（swarm）、层级式（嵌套子图）。
- 实现一个具有不可变状态、条件边和检查点/恢复循环的标准库状态图。

## 问题

Agent 和工作流面临同一个问题：当一个 40 步的运行在第 38 步失败时，你希望从第 38 步恢复，而不是从头开始。二流的状态模型让运维人员在假定每次都是全新运行的库之上拼凑重试逻辑。

LangGraph 的设计答案：状态是一流的类型化对象，变更是显式的，检查点在每个节点之后持久化。恢复只需一个 `load_state(session_id)` 调用。

## 概念

### 图

一个图由以下定义：

- **状态类型。** 一个类型化字典（或 Pydantic 模型），每个节点读取并修改它。
- **节点。** 纯函数 `(state) -> state_update`。更新在返回后合并到状态中。
- **边。** 节点之间的条件或直接转换。
- **入口和出口。** `START` 和 `END` 哨兵节点标记边界。

示例：一个具有 `classify`、`refund`、`bug`、`sales`、`done` 节点的 agent——一个路由工作流即图。

### 持久执行

每个节点返回后，运行时序列化状态并将其写入检查点存储器（SQLite、Postgres、Redis、自定义）。在第 N 步失败时，运行时可以 `resume(session_id)` 并从第 N+1 步以精确状态继续。

LangGraph 文档明确指出这至关重要的生产用户：Klarna、Uber、J.P. Morgan。其主张不是图的形状；而是图的形状加上检查点使恢复成本低廉。

### 流式传输

每个节点可以产出部分输出。图向调用者流式传输每个节点的增量事件，以便 UI 在图运行时更新。

### 人在回路

在节点之间检查和修改状态。实现方式：在关键节点前暂停，将状态展示给人类，接受修改，恢复。检查点存储器使这变得容易，因为状态已经序列化。

### 记忆

短期（单次运行内——状态中的对话历史）和长期（跨运行——通过检查点存储器加上单独的长期存储持久化）。LangGraph 通过工具与外部记忆系统（Mem0、自定义）集成。

### 三种拓扑

1. **监督者。** 中央路由 LLM 将任务分发给专业子 agent。`langgraph-supervisor` 中的 `create_supervisor()`（尽管 LangChain 团队在 2026 年建议通过工具调用直接实现以获得更多上下文控制）。
2. **Swarm / 点对点。** Agent 直接通过共享工具表面交接。没有中央路由器。
3. **层级式。** 监督者管理子监督者，实现为嵌套子图。

### 这种模式可能出错的地方

- **检查点太小。** 只检查对话轮次会导致工具状态和记忆写入不可恢复。完整状态必须序列化。
- **非确定性节点。** 恢复假设节点输入产生相同的状态更新。随机种子、墙钟时间、外部 API 必须被捕获。
- **条件边过度使用。** 每条边都是条件边的图是一个无法推理的状态机。优先选择线性链，偶尔使用分支。

## 构建它

`code/main.py` 实现了一个标准库有状态图：

- `State` — 一个带 `messages`、`step`、`route`、`output`、`human_approval` 的类型化字典。
- `Node` — 接收状态并返回更新字典的可调用对象。
- `StateGraph` — 节点 + 边 + 条件边 + 运行 + 恢复。
- `SQLiteCheckpointer`（内存假实现）— 在每个节点后序列化状态；`load(session_id)` 恢复。
- 演示图：classify -> branch(refund / bug / sales) -> human gate -> send。

运行它：

```
python3 code/main.py
```

跟踪显示第一次运行在 human gate 失败，持久化，然后恢复产生最终输出。

## 使用它

- **LangGraph** — 参考标准，生产就绪。使用 `create_react_agent`、`create_supervisor`，或构建你自己的图。
- **AutoGen v0.4**（第 14 课）— 高并发场景的 actor 模型替代方案。
- **Claude Agent SDK**（第 17 课）— 具有内置会话存储的托管 harness。
- **自定义** — 当你需要对状态形状或检查点后端的精确控制时。

## 交付它

`outputs/skill-state-graph.md` 在任何目标运行时中生成一个 LangGraph 风格的状态图，带有检查点和恢复功能。

## 练习

1. 当分类置信度低于阈值时，添加从 `classify` 到 `end` 的条件边。在人类手动设置 `route` 后恢复运行。
2. 将类 SQLite 假实现替换为真正的 SQLite 检查点存储器。测量每步序列化开销。
3. 实现并行边：两个节点并发运行，通过自定义 reducer 合并。不可变状态在这里有什么好处？
4. 阅读 `langgraph-supervisor` 参考。将玩具移植到 `create_supervisor`。比较跟踪形状。
5. 添加流式传输：每个节点在运行时产出部分状态。打印到达的增量。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| 状态图（State graph） | "Agent 即状态机" | 类型化状态 + 节点 + 边 + reducer |
| 检查点存储器（Checkpointer） | "持久化后端" | 在每个节点后序列化状态；支持恢复 |
| Reducer | "状态合并器" | 将当前状态与节点更新合并的函数 |
| 条件边（Conditional edge） | "分支" | 由状态函数选择的边 |
| 子图（Subgraph） | "嵌套图" | 在另一个图中作为节点使用的图 |
| 持久执行（Durable execution） | "从失败恢复" | 以精确状态在最后一个成功节点重新开始 |
| 监督者（Supervisor） | "路由 LLM" | 专业子 agent 的中央调度器 |
| Swarm | "点对点 agent" | Agent 通过共享工具交接；无中央路由器 |

## 进一步阅读

- [LangGraph 概览](https://docs.langchain.com/oss/python/langgraph/overview) — 参考文档
- [langgraph-supervisor 参考](https://reference.langchain.com/python/langgraph/supervisor/) — 监督者模式 API
- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — actor 模型替代方案
- [Claude Agent SDK 概览](https://platform.claude.com/docs/en/agent-sdk/overview) — 会话存储和子 agent
