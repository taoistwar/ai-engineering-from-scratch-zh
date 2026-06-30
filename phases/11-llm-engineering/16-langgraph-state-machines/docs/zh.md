# LangGraph — State Machines for Agents（LangGraph — 智能体的状态机）

> 手写的 ReAct 循环是一个 `while True`。用 LangGraph 编写的 ReAct 循环是一个你可以设检查点、中断、分支和时光旅行的图。智能体本身没有改变，但围绕它的框架改变了。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 11 · 09 (Function Calling), Phase 11 · 14 (Model Context Protocol)
**Time:** ~75 minutes

## 问题

你发布了一个函数调用智能体。它在三轮中正常运行，然后出了问题：模型尝试调用一个返回 500 的工具，用户在任务中途改变注意，或者智能体决定在没有人类签字的情况下退款。`while True:` 循环没有钩子。你无法暂停它、无法回退它，也无法分支到"如果模型选了另一个工具会怎样"。一旦你将此发布到演示之外，智能体就变成了一个要么成功要么失败的黑盒。

看得越清楚，下一步就越明显。智能体已经是一个状态机——系统提示词加消息历史加待处理的工具调用加下一步动作。使状态机显式化：节点表示"模型思考"、"工具运行"、"人类审批"，边表示它们之间的条件转移。一旦图变得显式，框架就免费获得了四样东西：检查点（在步骤之间保存状态）、中断（为人类暂停）、流式传输（流式推送 token 和中间事件）、时光旅行（回退到先前状态并尝试不同的分支）。

LangGraph 是提供此抽象的库。它不是 LangChain 意义上的智能体框架（"这是一个 AgentExecutor，祝你好运"）。它是一个具有一等状态、一等持久化和一等中断的图运行时。智能体循环是你画出来的，而不是你手写的。

## 概念

![LangGraph StateGraph: nodes, edges, and the checkpointer](../assets/langgraph-stategraph.svg)

一个 `StateGraph` 有三样东西。

1. **状态（State）。** 一个流过图的类型化字典（TypedDict 或 Pydantic 模型）。每个节点接收完整状态并返回部分更新，LangGraph 使用每个字段的*归约器*（reducer）合并——对于应累积的列表使用 `operator.add`，默认覆盖。
2. **节点（Nodes）。** Python 函数 `state -> partial_state`。每个是一个离散步骤："调用模型"、"运行工具"、"摘要"。
3. **边（Edges）。** 节点之间的转移。静态边去一个地方。条件边接受一个路由函数 `state -> next_node_name`，使图可以根据模型输出分支。

你编译图。编译绑定拓扑，附加一个检查点（可选但对生产至关重要），并返回一个可运行对象。你用初始状态和一个 `thread_id` 调用它。执行的每一步都将检查点持久化为以 `(thread_id, checkpoint_id)` 为键的状态。

### 四大超能力

**检查点（Checkpointing）。** 每个节点转移将新状态写入存储（测试用内存，生产用 Postgres/Redis/SQLite）。通过再次用相同的 `thread_id` 调用图来恢复。图从中断处继续。

**中断（Interrupts）。** 用 `interrupt_before=["human_review"]` 标记一个节点，执行将在该节点运行前停止。状态持久化。你的 API 回复用户"等待审批"。后续用 `Command(resume=...)` 对相同 `thread_id` 的请求恢复执行。

**流式传输（Streaming）。** `graph.stream(state, mode="updates")` 产出发生的状态增量。`mode="messages"` 在模型节点内部流式传输 LLM token。`mode="values"` 产出完整快照。你选择在你的 UI 中展示什么。

**时光旅行（Time-travel）。** `graph.get_state_history(thread_id)` 返回完整的检查点日志。将任何先前的 `checkpoint_id` 传递给 `graph.invoke`，你就从该点分叉。非常适合调试（"如果模型选了工具 B 会怎样？"）和重放生产轨迹的回归测试。

### 归约器才是重点

每个状态字段都有一个归约器。大多数默认值都不错——新值覆盖旧值。但消息列表需要 `operator.add` 以便新消息追加而不是替换。平行边通过归约器合并它们的更新。如果两个节点都更新 `messages` 而你忘记 `Annotated[list, add_messages]`，第二个会静默胜出，你丢失了一半的轮次。归约器是库中唯一微妙的地方；搞对了它，其余部分就顺畅了。

### 四节点的 ReAct 图

一个生产级 ReAct 智能体是四个节点和两条边：

1. `agent` — 以当前消息历史调用 LLM。返回助手消息（可能包含 tool_calls）。
2. `tools` — 执行最后一条助手消息中的任何 tool_calls，将工具结果作为 tool 消息追加。
3. 从 `agent` 的条件边：如果最后一条消息有 tool_calls 就路由到 `tools`，否则到 `END`。
4. 从 `tools` 回到 `agent` 的静态边。

仅此而已。你获得了完整的 ReAct 循环（Thought → Action → Observation → Thought → …），附带检查点、中断和流式传输，大约 40 行代码。

### StateGraph vs Send（扇出）

`Send(node_name, state)` 让一个节点派发并行子图。示例：智能体决定同时查询三个检索器。每个 `Send` 产生目标节点的一次并行执行；它们的输出通过状态归约器合并。这就是 LangGraph 表达编排器-工作者模式而不使用线程原语的方式。

### 子图

一个编译后的图可以是另一个图中的一个节点。外层图看到一个单一节点；内层图有自己的状态和自己的检查点。这就是团队构建监督者-工作者智能体的方式：监督者图将用户意图路由到每个领域的工作者子图。

## 构建

### 步骤 1：状态和节点

```python
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def agent_node(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: State) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END

tool_node = ToolNode(tools=[search_web, read_file])

graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

app = graph.compile(checkpointer=MemorySaver())
```

`add_messages` 是使消息列表累积而非覆盖的归约器。忘记它是 LangGraph 中最常见的 bug。

### 步骤 2：使用线程运行

```python
config = {"configurable": {"thread_id": "user-42"}}
for event in app.stream(
    {"messages": [HumanMessage("find the Anthropic headquarters address")]},
    config,
    stream_mode="updates",
):
    print(event)
```

每个更新是一个字典 `{node_name: state_delta}`。你的前端可以将这些流式推送到 UI，让用户看到"agent is thinking… calling search_web… got result… answering."

### 步骤 3：添加人类参与的中断

标记一个节点使执行在其运行前暂停。

```python
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["tools"],  # 在每次工具调用前暂停
)

state = app.invoke({"messages": [HumanMessage("delete the production database")]}, config)
# state["__interrupt__"] 已设置。检查提议的工具调用。
# 如果批准：
from langgraph.types import Command
app.invoke(Command(resume=True), config)
# 如果拒绝：写一条拒绝消息并恢复
app.update_state(config, {"messages": [AIMessage("Blocked by human reviewer.")]})
```

状态、检查点和线程都在中断期间持久化。除了执行期间，没有任何东西在内存中。

### 步骤 4：调试用的时光旅行

```python
history = list(app.get_state_history(config))
for snapshot in history:
    print(snapshot.values["messages"][-1].content[:80], snapshot.config)

# 从先前的检查点分叉
target = history[3].config  # 回退三步
for event in app.stream(None, target, stream_mode="values"):
    pass  # 从该点向前重放
```

传递 `None` 作为输入表示从给定检查点重放；传递一个值表示在恢复之前将其作为更新附加到该检查点的状态。这就是你在不重新运行整个对话的情况下重现糟糕智能体运行的方式。

### 步骤 5：为生产替换检查点

```python
from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string("postgresql://...") as checkpointer:
    checkpointer.setup()
    app = graph.compile(checkpointer=checkpointer)
```

SQLite、Redis 和 Postgres 都支持。`MemorySaver` 用于测试。任何跨重启持久化的内容都需要真实的存储。

## 经验法则

在你伸手拿 LangGraph 之前，做一个 60 秒的设计：

1. **命名节点。** 每个离散决策或产生副作用的动作是一个节点。"Agent thinks," "tool runs," "reviewer approves," "response streams." 如果你无法列出它们，任务还不是智能体形状的。
2. **声明状态。** 最小的 TypedDict，每个列表字段有一个归约器。不要把一切都塞进 `messages`；提升任务特定字段（一个工作的 `plan`、一个 `budget` 计数器、一个 `retrieved_docs` 列表）到顶层。
3. **画出边。** 静态边除非下一步取决于模型输出。每个条件边需要一个带命名分支的路由函数。
4. **预先选择检查点。** 测试用 `MemorySaver`，其他用 Postgres/Redis/SQLite。不要在没有检查点的情况下发布——没有检查点意味着没有恢复、没有中断、没有时光旅行。
5. **在工具运行前决定中断，而非之后。** 审批放在进入产生副作用的节点的边上，以便在损害前取消；验证放在模型出来的边上，以便廉价地拒绝错误调用。
6. **默认流式传输。** UI 用 `mode="updates"`，模型节点内部 token 级流式传输用 `mode="messages"`，评估时完整快照用 `mode="values"`。

拒绝发布没有检查点的 LangGraph 智能体。拒绝在副作用之后中断的。拒绝没有 `add_messages` 作为归约器的 `messages` 字段。

## 练习

1. **简单.** 使用计算器工具和网络搜索工具实现上述的四节点 ReAct 图。验证 `list(app.get_state_history(config))` 为一个两轮对话返回至少四个检查点。
2. **中等.** 添加一个 `planner` 节点，在 `agent` 之前运行，将结构化的 `plan: list[str]` 写入状态。让 `agent` 标记计划步骤为已完成。如果 `plan` 在跨检查点恢复时丢失（错误的归约器），测试失败。
3. **困难.** 构建一个监督者图，使用 `Send` 在三个子图之间路由（`researcher`、`writer`、`reviewer`）。每个子图有自己的状态和检查点。在外层图上添加 `interrupt_before=["writer"]`，使人类可以审批研究简报。确认从先前检查点的时光旅行只重放分叉的分支。

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| StateGraph | "The LangGraph graph" | The builder object you add nodes and edges to before compile. |
| Reducer | "How the field merges" | A function `(old, new) -> merged` applied when a node returns an update for that field; default is overwrite, `add_messages` appends. |
| Thread | "A conversation ID" | A `thread_id` string that scopes all checkpoints for one session. |
| Checkpoint | "A paused state" | A persisted snapshot of the full graph state after a node transition, keyed on `(thread_id, checkpoint_id)`. |
| Interrupt | "Pause for a human" | `interrupt_before` / `interrupt_after` stop execution at a node boundary; resume with `Command(resume=...)`. |
| Time-travel | "Fork from a prior step" | `graph.invoke(None, config_with_old_checkpoint_id)` replays from that checkpoint forward. |
| Send | "Parallel subgraph dispatch" | A constructor a node can return to spawn N parallel executions of a target node. |
| Subgraph | "A compiled graph as a node" | A compiled StateGraph used as a node in another graph; preserves its own state scope. |

## Further Reading

同英文版。
