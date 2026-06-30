# 移交与例程——无状态编排

> OpenAI的Swarm（2024年10月）将多智能体编排提炼为两个原语：**例程**（指令 + 工具作为系统提示）和**移交**（一个返回另一个Agent的工具）。没有状态机，没有分支DSL——LLM通过调用正确的移交工具进行路由。OpenAI Agents SDK（2025年3月）是生产后继者。Swarm本身仍然是最清晰的概念参考——其完整源代码仅几百行。该模式之所以病毒式传播，是因为API表面大致是"agent = prompt + tools; handoff = function returning agent。"局限性：无状态，因此内存是调用者的问题。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 04 (Primitive Model)
**Time:** ~60 minutes

## 问题

每个多智能体框架都希望你去学习它的DSL：LangGraph节点和边、CrewAI剧组和任务、AutoGen GroupChat和管理者。DSL是真实的抽象，但它们让事物感觉比需要的更重。

Swarm朝相反方向推动：使用模型已有的工具调用能力。移交成为工具调用。编排者是当前持有对话的任何智能体。状态机隐含在智能体的系统提示中。

## 概念

### 两个原语

**例程。** 一个定义智能体角色和可用工具的系统提示。可以将其视为限定范围的指令集："你是一个分诊智能体；如果用户询问退款，移交给退款智能体。"

**移交。** 智能体可以调用的返回新Agent对象的工具。Swarm运行时检测到Agent返回值，并为下一回合切换活跃智能体。

这就是整个抽象。

```
def transfer_to_refunds():
    return refund_agent  # Swarm看到Agent返回值 → 切换活跃智能体

triage_agent = Agent(
    name="triage",
    instructions="Route the user to the right specialist.",
    functions=[transfer_to_refunds, transfer_to_sales, transfer_to_support],
)
```

分诊智能体的系统提示使其根据用户消息选择正确的移交。LLM的工具调用完成路由。

### 为什么它是病毒式的

- **小API。** 两个概念要学。
- **使用模型已经做的事。** 工具调用已经是跨提供者的生产级别。
- **没有状态机负担。** 你不用描述图；智能体的提示描述它们移交给谁。

### 无状态交换

Swarm在运行间显式无状态。框架在运行期间维护消息历史，但它不持久化任何东西。内存、连续性、长时间运行的任务——全部是调用者的问题。

在生产中（OpenAI Agents SDK，2025年3月）这是改变的主要事情之一：SDK添加内置会话管理、护栏和追踪，同时保留移交原语。

### 何时Swarm/移交适合

- **分诊模式。** 一线智能体将用户路由给专家。
- **基于技能的移交。** "如果任务需要代码，调用编码者；如果需要研究，调用研究者。"
- **短小、有界对话。** 客户支持、FAQ到工单、简单工作流。

### 何时Swarm挣扎

- **长会话带共享内存。** 移交将对话状态重置为新智能体的提示加历史。没有调用者管理的内存在智能体间没有持久状态。
- **并行执行。** 移交是一次一个——活跃智能体切换。并行需要调用者编排多个Swarm运行。
- **审计和重放。** 无状态运行很难精确重放；LLM的移交选择不是确定性的。

### OpenAI Agents SDK（2025年3月）

生产后继者添加：

- **会话状态。** 跨运行持久线程。
- **护栏。** 输入/输出验证钩子。
- **追踪。** 每个工具调用和移交被记录。
- **移交过滤器。** 控制在移交上传输什么上下文。

移交原语存活；围绕它添加生产人机工程学。

### Swarm vs GroupChat

两者都使用LLM驱动的路由，但它们在**谁选择下一个**上不同：

- GroupChat：选择器（函数或LLM）从外部选择下一个发言者。
- Swarm：当前智能体通过调用移交工具选择其继任者。

Swarm是"智能体决定接下来是什么"；GroupChat是"管理者决定接下来是什么。"Swarm的决策存在于活跃智能体的工具调用中；GroupChat的存在于 `GroupChatManager` 中。

## 构建它

`code/main.py` 从零实现Swarm：一个Agent数据类，一个移交机制（工具返回Agent），和一个检测智能体切换的运行循环。

演示：分诊智能体路由到退款、销售或支持专家。每个专家有自己的工具。运行循环打印每次移交。

运行：

```
python3 code/main.py
```

## 运用

`outputs/skill-handoff-designer.md` 为给定任务设计移交拓扑：存在哪些智能体，它们可以调用哪些移交，什么上下文传输。

## 交付物

检查清单：

- **移交日志。** 每次移交写入带从智能体、到智能体、上下文快照的追踪事件。
- **上下文传输规则。** 决定移交上什么移动：完整历史（昂贵）、最近N条消息或摘要。
- **移交护栏。** 移交给具有不同工具权限的专家必须经过认证——否则提示注入可以强制不需要的移交。
- **循环检测。** 两个智能体来回移交是常见失败；用简单的最近K环检查检测。
- **回退智能体。** 如果移交目标不存在，回退到安全默认值。

## 练习

1. 运行 `code/main.py`，分诊到退款智能体。确认第二回合的活跃智能体是退款。
2. 添加循环检测规则：如果相同两个智能体连续移交了3次，强制退出。设计回退方案。
3. 阅读OpenAI Agents SDK移交过滤器文档。实现"移交时摘要"版本：外出智能体在接管智能体接手前将上下文压缩为要点摘要。
4. 比较Swarm移交与GroupChatManager选择器。哪种模式使提示注入更糟糕，为什么？
5. 阅读Swarm cookbook（https://developers.openai.com/cookbook/examples/orchestrating_agents）。识别Swarm做出的一个OpenAI Agents SDK更改或保留的显式设计决策。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|----------------|------------------------|
| 例程 | "智能体提示" | 系统提示 + 工具列表。定义角色和可用移交。 |
| 移交 | "转移到另一个智能体" | 活跃智能体可调用的返回新Agent的工具。运行时切换活跃智能体。 |
| 无状态 | "运行间无内存" | Swarm不持久化任何东西；内存是调用者的责任。 |
| 活跃智能体 | "现在谁在说话" | 当前持有对话的智能体。移交改变这一点。 |
| 上下文传输 | "移交上移动什么" | 接管智能体看到的历史策略：完整、最近N或已摘要。 |
| 移交循环 | "智能体乒乓" | 两个智能体不断互相移交回对方的失败模式。 |
| OpenAI Agents SDK | "生产Swarm" | 2025年3月后继者；在移交原语之上添加会话、护栏、追踪。 |
| 移交过滤器 | "传输上的门控" | SDK功能，在移交边界检查并修改上下文。 |

## 进一步阅读

- [OpenAI cookbook — Orchestrating Agents: Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) — 参考阐述
- [OpenAI Swarm repo](https://github.com/openai/swarm) — 原始实现，保留为概念参考
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) — 带会话和追踪的生产后继者
- [Anthropic handoff-in-Claude notes](https://docs.anthropic.com/en/docs/claude-code) — Claude Code子智能体如何通过 `Task` 使用类似移交的模式
