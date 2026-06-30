# 多智能体原语模型

> 2026年交付的每个多智能体框架——AutoGen、LangGraph、CrewAI、OpenAI Agents SDK、Microsoft Agent Framework——都是四维设计空间中的一个点。四个原语，仅此而已：智能体、移交、共享状态、编排者。本课从零开始构建它们，在所有四者上运行一个玩具系统，然后将每个主要框架映射到相同的轴上，以便你能用一段话读懂任何新发布。

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 (Agent Engineering), Phase 16 · 01 (Why Multi-Agent)
**Time:** ~60 minutes

## 问题

每六个月就有一个新的多智能体框架发布。2023年的AutoGen。2024年的CrewAI。2024年的LangGraph和OpenAI Swarm。2025年4月的Google ADK。2026年2月的Microsoft Agent Framework RC。每次新闻稿都声称自己是"正确的抽象"。

如果你试图一个一个地学习它们，你会精疲力竭。API看起来不同。文档在什么是"智能体"上意见不一。一个框架称其共享内存为"黑板"，另一个称为"消息池"，第三个称为"StateGraph"。你开始怀疑这个领域只是在搅动。

不是的。在营销之下，四个原语是稳定的。学一次，用一段话读懂每个新框架。

## 概念

### 四个原语

1. **智能体**——一个系统提示加一个工具列表。无状态的；每次运行从其系统提示和当前消息历史开始。
2. **移交**——从一���智能体到另一个智能体的结构化控制转移。机制上，是返回新智能体的工具调用或跟随条件的图边。
3. **共享状态**——多于一个智能体可以读取（有时写入）的任何数据结构。消息池、黑板、键值存储、向量内存。
4. **编排者**——决定谁下一个发言的任何东西。选项：显式图（确定性）、LLM发言者选择器（软）、最后一个发言者的移交调用（OpenAI Swarm），或队列上的调度器（群体架构）。

这就是整个设计空间。每个框架为每个轴选择默认值；其余的是表面语法。

### 每个2026年框架如何映射到它

| 框架 | 智能体 | 移交 | 共享状态 | 编排者 |
|-----------|-------|---------|--------------|--------------|
| OpenAI Swarm / Agents SDK | `Agent(instructions, tools)` | 工具返回Agent | 调用者的问题 | LLM的下一个移交调用 |
| AutoGen v0.4 / AG2 | `ConversableAgent` | GroupChat上的发言者选择器 | 消息池 | 选择器函数（LLM或轮询） |
| CrewAI | `Agent(role, goal, backstory)` | `Process.Sequential / Hierarchical` | 任务输出链式传递 | 管理者LLM或静态顺序 |
| LangGraph | 节点函数 | 图边 + 条件 | `StateGraph` reducer | 图，确定性的 |
| Microsoft Agent Framework | 智能体 + 编排模式 | 模式特定 | 线程/上下文 | 模式特定 |
| Google ADK | 智能体 + A2A卡片 | A2A任务 | A2A产物 | 宿主机决定 |

表面差异看起来巨大。底层：相同的四个旋钮。

### 为什么这很重要

一旦你看到原语，框架比较就变成了一个简短的检查清单：

- 编排者是否信任LLM进行路由（Swarm），还是将路由固定在代码中（LangGraph）？
- 共享状态是完整历史记录（GroupChat）还是投影（StateGraph reducer）？
- 智能体能否修改彼此的提示（CrewAI管理者），还是只能移交（Swarm）？

这三个问题回答了80%关于哪个框架适合给定问题的问题。你不再寻找"最好的多智能体框架"，而是开始设计你真正关心的轴。

### 无状态的洞见

除共享状态外的每个原语都是无状态的。智能体是（prompt, tools）的函数。移交是一个函数调用。编排者是一个调度器。**系统中唯一有状态的东西是共享状态。** 那就是所有有趣的Bug所在的地方：内存污染（第15课）、消息排序、版本控制、写入竞争。

隐藏共享状态的框架（Swarm）将问题推给调用者。集中化它的框架（LangGraph检查点、AutoGen池）使其可检查，但将协调成本转移到共享状态实现上。

### 单个原语的剖析

#### 智能体

```
Agent = (system_prompt, tools, model, optional_name)
```

没有内存。没有状态。两个具有相同系统提示和工具的智能体是可互换的。所有看起来像每个智能体状态的东西实际上在共享状态或移交协议中。

#### 移交

```
Handoff = (from_agent, to_agent, reason, payload)
```

三种实现占主导地位：

- **函数返回**——工具返回下一个智能体。这是OpenAI Swarm模式。智能体在其工具模式中携带路由。
- **图边**——LangGraph。边是声明式的。LLM产生一个值；一个条件选择下一个节点。
- **发言者选择**——AutoGen GroupChat。一个选择器函数（有时本身是LLM调用）读取池并选择谁下一个发言。

#### 共享状态

```
SharedState = { messages: [], artifacts: {}, context: {} }
```

至少是一个消息列表。通常更多：结构化产物（CrewAI Task输出）、类型化上下文（LangGraph reducers）、外部内存（MCP、向量DB）。

两种拓扑：**完整池**（每个智能体看到每条消息）和**投影**（智能体看到角色限定范围的视图）。完整池简单，扩展性差。投影池可扩展，但需要前期模式设计。

#### 编排者

```
Orchestrator = ({state, last_speaker}) -> next_agent
```

四种风格：

- **静态**——图在构建时固定（LangGraph确定性、CrewAI Sequential）。
- **LLM选择**——LLM读取池并选择下一个发言者（AutoGen、CrewAI Hierarchical）。
- **移交驱动**——当前智能体通过调用移交工具决定（Swarm）。
- **队列驱动**——工作者从共享队列中拉取；没有显式的下一个发言者（群体架构、Matrix）。

### 框架之间变化的是什么

一旦原语固定，剩余的设计决策是：

- **内存策略**——临时的 vs 持久检查点（LangGraph检查点器）。
- **安全边界**——谁可以批准移交（人在回路中）。
- **成本核算**——每个智能体的token预算。
- **可观测性**——追踪移交，持久化状态以进行重放。

所有这些都可以在原语之上实现。它们中没有一个是新的原语。

## 构建它

`code/main.py` 用约150行标准库Python实现四个原语。没有真实的LLM——每个智能体是一个脚本化策略，使焦点保持在协调结构上。

文件导出：

- `Agent`——由name、system prompt、tools、policy function组成的数据类。
- `Handoff`——返回新智能体的函数。
- `SharedState`——线程安全的消息池。
- `Orchestrator`——三种变体：`StaticOrchestrator`、`HandoffOrchestrator`、`LLMSelectorOrchestrator`（模拟）。

演示将相同的三智能体流水线（研究 → 编写 → 审查）通过所有三种编排者类型运行，并在最后打印消息池。你可以看到输出仅在*谁选择下一个*上有所不同；智能体和共享状态在运行间是相同的。

运行它：

```
python3 code/main.py
```

预期输出：三次编排者运行，每种模式一次。每次打印最终消息池。如果研究者早早决定完成，移交驱动的运行会到达更少的智能体——这是LLM路由权衡的缩影。

## 运用

`outputs/skill-primitive-mapper.md` 是一个技能，读取任何多智能体代码库或框架文档并返回四原语映射。在新的框架发布上运行它以在深入阅读文档之前获得一段话的理解。

## 交付物

在采用新框架之前，为其编写原语映射。如果你不能，文档就是不完整的，或者框架在发明第五个原语（罕见——检查你是否看到了没见过的共享状态变体）。

在你的架构文档中固定该映射。当新团队成员加入时，在API文档之前发送映射。当框架版本变化时，比较映射的差异，而不是变更日志。

## 练习

1. 用不同的智能体策略运行 `code/main.py` 三次。观察编排者选择如何改变哪些智能体运行。
2. 实现第四种编排者类型：队列驱动的，智能体轮询共享状态以获取工作。什么死锁可能发生，你如何检测它？
3. 取LangGraph快速入门（https://docs.langchain.com/oss/python/langgraph/workflows-agents）并用四个原语重写它。LangGraph的抽象中哪些是1:1映射，哪些是便利包装器？
4. 阅读OpenAI Swarm cookbook（https://developers.openai.com/cookbook/examples/orchestrating_agents）。识别Swarm使四个原语中哪一个最符合人体工程学，以及它将哪一个推给了调用者。
5. 在该表中找到一个完全隐藏共享状态的框架。解释当智能体需要在没有重新阅读历史的情况下跨移交协调时，什么会出问题。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|----------------|------------------------|
| 智能体 | "一个带有工具的LLM" | 一个 `(system_prompt, tools, model)` 三元组。无状态的。 |
| 移交 | "控制转移" | 一个命名下一个智能体和可选负载的结构化调用。三种实现：函数返回、图边、发言者选择。 |
| 共享状态 | "内存"/"上下文" | 多智能体系统中唯一有状态的部分。消息池或黑板。 |
| 编排者 | "协调者" | 决定谁下一个运行的任何东西。静态图、LLM选择器、移交驱动或队列驱动。 |
| 原语 | "抽象" | 每个框架参数化的四个轴之一。不是框架功能。 |
| 消息池 | "共享聊天历史" | 完整历史的共享状态。推理简单，扩展性差。 |
| 投影状态 | "限定视图" | 用于特定角色查看共享状态的视图。可扩展，需要模式设计。 |
| 发言者选择 | "谁下一个发言" | 编排者模式，其中函数（通常是LLM）从组中选择下一个智能体。 |

## 进一步阅读

- [OpenAI cookbook: Orchestrating Agents — Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) — 移交驱动编排的最清晰阐述
- [AutoGen stable docs](https://microsoft.github.io/autogen/stable/) — GroupChat + 发言者选择是LLM选择编排的参考
- [LangGraph workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — 图边编排和基于reducer的共享状态
- [CrewAI introduction](https://docs.crewai.com/en/introduction) — 角色-目标-背景故事智能体，顺序/层次化流程
- [AG2 (community AutoGen continuation)](https://github.com/ag2ai/ag2) — 在Microsoft将v0.4移入维护后的活跃AutoGen v0.2线
