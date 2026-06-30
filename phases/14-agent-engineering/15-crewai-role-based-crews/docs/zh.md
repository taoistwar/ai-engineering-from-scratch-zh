# CrewAI：基于角色的 Crew 和 Flow

> CrewAI 是 2026 年基于角色的多 agent 框架。四个原语：Agent、Task、Crew、Process。两种顶层形态：Crew（自主的、基于角色的协作）和 Flow（事件驱动的、确定性的）。文档直言不讳："对于任何生产就绪的应用，从 Flow 开始。"

**类型：** 学习 + 构建
**语言：** Python（标准库）
**前置条件：** 第 14 阶段 · 12（工作流模式），第 14 阶段 · 14（Actor 模型）
**时间：** ~75 分钟

## 学习目标

- 列举 CrewAI 的四个原语（Agent、Task、Crew、Process）以及各自拥有的内容。
- 区分 Sequential、Hierarchical 和计划中的 Consensus 流程；为每种工作负载选择一种。
- 区分 Crew（自主的基于角色）和 Flow（事件驱动的确定性），并解释文档的生产建议。
- 使用 `@tool` 装饰器和 `BaseTool` 子类接入工具；推理结构化输出与自由文本的区别。
- 列举 CrewAI 的四种记忆类型以及每种何时有价值。
- 实现一个标准库三 agent crew（研究员、写手、编辑），产出简报。
- 识别 CrewAI 的三种失败模式：提示词膨胀、manager-LLM 开销、脆弱的交接。

## 问题

采用多 agent 框架的团队会撞上同一堵墙。"自主协作"在演示中听起来很棒。然后客户提交了一个 bug，你需要确定性地重放。或者财务部门询问一个 LLM 路由的 crew 每次运行的成本是多少。或者值班人员需要知道凌晨 3 点是哪个 agent 卡住了。

自由形式的 LLM 路由 crew 无法干净地回答这些问题。纯 DAG 可以回答所有这些问题，但失去了头脑风暴 agent 所需的探索性形态。

CrewAI 的分割诚实地面对了这一权衡。Crew 用于协作的、基于角色的、探索性工作。Flow 用于事件驱动的、代码拥有的、可审计的生产。同一个框架，两种形态，根据场景选择。

## 概念

### 四个原语

CrewAI 的接口很小。记住这些，其余都是配置。

- **Agent。** `role + goal + backstory + tools + (可选) llm`。backstory 是承载关键作用的。它塑造了语气、判断力、agent 何时停止。tools 是 agent 可以调用的函数（详见下文）。
- **Task。** `description + expected_output + agent + (可选) context + (可选) output_pydantic`。可复用的工作单元。`expected_output` 是契约。`context` 列出上游任务，其输出被传入。`output_pydantic` 强制结构化形态。
- **Crew。** 容器。拥有 `agents` 列表、`tasks` 列表、`process` 以及可选的 `memory` + `verbose` + `manager_llm` 设置。
- **Process。** 执行策略。Sequential、Hierarchical、Consensus（计划中）。选择运行的形态。

Agent 彼此之间不直接看见对方。Task 引用 Agent。Crew 编排 Task。Process 决定谁选择下一个任务。这就是整个心智模型。

> **验证版本** CrewAI 0.86（2026-05）。较新版本可能重命名或合并流程类型；在依赖特定形态之前，请查看 [CrewAI Processes 文档](https://docs.crewai.com/concepts/processes)。

### Sequential vs Hierarchical vs Consensus

- **Sequential。** 任务按声明顺序运行。任务 N 的输出作为 `context` 可供任务 N+1 使用。成本最低。最可预测。当顺序固定时使用。
- **Hierarchical。** 一个管理者 Agent（独立的 LLM 调用）在专家之间路由。CrewAI 从你的 `manager_llm` 配置或默认值生成管理者。管理者每轮选择下一个任务，可以拒绝或重新路由。当你有四个或更多专家且顺序真正取决于先前输出时使用。
- **Consensus。** 计划中，目前未在公共 API 中实现。文档为该名称预留了未来基于投票的流程。目前不要依赖它。

Hierarchical 在每次专家调用之上增加了每轮一次的 LLM 调用（管理者）。在五步运行中，token 成本可能增加两倍。只有在需要路由时才为此付费。

### Crew vs Flow

这是 2026 年文档引导的框架。

- **Crew。** LLM 驱动的自主性。框架在运行时选择形态。适用于：研究、头脑风暴、初稿，任何路径本身就是答案一部分的场景。难以重放。难以测试。原型设计成本低。
- **Flow。** 你拥有的事件驱动图。`@start` 标记入口。`@listen(topic)` 标记一个步骤，当另一个步骤发出该主题时触发。每个步骤是纯 Python（内部可以调用 Crew）。适用于：生产。可观测。可测试。确定性的。

文档的 2026 年生产建议：从 Flow 开始。当自主性值得其成本时，将 Crew 作为 Flow 步骤内的 `Crew.kickoff()` 调用纳入。Flow 给你审计轨迹，Crew 给你探索能力。组合使用，不要二选一。

### 工具集成

三种方式给 Agent 添加工具。选择最适合的最简单方式。

1. **`@tool` 装饰器。** 纯函数变为工具。签名即 schema；文档字符串是 LLM 看到的描述。最适合一次性辅助函数。

   ```python
   from crewai.tools import tool

   @tool("Search the web")
   def search(query: str) -> str:
       """Return top results for the query."""
       return run_search(query)
   ```

2. **`BaseTool` 子类。** 基于类的工具，具有显式参数 schema、异步支持、重试。当工具有状态（客户端、缓存）或需要结构化参数时使用。

   ```python
   from crewai.tools import BaseTool
   from pydantic import BaseModel

   class SearchArgs(BaseModel):
       query: str
       limit: int = 10

   class SearchTool(BaseTool):
       name = "web_search"
       description = "Search the web and return top results."
       args_schema = SearchArgs

       def _run(self, query: str, limit: int = 10) -> str:
           return self.client.search(query, limit=limit)
   ```

3. **内置工具包。** CrewAI 提供第一方适配器：`SerperDevTool`、`FileReadTool`、`DirectoryReadTool`、`CodeInterpreterTool`、`RagTool`、`WebsiteSearchTool`。一次导入即可接入。

结构化输出使用 Pydantic。在 Task 上传递 `output_pydantic=MyModel`。CrewAI 根据模型验证 LLM 响应，并进行强制转换或重试。将其与严格的 `expected_output` 字符串配对。自由文本输出适合草稿；结构化输出是下游 Flow 可以消费的。

### 记忆钩子

CrewAI 开箱即用提供四种记忆类型。它们可以组合：一个 Crew 可以同时启用所有四种。

> **验证版本** CrewAI 0.86（2026-05）。最近的版本将所有内容路由到一个统一的 `Memory` 系统，该系统包装了这四种存储。下面的概念模型仍然成立，但公共类接口可能在较新版本中合并为单一的 `Memory` 入口点；检查 [CrewAI 记忆文档](https://docs.crewai.com/concepts/memory) 获取当前 API。

- **短期。** 单次运行内的对话缓冲区。运行结束时清除。
- **长期。** 跨运行持久化。存储在向量数据库中（默认 Chroma，可替换）。通过相似度检索与当前任务相关的内容。
- **实体。** 按实体的信息。"客户 X 用的是企业版计划。"按键是实体，不是相似度。跨运行存活。
- **上下文。** 组装时检索。在 Agent 需要的那一刻拉取相关记忆，而不是预加载。

在 Crew 上通过 `memory=True` 或按类型配置启用。由你配置的嵌入提供者支持（默认 OpenAI，可替换为本地）。记忆是 CrewAI 相对于较轻量框架的价值所在之一；纯 LangGraph 需要你自己接入每一个。

### CrewAI 适合的场景

- 三到六个具有命名角色和协作工作流的 agent。起草、审查、规划、头脑风暴。
- 路由中 LLM 对下一步的判断本身就是价值的一部分（Hierarchical）。
- 任何团队阅读 `role + goal + backstory` 比阅读图定义更开心的场景。

### CrewAI 不适合的场景

- 具有严格顺序的确定性 DAG。使用 LangGraph（第 13 课）。图的形状是正确的抽象；CrewAI 的角色框架是摩擦。
- 亚秒级延迟预算。Hierarchical 增加了往返。即使 Sequential 也会序列化包含 backstory 和先前输出的提示词。
- 单 agent 循环。跳过框架；一个 agent 循环（第 1 课）加上工具注册表更简短。

第 17 课（Agent 框架权衡）以矩阵形式呈现了这一点。简短版本：CrewAI 位于"基于角色的协作"角落。

### 依赖形态

独立于 LangChain。Python 3.10 到 3.13。使用 `uv`。Star 数：见 [crewAIInc/crewAI](https://github.com/crewAIInc/crewAI)（截至 2026-05 的快照）。AWS Bedrock 集成已文档化；供应商基准测试报告在 QA 工作负载上比 LangGraph 有显著加速，但方法论（数据集、硬件、评估指标）未公开，因此仅将框架供应商的数据视为方向性参考。

### 这种模式可能出错的地方

- **backstory 导致的提示词膨胀。** 每个 agent 2000 字的 backstory 和一个五 agent crew，在第一次工具调用之前就烧光了上下文预算。将 backstory 保持在 200 字以下。跨 agent 复用短语；不要把内部风格重复五次。
- **Manager-LLM token 开销。** Hierarchical 流程在每次专家调用之前增加一次管理者 LLM 调用。在五任务 crew 中，这是六次 LLM 调用而不是五次，且管理者调用携带完整的任务列表加上先前的输出。除非路由取决于输出，否则切换到 Sequential。
- **脆弱的交接。** 任务 N 的 `expected_output` 是"一个大纲"。任务 N+1 将其作为 `context` 读取，并试图解析三个章节。LLM 产出了四个。下游 Agent 即兴发挥。通过在任务 N 上使用 `output_pydantic` 修复，使任务 N+1 读取一个类型化对象，而不是自由文本。
- **Crew 直接上生产。** 自由形式的 Crew 在没有 Flow 包装的情况下交付到生产。输出变异性高；重放不可能；值班人员无法将坏的运行与好的运行进行 diff。用 Flow 包装。

## 构建它

`code/main.py` 实现了两种形态的标准库版本加上一个三 agent crew。

形态：

- `Agent`、`Task` 数据类，匹配 CrewAI 的接口。
- `SequentialCrew.kickoff(inputs)` 按声明顺序运行任务，将输出作为 `context` 传递。
- `HierarchicalCrew.kickoff(topic)` 增加一个管理者 Agent，每轮选择下一个专家，在"done"时停止。
- `Flow` 带有 `@start` 和 `@listen(topic)` 装饰器、一个小型事件循环和跟踪。
- `tool(name)` 装饰器，镜像 CrewAI 的 `@tool` 形态。
- `Memory` 带有 `short_term`、`long_term`、`entity` 存储；模拟相似度使用 numpy。
- 模拟 LLM 响应是根据角色加输入前缀键控的硬编码字符串。无网络。确定性。

具体演示：研究员、写手、编辑 crew 产出关于"2026 年 agent 工程"的简报。研究员拉取（模拟的）资料。写手起草。编辑精简。同一个 crew 通过 Flow 运行以展示确定性形态。

运行它：

```bash
python3 code/main.py
```

跟踪涵盖：sequential crew 通过 `context` 传递输出，hierarchical crew 带有管理者选择（研究员、写手、编辑，然后"done"），flow 以显式主题（`researched`、`drafted`、`edited`）运行相同的三个步骤，工具调用通过 `@tool` 路由，长期记忆在两次 kickoff 之间存活。

Crew 的跟踪是流动的；管理者原则上可以重新排序。Flow 的跟踪是固定的。这个选择就是本课的核心。

## 使用它

- **CrewAI Flow** 用于生产。即使 Flow 只有一个调用 `Crew.kickoff()` 的步骤。Flow 提供审计边界。
- **CrewAI Crew（Sequential）** 用于顺序明确的协作工作，特别是初稿和审查循环。
- **CrewAI Crew（Hierarchical）** 当路由取决于输出且你有四个或更多专家时。
- **LangGraph**（第 13 课）用于显式状态机、持久恢复、严格排序。
- **AutoGen v0.4**（第 14 课）用于 actor 模型并发和故障隔离。
- **OpenAI Agents SDK**（第 16 课）用于 OpenAI 优先的产品，带有交接和护栏。
- **Claude Agent SDK**（第 17 课）用于 Claude 优先的产品，带有子 agent 和会话存储。

## 交付它

`outputs/skill-crew-or-flow.md` 为任务选择 Crew 或 Flow，并构建最小实现。对于无 backstory 的 Crew、无显式主题的 Flow、少于三个专家的 Hierarchical 会硬性拒绝。

## 陷阱

- **Backstory 作为调味品。** 它塑造输出。为每个 agent 测试三种变体；差异是真实存在的。选择一种，冻结它。
- **跳过 `expected_output`。** 没有每个任务的契约，下游任务接收 LLM 产生的任何内容。Crew 运行了；审计失败了。
- **记忆始终开启。** 长期记忆每次运行都写入。向量数据库增长。检索变得嘈杂。将写入范围限定在信息是持久性的任务上。
- **管理者提示词漂移。** Hierarchical 的管理者提示词是隐式的。如果路由变得奇怪，在 verbose 模式下转储并阅读。
- **Crew 中的工具副作用。** 一个 Crew 可能比预期调用更多次工具。POST、DELETE、支付应放在 Flow 步骤中，永远不要放在 Crew 工具中。

## 练习

1. 将 Sequential crew 转换为 Flow。计算变异性下降的接触点。注意可读性下降的地方。
2. 向 crew 添加实体记忆：关于客户的信息在 kickoff 之间持久化。验证检索拉取了正确的实体。
3. 实现一个 Hierarchical 流程，管理者拒绝路由到编辑，直到写手的输出至少有三个段落。跟踪重试。
4. 为（模拟的）网页搜索接入一个 `BaseTool` 子类。比较跟踪形态与 `@tool` 装饰器版本的区别。
5. 向编辑任务添加 `output_pydantic=Brief`，其中 `Brief` 有 `title`、`summary`、`sections`。让写手任务输出一次格式错误的 JSON；在跟踪中验证 CrewAI 的重试行为。
6. 阅读 CrewAI 的文档简介。将玩具移植到真正的 `crewai` API。标准库版本跳过了哪些保证？
7. 将 AgentOps 或 Langfuse（第 24 课）接入到真实运行中。你在标准库版本中错过了哪些跟踪？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| Agent | "人设" | 角色 + 目标 + backstory + 工具 |
| Task | "工作单元" | 描述 + 期望输出 + 执行者 + 可选的结构化输出 |
| Crew | "Agent 团队" | Agent + Task + Process 的容器 |
| Process | "执行策略" | Sequential / Hierarchical / Consensus（计划中） |
| Flow | "确定性工作流" | 事件驱动的、代码拥有的、可测试的 |
| Backstory | "人设提示词" | 塑造 Agent 语气和判断力的内容 |
| `@tool` | "函数工具" | 将函数转换为 Agent 可调用工具的装饰器 |
| `BaseTool` | "类工具" | 基于类的工具，带有参数 schema、重试、异步支持 |
| 实体记忆（Entity memory） | "按实体的信息" | 限定于客户 / 账户 / 问题的记忆 |
| 长期记忆（Long-term memory） | "跨运行记忆" | 向量支持的记忆，在 kickoff 之间存活 |
| 上下文记忆（Contextual memory） | "即时检索" | 在 Agent 需要的那一刻拉取的记忆 |
| Manager LLM | "路由 agent" | Hierarchical 流程中选择下一个任务的额外 LLM |
| `expected_output` | "任务契约" | 告诉 Agent（和审计）返回什么形态的字符串 |

## 进一步阅读

- [CrewAI 文档简介](https://docs.crewai.com/en/introduction)：概念和推荐的生产路径
- [CrewAI Flow 指南](https://docs.crewai.com/en/concepts/flows)：事件驱动形态，`@start`，`@listen`
- [CrewAI 工具参考](https://docs.crewai.com/en/concepts/tools)：`@tool`，`BaseTool`，内置工具包
- [CrewAI 记忆](https://docs.crewai.com/en/concepts/memory)：短期、长期、实体、上下文
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)：多 agent 何时有帮助，何时没有
- [LangGraph 概览](https://docs.langchain.com/oss/python/langgraph/overview)：状态机替代方案
