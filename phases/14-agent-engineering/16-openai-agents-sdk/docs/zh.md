# OpenAI Agents SDK：交接、护栏、追踪

> OpenAI Agents SDK 是建立在 Responses API 之上的轻量级多 agent 框架。五个原语：Agent、Handoff、Guardrail、Session、Tracing。交接是名为 `transfer_to_<agent>` 的工具。护栏在输入或输出时触发。追踪默认开启。

**类型：** 学习 + 构建
**语言：** Python（标准库）
**前置条件：** 第 14 阶段 · 01（Agent 循环），第 14 阶段 · 06（工具使用）
**时间：** ~75 分钟

## 学习目标

- 列举 OpenAI Agents SDK 的五个原语。
- 解释交接：为什么它们被建模为工具、模型看到的名称形态以及上下文如何传输。
- 区分输入护栏、输出护栏和工具护栏；解释 `run_in_parallel` 与阻塞模式。
- 实现一个具有交接 + 护栏 + span 风格追踪的标准库运行时。

## 问题

无法干净委派的 agent 最终将所有内容塞进一个提示词。没有护栏的 agent 会泄露 PII、违反策略的输出或无限循环。OpenAI 的 SDK 将三个使多 agent 工作可驾驭的原语编码化了。

## 概念

### 五个原语

1. **Agent。** LLM + 指令 + 工具 + 交接。
2. **Handoff。** 委派给另一个 agent。对模型表示为名为 `transfer_to_<agent_name>` 的工具。
3. **Guardrail。** 对输入（仅第一个 agent）、输出（仅最后一个 agent）或工具调用（每个函数工具）的验证。
4. **Session。** 跨轮次的自动对话历史。
5. **Tracing。** 内置 span，覆盖 LLM 生成、工具调用、交接、护栏。

### 交接即工具

模型在其工具列表中看到 `transfer_to_billing_agent`。调用它告诉运行时：

1. 复制对话上下文（或通过 `nest_handoff_history` beta 折叠它）。
2. 用其指令初始化目标 agent。
3. 继续使用目标 agent 运行。

这是监督者模式（第 13 课 / 第 28 课）的产品化。

### 护栏

三种形式：

- **输入护栏。** 在第一个 agent 的输入上运行。在任何 LLM 调用之前拒绝不安全或超出范围的请求。
- **输出护栏。** 在最后一个 agent 的输出上运行。捕获 PII 泄露、策略违规、格式错误的响应。
- **工具护栏。** 按函数工具运行。验证参数、检查权限、审计执行。

模式：

- **并行**（默认）。护栏 LLM 与主 LLM 并行运行。尾部延迟更低。如果触发，主 LLM 的工作被丢弃（token 浪费）。
- **阻塞**（`run_in_parallel=False`）。护栏 LLM 先运行。如果触发，主调用不浪费任何 token。

触发引发 `InputGuardrailTripwireTriggered` / `OutputGuardrailTripwireTriggered`。

### 追踪

默认开启。每个 LLM 生成、工具调用、交接和护栏发出一个 span。`OPENAI_AGENTS_DISABLE_TRACING=1` 可退出。`add_trace_processor(processor)` 将 span 扇出到你自己的后端，与 OpenAI 的一起。

### 会话

`Session` 在后端（SQLite、Redis、自定义）存储对话历史。`Runner.run(agent, input, session=session)` 自动加载和追加。

### 这种模式可能出错的地方

- **交接漂移。** Agent A 交接给 Agent B，Agent B 又交接回 Agent A。添加跳数计数器。
- **护栏绕过。** 工具护栏仅在函数工具上触发；内置工具（文件读取器、网页抓取）需要单独的策略。
- **过度追踪。** Span 中的敏感内容。与 OTel GenAI 内容捕获规则（第 23 课）配对——存储到外部，按 ID 引用。

## 构建它

`code/main.py` 在标准库中实现了 SDK 形态：

- `Agent`、`FunctionTool`、`Handoff`（作为具有传输语义的函数工具）。
- `Runner` 具有输入/输出/工具护栏、交接调度和跳数计数器。
- 一个简单的 span 发出器以显示追踪形态。
- 一个分诊 agent，根据用户查询交接给 billing 或 support；护栏在一个输入上触发。

运行它：

```
python3 code/main.py
```

跟踪显示两次成功的交接、一次输入护栏触发，以及镜像真实 SDK 发出内容的 span 树。

## 使用它

- **OpenAI Agents SDK** 用于 OpenAI 优先的产品。
- **Claude Agent SDK**（第 17 课）用于 Claude 优先的产品。
- **LangGraph**（第 13 课）当你想要显式状态和持久恢复时。
- **自定义** 当你需要精确控制时（语音、多提供者、联邦部署）。

## 交付它

`outputs/skill-agents-sdk-scaffold.md` 构建一个 Agents SDK 应用，包含分诊 agent、交接、输入/输出/工具护栏、会话存储和追踪处理器。

## 练习

1. 添加交接跳数计数器：在 N 次传输后拒绝。追踪行为。
2. 实现 `nest_handoff_history` 作为选项——在传输前将之前的消息折叠为一个摘要。
3. 编写一个阻塞输出护栏。比较会触发它的提示词与通过的提示词的延迟。
4. 将 `add_trace_processor` 接入 JSON 日志记录器。它每个 span 发出什么形态？
5. 阅读 SDK 文档。将你的标准库玩具移植到 `openai-agents-python`。你建模错什么了？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| Agent | "LLM + 指令" | SDK 中的 Agent 类型；拥有工具和交接 |
| Handoff | "传输" | 模型调用以委派给另一个 agent 的工具 |
| Guardrail | "策略检查" | 对输入 / 输出 / 工具调用的验证 |
| Tripwire | "护栏触发" | 护栏拒绝时引发的异常 |
| Session | "历史存储" | 在运行之间持久化的对话记忆 |
| Tracing | "Span" | 覆盖 LLM + 工具 + 交接 + 护栏的内置可观测性 |
| 阻塞护栏（Blocking guardrail） | "顺序检查" | 护栏先运行；触发时不浪费 token |
| 并行护栏（Parallel guardrail） | "并发检查" | 护栏并行运行；延迟更低，触发时浪费 token |

## 进一步阅读

- [OpenAI Agents SDK 文档](https://openai.github.io/openai-agents-python/) — 原语、交接、护栏、追踪
- [Claude Agent SDK 概览](https://platform.claude.com/docs/en/agent-sdk/overview) — Claude 风格的对应物
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — 何时应该使用交接
- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — Agents SDK span 映射到的标准
