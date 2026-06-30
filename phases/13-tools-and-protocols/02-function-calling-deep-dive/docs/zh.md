# Function Calling 深入剖析——OpenAI、Anthropic、Gemini

> 2024 年，三家前沿提供商在相同的工具调用循环上达成了一致，然后在其他所有方面产生了分歧。OpenAI 使用 `tools` 和 `tool_calls`。Anthropic 使用 `tool_use` 和 `tool_result` 块。Gemini 使用 `functionDeclarations` 和唯一 id 关联。本课并排比较三者的差异，使在一个提供商上发布的代码在移植时不会出问题。

**Type:** Build
**Languages:** Python (stdlib, schema translators)
**Prerequisites:** Phase 13 · 01 (the tool interface)
**Time:** ~75 minutes

## 学习目标

- 陈述 OpenAI、Anthropic 和 Gemini function-calling 载荷之间的三种形状差异（声明、调用、结果）。
- 将一个工具声明转换为所有三种提供商格式，并预测严格模式约束将在何处不同。
- 在每个提供商中使用 `tool_choice` 来强制、禁止或自动选择工具调用。
- 了解每个提供商的硬性限制（工具数量、schema 深度、参数长度）以及每个在违反限制时发出的错误签名。

## 问题

Function-calling 请求的形状因提供商而异。来自 2026 年生产技术栈的三个具体示例：

**OpenAI Chat Completions / Responses API。** 你传递 `tools: [{type: "function", function: {name, description, parameters, strict}}]`。模型的响应包含 `choices[0].message.tool_calls: [{id, type: "function", function: {name, arguments}}]`，其中 `arguments` 是一个你必须解析的 JSON 字符串。严格模式（`strict: true`）通过受限解码强制 schema 合规。

**Anthropic Messages API。** 你传递 `tools: [{name, description, input_schema}]`。响应返回为 `content: [{type: "text"}, {type: "tool_use", id, name, input}]`。`input` 已经解析（是一个对象，不是字符串）。你以包含 `{type: "tool_result", tool_use_id, content}` 块的新 `user` 消息回复。

**Google Gemini API。** 你传递 `tools: [{functionDeclarations: [{name, description, parameters}]}]`（嵌套在 `functionDeclarations` 下）。响应到达为 `candidates[0].content.parts: [{functionCall: {name, args, id}}]`，其中 `id` 在 Gemini 3 及以上版本中是唯一的，用于并行调用关联。你以 `{functionResponse: {name, id, response}}` 回复。

相同的循环。不同的字段名称、不同的嵌套、不同的字符串-vs-对象约定、不同的关联机制。一个在 OpenAI 上编写天气代理的团队，要花两天移植到 Anthropic，再加一天移植到 Gemini，仅仅为了管道工作。

本课构建一个翻译器，将三种格式统一为一个规范的工具声明，并在边缘进行路由。Phase 13 · 17 将相同的模式泛化为 LLM 网关。

## 概念

### 共同结构

每个提供商需要五样东西：

1. **工具列表。** 每个工具的名称、描述和输入 schema。
2. **工具选择。** 强制使用特定工具、禁止工具或让模型决定。
3. **调用发出。** 指定工具和参数的结构化输出。
4. **调用 id。** 将响应关联到正确的调用（对并行很重要）。
5. **结果注入。** 将结果绑定回调用的消息或块。

### 形状差异，逐字段

| 方面 | OpenAI | Anthropic | Gemini |
|--------|--------|-----------|--------|
| 声明信封 | `{type: "function", function: {...}}` | `{name, description, input_schema}` | `{functionDeclarations: [{...}]}` |
| Schema 字段 | `parameters` | `input_schema` | `parameters` |
| 响应容器 | 助手消息上的 `tool_calls[]` | 类型 `tool_use` 的 `content[]` | 类型 `functionCall` 的 `parts[]` |
| 参数类型 | stringified JSON | 已解析的对象 | 已解析的对象 |
| Id 格式 | `call_...`（OpenAI 生成） | `toolu_...`（Anthropic） | UUID（Gemini 3+） |
| 结果块 | 角色 `tool`，`tool_call_id` | `user` 带 `tool_result`，`tool_use_id` | `functionResponse` 带匹配的 `id` |
| 强制工具 | `tool_choice: {type: "function", function: {name}}` | `tool_choice: {type: "tool", name}` | `tool_config: {function_calling_config: {mode: "ANY"}}` |
| 禁止工具 | `tool_choice: "none"` | `tool_choice: {type: "none"}` | `mode: "NONE"` |
| 严格 schema | `strict: true` | schema-即-schema（始终强制） | 请求级别的 `responseSchema` |

### 你实际会碰到的限制

- **OpenAI。** 每次请求 128 个工具。Schema 深度 5。参数字符串 <= 8192 字节。严格模式要求无 `$ref`，无重叠的 `oneOf`/`anyOf`/`allOf`，每个属性列在 `required` 中。
- **Anthropic。** 每次请求 64 个工具。Schema 深度实际上无界但实际限制为 10。无严格模式标志；schema 是契约，模型倾向于遵守。
- **Gemini。** 每次请求 64 个函数。Schema 类型是 OpenAPI 3.0 子集（与 JSON Schema 2020-12 有轻微差异）。自 Gemini 3 起并行调用使用唯一 id。

### `tool_choice` 行为

三种每个人都支持的模式，但命名不同。

- **自动。** 模型选择工具或文本。默认。
- **必需 / 任意。** 模型必须调用至少一个工具。
- **无。** 模型不得调用工具。

外加每个提供商独有的一个模式：

- **OpenAI。** 按名称强制特定工具。
- **Anthropic。** 按名称强制特定工具；`disable_parallel_tool_use` 标志分离单个 vs 多个。
- **Gemini。** `mode: "VALIDATED"` 将每个响应路由通过 schema 验证器，无论模型意图如何。

### 并行调用

OpenAI 的 `parallel_tool_calls: true`（默认）在一个助手消息中发出多个调用。你全部运行它们，并用包含每个 `tool_call_id` 一个条目的批量 tool-role 消息回复。Anthropic 历史上是单调用；`disable_parallel_tool_use: false`（自 Claude 3.5 起默认）启用多调用。Gemini 2 允许并行调用但不提供稳定的 id；Gemini 3 添加了 UUID 以便乱序响应可以干净地关联。

### 流式传输

三者都支持流式工具调用。线格式不同：

- **OpenAI。** `tool_calls[i].function.arguments` 的增量块逐步到达。你累积直到 `finish_reason: "tool_calls"`。
- **Anthropic。** block-start / block-delta / block-stop 事件。`input_json_delta` 块携带部分参数。
- **Gemini。** `streamFunctionCallArguments`（Gemini 3 新增）携带一个 `functionCallId` 的块，因此多个并行调用可以交错。

Phase 13 · 03 深入探讨并行 + 流式重组。本课重点在声明和单调用形状。

### 错误与修复

无效参数错误看起来也不同。

- **OpenAI（非严格）。** 模型返回 `arguments: "{bad json}"`，你的 JSON 解析失败，你注入错误消息并重新调用。
- **OpenAI（严格）。** 验证发生在解码期间；无效 JSON 不可能出现，但 `refusal` 可能出现。
- **Anthropic。** `input` 可能包含意外字段；schema 是建议性的。在服务端验证。
- **Gemini。** OpenAPI 3.0 怪癖：对象字段上的 `enum` 被静默忽略；自行验证。

### 翻译器模式

代码中的规范工具声明看起来像这样（你选择形状）：

```python
Tool(
    name="get_weather",
    description="Use when ...",
    input_schema={"type": "object", "properties": {...}, "required": [...]},
    strict=True,
)
```

三个小型函数将其转换为三种提供商的形状。`code/main.py` 中的框架正是这样做的，然后通过每个提供商的响应形状对假工具调用进行往返。不需要网络——本课教授的是形状，而非 HTTP。

生产团队将此翻译器包装在 `AbstractToolset`（Pydantic AI）、`UniversalToolNode`（LangGraph）或 `BaseTool`（LlamaIndex）中。Phase 13 · 17 发布一个在三者任一前面暴露 OpenAI 形状 API 的网关。

## 使用

`code/main.py` 定义了一个规范的 `Tool` dataclass 和三个翻译器，发出 OpenAI、Anthropic 和 Gemini 的声明 JSON。然后它将每种形状的手工制作提供商响应解析为相同的规范调用对象，展示了语义在表面之下是相同的。运行它并并排比较三种声明。

需要注意的地方：

- 三种声明块仅在外层信封和字段名称上不同。
- 三种响应块在调用的位置（顶层 `tool_calls`、`content[]` 块、`parts[]` 条目）上不同。
- 一个 `canonical_call()` 函数从所有三种响应形状中提取 `{id, name, args}`。

## 产出

本课产出 `outputs/skill-provider-portability-audit.md`。给定一个针对一个提供商的 function-calling 集成，该技能产生一份可移植性审计：它依赖哪些提供商限制，哪些字段需要重命名，以及在移植到其他每个提供商时会出什么问题。

## 练习

1. 运行 `code/main.py` 并验证三种提供商声明 JSON 都序列化了相同的底层 `Tool` 对象。修改规范工具添加一个 enum 参数，并确认只有 Gemini 翻译器需要处理 OpenAPI 怪癖。

2. 为每个提供商添加一个 `ListToolsResponse` 解析器，提取模型在 `list_tools` 或发现调用后返回的工具列表。OpenAI 原生没有；注意这种不对称。

3. 实现 `tool_choice` 转换：将规范 `ToolChoice(mode="force", tool_name="x")` 映射为所有三种提供商形状。然后映射 `mode="any"` 和 `mode="none"`。检查本课的差异表。

4. 选择三家提供商之一，从头到尾阅读其 function-calling 指南。找出其 schema 规范中其他两家不支持的字段。候选项：OpenAI `strict`、Anthropic `disable_parallel_tool_use`、Gemini `function_calling_config.allowed_function_names`。

5. 编写一个测试向量：一个参数违反声明 schema 的工具调用。通过每个提供商的验证器运行它（第 01 课中的标准库实现即可作为代理），并记录触发了哪些错误。记录你会将哪个提供商用于生产级严格性。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Function calling | "工具使用" | 用于结构化工具调用发出的提供商级 API |
| 工具声明 | "工具规格" | 名称 + 描述 + JSON Schema 输入载荷 |
| `tool_choice` | "强制 / 禁止" | 自动 / 必需 / 无 / 特定名称 模式 |
| 严格模式 | "Schema 强制" | 限制解码以匹配 schema 的 OpenAI 标志 |
| `tool_use` 块 | "Anthropic 的调用形状" | 带 id、name、input 的内联内容块 |
| `functionCall` 部件 | "Gemini 的调用形状" | 包含 name、args 和 id 的 `parts[]` 条目 |
| 参数为字符串 | "Stringified JSON" | OpenAI 将 args 作为 JSON 字符串返回，而非对象 |
| 并行工具调用 | "一个回合中扇出" | 一个助手消息中的多个工具调用 |
| 拒绝 | "模型拒绝" | 仅严格模式下，拒绝块而非调用 |
| OpenAPI 3.0 子集 | "Gemini schema 怪癖" | Gemini 使用类似 JSON Schema 但有细微差异的方言 |

## 拓展阅读

- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling) —— 权威参考，包含严格模式和并行调用
- [Anthropic — Tool use overview](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) —— `tool_use` 和 `tool_result` 块语义
- [Google — Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling) —— 并行调用、唯一 id 和 OpenAPI 子集
- [Vertex AI — Function calling reference](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling) —— Gemini 的企业接口
- [OpenAI — Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) —— 严格模式 schema 强制细节
