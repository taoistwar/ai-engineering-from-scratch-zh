# OpenTelemetry GenAI — 端到端追踪工具调用

> 一个代理调用五个工具、三个 MCP 服务器和两个子代理。你需要一条追踪线贯穿全部。OpenTelemetry GenAI 语义约定（v1.37 及以上版本中的稳定属性）是 2026 年的标准，由 Datadog、Langfuse、Arize Phoenix、OpenLLMetry 和 AgentOps 原生支持。本课程命名所需属性，演练 span 层次结构（agent → LLM → tool），并提供一个可插入任何 OTel 导出器的标准库 span 发射器。

**Type:** Build
**Languages:** Python（stdlib，OTel span 发射器）
**Prerequisites:** Phase 13 · 07（MCP 服务器），Phase 13 · 08（MCP 客户端）
**Time:** ~75 分钟

## 学习目标

- 为 LLM span 和工具执行 span 命名所需的 OTel GenAI 属性。
- 构建一个覆盖代理循环、LLM 调用、工具调用和 MCP 客户端分发的追踪层次结构。
- 决定捕获什么内容（选择性加入）vs 编辑什么内容（默认）。
- 将 span 发射到本地采集器（Jaeger、Langfuse）而无需重写工具代码。

## 问题

一个来自 2026 年 2 月的调试记录：用户报告"我的代理有时需要 30 秒响应；其他时候 3 秒。"没有追踪数据。日志显示 LLM 调用，但未显示工具分发、MCP 服务器往返、子代理。你猜测。最终你发现：一个 MCP 服务器偶尔在冷启动时挂起。

没有端到端追踪，你无法找到这个问题。OTel GenAI 解决了它。

这些约定在 2025-2026 年于 OpenTelemetry semantic-conventions 组下稳定下来。它们定义了稳定的属性名称，使 Datadog、Langfuse、Phoenix、OpenLLMetry 和 AgentOps 都能解析相同的 span。仅需插桩一次；即可发送到任何后端。

## 概念

### Span 层次结构

```
agent.invoke_agent  （顶层，INTERNAL span）
 ├── llm.chat       （CLIENT span）
 ├── tool.execute   （INTERNAL）
 │    └── mcp.call  （CLIENT span）
 ├── llm.chat       （CLIENT span）
 └── subagent.invoke （INTERNAL）
```

整个链路嵌套在同一个 trace id 下。Span id 链接父子关系。

### 必需属性

根据 2025-2026 年语义约定：

- `gen_ai.operation.name` — `"chat"`、`"text_completion"`、`"embeddings"`、`"execute_tool"`、`"invoke_agent"`。
- `gen_ai.provider.name` — `"openai"`、`"anthropic"`、`"google"`、`"azure_openai"`。
- `gen_ai.request.model` — 请求的模型字符串（例如 `"gpt-4o-2024-08-06"`）。
- `gen_ai.response.model` — 实际服务的模型。
- `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens`。
- `gen_ai.response.id` — 用于关联的提供者响应 ID。

工具 span：

- `gen_ai.tool.name` — 工具标识符。
- `gen_ai.tool.call.id` — 特定的调用 ID。
- `gen_ai.tool.description` — 工具描述（可选）。

代理 span：

- `gen_ai.agent.name` / `gen_ai.agent.id` / `gen_ai.agent.description`。

### Span 类型

- `SpanKind.CLIENT` 用于跨越进程边界的调用（LLM 提供者、MCP 服务器）。
- `SpanKind.INTERNAL` 用于代理自己的循环步骤和工具执行。

### 选择性加入的内容捕获

默认情况下，span 携带指标和计时信息——而不是提示词或补全内容。大型负载和 PII 默认关闭。设置 `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` 和特定的内容捕获环境变量以包含内容。在生产中启用前请仔细审查。

### Span 上的事件

令牌级事件可以作为 span 事件添加：

- `gen_ai.content.prompt` — 输入消息。
- `gen_ai.content.completion` — 输出消息。
- `gen_ai.content.tool_call` — 记录的工具调用。

事件在 span 内按时间排序以进行详细回放。

### 导出器

OTel span 导出到：

- **Jaeger / Tempo。** OSS，本地部署。
- **Langfuse。** LLM 专用可观测性；可视化令牌使用情况。
- **Arize Phoenix。** 评估 + 追踪一体化。
- **Datadog。** 商业；原生解析 `gen_ai.*` 属性。
- **Honeycomb。** 列导向；查询友好。

所有都使用 OTLP（线格式）。你的代码无关紧要。

### 跨 MCP 传播

当 MCP 客户端调用服务器时，将 W3C traceparent 头部注入请求。Streamable HTTP 支持标准头部。Stdio 本身不携带 HTTP 头部；规范的 2026 年路线图讨论了在 JSON-RPC 调用中添加 `_meta.traceparent` 字段。

在该功能发布之前：手动在每个请求的 `_meta` 中包含 traceparent。服务器记录 trace id。

### 指标

与 span 并列，GenAI 语义约定定义了指标：

- `gen_ai.client.token.usage` — 直方图。
- `gen_ai.client.operation.duration` — 直方图。
- `gen_ai.tool.execution.duration` — 直方图。

将这些用于不需要每次调用详细信息的仪表盘。

### AgentOps 层

AgentOps（成立于 2024 年）专注于 GenAI 可观测性。它包装了流行框架（LangGraph、Pydantic AI、CrewAI）以自动发出 OTel span。如果你的技术栈使用支持的框架，则很有用；否则使用手工插桩。

## 使用

`code/main.py` 以 OTel 形式的 span 输出到 stdout（类似 OTLP-JSON 的格式），用于一个调用 LLM、分发两个工具并进行一次 MCP 往返的代理。不涉及实际的导出器——课程侧重于 span 形态和属性集。将输出粘贴到 OTLP 兼容的查看器中或直接阅读。

关注要点：

- Trace id 在所有 span 中共享。
- 父子链接通过 `parentSpanId` 编码。
- 所需 `gen_ai.* ` 属性已填充。
- 内容捕获默认关闭；一个场景通过环境变量启用它。

## 交付物

本课程产出 `outputs/skill-otel-genai-instrumentation.md`。给定一个代理代码库，该技能生成一个插桩计划：在哪里添加 span、填充哪些属性以及瞄准哪些导出器。

## 练习

1. 运行 `code/main.py`。计数 span 并识别哪个是 CLIENT vs INTERNAL。

2. 开启内容捕获（环境变量）并确认 `gen_ai.content.prompt` 和 `gen_ai.content.completion` 事件出现。注意 PII 的影响。

3. 添加工具执行指标 `gen_ai.tool.execution.duration`，并将其作为每次调用的直方图样本发出。

4. 将一个 traceparent 从父代理 span 传播到 MCP 请求的 `_meta.traceparent` 字段。验证 MCP 服务器将看到相同的 trace id。

5. 阅读 OTel GenAI 语义约定规范。找出规范中列出但本课程代码未发出的一个属性。添加它。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|----------|
| OTel | "OpenTelemetry" | 追踪、指标、日志的开放标准 |
| GenAI 语义约定 | "GenAI 语义约定" | LLM / 工具 / 代理 span 的稳定属性名称 |
| `gen_ai.*` | "属性命名空间" | 所有 GenAI 属性共享此前缀 |
| Span | "计时操作" | 带有开始、结束和属性的工作单元 |
| Trace | "跨 span 的谱系" | 共享一个 trace id 的 span 树 |
| SpanKind | "CLIENT / SERVER / INTERNAL" | span 方向的提示 |
| OTLP | "OpenTelemetry 线协议" | 导出器的线格式 |
| 选择性加入内容 | "提示词 / 补全捕获" | 默认关闭；通过环境变量启用 |
| traceparent | "W3C 头部" | 跨服务的追踪上下文传播 |
| 导出器 | "后端特定的发送器" | 将 span 发送到 Jaeger / Datadog / 等的组件 |

## 进一步阅读

- [OpenTelemetry — GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — GenAI span、指标和事件的规范约定
- [OpenTelemetry — GenAI span](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/) — LLM 和工具执行 span 属性列表
- [OpenTelemetry — GenAI 代理 span](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/) — 代理级 `invoke_agent` span
- [open-telemetry/semantic-conventions — GenAI span](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md) — GitHub 托管的信源
- [Datadog — LLM OTel 语义约定](https://www.datadoghq.com/blog/llm-otel-semantic-convention/) — 生产集成演练
