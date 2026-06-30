# OpenTelemetry GenAI 语义约定

> OpenTelemetry 的 GenAI SIG（2024 年 4 月启动）定义了 Agent 遥测的标准模式。Span 名称、属性和内容捕获规则在各厂商间统一，使得 Agent 追踪信息在 Datadog、Grafana、Jaeger 和 Honeycomb 中含义一致。

**类型：** 学习 + 构建
**语言：** Python（标准库）
**前置条件：** 第 14 阶段 · 13（LangGraph）、第 14 阶段 · 24（可观测性平台）
**时间：** 约 60 分钟

## 学习目标

- 说出 GenAI span 类别：model/client、agent、tool。
- 区分 `invoke_agent` 的 CLIENT 与 INTERNAL span 以及各自适用场景。
- 列出顶层 GenAI 属性：provider name、request model、data-source ID。
- 解释内容捕获协议：opt-in、`OTEL_SEMCONV_STABILITY_OPT_IN`、推荐外部引用。

## 问题

每个厂商都发明自己的 span 命名。运维团队最终要为每个框架构建独立的仪表盘。OpenTelemetry 的 GenAI SIG 通过定义一个整个生态系统都遵循的标准来解决此问题。

## 概念

### Span 类别

1. **Model / client span。** 覆盖原始 LLM 调用。由厂商 SDK（Anthropic、OpenAI、Bedrock）和框架模型适配器发出。
2. **Agent span。** `create_agent`（Agent 构造时）和 `invoke_agent`（Agent 运行时）。
3. **Tool span。** 每次工具调用对应一个 span；通过父子关系连接到 agent span。

### Agent span 命名

- Span 名称：`invoke_agent {gen_ai.agent.name}`（如果有名称）；回退到 `invoke_agent`。
- Span 类型：
  - **CLIENT** — 用于远程 agent 服务（OpenAI Assistants API、Bedrock Agents）。
  - **INTERNAL** — 用于进程内 agent 框架（LangChain、CrewAI、本地 ReAct）。

### 关键属性

- `gen_ai.provider.name` — `anthropic`、`openai`、`aws.bedrock`、`google.vertex`。
- `gen_ai.request.model` — 模型 ID。
- `gen_ai.response.model` — 实际解析的模型（可能因路由而与请求不同）。
- `gen_ai.agent.name` — Agent 标识符。
- `gen_ai.operation.name` — `chat`、`completion`、`invoke_agent`、`tool_call`。
- `gen_ai.data_source.id` — 用于 RAG：查询了哪个语料库或存储。

对于 Anthropic、Azure AI Inference、AWS Bedrock、OpenAI 存在特定技术约定。

### 内容捕获

默认规则：检测工具默认不应捕获输入/输出。通过以下方式 opt-in 捕获：

- `gen_ai.system_instructions`
- `gen_ai.input.messages`
- `gen_ai.output.messages`

推荐的生产模式：将内容存储在外部（S3、你的日志存储），在 span 上记录引用（指针 ID，而非原文）。这就是第 27 课的内容投毒防御接入可观测性系统的做法。

### 稳定性

截至 2026 年 3 月，大多数约定仍处于实验阶段。通过以下方式选择使用稳定预览版：

```
OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
```

Datadog v1.37+ 将 GenAI 属性原生映射到其 LLM Observability 模式。其他后端（Grafana、Honeycomb、Jaeger）支持原始属性。

### 此模式的常见错误

- **在 span 中捕获完整提示。** PII、密钥、客户数据出现在运维人员可读的追踪中。存储在外部。
- **缺少 `gen_ai.provider.name`。** 多供应商仪表盘在缺少归属时会失效。
- **span 缺少父链接。** 孤立的 tool span。始终传播上下文。
- **未设置稳定性 opt-in。** 后端升级后你的属性可能被重命名。

## 构建它

`code/main.py` 实现了符合 GenAI 约定的标准库 span 发射器：

- 包含 GenAI 属性模式的 `Span`。
- 带有 `start_span`、嵌套上下文的 `Tracer`。
- 一次脚本化的 agent 运行，发出：`create_agent`、`invoke_agent`（INTERNAL）、每个工具 span、LLM 调用的 `chat` span。
- 一个内容捕获模式，将提示存储在外部并在 span 上记录 ID。

运行它：

```
python3 code/main.py
```

输出：一棵包含所有必需 GenAI 属性的 span 树，以及显示 opt-in 内容引用的"外部存储"。

## 使用它

- **Datadog LLM Observability**（v1.37+）原生映射属性。
- **Langfuse / Phoenix / Opik**（第 24 课）— 自动检测整个生态系统。
- **Jaeger / Honeycomb / Grafana Tempo** — 原始 OTel 追踪；基于 GenAI 属性构建仪表盘。
- **自托管** — 运行带有 GenAI 处理器的 OTel Collector。

## 交付它

`outputs/skill-otel-genai.md` 将 OTel GenAI span 接入现有 agent，包含内容捕获默认值和外部引用存储。

## 练习

1. 用 `invoke_agent`（INTERNAL）+ 每个工具 span 对你的第 01 课 ReAct 循环进行检测。发送到 Jaeger 实例。
2. 以"仅引用"模式添加内容捕获：提示存入 SQLite，span 属性仅携带行 ID。
3. 阅读 `gen_ai.data_source.id` 的规范。将其接入你的第 09 课 Mem0 搜索。
4. 设置 `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` 并验证你的属性不会被 collector 重命名。
5. 构建一个仪表盘：仅凭 GenAI 属性显示"哪些工具错误与哪些模型相关"。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| GenAI SIG | "OpenTelemetry GenAI 小组" | 定义该模式的 OTel 工作组 |
| invoke_agent | "Agent span" | 表示一次 agent 运行的 span 名称 |
| CLIENT span | "远程调用" | 调用远程 agent 服务的 span |
| INTERNAL span | "进程内" | 进程内 agent 运行的 span |
| gen_ai.provider.name | "提供商" | anthropic / openai / aws.bedrock / google.vertex |
| gen_ai.data_source.id | "RAG 来源" | 检索命中了哪个语料库/存储 |
| 内容捕获 | "提示日志记录" | 选择加入的消息捕获；生产环境存储在外部 |
| 稳定性 opt-in | "预览模式" | 用于锁定实验性约定的环境变量 |

## 扩展阅读

- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 规范本身
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — 默认包含 GenAI span
- [AutoGen v0.4（微软研究院）](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — 内置 OTel span
- [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) — W3C 追踪上下文传播
