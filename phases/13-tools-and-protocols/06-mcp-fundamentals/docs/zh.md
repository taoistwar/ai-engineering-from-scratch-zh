# MCP 基础——原语、生命周期、JSON-RPC 基础

> MCP 之前的每个集成都是一次性的。Model Context Protocol，最早由 Anthropic 于 2024 年 11 月发布，现在由 Linux 基金会的 Agentic AI Foundation 管理，将发现和调用标准化，使任何客户端都可以与任何服务器对话。2025-11-25 规范命名了六个原语（三个服务器端，三个客户端）、一个三阶段生命周期和 JSON-RPC 2.0 线格式。学习这些，本阶段 MCP 章节的其余内容就变成了阅读理解。

**Type:** Learn
**Languages:** Python (stdlib, JSON-RPC parser)
**Prerequisites:** Phase 13 · 01 through 05 (the tool interface and function calling)
**Time:** ~45 minutes

## 学习目标

- 命名所有六个 MCP 原语（服务器端的 tools、resources、prompts；客户端的 roots、sampling、elicitation），并为每个给出一个用例。
- 走过三阶段生命周期（initialize、operation、shutdown），并说明在每个阶段谁发送哪条消息。
- 解析和发出 JSON-RPC 2.0 请求、响应和通知信封。
- 解释 `initialize` 时的能力协商是什么，以及没有它会出什么问题。

## 问题

在 MCP 之前，每个使用工具的代理都有自己的协议。Cursor 有一个 MCP 形状但不兼容的工具系统。Claude Desktop 发布了一个不同的系统。VS Code 的 Copilot 扩展有第三个。一个构建 "Postgres query" 工具的团队将同一个工具编写了三次，每次适配不同的宿主 API。复用它需要复制代码。

结果是一次性集成的大爆发和生态系统速度的上限。

MCP 通过标准化线格式解决了这个问题。一个单一的 MCP 服务器可以在每个 MCP 客户端中工作：Claude Desktop、ChatGPT、Cursor、VS Code、Gemini、Goose、Zed、Windsurf，到 2026 年 4 月已有 300+ 客户端。每月 1.1 亿次 SDK 下载。10,000+ 公共服务器。Linux 基金会在 2025 年 12 月在新成立的 Agentic AI Foundation 下接管了管理工作。

本阶段使用的规范修订是 **2025-11-25**。它添加了异步 Tasks（SEP-1686）、URL 模式 elicitation（SEP-1036）、带工具的 sampling（SEP-1577）、增量范围同意（SEP-835）和 OAuth 2.1 资源指示符语义。Phase 13 · 09 至 16 涵盖这些扩展。本课停在基础层面。

## 概念

### 三个服务器原语

1. **Tools。** 可调用动作。与 Phase 13 · 01 相同的四步循环。
2. **Resources。** 暴露的数据。可通过 URI 寻址的只读内容：`file:///path`、`db://query/...`、自定义方案。
3. **Prompts。** 可重用模板。宿主 UI 中的斜杠命令；服务器提供模板，客户端填充参数。

### 三个客户端原语

4. **Roots。** 服务器允许接触的 URI 集合。客户端声明它们；服务器尊重它们。
5. **Sampling。** 服务器请求客户端的模型执行一个补全。支持无服务端 API 密钥的服务端托管的代理循环。
6. **Elicitation。** 服务器在中间步骤向客户端用户请求结构化输入。表单或 URL（SEP-1036）。

MCP 中的每个能力恰好属于这六个之一。Phase 13 · 10 至 14 深入介绍每一个。

### 线格式：JSON-RPC 2.0

每条消息是一个带有这些字段的 JSON 对象：

- 请求：`{jsonrpc: "2.0", id, method, params}`。
- 响应：`{jsonrpc: "2.0", id, result | error}`。
- 通知：`{jsonrpc: "2.0", method, params}`——没有 `id`，不期望响应。

基础规范有约 15 个方法，按原语分组。重要的有：

- `initialize` / `initialized`（握手）
- `tools/list`、`tools/call`
- `resources/list`、`resources/read`、`resources/subscribe`
- `prompts/list`、`prompts/get`
- `sampling/createMessage`（服务器到客户端）
- `notifications/tools/list_changed`、`notifications/resources/updated`、`notifications/progress`

### 三阶段生命周期

**阶段 1：初始化。**

客户端发送 `initialize`，包含其 `capabilities` 和 `clientInfo`。服务器以其自己的 `capabilities`、`serverInfo` 和它所使用的规范版本响应。客户端在消化了响应后发送 `notifications/initialized`。从此处开始，任一侧都可以根据协商的能力发送请求。

**阶段 2：运行。**

双向。客户端调用 `tools/list` 进行发现，然后 `tools/call` 进行调用。如果服务器声明了能力，可以发送 `sampling/createMessage`。当工具集发生变化时，服务器可以发送 `notifications/tools/list_changed`。当用户更改根范围时，客户端可以发送 `notifications/roots/list_changed`。

**阶段 3：关闭。**

任一侧关闭传输。MCP 中没有结构化的关闭方法；传输（stdio 或 Streamable HTTP，Phase 13 · 09）承载连接结束的信号。

### 能力协商

`initialize` 握手时的 `capabilities` 是契约。来自服务器的示例：

```json
{
  "tools": {"listChanged": true},
  "resources": {"subscribe": true, "listChanged": true},
  "prompts": {"listChanged": true}
}
```

服务器声明它可以发出 `tools/list_changed` 通知并支持 `resources/subscribe`。客户端通过声明自己的内容来同意：

```json
{
  "roots": {"listChanged": true},
  "sampling": {},
  "elicitation": {}
}
```

如果客户端没有声明 `sampling`，服务器不得调用 `sampling/createMessage`。对称地：如果服务器没有声明 `resources.subscribe`，客户端不得尝试订阅。

这就是防止生态系统漂移的原因。一个不支持 sampling 的客户端仍是一个有效的 MCP 客户端；一个不调用 `sampling` 的服务器仍是一个有效的 MCP 服务器。它们只是一起不使用该特性。

### 结构化内容与错误形状

`tools/call` 返回一个类型化块的 `content` 数组：`text`、`image`、`resource`。Phase 13 · 14 向该列表添加了 MCP Apps（`ui://` 交互式 UI）。

错误使用 JSON-RPC 错误码。规范定义的额外内容：`-32002` "资源未找到"，`-32603` "内部错误"，加上 MCP 特定的错误数据作为 `error.data`。

### 客户端能力 vs 工具调用细节

一个常见混淆：`capabilities.tools` 是客户端是否支持工具列表变更通知。客户端是否会调用特定工具是是由其模型驱动的运行时选择，而非能力标志。能力标志是规范级的契约。模型的选择是正交的。

### 为什么是 JSON-RPC 而不是 REST？

JSON-RPC 2.0（2010）是一个轻量级的双向协议。REST 是客户端发起的。MCP 需要服务器发起的消息（sampling、通知），因此带有对称请求/响应形状的 JSON-RPC 是自然的适配。JSON-RPC 还可以在 stdio 和 WebSocket/Streamable HTTP 上干净地组合，而无需重新发明 HTTP 的请求形状。

```figure
mcp-tool-call
```

## 使用

`code/main.py` 发布了一个最小的 JSON-RPC 2.0 解析器和发射器，然后手工走过 `initialize` → `tools/list` → `tools/call` → 关闭序列，打印每条消息。没有真正的传输；只有消息形状。与进一步阅读中链接的规范进行对比以验证每个信封。

需要注意的地方：

- `initialize` 双向声明能力；响应具有 `serverInfo` 和 `protocolVersion: "2025-11-25"`。
- `tools/list` 返回一个 `tools` 数组；每个条目有 `name`、`description`、`inputSchema`。
- `tools/call` 使用 `params.name` 和 `params.arguments`。
- 响应 `content` 是一个 `{type, text}` 块的数组。

## 产出

本课产出 `outputs/skill-mcp-handshake-tracer.md`。给定一个 MCP 客户端-服务器交互的 pcap 风格文本记录，该技能为每条消息标注它属于哪个原语、哪个生命周期阶段以及它依赖于哪个能力。

## 练习

1. 运行 `code/main.py`。识别发生能力协商的行，并描述如果服务器没有声明 `tools.listChanged` 会有什么变化。

2. 扩展解析器以处理 `notifications/progress`。消息形状：`{method: "notifications/progress", params: {progressToken, progress, total}}`。在长时间运行的 `tools/call` 进行中发出它，并确认客户端处理程序会显示进度条。

3. 从上到下阅读 MCP 2025-11-25 规范——整个文档约 80 页。识别大多数服务器不需要的一个能力标志。提示：它与资源订阅有关。

4. 在纸上草绘一个假设的 "cron job" 特性会属于的原语。（提示：服务器希望客户端在预定时间调用它。六个原语今天都不适合。）MCP 的 2026 年路线图对此有一个草案 SEP。

5. 从 GitHub 上的开源 MCP 服务器解析一个会话日志。计算请求 vs 响应 vs 通知消息的计数。计算生命周期 vs 运行阶段的流量占比。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| MCP | "Model Context Protocol" | 用于模型到工具发现和调用的开放协议 |
| 服务器原语 | "服务器暴露什么" | tools（动作）、resources（数据）、prompts（模板） |
| 客户端原语 | "客户端让服务器使用什么" | roots（范围）、sampling（LLM 回调）、elicitation（用户输入） |
| JSON-RPC 2.0 | "线格式" | 对称的请求/响应/通知信封 |
| `initialize` 握手 | "能力协商" | 第一对消息；服务器和客户端声明它们支持的特性 |
| `tools/list` | "发现" | 客户端向服务器请求其当前工具集 |
| `tools/call` | "调用" | 客户端请求服务器使用参数执行工具 |
| `notifications/*_changed` | "变更事件" | 服务器告诉客户端其原语列表已更改 |
| 内容块 | "类型化结果" | 工具结果中的 `{type: "text" \| "image" \| "resource" \| "ui_resource"}` |
| SEP | "规范演进提案" | 带编号的草案提案（例如用于异步 Tasks 的 SEP-1686） |

## 拓展阅读

- [Model Context Protocol — Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) —— 权威规范文档
- [Model Context Protocol — Architecture concepts](https://modelcontextprotocol.io/docs/concepts/architecture) —— 六个原语的心智模型
- [Anthropic — Introducing the Model Context Protocol](https://www.anthropic.com/news/model-context-protocol) —— 2024 年 11 月发布文章
- [MCP blog — First MCP anniversary](https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/) —— 一周年回顾和 2025-11-25 规范变化
- [WorkOS — MCP 2025-11-25 spec update](https://workos.com/blog/mcp-2025-11-25-spec-update) —— SEP-1686、1036、1577、835 和 1724 的摘要
