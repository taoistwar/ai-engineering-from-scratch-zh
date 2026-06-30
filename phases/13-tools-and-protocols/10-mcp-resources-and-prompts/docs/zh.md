# MCP 资源和提示词 — 超越工具的上下文暴露

> 工具获得了 MCP 90% 的注意力。另外两种服务器原语解决不同的问题。资源暴露数据用于读取；提示词暴露可复用模板作为斜杠命令。许多服务器应该使用资源来包装读取操作，而非用工具，应该使用提示词来替代在客户端提示词中硬编码工作流。本课程点明决策规则并演练 `resources/*` 和 `prompts/*` 消息。

**Type:** Build
**Languages:** Python（stdlib，资源和提示词处理程序）
**Prerequisites:** Phase 13 · 07（MCP 服务器）
**Time:** ~45 分钟

## 学习目标

- 针对给定领域，在工具、资源或提示词之间决定如何暴露能力。
- 实现 `resources/list`、`resources/read`、`resources/subscribe` 并处理 `notifications/resources/updated`。
- 实现带有参数模板的 `prompts/list` 和 `prompts/get`。
- 识别宿主何时将提示词呈现为斜杠命令 vs 自动注入的上下文。

## 问题

一个天真的 MCP 笔记应用服务器把一切都暴露为工具：`notes_read`、`notes_list`、`notes_search`。这会把每个数据访问都包裹在模型驱动的工具调用中。后果：

- 模型必须为每个可能受益于上下文的查询决定是否调用 `notes_read`。
- 只读内容无法被订阅或流式传输到宿主的侧边面板。
- 客户端 UI（Claude Desktop 的资源附加面板、Cursor 的"包含文件"选择器）无法呈现数据。

正确的划分：将数据暴露为资源，将变更或计算操作暴露为工具，将可复用的多步工作流暴露为提示词。每种原语都有其 UX 载体和访问模式。

## 概念

### 工具 vs 资源 vs 提示词 — 决策规则

| 能力 | 原语 |
|------|------|
| 用户想要搜索、筛选或转换数据 | 工具 |
| 用户希望宿主将此数据作为上下文包含 | 资源 |
| 用户想要一个可重新运行的可模板化工作流 | 提示词 |

指导原则：如果模型在每个相关查询上调用它都会受益，它就是工具。如果用户在对话中附加它会受益，它就是资源。如果整个多步工作流是用户想要复用的单元，它就是提示词。

### 资源

`resources/list` 返回 `{resources: [{uri, name, mimeType, description?}]}`。`resources/read` 接受 `{uri}` 并返回 `{contents: [{uri, mimeType, text | blob}]}`。

URI 可以是任何可寻址的：

- `file:///Users/alice/notes/mcp.md`
- `postgres://my-db/query/SELECT ...`
- `notes://note-14`（自定义方案）
- `memory://session-2026-04-22/recent`（服务器特定）

`contents[]` 支持文本和二进制。二进制使用 `blob` 作为 base64 编码字符串加上 `mimeType`。

### 资源订阅

在能力中声明 `{resources: {subscribe: true}}`。客户端调用 `resources/subscribe {uri}`。当资源发生变化时，服务器发送 `notifications/resources/updated {uri}`。客户端重新读取。

用例：一个笔记服务器，其资源是磁盘上的文件；文件监视器在外部编辑时触发更新通知；Claude Desktop 在文件外部编辑时将其重新拉入上下文。

### 资源模板（2025-11-25 新增）

`resourceTemplates` 允许你暴露参数化的 URI 模式：`notes://{id}`，其中 `id` 作为补全目标。客户端可以在资源选择器中自动补全 ID。

### 提示词

`prompts/list` 返回 `{prompts: [{name, description, arguments?}]}`。`prompts/get` 接受 `{name, arguments}` 并返回 `{description, messages: [{role, content}]}`。

提示词是一个模板，填充后变为宿主馈送给模型的消息列表。例如，一个 `code_review` 提示词接受 `file_path` 参数并返回一个三消息序列：系统消息、包含文件正文的用户消息，以及带有推理模板的助手启动消息。

### 宿主和提示词

Claude Desktop、VS Code 和 Cursor 将提示词呈现为聊天 UI 中的斜杠命令。用户输入 `/code_review` 并从表单中选择参数。服务器的提示词是"用户快捷方式"和"发送给模型的完整提示词"之间的契约。

并非每个客户端都支持提示词——检查能力协商。声明了提示词能力但客户端不支持提示词，则将看不到斜杠命令。

### "列表已变更"通知

资源和提示词在集合变更时都发出 `notifications/list_changed`。一个刚导入了 20 条新笔记的笔记服务器发出 `notifications/resources/list_changed`；客户端重新调用 `resources/list` 以获取新增项。

### 内容类型约定

文本：`mimeType: "text/plain"`、`text/markdown`、`application/json`。
二进制：`image/png`、`application/pdf`，加上 `blob` 字段。
MCP Apps（第 14 课）：`text/html;profile=mcp-app` 在 `ui://` URI 中。

### 动态资源

资源 URI 不必对应静态文件。`notes://recent` 可以在每次读取时返回最近五条笔记。`db://query/users/active` 可以执行参数化查询。服务器可以自由动态计算内容。

规则：如果客户端可以按 URI 缓存，URI 必须稳定。如果计算是一次性的，URI 应包含时间戳或 nonce，以免客户端缓存过时。

### 订阅 vs 轮询

支持订阅的客户端通过 `notifications/resources/updated` 接收服务器推送。不支持订阅的客户端或在订阅之前的宿主则通过重新读取来轮询。两者都符合规范。服务器的能力声明告诉客户端它支持哪种。

订阅的成本：服务器上每个会话的状态（谁订阅了什么）。保持订阅集合有界；断开连接的客户端应该超时清理。

### 提示词 vs 系统提示词

MCP 中的提示词不是系统提示词。宿主的系统提示词（其自身的操作指令）和 MCP 提示词（由用户调用的服务器提供的模板）并存。一个良好的客户端绝不允许服务器提示词覆盖其自身的系统提示词；它分层叠加。

## 使用

`code/main.py` 扩展了第 07 课的笔记服务器，添加了：

- 每个笔记的资源（`notes://note-1` 等），支持 `resources/subscribe`。
- 一个 `review_note` 提示词，渲染为三消息模板。
- 一个文件监视器模拟，在笔记被修改时发出 `notifications/resources/updated`。
- 一个 `notes://recent` 动态资源，始终返回最近五条笔记。

运行演示来查看完整流程。

## 交付物

本课程产出 `outputs/skill-primitive-splitter.md`。给定一个拟议的 MCP 服务器，该技能将每个能力分类为工具/资源/提示词，并附上理由。

## 练习

1. 运行 `code/main.py`。观察初始资源列表，然后触发笔记编辑并验证 `notifications/resources/updated` 事件是否触发。

2. 添加一个 `resources/list_changed` 发射器：当创建新笔记时，发送通知以便客户端重新发现。

3. 为 GitHub MCP 服务器设计三个提示词：`summarize_pr`、`triage_issue`、`release_notes`。每个都要有参数模式。提示词正文应无需进一步编辑即可运行。

4. 选取第 07 课服务器中的一个现有工具，判断它是应该保持为工具还是拆分为资源加工具对。用一句话说明理由。

5. 阅读规范中的 `server/resources` 和 `server/prompts` 部分。找出 `resources/read` 中很少被填充但得到规范支持的一个字段。提示：查看资源内容上的 `_meta`。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|----------|
| 资源 | "暴露的数据" | 宿主可以读取的 URI 可寻址内容 |
| 资源 URI | "指向数据的指针" | 带方案前缀的标识符（`file://`、`notes://` 等） |
| `resources/subscribe` | "监视变更" | 客户端选择性加入的对特定 URI 的服务器推送更新 |
| `notifications/resources/updated` | "资源已变更" | 向客户端发出信号：已订阅资源有新内容 |
| 资源模板 | "参数化 URI" | 带有补全提示的 URI 模式，供宿主选择器使用 |
| 提示词 | "斜杠命令模板" | 带有参数槽的命名多消息模板 |
| 提示词参数 | "模板输入" | 宿主在渲染前收集的有类型参数 |
| `prompts/get` | "渲染模板" | 服务器返回填充后的消息列表 |
| 内容块 | "有类型块" | `{type: text | image | resource | ui_resource}` |
| 斜杠命令 UX | "用户快捷方式" | 宿主将提示词呈现为以 `/` 开头的命令 |

## 进一步阅读

- [MCP — 概念：资源](https://modelcontextprotocol.io/docs/concepts/resources) — 资源 URI、订阅和模板
- [MCP — 概念：提示词](https://modelcontextprotocol.io/docs/concepts/prompts) — 提示词模板和斜杠命令集成
- [MCP — 服务器资源规范 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/resources) — `resources/*` 消息完整参考
- [MCP — 服务器提示词规范 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/prompts) — `prompts/*` 消息完整参考
- [MCP — 协议信息站点：资源](https://modelcontextprotocol.info/docs/concepts/resources/) — 扩展官方文档的社区指南
