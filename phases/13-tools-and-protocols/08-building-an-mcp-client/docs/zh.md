# 构建 MCP 客户端 — 发现、调用、会话管理

> 大多数 MCP 内容发布的是服务器教程，对客户端只一笔带过。客户端代码才是复杂编排的所在：进程启动、能力协商、跨多服务器的工具列表合并、采样回调、重连以及命名空间冲突解决。本课程构建一个多服务器客户端，将三个不同的 MCP 服务器整合为一个扁平的模型工具命名空间。

**Type:** Build
**Languages:** Python (stdlib, 多服务器 MCP 客户端)
**Prerequisites:** Phase 13 · 07（构建 MCP 服务器）
**Time:** ~75 分钟

## 学习目标

- 将 MCP 服务器作为子进程启动，完成 `initialize`，并发送 `notifications/initialized`。
- 维护每个服务器的会话状态（能力、工具列表、最近的通知 ID）。
- 将跨多个服务器的工具列表合并为一个带有冲突处理的命名空间。
- 将工具调用路由到拥有该工具的服务器，并重组响应。

## 问题

一个真正的代理宿主（Claude Desktop、Cursor、Goose、Gemini CLI）同时加载多个 MCP 服务器。一个用户可能同时运行文件系统服务器、Postgres 服务器和 GitHub 服务器。客户端的工作是：

1. 启动每个服务器。
2. 独立完成每次握手。
3. 对每个服务器调用 `tools/list` 并扁平化结果。
4. 当模型发出 `notes_search` 时，在合并的命名空间中查找并路由到正确的服务器。
5. 处理来自任何服务器的通知（`tools/list_changed`）而不阻塞。
6. 在传输失败时重连。

手工实现所有这些正是"玩具"与"可用的"之间的分界线。官方 SDK 封装了这些，但心智模型必须是你的。

## 概念

### 子进程启动

`subprocess.Popen` 配合 `stdin=PIPE, stdout=PIPE, stderr=PIPE`。设置 `bufsize=1` 并使用文本模式进行逐行读取。每个服务器是一个进程；客户端为每个服务器持有一个 `Popen` 句柄。

### 每个服务器的会话状态

每个服务器一个 `Session` 对象，持有：

- `process` — Popen 句柄。
- `capabilities` — 服务器在 `initialize` 时声明的内容。
- `tools` — 上次的 `tools/list` 结果。
- `pending` — 请求 ID 到等待响应的 promise/future 的映射。

请求本质上是异步的；在服务器 B 正在调用期间，向服务器 A 发送的 `tools/call` 不能阻塞。要么使用带队列的线程，要么使用 asyncio。

### 合并的命名空间

当客户端看到聚合的工具列表时，名称可能冲突。两个服务器可能都暴露了 `search`。客户端有三种选择：

1. **按服务器名称加前缀。** `notes/search`、`files/search`。清晰但丑陋。
2. **静默先到先得。** 后来的服务器的 `search` 覆盖先前的。有风险；隐藏冲突。
3. **冲突拒绝。** 拒绝加载第二个服务器；通知用户。对安全敏感的宿主最安全。

Claude Desktop 使用按服务器加前缀。Cursor 使用冲突拒绝并给出明确错误。VS Code MCP 也采用按服务器加前缀。

### 路由

合并后，分发表将 `tool_name -> session` 映射。模型按名称发出调用；客户端找到会话并向该服务器的 stdin 写入 `tools/call` 消息，然后等待响应。

### 采样回调

如果服务器在 `initialize` 时声明了 `sampling` 能力，它可能发送 `sampling/createMessage` 请求客户端运行其 LLM。客户端必须：

1. 在该样本解析之前阻止对该服务器的进一步请求，或者如果实现支持并发则可以管线化。
2. 调用其 LLM 提供者。
3. 将响应发送回服务器。

第 11 课端到端涵盖采样。本课为了完整性只做存根。

### 通知处理

`notifications/tools/list_changed` 意味着重新调用 `tools/list`。`notifications/resources/updated` 意味着如果资源正在使用则重新读取。通知不能产生响应——不要尝试确认它们。

一个常见的客户端 bug：在 `tools/call` 上阻塞读取循环而通知在流中等待。使用后台读取线程将每条消息推送到队列；主线程出队并分发。

### 重连

传输可能失败：服务器崩溃、操作系统杀死进程、stdio 管道断裂。客户端检测到 stdout 的 EOF 并将会话视为已死。选项：

- 静默重启服务器并重新握手。适用于纯只读服务器。
- 将故障呈现给用户。适用于具有用户可见会话的有状态服务器。

第 13 阶段 · 09 涵盖 Streamable HTTP 的重连语义；stdio 更简单。

### 保活和会话 ID

Streamable HTTP 使用 `Mcp-Session-Id` 头部。Stdio 没有会话 ID——进程身份就是会话。保活 ping 是可选的；stdio 管道不会因不活动而断开。

## 使用

`code/main.py` 将三个模拟 MCP 服务器作为子进程启动，与每个服务器握手，合并其工具列表，并将工具调用路由到正确的服务器。"服务器"实际上是运行玩具响应器的其他 Python 进程（没有真实的 LLM）。运行它以查看：

- 三次初始化，每次都有各自的能力集。
- 三个 `tools/list` 结果合并为一个 7 工具命名空间。
- 基于工具名称的路由决策。
- 通过命名空间前缀阻止的冲突。

关注要点：

- `Session` 数据类干净地持有每个服务器的状态。
- 后台读取线程在不阻塞主线程的情况下从 stdout 逐行出队。
- 分发表是一个简单的 `dict[str, Session]`。
- 冲突处理是显式的：当两个服务器声明相同名称时，后来的那个被加上前缀重命名。

## 交付物

本课程产出 `outputs/skill-mcp-client-harness.md`。给定声明性的 MCP 服务器列表（名称、命令、参数），该技能生成一个能启动它们、合并工具列表并提供带冲突解决的路由函数的测试框架。

## 练习

1. 运行 `code/main.py` 并观察服务器启动日志。用 SIGTERM 杀死一个模拟服务器进程，观察客户端如何检测 EOF 并将该会话标记为已死。

2. 实现命名空间前缀。当两个服务器暴露 `search` 时，将第二个重命名为 `<server>/search`。更新分发表并验证工具调用路由正确。

3. 为服务器重启添加连接池风格的退避：连续失败时指数退避，上限 30 秒，三次失败后向用户发通知。

4. 设计一个支持 100 个并发 MCP 服务器的客户端。什么数据结构替代简单的分发表？（提示：前缀命名空间的 trie，加上每个服务器工具数量的指标。）

5. 将客户端迁移到官方 MCP Python SDK。SDK 封装了 `stdio_client` 和 `ClientSession`。代码应从约 200 行缩减到约 40 行，同时保留多服务器路由。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|----------|
| MCP 客户端 | "代理宿主" | 启动服务器并编排工具调用的进程 |
| 会话 | "每个服务器的状态" | 能力、工具列表和待处理请求的簿记 |
| 合并命名空间 | "一个工具列表" | 所有活动服务器中工具名称的扁平集合 |
| 命名空间冲突 | "两个服务器同名工具" | 客户端必须对重复项加前缀、拒绝或先到先得 |
| 路由 | "谁接收这个调用？" | 从工具名称到拥有服务器的分发 |
| 后台读取器 | "非阻塞 stdout" | 将服务器 stdout 排入队列的线程或任务 |
| 采样回调 | "LLM 即服务" | 客户端处理来自服务器的 `sampling/createMessage` |
| `notifications/*_changed` | "原语已变更" | 向客户端发出必须重新发现或重新读取的信号 |
| 重连策略 | "服务器挂掉时" | 传输失败时的重启语义 |
| Stdio 会话 | "进程 = 会话" | 没有会话 ID；子进程生命周期即会话 |

## 进一步阅读

- [MCP — 客户端规范](https://modelcontextprotocol.io/specification/2025-11-25/client) — 规范客户端行为
- [MCP — 快速入门客户端指南](https://modelcontextprotocol.io/quickstart/client) — 使用 Python SDK 的 hello-world 客户端教程
- [MCP Python SDK — 客户端模块](https://github.com/modelcontextprotocol/python-sdk) — `ClientSession` 和 `stdio_client` 参考
- [MCP TypeScript SDK — Client](https://github.com/modelcontextprotocol/typescript-sdk) — TS 并行实现
- [VS Code — 扩展中的 MCP](https://code.visualstudio.com/api/extension-guides/ai/mcp) — VS Code 如何在单个编辑器宿主中多路复用多个 MCP 服务器
