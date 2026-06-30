# MCP 传输 — stdio vs Streamable HTTP vs SSE 迁移

> stdio 仅适用于本地。Streamable HTTP（2025-03-26）是远程标准。旧的 HTTP+SSE 传输已弃用，正在 2026 年年中移除。选错传输会导致迁移成本；选对传输可获得支持远程部署的 MCP 服务器，附带会话连续性和 DNS 重绑定防护。

**Type:** Learn
**Languages:** Python（stdlib，Streamable HTTP 端点骨架）
**Prerequisites:** Phase 13 · 07、08（MCP 服务器和客户端）
**Time:** ~45 分钟

## 学习目标

- 根据部署形态（本地 vs 远程，单进程 vs 集群）在 stdio 和 Streamable HTTP 之间做出选择。
- 实现 Streamable HTTP 单端点模式：POST 用于请求，GET 用于会话流。
- 强制 `Origin` 验证和会话 ID 语义以防止 DNS 重绑定。
- 在 2026 年年中移除截止日期前将旧的 HTTP+SSE 服务器迁移到 Streamable HTTP。

## 问题

第一个 MCP 远程传输（2024-11）是 HTTP+SSE：两个端点，一个用于客户端的 POST，一个服务器发送事件通道用于服务器到客户端流。它确实能用。但也有缺点：每个会话两个端点，在某些 CDN 前缓存损坏，以及依赖长连接 SSE（某些 WAF 会主动终止）。

2025-03-26 规范用 Streamable HTTP 替代了它：一个端点，POST 用于客户端请求，GET 用于建立会话流，两者共享 `Mcp-Session-Id` 头部。此后构建或迁移的每个服务器都使用 Streamable HTTP。旧的 SSE 模式正在被弃用——Atlassian Rovo 于 2026 年 6 月 30 日移除；Keboola 于 2026 年 4 月 1 日移除；大多数剩余的 enterprise 服务器在 2026 年底前。

而 stdio 对本地服务器仍然重要。Claude Desktop、VS Code 以及每个 IDE 形态的客户端都通过 stdio 启动服务器。正确的心智模型：stdio 用于"这台机器"，Streamable HTTP 用于"网络上的"。没有交叉。

## 概念

### stdio

- 子进程传输。客户端启动服务器，通过 stdin/stdout 通信。
- 每行一个 JSON 对象。换行分隔。
- 没有会话 ID；进程身份就是会话。
- 无需认证（子进程继承父进程的信任边界）。
- 绝不要用于远程服务器——你需要 SSH 或 socat 做隧道，此时应直接用 Streamable HTTP。

### Streamable HTTP

单个端点 `/mcp`（或任意路径）。支持三种 HTTP 方法：

- **POST /mcp。** 客户端发送 JSON-RPC 消息。服务器返回单个 JSON 响应，或者一个包含一个或多个响应的 SSE 流（用于批处理响应和与该请求相关的通知）。
- **GET /mcp。** 客户端打开长连接 SSE 通道。服务器用它进行服务器到客户端请求（采样、通知、引导）。
- **DELETE /mcp。** 客户端显式终止会话。

会话由服务器在第一个响应上设置的 `Mcp-Session-Id` 头部标识，客户端在后续每个请求上回显。会话 ID 必须是加密随机的（128+ 位）；出于安全原因，拒绝客户端选择的 ID。

### 单端点 vs 双端点

旧规范中的双端点模式在 2026 年仍可调用——规范声明其为"旧版兼容"。但所有新服务器应为单端点。官方 SDK 生成单端点；仅在与未迁移的远程服务器通信时使用旧版模式。

### `Origin` 验证和 DNS 重绑定

浏览器不是 MCP 客户端（目前），但攻击者可以构建一个网页，诱使浏览器向 `localhost:1234/mcp` 发送 POST——这是用户本地 MCP 服务器监听的地方。如果服务器不检查 `Origin`，浏览器的同源策略无法保护它，因为 `Origin: http://evil.com` 是有效的跨源。

2025-11-25 规范要求服务器拒绝 `Origin` 不在白名单中的请求。白名单通常包含 MCP 客户端宿主（`https://claude.ai`、`vscode-webview://*`）和本地 UI 的 localhost 变体。

### 会话 ID 生命周期

1. 客户端在没有 `Mcp-Session-Id` 的情况下发送第一个请求。
2. 服务器分配一个随机 ID，在响应头部设置 `Mcp-Session-Id`。
3. 客户端在所有后续请求和流的 `GET /mcp` 上回显该头部。
4. 服务器可以撤销会话；客户端在后续请求上看到 404 并必须重新初始化。
5. 客户端可以显式 DELETE 会话以进行干净关闭。

### 保活和重连

SSE 连接会断开。客户端通过使用相同的 `Mcp-Session-Id` 重新 GET 来重建。服务器必须排队保存在中断期间错过的事件（在合理窗口内），并通过客户端回显的 `last-event-id` 头部重播。

第 13 阶段 · 13 涵盖任务（Tasks），它允许长时间运行的工作即使在完整会话重连后也能存活。

### 向后兼容性探测

一个希望同时支持新旧服务器的客户端：

1. POST 到 `/mcp`。
2. 如果响应是 `200 OK` 带 JSON 或 SSE，这是 Streamable HTTP。
3. 如果响应是 `200 OK` 带 `Content-Type: text/event-stream` 且 `Location` 头部指向次要端点，这是旧版 HTTP+SSE；跟随 `Location`。

### Cloudflare、ngrok 和部署

2026 年的生产远程 MCP 服务器运行在 Cloudflare Workers（配合其 MCP Agents SDK）、Vercel Functions 或容器化的 Node/Python 上。关键：你的部署必须支持 SSE GET 的长连接 HTTP。Vercel 的免费层上限 10 秒，不适合。Cloudflare Workers 支持无限流。

### 网关组合

当你用网关（第 13 阶段 · 17）前置多个 MCP 服务器时，网关是一个单一的 Streamable HTTP 端点，重写会话 ID 并多路复用上游。工具在网关层合并；客户端看到单个逻辑服务器。

### 传输失败模式

- **stdio SIGPIPE。** 写入中间子进程死亡引发 SIGPIPE；服务器应干净退出。客户端应检测 EOF 并将会话标记为已死。
- **HTTP 502 / 504。** Cloudflare、nginx 等代理在上游失败时发出这些。Streamable HTTP 客户端应在短暂退避后重试一次。
- **SSE 连接断开。** TCP RST、代理超时或客户端网络变更关闭流。客户端使用 `Mcp-Session-Id` 和可选的 `last-event-id` 重连以恢复。
- **会话撤销。** 服务器使会话 ID 失效；客户端在下一次请求时看到 404。客户端必须重新握手。
- **时钟偏差。** 客户端上的资源 TTL 计算与服务器不一致。客户端应将服务器时间戳视为权威。

### 何时绕过 Streamable HTTP

一些企业在自己的网络内部署基于 gRPC 或消息队列传输的 MCP 服务器。这是非标准的——MCP 的规范没有正式定义这些。网关可以对 MCP 客户端暴露 Streamable HTTP 接口，同时在内部使用 gRPC。保持外部接口符合规范；网关负责翻译。

## 使用

`code/main.py` 使用 `http.server`（标准库）实现了一个最小 Streamable HTTP 端点。它处理 `/mcp` 上的 POST、GET 和 DELETE，在第一个响应上设置 `Mcp-Session-Id`，验证 `Origin`，并拒绝来自非白名单来源的请求。处理程序复用了第 07 课笔记服务器的分发逻辑。

关注要点：

- POST 处理程序读取 JSON-RPC 正文，分发并写入 JSON 响应（单响应变体；SSE 变体结构相似）。
- `Origin` 检查拒绝默认的 `http://evil.example` 探测但接受 `http://localhost`。
- 会话 ID 是随机 128 位十六进制字符串；服务器在内存中保存每个会话的状态。

## 交付物

本课程产出 `outputs/skill-mcp-transport-migrator.md`。给定一个 HTTP+SSE（旧版）MCP 服务器，该技能生成一个迁移计划，包含会话 ID 连续性、Origin 检查和向后兼容探测支持。

## 练习

1. 运行 `code/main.py`。用 `curl` POST 一个 `initialize` 并观察 `Mcp-Session-Id` 响应头部。POST 第二个请求回显头部并验证会话连续性。

2. 添加打开 SSE 流的 GET 处理程序。每五秒发送一个 `notifications/progress` 事件。使用相同的会话 ID 重新 GET 重连，确认服务器接受。

3. 实现 `last-event-id` 重播逻辑。重连时，重播在该 ID 之后生成的任何事件。

4. 扩展 `Origin` 验证以支持通配符模式（`https://*.example.com`），确认它接受 `https://app.example.com` 但拒绝 `https://evil.example.com.attacker.net`。

5. 从官方注册表获取一个旧版 HTTP+SSE 服务器（有好几个）并绘制迁移方案：端点处理、会话 ID 生成和头部语义的变化。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|----------|
| stdio 传输 | "本地子进程" | 基于 stdin/stdout 的 JSON-RPC，换行分隔 |
| Streamable HTTP | "远程传输" | 单端点 POST + GET + 可选 SSE，2025-03-26 规范 |
| HTTP+SSE | "旧版" | 双端点模型，2026 年年中移除 |
| `Mcp-Session-Id` | "会话头部" | 服务器分配的随机 ID，在后续每个请求上回显 |
| `Origin` 白名单 | "DNS 重绑定防御" | 拒绝 Origin 未获批准的请求 |
| 单端点 | "一个 URL" | `/mcp` 处理所有会话操作的 POST / GET / DELETE |
| `last-event-id` | "SSE 重放" | 用于恢复丢失事件而不错过的头部 |
| 向后兼容探测 | "新旧检测" | 客户端响应形态检查，自动选择传输 |
| 长连接 HTTP | "SSE 流" | 服务器在一个 TCP 连接上推送事件持续数分钟或数小时 |
| 会话撤销 | "强制重新初始化" | 服务器使会话 ID 失效；客户端必须再次握手 |

## 进一步阅读

- [MCP — 基础传输规范 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports) — stdio 和 Streamable HTTP 的规范参考
- [MCP — 基础传输规范 2025-03-26](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports) — 引入 Streamable HTTP 的修订版
- [Cloudflare — MCP 传输](https://developers.cloudflare.com/agents/model-context-protocol/transport/) — Workers 宿主的 Streamable HTTP 模式
- [AWS — MCP 传输机制](https://builder.aws.com/content/35A0IphCeLvYzly9Sw40G1dVNzc/mcp-transport-mechanisms-stdio-vs-streamable-http) — 跨部署形态的比较
- [Atlassian — HTTP+SSE 弃用通知](https://community.atlassian.com/forums/Atlassian-Remote-MCP-Server/HTTP-SSE-Deprecation-Notice/ba-p/3205484) — 具体迁移截止日期示例
