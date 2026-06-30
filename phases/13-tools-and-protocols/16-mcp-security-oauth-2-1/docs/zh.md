# MCP 安全 II — OAuth 2.1、资源指示器、增量范围

> 远程 MCP 服务器需要的是授权，而不仅仅是认证。2025-11-25 规范对齐到 OAuth 2.1 + PKCE + 资源指示器（RFC 8707）+ 受保护资源元数据（RFC 9728）。SEP-835 在 403 WWW-Authenticate 上添加带逐步授权升级的增量范围同意。本课程将逐步授权升级流程作为状态机实现，让你能看到每一跳。

**Type:** Build
**Languages:** Python（stdlib，OAuth 状态机模拟器）
**Prerequisites:** Phase 13 · 09（传输），Phase 13 · 15（安全 I）
**Time:** ~75 分钟

## 学习目标

- 区分资源服务器和授权服务器的职责。
- 遍历 PKCE 保护的 OAuth 2.1 授权码流程。
- 使用 `resource`（RFC 8707）和受保护资源元数据（RFC 9728）防止混淆代理攻击。
- 实现逐步授权升级：服务器响应 403 并带有 WWW-Authenticate，要求更高级别的范围；客户端重新提示用户同意并重试。

## 问题

早期 MCP（2025 年之前）的远程服务器使用临时 API 密钥或甚至没有认证。2025-11-25 规范通过完整的 OAuth 2.1 配置文件弥补了这一缺口。

三种真实需求：

- **普通远程服务器。** 用户安装一个远程 MCP 服务器，访问其 Notion / GitHub / Gmail。OAuth 2.1 配合 PKCE 是正确的形态。
- **范围升级。** 一个授予了 `notes:read` 的笔记服务器后来可能需要对特定操作的 `notes:write`。逐步授权升级（SEP-835）请求额外范围，而不是重做整个流程。
- **混淆代理预防。** 客户端持有一个受众范围为服务器 A 的令牌。服务器 A 是恶意的，试图将令牌提交给服务器 B。资源指示器（RFC 8707）将令牌固定到其预期受众。

OAuth 2.1 并不新鲜。新鲜的是 MCP 的配置文件：特定的必需流程（仅授权码 + PKCE；没有 implicit，默认没有 client credentials）、每个令牌请求上强制的资源指示器，以及发布受保护资源元数据以便客户端知道去哪里。

## 概念

### 角色

- **客户端。** MCP 客户端（Claude Desktop、Cursor 等）。
- **资源服务器。** MCP 服务器（笔记、GitHub、Postgres 等）。
- **授权服务器。** 签发令牌。可以是与资源服务器相同的服务，也可以是独立的 IdP（Auth0、Keycloak、Cognito）。

在 MCP 的配置文件中，资源和授权服务器可以是同一宿主，但应当通过 URL 来区分。

### 授权码 + PKCE

流程：

1. 客户端生成 `code_verifier`（随机）和 `code_challenge`（SHA256）。
2. 客户端将用户重定向到 `/authorize?response_type=code&client_id=...&redirect_uri=...&scope=notes:read&code_challenge=...&resource=https://notes.example.com`。
3. 用户同意。授权服务器重定向到 `redirect_uri?code=...`。
4. 客户端 POST 到 `/token?grant_type=authorization_code&code=...&code_verifier=...&resource=...`。
5. 授权服务器根据存储的 challenge 验证 verifier 的哈希并签发访问令牌。
6. 客户端使用令牌：`Authorization: Bearer ...` 在每个对资源服务器的请求上。

PKCE 防止授权码拦截攻击。资源指示器防止令牌在其他地方有效。

### 受保护资源元数据（RFC 9728）

资源服务器发布 `.well-known/oauth-protected-resource` 文档：

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com"],
  "scopes_supported": ["notes:read", "notes:write", "notes:delete"]
}
```

客户端从资源服务器发现授权服务器。减少配置——客户端只需要资源 URL。

### 资源指示器（RFC 8707）

令牌请求中的 `resource` 参数固定令牌的预期受众。签发的令牌包含 `aud: "https://notes.example.com"`。另一个接收此令牌的 MCP 服务器检查 `aud` 并拒绝。

### 范围模型

范围是由空格分隔的字符串。常见的 MCP 约定：

- `notes:read`、`notes:write`、`notes:delete`
- `admin:*` 用于管理员能力（谨慎使用）
- `profile:read` 用于身份

范围选择应该是最小权限：现在请求你需要的，需要更多时逐步升级。

### 逐步授权升级（SEP-835）

用户授予 `notes:read`。他们后来要求代理删除一条笔记。服务器响应：

```
HTTP/1.1 403 Forbidden
WWW-Authenticate: Bearer error="insufficient_scope",
    scope="notes:delete", resource="https://notes.example.com"
```

客户端看到 insufficient_scope 错误，用额外范围提示用户同意对话框，执行一个迷你 OAuth 流程，用新令牌重试请求。

### 令牌受众验证

每个请求：服务器检查 `token.aud == self.resource_url`。不匹配 = 401。这阻止跨服务器令牌重用。

### 短生命周期令牌和轮换

访问令牌应当是短生命周期的（默认 1 小时）。刷新令牌在每次刷新时轮换。客户端在后台处理静默刷新。

### 无令牌透传

采样服务器（第 13 阶段 · 11）禁止将客户端的令牌透传给其他服务。采样请求是边界。

### 混淆代理预防

令牌绑定到 `aud`。客户端绑定到 `client_id`。每个请求对两者验证。规范明确禁止在 MCP 之前远程工具生态系统中常见的旧的"传递令牌"模式。

### 客户端 ID 发现

每个 MCP 客户端在固定 URL 上发布其元数据。授权服务器可以获取客户端的元数据文档以发现重定向 URI 和联系信息。这消除了手动客户端注册。

### 网关和 OAuth

第 13 阶段 · 17 展示企业网关如何处理 OAuth：网关持有上游服务器的凭据，给客户端的令牌是网关签发的，上游令牌从不离开网关。这翻转了信任模型——用户向网关认证一次；网关处理 N 个服务器授权。

## 使用

`code/main.py` 将完整的 OAuth 2.1 逐步升级流程模拟为状态机。它实现了：

- PKCE code-verifier / challenge 生成。
- 带有资源指示器的授权码流程。
- 受保护资源元数据端点。
- 带有受众检查的令牌验证。
- 在 `insufficient_scope` 上的逐步升级。

本课程中没有 HTTP 服务器；状态机在内存中运行，让你能跟踪每一跳。第 13 阶段 · 17 的网关课程将其连接到实际传输。

## 交付物

本课程产出 `outputs/skill-oauth-scope-planner.md`。给定一个带有工具的远程 MCP 服务器，该技能设计范围集合、固定规则和逐步升级策略。

## 练习

1. 运行 `code/main.py`。跟踪两个范围的逐步升级流程。注意在逐步升级时哪些跳会重复。

2. 添加刷新令牌轮换：每次刷新签发一个新的刷新令牌并使旧令牌失效。模拟一个被盗的刷新令牌在轮换后被使用并确认其失败。

3. 使用 stdlib http.server 将受保护资源元数据端点实现为真正的 HTTP 响应。镜像第 09 课的 /mcp 端点。

4. 为 GitHub MCP 服务器设计范围层次结构：读取仓库、写入 PR、批准 PR、合并 PR、管理员。在每层之间使用逐步升级。

5. 阅读 RFC 8707 和 RFC 9728。找出 9728 中 MCP 用于与 RFC 示例不同的一个字段。（提示：涉及 `scopes_supported`。）

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|----------|
| OAuth 2.1 | "现代 OAuth" | 强制 PKCE 并禁止 implicit 流程的综合 RFC |
| PKCE | "持有权证明" | 代码验证器 + challenge，击败授权码拦截 |
| 资源指示器 | "令牌受众" | RFC 8707 `resource` 参数，将令牌固定到一个服务器 |
| 受保护资源元数据 | "发现文档" | RFC 9728 `.well-known/oauth-protected-resource` |
| 逐步授权升级 | "增量同意" | SEP-835 流程，按需添加范围 |
| `insufficient_scope` | "403 带 WWW-Authenticate" | 服务器信号，需要重新同意获取更大范围 |
| 混淆代理 | "跨服务令牌重用" | 攻击者不适当地转发令牌的攻击 |
| 短生命周期令牌 | "访问令牌 TTL" | 携带者令牌快速过期；刷新令牌续期 |
| 范围层次结构 | "最小权限栈" | 分级范围集合，级别间使用逐步升级 |
| 客户端 ID 元数据 | "客户端发现文档" | 客户端发布自身 OAuth 元数据的 URL |

## 进一步阅读

- [MCP — 授权规范](https://modelcontextprotocol.io/specification/draft/basic/authorization) — 规范 MCP OAuth 配置文件
- [den.dev — MCP 11 月授权规范](https://den.dev/blog/mcp-november-authorization-spec/) — 2025-11-25 变更演练
- [RFC 8707 — OAuth 2.0 的资源指示器](https://datatracker.ietf.org/doc/html/rfc8707) — 受众固定 RFC
- [RFC 9728 — OAuth 2.0 受保护资源元数据](https://datatracker.ietf.org/doc/html/rfc9728) — 发现文档 RFC
- [Aembit — MCP OAuth 2.1、PKCE 和 AI 授权的未来](https://aembit.io/blog/mcp-oauth-2-1-pkce-and-the-future-of-ai-authorization/) — 实战逐步升级流程演练
