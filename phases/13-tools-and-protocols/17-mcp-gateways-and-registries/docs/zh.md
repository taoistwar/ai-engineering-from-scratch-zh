# MCP 网关和注册表 — 企业控制平面

> 企业不能让每个开发者随意安装 MCP 服务器。网关集中认证、RBAC、审计、速率限制、缓存和工具投毒检测，然后将合并后的工具界面作为单个 MCP 端点暴露。官方 MCP 注册表（Anthropic + GitHub + PulseMCP + Microsoft，命名空间已验证）是规范的上游。本课程点明网关适合放置在哪里，演练一个最小实现，并概览 2026 年的供应商格局。

**Type:** Learn
**Languages:** Python（stdlib，最小网关）
**Prerequisites:** Phase 13 · 15（工具投毒），Phase 13 · 16（OAuth 2.1）
**Time:** ~45 分钟

## 学习目标

- 解释 MCP 网关位于何处（位于 MCP 客户端和多个后端 MCP 服务器之间）。
- 实现五项网关职责：认证、RBAC、审计、速率限制、策略。
- 在网关层强制固定工具哈希清单。
- 区分官方 MCP 注册表与元注册表（Glama、MCPMarket、MCP.so、Smithery、LobeHub）。

## 问题

一个财富 500 强公司拥有 30 台已批准的 MCP 服务器、5000 名开发者、合规和审计要求，以及一个希望集中策略的安全团队。让每个开发者在其 IDE 中安装任意服务器是不可能的。

网关模式：

1. 网关作为开发者连接的单个 Streamable HTTP 端点运行。
2. 网关为每个后端 MCP 服务器持有凭据。
3. 每个开发者请求通过网关自己的 OAuth 进行认证和范围限定。
4. 网关将调用路由到后端服务器，应用策略。
5. 所有调用记录日志以供审计。

Cloudflare MCP Portals、Kong AI Gateway、IBM ContextForge、MintMCP、TrueFoundry、Envoy AI Gateway——都在 2025-2026 年发布了网关或网关功能。

同时，官方 MCP 注册表作为规范上游启动：经过策展、命名空间验证、反向 DNS 命名的服务器，网关可以从中拉取。元注册表（Glama、MCPMarket、MCP.so、Smithery、LobeHub）聚合来自多个来源的服务器。

## 概念

### 五项网关职责

1. **认证。** OAuth 2.1 识别开发者；映射到用户角色。
2. **RBAC。** 每位用户的策略：哪些服务器、哪些工具、哪些范围。
3. **审计。** 每个调用记录：谁、什么、何时、结果。
4. **速率限制。** 每用户/每工具/每服务器上限以防止滥用。
5. **策略。** 拒绝投毒描述、强制两条规则、编辑 PII。

### 作为单个端点的网关

对开发者而言，网关看起来像是一个 MCP 服务器。内部它路由到 N 个后端。会话 ID（第 13 阶段 · 09）在边界处被重写。

### 凭据保管

开发者永远看不到后端令牌。网关持有它们（或代理到持有它们的身份提供者）。一个具有 `notes:read` 的开发者可以在网关上传递性访问笔记 MCP 服务器，使用网关自己的后端凭据——但仅限于绑定传递性访问的策略下。

### 在网关上固定工具哈希

网关持有已批准工具描述的清单（SHA256 哈希）。在发现时，它获取每个后端的 `tools/list`，将哈希与清单比较，并移除任何描述已变更的工具。这是第 13 阶段 · 15 的地毯抽走防御在中央统一应用。

### 策略即代码

高级网关在 OPA/Rego、Kyverno 或 Styra 中表达策略。像"用户 `alice` 只能对组织 `acme` 的仓库调用 `github.open_pr`"这样的规则被声明式编码。简单网关使用手工编码的 Python。两种形态都有效。

### 会话感知路由

当用户的会话包含混合服务器时，网关多路复用：开发者的单个 MCP 会话持有 N 个后端会话，每个服务器一个。来自任何后端的通知通过网关路由到开发者的会话。

### 命名空间合并

网关合并来自所有后端的工具命名空间，通常在冲突时加前缀。`github.open_pr`、`notes.search`。这使路由无歧义。

### 注册表

- **官方 MCP 注册表（`registry.modelcontextprotocol.io`）。** 在 Anthropic、GitHub、PulseMCP、Microsoft 的管理下启动。命名空间已验证（反向 DNS：`io.github.user/server`）。预先筛选了基本质量。
- **Glama。** 以搜索为中心的元注册表，聚合许多来源。
- **MCPMarket。** 偏商业的目录，带有供应商列表。
- **MCP.so。** 社区目录；开放提交。
- **Smithery。** 包管理器风格的安装流程。
- **LobeHub。** 在其 LobeChat 应用中集成 UI 的注册表。

企业网关默认从官方注册表拉取，允许从元注册表中管理员策展的添加，并拒绝任何未固定的项目。

### 反向 DNS 命名

官方注册表强制对公共服务器使用反向 DNS 命名：`io.github.alice/notes`。命名空间防止占用并使信任委托更清晰。

### 供应商概览，2026 年 4 月

| 供应商 | 优势 |
|--------|------|
| Cloudflare MCP Portals | 边缘托管；集成 OAuth；免费层 |
| Kong AI Gateway | K8s 原生；细粒度策略；日志到 OpenTelemetry |
| IBM ContextForge | 企业 IAM；合规；审计导出 |
| TrueFoundry | 偏 DevOps；以指标为先 |
| MintMCP | 面向开发者平台 |
| Envoy AI Gateway | 开源；可自定义的过滤器 |

第 17 阶段（生产基础设施）更深入地探讨网关运营。

## 使用

`code/main.py` 在约 150 行中提供一个最小网关：通过虚假的 Bearer 令牌认证用户，持有每用户 RBAC 策略，将请求路由到两个后端 MCP 服务器，将每个调用写入审计日志，强制速率限制，并拒绝任何后端工具——如果其描述哈希与固定清单不匹配。

关注要点：

- `RBAC` 字典以 `user_id` 为键，包含允许的 `server_tool` 条目。
- `AUDIT_LOG` 是一个仅追加的事件列表。
- 速率限制使用每用户的令牌桶。
- 固定清单是一个 `server::tool -> hash` 的字典。

## 交付物

本课程产出 `outputs/skill-gateway-bootstrap.md`。给定一个企业 MCP 计划（用户、后端、合规），该技能生成网关配置规范。

## 练习

1. 运行 `code/main.py`。作为允许用户进行调用；然后作为不允许的用户；然后发起超出速率限制的突发调用。验证所有三种流程。

2. 添加一个策略，在结果返回客户端之前从结果中编辑 PII。对 SSN 形状的字符串使用简单正则表达式通过；注意缺口（电子邮件、电话号码）。

3. 扩展审计日志以发出 OpenTelemetry GenAI span。第 13 阶段 · 20 涵盖确切的属性。

4. 为一个 50 人开发者团队设计 RBAC 策略，包含 5 个后端（notes、github、postgres、jira、slack）。谁对每个后端只读？谁可以写入？

5. 从头到尾阅读 Cloudflare 企业 MCP 帖子。识别此标准库网关不具备的 Cloudflare 的一个功能。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|----------|
| 网关 | "MCP 代理" | 位于客户端和后端之间的集中服务器 |
| 凭据保管 | "后端令牌留在服务器端" | 开发者永远看不到上游令牌 |
| 会话感知路由 | "多后端会话" | 网关为每个开发者会话多路复用 N 个后端会话 |
| 工具哈希固定 | "已批准清单" | 每个已批准工具描述的 SHA256；在中央阻止地毯抽走 |
| RBAC | "每用户策略" | 对工具和服务器的基于角色的访问控制 |
| 策略即代码 | "声明式规则" | 在网关上强制执行的 OPA/Rego、Kyverno、Styra 策略 |
| 审计日志 | "谁、什么、何时" | 用于合规的仅追加事件日志 |
| 速率限制 | "每用户令牌桶" | 每分钟上限以防止滥用 |
| 官方 MCP 注册表 | "规范上游" | `registry.modelcontextprotocol.io`，命名空间已验证 |
| 反向 DNS 命名 | "注册表命名空间" | `io.github.user/server` 约定 |

## 进一步阅读

- [官方 MCP 注册表](https://registry.modelcontextprotocol.io/) — 规范上游，命名空间已验证
- [Cloudflare — 企业 MCP](https://blog.cloudflare.com/enterprise-mcp/) — 带有 OAuth 和策略的网关模式
- [agentic-community — MCP 网关注册表](https://github.com/agentic-community/mcp-gateway-registry) — 开源参考网关
- [TrueFoundry — 什么是 MCP 网关？](https://www.truefoundry.com/blog/what-is-mcp-gateway) — 功能比较文章
- [IBM — MCP Context Forge](https://github.com/IBM/mcp-context-forge) — IBM 的企业网关
