# 生产环境中的 MCP 认证 — 注册、JWKS 刷新、受众固定令牌

> 第 16 课在内存中搭建了 OAuth 2.1 状态机框架。到 2026 年，你要向真实组织发布的每个 MCP 服务器都位于生产认证之后：可扩展到无界客户端数量的客户端注册（客户端 ID 元数据文档优先，动态客户端注册作为向后兼容的备选方案）、授权服务器元数据发现（RFC 8414 或 OpenID Connect 发现）、不会在凌晨 3 点破坏令牌验证的 JWKS 缓存刷新，以及拒绝跨资源重放的受众固定令牌。本课程使用三个角色对完整的认证交互界面建模——授权服务器、资源服务器（MCP 服务器）和客户端——这样你就可以跟踪从发现到经过验证的工具调用的每一跳。
>
> **规范说明（2025-11-25）：** 2025 年 11 月 MCP 授权规范将动态客户端注册从 `SHOULD` 降级为 `MAY`，并将**客户端 ID 元数据文档（CIMD）**作为推荐的默认注册机制。本课程按规范优先级顺序教授两者，代码出于演练目的保留了 DCR，因为它在单个进程中是完全自包含的。

**Type:** Build
**Languages:** Python（stdlib）
**Prerequisites:** Phase 13 · 16（OAuth 2.1 状态机），Phase 13 · 17（网关）
**Time:** ~90 分钟

## 学习目标

- 通过 RFC 8414 元数据发现授权服务器并验证契约。
- 实现 RFC 7591 动态客户端注册，使 MCP 客户端无需管理员干预即可注册。
- 按计划缓存和刷新 JWKS 密钥，使签名验证在密钥轮换后依然有效。
- 使用 RFC 8707 资源指示器将令牌固定到单个 MCP 资源，并拒绝混淆代理重用。
- 干净地分离三个角色——授权服务器、资源服务器、客户端——使每个角色仅强制执行属于它的检查。
- 阅读 IdP 能力矩阵，并在 IdP 无法满足 MCP 认证配置文件时拒绝部署。

## 问题

第 16 课的模拟器在内存中运行 OAuth 2.1。生产环境有三个仅内存模拟器看不到的操作缺口。

第一个缺口是注册。一个真实组织运行数百个 MCP 服务器和数千个 MCP 客户端。运维人员不会手工注册每个 Cursor 用户作为 OAuth 客户端。2025-11-25 规范为客户端提供了解决此问题的优先级顺序：如果你有预注册的 `client_id` 就用它，否则使用**客户端 ID 元数据文档**（客户端用自己控制的 HTTPS URL 标识自身，授权服务器拉取元数据），否则回退到 **RFC 7591 动态客户端注册**（客户端推送 `POST /register` 并当场接收 `client_id`），否则提示用户。CIMD 是推荐的默认选项，因为它完全消除了每个服务器的注册，同时保持了基于 DNS 的信任模型；DCR 保留用于向后兼容。两者都从授权服务器的元数据中发现其入口点：CIMD 使用 `client_id_metadata_document_supported`，DCR 使用 `registration_endpoint`。

第二个缺口是密钥轮换。JWT 验证依赖于授权服务器的签名密钥，以 JSON Web Key Set（JWKS）形式发布。授权服务器按计划轮换这些密钥（通常每小时，在事件响应下有时更快）。一个在启动时获取一次 JWKS 的 MCP 服务器在轮换窗口之前验证正常——然后每个请求都失败，直到重启。生产环境中，JWKS 作为缓存值与一个在之前的密钥过期前覆盖缓存的刷新任务配合使用，再加上一个缓存未命中时的回退获取，以处理由比缓存更新的密钥签名的令牌到达的情况。

第三个缺口是受众绑定。第 16 课介绍了 RFC 8707 资源指示器。在生产中，该指示器成为每个请求上的硬声明检查。MCP 服务器将 `token.aud` 与其自身的规范资源 URL 进行比较并拒绝不匹配（HTTP 401）。这是唯一的防线，阻止上游 MCP 服务器（或持有为某个服务器签发的令牌的恶意客户端）对同一信任网格中的另一个服务器重放该令牌。

本课程将每个缺口映射到交互界面的具体部分。元数据文档是一个 HTTP 端点。JWKS 缓存刷新是一个计划任务加上键值缓存。JWT 验证是资源服务器在分发任何工具之前运行的例行程序。保持三个角色分离，每个角色仅强制执行自己拥有的检查：授权服务器签发并轮换密钥，资源服务器缓存并验证，客户端发现并注册。

## 概念

### RFC 8414 — OAuth 授权服务器元数据

位于 `/.well-known/oauth-authorization-server` 的文档描述了客户端所需的一切：

```json
{
  "issuer": "https://auth.example.com",
  "authorization_endpoint": "https://auth.example.com/authorize",
  "token_endpoint": "https://auth.example.com/token",
  "jwks_uri": "https://auth.example.com/.well-known/jwks.json",
  "registration_endpoint": "https://auth.example.com/register",
  "response_types_supported": ["code"],
  "grant_types_supported": ["authorization_code", "refresh_token"],
  "code_challenge_methods_supported": ["S256"],
  "scopes_supported": ["mcp:tools.read", "mcp:tools.invoke"],
  "token_endpoint_auth_methods_supported": ["none", "private_key_jwt"]
}
```

给定 MCP 资源 URL 的客户端链式发现：来自 RFC 9728（资源服务器的文档）的 `oauth-protected-resource` 指定颁发者，然后来自此 RFC 的 `oauth-authorization-server` 指定每个端点。客户端从不硬编码授权 URL。

在信任 IdP 用于 MCP 之前你需要验证的契约：

- `code_challenge_methods_supported` 包含 `S256`（RFC 7636 中的 PKCE）。规范明确：如果此字段**缺失**，则授权服务器不支持 PKCE，客户端**必须**拒绝继续。
- `grant_types_supported` 包含 `authorization_code` 并拒绝 `password` 和 `implicit`。
- 至少有一条注册路径被广告：`client_id_metadata_document_supported: true`（CIMD，首选）**或** `registration_endpoint`（RFC 7591 DCR，备选）。任一满足契约；你不再硬性要求 DCR。
- 对于 OAuth 2.1，`response_types_supported` 恰好是 `["code"]`。

如果 `S256` 缺失，MCP 服务器拒绝针对此 IdP 部署——PKCE 没有降级模式。如果两者注册路径均未广告且你没有预注册的 `client_id`，你也无法注册；部署清单本身有误，而非代码。

### RFC 9728（复习）— 受保护资源元数据

第 16 课涵盖了 RFC 9728。生产中的变化：此文档是客户端寻找被此 MCP 服务器信任的授权服务器的唯一位置。单个 MCP 服务器可以接受来自多个 IdP 的令牌（一个用于员工，一个用于合作伙伴）。RFC 9728 声明该集合；RFC 8414 记录每个 IdP 支持的内容。

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com", "https://partners.example.com"],
  "scopes_supported": ["mcp:tools.invoke"],
  "bearer_methods_supported": ["header"],
  "resource_documentation": "https://notes.example.com/docs"
}
```

### 客户端 ID 元数据文档（推荐的默认选项）

CIMD 将注册从推送转变为拉取。客户端不要求授权服务器生成 `client_id`，而是将自己控制的 HTTPS URL 用作 `client_id`。该 URL 解析为一个 JSON 元数据文档；授权服务器在 OAuth 流程中按需获取它。信任根植于 DNS：如果服务器运维者信任 `app.example.com`，它就信任源自 `https://app.example.com/client.json` 的客户端。没有注册往返，没有可耗尽的 `client_id` 命名空间，没有需要同步的按服务器状态。

客户端托管的元数据文档：

```json
{
  "client_id": "https://app.example.com/oauth/client.json",
  "client_name": "Example MCP Client",
  "client_uri": "https://app.example.com",
  "redirect_uris": ["http://127.0.0.1:7333/callback", "http://localhost:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none"
}
```

文档中的 `client_id` 值**必须**等于它所服务的 URL（授权服务器验证此项；不匹配将被拒绝）。授权服务器在其 RFC 8414 元数据中用 `client_id_metadata_document_supported: true` 宣布对此项的支持。

规范明确指出的两个安全事实：

- **SSRF。** 授权服务器获取攻击者提供的 URL。它必须防范服务器端请求伪造（不获取内部/管理端点）。
- **localhost 冒充。** CIMD 单独无法阻止本地攻击者声称合法客户端的元数据 URL 并绑定任何 `localhost` 重定向。授权服务器**必须**在同意过程中清晰显示重定向 URI 主机名，**并且应当**对仅限 `localhost` 的重定向发出警告。

由于 CIMD 不需要服务器端状态，因此不需要像 DCR 那样设置注册器。客户端是只读的：从一个静态 HTTPS 端点提供你的元数据文档，授权服务器会将其拉取。

### RFC 7591 — 动态客户端注册（备选 / 向后兼容）

DCR 现在是一个 `MAY`，保留用于预 2025-11-25 部署和尚未支持 CIMD 的 IdP 的向后兼容。没有它（也没有 CIMD 或预注册），每个 MCP 客户端（Cursor、Claude Desktop、自定义代理）都需要与 IdP 管理员进行带外交换。有 DCR 的情况下，客户端发送：

```json
POST /register
Content-Type: application/json

{
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none",
  "scope": "mcp:tools.invoke",
  "client_name": "Cursor",
  "software_id": "com.cursor.cursor",
  "software_version": "0.42.0"
}
```

服务器以 `client_id` 和一个 `registration_access_token` 响应以供后续更新：

```json
{
  "client_id": "c_3e7f1a",
  "client_id_issued_at": 1769472000,
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "registration_access_token": "regt_b2...",
  "registration_client_uri": "https://auth.example.com/register/c_3e7f1a"
}
```

`token_endpoint_auth_method: none` 是对在用户设备上运行的 MCP 客户端的正确默认值。他们只获得一个 `client_id`——没有可外泄的 `client_secret`。PKCE 提供了公共客户端所需的持有权证明。

三个生产陷阱：

- 注册端点必须按源 IP 进行速率限制。没有这一点，恶意行为者会脚本化数百万个虚假注册并耗尽 `client_id` 命名空间。在注册器处理请求之前运行速率限制检查。
- 某些企业 IdP 要求 `software_statement`（一个为客户端担保的签名 JWT）。本课程的模拟跳过它；生产环境须连接一个拒绝除 localhost 重定向 URI 外未签名注册的验证步骤。
- `registration_access_token` 必须以哈希形式存储，而非明文。此令牌被盗意味着攻击者可以重写客户端的重定向 URI。

### RFC 8707（复习）— 资源指示器

第 16 课建立了形态。生产规则：每个令牌请求包含 `resource=<canonical-mcp-url>`，并且 MCP 服务器对每个调用验证 `token.aud` 匹配其自身的资源 URL。规范 URI 是服务器的*最具体*标识符：它使用小写方案和主机名，无片段，并且按惯例没有尾部斜杠。路径组件根据规则**不**会被剥离——规范在需要用其识别单个 MCP 服务器时保留它。`https://mcp.example.com`、`https://mcp.example.com/mcp`、`https://mcp.example.com:8443` 和 `https://mcp.example.com/server/mcp` 都是有效的规范 URI。每服务器选一个并将 `aud` 固定到恰为该值。（本课程的模拟为简洁使用了裸主机名受众，如 `https://notes.example.com`；在同一来源下共托管多个 MCP 服务器的部署通过路径来区分。）

### RFC 7636（复习）— PKCE

PKCE 在 OAuth 2.1 中是强制的。本课程的授权码流程始终携带 `code_challenge` 和 `code_verifier`。服务器拒绝任何没有验证器或验证器无法哈希到已存储 challenge 的令牌请求。

### MCP 规范 2025-11-25 认证配置文件

MCP 规范（2025-11-25）精确规定了 MCP 服务器的授权层必须做什么：

- 实现 RFC 9728 受保护资源元数据，并通过 401 上的 `WWW-Authenticate: Bearer resource_metadata="..."` 头部或众所周知 URI `/.well-known/oauth-protected-resource` 提供其位置（SEP-985 通过众所周知备选使头部可选）。元数据 `authorization_servers` 字段**必须**命名至少一个服务器。
- 仅通过**每个**请求上的 `Authorization: Bearer ...` 接受令牌——绝不在查询字符串中，绝不会仅在会话开始时验证。
- 对每个请求验证 `aud`、`iss`、`exp` 和所需范围。服务器**必须**验证令牌是专门为它签发的（受众）；缺失或不匹配的 `aud` 被拒绝，绝不会视为通配符。
- 在 401/403 上，返回承载 `error=...`、`resource_metadata="<PRM-URL>"` 参数（元数据文档的 URL，*不是*裸资源）的 `WWW-Authenticate: Bearer`，并在 `insufficient_scope`（403）上携带 `scope="..."`。注意：该参数是 `resource_metadata`，一个发现指针——challenge 中没有 `resource` 参数。
- 授权服务器发现接受 **RFC 8414 OAuth 元数据或 OpenID Connect 发现 1.0**；客户端必须以优先级顺序尝试两个众所周知的后缀。
- 客户端（而非服务器）防御**混合攻击**：它在重定向前记录预期的 `issuer`，并在兑换代码前验证 `iss` 授权响应参数（RFC 9207）。单独 PKCE 无法阻止混合攻击，因为客户端将其 `code_verifier` 交予任何被导向的令牌端点。

OAuth 2.1 草案是基底；RFC 8414/7591/8707/9728/9207 + RFC 7636 + CIMD 是交互界面；MCP 规范是配置文件。

### IdP 能力矩阵

并非每个 IdP 都支持完整的 MCP 配置文件。下面的矩阵记录了截至 2025-11-25 规范的事实能力陈述。它是一个*部署门槛*，而非推荐。

CIMD 在 2025-11-25 规范中发布，底层 OAuth 草案仅在 2025 年 10 月被采用，因此供应商支持仍在逐步到来——将下面的"CIMD"视为"今日的现状，在你的租户中验证"，而非永久性陈述。

| IdP 类别 | AS 元数据（8414/OIDC） | CIMD | RFC 7591 DCR | RFC 8707 resource | RFC 7636 S256 PKCE | 备注 |
|---|---|---|---|---|---|---|
| 自托管（Keycloak） | 是 | 即将到来 | 是 | 是（自 24.x） | 是 | 本课程中 MCP 配置文件的参考 IdP；完整 DCR 路径端到端，CIMD 跟踪新规范。 |
| 企业 SSO（Microsoft Entra ID） | 是 | 即将到来 | 是（高级层） | 是 | 是 | DCR 可用性因租户层级而异；在部署前在目标租户中验证。 |
| 企业 SSO（Okta） | 是 | 即将到来 | 是（Okta CIC / Auth0） | 是 | 是 | DCR 在 Auth0（现在为 Okta CIC）上可用；经典 Okta 组织需要管理员预注册。 |
| 社交登录 IdP（通用） | 各异 | 否 | 极少 | 极少 | 是 | 大多数社交 IdP 将客户端视为静态合作伙伴；没有自助注册。仅用作身份源，在之上叠加你自己的 MCP 感知授权服务器。 |
| 自定义 / 自研 | 视情况 | 视情况 | 视情况 | 视情况 | 视情况 | 如果你自己发布，发布完整配置文件并首选 CIMD。跳过 PKCE 或受众绑定会破坏 MCP 认证契约。 |

部署清单的拒绝规则：如果选定的 IdP 在 `code_challenge_methods_supported` 中未列出 `S256`，则 MCP 服务器拒绝启动——PKCE 没有降级模式。注册是一个较软的门槛：你需要一个可行的路径（预注册的 `client_id`、`client_id_metadata_document_supported: true`，或者 `registration_endpoint`）。DCR 的缺失本身不再触发拒绝，因为 CIMD 或预注册可以覆盖它。

### JWKS 刷新模式（在 AS 上轮换，在资源服务器上刷新）

将两个动词分开，因为混淆它们是一个真实的生产 bug：

- **轮换**是*授权服务器*做的事：生成新的签名密钥，将其发布到 JWKS 中，稍后淘汰旧密钥。资源服务器与此无关，也不能执行——它不持有 IdP 的私钥。
- **刷新**是*资源服务器*做的事：将已发布的 JWKS 重新 GET 到其缓存中。这是资源服务器唯一执行过的 JWKS 操作。

生产故障模式是过时缓存。使用计划刷新任务加上键值缓存来解决它。资源服务器运行一个任务（cron、计时器，或运行时提供的任何东西），在固定间隔获取 `<issuer>/.well-known/jwks.json` 并覆盖 `cache[issuer] = {keys, fetched_at}`。验证器从该缓存读取。如果一个令牌的 `kid` 在缓存中缺失，会触发**一次**同步刷新作为回退，然后重新检查。这同时处理两种情况：计划刷新，以及由全新密钥签名的令牌在下一次计划刷新之前到达的密钥重叠窗口。

回退**必须是重新获取，绝不能是轮换**。如果你将缓存未命中路径连接到轮换并生成密钥，会立即破坏两件事：（1）生成新密钥产生一个*依然*不匹配令牌的 `kid`，因此查找仍然失败；（2）发起虚假 `kid` 洪流的攻击者会触发无界的密钥创建序列——一次自发的 DoS。重新获取是幂等的，因此一个伪造的 `kid` 最多花费一次无效的获取。

缓存形态：

```json
{
  "https://auth.example.com": {
    "keys": [
      {"kid": "k_2026_03", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"},
      {"kid": "k_2026_04", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"}
    ],
    "fetched_at": 1772668800
  }
}
```

同时存在两个密钥是稳态。授权服务器通过在淘汰旧密钥前引入下一个密钥来轮换，这样在旧密钥下签发的令牌在过期之前仍然有效。缓存持有并集；验证器按 `kid` 选择。

### 验证例程

MCP 服务器在分发任何工具之前运行验证。`code/main.py` 使用的形态：

```python
result = server.validate(bearer_token, required_scope="mcp:tools.invoke")
if not result["valid"]:
    return {"status": result["status"], "WWW-Authenticate": result["www_authenticate"]}
```

`validate` 解码 JWT，从 JWKS 缓存解析签名密钥（在 miss 时刷新一次），验证签名，然后根据白名单检查 `iss`，根据此服务器的规范资源检查 `aud`，检查 `exp` 和所需范围——在第一次失败时返回 `WWW-Authenticate` 挑战。将其作为资源服务器上的单个例程意味着每个入口点（每个工具调用，每种传输）都经过相同的检查；没有绕过验证到达工具的路径。

### 受众重放演练（访问令牌权限限制）

服务器 A（`notes.example.com`）和服务器 B（`tasks.example.com`）都在相同的授权服务器上注册。服务器 A 被攻破。攻击者获取用户的笔记令牌并向服务器 B 重放。

服务器 B 的验证器：

1. 解码 JWT，按 `kid` 获取 JWKS，验证签名。
2. 根据其受保护资源元数据的 `authorization_servers` 检查 `iss`。（通过——同一个 IdP。）
3. 检查 `aud == "https://tasks.example.com"`。（失败——令牌的 `aud` 是 `https://notes.example.com`。）
4. 返回 401 并带有 `WWW-Authenticate: Bearer error="invalid_token", error_description="audience mismatch", resource_metadata="https://tasks.example.com/.well-known/oauth-protected-resource"`。

受众声明是在协议层对抗此攻击的唯一防御措施。因性能跳过它是生产中最常见的错误；验证器必须在每个请求上运行，而不仅仅在会话开始时。规范将此称为**访问令牌权限限制**：一个 MCP 服务器**必须**拒绝任何未将其命名为受众的令牌。

> **命名说明。** 规范将混淆代理这一术语保留给一个相关但不同的问题：一个 MCP 服务器作为 OAuth **代理**作用于第三方 API，使用静态客户端 ID，在没有获得每个客户端用户同意的情况下转发令牌。受众绑定修复了上述重放问题；混淆代理的修复是每个客户端同意**加上**绝不将入站令牌传递给上游 API（MCP 服务器**必须**获取自己独立的上游令牌）。

### 混合攻击（服务器无法提供的客户端侧防御）

客户端在其生命周期中与许多授权服务器通信。恶意 AS 可以试图让客户端在攻击者的令牌端点上兑换诚实 AS 的授权码。受众绑定在这里无济于事——攻击发生在任何令牌存在之前。防御存在于客户端（RFC 9207）：

1. 在重定向前，客户端从已验证的 AS 元数据记录预期的 `issuer`。
2. 在授权响应时，客户端比较返回的 `iss` 参数与已记录的 issuer（简单字符串比较，无标准化），然后再将代码发送到任何地方。
3. 不匹配（或者 `iss` 缺失但 AS 广告了 `authorization_response_iss_parameter_supported`）→ 拒绝，甚至不显示 `error` 字段。

单独 PKCE 无法阻止混合攻击，因为客户端将其 `code_verifier` 交给了任何被导向的令牌端点。这就是规范按请求记录 issuer 与 PKCE verifier 和 `state` 并存的原因。

### 故障模式

- **过时 JWKS。** 验证器在 AS 轮换密钥后拒绝有效令牌。修复方案是上面的 cron-refresh + cache-miss-refetch 模式。绝不要在没有刷新任务的情况下缓存 JWKS。
- **轮换作为回退。** 将缓存未命中路径连接到轮换并生成密钥而非重新获取是一个真实的 bug：它永远不会产生缺失的 `kid`，并且会将攻击者控制的 `kid` 值转变为密钥创建 DoS。回退必须是幂等的 `refresh-jwks`。
- **缺失 `aud` 声明。** 某些 IdP 默认省略 `aud`，除非 `resource` 出现在令牌请求中。验证器必须拒绝缺失 `aud` 的令牌，不将缺失视为通配符。
- **通过缺失 `iss` 检查导致的混合攻击。** 不根据重定向前记录的 issuer 验证 RFC 9207 `iss` 授权响应参数的客户端可能被导向攻击者的令牌端点来兑换诚实 AS 的代码。这是一个客户端侧失败；资源服务器无法弥补。
- **范围升级竞争。** 同一用户的两次并发逐步升级流程可能都成功并产生两个不同范围的访问令牌。验证器必须使用请求中呈现的令牌，而不是查找"用户的当前范围"——这会产生 TOCTOU 窗口。
- **注册令牌被盗。** 泄露的 `registration_access_token` 让攻击者可以重写重定向 URI。这些在存储时加盐；要求客户端在每次更新时呈现明文；有嫌疑时轮换。
- **`iss` 未固定。** 一个接受任何 `iss` 的验证器让攻击者可以搭建自己的授权服务器，为目标受众注册客户端，并签发令牌。受保护资源元数据的 `authorization_servers` 列表是白名单；强制执行它。

## 使用

`code/main.py` 使用标准库 Python 和三个角色——`AuthorizationServer`、`ResourceServer` 和 `Client`——演练完整生产流程。流程：

1. 授权服务器在 `/.well-known/oauth-authorization-server` 上发布 RFC 8414 元数据。
2. MCP 客户端调用元数据端点并检查其注册选项（`client_id_metadata_document_supported` 用于 CIMD，`registration_endpoint` 用于 DCR）和 `S256` PKCE 支持。
3. 演练采用 DCR 回退路径：客户端 POST 到 `/register`（RFC 7591）并接收 `client_id`。（CIMD 客户端则会改为呈现自己的 HTTPS `client_id` URL 并跳过此步骤。）
4. MCP 客户端运行受 PKCE 保护的授权码流程（RFC 7636），带有 `resource` 指示器（RFC 8707）。
5. MCP 客户端使用 `Authorization: Bearer ...` 调用 MCP 服务器上的工具。
6. MCP 服务器运行 `validate`，从 JWKS 缓存解析签名密钥。
7. IdP 轮换一个密钥；计划刷新将 JWKS 重新拉取到缓存中。
8. 下一次调用对刷新后的密钥进行验证而无需重启，并且先前的令牌在重叠窗口期间仍然验证通过。
9. 针对不同 MCP 资源的受众重放尝试获得带有 `audience mismatch` 和 `resource_metadata` 指针的 401。

这里的 JWT 使用 HS256 配合共享秘密（以便课程仅使用标准库运行）。生产环境使用 RS256 或 EdDSA 配合上述的 JWKS 模式；验证逻辑在其他方面完全相同。由于 IdP 和资源服务器在单个进程中存活，`refresh_jwks` 直接读取授权服务器的密钥列表；在真实网络中，它是向 `jwks_uri` 发出的 HTTP `GET` 请求。

## 交付物

本课程产出 `outputs/skill-mcp-auth.md`。给定 MCP 服务器配置和 IdP 能力集，该技能发出需要搭建的认证交互界面：受保护资源元数据、要使用的注册路径（CIMD、预注册或 DCR 回退）、JWKS 刷新计划、范围映射，以及在 IdP 不支持完整 RFC 配置文件时应用的拒绝规则。

## 练习

1. 运行 `code/main.py`。跟踪流程。注意 IdP 如何在步骤 6 中轮换密钥，计划 `refresh_jwks` 重新拉取已发布的集合，以及旧令牌（重叠窗口）和新令牌都无需重启即可验证。

2. 向受保护资源元数据的 `authorization_servers` 列表添加一个新的 IdP。签发一个由新 IdP 签名的令牌并确认验证器接受它。签发一个由未列出 IdP 签名的令牌并确认验证器拒绝并带有 `WWW-Authenticate: Bearer error="invalid_token", error_description="iss not allowed"`。

3. 在 `register_client` 上添加一个在注册器接受请求之前运行的速率限制检查。使用一个按源 IP 的令牌桶，存储在一个按 IP 键的小字典中。

4. 阅读 RFC 7591 并找出本课程的 `/register` 处理程序未验证的两个字段。添加验证。（提示：`software_statement` 和 `redirect_uris` URI 方案。）

5. 添加一个客户端 ID 元数据文档路径。提供一个 `client.json`，其 `client_id` 等于自身的 URL，并让授权服务器获取并验证它（如果 `client_id` ≠ URL 则拒绝）。确认 CIMD 客户端无需 `register_client` 调用即可注册。

6. 证明 DoS 修复。向验证器发送一个具有随机 `kid` 的令牌，确认 `refresh_jwks` 最多运行一次，且授权服务器的密钥数量没有增长。然后故意将回退重新连接到轮换并生成密钥，观察每个虚假令牌密钥数量的增长——之后将回退恢复为重新获取。

7. 实现混合攻击部分的客户端侧 RFC 9207 `iss` 检查：在授权请求前记录预期的 issuer，然后拒绝 `iss` 不匹配的授权响应。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|----------|
| ASM | "OAuth 元数据文档" | RFC 8414 `/.well-known/oauth-authorization-server` JSON |
| CIMD | "客户端元数据 URL" | 客户端 ID 元数据文档——一个用作 `client_id` 的 HTTPS URL；AS 拉取 JSON。自 2025-11-25 起推荐的默认选项 |
| DCR | "自助客户端注册" | RFC 7591 `POST /register` 流程；在 2025-11-25 中降级为 `MAY` 回退 |
| JWKS | "用于 JWT 验证的公钥" | JSON Web Key Set，从 `jwks_uri` 获取，按 `kid` 索引 |
| 轮换 vs 刷新 | "更新密钥" | *轮换* = AS 生成/淘汰签名密钥；*刷新* = 资源服务器重新获取已发布的集合。资源服务器仅执行刷新 |
| 资源指示器 | "受众参数" | RFC 8707 `resource` 参数，将令牌固定到一个服务器 |
| `aud` 声明 | "受众" | 验证器将其与规范资源 URL 进行比较的 JWT 声明 |
| 受众重放 | "令牌重放" | 为服务器 A 签发的令牌被提交给服务器 B；由受众验证防御（规范：访问令牌权限限制） |
| 混淆代理 | "代理令牌滥用" | 一个使用静态客户端 ID 的 MCP 代理在没有每个客户端同意的情况下转发令牌；与受众重放不同 |
| 混合攻击 | "错误的令牌端点" | 客户端被导向攻击者的端点兑换诚实 AS 的代码；由客户端通过 RFC 9207 `iss` 防御 |
| `iss` 白名单 | "受信任的授权服务器" | 受保护资源元数据的 `authorization_servers` 中命名的集合 |
| `resource_metadata` | "在哪里找到 PRM 文档" | 在 401/403 上命名 RFC 9728 元数据 URL 的 `WWW-Authenticate` 参数 |
| 公共客户端 | "原生或浏览器客户端" | 没有 `client_secret` 的 OAuth 客户端；PKCE 补偿 |
| `WWW-Authenticate` | "401/403 响应头部" | 携带 `Bearer error=...` 指令，驱动客户端恢复 |

## 进一步阅读

- [MCP — 授权规范（2025-11-25）](https://modelcontextprotocol.io/specification/2025-11-25/basic/authorization) — 本课程实现的 MCP 认证配置文件
- [MCP 博客 — MCP 一周年：2025 年 11 月规范发布](https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/) — 2025-11-25 中的变更（CIMD、XAA、DCR 降级）
- [Aaron Parecki — 2025 年 11 月 MCP 授权规范中的客户端注册](https://aaronparecki.com/2025/11/25/1/mcp-authorization-spec-update) — CIMD 优于 DCR 的理由
- [OAuth 客户端 ID 元数据文档（draft-ietf-oauth-client-id-metadata-document-00）](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-client-id-metadata-document-00) — CIMD
- [RFC 8414 — OAuth 2.0 授权服务器元数据](https://datatracker.ietf.org/doc/html/rfc8414) — 发现契约
- [RFC 7591 — OAuth 2.0 动态客户端注册协议](https://datatracker.ietf.org/doc/html/rfc7591) — DCR（回退路径）
- [RFC 7636 — 代码交换的证明密钥（PKCE）](https://datatracker.ietf.org/doc/html/rfc7636) — 公共客户端持有权证明
- [RFC 8707 — OAuth 2.0 的资源指示器](https://datatracker.ietf.org/doc/html/rfc8707) — 受众固定
- [RFC 9728 — OAuth 2.0 受保护资源元数据](https://datatracker.ietf.org/doc/html/rfc9728) — 资源服务器发现
- [RFC 9207 — OAuth 2.0 授权服务器颁发者标识](https://datatracker.ietf.org/doc/html/rfc9207) — 防御混合攻击的 `iss` 参数
- [OAuth 2.1 草案](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-v2-1) — 综合 OAuth 基底
