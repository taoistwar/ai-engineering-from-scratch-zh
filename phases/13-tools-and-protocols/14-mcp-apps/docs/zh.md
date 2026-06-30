# MCP Apps — 通过 `ui://` 实现交互式 UI 资源

> 纯文本工具输出限制了代理可以展示的内容。MCP Apps（SEP-1724，2026 年 1 月 26 日正式发布）允许工具返回内联渲染在 Claude Desktop、ChatGPT、Cursor、Goose 和 VS Code 中的沙盒化交互式 HTML。仪表盘、表单、地图、3D 场景，全部通过一个扩展实现。本课程演练 `ui://` 资源方案、`text/html;profile=mcp-app` MIME、iframe-sandbox postMessage 协议以及随之而来的让服务器渲染 HTML 的安全界面。

**Type:** Build
**Languages:** Python（stdlib，UI 资源发射器），HTML（示例应用）
**Prerequisites:** Phase 13 · 07（MCP 服务器），Phase 13 · 10（资源）
**Time:** ~75 分钟

## 学习目标

- 从工具调用返回 `ui://` 资源并设置正确的 MIME 和元数据。
- 使用 `_meta.ui.resourceUri`、`_meta.ui.csp` 和 `_meta.ui.permissions` 声明工具的关联 UI。
- 实现 iframe 沙箱 postMessage JSON-RPC 用于 UI 到宿主的通信。
- 应用 CSP 和 permissions-policy 默认值以防御 UI 发起的攻击。

## 问题

2025 年的 `visualize_timeline` 工具可以返回"这是按时间顺序排列的 14 条笔记：……"。那只是一段话。用户真正想要的是交互式时间线。在 MCP Apps 之前，选项是：客户端特定的组件 API（Claude artifacts、OpenAI Custom GPT HTML），或者根本没有 UI。

MCP Apps（SEP-1724，2026 年 1 月 26 日发布）标准化了契约。工具结果包含一个 `resource`，其 URI 为 `ui://...`，MIME 为 `text/html;profile=mcp-app`。宿主在沙盒化 iframe 中渲染它，具有有限的 CSP，除非明确授权否则无网络访问。iframe 内的 UI 通过一个微小的 postMessage JSON-RPC 方言向宿主发送消息。

每个兼容的客户端（Claude Desktop、ChatGPT、Goose、VS Code）以相同方式渲染相同的 `ui://` 资源。一个服务器，一个 HTML 包，通用 UI。

## 概念

### `ui://` 资源方案

工具返回：

```json
{
  "content": [
    {"type": "text", "text": "这是你的笔记时间线："},
    {"type": "ui_resource", "uri": "ui://notes/timeline"}
  ],
  "_meta": {
    "ui": {
      "resourceUri": "ui://notes/timeline",
      "csp": {
        "defaultSrc": "'self'",
        "scriptSrc": "'self' 'unsafe-inline'",
        "connectSrc": "'self'"
      },
      "permissions": []
    }
  }
}
```

宿主然后对 `ui://notes/timeline` URI 调用 `resources/read`，并获得：

```json
{
  "contents": [{
    "uri": "ui://notes/timeline",
    "mimeType": "text/html;profile=mcp-app",
    "text": "<!doctype html>..."
  }]
}
```

### Iframe 沙箱

宿主在沙盒化 `<iframe>` 内渲染 HTML，带有：

- `sandbox="allow-scripts allow-same-origin"`（或根据服务器声明更严格）
- 通过响应头部应用的服务器声明的 CSP。
- 没有来自宿主源的 cookies 和 localStorage。
- 网络访问限于 CSP 中的 `connectSrc`。

### postMessage 协议

iframe 通过 `window.postMessage` 与宿主通信。一个微小的 JSON-RPC 2.0 方言：

始终将 `targetOrigin` 固定为对等方的确切来源，并在接收端验证 `event.origin` 是否在白名单内，然后再处理任何负载。绝不要在此通道的任何一方使用 `"*"`——主体携带工具调用和资源读取。

```js
// iframe 到宿主（固定到宿主来源）
window.parent.postMessage({
  jsonrpc: "2.0",
  id: 1,
  method: "host.callTool",
  params: { name: "notes_update", arguments: { id: "note-14", title: "..." } }
}, "https://host.example.com");

// 宿主到 iframe（固定到 iframe 来源）
iframe.contentWindow.postMessage({
  jsonrpc: "2.0",
  id: 1,
  result: { content: [...] }
}, "https://iframe.example.com");

// 双方的接收方
window.addEventListener("message", (event) => {
  if (event.origin !== "https://expected-peer.example.com") return;
  // 安全处理 event.data
});
```

UI 可以调用的可用宿主端方法：

- `host.callTool(name, arguments)` — 调用服务器工具。
- `host.readResource(uri)` — 读取 MCP 资源。
- `host.getPrompt(name, arguments)` — 获取提示词模板。
- `host.close()` — 关闭 UI。

每个调用仍然通过 MCP 协议并继承服务器的权限。

### 权限

`_meta.ui.permissions` 列表请求额外能力：

- `camera` — 访问用户摄像头（用于扫描文档 UI）。
- `microphone` — 语音输入。
- `geolocation` — 定位。
- `network:*` — 比仅 `connectSrc` 更广泛的网络访问。

每个权限是用户在 UI 渲染前看到的提示。

### 安全风险

iframe 中的 HTML 仍然是 HTML。新的攻击面：

- **通过 UI 的提示词注入。** 恶意服务器 UI 可以显示看起来像系统消息的文本并欺骗用户。宿主渲染应在视觉上区分服务器 UI 与宿主 UI。
- **通过 `connectSrc` 的外泄。** 如果 CSP 允许 `connect-src: *`，UI 可以向任何地方发送数据。默认应严格。
- **点击劫持。** UI 覆盖宿主 chrome。宿主必须防止 z-index 操纵并强制不透明度规则。
- **窃取焦点。** UI 获取键盘焦点并捕获下一条消息。宿主必须拦截。

第 13 阶段 · 15 作为 MCP 安全的一部分深入涵盖了这些内容；本课程介绍它们。

### `ui/initialize` 握手

iframe 加载后，通过 postMessage 发送 `ui/initialize`：

```json
{"jsonrpc": "2.0", "id": 0, "method": "ui/initialize",
 "params": {"theme": "dark", "locale": "en-US", "sessionId": "..."}}
```

宿主用能力和会话令牌响应。UI 在后续每个宿主调用中使用会话令牌。

### AppRenderer / AppFrame SDK 原语

ext-apps SDK 公开了两个便捷原语：

- `AppRenderer`（服务器端）— 包装 React / Vue / Solid 组件并发出带有正确 MIME 和元数据的 `ui://` 资源。
- `AppFrame`（客户端）— 接收资源，挂载 iframe，并中介 postMessage。

你可以使用这些或手工编写 HTML 和 JSON-RPC。

### 生态系统状态

MCP Apps 于 2026 年 1 月 26 日发布。截至 2026 年 4 月的客户端支持：

- **Claude Desktop。** 自 2026 年 1 月起完全支持。
- **ChatGPT。** 通过 Apps SDK 完全支持（相同的底层 MCP Apps 协议）。
- **Cursor。** Beta 版；通过设置启用。
- **VS Code。** 仅 Insider 构建。
- **Goose。** 完全支持。
- **Zed、Windsurf。** 已规划路线。

生产中的服务器：仪表盘、地图可视化、数据表、图表构建器、沙盒 IDE 预览。

## 使用

`code/main.py` 扩展了笔记服务器，添加了返回 `ui://notes/timeline` 资源的 `visualize_timeline` 工具，以及在 `resources/read` 上返回带有 SVG 时间线的小而完整 HTML 包的处理程序。HTML 是标准库模板化的——没有构建系统。postMessage 在 JS 注释中绘制——因为标准库无法驱动浏览器。

关注要点：

- 工具响应上的 `_meta.ui` 携带 resourceUri、CSP、permissions。
- HTML 在没有网络访问的情况下渲染；所有数据内联。
- JS 通过 `window.parent.postMessage` 调用 `host.callTool`（在此标准库演示中记录但无实际动作）。

## 交付物

本课程产出 `outputs/skill-mcp-apps-spec.md`。给定一个将受益于交互式 UI 的工具，该技能生成完整的 MCP Apps 契约：`ui://` URI、CSP、permissions、postMessage 入口点和安全清单。

## 练习

1. 运行 `code/main.py` 并检查发出的 HTML。直接在浏览器中打开 HTML；验证 SVG 渲染。然后绘制 UI 将用于调用 `host.callTool("notes_update", ...)` 的 postMessage 契约。

2. 收紧 CSP：移除 `'unsafe-inline'` 并使用基于 nonce 的脚本策略。HTML 生成代码中有什么变化？

3. 添加第二个 UI 资源 `ui://notes/editor`，包含一个用于原地编辑笔记的表单。当用户提交时，iframe 调用 `host.callTool("notes_update", ...)`。

4. 审计 UI 的攻击面。恶意服务器可能在哪里注入内容？iframe 沙箱防御了什么，没有防御什么？

5. 阅读 SEP-1724 规范并找出此玩具实现未使用的 MCP Apps SDK 中的一个能力。（提示：组件级状态同步。）

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|----------|
| MCP Apps | "交互式 UI 资源" | SEP-1724 扩展，2026-01-26 发布 |
| `ui://` | "应用 URI 方案" | UI 包的资源方案 |
| `text/html;profile=mcp-app` | "该 MIME" | MCP 应用 HTML 的内容类型 |
| Iframe 沙箱 | "渲染容器" | 使用 CSP 和权限的 UI 的浏览器沙盒 |
| postMessage JSON-RPC | "UI 到宿主的线协议" | 用于宿主调用的微小 JSON-RPC-over-postMessage 方言 |
| `_meta.ui` | "工具-UI 绑定" | 将工具结果链接到 UI 资源的元数据 |
| CSP | "内容安全策略" | 声明脚本、网络、样式的允许来源 |
| AppRenderer | "服务器 SDK 原语" | 将框架组件转换为 `ui://` 资源 |
| AppFrame | "客户端 SDK 原语" | 中介 postMessage 的 iframe 挂载助手 |
| `ui/initialize` | "握手" | 从 UI 到宿主的第一个 postMessage |

## 进一步阅读

- [MCP ext-apps — GitHub](https://github.com/modelcontextprotocol/ext-apps) — 参考实现和 SDK
- [MCP Apps 规范 2026-01-26](https://github.com/modelcontextprotocol/ext-apps/blob/main/specification/2026-01-26/apps.mdx) — 正式规范文档
- [MCP — Apps 扩展概述](https://modelcontextprotocol.io/extensions/apps/overview) — 高层文档
- [MCP 博客 — MCP Apps 发布](https://blog.modelcontextprotocol.io/posts/2026-01-26-mcp-apps/) — 2026 年 1 月发布帖
- [MCP Apps API 参考](https://apps.extensions.modelcontextprotocol.io/api/) — JSDoc 风格 SDK 参考
