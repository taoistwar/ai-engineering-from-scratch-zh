# 根和作用域与引导 — 作用域和飞行中用户输入

> 硬编码路径在用户打开不同项目时就失效了。预填的工具参数在用户未充分指定时失效。根作用域将服务器限制在用户控制的一组 URI 内；引导在工具调用中途暂停，通过表单或 URL 向用户请求结构化输入。两个客户端原语，两个针对常见 MCP 故障模式的修复。SEP-1036（URL 模式引导，2025-11-25）在 2026 年上半年期间是实验性的——在使用前检查 SDK 版本。

**Type:** Build
**Languages:** Python（stdlib，根作用域 + 引导演示）
**Prerequisites:** Phase 13 · 07（MCP 服务器）
**Time:** ~45 分钟

## 学习目标

- 声明 `roots` 并响应 `notifications/roots/list_changed`。
- 将服务器文件操作限制在声明的根作用域 URI 集合内。
- 使用 `elicitation/create` 在工具调用中途向用户请求确认或结构化输入。
- 在表单模式和 URL 模式引导之间选择（后者是实验性的；标注漂移风险）。

## 问题

在生产中，笔记 MCP 服务器会遇到两个具体故障。

**损坏的路径假设。** 服务器针对 `~/notes` 编写。用户在不同机器上，笔记位于 `~/Documents/Notes`，工具调用静默失败（找不到文件）或更糟——写入了错误的位置。

**用户知道的缺失参数。** 用户说"删除旧的 TPS 报告笔记"。模型调用 `notes_delete(title: "TPS report")` 但有三个匹配的笔记，分别来自 2023、2024 和 2025。工具无从猜测。返回"模糊"令人烦恼；全部运行则是灾难性的。

根作用域解决第一个：客户端在 `initialize` 时声明服务器可以触碰的 URI 集合。引导解决第二个：服务器暂停工具调用并发送 `elicitation/create` 让用户选择哪一个。

## 概念

### 根作用域

客户端在 `initialize` 时声明根列表：

```json
{
  "capabilities": {"roots": {"listChanged": true}}
}
```

服务器然后可以调用 `roots/list`：

```json
{"roots": [{"uri": "file:///Users/alice/Documents/Notes", "name": "Notes"}]}
```

服务器必须将根作用域视为边界：任何在根作用域集合外的文件读写都被拒绝。这不由客户端强制执行（服务器仍然是用户信任的代码），但符合规范的服务器会遵守。

当用户添加或删除根时，客户端发送 `notifications/roots/list_changed`。服务器重新调用 `roots/list` 并更新其边界。

### 根作用域为什么是客户端原语

根作用域由客户端声明，因为它们代表用户的同意模型。用户告诉 Claude Desktop "给此笔记服务器访问这两个目录的权限"。服务器不能扩大该范围。

### 引导：默认的表单模式

`elicitation/create` 接受一个表单模式加上自然语言提示：

```json
{
  "method": "elicitation/create",
  "params": {
    "message": "删除 'TPS report'？多个笔记匹配；选择一个。",
    "requestedSchema": {
      "type": "object",
      "properties": {
        "note_id": {
          "type": "string",
          "enum": ["note-3", "note-7", "note-14"]
        },
        "confirm": {"type": "boolean"}
      },
      "required": ["note_id", "confirm"]
    }
  }
}
```

客户端渲染一个表单，收集用户的答案，返回：

```json
{
  "action": "accept",
  "content": {"note_id": "note-14", "confirm": true}
}
```

三种可能的操作：`accept`（用户填写完毕）、`decline`（用户关闭了它）、`cancel`（用户中止了整个工具调用）。

表单模式是扁平的——嵌套对象在 v1 中不受支持。SDK 通常会拒绝任何比单层更复杂的内容。

### 引导：URL 模式（SEP-1036，实验性）

2025-11-25 新增功能。替代模式，服务器发送一个 URL：

```json
{
  "method": "elicitation/create",
  "params": {
    "message": "登录 GitHub",
    "url": "https://github.com/login/oauth/authorize?client_id=..."
  }
}
```

客户端在浏览器中打开 URL，等待完成，在用户返回后响应。适用于表单不足的 OAuth 流程、支付授权和文档签署场景。

漂移风险说明：SEP-1036 的响应形态仍在稳定中；一些 SDK 返回回调 URL，另一些返回完成令牌。在将 URL 模式用于生产前请阅读 SDK 的发布说明。

### 引导在何种情况下是正确的工具

- 破坏性操作前的用户确认（破坏性提示 + 引导）。
- 消歧（选择 N 个候选项中的一个）。
- 首次运行设置（API 密钥、目录、偏好）。
- OAuth 风格流程（URL 模式）。

### 引导在何种情况下是错误的

- 填写模型本可以用自然语言要求的工具必需参数。使用正常的重新提示，而非引导对话框。
- 高频调用。引导会中断对话；不要在循环中触发它。
- 任何服务器可以在事后验证的内容。验证，返回错误，让模型以文本形式向用户提问。

### 人机协同桥梁

引导和采样一起实现了 MCP 的"人机协同"模型。服务器的代理循环可以暂停以获取用户输入（引导）或模型推理（采样）。第 13 阶段 · 11 覆盖了采样；本课程覆盖引导。将它们组合以实现完整的中途循环控制。

## 使用

`code/main.py` 扩展了笔记服务器，添加了：

- `roots/list` 响应，服务器在 root-list-changed 通知后重新查询。
- 一个 `notes_delete` 工具，当多个笔记匹配时使用 `elicitation/create` 进行消歧。
- 一个 `notes_setup` 工具，使用 URL 模式引导打开首次运行配置页面（模拟）。
- 一个边界检查，拒绝在声明根作用域外的 URI 上操作。

演示运行三种场景：顺利路径（一个匹配）、消歧（三个匹配，引导触发）、超出根作用域写入（被拒绝）。

## 交付物

本课程产出 `outputs/skill-elicitation-form-designer.md`。给定一个可能需要用户确认或消歧的工具，该技能设计引导表单模式和消息模板。

## 练习

1. 运行 `code/main.py`。触发消歧路径；确认模拟的用户答案被路由回工具。

2. 添加一个新工具 `notes_archive`，每次都要求引导确认（破坏性提示）。检查 UX：这与模型以文本形式重新询问相比如何？

3. 为首次运行 OAuth 流程实现 URL 模式引导。注意漂移风险并添加 SDK 版本保护。

4. 扩展 `roots/list` 处理：当通知到达时，服务器应原子性地重新读取并重新扫描现在可能超出范围的打开文件句柄。

5. 阅读 GitHub 上的 SEP-1036 问题讨论帖。找出一个影响服务器应如何处理 URL 模式回调的未解决问题。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|----------|
| 根作用域 | "同意边界" | 客户端已允许服务器触碰的 URI |
| `roots/list` | "服务器请求范围" | 客户端返回当前根作用域集合 |
| `notifications/roots/list_changed` | "用户更改了范围" | 客户端信号：根作用域集合已变更 |
| 引导 | "中途向用户提问" | 服务器发起的请求，获取结构化用户输入 |
| `elicitation/create` | "该方法" | 引导请求的 JSON-RPC 方法 |
| 表单模式 | "模式驱动的表单" | 扁平 JSON Schema 在客户端 UI 中渲染为表单 |
| URL 模式 | "浏览器重定向" | SEP-1036 实验性；打开 URL 并等待 |
| `accept` / `decline` / `cancel` | "用户响应结果" | 服务器处理的三个分支 |
| 消歧 | "选择一个" | 工具具有 N 个候选项时的常见引导用例 |
| 扁平表单 | "仅顶层属性" | 引导模式不支持嵌套 |

## 进一步阅读

- [MCP — 客户端根作用域规范](https://modelcontextprotocol.io/specification/draft/client/roots) — 规范根作用域参考
- [MCP — 客户端引导规范](https://modelcontextprotocol.io/specification/draft/client/elicitation) — 规范引导参考
- [Cisco — MCP 引导、结构化内容、OAuth 增强的新增内容](https://blogs.cisco.com/developer/whats-new-in-mcp-elicitation-structured-content-and-oauth-enhancements) — 2025-11-25 新增功能演练
- [MCP — GitHub SEP-1036](https://github.com/modelcontextprotocol/modelcontextprotocol) — URL 模式引导提案（实验性，漂移风险）
- [The New Stack — 引导如何为 AI 工具带来人机协同](https://thenewstack.io/how-elicitation-in-mcp-brings-human-in-the-loop-to-ai-tools/) — UX 演练
