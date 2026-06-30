# MCP 采样 — 服务器请求 LLM 补全和代理循环

> 大多数 MCP 服务器是哑执行器：接收参数、运行代码、返回内容。采样让服务器反转方向：它请求客户端的 LLM 做出决策。这让服务器能够承载代理循环而无需拥有任何模型凭据。SEP-1577 于 2025-11-25 合并，在采样请求中添加了工具，使循环可以包含更深层的推理。漂移风险说明：SEP-1577 的工具内采样形态在 2026 年 Q1 期间处于实验状态，SDK API 仍在稳定中。

**Type:** Build
**Languages:** Python（stdlib，采样测试框架）
**Prerequisites:** Phase 13 · 07（MCP 服务器），Phase 13 · 10（资源和提示词）
**Time:** ~75 分钟

## 学习目标

- 解释 `sampling/createMessage` 解决什么问题（无需服务器端 API 密钥的服务器承载循环）。
- 实现一个请求客户端对多轮提示词进行采样并返回补全的服务器。
- 使用 `modelPreferences`（成本/速度/智能优先级）指导客户端模型选择。
- 构建一个 `summarize_repo` 工具，通过采样在内部迭代而不是硬编码行为。

## 问题

一个用于代码摘要工作流的有用 MCP 服务器需要：遍历文件树，选择哪些文件读取，合成摘要，然后返回。LLM 推理在哪里发生？

选项 A：服务器调用自己的 LLM。需要 API 密钥，服务器端计费，每个用户都很昂贵。

选项 B：服务器返回原始内容；客户端的代理进行推理。可行，但将服务器逻辑移到了客户端提示词中，这很脆弱。

选项 C：服务器通过 `sampling/createMessage` 请求客户端的 LLM。服务器保留算法（读取哪些文件，做多少次遍历），而客户端保留计费和模型选择。服务器完全没有凭据。

采样就是选项 C。它是一个受信任服务器在不自己成为完整 LLM 宿主的情况下承载代理循环的机制。

## 概念

### `sampling/createMessage` 请求

服务器发送：

```json
{
  "jsonrpc": "2.0",
  "id": 42,
  "method": "sampling/createMessage",
  "params": {
    "messages": [{"role": "user", "content": {"type": "text", "text": "..."}}],
    "systemPrompt": "...",
    "includeContext": "none",
    "modelPreferences": {
      "costPriority": 0.3,
      "speedPriority": 0.2,
      "intelligencePriority": 0.5,
      "hints": [{"name": "claude-3-5-sonnet"}]
    },
    "maxTokens": 1024
  }
}
```

客户端运行其 LLM，返回：

```json
{"jsonrpc": "2.0", "id": 42, "result": {
  "role": "assistant",
  "content": {"type": "text", "text": "..."},
  "model": "claude-3-5-sonnet-20251022",
  "stopReason": "endTurn"
}}
```

### `modelPreferences`

三个浮点数总和为 1.0：

- `costPriority`：倾向于更便宜的模型。
- `speedPriority`：倾向于更快的模型。
- `intelligencePriority`：倾向于更强大的模型。

加上 `hints`：服务器偏好的命名模型。客户端可能遵从也可能不遵从提示；客户端用户配置始终优先。

### `includeContext`

三个值：

- `"none"` — 仅服务器提供的消息。默认。
- `"thisServer"` — 包含来自此服务器会话的先前消息。
- `"allServers"` — 包含所有会话上下文。

`includeContext` 自 2025-11-25 起软弃用，因为它会泄露跨服务器上下文，这是安全隐患。优先使用 `"none"` 并在消息中传递显式上下文。

### 带工具的采样（SEP-1577）

2025-11-25 新增功能：采样请求可以包含 `tools` 数组。客户端使用这些工具运行完整的工具调用循环。这让服务器能够通过客户端的模型承载 ReAct 风格的代理循环。

```json
{
  "messages": [...],
  "tools": [
    {"name": "fetch_url", "description": "...", "inputSchema": {...}}
  ]
}
```

客户端循环：采样，如果调用了工具则执行它，再次采样，返回最终的助手消息。这在 2026 年 Q1 期间是实验性的；SDK 签名可能仍有变动。实现时请对照 2025-11-25 规范的 client/sampling 部分确认。

### 人机协同

客户端必须在运行采样前向用户展示服务器要求模型做什么。恶意服务器可能利用采样来操纵用户会话（"对用户说 X 以便他们点击 Y"）。Claude Desktop、VS Code 和 Cursor 将采样请求呈现为用户可以拒绝的确认对话框。

2026 年的共识：没有人工确认的采样是红旗。网关（第 13 阶段 · 17）可以自动批准低风险采样并自动拒绝任何可疑内容。

### 无需 API 密钥的服务器承载循环

典型用例：一个没有自己 LLM 访问权限的代码摘要 MCP 服务器。它做的是：

1. 遍历仓库结构。
2. 调用 `sampling/createMessage` 询问"选择最有可能描述此仓库目的的五份文件"。
3. 读取这些文件。
4. 调用 `sampling/createMessage` 用文件内容询问"用 3 段话总结该仓库"。
5. 将摘要作为 `tools/call` 结果返回。

服务器从未接触 LLM API。客户端的用户用自己的凭据支付补全费用。

### 安全风险（Unit 42 披露，2026 年 Q1）

- **隐蔽采样。** 一个工具总是用"从会话上下文中回复用户的电子邮件"来调用采样。第 13 阶段 · 15 涵盖攻击向量。
- **通过采样的资源窃取。** 服务器要求客户端总结攻击者的 payload，让用户为此付费。
- **循环炸弹。** 服务器在紧密循环中调用采样。客户端必须强制每个会话的速率限制。

## 使用

`code/main.py` 提供一个虚假的服务器到客户端采样测试框架。一个模拟的 `summarize_repo` 工具调用两轮采样（选择文件，然后摘要），虚假客户端返回预设响应。测试框架展示了：

- 服务器使用 `modelPreferences` 发送 `sampling/createMessage`。
- 客户端返回补全。
- 服务器继续其循环。
- 速率限制器限制每次工具调用的总采样调用数。

关注要点：

- 服务器只暴露一个工具（`summarize_repo`）；所有推理都在采样调用中进行。
- 模型偏好权重影响客户端的模型选择；提示列出首选模型。
- 循环在 `stopReason: "endTurn"` 时终止。
- `max_samples_per_tool = 5` 限制捕获失控循环。

## 交付物

本课程产出 `outputs/skill-sampling-loop-designer.md`。给定一个需要 LLM 调用的服务器端算法（研究、摘要、规划），该技能设计一个基于采样的实现，包含正确的 modelPreferences、速率限制和安全确认。

## 练习

1. 运行 `code/main.py`。将 `max_samples_per_tool` 改为 2 并观察速率限制截断。

2. 实现 SEP-1577 的工具内采样变体：采样请求携带 `tools` 数组。验证客户端侧循环在返回最终补全前执行这些工具。注意漂移风险：SDK 签名在 2026 年上半年可能仍会变化。

3. 添加人机协同确认：在服务器的第一个 `sampling/createMessage` 之前，暂停并等待用户批准。被拒绝的调用返回有类型的拒绝。

4. 添加按客户端会话为键的每用户速率限制器。同一用户的同一服务器循环应共享预算。

5. 设计一个 `summarize_pdf` 工具，使用采样来选择要包含的块。绘制发送的消息。`modelPreferences.intelligencePriority` 在 0.1 vs 0.9 时如何改变行为？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|----------|
| 采样 | "服务器到客户端 LLM 调用" | 服务器请求客户端模型进行补全 |
| `sampling/createMessage` | "该方法" | 采样请求的 JSON-RPC 方法 |
| `modelPreferences` | "模型优先级" | 成本/速度/智能权重加上名称提示 |
| `includeContext` | "跨会话泄露" | 软弃用的上下文包含模式 |
| SEP-1577 | "采样中的工具" | 允许采样内部使用工具，支持服务器承载的 ReAct |
| 人机协同 | "用户确认" | 客户端在运行前向用户展示采样请求 |
| 循环炸弹 | "失控采样" | 服务器端无限采样循环；客户端必须速率限制 |
| 隐蔽采样 | "隐藏推理" | 恶意服务器在采样提示词中隐藏意图 |
| 资源窃取 | "使用用户的 LLM 预算" | 服务器强制客户端在不想要的采样上花费 |
| `stopReason` | "生成为何停止" | `endTurn`、`stopSequence` 或 `maxTokens` |

## 进一步阅读

- [MCP — 概念：采样](https://modelcontextprotocol.io/docs/concepts/sampling) — 采样的高层概述
- [MCP — 客户端采样规范 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/client/sampling) — 规范 `sampling/createMessage` 形态
- [MCP — GitHub SEP-1577](https://github.com/modelcontextprotocol/modelcontextprotocol) — 采样中工具的 Spec Evolution Proposal（实验性）
- [Unit 42 — MCP 攻击向量](https://unit42.paloaltonetworks.com/model-context-protocol-attack-vectors/) — 隐蔽采样和资源窃取模式
- [Speakeasy — MCP 采样核心概念](https://www.speakeasy.com/mcp/core-concepts/sampling) — 带客户端侧代码示例的演练
