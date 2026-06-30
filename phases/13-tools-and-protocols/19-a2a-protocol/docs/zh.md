# A2A — 代理到代理协议

> MCP 是代理到工具。A2A（Agent2Agent）是代理到代理——一个开放协议，让构建在不同框架上的不透明代理能够协作。由 Google 于 2025 年 4 月发布，2025 年 6 月捐赠给 Linux 基金会，2026 年 4 月达到 v1.0 版本，拥有 150 多个支持者，包括 AWS、Cisco、Microsoft、Salesforce、SAP 和 ServiceNow。它吸收了 IBM 的 ACP 并添加了 AP2 支付扩展。本课程演练代理卡片、任务生命周期和两种传输绑定。

**Type:** Build
**Languages:** Python（stdlib，代理卡片 + 任务测试框架）
**Prerequisites:** Phase 13 · 06（MCP 基础），Phase 13 · 08（MCP 客户端）
**Time:** ~75 分钟

## 学习目标

- 区分代理到工具（MCP）和代理到代理（A2A）用例。
- 在 `/.well-known/agent.json` 上发布代理卡片，包含技能和端点元数据。
- 遍历任务生命周期（submitted → working → input-required → completed / failed / canceled / rejected）。
- 使用带有 Parts（text、file、data）的消息和作为输出的 Artifacts。

## 问题

一个客服代理需要将报告编写委托给一个专业写作代理。A2A 之前的选项：

- 自定义 REST API。能用但每个配对都是独特的。
- 共享代码库。要求两个代理运行同一框架。
- MCP。不合适：MCP 用于调用工具，而不是两个代理在保留各自不透明内部推理的情况下协作。

A2A 填补了空白。它将交互建模为一个代理向另一个代理发送一个任务，附带生命周期、消息和产物。被调用代理的内部状态保持不透明——调用者只看到任务状态转换和最终输出。

A2A 是"让跨框架的代理相互通信"的协议。它不替代 MCP；两者是互补的。

## 概念

### 代理卡片

每个符合 A2A 的代理在 `/.well-known/agent.json` 上发布一张卡片：

```json
{
  "schemaVersion": "1.0",
  "name": "research-agent",
  "description": "总结学术论文并草拟引文。",
  "url": "https://research.example.com/a2a",
  "version": "1.2.0",
  "skills": [
    {
      "id": "summarize_paper",
      "name": "总结一篇论文",
      "description": "阅读论文 PDF 并生成 3 段总结。",
      "inputModes": ["text", "file"],
      "outputModes": ["text", "artifact"]
    }
  ],
  "capabilities": {"streaming": true, "pushNotifications": true}
}
```

发现是基于 URL 的：获取卡片，学习 A2A 端点的 URL，枚举技能。

### 签名代理卡片（AP2）

AP2 扩展（2025 年 9 月）向代理卡片添加了加密签名。发布者用自己的 JWT 签名其卡片；消费者验证。防止冒充。

### 任务生命周期

```
submitted -> working -> completed | failed | canceled | rejected
             -> input_required -> working （通过消息循环）
```

客户端用 `tasks/send` 发起。被调用代理穿越状态；客户端通过 SSE 或轮询订阅状态更新。

### 消息和 Parts

一条消息携带一个或多个 Part：

- `text` — 纯文本内容。
- `file` — 带有 mimeType 的 base64 二进制数据。
- `data` — 有类型的 JSON payload（被调用代理的结构化输入）。

示例：

```json
{
  "role": "user",
  "parts": [
    {"type": "text", "text": "总结这篇论文。"},
    {"type": "file", "file": {"name": "paper.pdf", "mimeType": "application/pdf", "bytes": "..."}},
    {"type": "data", "data": {"targetLength": "3 段"}}
  ]
}
```

### Artifacts

输出是 Artifacts，而不是裸字符串。一个 Artifact 是一个命名的、有类型的输出：

```json
{
  "name": "summary",
  "parts": [{"type": "text", "text": "..."}],
  "mimeType": "text/markdown"
}
```

Artifacts 可以作为块流式传输。调用者累积。

### 两种传输绑定

1. **JSON-RPC over HTTP。** `/a2a` 端点，POST 用于请求，可选 SSE 用于流式传输。默认绑定。
2. **gRPC。** 适用于 gRPC 是原生的企业环境。

两种绑定携带相同的逻辑消息形态。

### 不透明度保留

一个关键设计原则：被调用代理的内部状态是不透明的。调用者看到任务状态和 Artifacts。被调用代理的思维链、其工具调用、其子代理委托——全部不可见。这不同于 MCP，在 MCP 中工具调用是透明的。

理由：A2A 使竞争者能够在不对内部进行暴露的情况下协作。A2A 可以"调用这个客服代理"而无需要求调用者了解该代理如何实现该服务。

### 时间线

- **2025-04-09。** Google 宣布 A2A。
- **2025-06-23。** 捐赠给 Linux 基金会。
- **2025-08。** 吸收 IBM 的 ACP。
- **2025-09。** AP2 扩展（Agent Payments）发布。
- **2026-04。** v1.0 发布，拥有 150+ 支持组织。

### 与 MCP 的关系

| 维度 | MCP | A2A |
|------|-----|-----|
| 用例 | 代理到工具 | 代理到代理 |
| 不透明度 | 透明的工具调用 | 不透明的内部推理 |
| 典型调用者 | 代理运行时 | 另一个代理 |
| 状态 | 工具调用结果 | 带生命周期的任务 |
| 授权 | OAuth 2.1（第 13 阶段 · 16） | JWT 签名的代理卡片（AP2） |
| 传输 | Stdio / Streamable HTTP | JSON-RPC over HTTP / gRPC |

当你想要调用特定工具时使用 MCP。当你想要将整个任务委托给另一个代理时使用 A2A。许多生产系统两者都用：一个代理将 MCP 用于其工具层，将 A2A 用于其协作层。

## 使用

`code/main.py` 实现了一个最小 A2A 测试框架：一个研究代理发布其卡片，一个写作代理接收 `tasks/send` 带有 Parts，包括一个 PDF 和一个文本指令，穿越 working → input_required → working → completed，并返回一个文本 artifact。全部使用标准库；使用内存传输来聚焦消息形态。

关注要点：

- 代理卡片 JSON 形态。
- 任务 id 分配和状态转换。
- 带有混合类型 Parts 的消息。
- 任务中期 input-required 分支。
- 完成时返回 Artifact。

## 交付物

本课程产出 `outputs/skill-a2a-agent-spec.md`。给定一个应可被其他代理调用的新代理，该技能生成代理卡片 JSON、技能模式和端点蓝图。

## 练习

1. 运行 `code/main.py`。跟踪完整的任务生命周期，包括被调用代理要求澄清的 input-required 暂停。

2. 添加一个签名的代理卡片。使用 HMAC 对卡片的规范 JSON 签名。编写验证器并确认它对被篡改的卡片失败。

3. 实现任务流式传输：写作代理通过 SSE 发出三个增量 artifact 块，调用者累积它们。

4. 设计一个包装 MCP 服务器的 A2A 代理。将每个 MCP 工具映射到一个 A2A 技能。注意权衡——失去了什么不透明度？

5. 阅读 A2A v1.0 公告，找出截至 2026 年 4 月任何框架尚未实现的一项功能。（提示：与多跳任务委托有关。）

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|----------|
| A2A | "代理到代理协议" | 用于不透明代理协作的开放协议 |
| 代理卡片 | "`.well-known/agent.json`" | 描述代理技能和端点的已发布元数据 |
| 技能 | "一个可调用单元" | 代理支持的命名操作（类似于 MCP 工具） |
| 任务 | "委托单元" | 带生命周期和最终 artifact 的工作项 |
| 消息 | "任务输入" | 携带 Parts（text、file、data） |
| Part | "有类型块" | 消息的 `text` / `file` / `data` 元素 |
| Artifact | "任务输出" | 完成时返回的命名、有类型输出 |
| AP2 | "代理支付协议" | 用于信任和支付的签名代理卡片扩展 |
| 不透明度 | "黑盒协作" | 被调用代理的内部对调用者隐藏 |
| input-required | "任务暂停" | 代理需要更多信息时的生命周期状态 |

## 进一步阅读

- [a2a-protocol.org](https://a2a-protocol.org/latest/) — 规范 A2A 规范
- [a2aproject/A2A — GitHub](https://github.com/a2aproject/A2A) — 参考实现和 SDK
- [Linux Foundation — A2A 启动新闻稿](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents) — 2025 年 6 月治理转移
- [Google Cloud — A2A 协议升级](https://cloud.google.com/blog/products/ai-machine-learning/agent2agent-protocol-is-getting-an-upgrade) — 路线图和合作伙伴势头
- [Google Dev — A2A 1.0 里程碑](https://discuss.google.dev/t/the-a2a-1-0-milestone-ensuring-and-testing-backward-compatibility/352258) — v1.0 发布说明和向后兼容指导
