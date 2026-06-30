# A2A——智能体对智能体协议

> Google于2025年4月宣布A2A；到2026年4月，规范位于https://a2a-protocol.org/latest/specification/，150+个组织支持它。A2A是MCP（第13课）的水平补充：MCP是垂直的（智能体 ↔ 工具），A2A是对等（智能体 ↔ 智能体）。它定义了智能体卡片（发现）、带产物的任务（文本、结构化数据、视频）、不透明任务生命周期和认证。生产系统越来越多地将MCP与A2A配对。Google Cloud在2025-2026年间将A2A支持推出到Vertex AI Agent Builder。

**Type:** Learn + Build
**Languages:** Python (stdlib, `http.server`, `json`)
**Prerequisites:** Phase 16 · 04 (Primitive Model)
**Time:** ~75 minutes

## 问题

你的智能体需要调用另一个系统上的另一个智能体。怎么做？你可以暴露一个HTTP端点，定义一个定制JSON模式，并希望另一端说它。每对智能体都成为一个定制集成。

A2A是该调用的通用有线路协议。标准发现、标准任务模型、标准传输、标准产物。像HTTP+REST但智能体是一等公民。

## 概念

### 四个元素

**智能体卡片。** 位于 `/.well-known/agent.json` 的JSON文档，描述智能体：名称、技能、端点、支持的模态、认证要求。发现通过读取卡片发生。

```
GET https://agent.example.com/.well-known/agent.json
→ {
    "name": "code-review-agent",
    "skills": ["review-python", "review-typescript"],
    "endpoints": {
      "tasks": "https://agent.example.com/tasks"
    },
    "auth": {"type": "bearer"},
    "modalities": ["text", "structured"]
  }
```

**任务。** 工作单元。一个异步、有状态的对象，具有生命周期：`submitted → working → completed / failed / canceled`。客户端发送任务，轮询或订阅更新。

**产物。** 由任务产生的结果类型。文本、结构化JSON、图像、视频、音频。产物是有类型的，使不同模态成为一等公民。

**不透明生命周期。** A2A不规定*远程智能体如何*解决任务。客户端看到状态转换和产物；实现可以自由使用任何框架。

### MCP/A2A分离

- **MCP**（第13课）：智能体 ↔ 工具。智能体通过JSON-RPC向工具服务器读/写。默认无状态。
- **A2A**：智能体 ↔ 智能体。对等协议；双方都是有自己推理的智能体。

生产多智能体系统使用两者。A2A对对方在其一侧调用MCP工具。分离保持这两个关注点清洁。

### 发现流程

```
客户端                     智能体服务器
  ├──GET /.well-known/agent.json──>
  <──智能体卡片 JSON─────────────
  ├──POST /tasks {skill, input}──>
  <──201 task_id, state=submitted
  ├──GET /tasks/{id}──────────────>
  <──state=working, 42% done──────
  ├──GET /tasks/{id}──────────────>
  <──state=completed, artifacts──
```

或流式：SSE订阅 `/tasks/{id}/events` 用于推送更新。

### 认证

A2A支持三种常见模式：

- **Bearer令牌**——OAuth2或不透明。
- **mTLS**——相互TLS；组织互相证明身份。
- **签名请求**——负载上的HMAC。

认证在智能体卡片中声明；客户端发现并遵守。

### 到2026年4月150+个组织

企业采用推动了A2A的规模。头条：A2A成为企业智能体系统跨越信任边界的方式。Google Cloud发布了Vertex AI Agent Builder A2A支持；Microsoft Agent Framework支持它；大多数主要框架（LangGraph、CrewAI、AutoGen）发布A2A适配器。

### 哪里A2A获胜

- **跨组织调用。** 公司A的智能体调用公司B的智能体。没有A2A，每对都是定制契约。
- **异构框架。** LangGraph智能体调用CrewAI智能体调用自定义Python智能体。A2A归一化。
- **类型化产物。** 视频结果、结构化JSON、音频——全都一等公民。
- **长时间运行的任务。** 不透明生命周期 + 轮询使数小时长的任务直接了当。

### 哪里A2A挣扎

- **延迟敏感的微调用。** A2A的生命周期是异步的。亚毫秒智能体间不适合；使用直接RPC。
- **紧密耦合的进程内智能体。** 如果两个智能体在同一Python进程中运行，A2A的HTTP往返是过度的。
- **小团队。** 规范开销是真实的；仅内部智能体可能不需要此形式。

### A2A vs ACP、ANP、NLIP

2024-2026年出现了几个相关规范：

- **ACP**（IBM/Linux基金会）——A2A的前身，范围更窄。
- **ANP**（智能体网络协议）——对等发现为重、去中心化优先。
- **NLIP**（Ecma自然语言交互协议，2025年12月标准化）——自然语言内容类型。

A2A是截至2026年4月采用最广泛的对等协议。参见arXiv:2505.02279（Liu等人，"A Survey of Agent Interoperability Protocols"）进行比较。

## 构建它

`code/main.py` 使用 `http.server` 和JSON实现一个A2A极简服务器和客户端。服务器：

- 暴露 `/.well-known/agent.json`，
- 接受 `POST /tasks`，
- 管理任务状态，
- 在 `GET /tasks/{id}` 返回产物。

客户端：

- 获取智能体卡片，
- 提交任务，
- 轮询直到完成，
- 读取产物。

运行：

```
python3 code/main.py
```

脚本在后台线程中启动服务器，然后运行客户端。你看到完整流程：发现、提交、轮询、产物。

## 运用

`outputs/skill-a2a-integrator.md` 设计A2A集成：智能体卡片内容、任务模式、认证选择、流式 vs 轮询。

## 交付物

检查清单：

- **固定规范版本。** A2A仍在演进；智能体卡片应声明协议版本。
- **幂等任务创建。** 重复提交（网络重试）应产生一个任务。
- **产物模式。** 声明智能体返回什么形状；消费者应验证。
- **速率限制 + 认证。** A2A是面向公众的；应用标准Web安全。
- **失败任务的死信。** 随时间检查模式以发现反复失败类型。

## 练习

1. 运行 `code/main.py`。确认客户端发现服务器并收到正确产物。
2. 添加第二个技能到服务器（例如"summarize"）。更新智能体卡片。编写一个根据任务类型选择技能的客户端。
3. 实现SSE流端点：`/tasks/{id}/events` 发出状态变化。客户端需要做什么不同的事？
4. 阅读A2A规范（https://a2a-protocol.org/latest/specification/）。识别规范要求但此演示未实现的三件事。
5. 比较A2A（智能体卡片发现）与MCP（通过 `listTools` 的服务器端能力列表）。自描述智能体与能力探测之间的权衡是什么？

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|----------------|------------------------|
| A2A | "智能体到智能体" | 智能体跨系统调用其他智能体的对等协议。Google 2025年。 |
| 智能体卡片 | "智能体的名片" | 位于 `/.well-known/agent.json` 的JSON，描述技能、端点、认证。 |
| 任务 | "工作单元" | 带生命周期的异步有状态对象；完成时产物产生。 |
| 产物 | "结果" | 类型化输出：文本、结构化JSON、图像、视频、音频。一等媒体。 |
| 不透明生命周期 | "如何解决是智能体的事" | 客户端看到状态转换；服务器自由选择框架/工具。 |
| 发现 | "找到智能体" | `GET /.well-known/agent.json` 返回卡片。 |
| MCP vs A2A | "工具 vs 对等" | MCP：垂直智能体 ↔ 工具。A2A：水平智能体 ↔ 智能体。 |
| ACP / ANP / NLIP | "兄弟协议" | 相邻规范；A2A是2026年采用最广泛的。 |

## 进一步阅读

- [A2A specification](https://a2a-protocol.org/latest/specification/) — 权威规范
- [Google Developers Blog — A2A announcement](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/) — 2025年4月发布文章
- [A2A GitHub repo](https://github.com/a2aproject/A2A) — 参考实现和SDK
- [Liu et al. — A Survey of Agent Interoperability Protocols](https://arxiv.org/html/2505.02279v1) — MCP、ACP、A2A、ANP比较
