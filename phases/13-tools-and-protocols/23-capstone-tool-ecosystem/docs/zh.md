# 顶点项目 — 构建完整的工具生态系统

> 第 13 阶段教授了每一个部分。本顶点项目将它们连接为一个生产形态的系统：一个带有工具 + 资源 + 提示词 + 任务 + UI 的 MCP 服务器，边缘处有 OAuth 2.1，一个 RBAC 网关，一个多服务器客户端，一个 A2A 子代理调用，OTel 追踪进入采集器，CI 中的工具投毒检测，以及一个 AGENTS.md + SKILL.md 包。最后，你将能够为每一个架构选择辩护。

**Type:** Build
**Languages:** Python（stdlib，端到端生态系统测试框架）
**Prerequisites:** Phase 13 · 01 至 21
**Time:** ~120 分钟

## 学习目标

- 组合一个 MCP 服务器，暴露工具、资源、提示词和一个带有 `ui://` 应用的任务。
- 用强制 RBAC 和固定哈希的 OAuth 2.1 网关前置该服务器。
- 编写一个带有端到端 OTel GenAI 属性追踪的多服务器客户端。
- 将部分工作负载委托给一个 A2A 子代理；验证保留了不透明度。
- 使用 AGENTS.md + SKILL.md 打包整个技术栈，使其他代理能够驱动它。

## 问题

发布"研究和报告"系统：

- 用户问："总结三篇被引用次数最多的 2026 年关于代理协议的 arXiv 论文。"
- 系统：通过 MCP 搜索 arXiv；通过 A2A 将论文总结委托给专业写作代理；聚合结果；将交互式报告渲染为 MCP Apps `ui://` 资源；将每个步骤记录到 OTel。

第 13 阶段的所有原语都出现了。这不是玩具——2026 年由 Anthropic（Claude Research 产品）、OpenAI（带 Apps SDK 的 GPTs）和第三方发布的生产研究助手系统具有与此完全相同的形态。

## 概念

### 架构

```
[user] -> [client] -> [gateway（OAuth 2.1 + RBAC）] -> [research MCP server]
                                                      |
                                                      +- MCP 工具：arxiv_search（纯读）
                                                      +- MCP 资源：notes://recent
                                                      +- MCP 提示词：/research_topic
                                                      +- MCP 任务：generate_report（长时运行）
                                                      +- MCP Apps UI：ui://report/current
                                                      +- A2A 调用：writer-agent（tasks/send）
                                                      |
                                                      +- OTel GenAI span
```

### 追踪层次结构

```
agent.invoke_agent
 ├── llm.chat（启动）
 ├── mcp.call -> tools/call arxiv_search
 ├── mcp.call -> resources/read notes://recent
 ├── mcp.call -> prompts/get research_topic
 ├── a2a.tasks/send -> writer-agent
 │    └── task transitions（不透明内部）
 ├── mcp.call -> tools/call generate_report（任务增强）
 │    └── tasks/status 轮询
 │    └── tasks/result（已完成，返回 ui:// resource）
 └── llm.chat（最终合成）
```

一个 trace id。每个 span 有正确的 `gen_ai.*` 属性。

### 安全态势

- OAuth 2.1 + PKCE，带资源指示器将受众固定到网关。
- 网关持有上游凭据；用户永远看不到它们。
- RBAC：`alice` 有 `research:read`、`research:write`，可以调用所有工具。`bob` 有 `research:read`，不能调用 `generate_report`。
- 固定的描述清单：丢弃任何工具哈希发生变化的服务器。
- 两条规则审计：没有工具同时结合不受信任的输入、敏感数据和后果性操作。

### 渲染

最终的 `generate_report` 任务返回内容块加上 `ui://report/current` 资源。客户端的宿主（Claude Desktop 等）在沙盒 iframe 中渲染交互式仪表盘。仪表盘包含按引用次数排序的论文列表、引用数量，以及一个按钮，当用户点击任何论文时调用 `host.callTool('summarize_paper', {arxiv_id})`。

### 打包

整个系统以如下形式发布：

```
research-system/
  AGENTS.md                     # 项目约定
  skills/
    run-research/
      SKILL.md                  # 顶级工作流
  servers/
    research-mcp/               # MCP 服务器
      pyproject.toml
      src/
  agents/
    writer/                     # A2A 代理
  gateway/
    config.yaml                 # RBAC + 固定清单
```

用户通过 `docker compose up` 部署。Claude Code、Cursor、Codex 和 opencode 用户可以通过调用 `run-research` 技能来驱动系统。

### 每个第 13 阶段课程的贡献

| 课程 | 顶点项目使用的部分 |
|------|--------------------|
| 01-05 | 工具接口、提供者可移植性、并行调用、模式、检查 |
| 06-10 | MCP 原语、服务器、客户端、传输、资源 + 提示词 |
| 11-14 | 采样、根和作用域 + 引导、异步任务、`ui://` 应用 |
| 15-17 | 工具投毒、OAuth 2.1、网关 + 注册表 |
| 18 | A2A 子代理委托 |
| 19 | OTel GenAI 追踪 |
| 20 | LLM 层的路由网关 |
| 21 | SKILL.md + AGENTS.md 打包 |

## 使用

`code/main.py` 将前几课的模式缝合到一个可运行的演示中。全部使用标准库，全部在进程中，这样你可以从头到尾阅读。它运行研究和报告场景的完整流程：与网关握手，模拟 OAuth 2.1，合并 tools/list，generate_report 作为任务，A2A 调用 writer，返回 ui:// 资源，发出 OTel span。

关注要点：

- 一个 trace id 贯穿每一跳。
- 网关策略阻止第二个用户写入。
- 任务生命周期从 working 到 completed，返回文本和 ui:// 内容。
- A2A 调用的内部状态对编排器是不透明的。
- AGENTS.md 和 SKILL.md 是另一个代理复现此工作流所需的唯一文件。

## 交付物

本课程产出 `outputs/skill-ecosystem-blueprint.md`。给定一个产品需求（研究、总结、自动化），该技能生成完整架构：哪些 MCP 原语、哪些网关控制、哪些 A2A 调用、哪些遥测、如何打包。

## 练习

1. 运行 `code/main.py`。注意单个 trace id 以及 span 如何嵌套。计算演示触及了多少第 13 阶段的原语。

2. 扩展演示：添加第二个后端 MCP 服务器（例如 `bibliography`），并确认网关将其工具合并到相同的命名空间。

3. 将虚假的 A2A 写作代理替换为在子进程上运行的真正代理。使用第 19 课的测试框架。

4. 在编排器和 LLM 之间的路由网关中添加 PII 编辑步骤。确认用户查询中的电子邮件被清洗。

5. 为将要维护此系统的队友编写一份 AGENTS.md。它应能在五分钟内读完，并为他们提供在 Cursor 或 Codex 中驱动此顶点项目所需的一切。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|----------|
| 顶点项目 | "第 13 阶段集成演示" | 使用每个原语的端到端系统 |
| 研究和报告 | "场景" | 搜索、总结、渲染模式 |
| 生态系统 | "所有部件组合" | 服务器 + 客户端 + 网关 + 子代理 + 遥测 + 包 |
| 追踪层次结构 | "单个 trace id" | 每一跳的 span 共享 trace；父子关系通过 span id |
| 网关签发令牌 | "传递性认证" | 客户端只看到网关的令牌；网关持有上游凭据 |
| 合并命名空间 | "扁平列表中的所有工具" | 网关上的多服务器合并，冲突时加前缀 |
| 不透明度边界 | "A2A 调用隐藏内部" | 子代理的推理对编排器不可见 |
| 三层技术栈 | "AGENTS.md + SKILL.md + MCP" | 项目上下文 + 工作流 + 工具 |
| 纵深防御 | "多层安全" | 固定哈希、OAuth、RBAC、两条规则、审计日志 |
| 规范合规矩阵 | "我们发布的符合规范要求的清单" | 将可交付物映射到 2025-11-25 要求的检查表 |

## 进一步阅读

- [MCP — 规范 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — 综合参考
- [MCP 博客 — 2026 路线图](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — 协议的发展方向
- [a2a-protocol.org](https://a2a-protocol.org/latest/) — A2A v1.0 参考
- [OpenTelemetry — GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 规范追踪约定
- [Anthropic — Claude Agent SDK 概述](https://code.claude.com/docs/en/agent-sdk/overview) — 生产代理运行时模式
