# Agno 与 Mastra：生产运行时

> Agno（Python）和 Mastra（TypeScript）是 2026 年的生产运行时组合。Agno 瞄准微秒级 agent 实例化和无状态 FastAPI 后端。Mastra 提供 agent、工具、工作流、统一模型路由和复合存储，基于 Vercel AI SDK 底层。

**类型：** 学习
**语言：** Python、TypeScript
**前置条件：** 第 14 阶段 · 01（Agent 循环），第 14 阶段 · 13（LangGraph）
**时间：** ~45 分钟

## 学习目标

- 识别 Agno 的性能目标以及它们何时重要。
- 列举 Mastra 的三个原语——Agent、Tool、Workflow——以及支持的服务器适配器。
- 解释为什么无状态会话范围的 FastAPI 后端是推荐的 Agno 生产路径。
- 为给定技术栈选择 Agno 还是 Mastra（Python 优先 vs TypeScript 优先）。

## 问题

LangGraph、AutoGen、CrewAI 是框架重量级的。想要"只需要 agent 循环，要快，在我的运行时里"的团队会选择 Agno（Python）或 Mastra（TypeScript）。两者都用一些框架拥有的原语换取了原始速度和与周围技术栈更紧密的契合。

## 概念

### Agno

- Python 运行时，前身是 Phi-data。
- "没有图、链或复杂的模式——只有纯 Python。"
- 文档中的性能目标：约 2μs agent 实例化，约 3.75 KiB 内存每个 agent，约 23 个模型提供者。
- 生产路径：无状态会话范围的 FastAPI 后端。每个请求启动一个全新的 agent；会话状态存储在数据库中。
- 原生多模态（文本、图像、音频、视频、文件）和 agentic RAG。

当你有每秒数千个短生命周期的 agent（聊天扇入、评估管道）时，速度目标很重要。当一个 agent 运行 10 分钟时，它们就不那么重要了。

### Mastra

- TypeScript，基于 Vercel AI SDK 构建。
- 三个原语：**Agent**、**Tool**（Zod 类型化）、**Workflow**。
- 统一模型路由——3,300+ 模型覆盖 94 个提供者（2026 年 3 月）。
- 复合存储：记忆、工作流、可观测性到不同后端；推荐 ClickHouse 用于规模化的可观测性。
- Apache 2.0，`ee/` 目录为源码可用的企业许可证。
- 服务器适配器：Express、Hono、Fastify、Koa；一流的 Next.js 和 Astro 集成。
- 提供 Mastra Studio（localhost:4111）用于调试。
- 1.0 版本（2026 年 1 月）获得 22k+ GitHub stars，300k+ 周 npm 下载量。

### 定位

两者都没有试图成为 LangGraph。它们竞争的是：

- **语言契合。** Agno 适合 Python 优先的团队；Mastra 适合 TypeScript 优先的团队。
- **运行时人体工学。** Agno = 近乎零开销；Mastra = 与 Vercel 生态系统集成。
- **可观测性。** 两者都集成了 Langfuse/Phoenix/Opik（第 24 课），但 Mastra Studio 是第一方的。

### 何时选择哪个

- **Agno** — Python 后端，许多短生命周期 agent，强性能需求，FastAPI 团队。
- **Mastra** — TypeScript 后端，Next.js / Vercel 部署，统一多提供者模型路由，Zod 类型化工具。
- **LangGraph**（第 13 课）— 当持久状态和显式图推理比原始速度更重要时。
- **OpenAI / Claude Agent SDK** — 当你想要提供者产品化的形态时（第 16–17 课）。

### 这种模式可能出错的地方

- **为性能而性能。** 因为"2μs"听起来很好而选择 Agno，但工作负载是每次请求一次慢速 agent 调用。开销不是瓶颈。
- **生态系统锁定。** Mastra 的 Vercel 风格集成在 Vercel 上是加分项，在其他地方是减分项。
- **企业许可证混淆。** Mastra 的 `ee/` 目录是源码可用的，不是 Apache 2.0。如果你计划 fork，先阅读许可证。

## 构建它

本课主要是比较性的——没有一个代码工件能公平对待两个框架。参见 `code/main.py` 中的并排玩具：一个最小化的"运行 agent、流式输出、持久化会话"流程，实现两次（一次 Agno 形态，一次 Mastra 形态）。

运行它：

```
python3 code/main.py
```

两个结构不同但功能等效的跟踪。

## 使用它

- **Agno** — 需要速度和 FastAPI 形态的 Python 后端。
- **Mastra** — 具有许多提供者和工作流原语的 TypeScript 后端。
- 两者都提供第一方可观测性钩子。两者都集成了 Langfuse。

## 交付它

`outputs/skill-runtime-picker.md` 根据技术栈、延迟预算和运维形态选择 Agno、Mastra、LangGraph 或提供者 SDK。

## 练习

1. 阅读 Agno 的文档。将标准库 ReAct 循环（第 01 课）移植到 Agno。什么消失了？什么留下了？
2. 阅读 Mastra 的文档。将相同的循环移植到 Mastra。工具类型化（Zod vs 无）有什么变化？
3. 基准测试：在你的技术栈上测量 agent 实例化延迟。Agno 的 2μs 对你的工作负载重要吗？
4. 设计一个迁移方案：如果你一直在 Python 中使用 CrewAI，迁移到 Agno 会破坏什么？
5. 阅读 Mastra 的 `ee/` 许可证条款。什么限制会影响开源 fork？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| Agno | "快速的 Python agent" | 无状态会话范围的 agent 运行时 |
| Mastra | "基于 Vercel AI SDK 的 TypeScript agent" | Agent + Tool + Workflow + 模型路由 |
| 统一模型路由（Unified Model Router） | "多提供者访问" | 覆盖 94 个提供者 3,300+ 模型的单一客户端 |
| 复合存储（Composite storage） | "多后端" | 记忆/工作流/可观测性各自到不同的存储 |
| Mastra Studio | "本地调试器" | localhost:4111 UI，用于内省 agent |
| 源码可用（Source-available） | "非开源" | 许可证允许源码阅读但限制商业使用 |

## 进一步阅读

- [Agno Agent Framework 文档](https://www.agno.com/agent-framework) — 性能目标，FastAPI 集成
- [Mastra 文档](https://mastra.ai/docs) — 原语，服务器适配器，模型路由
- [LangGraph 概览](https://docs.langchain.com/oss/python/langgraph/overview) — 有状态图替代方案
- [Comet Opik](https://www.comet.com/site/products/opik/) — Mastra 集成引用的可观测性对比
