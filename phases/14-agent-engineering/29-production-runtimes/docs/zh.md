# 生产运行时：队列、事件、定时任务

> 生产 Agent 在六种运行时形态上运行：请求-响应、流式、持久执行、基于队列的后台、事件驱动和定时调度。在选择框架之前先选择形态。可观测性在每种形态中都是承重的关键。

**类型：** 学习
**语言：** Python（标准库）
**前置条件：** 第 14 阶段 · 13（LangGraph）、第 14 阶段 · 22（语音）
**时间：** 约 60 分钟

## 学习目标

- 说出六种生产运行时形态并将每种匹配到框架/产品模式。
- 解释为何持久执行（LangGraph）对长时间任务至关重要。
- 描述事件驱动运行时以及 Claude Managed Agents 适用时机。
- 解释多步 Agent 中"可观测性是承重关键"这一主张。

## 问题

生产 Agent 的失败方式在 Jupyter notebook 中不会显现：第 37 步的网络超时、用户在语音通话中途挂断、机器重启导致定时任务失效、后台工作线程内存不足。运行时形态决定了哪些失败是可以存活的。

## 概念

### 请求-响应

- 同步 HTTP。用户等待完成。
- 仅适用于短任务（< 30 秒）。
- 技术栈：Agno（Python + FastAPI）、Mastra（TypeScript + Express/Hono/Fastify/Koa）。
- 可观测性：标准 HTTP 访问日志 + OTel span。

### 流式

- 使用 SSE 或 WebSocket 进行渐进式输出。
- LiveKit 将其扩展到 WebRTC 用于语音/视频（第 22 课）。
- 技术栈：任何支持流式传输的框架 + 处理 SSE/WS 的前端。
- 可观测性：每个块的时序、首词延迟、尾部延迟。

### 持久执行

- 每一步后状态检查点化；失败时自动恢复。
- AutoGen v0.4 actor 模型将故障隔离到单个 agent（第 14 课）。
- LangGraph 的核心差异化优势（第 13 课）。
- 当步数未知且恢复成本高时至关重要。

### 基于队列 / 后台

- 作业进入队列，worker 拾取，结果通过 webhook 或 pub/sub 回流。
- 对长时间运行的 agent 至关重要（每个任务数十到数百步，根据 Anthropic 的 computer use 公告）。
- 技术栈：Celery（Python）、BullMQ（Node）、SQS + Lambda（AWS）、自定义。
- 可观测性：队列深度、每个作业的延迟分布、死信队列大小。

### 事件驱动

- Agent 订阅触发器：新邮件、新 PR、定时任务触发。
- Claude Managed Agents 开箱即用地覆盖这一点（第 17 课）。
- CrewAI Flows（第 15 课）构建事件驱动的确定性工作流。
- 可观测性：触发源、事件到启动的延迟、Agent 延迟。

### 定时调度

- 定期运行的定时任务形态的 Agent。
- 结合持久执行，使失败的夜间运行在下一个周期恢复。
- 技术栈：Kubernetes CronJob + 持久框架；托管平台（Render cron、Vercel cron）。

### 2026 年部署模式

- **CrewAI Flows** 用于事件驱动生产。
- **Agno** 无状态 FastAPI 用于 Python 微服务。
- **Mastra** 服务器适配器（Express、Hono、Fastify、Koa）用于嵌入。
- **Pipecat Cloud / LiveKit Cloud** 用于托管语音（第 22 课）。
- **Claude Managed Agents** 用于托管的长时间异步运行。

### 可观测性是承重的关键

没有 OpenTelemetry GenAI span（第 23 课）加上 Langfuse/Phoenix/Opik 后端（第 24 课），你无法调试一个在第 40 步失败的多步 Agent。这对生产环境不是可选项。它是"我们快速调试"与"我们从零开始重新运行并加更多日志"之间的区别。

### 生产运行时失败的常见情况

- **形态选择错误。** 为 5 分钟的任务选择请求-响应。用户挂断；worker 堆积；重试加剧问题。
- **无死信队列。** 不带死信队列的队列 worker。失败的作业消失无踪。
- **不透明的后台工作。** 后台 Agent 运行不导出追踪。故障在用户报告前是不可见的。
- **跳过持久状态。** 任何超过 30 秒且你无法承受重启的运行都需要持久执行。

## 构建它

`code/main.py` 是一个标准库多形态演示：

- 请求-响应端点（纯函数）。
- 流式处理器（生成器）。
- 带死信队列的基于队列的 worker。
- 事件触发器注册表。
- 定时任务调度器。

运行它：

```bash
python3 code/main.py
```

输出：五条追踪显示每种形态在相同任务上的行为。相同的 Agent 逻辑，不同的外层。持久执行（第六种形态）在第 13 课中通过 LangGraph 检查点有意覆盖。

## 使用它

- **请求-响应** 用于聊天式 UX。
- **流式** 用于渐进式响应。
- **持久** 用于长时间任务。
- **队列** 用于批量/异步/长时间运行。
- **事件** 用于 Agent 的反应能力。
- **定时任务** 用于日常任务（记忆整合、评估、成本报告）。

## 交付它

`outputs/skill-runtime-shape.md` 为任务选择一种运行时形态并接入可观测性要求。

## 练习

1. 将你的第 01 课 ReAct 循环移植到你的技术栈中所有六种形态。哪种形态适合哪种产品面？
2. 为基于队列的演示添加死信队列。模拟 10% 的作业失败率；展现死信队列大小。
3. 编写一个定时触发的评估 Agent，每天夜间针对当天前 20 条追踪运行。
4. 实现带背压的流式传输：如果客户端慢，暂停 Agent。这与轮次预算如何交互？
5. 阅读 Claude Managed Agents 文档。何时你会将自托管的长时间 Agent 迁移到托管平台？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| 请求-响应 | "同步" | 用户等待；仅短任务 |
| 流式 | "SSE / WS" | 渐进式输出；更好的 UX；每个块的延迟可观测 |
| 持久执行 | "从失败恢复" | 检查点化状态；在最后一步重启 |
| 基于队列 | "后台作业" | 生产者 / worker 池 / 死信队列 |
| 事件驱动 | "基于触发器" | Agent 对外部事件做出反应 |
| 死信队列 | "Dead-letter queue" | 失败作业的停放场 |
| Claude Managed Agents | "托管平台" | Anthropic 托管的长时间异步运行，带缓存 + 压缩 |

## 扩展阅读

- [LangGraph 概述](https://docs.langchain.com/oss/python/langgraph/overview) — 持久执行细节
- [Claude Managed Agents 概述](https://platform.claude.com/docs/en/managed-agents/overview) — 托管的长时间异步运行
- [Anthropic，Introducing computer use](https://www.anthropic.com/news/3-5-models-and-computer-use) — "每个任务数十到数百步"
- [AutoGen v0.4（微软研究院）](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — actor 模型故障隔离
