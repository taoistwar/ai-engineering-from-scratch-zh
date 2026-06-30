# LLM 可观测性栈选择

> 2026 年的可观测性市场分为两个类别。开发平台（LangSmith、Langfuse、Comet Opik）将监控与评估、提示管理、会话回放捆绑。网关/仪器化工具（Helicone、SigNoz、OpenLLMetry、Phoenix）专注于遥测。Langfuse 以 MIT 许可的核心开源，有很强的开源平衡（免费云端 50K 事件/月）。Phoenix 在 Elastic License 2.0 下原生支持 OpenTelemetry — 对漂移/RAG 可视化出色，不是持久化生产后端。Arize AX 使用零拷贝 Iceberg/Parquet 集成，声称比单体可观测性便宜 100 倍。LangSmith 在 LangChain/LangGraph 上领先，$39/用户/月，仅企业版支持自托管。Helicone 是基于代理的，15-30 分钟设置，100K req/月免费，但在代理追踪上深度较少。常见生产模式：网关（Helicone/Portkey）+ 评估平台（Phoenix/TruLens），由 OpenTelemetry 粘合。

**Type:** Learn
**Languages:** Python (stdlib, toy trace-sampling simulator)
**Prerequisites:** Phase 17 · 08 (Inference Metrics), Phase 14 (Agent Engineering)
**Time:** ~60 minutes

## 学习目标

- 区分开发平台（捆绑：评估 + 提示 + 会话）和网关/遥测工具（仅追踪 + 指标）。
- 映射六种主要工具（Langfuse、LangSmith、Phoenix、Arize AX、Helicone、Opik）到其许可、定价和最佳甜点用例。
- 解释 OpenTelemetry 粘合模式，让你将网关工具与独立的评估平台组合。
- 说出 2026 年的成本差异因素（Arize AX 的零拷贝方法 vs 单体摄取），并陈述粗略的 100 倍乘数。

## 问题

你交付了一个 LLM 功能。它能工作。你对提示失败、工具循环、延迟回归、成本尖峰或提示缓存命中率一无所知。你搜索"LLM 可观测性"，得到八个工具，它们都声称以三种不同的价位解决同样的问题。

它们解决的问题不同。LangSmith 回答"这个 LangGraph 运行为什么失败？"Phoenix 回答"我的 RAG 管线是否正在漂移？"Helicone 回答"哪个应用在燃烧 token？"Langfuse 回答"我能自托管整个东西吗？"不同的工具，不同的受众。

选择涉及四个轴：栈（LangChain？原生 SDK？多供应商？）、许可容忍度（只 MIT？Elastic OK？商业 fine？）、预算（免费层？$100/月？$1000/月？）和自托管（必须？加分项？从不？）。

## 概念

### 两个类别

**开发平台**将可观测性与评估、提示管理、数据集版本管理、会话回放捆绑。你运行实验，看哪个提示有效，将新提示与旧优胜者进行数据集回归。LangSmith、Langfuse、Comet Opik。

**网关/遥测工具**仪器化推理调用 — 提示、响应、token、延迟、模型、成本。Helicone、SigNoz、OpenLLMetry、Phoenix。极简主义。可通过 OpenTelemetry 与独立的评估工具组合。

### Langfuse — 开源平衡

- 核心 Apache / MIT 许可；通过 Docker 自托管。
- 云端免费层：50K 事件/月。付费：$29/月团队。
- 评估、提示管理、追踪、数据集。合理覆盖所有四个开发平台特性。
- 甜点：你想要 LangSmith 级别的特性但必须自托管或保持开源许可。

### Phoenix (Arize) — 遥测优先，OpenTelemetry 原生

- Elastic License 2.0；自托管轻松。
- 对 RAG 和漂移可视化出色。嵌入空间散点图作为一等特性提供。
- 不是设计为持久化生产后端 — 主要是开发时可观测性。
- 甜点：RAG 管线开发、漂移调试，与独立的生产网关配对。

### Arize AX — 规模化方案

- 商业。通过 Iceberg/Parquet 零拷贝数据湖集成。
- 声称规模化下比单体可观测性（Datadog 级别）便宜约 100 倍。数学：你将自己的追踪存储在 S3 上的 Parquet 中；Arize 直接读取。
- 甜点：>1000 万追踪/天，已有数据湖，想要 LLM 特定仪表板而无需 Datadog 价格。

### LangSmith — LangChain/LangGraph 优先

- 商业，$39/用户/月。仅企业版自托管。
- 对 LangChain 和 LangGraph 栈最佳。如果你不在任一栈上，吸引力较小。
- 甜点：团队已投入 LangChain，愿意付费。

### Helicone — 基于代理的最小可行方案

- 将 `OPENAI_API_BASE` 切换到 Helicone 代理，15-30 分钟设置。
- MIT 许可；100K req/月免费，付费 $20/月+。
- 包含故障转移、缓存、速率限制 — 也充当网关。
- 对代理/多步追踪深度较少。
- 甜点：快速启动、单栈应用、需要网关 + 可观测性合二为一。

### Opik (Comet) — 开源开发平台

- Apache 2.0，完全开源。
- 与 Langfuse 类似的功能集，兼具 Comet 传统。
- 甜点：已在 Comet 上的 ML 团队，想要在同一窗格中的 LLM 可观测性。

### SigNoz — OpenTelemetry 优先的全 APM

- Apache 2.0。通过 OpenTelemetry 处理通用 APM 加 LLM。
- 甜点：跨服务和 LLM 调用的统一可观测性。

### 粘合剂：OpenTelemetry + GenAI 语义惯例

OpenTelemetry 在 2025 年末发布了 GenAI 语义惯例（`gen_ai.system`、`gen_ai.request.model`、`gen_ai.usage.input_tokens`）。消费 OTel 的工具可以互操作。正在涌现的生产模式：

1. 从每个 LLM 调用发出带 GenAI 惯例的 OTel。
2. 路由到网关（Helicone / Portkey）用于日常工作。
3. 双重发送到评估平台（Phoenix / Langfuse）用于回归。
4. 归档到数据湖（Iceberg）通过 Arize AX 或 DuckDB 做长期分析。

### 陷阱：在错误的层面仪器化

在你的代理框架内部仪器化（例如添加 LangSmith 追踪）将你耦合到该框架。在 HTTP/OpenAI-SDK 层面仪器化（通过 OpenLLMetry 或你的网关）是可移植的。

### 采样 — 你无法保留所有

在 >100 万请求/天下，全追踪保留的成本超过 LLM 调用的成本。按规则采样：100% 错误、100% 高成本、5% 成功。始终保留聚合；为长尾保留原始数据。

### 你应该记住的数字

- Langfuse 免费云端：50K 事件/月。
- LangSmith：$39/用户/月。
- Helicone 免费：100K req/月。
- Arize AX 声称：规模化下比单体便宜约 100 倍。
- OpenTelemetry GenAI 惯例：2025 年推出，2026 年广泛采用。

## 使用它

`code/main.py` 跨保留策略模拟 100 万追踪的一天（100% 摄取、采样、采样 + 错误）。报告存储成本和每种策略下丢失的内容。

## 交付它

本课产出 `outputs/skill-observability-stack.md`。给定栈、规模、预算、许可立场，选择工具。

## 练习

1. 你在 LangChain 上的团队想要开源自托管可观测性。选择 Langfuse 或 Opik 并论证。
2. 在 500 万追踪/天下，Datadog 报价 $150K/月，计算 Arize AX 的盈亏平衡。
3. 设计你的组织指南应强制在每个 LLM 调用上的 OpenTelemetry GenAI 属性集。
4. 论证 Phoenix 单独是否足以用于生产。何时不够？
5. Helicone 有 20 毫秒代理开销。在 P99 TTFT 300 毫秒下，可接受吗？如果 SLA 是 100 毫秒呢？

## 核心术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| OpenLLMetry | "LLM 的 OTel" | LLM 的开源 OpenTelemetry 仪器化 |
| GenAI 惯例 | "OTel 属性" | LLM 调用的标准 OTel 属性名称 |
| LangSmith | "LangChain 可观测性" | 与 LangChain 生态系统捆绑的商业平台 |
| Langfuse | "开源 LangSmith" | MIT 开源，类似功能集 |
| Phoenix | "Arize 开发工具" | OpenTelemetry 原生的开发/评估平台 |
| Arize AX | "规模化可观测性" | 商业零拷贝 Iceberg/Parquet 可观测性 |
| Helicone | "代理可观测性" | 收集 LLM 遥测的 HTTP 代理 + 网关特性 |
| Opik | "Comet LLM" | Comet 的 Apache 2.0 开源开发平台 |
| 会话回放 | "追踪重新运行" | 带工具调用的全代理会话回放 |
| 评估 | "离线测试" | 在标记数据集上运行候选模型/提示 |

## 进一步阅读

- [SigNoz — 顶级 LLM 可观测性工具 2026](https://signoz.io/comparisons/llm-observability-tools/)
- [Langfuse — Arize AX 替代方案分析](https://langfuse.com/faq/all/best-phoenix-arize-alternatives)
- [PremAI — 设置 Langfuse、LangSmith、Helicone、Phoenix](https://blog.premai.io/llm-observability-setting-up-langfuse-langsmith-helicone-phoenix/)
- [OpenTelemetry GenAI 语义惯例](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Arize Phoenix 文档](https://docs.arize.com/phoenix)
- [Helicone 文档](https://docs.helicone.ai/)
