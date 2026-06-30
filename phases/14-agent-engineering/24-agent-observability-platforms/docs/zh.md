# Agent 可观测性：Langfuse、Phoenix、Opik

> 2026 年主导的三个开源 Agent 可观测性平台。Langfuse（MIT）— 每月 6M+ 安装量，追踪 + 提示管理 + 评估 + 会话回放。Arize Phoenix（Elastic 2.0）— 深度 agent 专用评估、RAG 相关性、OpenInference 自动检测。Comet Opik（Apache 2.0）— 自动提示优化、护栏、LLM 裁判幻觉检测。

**类型：** 学习
**语言：** Python（标准库）
**前置条件：** 第 14 阶段 · 23（OTel GenAI）
**时间：** 约 45 分钟

## 学习目标

- 说出三个顶级开源 Agent 可观测性平台及其许可证。
- 区分各自最强项：Langfuse（提示管理 + 会话）、Phoenix（RAG + 自动检测）、Opik（优化 + 护栏）。
- 解释为何到 2026 年 89% 的组织报告已部署 agent 可观测性。
- 实现一个带 LLM 裁判评估的标准库追踪到仪表盘流水线。

## 问题

OTel GenAI（第 23 课）给出了模式。你仍然需要一个能够摄取 span、运行评估、存储提示版本和发现回归的平台。这三个竞争者各自强调生命周期的不同部分。

## 概念

### Langfuse（MIT）

- 每月 6M+ SDK 安装量，19k+ GitHub stars。
- 功能：追踪、带版本控制与演练场的提示管理、评估（LLM 作裁判、用户反馈、自定义）、会话回放。
- 2025 年 6 月：此前为商业模块的功能（LLM 作裁判、标注队列、提示实验、Playground）以 MIT 协议开源。
- 最强项：端到端可观测性，具有紧密的提示管理闭环。

### Arize Phoenix（Elastic License 2.0）

- 更深度的 agent 专用评估：追踪聚类、异常检测、RAG 检索相关性。
- 原生 OpenInference 自动检测。
- 与托管 Arize AX 配对用于生产环境。
- 无提示版本管理 — 定位为与其他平台并用的漂移/行为回归工具。
- 最强项：RAG 相关性、行为漂移、异常检测。

### Comet Opik（Apache 2.0）

- 通过 A/B 实验进行自动提示优化。
- 护栏（PII 脱敏、主题约束）。
- LLM 裁判幻觉检测。
- Comet 自身测量的基准：Opik 日志 + 评估 23.44 秒 vs Langfuse 327.15 秒（约 14 倍差距）— 将厂商基准视为方向性参考。
- 最强项：优化闭环、自动实验、护栏执行。

### 行业数据

根据 Maxim（2026 年现场分析）：89% 的组织已部署 agent 可观测性；质量问题是最主要的生产障碍（32% 的受访者提及）。

### 选择平台

| 需求 | 选择 |
|------|------|
| 带提示管理的一站式方案 | Langfuse |
| 深度 RAG 评估 + 漂移 | Phoenix |
| 自动优化 + 护栏 | Opik |
| 开放许可，无 ELv2 | Langfuse（MIT）或 Opik（Apache 2.0） |
| Datadog / New Relic 集成 | 任选 — 它们都导出 OTel |

### 此模式的常见错误

- **无评估策略。** 没有评估的追踪只是昂贵的日志记录。
- **自建无事实依据的 LLM 裁判。** 适用 CRITIC 模式（第 05 课）— 裁判需要外部工具来进行事实性验证。
- **提示版本未与追踪关联。** 当生产出现回归时，你无法二分定位到导致问题的提示。

## 构建它

`code/main.py` 实现了标准库追踪收集器 + LLM 裁判评估器：

- 摄取 GenAI 格式的 span。
- 按会话分组，标记失败的运行（护栏触发、低置信度评估）。
- 基于量规对 agent 响应打分的脚本化 LLM 裁判。
- 仪表盘式摘要：失败率、顶级失败原因、评估分数分布。

运行它：

```
python3 code/main.py
```

输出：每个会话的评估分数和失败分类，与 Langfuse/Phoenix/Opik 的显示内容一致。

## 使用它

- **Langfuse** 自托管或云端；通过 OTel 或其 SDK 接入。
- **Arize Phoenix** 自托管；自动检测 OpenInference。
- **Comet Opik** 自托管或云端；自动优化循环。
- **Datadog LLM Observability** 适用于已在使用 Datadog 的混合运维+ML 团队。

## 交付它

`outputs/skill-obs-platform-wiring.md` 选择一个平台并将追踪 + 评估 + 提示版本接入现有 agent。

## 练习

1. 将一周的 OTel 追踪导出到 Langfuse 云端（免费层）。哪些会话失败了？为什么？
2. 为你的领域编写 LLM 裁判量规（事实正确性、语气、范围遵守）。在 50 条追踪上测试。
3. 比较 Langfuse 提示版本管理与 Phoenix 追踪聚类。哪个能更快告诉你哪里出错了？
4. 阅读 Opik 的护栏文档。将 PII 脱敏护栏接入你的一次 agent 运行。
5. 在你的语料库上对三者进行基准测试。忽略厂商发布的数字；测量你自己的。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| 追踪 | "Span 收集器" | 摄取 OTel / SDK span；按会话索引 |
| 提示管理 | "提示 CMS" | 与追踪关联的版本化提示 |
| LLM 作裁判 | "自动评估" | 独立 LLM 根据量规对 agent 输出打分 |
| 会话回放 | "追踪回放" | 逐步走过历史运行用于调试 |
| RAG 相关性 | "检索质量" | 检索到的上下文是否匹配查询 |
| 追踪聚类 | "行为分组" | 将相似运行聚类用于漂移检测 |
| 护栏执行 | "日志时策略" | 对记录内容进行 PII/毒性/范围检查 |

## 扩展阅读

- [Langfuse 文档](https://langfuse.com/) — 追踪、评估、提示管理
- [Arize Phoenix 文档](https://docs.arize.com/phoenix) — 自动检测、漂移
- [Comet Opik](https://www.comet.com/site/products/opik/) — 优化 + 护栏
- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 三者都消费的模式
