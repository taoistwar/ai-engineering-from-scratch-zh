# 实践项目 11 — LLM 可观测性与评估仪表盘

> Langfuse 走向开放核心。Arize Phoenix 发布了 2026 年 GenAI 语义约定映射。Helicone 和 Braintrust 都加倍投入按用户成本归属。Traceloop 的 OpenLLMetry 成为事实上的 SDK 插桩。生产形态是 ClickHouse 用于追踪，Postgres 用于元数据，Next.js 用于 UI，以及一组小型评估任务（DeepEval、RAGAS、LLM-judge）在采样的追踪上运行。构建一个自托管版本，从至少四个 SDK 系列摄入，并演示在五分钟内捕获注入的回归。

**类型:** 实践项目
**语言:** TypeScript（UI），Python / TypeScript（摄入 + 评估），SQL（ClickHouse）
**前置条件:** 阶段 11（LLM 工程），阶段 13（工具），阶段 17（基础设施），阶段 18（安全）
**涉及的阶段:** P11 · P13 · P17 · P18
**时间:** 25 小时

## 问题

2026 年每个运行生产流量的 AI 团队都在模型旁边维护一个可观测性平面。成本归属。幻觉检测。漂移监控。越狱信号。SLO 仪表盘。PII 泄漏告警。开源参考——Langfuse、Phoenix、OpenLLMetry——收敛于 OpenTelemetry GenAI 语义约定作为摄入模式。你现在可以用一个 SDK 插桩 OpenAI、Anthropic、Google、LangChain、LlamaIndex 和 vLLM，并发送兼容的 span。

你将构建一个自托管仪表盘，从至少四个 SDK 系列摄入，在采样的追踪上运行一组小型评估任务，检测漂移，并报警。衡量标准：给定一个故意注入的回归（开始产生 PII 的提示），仪表盘在五分钟内捕获它并触发警报。

## 概念

摄入是 OTLP HTTP。SDK 产生 GenAI 语义约定的 span：`gen_ai.system`、`gen_ai.request.model`、`gen_ai.usage.input_tokens`、`gen_ai.response.id`、`llm.prompts`、`llm.completions`。Span 进入 ClickHouse 进行列式分析；元数据（用户、会话、应用）进入 Postgres。

评估作为批处理任务在采样的追踪上运行。DeepEval 评分忠实度、毒性和答案相关性。当追踪携带检索上下文时，RAGAS 评分检索指标。自定义 LLM-judges 运行领域特定检查（PII 泄漏、偏离策略的响应）。评估运行将评估 span 写回同一个 ClickHouse，链接到父追踪。

漂移检测随时间监控嵌入空间分布（提示嵌入上的 PSI 或 KL 散度）加上评估分数趋势。告警输入 Prometheus Alertmanager，然后 Slack / PagerDuty。UI 是 Next.js 15 带 Recharts。

## 架构

```
production apps:
  OpenAI SDK  +  Anthropic SDK  +  Google GenAI SDK
  LangChain + LlamaIndex + vLLM
       |
       v
  OpenTelemetry SDK with GenAI semconv
       |
       v  OTLP HTTP
  collector (ingest, sample, fan-out)
       |
       +-------------+-----------+
       v             v           v
   ClickHouse    Postgres    S3 archive
   (spans)       (metadata)  (raw events)
       |
       +---> eval jobs (DeepEval, RAGAS, LLM-judge)
       |     sampled or all-trace
       |     write eval spans back
       |
       +---> drift detector (PSI / KL on prompt embeddings)
       |
       +---> Prometheus metrics -> Alertmanager -> Slack / PagerDuty
       |
       v
   Next.js 15 dashboard (Recharts)
```

## 技术栈

- 摄入: OpenTelemetry SDK + GenAI 语义约定；OTLP HTTP 传输
- 采集器: OpenTelemetry Collector 带尾采样处理器（用于成本控制）
- 存储: ClickHouse 用于 span，Postgres 用于元数据，S3 用于原始事件归档
- 评估: DeepEval、RAGAS 0.2、Arize Phoenix 评估器包、自定义 LLM-judge
- 漂移: 提示嵌入上的 PSI / KL（sentence-transformers），每周
- 告警: Prometheus Alertmanager -> Slack / PagerDuty
- UI: Next.js 15 App Router + Recharts + server actions
- 开箱即用支持的 SDK：OpenAI、Anthropic、Google GenAI、LangChain、LlamaIndex、vLLM

## 构建它

1. **采集器配置。** OpenTelemetry Collector 带 OTLP HTTP 接收器，尾采样器保留 100% 的错误追踪和 10% 的成功，并将输出发送到 ClickHouse 和 S3。

2. **ClickHouse 模式。** 表 `spans` 包含映射 GenAI 语义约定的列：`gen_ai_system`、`gen_ai_request_model`、`input_tokens`、`output_tokens`、`latency_ms`、`prompt_hash`、`trace_id`、`parent_span_id`，加上用于长载荷的 JSON 包。添加按 user_id 和 app_id 的次级索引。

3. **SDK 覆盖率测试。** 使用每个 SDK（OpenAI、Anthropic、Google、LangChain、LlamaIndex、vLLM）编写一个小型客户端应用，使用 OpenLLMetry 自动插桩。验证每个产生落在 ClickHouse 中的规范 GenAI span。

4. **评估任务。** 一个计划任务读取最近 15 分钟的采样追踪，运行 DeepEval 忠实度、毒性和答案相关性。输出是链接到父追踪的评估 span。

5. **自定义 LLM-judge。** 一个 PII 泄漏判断器：给定一个响应，调用守卫 LLM 对 PII 泄漏可能性进行评分。高分数响应落入分类队列。

6. **漂移检测。** 每周作业计算本周汇聚提示嵌入与追踪的 4 周基线之间的 PSI。如果 PSI 超过阈值，报警。

7. **仪表盘。** Next.js 15 包含页面：概览（spans/秒、成本/用户、p95 延迟）、追踪（搜索 + 瀑布）、评估（忠实度趋势、毒性）、漂移（PSI 随时间变化）、告警。

8. **告警链路。** Prometheus 输出器读取评估分数聚合和延迟百分位；Alertmanager 路由到 Slack（警告）和 PagerDuty（严重违反）。

9. **回归探测。** 注入一个错误：被评估的聊天机器人开始 1% 的时间泄漏虚假的 SSN。测量 MTTR：从错误部署到 Slack 告警。

## 使用它

```
$ curl -X POST https://my-otel-collector/v1/traces -d @trace.json
[collector]  accepted 1 trace, 3 spans
[clickhouse] inserted 3 spans (app=chat, user=u_42)
[eval]       DeepEval faithfulness 0.82, toxicity 0.03
[drift]      weekly PSI 0.08 (below 0.2 threshold)
[ui]         live at https://obs.example.com
```

## 交付它

`outputs/skill-llm-observability.md` 是可交付成果。给定一个 LLM 应用，仪表盘摄入其追踪，运行评估，在漂移上报警，并在 Next.js 中展示成本/用户细分。

| 权重 | 标准 | 如何衡量 |
|:-:|---|---|
| 25 | 追踪模式覆盖率 | 产生规范 GenAI span 的 SDK 系列数量（目标：6+） |
| 20 | 评估正确性 | DeepEval / RAGAS 分数与手动标注集对比 |
| 20 | 仪表盘 UX | 注入回归的 MTTR（目标低于 5 分钟） |
| 20 | 成本 / 规模 | 持续以 1k spans/秒摄入而无积压 |
| 15 | 告警 + 漂移检测 | Prometheus/Alertmanager 链路端到端运用 |
| **100** | | |

## 练习

1. 为 Haystack 框架添加自定义插桩。验证规范 span 带有忠实的 `gen_ai.*` 属性落入 ClickHouse。

2. 在相同追踪上将 DeepEval 替换为 Phoenix 评估器。测量两个评估引擎之间的分数漂移。

3. 锐化漂移检测器：按 app-id 而非全局计算 PSI。显示每个应用的漂移轨迹。

4. 添加"用户影响"页面：每用户成本和每用户失败率带迷你图。

5. 构建尾采样策略，保留 100% 毒性 > 0.5 的追踪加上其余 10% 的分层样本。测量引入的采样偏差。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| GenAI semconv | "OTel LLM 属性" | 2025 年 OpenTelemetry 规范，用于 LLM span 属性（system、model、tokens） |
| Tail sampling | "追踪后采样" | 采集器在追踪完成后决定保留或丢弃（可以查看错误） |
| PSI | "群体稳定性指数" | 比较两个分布的漂移指标；> 0.2 通常表示有意义的漂移 |
| LLM-judge | "评估即模型" | 一个 LLM 根据评分标准对另一个 LLM 的输出进行评分（忠实度、毒性、PII） |
| Tail-sampling policy | "保留规则" | 决定哪些追踪持久化 vs 丢弃的规则；错误 + 采样率 |
| Eval span | "链接的评估追踪" | 携带链接到原始 LLM 调用 span 的评估分数的子 span |
| Cost per user | "单位经济学" | 在时间窗口内归因于 user_id 的美元成本；关键产品指标 |

## 扩展阅读

- [Langfuse](https://github.com/langfuse/langfuse) — 参考开放核心可观测性平台
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — 替代参考，具有强大的漂移支持
- [OpenLLMetry (Traceloop)](https://github.com/traceloop/openllmetry) — 自动插桩 SDK 家族
- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 摄入模式
- [Helicone](https://www.helicone.ai) — 替代托管可观测性
- [Braintrust](https://www.braintrust.dev) — 替代评估优先平台
- [ClickHouse 文档](https://clickhouse.com/docs) — 列式 span 存储
- [DeepEval](https://github.com/confident-ai/deepeval) — 评估器库
