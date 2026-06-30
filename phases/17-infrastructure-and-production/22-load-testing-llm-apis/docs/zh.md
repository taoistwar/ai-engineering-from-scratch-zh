# LLM API 负载测试 — 为什么 k6 和 Locust 会撒谎

> 传统负载测试工具不是为流式响应、可变输出长度、token 级指标或 GPU 饱和设计的。两个陷阱咬到大多数团队。GIL 陷阱：Locust 的 token 级测量在 Python GIL 下运行分词器，在高并发下与请求生竞争；分词积压然后膨胀报告的 token 间延迟 — 你的客户端才是瓶颈，而非服务器。提示均匀性陷阱：循环中的相同提示测试 token 分布上的一个点；真实流量具有可变长度和多样的前缀匹配。LLMPerf 用 `--mean-input-tokens` + `--stddev-input-tokens` 解决了这个问题。2026 年工具映射：LLM 专用（GenAI-Perf、LLMPerf、LLM-Locust、guidellm）用于 token 级精度；**k6 v2026.1.0** + **k6 Operator 1.0 GA（2025 年 9 月）** — 感知流式传输、通过 TestRun/PrivateLoadZone CRD 实现 Kubernetes 原生分布式，最适合 CI/CD 门控；Vegeta 用于 Go 恒定速率饱和；Locust 2.43.3 仅在使用 LLM-Locust 扩展用于流式传输时。负载模式：稳态、斜坡、尖峰（自动扩展测试）、浸泡（内存泄漏）。

**Type:** Build
**Languages:** Python (stdlib, toy realistic-prompt generator + latency collector)
**Prerequisites:** Phase 17 · 08 (Inference Metrics), Phase 17 · 03 (GPU Autoscaling)
**Time:** ~75 minutes

## 学习目标

- 解释使通用负载测试工具对 LLM API 撒谎的两个反模式（GIL 陷阱、提示均匀性陷阱）。
- 为给定目的选择工具：LLMPerf（基准测试运行）、k6 + 流式扩展（CI 门控）、guidellm（大规模合成）、GenAI-Perf（NVIDIA 参考）。
- 设计四种负载模式（稳态、斜坡、尖峰、浸泡）并说出每种捕获的失败模式。
- 使用输入 token 的均值 + 标准差构建逼真的提示分布，而非固定长度。

## 问题

你在 500 并发用户下对 LLM 端点进行了 k6 测试。它顶住了。你发布了。在 200 个实际用户的生产中，服务崩溃了 — P99 TTFT 爆炸，GPU 被占用。

两件事发生了。首先，k6 发送了 500 个相同提示 — 你的请求合并和前缀缓存让看起来你在处理 500 个并发的解码，而实际上你在处理一个。其次，k6 不跟踪流式响应上 token 间延迟，不像人眼体验的那样；它看到一个 HTTP 连接，而非以不同间隔到达的 500 个 token。

LLM 负载测试是其自身的学科。

## 概念

### GIL 陷阱（Locust）

Locust 使用 Python 并在 GIL 下运行客户端分词。高并发下，分词器排在请求生后面。报告的 token 间延迟包括客户端分词积压。你认为服务器慢了；是测试工具本身。

修复：LLM-Locust 扩展将分词移到独立进程，或使用编译语言工具（k6、使用 tokenizers.rs 的 LLMPerf）。

### 提示均匀性陷阱

所有已知负载测试器让你配置一个提示。在 10,000 次迭代的循环测试中，每次都发送完全相同的提示。服务器每次都看到相同的前缀 — 前缀缓存命中接近 100%，吞吐量看起来很棒。

修复：从提示分布中采样。LLMPerf 使用 `--mean-input-tokens 500 --stddev-input-tokens 150` — 多种长度，多种内容。

### 四种负载模式

1. **稳态** — 恒定 RPS 持续 30-60 分钟。捕获：基线性能退化。
2. **斜坡** — 线性从 0 增加到目标 RPS 超过 15 分钟。捕获：容量断点、预热异常。
3. **尖峰** — 突然 3-10 倍 RPS 持续 2 分钟然后返回。捕获：自动扩展延迟、队列饱和、冷启动影响。
4. **浸泡** — 稳态持续 4-8 小时。捕获：内存泄漏、连接池漂移、可观测性溢出。

### 2026 年工具映射

**LLMPerf** (Anyscale) — Python 但 Rust 支持分词。均值/标准差提示。感知流式。性能运行的最佳默认。

**NVIDIA GenAI-Perf** — NVIDIA 参考。使用 Triton 客户端；全面指标覆盖。注意其 ITL 排除 TTFT；LLMPerf 包含它。两个工具为相同服务器生成不同的 TPOT。

**LLM-Locust** (TrueFoundry) — 修复 GIL 陷阱的 Locust 扩展。熟悉的 Locust DSL + 流式指标。

**guidellm** — 大规模合成基准测试。

**k6 v2026.1.0** + **k6 Operator 1.0 GA（2025 年 9 月）**：
- k6 本身（Go、编译、无 GIL）添加了感知流式传输的指标。
- k6 Operator 使用 TestRun / PrivateLoadZone CRD 进行 Kubernetes 原生分布式测试。
- 最适合 CI/CD 门控和 SLA 测试。

**Vegeta** — Go，比 k6 更简单。恒定速率 HTTP 饱和。不感知 LLM 但适合网关/速率限制测试。

**Locust 2.43.3 原生** — 对 LLM 有 GIL 陷阱。仅在使用 LLM-Locust 扩展时适用。

### CI 中的 SLA 门控

在 PR 上运行 k6 使用：

- 每个在基线 RPS 下 30-50 次迭代。
- 门控：P50/P95 TTFT、5xx < 5%、TPOT 在阈值下。
- 突破时构建失败。

### 逼真的提示分布

从真实流量样本构建（如果你有的话）或从发布的分布构建（例如，ShareGPT 提示用于聊天，HumanEval 用于代码）。将均值 + 标准差输入 LLMPerf。不惜一切代价避免单提示循环。

### 你应该记住的数字

- k6 Operator 1.0 GA：2025 年 9 月。
- k6 v2026.1.0：感知流式传输的指标。
- 典型 LLMPerf 运行：并发 X 下 100-1000 个请求。
- 典型 CI 门控：每 PR 30-50 次迭代。
- 四种模式：稳态、斜坡、尖峰、浸泡。

## 使用它

`code/main.py` 用逼真提示分布模拟负载测试，测量有效 TPOT，并演示统一提示陷阱。

## 交付它

本课产出 `outputs/skill-load-test-plan.md`。给定工作负载和 SLA，选择工具并设计四种负载模式。

## 练习

1. 运行 `code/main.py`。比较统一 vs 逼真分布 — 差距在哪里？
2. 编写 CI 门控的 k6 脚本：100 并发下 TTFT P95 < 800 毫秒，运行 5 分钟。
3. 你的浸泡测试显示内存以 50 MB/小时增长。说出三个原因以及从中选择哪个的仪器化。
4. 从 10 RPS 到 100 RPS 的尖峰测试。如果 Karpenter + vLLM 生产栈就位（Phase 17 · 03 + 18），预期恢复时间是多少？
5. GenAI-Perf 报告 TPOT=6ms；LLMPerf 在相同服务器上报告 TPOT=11ms。解释。

## 核心术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| LLMPerf | "LLM 工具" | Anyscale 基准测试工具，感知流式 |
| GenAI-Perf | "NVIDIA 工具" | NVIDIA 参考工具 |
| LLM-Locust | "Locust for LLMs" | 修复 GIL 陷阱的 Locust 扩展 |
| guidellm | "合成基准" | 大规模合成工具 |
| k6 Operator | "K8s k6" | 基于 CRD 的分布式 k6 |
| GIL trap | "Python 客户端开销" | 分词积压膨胀报告的延迟 |
| Prompt-uniformity trap | "单提示谎言" | 相同提示循环命中缓存，膨胀吞吐量 |
| Steady-state | "恒定负载" | 恒定 RPS 持续 N 分钟 |
| Ramp | "线性上升" | 持续时间内从 0 到目标 |
| Spike | "突发测试" | 突然倍增然后恢复 |
| Soak | "长测试" | 数小时用于泄漏检测 |

## 进一步阅读

- [TianPan — 负载测试 LLM 应用](https://tianpan.co/blog/2026-03-19-load-testing-llm-applications)
- [PremAI — 负载测试 LLM 2026](https://blog.premai.io/load-testing-llms-tools-metrics-realistic-traffic-simulation-2026/)
- [NVIDIA NIM — LLM 推理基准测试介绍](https://docs.nvidia.com/nim/large-language-models/1.0.0/benchmarking.html)
- [TrueFoundry — LLM-Locust](https://www.truefoundry.com/blog/llm-locust-a-tool-for-benchmarking-llm-performance)
- [LLMPerf](https://github.com/ray-project/llmperf)
- [k6 Operator](https://github.com/grafana/k6-operator)
