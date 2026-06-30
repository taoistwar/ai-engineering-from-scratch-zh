# 推理指标 — TTFT、TPOT、ITL、Goodput、P99

> 四个指标决定推理部署是否正常工作。TTFT 是预填充加队列加网络。TPOT（等价于 ITL）是每个 token 的内存受限解码成本。端到端延迟是 TTFT 加 TPOT 乘以输出长度。吞吐量是整个集群聚合的每秒 token 数。但对产品重要的是有效吞吐量 — 同时满足所有 SLO 的请求占比。高吞吐量低有效吞吐量意味着你在处理从未按时到达用户的 token。2026 年 Llama-3.1-8B-Instruct 在 TRT-LLM 上的参考数字：均值 TTFT 162 毫秒，均值 TPOT 7.33 毫秒，均值 E2E 1,093 毫秒。始终报告 P50、P90、P99 — 绝不仅仅报告均值。还要注意测量陷阱：GenAI-Perf 将 TTFT 排除在 ITL 计算之外，LLMPerf 将其包含在内；两个工具对同一运行的 TPOT 不一致。

**Type:** Learn
**Languages:** Python (stdlib, toy percentile calculator and goodput reporter)
**Prerequisites:** Phase 17 · 04 (vLLM Serving Internals)
**Time:** ~60 minutes

## 学习目标

- 精确定义 TTFT、TPOT、ITL、E2E、吞吐量和有效吞吐量，并说出每个指标测量的组件。
- 解释为什么均值是 LLM 推理的错误统计量，以及如何解读 P50/P90/P99。
- 构建 SLO 多约束（例如 TTFT<500ms 且 TPOT<15ms 且 E2E<2s）并据此计算有效吞吐量。
- 说出两个对同一运行的 TPOT 不一致的基准测试工具，并解释原因。

## 问题

"我们的吞吐量是每秒 15,000 token。"那又怎样？如果 40% 的请求端到端超过了 2 秒，用户就放弃了会话。吞吐量本身不能告诉你产品是否正常工作。

推理有多条延迟轴，每条失败方式不同。预填充是计算受限的，随提示长度缩放。解码是内存受限的，随批次大小缩放。排队延迟是运维问题。网络是物理距离问题。你需要针对每个的不同指标、需要百分位数、需要一个单一的复合指标来表述"用户得到了他们期望的结果吗" — 那就是有效吞吐量。

## 概念

### TTFT — 首 token 时间

`TTFT = queue_time + network_request + prefill_time`

当提示很长时，预填充占主导。在 H100 上的 Llama-3.3-70B FP8，32k 提示需要约 800 毫秒的纯预填充。队列时间是负载下调度器的行为。网络请求是包括 TLS 的线路时间。TTFT 是用户在收到任何流式回复前看到的延迟。

### TPOT / ITL — token 间延迟

同一个量的多种名称。`TPOT`（每输出 token 时间）、`ITL`（token 间延迟）、`每 token 解码延迟` — 都一样。它是首个 token 之后连续流式 token 之间的时间。

`TPOT = (decode_forward_time + scheduler_overhead) / tokens_produced`

在相同带分块预填充的 Llama-3.3-70B H100 栈上，TPOT 均值约 7 毫秒。没有分块预填充时，在相邻序列的长预填充期间，TPOT 可能飙升至 50 毫秒。关注 P99，而非均值。

### E2E 延迟

`E2E = TTFT + TPOT * output_tokens + network_response`

对于长输出（>500 token），E2E 由 TPOT 主导。对于短输出加长提示，E2E 由 TTFT 主导。报告以输出长度为条件的 E2E。

### 吞吐量

`throughput = total_output_tokens / elapsed_time`

聚合指标。告诉你集群效率。不告诉你单个请求的健康状况。

### Goodput（有效吞吐量）— 你真正关心的指标

`goodput = fraction of requests meeting (TTFT <= a) AND (TPOT <= b) AND (E2E <= c)`

SLO 是多约束的。一个请求只有在每个约束都满足时才是"好"的。有效吞吐量就是这个占比。60% 有效吞吐量的高吞吐量是失败的。99% 有效吞吐量的较低吞吐量才是目标。

在 2026 年，有效吞吐量是 MLPerf Inference v6.0 提交和 AI 平台供应商内部 SLA 跟踪中使用的指标。

### 为什么均值是错误的统计量

LLM 延迟分布是右偏的。一个有长预填充相邻序列的解码批次可以以 TPOT ~7 毫秒发送 500 个 token，以 TPOT ~60 毫秒发送 20 个 token。均值 TPOT 是 9 毫秒。P99 TPOT 是 65 毫秒。用户频繁命中 P99 — 这就是他们离开的原因。

始终报告三元组 (P50, P90, P99)。对于用户体验，P99 是你优化的目标。

### 参考数字 — Llama-3.1-8B-Instruct 在 TRT-LLM，2026

- 均值 TTFT：162 毫秒
- 均值 TPOT：7.33 毫秒
- 均值 E2E：1,093 毫秒
- P99 TPOT：根据分块预填充配置在 10-25 毫秒之间变化。

这些是发布的 NVIDIA 参考点。它们随模型大小（70B 会显示 3-5 倍）、硬件（H100 vs B200 约 3 倍）和负载而变化。

### 测量陷阱

2026 年最常用的两个基准测试工具对同一运行的 TPOT 不一致：

- **NVIDIA GenAI-Perf**：将 TTFT 排除在 ITL 计算之外。ITL 从 token 2 开始。
- **LLMPerf**：包含 TTFT。ITL 从 token 1 开始。

对于一个 TTFT 500 毫秒、100 个输出 token 总计 700 毫秒解码的请求，GenAI-Perf 报告 `ITL = 700/99 = 7.07 毫秒`，LLMPerf 报告 `ITL = 1200/100 = 12.00 毫秒`。工具的选择改变了数字。

始终说明工具。始终发布定义。

### 构建 SLO

2026 年面向消费者的 70B 聊天模型的合理 SLO：

- TTFT P99 <= 800 毫秒。
- TPOT P99 <= 25 毫秒。
- E2E P99 <= 3 秒（对于 <300 token 输出）。
- 有效吞吐量目标 >= 99%。

企业 SLO 收紧 TTFT（200-400 毫秒）并放宽 E2E。要点是写下它们，测量所有三者，并将有效吞吐量作为单一复合指标跟踪。

### 如何测量

- 运行真实流量或逼真的合成流量（LLMPerf 使用 `--mean-input-tokens 800 --stddev-input-tokens 300 --mean-output-tokens 150`）。
- 以 2 倍峰值并发作为基准运行目标。
- 运行 30-50 次迭代，取组合样本的百分位数。
- 与工具名称、工具版本、模型、硬件、并发、提示分布一起发布。

```figure
throughput-latency
```

## 使用它

`code/main.py` 是一个玩具有效吞吐量计算器。生成合成延迟分布、应用 SLO 并计算有效吞吐量。还展示了同一追踪上 GenAI-Perf vs LLMPerf 的 TPOT 差异。

## 交付它

本课产出 `outputs/skill-slo-goodput-gate.md`。给定工作负载和 SLO，生成 CI/CD 就绪的基准配方，以有效吞吐量而非吞吐量作为部署门控。

## 练习

1. 运行 `code/main.py`。生成一个有 1% 尾部尖峰的分布。当你将 P99 TPOT 从 30 毫秒收紧到 15 毫秒时，有效吞吐量如何变化？
2. 供应商引用"Llama 3.3 70B H100 上 15,000 tok/s"。在相信之前说出要问的三个问题。
3. 为什么分块预填充保护 P99 TPOT 而非均值 TPOT？
4. 为语音助手构建消费者 SLO（首 token 是被听到的，不是被读到的）。哪个指标是用户最能感知的？
5. 阅读 LLMPerf README 和 GenAI-Perf 文档。找出三个其他工具不一致的指标。

## 核心术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| TTFT | "首 token 时间" | 队列 + 网络 + 预填充；长提示时由预填充主导 |
| TPOT | "每输出 token 时间" | 首 token 后每个 token 的内存受限解码成本 |
| ITL | "token 间延迟" | 大多数工具中与 TPOT 相同（不是全部 — 见 GenAI-Perf） |
| E2E | "端到端" | TTFT + TPOT * 输出长度；其上加上响应侧网络 |
| 吞吐量 | "tok/s" | 集群效率；没有延迟百分位数则无意义 |
| Goodput（有效吞吐量） | "SLO 满足率" | 同时满足所有 SLO 约束的请求占比 |
| P99 | "尾部" | 100 次中的 1 次最坏情况延迟；用户体验指标 |
| SLO 多约束 | "联合约束" | 所有三个延迟边界的 AND；任一违反则请求失败 |
| GenAI-Perf vs LLMPerf | "工具陷阱" | 工具在 ITL 是否包含 TTFT 上有分歧 |

## 进一步阅读

- [NVIDIA NIM — LLM 基准测试指标](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html) — TTFT、ITL、TPOT 的规范定义。
- [Anyscale — LLM 推理基准测试指标](https://docs.anyscale.com/llm/serving/benchmarking/metrics) — 替代定义和测量配方。
- [BentoML — LLM 推理指标](https://bentoml.com/llm/inference-optimization/llm-inference-metrics) — 真实部署上的应用测量。
- [LLMPerf](https://github.com/ray-project/llmperf) — 基于 Ray 的开源基准。
- [GenAI-Perf](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client/src/c++/perf_analyzer/genai-perf/README.html) — NVIDIA 的基准测试工具。
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) — 行业接受的基于有效吞吐量的基准。
