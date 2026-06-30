# EAGLE-3 生产环境中的推测解码

> 推测解码将一个快速草稿模型与目标模型配合。草稿提出 K 个 token；目标在一次前向传播中验证；被接受的 token 是免费的。在 2026 年，EAGLE-3 是生产级别的变体 — 它在目标模型的隐藏状态上训练草稿头，而非原始 token，在通用聊天上将接受率 alpha 推入 0.6-0.8 区间。正确的问题不是"草稿多快"而是"在我流量上的 alpha 是多少？"如果 alpha 降到约 0.55 以下，推测解码在高并发下是净负面的，因为每次拒绝的草稿都要付出第二次目标前向的代价。本课教你首先测量 alpha，其次再拨动开关。

**Type:** Learn
**Languages:** Python (stdlib, toy acceptance-rate simulator)
**Prerequisites:** Phase 17 · 04 (vLLM Serving Internals), Phase 10 · 18 (Multi-Token Prediction)
**Time:** ~60 minutes

## 学习目标

- 说出推测解码的三代演进，并解释 EAGLE-3 相比 EAGLE-2 和经典草稿模型改变了什么。
- 定义接受率 alpha，从 alpha 和 K（草稿长度）计算期望加速比，并识别目标并发下的盈亏平衡 alpha。
- 解释为什么 2026 年 vLLM 中推测解码是选择性开启（而非默认），以及为什么在没有测量 alpha 的情况下开启它是生产反模式。
- 编写一个测量计划：使用哪个基准测试、哪种提示分布、哪个并发点、以哪个指标作为门控。

## 问题

解码是内存受限的。在运行 Llama 3.3 70B FP8 的 H100 上，每个解码 token 读取约 140 GB/s 的权重并发出一个 token。GPU 计算在解码期间几乎空闲 — 瓶颈是 HBM 带宽，而非矩阵乘法吞吐量。

推测解码利用这一差距。用一个廉价的草稿模型生成 K 个候选 token，然后让目标模型在一次前向传播中验证所有 K 个 token。每个被验证的 token 实际上是免费的（分摊到目标模型本来也必须做的 K 个批次前向传播中）。

经典的草稿模型方法使用同家族的较小模型（Llama 3.2 1B 为 Llama 3.3 70B 起草）。它能工作但接受率一般 — 较小模型的分布与目标分化。EAGLE，然后 EAGLE-2，然后 EAGLE-3 直接在目标模型的内部状态上训练一个轻量草稿头，因此草稿的分布更紧密地跟踪目标。这就是为什么 alpha 从使用草稿模型的 0.4 变为 EAGLE-3 的 0.6-0.8。

陷阱：2026 年 vLLM 中 EAGLE-3 是选择性开启的。`speculative_config` 必须显式设置。没有标志，没有加速。在真实流量上不先测量 alpha 就开启它的团队常常看到尾部延迟变差，而非变好。

## 概念

### 推测解码实际带来了什么

没有推测解码时，每个 token 的成本是一次目标前向传播。在草稿长度 K 和接受率 alpha 下使用推测解码，每次目标前向传播的预期 token 数是 `1 + K * alpha`。加速比是 `(1 + K * alpha) / (1 + epsilon)`，其中 epsilon 是草稿加验证的开销。对 K=5，alpha=0.7：`(1 + 5*0.7) / (1 + 0.1) = 4.5 / 1.1 = 4.1 倍`。真实世界中数字聚集在 2-3 倍，因为 alpha 在生产流量上很少那么高，且 epsilon 在大批次下增长。

### 为什么 alpha 是唯一重要的指标

被拒绝的 token 并未消失 — 它们为第一个被拒绝的 token 强制了一次额外的目标前向传播。在 alpha 降至 0.4 的工作负载上，你支付了草稿开销加验证再加重新生成。在高并发下（比如 256 并发），解码批次已经大到"仅目标"和"目标带验证"之间的内存带宽差距缩小。在大多数 2026 年硬件上，alpha 低于 0.55 时推测解码是净负面的。

Alpha 因工作负载而异。在 ShareGPT 风格的通用聊天上，在 ShareGPT 上训练的 EAGLE-3 达到 0.6-0.8。在领域特定流量上（代码、医疗、法律），在通用数据上训练的草稿头降至 0.4-0.6。训练一个领域特定的草稿头可以恢复 alpha — 相对于目标微调，这是一个轻量快速的训练作业。

### EAGLE 各代一览

- **经典草稿模型**：同家族的较小模型。Alpha 0.3-0.5。基础设施简单 — 加载两个模型，草稿每次目标前向运行 K 次前向。
- **EAGLE-1 (2024)**：在目标隐藏状态（最后一层）上训练的单个草稿头。Alpha ~0.5-0.6。在目标之上参数开销小。
- **EAGLE-2 (2025)**：自适应草稿长度和基于树的草稿（在一次目标传播中验证多个分支）。Alpha ~0.6-0.7。草稿调度器更复杂。
- **EAGLE-3 (2025-2026)**：在多个目标层（不仅是最后一层）上训练的草稿头，更好的对齐。通用聊天上 Alpha ~0.6-0.8。

### 2026 年生产配方

1. 以普通方式交付目标模型。在目标并发下测量基线 TTFT、ITL、吞吐量。
2. 通过 vLLM `speculative_config` 启用 EAGLE-3 草稿。重新运行基准测试。
3. 记录接受率 alpha。vLLM V1 将其报告为 `spec_decode_metrics.accepted_tokens_per_request`。除以请求的草稿长度得到 alpha。
4. 如果在生产流量分布上 alpha < 0.55，禁用推测解码或训练一个领域特定的 EAGLE-3 草稿。
5. 在生产并发下重新运行。确认 P99 ITL 没有变差。

### 生产陷阱：P99 尾部

均值 ITL 随推测解码下降。如果未调优，P99 可能变差。被拒绝的草稿触发两阶段序列（草稿 + 验证失败 + 重新生成）。在满批次下，这两个阶段是序列化的。关注 P99 ITL，而不是 P50。

### EAGLE-3 的部署现状

Google 在 2025 年将推测解码部署到 AI Overviews 中（同等质量，更快响应）。vLLM V1 将 `speculative_config` 作为文档记录的接口提供；V1 中的 N-gram GPU 推测解码是与分块预填充兼容的变体。SGLang 将 EAGLE-3 作为前缀密集型工作负载的推荐草稿路径。

### 一行中的盈亏平衡数学

期望加速比：`S(alpha, K) = (1 + K*alpha) / (1 + verify_overhead)`。设置 `S = 1` 解出 alpha：`alpha_breakeven = verify_overhead / K`。对于典型的 verify_overhead ~0.15 和 K=5：`alpha_breakeven = 0.03`。但那是原始解码数学。在高并发下验证开销上升，并且解码批次已经在序列间分摊内存读取，因此实际有效 alpha_breakeven 在实践中攀升至约 0.45-0.55。

### 何时不使用推测解码

- 延迟不重要的单批次离线生成。使用普通目标。
- 非常短的输出（少于 50 个 token）。草稿开销和验证成本占主导。
- 没有领域训练草稿头的专业领域。Alpha 太低。
- vLLM v0.18.0 加上草稿模型推测解码加上 `--enable-chunked-prefill`。此组合不编译。V1 中记录的例外是 N-gram GPU 推测解码。

## 使用它

`code/main.py` 在一系列 alpha 值和草稿长度 K 上模拟有和没有推测解码的解码循环。它打印盈亏平衡 alpha、测量的加速比和尾部行为。在多个 (alpha, K) 组合上运行，精确查看推测解码在哪里停止回报。

## 交付它

本课产出 `outputs/skill-eagle3-rollout.md`。给定目标模型、流量分布描述和并发目标，它生成一个分阶段的 EAGLE-3 部署计划 — 基准基线、启用配置、测量 alpha、以 alpha >= 0.55 为门控、观察 P99 ITL。

## 练习

1. 运行 `code/main.py`。在 K=5 下，2 倍加速需要什么 alpha？3 倍加速呢？这对 verify_overhead 有多敏感？
2. 想象生产流量分为 70% 通用聊天、30% 代码。通用聊天在 ShareGPT 上训练的 EAGLE-3 上命中 alpha 0.7；代码命中 alpha 0.4。混合 alpha 是多少？推测解码是否为净正面？
3. 阅读 vLLM `speculative_config` 文档。说出三种模式（草稿模型、EAGLE、N-gram）以及哪种与分块预填充兼容。
4. 你看到启用 EAGLE-3 后均值 ITL 下降 25% 但 P99 ITL 上升了 15%。诊断并提出缓解方案。
5. 计算 Llama 3.3 70B 上 EAGLE-3 草稿头的内存成本。与作为经典草稿运行 Llama 3.2 1B 相比如何？

## 核心术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 推测解码 | "草稿加验证" | 用廉价模型提出 K 个 token，在一次目标前向中验证所有 K 个 |
| 接受率 alpha | "推测接受率" | 被目标接受的草稿 token 比例；唯一重要的指标 |
| 草稿长度 K | "推测 k" | 草稿每次目标前向提出多少 token；典型 4-8 |
| 验证开销 epsilon | "推测开销" | 验证加重新生成相比普通目标前向的额外成本；随批次增长 |
| EAGLE-3 | "最新 EAGLE" | 2025-2026 变体；在多个目标层上训练草稿头；通用聊天上 alpha 0.6-0.8 |
| `speculative_config` | "vLLM 推测配置" | vLLM V1 中的显式选择性启用配置；无默认意味着无加速 |
| N-gram 推测解码 | "N-gram 草稿" | 在提示中使用 N-gram 查找的 GPU 端草稿；与分块预填充兼容 |
| 盈亏平衡 alpha | "无操作 alpha" | 推测解码给出零加速的 alpha；在生产并发下关注此值 |
| 拒绝草稿两阶段 | "重新生成成本" | 草稿被拒绝时的两次目标前向传播；驱动 P99 尾部 |

## 进一步阅读

- [vLLM — 推测解码文档](https://docs.vllm.ai/en/latest/features/spec_decode/) — V1 中 `speculative_config` 和分块预填充兼容性的权威来源。
- [vLLM 推测配置 API](https://docs.vllm.ai/en/latest/api/vllm/config/speculative/) — 精确的字段集。
- [EAGLE 论文 (arXiv:2401.15077)](https://arxiv.org/abs/2401.15077) — 原始 EAGLE 草稿头公式。
- [EAGLE-2 论文 (arXiv:2406.16858)](https://arxiv.org/abs/2406.16858) — 自适应草稿和树形结构。
- [UC Berkeley EECS-2025-224](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-224.html) — 带推测解码的高效 LLM 系统。
- [BentoML — 推测解码](https://bentoml.com/llm/inference-optimization/speculative-decoding) — 生产部署检查清单。
