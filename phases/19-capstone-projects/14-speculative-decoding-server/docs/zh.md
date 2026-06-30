# 实践项目 14 — 推测解码推理服务器

> vLLM 0.7 中的 EAGLE-3 在真实流量上提供 2.5-3 倍的吞吐量。P-EAGLE（AWS 2026）进一步推动了并行推测。SGLang 的 SpecForge 在规模上训练了草稿头。Red Hat 的 Speculators 中心发布了针对常见开放模型的对齐草稿。TensorRT-LLM 使推测解码在 NVIDIA 上成为一等公民。2026 年的生产服务栈是 vLLM 或 SGLang 配合 EAGLE 系列草稿、FP8 或 INT4 量化，以及队列等待上的 HPA。这个实践项目是以 2.5 倍以上的基线吞吐量服务两个开放模型，并提供完整的尾部延迟报告。

**类型:** 实践项目
**语言:** Python（服务），C++ / CUDA（内核检查），YAML（配置）
**前置条件:** 阶段 3（深度学习），阶段 7（transformers），阶段 10（从头构建 LLM），阶段 17（基础设施）
**涉及的阶段:** P3 · P7 · P10 · P17
**时间:** 30 小时

## 问题

推测解码在 2026 年成为了商品。EAGLE-3 草稿头在目标模型的隐藏状态上进行训练，并预测前方 N 个 token；目标模型在单次传递中验证。60-80% 的接受率转化为 2-3 倍端到端吞吐量。vLLM 0.7 原生集成了这一点。SGLang + SpecForge 为你提供训练管道。Red Hat 的 Speculators 发布了针对 Llama 3.3 70B、Qwen3-Coder-30B MoE、GPT-OSS-120B 的对齐草稿。

工艺在于服务运营，而不是模型本身。接受率随流量分布漂移（ShareGPT vs 代码 vs 领域数据）。拒绝下的尾部延迟比没有推测时更高——你必须报告多个批处理大小下的 p99，而不仅仅是稳态 token/秒。与 Anthropic / OpenAI API 对比的每 1M token 成本是可信度杠杆。

## 概念

推测解码有两层。一个**草稿**模型（EAGLE-3 头、ngram 或较小的目标对齐模型）在每一步提出 k 个候选 token。**目标**模型在单次传递中验证所有 k 个；任何被接受的前缀替换贪婪路径。接受率取决于草稿-目标对齐度和输入分布。

EAGLE-3 在大多数流量上击败 ngram 草稿。P-EAGLE 为更深的草稿树运行并行推测。权衡：拒绝时的 P99 延迟更高，因为验证传递更大。服务配置必须报告按批处理大小分桶的延迟来展示这一点。

部署是 Kubernetes。vLLM 0.7 在每 GPU 或张量并行分片上运行一个副本。HPA 根据队列等待而不是 CPU 自动扩展。FP8（Marlin）和 INT4（AWQ）量化将 GPU 内存保持在 H100 / H200 包络内。端到端报告是吞吐量、接受率、批处理 1/8/32 下的 p50/p99，以及 $/1M token。

## 架构

```
request ingress
    |
    v
vLLM server (0.7) or SGLang (0.4)
    |
    +-- draft: EAGLE-3 heads | P-EAGLE parallel | ngram fallback
    +-- target: Llama 3.3 70B | Qwen3-Coder-30B | GPT-OSS-120B
    |     quantized FP8-Marlin or INT4-AWQ
    |
    v
verify pass: batch k draft tokens through target
    |
    v (accept prefix; resample for rejected suffix)
    v
token stream back to client
    |
    v
Prometheus metrics: throughput, acceptance rate, queue wait, latency p50/p99
    |
    v
HPA on queue-wait metric
```

## 技术栈

- 服务: vLLM 0.7 或 SGLang 0.4
- 推测方法: EAGLE-3 草稿头、P-EAGLE 并行推测、ngram 回退
- 草稿训练: SpecForge（SGLang）或 Red Hat Speculators
- 目标模型: Llama 3.3 70B、Qwen3-Coder-30B MoE、GPT-OSS-120B
- 量化: FP8（Marlin）、INT4 AWQ
- 部署: Kubernetes + NVIDIA device plugin；队列等待指标上的 HPA
- 评估: ShareGPT、MT-Bench-v2、GSM8K、HumanEval 用于领域分布的接受率测量
- 参考: TensorRT-LLM 推测解码用于供应商基线

## 构建它

1. **目标模型准备。** 选择 Llama 3.3 70B。通过 Marlin 量化到 FP8。在 1xH100（或 2x 张量并行）上以 vLLM 0.7 部署。

2. **草稿源。** 从 Red Hat Speculators 拉取对齐的 EAGLE-3 草稿头（或通过 SpecForge 训练一个）。加载到 vLLM 的推测解码配置中。

3. **基线数字。** 在推测之前：批处理 1/8/32 下的 token/s、p50/p99 延迟、GPU 利用率。发布。

4. **启用 EAGLE-3。** 翻转配置；重新运行相同的基准。报告加速、接受率、p99 尾部延迟差值。

5. **P-EAGLE。** 启用并行推测；测量更深的草稿树 vs 串行 EAGLE-3。报告 P-EAGLE 有帮助 vs 有害的拐点。

6. **领域流量。** 通过同一服务器运行 ShareGPT vs HumanEval vs 领域特定流量。测量每个分布的接受率。识别草稿何时漂移。

7. **第二个目标模型。** 在 Qwen3-Coder-30B MoE 上运行相同的管道。草稿更棘手（MoE 路由噪声）。报告。

8. **K8s HPA。** 在 K8s 下部署，HPA 跟踪 `queue_wait_ms`。在负载增加三倍时演示横向扩展。

9. **成本比较。** 在相同评估上计算 $/1M token vs Anthropic Claude Sonnet 4.7 和 OpenAI GPT-5.4。发布。

## 使用它

```
$ curl https://infer.example.com/v1/chat/completions -d '{"messages":[...]}'
[serve]     vLLM 0.7, Llama 3.3 70B FP8, EAGLE-3 active
[decode]    bs=8, accepted_tokens_per_step=3.2, acceptance_rate=0.76
[latency]   first-token 42ms, full-response 980ms (620 tokens)
[cost]      $0.34 per 1M output tokens at sustained throughput
```

## 交付它

`outputs/skill-inference-server.md` 描述了可交付成果。一个经过测量的服务栈，带有推测解码、完整基准报告和 K8s 部署。

| 权重 | 标准 | 如何衡量 |
|:-:|---|---|
| 25 | 测量到的加速 vs 基线 | 在两个模型上匹配质量下 2.5 倍+ 吞吐量 |
| 20 | 真实流量上的接受率 | 每个分布的接受率报告 |
| 20 | P99 尾部延迟纪律 | 带有和没有推测时批处理 1/8/32 下的 p99 |
| 20 | 运维 | K8s 部署、队列等待上的 HPA、平稳部署 |
| 15 | 撰写和方法论 | 清晰解释什么改变了以及为什么 |
| **100** | | |

## 练习

1. 测量当草稿比目标落后一个版本时（例如，Llama 3.3 -> 3.4 漂移）接受率的下降。构建一个监控告警。

2. 实现 ngram 回退：如果 EAGLE-3 接受率降到阈值以下，切换到 ngram 草稿。报告可靠性改进。

3. 运行受控 MoE 实验：相同 Qwen3-Coder-30B 注入路由噪声 vs 不注入。测量草稿接受敏感性。

4. 扩展到 H200（141 GB）。报告获得的每副本模型大小余量，以及是否可以服务未量化的 Llama 3.3 70B。

5. 在相同 H100 硬件上对 TensorRT-LLM 推测解码进行基准测试。报告它在何处胜过 vLLM。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| Draft model | "推测器" | 小模型，提出 N 个 token 供目标验证 |
| EAGLE-3 | "2026 年草稿架构" | 在目标隐藏状态上训练的草稿头；约 75% 接受率 |
| P-EAGLE | "并行推测" | 草稿分支树在一次目标传递中验证 |
| Acceptance rate | "命中率" | 无需重新采样就接受的草稿 token 比例 |
| Quantization | "FP8 / INT4" | 更低精度的权重，以在 GPU 内存中容纳更大的模型 |
| Queue wait | "HPA 指标" | 请求在推理开始前在待处理队列中等待的时间 |
| Speculators hub | "对齐草稿" | Red Hat Neural Magic 中心，包含针对常见开放模型的 EAGLE 草稿 |

## 扩展阅读

- [vLLM EAGLE 和 P-EAGLE 文档](https://docs.vllm.ai) — 参考服务栈
- [P-EAGLE (AWS 2026)](https://aws.amazon.com/blogs/machine-learning/p-eagle-faster-llm-inference-with-parallel-speculative-decoding-in-vllm/) — 并行推测解码论文 + 集成
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) — 草稿头训练管道
- [Red Hat Speculators](https://github.com/neuralmagic/speculators) — 对齐草稿中心
- [TensorRT-LLM 推测解码](https://nvidia.github.io/TensorRT-LLM/) — 供应商替代方案
- [Fireworks.ai 服务架构](https://fireworks.ai/blog) — 商业参考
- [EAGLE-3 论文 (arXiv:2503.01840)](https://arxiv.org/abs/2503.01840) — 方法论文
- [vLLM 仓库](https://github.com/vllm-project/vllm) — 代码和基准
