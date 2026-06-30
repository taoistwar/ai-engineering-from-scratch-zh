# 多区域 LLM 推理与 KV 缓存局部性

> 轮询负载均衡对缓存 LLM 推理是有害的。一个没有落在持有其前缀的节点上的请求需要付出全额预填充成本 — 在长提示上 P50 约 800 毫秒，而缓存命中时约 80 毫秒。2026 年的生产模式是缓存感知路由器（Rust 编写的 vLLM Router、llm-d router），它消费 KV 缓存事件并根据前缀哈希匹配进行路由。最近的研究（GORGO）将跨区域网络延迟作为路由目标中的显式项。商业"跨区域推理"产品（Bedrock 跨区域推理、GKE 多集群网关）将推理视为不透明的 — 它们处理可用性，而非 TTFT。JPMorgan 和 Mayo Clinic 在 2024 年 11 月运行 us-east-1 故障转移约 22 分钟。DR 现实：32% 的 LLM DR 失败是因为团队备份了权重但忘记了分词器文件或量化配置。

**Type:** Learn
**Languages:** Python (stdlib, toy prefix-cache-aware router simulator)
**Prerequisites:** Phase 17 · 04 (vLLM Serving), Phase 17 · 06 (SGLang RadixAttention)
**Time:** ~60 minutes

## 学习目标

- 解释为什么轮询负载均衡破坏缓存推理，并量化 TTFT 损失。
- 绘制缓存感知路由器：输入（KV 缓存事件）、算法（前缀哈希匹配）、平局决胜（GPU 利用率）。
- 说出 LLM 的 32% DR 失败驱动因素（缺失分词器文件/量化配置），并陈述三文件 DR 检查清单。
- 区分商业跨区域产品（Bedrock CRI、GKE 多集群网关）和 KV 感知路由。

## 问题

你的服务在 us-east-1、us-west-2 和 eu-west-1 运行。你在前面放了一个带轮询的 ALB。生产中的前缀缓存命中率降至 8%。TTFT P50 翻三倍。你的 vLLM 日志显示每个请求都在付出全额预填充成本。

轮询对无状态服务是最优的。LLM 推理本质上是带状态的 — KV 缓存编码了模型见过的所有内容。盲目路由就是路由到错误的缓存。

另外，你的团队有一个 DR 计划。你将模型权重备份到跨区域 S3。一个区域中断发生；你尝试故障转移；副本拒绝启动。你忘记了 `tokenizer.json`、量化配置和 RoPE 缩放配置在一个你没有同步的独立桶中。

多区域 LLM 推理是缓存问题、路由问题和 DR 规范问题 — 不是负载均衡器问题。

## 概念

### 缓存感知路由

请求带着提示到达。路由器对前缀（例如前 512 个 token）进行哈希；它询问每个副本"你有此外缀缓存吗？"副本在分配和驱逐块时在发布/订阅频道上发布 KV 缓存事件。路由器选择有匹配的副本，如果没有匹配则落在基于 GPU 利用率的平局决胜。

**vLLM Router**（Rust，2026 生产栈）：订阅 `kv.cache.block_added` 事件，维护前缀哈希→副本索引，以 O(1) 查找路由。无匹配时落在最少队列深度。

**llm-d router**：相同模式，Kubernetes 原生。通过 ControlPlane API 发布事件。

**SGLang RadixAttention**（Phase 17 · 06）是副本内等价物。跨副本路由严格在上游。

### 数字

2K-token 提示上的 TTFT P50，Llama 3.3 70B FP8，H100：
- 缓存命中（相同副本，前缀驻留）：~80 毫秒。
- 缓存未命中（冷预填充）：~800 毫秒。

10 倍差距。如果你的路由器在副本间命中 60-80% 的前缀缓存，你以 N 副本容量近似单副本性能。如果命中 10%，你近似朴素扩展。

### 跨区域有一个新约束 — 网络延迟

区域间 RTT：
- us-east-1 ↔ us-west-2：~65 毫秒。
- us-east-1 ↔ eu-west-1：~75 毫秒。
- us-east-1 ↔ ap-southeast-1：~220 毫秒。

如果路由将请求从 us-east-1 带到 ap-southeast-1 的热前缀，节省的预填充（800 → 80 毫秒）被 440 毫秒往返吞没。GORGO（2026 年研究）使其明确 — 联合最小化 `prefill_time + network_latency`，而非仅预填充。通常答案是保持区域路由，除非在低 MB 级前缀上预填充占主导。

### 商业"跨区域推理"在这里没有帮助

AWS Bedrock 跨区域推理在容量紧张时自动将请求路由到其他区域。它优化可用性，而非 TTFT，且将推理视为不透明。GKE 多集群网关相同 — 服务级故障转移，无 KV 缓存感知。

你仍然需要一个应用层缓存感知路由器，即使使用这些。它们处理"us-east-1 着火了"的情况。缓存感知路由处理 TTFT 情况。

### DR 规范 — 32% 缺失文件问题

2026 年广泛引用的统计数据：32% 的 LLM DR 失败是因为团队备份了权重但忘记了：

- `tokenizer.json` 或 `tokenizer.model`
- 量化配置（`quantize_config.json`、AWQ 缩放因子、GPTQ 零点）
- 模型特定配置（RoPE 缩放、注意力掩码、聊天模板）
- 引擎配置（`vllm_config.yaml`、采样默认值、LoRA 适配器清单）

修复是三文件最低 DR 清单：

1. HF 模型仓库下的所有文件（权重 + 配置 + 分词器）。
2. 引擎特定的推理配置。
3. 部署清单（K8s YAML、Dockerfile、依赖锁定）。

另外：每季度运行一次 DR 演练。JPMorgan 的 us-east-1 演练在 2024 年 11 月用了 22 分钟恢复，仅因为剧本经过了排练。

### 数据驻留是正交的

欧盟客户 PHI 不能离开欧盟。如果你的缓存感知路由器将巴黎发起的请求发送到 us-east-1 进行前缀匹配，无论 TTFT 提升多少你已经违反了 GDPR。在优化缓存之前按驻留边界分区路由器。

### 你应该记住的数字

- 缓存命中 vs 未命中 TTFT 差距：约 10 倍（2K 提示上 80 毫秒 vs 800 毫秒）。
- 区域间 RTT 美国-欧盟：~75 毫秒。
- DR 失败：32% 缺失分词器/量化配置。
- JPMorgan us-east-1 故障转移 2024 年 11 月：22 分钟（30 分钟 SLA）。

## 使用它

`code/main.py` 在多区域工作负载上模拟三种路由策略（轮询、缓存感知区域、缓存感知全球）。报告缓存命中率、TTFT P50/P99 和跨区域账单。

## 交付它

本课产出 `outputs/skill-multi-region-router.md`。给定区域、驻留约束和 SLA，设计路由计划。

## 练习

1. 运行 `code/main.py`。在什么提示长度下跨区域路由击败纯本地路由，给定 75 毫秒 RTT？
2. 你的缓存命中率从 70% 降至 12%。诊断三个可能原因和可确认每个原因的可观测数据。
3. 为在带 5 个 LoRA 适配器的 vLLM 中服务的 70B AWQ 量化模型设计 DR 清单。列出每个文件和配置。
4. 论证 Bedrock 跨区域推理是否对有严格 TTFT SLO 的金融科技"足够"。引用具体行为。
5. 巴黎发出的请求匹配 us-east-1 的前缀。你路由吗？写下策略。

## 核心术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 缓存感知路由 | "智能 LB" | 基于前缀哈希匹配路由到持有 KV 缓存的副本 |
| KV 缓存事件 | "缓存发布订阅" | 副本发布块添加/驱逐；路由器建立索引 |
| 前缀哈希 | "缓存键" | 前 N 个 token 的哈希值，用作路由器查找 |
| GORGO | "跨区域路由研究" | arXiv 2602.11688；网络延迟作为显式项 |
| 跨区域推理 | "Bedrock CRI" | AWS 产品；可用性故障转移，非 TTFT 感知 |
| DR 清单 | "备份列表" | 每个恢复所需文件 — 不仅仅是权重 |
| 数据驻留 | "GDPR 边界" | 哪个区域能看到用户数据的法律约束 |
| RTT | "往返时间" | 网络延迟；美国-欧盟 75 毫秒，美国-亚太 220 毫秒 |
| LLM 感知 LB | "缓存命中 LB" | 缓存感知路由器作为产品类别 |

## 进一步阅读

- [BentoML — 多云和跨区域推理](https://bentoml.com/llm/infrastructure-and-operations/multi-cloud-and-cross-region-inference)
- [arXiv — GORGO (2602.11688)](https://arxiv.org/html/2602.11688v1) — 带网络延迟项的跨区域 KV 缓存重用。
- [TianPan — 多区域 LLM 推理缓存局部性](https://tianpan.co/blog/2026-04-17-multi-region-llm-serving-data-residency-routing)
- [AWS Bedrock 跨区域推理](https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html) — 可用性故障转移文档。
- [vLLM 生产栈路由器](https://github.com/vllm-project/production-stack) — 缓存感知路由器源代码。
