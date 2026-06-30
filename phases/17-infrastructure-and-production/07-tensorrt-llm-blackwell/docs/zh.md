# Blackwell 上的 TensorRT-LLM，使用 FP8 和 NVFP4

> TensorRT-LLM 仅限 NVIDIA，但在 Blackwell 上它赢了。在配备 Dynamo 编排的 GB200 NVL72 上，SemiAnalysis InferenceX 在 2026 年 Q1-Q2 测得 120B 模型的 $0.012/M token，相对于 H100 + vLLM 的 $0.09/M — 7 倍的经济差距。该栈是三种浮点精度的复合叠加：FP8 对 KV 缓存和注意力内核仍然至关重要，因为它们需要动态范围；NVFP4（4 位微缩放）处理权重和激活值；多 token 预测 (MTP) 和解耦预填充/解码在上层再叠加 2-3 倍。第 0 天模型支持直接加载 FP4 权重，无需训练后转换。2026 年工程团队的陷阱：TRT-LLM 是封闭的 NVIDIA 栈，采用它意味着用可移植性换吞吐量。在提交之前，对你的模型和硬件组合进行计算。

**Type:** Learn
**Languages:** Python (stdlib, toy FP8/NVFP4 memory and cost calculator)
**Prerequisites:** Phase 17 · 04 (vLLM Serving Internals), Phase 10 · 13 (Quantization)
**Time:** ~75 minutes

## 学习目标

- 解释为什么当权重使用 NVFP4 时 FP8 对 KV 缓存和注意力仍然至关重要。
- 计算前沿模型在 BF16、FP8 和 NVFP4 下的 HBM 占用，并推理节省从何而来。
- 说出 TRT-LLM 利用的 Blackwell 特定特性（第 0 天 FP4、MTP、解耦推理、全对全原语）。
- 决定 TRT-LLM 的 NVIDIA 锁定在什么时候值得相较于 Hopper 上 vLLM 的 7 倍成本差距。

## 问题

2026 年推理经济学的前沿是"每美元的 token 数"。答案取决于四个叠加的选择：硬件代次（Hopper H100/H200 vs Blackwell B200/GB200）、精度（BF16 → FP8 → NVFP4）、推理引擎（vLLM vs SGLang vs TRT-LLM）和编排（普通 vs 解耦 vs Dynamo）。

在 Hopper 上使用 vLLM，一个 120B MoE 运行在约 $0.09/M token。在 Blackwell 上使用 TRT-LLM + Dynamo，同一模型运行在约 $0.012 — 7 倍便宜。部分差距来自硬件（Blackwell 单 GPU LLM 吞吐量是 Hopper 的 11-15 倍）。部分来自栈：FP4 权重、MTP 草稿、解耦预填充/解码，以及用于 MoE 专家通信的 NVLink 5 全对全。

你不能在 NVIDIA 栈之外复现这些。这就是权衡 — 用可移植性换经济学。理解每个栈选择贡献了多少份额的差距是本课的意义。

## 概念

### 为什么 FP8 仍然是 KV 缓存的下限

2026 年的一个常见错误：假设 NVFP4 到处适用。并非如此。KV 缓存需要 FP8（8 位浮点），因为它存储的注意力键和值跨越很宽的动态范围。将 KV 量化到 FP4 会导致灾难性的精度损失 — 分布的尾部掉落了，注意力分数崩溃。FP8 的指数位给了 KV 缓存需要的范围。

NVFP4（2025-2026）适用于权重和激活值。微缩放：每个权重块有自己的缩放因子，使小块可以跨越不同的动态范围而没有每张量缩放损失。对于激活值，FP4 可以维持，因为激活值在层内是小范围的。

典型的 Blackwell 配置：

- 权重：NVFP4（4 位微缩放）。
- 激活值：NVFP4。
- KV 缓存：FP8。
- 注意力累加器：FP32（softmax 稳定性）。

### TRT-LLM 使用的 Blackwell 特定原语

- **第 0 天 FP4 权重**：模型供应商直接发布 FP4 权重；TRT-LLM 无需训练后转换即可加载。FP4 无需 AWQ / GPTQ 步骤。
- **多 token 预测 (MTP)**：与 EAGLE（Phase 17 · 05）相同的思想，但集成到 TRT-LLM 构建中。
- **解耦推理**：预填充和解码在独立的 GPU 池上，KV 缓存通过 NVLink 或 InfiniBand 传输。与 Dynamo 相同的思想（Phase 17 · 20）。
- **全对全通信原语**：NVLink 5 将 MoE 专家通信延迟相比 Hopper 削减了 3 倍。TRT-LLM 的 MoE 内核为此调优。
- **NVFP4 + MXFP8 微缩放**：Blackwell Tensor Cores 上的硬件加速缩放因子处理。

### 你应该记住的数字

- HGX B200 通过 TRT-LLM 在 GPT-OSS-120B 上 $0.02/M token。
- GB200 NVL72 通过 Dynamo（编排 TRT-LLM）$0.012/M token。
- H100 + vLLM 在同类工作负载上 ≈ $0.09/M token。
- 三个月内 TRT-LLM 更新带来 2.8 倍吞吐量提升（2026 年）。
- Blackwell vs Hopper 单 GPU LLM 吞吐量：11-15 倍。
- MLPerf Inference v6.0（2026 年 4 月）：Blackwell 在每个提交任务上占主导。

### FP4 在质量上实际付出了什么

NVFP4 是激进的。在推理密集型工作负载（思维链、数学、长上下文代码生成）上，FP4 权重明显退化。每块校准缓解但不能消除。交付推理模型的团队经常使用 FP8 权重 + FP4 激活值作为折中，或坚持使用全 FP8 的 H200。

规则：在承诺使用 NVFP4 权重之前，始终在你的评估集上验证任务质量。

### 为什么这是 NVIDIA 锁定决策

TRT-LLM 是 C++ + CUDA + 闭源内核。模型需要为特定 GPU SKU 编译。没有 AMD、没有 Intel、没有 ARM。如果你的基础设施策略是多供应商，TRT-LLM 对于 TRT-LLM 服务层级是不可行的 — 你仍然可以在混合硬件上通过 vLLM 提供服务。如果你是纯 NVIDIA，7 倍的差距就值得锁定。

### 2026 年实践配方

对于每年 $1 亿以上的推理账单，在 Hopper + vLLM 上运行意味着错失了 7-10 倍的产出。将成本主导的工作负载迁移到 Blackwell + TRT-LLM + Dynamo。保留实验层在 H100 + vLLM 上以获取模型迭代速度。在投入生产之前验证每个 NVFP4 转换模型的质量。

### 解耦奖励

TRT-LLM 的解耦推理（分离的预填充和解码池）在 Phase 17 · 20 中深度涵盖。在 Blackwell 上，乘数叠加：FP4 权重 × MTP 加速 × 解耦放置 × 缓存感知路由。7 倍的数值假设了完整的栈。

```figure
pipeline-parallel
```

## 使用它

`code/main.py` 计算跨三个栈的模型 HBM 占用、解码吞吐量（内存带宽限制区域）和 $/M-token：H100 + BF16 + vLLM、H100 + FP8 + vLLM、B200 + NVFP4/FP8 + TRT-LLM。运行它以查看复合效应和每个变化贡献的差距份额。

## 交付它

本课产出 `outputs/skill-trtllm-blackwell-advisor.md`。给定工作负载、模型大小和年度 token 量，决定 Blackwell + TRT-LLM 栈是否值得 NVIDIA 锁定。

## 练习

1. 运行 `code/main.py`。对于 30% 活跃参数的 120B MoE，在 H100 BF16、H100 FP8 和 B200 NVFP4/FP8 上计算内存带宽限制的解码吞吐量。最大跃升从哪里来？
2. 一个客户在 H100 + vLLM 上每年花费 $2M。给定 7 倍经济差距，他们需要购买多少 Blackwell GPU 的盈亏平衡数才能在 12 个月内摊销迁移到 TRT-LLM？
3. 你在 NVFP4 权重转换后看到 MATH 准确率下降 3 分。说出两条恢复路径：一条质量优先（保持 FP8 权重）、一条成本优先（用领域内数据校准）。
4. 阅读 MLPerf v6.0 推理结果。哪个任务有最小的 Blackwell-over-Hopper 差距，为什么？
5. 计算一个 405B 模型在 NVFP4 权重 + FP8 KV 缓存 128k 上下文下所需的 HBM。它能否放入单个 GB200 NVL72 节点？

## 核心术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| FP8 | "八位浮点" | 8 位浮点；由于动态范围用于 KV 缓存和注意力 |
| NVFP4 | "四位微" | NVIDIA 的 4 位微缩放 FP 格式；Blackwell 上的权重和激活值 |
| MXFP8 | "MX 八" | 微缩放 FP8 变体；Blackwell Tensor Cores 上硬件加速 |
| 第 0 天 FP4 | "发布 FP4 权重" | 模型供应商发布已是 FP4 的权重；无需训练后转换步骤 |
| MTP | "多 token 预测" | TRT-LLM 集成的推测解码草稿 (Phase 17 · 05) |
| 解耦推理 | "分离预填充/解码" | 预填充和解码在独立 GPU 池上；KV 通过 NVLink/IB 传输 |
| 全对全 | "MoE 专家通信" | 将 token 路由到专家 GPU 的通信模式；NVLink 5 削减 3 倍 |
| InferenceX | "SemiAnalysis 推理基准" | 2026 年行业接受的每 token 成本基准 |

## 进一步阅读

- [NVIDIA — Blackwell Ultra MLPerf Inference v6.0](https://developer.nvidia.com/blog/nvidia-blackwell-ultra-sets-new-inference-records-in-mlperf-debut/) — 2026 年 4 月 MLPerf 结果。
- [NVIDIA — Blackwell 上的 MoE 推理](https://developer.nvidia.com/blog/delivering-massive-performance-leaps-for-mixture-of-experts-inference-on-nvidia-blackwell/) — NVLink 5 全对全和 MoE 内核。
- [TensorRT-LLM 概览](https://nvidia.github.io/TensorRT-LLM/overview.html) — 官方引擎文档。
- [NVIDIA — 引入 Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/) — TRT-LLM 之上的解耦编排。
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) — 发布 Blackwell 数字的基准套件。
