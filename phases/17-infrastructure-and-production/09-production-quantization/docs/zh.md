# 生产量化 — AWQ、GPTQ、GGUF K-quants、FP8、MXFP4/NVFP4

> 量化格式不是一个通用选择 — 它是硬件、推理引擎和工作负载的函数。GGUF Q4_K_M 或 Q5_K_M 统治 CPU 和边缘，通过 llama.cpp 和 Ollama 交付。GPTQ 在 vLLM 中胜出，当你需要在同一基础模型上使用多 LoRA 时。AWQ 加 Marlin-AWQ 内核在 7B 级别模型上提供约 741 tok/s，并具有 INT4 下的最佳 Pass@1 — 2026 年数据中心生产的默认选择。FP8 在 Hopper、Ada 和 Blackwell 上保持中间地带 — 接近无损且广泛支持。NVFP4 和 MXFP4（Blackwell 微缩放）是激进的，需要逐块验证。两个陷阱会咬到团队：校准数据集必须匹配部署领域，且 KV 缓存与权重量化是分开的 — AWQ 课程中"我的模型现在是 4 GB"这句话忘记了生产批次大小下的 10-30 GB KV 缓存。

**Type:** Learn
**Languages:** Python (stdlib, toy memory and throughput comparison across formats)
**Prerequisites:** Phase 10 · 13 (Quantization foundations), Phase 17 · 04 (vLLM Serving Internals)
**Time:** ~75 minutes

## 学习目标

- 说出 2026 年六种生产量化格式及其最佳甜点。
- 给定硬件（CPU vs GPU、Hopper vs Blackwell）、引擎（vLLM、TRT-LLM、llama.cpp）和工作负载（常规聊天、推理、多 LoRA），选择格式。
- 计算所选格式节省的权重内存和保持不变的 KV 缓存。
- 指出会在领域流量上降低量化模型质量的校准数据集陷阱。

## 问题

量化减少了内存和 HBM 带宽，这正是解码所需要的。一个 FP16 70B 模型是 140 GB 的权重。将权重量化到 INT4（AWQ 或 GPTQ），模型是 35 GB — 可放入一个 H100，并为 KV 缓存留出空间，这一点很重要，因为在 128 并发序列和 2k 上下文下，仅 KV 缓存就是 20-30 GB。

但量化不是免费的。激进的量化会降低质量，尤其是在推理密集型任务上。不同格式与不同引擎配合。不同硬件原生支持不同精度。2026 年的格式种类繁多，你不能复制别人的选择 — 你必须根据自己的栈来选择。

## 概念

### 六种格式

| 格式 | 位 | 甜点 | 引擎 |
|--------|------|-----------|---------|
| GGUF Q4_K_M / Q5_K_M | 4-5 | CPU、边缘、笔记本 | llama.cpp、Ollama |
| GPTQ | 4-8 | vLLM 上的多 LoRA | vLLM、TGI |
| AWQ | 4 | 数据中心 GPU 生产 | vLLM (Marlin-AWQ)、TGI |
| FP8 | 8 | Hopper/Ada/Blackwell 数据中心 | vLLM、TRT-LLM、SGLang |
| MXFP4 | 4 | Blackwell 多用户 | TRT-LLM |
| NVFP4 | 4 | Blackwell 多用户 | TRT-LLM |

### GGUF — CPU/边缘默认

GGUF 是一种文件格式，本身不是一种量化方案 — 它将 K-quant 变体（Q2_K、Q3_K_M、Q4_K_M、Q5_K_M、Q6_K、Q8_0）打包在一个容器中。Q4_K_M 和 Q5_K_M 是生产默认 — 4-5 位下接近 BF16 的质量。CPU 或边缘推理的最优选择，因为 llama.cpp 是遥遥领先的最快 CPU 推理引擎。

在 vLLM 中的吞吐量损失：7B 上约 93 tok/s — 该格式未针对 GPU 内核优化。当部署目标是 CPU/边缘时使用 GGUF。否则不。

### GPTQ — vLLM 中的多 LoRA

GPTQ 是一种带校准传递的训练后量化算法。Marlin 内核使其在 GPU 上快速（相比非 Marlin GPTQ 有 2.6 倍加速）。7B 上约 712 tok/s。

独特的胜利：GPTQ-Int4 在 vLLM 中支持 LoRA 适配器。如果你正在服务一个基础模型加 10-50 个微调变体（每个作为 LoRA），GPTQ 是你的路径。截至 2026 年初，NVFP4 尚不支持 LoRA。

### AWQ — 数据中心 GPU 默认

激活感知权重量化。在量化过程中保护约 1% 最显著的权重。Marlin-AWQ 内核：相比朴素实现 10.9 倍加速。7B 上约 741 tok/s，INT4 格式中最佳 Pass@1。

为新 GPU 推理选择 AWQ，除非你需要多 LoRA（GPTQ）或激进的 Blackwell FP4（NVFP4）。

### FP8 — 可靠的中庸之选

8 位浮点。接近无损。广泛支持。Hopper Tensor Cores 原生加速 FP8。Blackwell 继承。当质量不可妥协时（推理、医疗、代码生成），FP8 是 2026 年安全默认。内存节省是 INT4 的一半，但质量风险低得多。

### MXFP4 / NVFP4 — Blackwell 激进之选

微缩放 FP4。每个权重块有自己的缩放因子。激进但在 Blackwell Tensor Cores 上硬件加速。相比 FP8 半字节每 token — Phase 17 · 07 中的经济收益。

注意事项：
- 尚无 LoRA 支持（2026 年初）。
- 推理密集型工作负载上可见质量下降。
- 按模型在你的评估集上验证。

### 校准陷阱

AWQ 和 GPTQ 需要校准数据集 — 通常是 C4 或 WikiText。对于领域模型（代码、医疗、法律），在通用网页文本上校准会让算法在保护哪些权重上做出错误决策。HumanEval 上的 Pass@1 可能下降数个百分点。

修复：在领域内数据上校准。数百个领域样本通常足够。在发布前在评估集上测试。

### KV 缓存陷阱

AWQ 将权重缩小到 4 位。KV 缓存是独立的，保持 FP16/FP8。对于带 AWQ 的 70B 模型：

- 权重：~35 GB（INT4，从 140 GB）。
- KV 缓存 128 并发 × 2k 上下文：~20 GB。
- 激活值：~5 GB。
- 总计：~60 GB — 可放入 H100 80GB。

天真的"我把模型量化到了 4 GB"忘记了另外 30-50 GB。全面预算 HBM。

另外，KV 缓存量化（FP8 KV 或 INT8 KV）是一个不同的选择，有自身的权衡 — 它直接影响注意力精度，不是免费收益。

### AWQ INT4 对推理是危险的

思维链、数学、长上下文代码生成 — 这些在激进量化下明显受损。AWQ INT4 在 MATH 上损失约 3-5 分。对于推理密集型工作负载，部署 FP8 或 BF16；接受内存成本。

### 2026 年选择指南

- CPU/边缘推理：GGUF Q4_K_M。完毕。
- GPU 推理、常规聊天、无 LoRA：AWQ。
- GPU 推理、多 LoRA：GPTQ 加 Marlin。
- 推理工作负载：FP8。
- Blackwell 数据中心、已验证质量：NVFP4 + FP8 KV。
- 模糊不清：在每个候选格式上运行 1,000 样本评估。

```figure
gpu-memory-breakdown
```

## 使用它

`code/main.py` 计算跨六种格式在多种模型大小下的内存占用（权重 + KV + 激活值）和相对吞吐量。展示 KV 缓存在哪里占主导、权重压缩在哪里有回报以及 FP8 在哪里是安全选择。

## 交付它

本课产出 `outputs/skill-quantization-picker.md`。给定硬件、模型大小、工作负载类型和质量容忍度，选择格式并生成校准/验证计划。

## 练习

1. 运行 `code/main.py`。对于 128 并发 2k 上下文的 70B 模型，计算每种格式的总 HBM。哪种格式让你能放入一个 H100 80GB？
2. 你有一个 7B 编码模型。选择一个格式并论证。如果你对质量容忍度的判断错了，恢复路径是什么？
3. 计算为医疗领域模型校准 AWQ 所需的校准数据集大小。为什么更多数据并不总是更好？
4. 阅读 Marlin-AWQ 内核论文或发布说明。用三句话解释为什么 AWQ 在 7B 上达到 741 tok/s 而原始 GPTQ 达到 ~712。
5. 什么时候将 AWQ 权重与 FP8 KV 缓存组合有意义，对比保持 KV 在 BF16？

## 核心术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| GGUF | "llama.cpp 格式" | 打包 K-quant 变体的文件格式；CPU/边缘默认 |
| Q4_K_M | "Q4 K M" | 4 位 K-quant 中等；生产 GGUF 默认 |
| GPTQ | "jee pee tee q" | 带校准的训练后 INT4；在 vLLM 中支持 LoRA |
| AWQ | "a w q" | 激活感知 INT4；Marlin 内核；INT4 中最佳 Pass@1 |
| Marlin 内核 | "快速 INT4 内核" | 在 Hopper 上为 INT4 定制的 CUDA 内核；10 倍加速 |
| FP8 | "八位浮点" | Hopper/Ada/Blackwell 上的安全精度默认 |
| MXFP4 / NVFP4 | "微缩放四" | Blackwell 4 位 FP，具有每块缩放因子 |
| 校准数据集 | "校准数据" | 用于选择量化参数的输入文本；必须匹配领域 |
| KV 缓存量化 | "KV INT8" | 与权重不同的独立选择；影响注意力精度 |

## 进一步阅读

- [VRLA Tech — LLM 量化 2026](https://vrlatech.com/llm-quantization-explained-int4-int8-fp8-awq-and-gptq-in-2026/) — 比较基准。
- [Jarvis Labs — vLLM 量化完整指南](https://jarvislabs.ai/blog/vllm-quantization-complete-guide-benchmarks) — 按格式的吞吐量数字。
- [PremAI — GGUF vs AWQ vs GPTQ vs bitsandbytes 2026](https://blog.premai.io/llm-quantization-guide-gguf-vs-awq-vs-gptq-vs-bitsandbytes-compared-2026/) — 逐格式选择。
- [vLLM 文档 — 量化](https://docs.vllm.ai/en/latest/features/quantization/index.html) — 支持的格式和标志。
- [AWQ 论文 (arXiv:2306.00978)](https://arxiv.org/abs/2306.00978) — 原始 AWQ 公式。
- [GPTQ 论文 (arXiv:2210.17323)](https://arxiv.org/abs/2210.17323) — 原始 GPTQ 公式。
