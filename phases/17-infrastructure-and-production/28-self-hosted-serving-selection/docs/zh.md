# 自托管推理选择 — llama.cpp、Ollama、TGI、vLLM、SGLang

> 2026 年四个引擎主导自托管推理。基于硬件、规模和工作负载进行选择。**llama.cpp** 在 CPU 上最快 — 最广泛的模型支持、对量化和线程的完全控制。**Ollama** 是开发笔记本的一键安装，比 llama.cpp 慢 15-30%（Go + CGo + HTTP 序列化），类似生产负载下有 3 倍吞吐量差距。**TGI 于 2025 年 12 月 11 日进入维护模式** — 仅错误修复，原始吞吐量比 vLLM 慢约 10%，但历史上可观测性和 HF 生态系统集成最佳。该维护状态使其成为长期的风险赌注 — SGLang 或 vLLM 对新项目是更安全的默认选择。**vLLM** 是通用生产默认 — v0.15.1（2026 年 2 月）添加了 PyTorch 2.10、RTX Blackwell SM120、H200 优化。**SGLang** 是代理多轮/前缀密集型专家 — 生产中有 400,000+ GPU（xAI、LinkedIn、Cursor、Oracle、GCP、Azure、AWS）。硬件约束：仅 CPU → 只 llama.cpp。AMD / 非 NVIDIA → 只 vLLM（TRT-LLM 是 NVIDIA 锁定的）。2026 年流水线模式：开发 = Ollama、预发 = llama.cpp、生产 = vLLM 或 SGLang。全程使用相同的 GGUF/HF 权重。

**Type:** Learn
**Languages:** Python (stdlib, engine-decision tree walker)
**Prerequisites:** All Phase 17 lessons covering engines (04, 06, 07, 09, 18)
**Time:** ~45 minutes

## 学习目标

- 给定硬件（CPU / AMD / NVIDIA Hopper / Blackwell）、规模（1 用户 / 100 / 10,000）和工作负载（通用聊天 / 代理 / 长上下文），选择引擎。
- 说出 2026 年 TGI 维护模式状态（2025 年 12 月 11 日）以及为什么它使新项目偏向 vLLM 或 SGLang。
- 描述使用相同 GGUF 或 HF 权重的开发/预发/生产流水线。
- 解释为什么"仅 CPU"强制 llama.cpp，"AMD"排除 TRT-LLM。

## 问题

你的团队开始一个新的自托管 LLM 项目。一个工程师说 Ollama，另一个说 vLLM，第三个说"TGI 不是开箱即用吗？"对于不同的上下文，三者都是正确的。对于所有上下文，没有一个是对的。

2026 年的选择树很重要：硬件第一、规模第二、工作负载第三。而一个特定的 2025 年事件 — TGI 于 12 月 11 日进入维护模式 — 改变了新项目的默认选择。

## 概念

### 五个引擎

| 引擎 | 最适合 | 备注 |
|--------|----------|-------|
| **llama.cpp** | CPU / 边缘 / 最少依赖 / 最广泛模型支持 | CPU 上最快，完全控制 |
| **Ollama** | 开发笔记本、单用户、一键安装 | 比 llama.cpp 慢 15-30%；3 倍生产吞吐量差距 |
| **TGI** | HF 生态系统、受监管行业 | **2025 年 12 月 11 日维护模式** |
| **vLLM** | 通用生产、100+ 用户 | 广泛的生产默认；v0.15.1 2026 年 2 月 |
| **SGLang** | 代理多轮、前缀密集型工作负载 | 生产中有 400,000+ GPU |

### 硬件优先决策

**仅 CPU** → llama.cpp。Ollama 也可以但更慢。没有其他引擎在 CPU 上有竞争力。

**AMD GPU** → vLLM（AMD ROCm 支持）。SGLang 也可以。TRT-LLM 是 NVIDIA 锁定的，所以排除。

**NVIDIA Hopper (H100 / H200)** → vLLM 或 SGLang 或 TRT-LLM。三者都是顶级。

**NVIDIA Blackwell (B200 / GB200)** → TRT-LLM 是吞吐量领先者（Phase 17 · 07）。vLLM 和 SGLang 紧随。

**Apple Silicon (M 系列)** → llama.cpp (Metal)。Ollama 包装了这。

### 规模第二决策

**1 用户 / 本地开发** → Ollama。一条命令，首 token 秒出。

**10-100 用户 / 小团队** → vLLM 单 GPU。

**100-10k 用户 / 生产** → vLLM 生产栈（Phase 17 · 18）或 SGLang。

**10k+ 用户 / 企业** → vLLM 生产栈 + 解耦（Phase 17 · 17）+ LMCache（Phase 17 · 18）。

### 工作负载第三决策

**通用聊天 / 问答** → vLLM 在广泛默认上胜出。

**代理多轮（工具、规划、记忆）** → SGLang 的 RadixAttention（Phase 17 · 06）占主导。

**前缀重用密集的 RAG** → SGLang。

**代码生成** → vLLM 可以；SGLang 在缓存上略好。

**长上下文 (128K+)** → vLLM + 分块预填充；SGLang + 分层 KV。

### TGI 维护陷阱

Hugging Face TGI 于 2025 年 12 月 11 日进入维护模式 — 只进行错误修复。历史上：顶级的可观测性、最佳的 HF 生态系统集成（模型卡、安全工具），原始吞吐量略落后于 vLLM。

对 2026 年的新项目：默认不选 TGI。现有的 TGI 部署可以继续但应最终迁移。SGLang 和 vLLM 是更安全的默认选择。

### 流水线模式

开发（Ollama）→ 预发（llama.cpp）→ 生产（vLLM）。全程使用相同的 GGUF 或 HF 权重。工程师在笔记本上快速迭代；预发镜像生产量化；生产是推理目标。

### Ollama 警告

Ollama 对开发很棒。对共享生产不棒：Go HTTP 序列化增加开销，并发管理比 vLLM 简单，OpenTelemetry 支持滞后。在 Ollama 闪光的地方使用它 — 单用户、一键 — 并对共享切换到 vLLM。

### 自托管 vs 托管是独立的决策

Phase 17 · 01（托管超大规模云商）、· 02（推理平台）涵盖托管。本课假设你已经决定自托管。自托管的理由：数据驻留、自定义微调、规模化总拥有成本、领域模型在托管上不可用。

### 你应该记住的数字

- TGI 维护模式：2025 年 12 月 11 日。
- vLLM v0.15.1：2026 年 2 月；PyTorch 2.10；Blackwell SM120 支持。
- SGLang 生产足迹：400,000+ GPU。
- Ollama 吞吐量差距 vs llama.cpp：慢 15-30%；生产负载下 3 倍。

```figure
data-parallel
```

## 使用它

`code/main.py` 是一个决策树遍历器：给定硬件 + 规模 + 工作负载，选择引擎并解释原因。

## 交付它

本课产出 `outputs/skill-engine-picker.md`。给定约束，选择引擎并编写迁移计划。

## 练习

1. 用你的硬件/规模/工作负载运行 `code/main.py`。输出与你的直觉匹配吗？
2. 你的基础设施是 12 个 H100 和 8 个 MI300X AMD。什么引擎？为什么 TRT-LLM 排除在外？
3. 一个团队想在 2026 年使用 TGI，因为"这是我们所知道的。"论证迁移案例。
4. Ollama 开发到 vLLM 生产：量化、配置和可观测性上有什么变化？
5. RAG 产品具有 P99 前缀长度 8K 和租户间高重用。选择一个引擎并将其与 Phase 17 · 11 + 18 叠加。

## 核心术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| llama.cpp | "CPU 那个" | 最广泛模型支持，CPU 上最快 |
| Ollama | "笔记本那个" | 一键安装，开发级吞吐量 |
| TGI | "HF 的推理服务" | 自 2025 年 12 月起维护模式 |
| vLLM | "默认" | 2026 年广泛的生产基线 |
| SGLang | "代理那个" | 前缀密集，RadixAttention |
| TRT-LLM | "NVIDIA 锁定" | Blackwell 吞吐量领先者，仅 NVIDIA |
| GGUF | "llama.cpp 格式" | 打包的 K-quant 变体 |
| Production-stack | "vLLM K8s" | Phase 17 · 18 参考部署 |
| 流水线模式 | "开发→预发→生产" | Ollama → llama.cpp → vLLM 相同权重 |

## 进一步阅读

- [AI Made Tools — vLLM vs Ollama vs llama.cpp vs TGI 2026](https://www.aimadetools.com/blog/vllm-vs-ollama-vs-llamacpp-vs-tgi/)
- [Morph — llama.cpp vs Ollama 2026](https://www.morphllm.com/comparisons/llama-cpp-vs-ollama)
- [n1n.ai — 综合 LLM 推理引擎比较](https://explore.n1n.ai/blog/llm-inference-engine-comparison-vllm-tgi-tensorrt-sglang-2026-03-13)
- [PremAI — 10 个最佳 vLLM 替代方案 2026](https://blog.premai.io/10-best-vllm-alternatives-for-llm-inference-in-production-2026/)
- [TGI 维护公告](https://github.com/huggingface/text-generation-inference) — 发布说明。
- [vLLM v0.15.1 发布说明](https://github.com/vllm-project/vllm/releases)
