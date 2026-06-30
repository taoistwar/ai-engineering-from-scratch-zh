# 推理平台经济学 — Fireworks、Together、Baseten、Modal、Replicate、Anyscale

> 2026 年的推理市场不再是 GPU 时间租赁。它分叉为定制芯片（Groq、Cerebras、SambaNova）、GPU 平台（Baseten、Together、Fireworks、Modal）和 API 优先市场（Replicate、DeepInfra）。Fireworks 于 2026 年 5 月 1 日将 GPU 租赁价格上调 $1/小时，40 亿美元估值和每天 10 万亿+ token 的处理量告诉你以量为驱动的模式是奏效的。Baseten 于 2026 年 1 月以 50 亿美元估值完成 3 亿美元的 E 轮融资。竞争定位规则很简单：Fireworks 优化延迟，Together 优化目录广度，Baseten 优化企业细节，Modal 优化 Python 原生开发体验，Replicate 优化多模态覆盖范围，Anyscale 优化分布式 Python。本课给你一个可以直接交给创始人的矩阵。

**Type:** Learn
**Languages:** Python (stdlib, toy per-call economics comparator)
**Prerequisites:** Phase 17 · 01 (Managed LLM Platforms), Phase 17 · 04 (vLLM Serving Internals)
**Time:** ~60 minutes

## 学习目标

- 说出三个市场细分（定制芯片、GPU 平台、API 优先），并将每个供应商映射到一个细分。
- 解释为什么"按 token"的 API 定价模式向推理引擎的成本曲线收敛，而不是硬件成本曲线。
- 在至少三家供应商之间计算每请求的有效成本，并解释何时按分钟计费（Baseten、Modal）优于按 token 计费。
- 确定哪个平台是给定工作负载的合适默认选择（无服务器突发型、稳定高吞吐量型、微调变体型、多模态型）。

## 问题

你已经评估了托管超大规模云商平台。你决定需要一个更窄、更快的供应商 — Fireworks 追求延迟，Together 追求广度，Baseten 追求微调自定义模型。现在你有六个真正的选择，而定价页面并不对齐。Fireworks 显示 $/M token；Baseten 显示 $/分钟；Modal 显示 $/秒；Replicate 显示 $/次预测。如果不建模工作负载，无法逐对比对。

更糟糕的是，每个定价页面背后的商业模式不同。Fireworks 在共享 GPU 上运行自己的定制引擎（FireAttention）；按 token 的费率反映他们的利用率曲线。Baseten 给你 Truss + 专用 GPU；按分钟计费反映独占性。Modal 是真正的 Python 无服务器 — 按秒计费，亚秒级冷启动。同样的输出（LLM 响应），三种不同的成本函数。

本课建模这六家并告诉你每种何时胜出。

## 概念

### 三个细分市场

**定制芯片** — Groq (LPU)、Cerebras (WSE)、SambaNova (RDU)。在同一模型上通常比基于 GPU 的集群快 5-10 倍的解码速度。每个 token 价格更高（Groq 在 2025 年末 Llama-70B 上约为 $0.99/M），但对延迟敏感的用例无可匹敌。Groq 是语音代理和实时翻译的生产级选择。

**GPU 平台** — Baseten、Together、Fireworks、Modal、Anyscale。运行在 NVIDIA（2026 年的 H100、H200、B200）或有时 AMD 上。是"原始 GPU 租赁"（RunPod、Lambda）和"超大规模云商托管服务"（Bedrock）之间的经济学层。

**API 优先市场** — Replicate、DeepInfra、OpenRouter、Fal。广泛目录，按次预测或按秒付费，强调首次调用时间。

### Fireworks — 延迟优化的 GPU 平台

- FireAttention 引擎（定制）；宣传在等效配置下延迟比 vLLM 低 4 倍。
- 批处理层级约为无服务器费率的 50%，用于非交互式工作负载。
- 微调模型以与基础模型相同的费率提供服务 — 与对 LoRA 收取溢价的供应商相比，这是真正的差异化优势。
- 2026 年中：于 2026 年 5 月 1 日将按需 GPU 租赁价格上调 $1/小时。规模化时可协商批量定价。
- 财务信号：40 亿美元估值，每天处理 10 万亿+ token。

### Together — 广度优化

- 200+ 模型，包括在上游发布后数日内推出的开源版本。
- 在等效 LLM 模型上比 Replicate 便宜 50-70% — "AI 原生云"的定位是量和目录。
- 推理 + 微调 + 训练，一个 API。

### Baseten — 企业细节优化

- Truss 框架：模型打包，在一个清单中包含依赖、密钥、服务配置。
- GPU 范围从 T4 到 B200。按分钟计费，冷启动缓解方案合理。
- SOC 2 Type II、HIPAA 就绪。金融科技和医疗的常见选择。
- 50 亿美元估值，2026 年 1 月 E 轮（3 亿美元，来自 CapitalG、IVP、NVIDIA）。

### Modal — Python 原生优化

- 纯 Python 基础设施即代码。用 `@modal.function(gpu="A100")` 装饰一个函数，一条命令部署。
- 按秒计费。预热后冷启动 2-4 秒；小模型 <1 秒。
- B 轮 8700 万美元，11 亿美元估值（2025 年）。独立调查中开发体验评分最强。

### Replicate — 多模态广度

- 按次预测付费。图像、视频和音频模型的默认平台。
- 集成生态（Zapier、Vercel、CMS 插件）。
- LLM 按 token 费率竞争力较弱，但在多模态多样性上胜出。

### Anyscale — Ray 原生

- 构建在 Ray 之上；RayTurbo 是 Anyscale 的专有推理引擎（与 vLLM 竞争）。
- 最适合推理步骤是更大图中一个节点的分布式 Python 工作负载。
- 托管 Ray 集群；与 Ray AIR 和 Ray Serve 紧密集成。

### 按 token 计费 vs 按分钟计费 — 各自何时胜出

按 token 计费在工作负载对延迟不敏感且突发时合理 — 你只为你使用的东西付费。按分钟计费在利用率高且可预测时合理 — 一旦你占满 GPU，就比按 token 划算。

粗略规则：对专用 GPU 持续利用率高于约 30% 的工作负载，按分钟计费（Baseten、Modal）开始优于按 token 计费（Fireworks、Together）。低于此值时按 token 计费胜出，因为你避免了为空闲付费。

### 定制引擎才是真正的护城河

每个在 vLLM 和 SGLang 之上的平台都声称拥有定制引擎。FireAttention、RayTurbo、Baseten 的推理栈。定制引擎的声明有营销成分 — 诚实的框架是 vLLM + SGLang 代表约 80% 的生产级开源推理，平台层的差异化在于开发体验、归因和 SLA。

### 你应该记住的数据

- Fireworks GPU 租赁：2026 年 5 月 1 日起上调 $1/小时。
- Fireworks 声明：等效配置下延迟比 vLLM 低 4 倍。
- Together：LLM 上比 Replicate 便宜 50-70%。
- Baseten 估值：50 亿美元（E 轮，2026 年 1 月，3 亿美元融资）。
- Modal 估值：11 亿美元（B 轮，2025 年）。
- 持续利用率约 30% 以上，按分钟计费优于按 token 计费。

```figure
cost-per-token
```

## 使用它

`code/main.py` 在合成工作负载上跨定价模式比较六家供应商。报告 $/天和有效 $/M token。运行它找到按 token 和按分钟之间的盈亏平衡点。

## 交付它

本课产出 `outputs/skill-inference-platform-picker.md`。给定工作负载画像、SLA 和预算，选择主推理平台并指定第二名。

## 练习

1. 运行 `code/main.py`。在什么持续利用率下 Baseten（按分钟）对一台 H100 上的 70B 模型优于 Fireworks（按 token）？自己推导交叉点并与经验法则比较。
2. 你的产品同时提供图像生成、聊天和语音转文本。为每种模态选择平台，并命名统一它们的网关模式。
3. Fireworks 将你的主模型涨价 $1/小时。如果 40% 的流量转移到批处理层级（50% 折扣），建模混合成本影响。
4. 一个受监管客户要求 SOC 2 Type II + HIPAA + 专用 GPU。哪三家平台可行，哪个在 FinOps 上获胜？
5. 比较 Llama 3.1 70B 在 Fireworks 无服务器、Together 按需、Baseten 专用和 Replicate API 上每 1,000 次预测的成本。每天 10 次预测时哪个最便宜？每天 10,000 次呢？

## 核心术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 定制芯片 | "非 GPU 芯片" | Groq LPU、Cerebras WSE、SambaNova RDU — 针对解码优化 |
| FireAttention | "Fireworks 引擎" | 定制注意力内核；宣传延迟比 vLLM 低 4 倍 |
| Truss | "Baseten 的格式" | 模型打包清单；依赖 + 密钥 + 服务配置 |
| 按 token | "API 定价" | 按消耗的 token 收费；不为空闲付费 |
| 按分钟 | "专用定价" | 按墙上时钟 GPU 时间收费；高利用率时划算 |
| 按次预测 | "Replicate 定价" | 按模型调用收费；图像/视频中常见 |
| RayTurbo | "Anyscale 引擎" | 基于 Ray 的专有推理；在 Ray 集群上与 vLLM 竞争 |
| 批处理层级 | "5 折" | 以降低费率排队的非交互式工作负载队列；Fireworks、OpenAI 上常见 |
| 以基础费率微调 | "Fireworks LoRA" | 以基础模型费率对 LoRA 服务的请求收费（差异化优势） |

## 进一步阅读

- [Fireworks 定价](https://fireworks.ai/pricing) — 按 token 费率、批处理层级、GPU 租赁。
- [Baseten 定价](https://www.baseten.co/pricing/) — 按分钟费率、承诺容量、企业层级。
- [Modal 定价](https://modal.com/pricing) — 按秒 GPU 费率和免费层级。
- [Together AI 定价](https://www.together.ai/pricing) — 模型目录和按 token 费率。
- [Anyscale 定价](https://www.anyscale.com/pricing) — RayTurbo 和托管 Ray 定价。
- [Northflank — Fireworks AI 替代方案](https://northflank.com/blog/7-best-fireworks-ai-alternatives-for-inference) — 对比评估。
- [Infrabase — AI 推理 API 供应商 2026](https://infrabase.ai/blog/ai-inference-api-providers-compared) — 供应商格局。
