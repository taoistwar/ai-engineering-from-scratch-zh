# 托管 LLM 平台 — Bedrock、Vertex AI、Azure OpenAI

> 三家超大规模云服务商，三种截然不同的策略。AWS Bedrock 是一个模型市场 — Claude、Llama、Titan、Stability、Cohere 在一个 API 背后运行。Azure OpenAI 是与 OpenAI 的独家合作，外加用于专用容量的预置吞吐量单元 (PTU)。Vertex AI 以 Gemini 优先，拥有最佳的長上下文和多模态方案。2026 年，Artificial Analysis 在 Llama 3.1 405B 同类模型上测得 Azure OpenAI 的中位延迟约为 50 ms，Bedrock 约为 75 ms — PTU 解释了这一差距，因为专用容量优于共享的按需容量。决策规则不是"哪个最快"，而是"哪个模型目录和 FinOps 层面对我的产品最匹配"。本课教你带着书面权衡进行选择，而不是凭感觉。

**Type:** Learn
**Languages:** Python (stdlib, toy cost-and-latency comparator)
**Prerequisites:** Phase 11 (LLM Engineering), Phase 13 (Tools & Protocols)
**Time:** ~60 minutes

## 学习目标

- 说出三种平台策略（市场模式 vs 独家模式 vs Gemini 优先模式），并将每种策略匹配到一个产品用例。
- 解释预置吞吐量单元 (PTU) 在 Azure OpenAI 中为你带来什么，以及为什么 Bedrock 按需服务在 405B 规模下通常慢约 25 ms。
- 绘制每个平台的 FinOps 归因面（Bedrock Application Inference Profiles vs Vertex 按项目归队 vs Azure 作用域 + PTU 预留）。
- 制定"最低双供应商"策略，并解释为什么在 2026 年单供应商锁定是代价高昂的错误。

## 问题

你为产品选择了 Claude 3.7 Sonnet。现在你需要部署它。你可以直接调用 Anthropic API，也可以通过 AWS Bedrock 调用，或者通过网关调用。直接 API 最简单；Bedrock 附加了 BAA、VPC 端点、IAM 和 CloudWatch 归因。网关提供故障转移、统一计费和跨供应商的速率限制。

更深层次的问题是目录。如果你需要同一产品中同时使用 Claude、Llama 和 Gemini，你不能从单一位置买到全部 —— 除非那个位置同时是 Bedrock、Vertex 和 Azure OpenAI。超大规模云服务商不是可互换的 —— 他们各自在谁拥有模型层上下了不同的赌注。

本课梳理了三家赌注、延迟差距、FinOps 差距和锁定风险。

## 概念

### 三种策略

**AWS Bedrock** — 市场模式。Claude (Anthropic)、Llama (Meta)、Titan (AWS 自研)、Stability (图像)、Cohere (嵌入)、Mistral，外加图像和嵌入子目录。一个 API，一个 IAM 面，一个 CloudWatch 导出。Bedrock 的赌注是：客户想要选择权多于想要单一模型。

**Azure OpenAI** — 独家合作。你获得 GPT-4 / 4o / 5 / o 系列、DALL·E、Whisper，以及在 Azure 数据中心内对 OpenAI 模型进行微调。"Azure OpenAI 服务"目录中没有非 OpenAI 模型 —— 这些模型归入 Azure AI Foundry（独立产品）。Azure 的赌注是：OpenAI 仍然是前沿，客户希望在这一特定关系上获得企业级控制。

**Vertex AI** — Gemini 优先，其它其次。Gemini 1.5 / 2.0 / 2.5 Flash 和 Pro，外加 Model Garden（第三方模型）。Vertex 的赌注是多模态長上下文 —— 100 万 token 的 Gemini 上下文是差异化优势。

### 规模化时的延迟差距

Artificial Analysis 运行持续基准测试。在等效的 Llama 3.1 405B 部署（共享按需）上，Azure OpenAI 的首 token 中位延迟约为 50 ms；Bedrock 约为 75 ms。这一差距不是 AWS 的失败 —— 而是容量模型的不同。Azure 销售 PTU（预置吞吐量单元），为你的租户预留 GPU 容量。Bedrock 的等效服务（预置吞吐量）存在但起价约 $21/小时每单元，大多数客户留在共享按需上。

按需共享容量与其他所有客户的流量竞争。专用容量则不会。如果你的产品 SLA 要求 P99 TTFT < 100 ms，你要么在 Azure 购买 PTU，购买 Bedrock 预置吞吐量，要么接受默认的方差。

### 预置吞吐量经济学

Azure PTU：一个预留的推理计算块。相比可预测工作负载的按需服务，最高节省约 70%。每小时固定成本，无论流量如何 —— 即使空闲也要为预留付费。盈亏平衡点通常在 40-60% 的持续利用率左右。

Bedrock 预置吞吐量：$21-$50/小时，取决于模型和区域。类似的数学 —— 盈亏平衡点约在峰值利用率的一半。需要月度承诺。

Vertex 预置容量按 Gemini SKU 销售；定价因模型和区域而异，公开披露较少。

### FinOps 面 —— 真正的差异化因素

**Bedrock Application Inference Profiles** 是市场中最干净的归因。为 profile 打上 `team`、`product`、`feature` 标签；将所有模型调用路由通过它；CloudWatch 无需后处理即可按 profile 分解成本。2025 年新增，仍是粒度最细的超大规模原生方案。

**Vertex** 归因是按项目归队加上全标签。你将每个团队建模为一个 GCP 项目，在每个资源上打标签，并使用 BigQuery Billing Export + DataStudio 进行汇总。工作量更大，但 BigQuery 允许你对成本数据进行任意 SQL 查询。

**Azure** 依赖订阅/资源组作用域加上标签，PTU 预留作为一等成本对象。标签从资源组继承，而不是从请求继承，因此按请求归因需要 Application Insights 自定义指标或一个能标记请求头的网关。

模式：Bedrock 原生最干净，Vertex 通过 BigQuery 最灵活，Azure 最不透明，除非你进行仪器化。

### 锁定是 2026 年的风险

当单一模型主导时，单超大规模云商承诺是可以的。在 2026 年，前沿每个月都在变动 —— Claude 3.7 一个季度，Gemini 2.5 下个季度，GPT-5 再下个季度。锁定在一个平台上就等于放弃了三分之二的前沿。

有效团队采用的模式：对任何产品关键的 LLM 调用保持最低双供应商。Bedrock 加上 Azure OpenAI 是常见的组合 —— 一个来源 Claude，另一个来源 GPT，在两者之间故障转移，共用同一网关。成本增加微乎其微，因为网关会路由最优路径；在中断期间（如 Azure OpenAI 2025 年 1 月事件、AWS us-east-1 中断）的可用性提升是决定性的。

### 数据驻留、BAA 和受监管行业

Bedrock：大多数区域提供 BAA；VPC 端点；护栏。常见的金融科技默认选择。
Azure OpenAI：HIPAA、SOC 2、ISO 27001；欧盟数据驻留；企业受监管默认选择。
Vertex：HIPAA、GDPR、按区域数据驻留；Google Cloud 的合规体系。

三家都满足基本要求。差异在于数据保留策略、日志处理方式以及滥用监控是否读取你的流量（大多数默认启用；企业可申请退出）。

### 你应该记住的数据

- Azure OpenAI Llama 3.1 405B 同类模型 TTFT 中位数：~50 ms（使用 PTU）。
- Bedrock 按需 TTFT 中位数：~75 ms。
- Bedrock 预置吞吐量：$21-$50/小时每单元。
- Azure PTU 盈亏平衡：~40-60% 持续利用率。
- 高利用率下 PTU 相比按需的节省：最高 70%。

## 使用它

`code/main.py` 在一个合成工作负载上比较三家平台 —— 它建模按需 vs PTU 经济学、TTFT 方差和成本归因精度。运行它可以看到 PTU 在何处划算，以及市场的模型广度何时超过了 TTFT 差距。

## 交付它

本课产出 `outputs/skill-managed-platform-picker.md`。给定一个工作负载画像（所需模型、TTFT SLA、日均量、合规要求），它推荐一个主平台、一个备用平台和一个 FinOps 仪器化计划。

## 练习

1. 运行 `code/main.py`。在什么持续利用率下 Azure PTU 对 70B 级别模型优于按需？计算盈亏平衡点并与广告的 40-60% 区间比较。
2. 你的产品需要 Claude 3.7 Sonnet 和 GPT-4o。设计一个双供应商部署 —— 哪个模型归哪个超大规模云商，前面放什么网关，故障转移策略是什么？
3. 一个受监管的医疗客户要求 BAA、美国东部数据驻留和 P99 TTFT 低于 100ms。选择平台并用三个具体特性来论证。
4. 你发现这个月 Bedrock 账单上涨了 4 倍而流量没变。没有 Application Inference Profiles，你如何找出原因？有了 profiles，需要多长时间？
5. 阅读 Azure OpenAI 和 Bedrock 定价页面。对于每月 1 亿 token 的 Claude 工作负载，哪个更便宜 —— 直接 Anthropic API、Bedrock 按需还是 Bedrock 预置吞吐量？

## 核心术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Bedrock | "AWS LLM 服务" | 涵盖 Claude、Llama、Titan、Mistral、Cohere 的模型市场 |
| Azure OpenAI | "Azure 的 ChatGPT" | Azure 数据中心中的独家 OpenAI 模型，附带企业级控制 |
| Vertex AI | "Google 的 LLM" | 以 Gemini 优先的平台，Model Garden 用于第三方模型 |
| PTU | "专用容量" | 预置吞吐量单元 — 预留推理 GPU，按小时定价 |
| Application Inference Profile | "Bedrock 标记" | 每个产品的成本/使用量 profile，附带标签，CloudWatch 原生 |
| Model Garden | "Vertex 目录" | Vertex AI 的第三方模型区，独立于 Gemini |
| 最低双供应商 | "LLM 冗余" | 在至少 2 家超大规模云商上运行每个关键 LLM 路径的策略 |
| BAA | "HIPAA 文件" | 业务伙伴协议；处理 PHI 时必须；三家都提供 |
| 滥用监控 | "日志监控器" | 供应商侧对提示/输出的安全检查；企业可申请退出 |

## 进一步阅读

- [AWS Bedrock 定价](https://aws.amazon.com/bedrock/pricing/) — 权威费率卡和预置吞吐量定价。
- [Azure OpenAI 服务定价](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/) — PTU 经济学和费率卡。
- [Vertex AI 生成式 AI 定价](https://cloud.google.com/vertex-ai/generative-ai/pricing) — Gemini 层级和 Model Garden 附加费。
- [Artificial Analysis LLM 排行榜](https://artificialanalysis.ai/) — 跨供应商的持续延迟和吞吐量基准测试。
- [The AI Journal — AWS Bedrock vs Azure OpenAI CTO 指南 2026](https://theaijournal.co/2026/03/aws-bedrock-vs-azure-openai/) — 企业决策框架。
- [Finout — Bedrock vs Vertex vs Azure FinOps](https://www.finout.io/blog/bedrock-vs.-vertex-vs.-azure-cognitive-a-finops-comparison-for-ai-spend) — 归因机制并排对比。
