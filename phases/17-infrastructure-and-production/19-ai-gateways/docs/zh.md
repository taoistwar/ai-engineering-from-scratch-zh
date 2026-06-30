# AI 网关 — LiteLLM、Portkey、Kong AI Gateway、Bifrost

> 网关位于你的应用和模型供应商之间。核心功能是供应商路由、降级、重试、速率限制、秘密引用、可观测性、护栏。2026 年市场分裂：**LiteLLM** 是 MIT 开源，100+ 供应商，兼容 OpenAI，但在约 2000 RPS 时崩溃（8 GB 内存，发布基准中的级联失败）；最适合 Python、<500 RPS、开发/原型。**Portkey** 是控制平面定位（护栏、PII 脱敏、越狱检测、审计追踪），2026 年 3 月以 Apache 2.0 开源，20-40 毫秒延迟开销，$49/月生产层。**Kong AI Gateway** 基于 Kong Gateway 构建 — Kong 自身在相同 12 CPU 上的基准测试：比 Portkey 快 228%，比 LiteLLM 快 859%；$100/模型/月定价（Plus 层最多 5 个）；如果你已在 Kong 上是企业合适的选择。**Bifrost**（Maxim AI）— 自动重试与可配置退避，OpenAI 429 时降级到 Anthropic。**Cloudflare / Vercel AI Gateways** — 托管、零运维、基本重试。数据驻留驱动自托管决策；Portkey 和 Kong 以开源 + 可选托管处于中间。

**Type:** Learn
**Languages:** Python (stdlib, toy gateway-routing simulator)
**Prerequisites:** Phase 17 · 01 (Managed LLM Platforms), Phase 17 · 16 (Model Routing)
**Time:** ~60 minutes

## 学习目标

- 列举六项核心网关功能（路由、降级、重试、速率限制、秘密、可观测性、护栏）。
- 将四种 2026 年网关（LiteLLM、Portkey、Kong AI、Bifrost）映射到规模上限和用例。
- 引用 Kong 基准测试（228% vs Portkey，859% vs LiteLLM）并解释为什么对 >500 RPS 重要。
- 给定数据驻留和运维预算，选择自托管 vs 托管。

## 问题

你的产品调用 OpenAI、Anthropic 和自托管 Llama。每个供应商有不同的 SDK、错误模型、速率限制和认证方案。你想要故障转移（如果 OpenAI 429，尝试 Anthropic）、单一凭据存储、统一可观测性和每租户速率限制。

在应用层重新发明这些会将每个服务耦合到每个供应商。网关层将其合并到一个进程中，用一个 API（通常兼容 OpenAI）扇出到供应商。

## 概念

### 六项核心功能

1. **供应商路由** — OpenAI、Anthropic、Gemini、自托管等在一个 API 后面。
2. **降级** — 在 429、5xx 或质量失败时，在其他地方重试。
3. **重试** — 指数退避，有限尝试。
4. **速率限制** — 每租户、每密钥、每模型。
5. **秘密引用** — 在运行时从保管库拉取凭据（绝不在应用中）。
6. **可观测性** — OTel + GenAI 属性（Phase 17 · 13）+ 成本归因。
7. **护栏** — PII 脱敏、越狱检测、允许主题过滤。

### LiteLLM — MIT 开源，Python

- 100+ 供应商，兼容 OpenAI，路由器配置，降级，基本可观测性。
- 在 Kong 基准测试中约 2000 RPS 时崩溃；8 GB 内存占用，持续负载下级联失败。
- 最适合：Python 应用、<500 RPS、开发/预发网关、实验性路由。
- 成本：开源 $0；存在云端免费层。

### Portkey — 控制平面定位

- 2026 年 3 月起 Apache 2.0 开源。护栏、PII 脱敏、越狱检测、审计追踪。
- 每请求 20-40 毫秒延迟开销。
- $49/月生产层，带保留 + SLA。
- 最适合：需要捆绑护栏 + 可观测性的受监管行业。

### Kong AI Gateway — 规模化方案

- 基于 Kong Gateway 构建（成熟的 API 网关产品，lua+OpenResty）。
- Kong 自身在 12-CPU 等效上的基准测试：比 Portkey 快 228%，比 LiteLLM 快 859%。
- 定价：$100/模型/月，Plus 层最多 5 个。
- 最适合：已在 Kong 上；>1000 RPS；愿意许可。

### Bifrost (Maxim AI)

- 自动重试与可配置退避。
- OpenAI 429 时降级到 Anthropic 是规范的配方。
- 较新进入者；商业。

### Cloudflare AI Gateway / Vercel AI Gateway

- 托管、零运维。基本重试和可观测性。
- 最适合：Cloudflare/Vercel 上的边缘推理 JavaScript 应用。
- 在护栏和速率限制上相比 Kong/Portkey 有限。

### 自托管 vs 托管

数据驻留是强制函数。医疗和金融默认自托管（LiteLLM 或 Portkey OSS 或 Kong）。消费产品默认托管（Cloudflare AI Gateway）或中层（Portkey 托管）。混合：自托管用于受监管租户，托管用于其他。

### 延迟预算

- LiteLLM：典型 5-15 毫秒开销。
- Portkey：20-40 毫秒开销。
- Kong：3-8 毫秒开销。
- Cloudflare/Vercel：1-3 毫秒开销（边缘优势）。

网关延迟直接加到 TTFT 上。对于 TTFT P99 < 100 毫秒 SLA，Kong 或 Cloudflare。对于 P99 < 500 毫秒，任意均可。

### 速率限制语义很重要

简单令牌桶在中等规模下工作。多租户需要滑动窗口 + 突发允许 + 每租户分层。LiteLLM 提供令牌桶；Kong 提供滑动窗口；Portkey 提供分层。

### 网关 + 可观测性 + 路由组合

Phase 17 · 13（可观测性）+ 16（模型路由）+ 19（网关）在生产中是同一层。选择一个覆盖所有三者或仔细连接它们：大多数 2026 年部署将 Helicone（可观测性）或 Portkey（护栏）与 Kong（规模）以分离角色组合。

### 你应该记住的数字

- LiteLLM：约 2000 RPS 崩溃，8 GB 内存。
- Portkey：20-40 毫秒开销；2026 年 3 月起 Apache 2.0。
- Kong：比 Portkey 快 228%，比 LiteLLM 快 859%。
- Kong 定价：$100/模型/月，Plus 层最多 5 个。
- Cloudflare/Vercel：边缘处 1-3 毫秒开销。

## 使用它

`code/main.py` 模拟在 429/5xx 注入下跨 3 个供应商的网关路由与降级。报告延迟、重试率和降级命中率。

## 交付它

本课产出 `outputs/skill-gateway-picker.md`。给定规模、运维姿态、合规、延迟预算，选择网关。

## 练习

1. 运行 `code/main.py`。配置从 OpenAI → Anthropic → 自托管的降级。在 5% 供应商错误率下预期命中率是多少？
2. 你的 SLA 是 TTFT P99 < 200 毫秒，基线 300 毫秒。哪些网关保持在预算内？
3. 一个医疗客户要求自托管 + PII 脱敏 + 审计。选择 Portkey OSS 或 Kong。
4. 比较 LiteLLM vs Kong：在什么 RPS 上限下团队应该迁移？
5. 为多租户 SaaS 设计速率限制策略：免费层、试用层、付费层。令牌桶还是滑动窗口？

## 核心术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 网关 | "API 代理" | 位于应用和供应商之间的进程 |
| LiteLLM | "MIT 那个" | Python 开源，100+ 供应商，2K RPS 崩溃 |
| Portkey | "护栏网关" | 控制平面 + 可观测性，Apache 2.0 |
| Kong AI Gateway | "规模那个" | 基于 Kong Gateway 构建，基准领先 |
| Bifrost | "Maxim 的网关" | 重试 + Anthropic 降级配方 |
| Cloudflare AI Gateway | "边缘托管" | 边缘部署的托管网关，零运维 |
| PII 脱敏 | "数据清洗" | 发送到模型前的正则 + NER 掩码 |
| 越狱检测 | "提示注入护栏" | 用户输入上的分类器 |
| 审计追踪 | "受监管日志" | 每次 LLM 调用的不可变记录 |
| 令牌桶 | "简单速率限制" | 基于补充的速率限制器 |
| 滑动窗口 | "精确速率限制" | 时间窗口速率限制器；更好的公平性 |

## 进一步阅读

- [Kong AI Gateway 基准测试](https://konghq.com/blog/engineering/ai-gateway-benchmark-kong-ai-gateway-portkey-litellm)
- [TrueFoundry — AI 网关 2026 比较](https://www.truefoundry.com/blog/a-definitive-guide-to-ai-gateways-in-2026-competitive-landscape-comparison)
- [Techsy — 顶级 LLM 网关工具 2026](https://techsy.io/en/blog/best-llm-gateway-tools)
- [LiteLLM GitHub](https://github.com/BerriAI/litellm)
- [Portkey GitHub](https://github.com/Portkey-AI/gateway)
- [Kong AI Gateway 文档](https://docs.konghq.com/gateway/latest/ai-gateway/)
