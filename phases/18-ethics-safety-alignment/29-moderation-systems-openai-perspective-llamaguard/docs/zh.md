# 审核系统 — OpenAI、Perspective、Llama Guard

> 生产审核系统操作化第 12-16 课中定义的安全策略。OpenAI Moderation API：`omni-moderation-latest`（2024 年）基于 GPT-4o，在单一调用中分类文本+图像；在多语言测试集上比先前版本好 42%；响应模式返回 13 个类别布尔值 — 骚扰、骚扰/威胁、仇恨、仇恨/威胁、非法、非法/暴力、自残、自残/意图、自残/指令、性、性/未成年人、暴力、暴力/图像；对大多数开发者免费。分层模式：输入审核（生前）、输出审核（生后）、自定义审核（领域规则）。异步并行调用隐藏延迟；标记时使用占位符响应。Llama Guard 3/4（第 16 课）：14 个 MLCommons 危害，代码解释器滥用，8 种语言（v3），多图像（v4）。Perspective API（Google Jigsaw）：先于 LLM 作为审核者的毒性评分；主要是带有严重毒性/侮辱/亵渎变体的单维度毒性；内容审核研究的基线。废弃：Azure Content Moderator 于 2024 年 2 月废弃，2027 年 2 月退役，由 Azure AI Content Safety 替代。

**Type:** Build
**Languages:** Python (stdlib, three-layer moderation harness)
**Prerequisites:** Phase 18 · 16 (Llama Guard / Garak / PyRIT)
**Time:** ~60 minutes

## 学习目标

- 描述 OpenAI Moderation API 的类别分类法以及它与 Llama Guard 3 的 MLCommons 集有何不同。
- 描述三层审核模式（输入、输出、自定义）并说出每种的一个失败模式。
- 描述 Perspective API 作为前 LLM 时代基线的定位以及为什么它在研究中仍被使用。
- 陈述 Azure 废弃时间线。

## 问题

第 12-16 课描述攻击和防御工具。第 29 课涵盖在用户接触产品的表面上操作化防御的已部署审核系统。三层模式是 2026 年的默认配置。

## 概念

### OpenAI Moderation API

`omni-moderation-latest`（2024 年）。基于 GPT-4o。在单一调用中分类文本+图像。对大多数开发者免费。

类别（响应模式中的 13 个布尔值）：
- harassment、harassment/threatening
- hate、hate/threatening
- self-harm、self-harm/intent、self-harm/instructions
- sexual、sexual/minors
- violence、violence/graphic
- illicit、illicit/violent

多模态支持适用于 `violence`、`self-harm` 和 `sexual` 但不适用于 `sexual/minors`；其余仅文本。

### Llama Guard 3/4

在第 16 课中涵盖。14 个 MLCommons 危害类别（与 OpenAI 的 13 个布尔值组织不同）。支持 8 种语言（v3）。Llama Guard 4（2025 年 4 月）是原生多模态，12B。

OpenAI 和 Llama Guard 的分类法重叠但分化。OpenAI 将"illicit"作为广泛类别；Llama Guard 将"violent crimes"和"non-violent crimes"分开。部署基于策略分类法匹配进行选择。

### Perspective API（Google Jigsaw）

毒性评分系统，先于 LLM 作为审核者（前 2020 年）。类别：TOXICITY、SEVERE_TOXICITY、INSULT、PROFANITY、THREAT、IDENTITY_ATTACK。带有子维度变体的单维度主要分数（TOXICITY）。基线内容审核研究工具。

### 三层审核

1. **输入审核。** 在提示到达 LLM 之前。失败模式：假阳性阻止合法查询（"帮我写关于自杀预防的博客"）。
2. **输出审核。** 在 LLM 响应后。失败模式：攻击者在审核前捕获输出（流式协议意味着 token 逐 token 到达用户）。
3. **自定义审核。** 领域特定规则（代码审核、医学保障）。失败模式：跨领域覆盖缺失。

### Azure 废弃

Azure Content Moderator 于 2024 年 2 月废弃，2027 年 2 月退役。由 Azure AI Content Safety 替代，涵盖文本+图像+多语言。

## 使用它

`code/main.py` 实现三层审核工具：输入过滤器（关键词黑名单）、输出过滤器（OpenAI 类别模拟）和表明两层控制都在生产日志中的分层日志器。

## 交付它

本课产出 `outputs/skill-moderation-config.md`。给定部署上下文和内容策略，选择类别分类法、审核层和延迟/安全权衡。

## 练习

1. 运行 `code/main.py`。发送一个多模态测试案例（文本+图像标记）。输入和输出过滤器如何交互？

2. 比较 OpenAI 13 类别与 Llama Guard 14 MLCommons 类别。在什么策略下哪一者更好？

3. 流式输出意味着 token 在审核前到达用户。设计一个满足 < 50ms 延迟预算且不丢失安全控制的流式协议。

4. Perspective API 评分是单维度的（TOXICITY）。多类别系统（OpenAI、Llama Guard）的优势和局限性是什么？

5. 三层审核：在每一层，攻击者假阳性（拒绝合法）和假阴性（通过有害）之间的权衡。哪个最严重？

## 核心术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Omni-moderation | "OpenAI 审核" | 文本+图像分类器，13 个类别，免费 |
| Llama Guard | "Meta 审核" | 14 个 MLCommons 危害，8 种语言，多模态（v4） |
| Perspective | "Google 毒性" | 前 LLM 毒性评分基线 |
| 输入审核 | "生前" | 在 LLM 调用前过滤提示 |
| 输出审核 | "生后" | 在 LLM 响应后过滤 |
| 自定义审核 | "领域规则" | 特定领域的安全规则 |

## 进一步阅读

- [OpenAI Moderation API](https://platform.openai.com/docs/guides/moderation) — 13 个类别
- [Llama Guard (Meta)](https://ai.meta.com/research/publications/llama-guard/) — MLCommons 分类法
- [Perspective API (Google Jigsaw)](https://perspectiveapi.com/) — 毒性评分
- [Azure AI Content Safety](https://azure.microsoft.com/en-us/products/ai-services/ai-content-safety)
