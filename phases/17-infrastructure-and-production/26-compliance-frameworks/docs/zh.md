# 合规 — SOC 2、HIPAA、GDPR、PCI-DSS、EU AI Act、ISO 42001

> 多框架覆盖是 2026 年企业交易的入门要求。**EU AI Act**：自 2024 年 8 月 1 日生效。大多数高风险要求于 2026 年 8 月 2 日执行。高风险系统义务罚款最高 €1500 万或全球年营业额的 3%（第 99(4) 条）；禁止的 AI 实践罚款最高 €3500 万或 7%（第 99(3) 条）。如果为欧盟用户服务则全球适用。**Colorado AI Act**：2026 年 6 月 30 日生效（由 SB25B-004 从 2026 年 2 月延迟）— 高风险系统的影响评估、对 AI 决定的申诉权。Virginia 对信用/就业/住房/教育类似。**SOC 2 Type II**：事实上的 B2B AI 要求（Type II，不是 Type I，用于金融科技）。**GDPR**：最大的 AI 特定罚款记录是 Clearview AI 的 €3050 万（荷兰 DPA，2024 年 9 月）；意大利 Garante 于 2024 年 12 月对 OpenAI 处以 €1500 万罚款（后在 2026 年 3 月上诉中被推翻）。推理时的实时 PII 脱敏是可辩护的标准；后处理清理不够。**HIPAA**：医疗绑定 — 没有 BAA 不能将 PHI 发送到外部 AI 服务。**PCI-DSS**：AI 交互层覆盖需要配置 + 合同协议，非自动。**ISO 42001**：新兴的 AI 治理标准，随 ISO 27001 增长的采购要求。参考画像：OpenAI 维持 SOC 2 Type 2、ISO/IEC 27001:2022、ISO/IEC 27701:2019、GDPR/CCPA/HIPAA (BAA)/FERPA、ChatGPT 支付组件的 PCI-DSS。跨框架映射减少审计疲劳：访问控制映射到 ISO 27001 A.5.15-5.18、GDPR 第 32 条、HIPAA §164.312(a)。

**Type:** Learn
**Languages:** (Python optional — compliance is policy + process, not code)
**Prerequisites:** Phase 17 · 25 (Security), Phase 17 · 13 (Observability)
**Time:** ~60 minutes

## 学习目标

- 列举与 LLM 产品相关的 2026 年七个框架，并将每个匹配到客户细分。
- 引用 EU AI Act 执行时间线（2024 年 8 月生效；2026 年 8 月高风险执行）和两级罚款上限（高风险义务 €1500 万 / 3%，禁止实践 €3500 万 / 7%）。
- 解释为什么后处理 PII 清理对 GDPR 不够，并指出实时推理层脱敏是可辩护的标准。
- 描述跨框架控制映射（例如，访问控制映射到 ISO 27001 A.5.15-5.18 + GDPR 第 32 条 + HIPAA §164.312(a)）。

## 问题

一个企业客户的采购要求 SOC 2 Type II、GDPR、HIPAA BAA、ISO 27001 和"EU AI Act 合规声明"。你的团队有 SOC 2 Type I。你距离 Type II 还有六个月，且尚未启动 GDPR 第 30 条记录。

多框架覆盖不是 LLM 问题 — 是带 LLM 特定叠加的企业 SaaS 问题。2026 年的采购团队想要一个每行一个框架、每列一个控制的矩阵，而不是 PDF。

## 概念

### 七个框架

| 框架 | 范围 | LLM 特定要求 |
|-----------|-------|--------------------------|
| SOC 2 Type II | B2B SaaS 基线 | 经过 6-12 个月审计的过程控制 |
| HIPAA | 美国医疗 | 需要 BAA；没有签署协议 PHI 不能离开基础设施 |
| GDPR | 欧盟用户 | 实时 PII 脱敏；数据主体权利；第 30 条记录 |
| PCI-DSS | 支付数据 | 触及支付的 AI 配置 + 合同 |
| EU AI Act | 服务欧盟用户 | 风险层级分类；高风险系统：合规评估、文档、日志 |
| Colorado AI Act | 服务 CO 居民 | 影响评估；申诉权 |
| ISO 42001 | AI 治理 | 新兴；与 ISO 27001 配对 |

### EU AI Act 时间线

- 2024 年 8 月 1 日：生效。
- 2025 年 2 月 2 日：禁止 AI 实践执行。
- 2026 年 8 月 2 日：高风险系统执行（合规评估、文档、日志）。
- 2027 年 8 月：协调立法下的产品中高风险系统。

风险层级：不可接受（禁止）、高风险（合规 + 日志）、有限风险（透明度）、最低风险（无约束）。大多数 B2B LLM SaaS 是有限风险；高风险在就业、信用、教育、执法、移民、基本服务中触发。

罚款（第 99 条）：违反高风险系统义务最高 €1500 万或全球年营业额的 3%（第 99(4) 条）；禁止 AI 实践最高 €3500 万或 7%（第 99(3) 条）；以较高者为准。

### GDPR — 实时脱敏是标准

后处理清理（LLM 看到 PII 后再脱敏）不是可辩护的姿态 — 模型已经看到了数据。实时推理层脱敏是 2026 年的标准：

- LLM 调用前的实体识别。
- 一致分词（Mesh 方法）保留语义。
- 仅存储脱敏后的提示 + 已同意的原始数据。

最近的执法：Clearview AI 的 €3050 万（荷兰 DPA，2024 年 9 月）是迄今为止最大的 AI 特定 GDPR 罚款记录；OpenAI 的 €1500 万（意大利 Garante，2024 年 12 月）是最大的 LLM 特定罚款，尽管在 2026 年 3 月上诉中被推翻，裁决仍在进一步审查中。后处理声明已在审计中失败。

### HIPAA — BAA 不是可选的

没有签署的业务伙伴协议，你不能将 PHI 发送到外部 AI 服务。三家超大规模云商 LLM 平台（Bedrock、Azure OpenAI、Vertex）都提供 BAA。OpenAI 直接 API 提供 BAA。Anthropic 直接 API 提供 BAA。在发送 PHI 之前确认。

### SOC 2 Type II

Type I：控制已设计并记录。
Type II：控制在 6-12 个月内有效运行。

2026 年的 B2B 采购默认 Type II。Type I 是起点；Type II 是门槛。

常见审计驱动因素：访问日志（谁看了什么）、变更管理（如何部署的）、风险评估（每季度）、事件响应（测试了吗？）。Phase 17 · 25 的审计日志可直接重用。

### 跨框架映射

一个访问控制策略满足多个框架控制：

| 控制 | 框架 |
|---------|-----------|
| 访问日志 | ISO 27001 A.5.15-5.18、GDPR 第 32 条、HIPAA §164.312(a) |
| 变更管理 | ISO 27001 A.8.32、PCI DSS 要求 6、HIPAA 泄露通知范围 |
| 传输加密 | ISO 27001 A.8.24、GDPR 第 32 条、HIPAA §164.312(e) |
| 机密管理 | ISO 27001 A.8.19、PCI DSS 要求 8、SOC 2 CC6.1 |

合规工具（Drata、Vanta、Secureframe）自动进行此映射。规模化时值得成本。

### ISO 42001 — 新兴

2023 年末发布。随 ISO 27001 增长的采购要求。AI 治理框架，包括风险管理、数据质量、透明度、人工监督。

### OpenAI 的参考画像

OpenAI 维持 SOC 2 Type 2、ISO/IEC 27001:2022、ISO/IEC 27701:2019、GDPR/CCPA/HIPAA (BAA)/FERPA、ChatGPT 支付组件的 PCI-DSS。这大致是 2026 年的企业入门要求。

### 你应该记住的数字

- EU AI Act 罚款：最高 €1500 万 / 3%（高风险义务，第 99(4) 条）；最高 €3500 万 / 7%（禁止实践，第 99(3) 条）。
- EU AI Act 高风险执行：2026 年 8 月 2 日。
- 最大 AI 特定 GDPR 罚款记录：€3050 万，Clearview AI（荷兰 DPA，2024 年 9 月）。
- 最大 LLM 特定 GDPR 罚款：€1500 万，OpenAI（意大利 Garante，2024 年 12 月；2026 年 3 月上诉被推翻）。
- SOC 2 Type II 窗口：6-12 个月运行的控制。
- Colorado AI Act 生效日期：2026 年 6 月 30 日（由 SB25B-004 从 2026 年 2 月延迟）。

## 使用它

`code/main.py` 是 Python 中的合规映射电子表格 — 给定一个控制，列出其满足的框架。

## 交付它

本课产出 `outputs/skill-compliance-matrix.md`。给定客户细分和地理位置，指定所需框架和控制。

## 练习

1. 你的第一个企业客户要求 SOC 2 Type II、HIPAA BAA、EU AI Act 声明。赢得交易的最低可行合规姿态是什么？
2. 根据 EU AI Act 风险层级分类三个假设的 LLM 产品。高风险时有什么变化？
3. 你没有 BAA 意外将 PHI 发送到供应商。走查事件响应。
4. 论证 ISO 42001 在 2026 年对中型 AI 供应商是否"必要"。
5. 将你的 LLM 审计日志字段（Phase 17 · 25）映射到至少三个框架控制。

## 核心术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| SOC 2 Type II | "审计控制" | 控制运行 6-12 个月，独立证明 |
| HIPAA BAA | "医疗合同" | 业务伙伴协议；PHI 必需 |
| GDPR | "欧盟隐私" | 实时 PII 脱敏是 2026 年可辩护的标准 |
| EU AI Act | "欧盟 AI 规则" | 2026 年 8 月高风险执行；€1500 万 / 3%（高风险义务）— €3500 万 / 7%（禁止实践） |
| Colorado AI Act | "美国 AI 州法" | 2026 年 6 月 30 日生效（SB25B-004 延迟）；影响评估 |
| ISO 42001 | "AI 治理" | AI 风险 + 透明度的新兴框架 |
| ISO 27001 | "安全 ISMS" | 信息安全管理系统基线 |
| Conformity assessment | "EU AI 文档包" | 高风险要求：文档、测试、日志 |
| Cross-framework mapping | "一个控制，多个框架" | 单一策略满足多个框架控制 |

## 进一步阅读

- [OpenAI 安全和隐私](https://openai.com/security-and-privacy/) — 参考合规画像。
- [GuardionAI — LLM 合规 2026: ISO 42001、EU AI Act、SOC 2、GDPR](https://guardion.ai/blog/llm-compliance-guide-iso-42001-eu-ai-act-soc2-gdpr-2026)
- [Dsalta — SOC 2 Type 2 审计指南 2026: 10 项 AI 控制](https://www.dsalta.com/resources/ai-compliance/soc-2-type-2-audit-guide-2026-10-ai-powered-controls-every-saas-team-needs)
- [EU AI Act 官方文本](https://eur-lex.europa.eu/eli/reg/2024/1689/oj) — 主要来源。
- [Colorado AI Act](https://leg.colorado.gov/bills/sb24-205) — 主要来源。
- [ISO/IEC 42001:2023](https://www.iso.org/standard/81230.html) — AI 管理系统标准。
