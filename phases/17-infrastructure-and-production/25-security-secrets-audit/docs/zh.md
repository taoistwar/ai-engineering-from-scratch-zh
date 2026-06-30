# 安全 — 机密、API 密钥轮换、审计日志、护栏

> 通过集中化保管库（HashiCorp Vault、AWS Secrets Manager、Azure Key Vault）消除机密蔓延。绝不要将凭据存储在配置文件、VCS 中的 env 文件、电子表格中。使用 IAM 角色而非静态密钥；CI/CD 使用 OIDC。AI 网关模式是 2026 年的解决方案：应用 → 网关 → 模型供应商，网关在运行时从保管库拉取凭据。在保管库中轮换，所有应用在数分钟内获取 — 无需重新部署，无需 Slack "谁有新密钥"消息。轮换策略 ≤ 90 天；对每次提交使用 TruffleHog / GitGuardian / Gitleaks 扫描。零信任：MFA、SSO、RBAC/ABAC、短期令牌、设备姿态。PII 清理使用实体识别在转发前（转发到 LLM 之前）掩码 PHI/PII；一致的分词（Mesh 方法）将敏感值映射到稳定的占位符，使 LLM 保留代码/关系语义。网络出口：LLM 服务在专用 VPC/VNet 子网中，仅白名单 `api.openai.com`、`api.anthropic.com` 等；阻止所有其他出站。2026 年事件驱动因素：Vercel 供应链攻击通过被入侵的 CI/CD 凭据泄露了数千客户部署的环境变量。

**Type:** Learn
**Languages:** Python (stdlib, toy PII-scrubber + audit-log writer)
**Prerequisites:** Phase 17 · 19 (AI Gateways), Phase 17 · 13 (Observability)
**Time:** ~60 minutes

## 学习目标

- 列举四种机密管理反模式（VCS 中的配置文件、硬编码 env、电子表格、静态密钥）并说出其替代方案。
- 解释 AI 网关从保管库拉取模式作为 2026 年生产标准。
- 实现带一致分词的 PII 清理器（相同值 → 相同占位符），使语义得以保留。
- 说出 2026 年 Vercel 供应链事件及其关于 CI/CD 凭据规范的教训。

## 问题

一名实习生提交了带 API 密钥的 `.env`。他们很快删除了它。但密钥已在 git 历史中 — GitGuardian 扫描捕获到它，你的轮换过程是"在 Slack 通知团队，更新 40 个配置文件，重新部署所有服务。"8 小时后，一半服务上线，一半在等待部署窗口。

另外，用户提示包括"我的 SSN 是 123-45-6789。"提示发送到 OpenAI。你有 BAA 但你的内部策略是在转发前掩码 PII。你没有。

另外，你的 EKS 集群的 LLM pod 可以访问任何互联网主机。有人通过 DNS 查询向攻击者控制的域泄露数据。没有任何东西阻止它。

LLM 服务的安全必须处理所有三个向量。保管库支持的凭据。PII 清理。网络出口过滤。审计日志。

## 概念

### 集中化保管库 + IAM 角色拉取

**保管库**：HashiCorp Vault、AWS Secrets Manager、Azure Key Vault、GCP Secret Manager。一个真相来源。

**IAM 角色**：应用/网关通过其 IAM 身份认证，而非静态密钥。保管库返回有效期为 token 生命周期的机密。

**AI 网关模式**：网关在请求时从保管库拉取 `OPENAI_API_KEY`。在保管库中轮换；下一个请求获得新密钥。无需重新部署。

### 轮换策略 ≤ 90 天

所有 API 密钥、保管库根 token、CI/CD 凭据。尽可能自动化轮换。手动轮换记录并跟踪。

### 机密扫描

- **TruffleHog** — 基于提交的正则 + 熵。
- **GitGuardian** — 商业，高准确率。
- **Gitleaks** — OSS，在 CI 中运行。

对每次提交运行。如果检测到新机密，阻止 PR。

### 零信任姿态

- 所有账户强制 MFA。
- 通过 SAML/OIDC 实现 SSO。
- RBAC（基于角色）或 ABAC（基于属性）用于细粒度访问。
- 短期 token（小时，不是天）。
- 设备姿态 — 仅带磁盘加密的公司设备。

### PII / PHI 清理

在提示离开你的基础设施之前：

1. 实体识别（spaCy NER、Presidio、商业）。
2. 掩码匹配的实体：`"My SSN is 123-45-6789"` → `"My SSN is [SSN_TOKEN_A3F]"`。
3. 一致分词（Mesh 方法）：相同值映射到相同占位符，使 LLM 保留关系。
4. 可选反向映射用于 LLM 响应。

静态正则过滤器捕获基本模式；NER 捕获更多。两者都使用。

### 输入 + 输出护栏

输入：阻止已知越狱、禁止话题；按用户速率限制。

输出：正则清理泄露的机密（拒绝上下文中的 API 密钥模式、电子邮件模式），分类器检测策略违规。

### 网络出口白名单

LLM 服务在专用子网中：
- 白名单：`api.openai.com`、`api.anthropic.com`、向量数据库端点、保管库端点。
- 其他一切：丢弃。
- 通过仅白名单解析器的 DNS（避免 DNS 隧道泄露）。

### 审计日志

每次 LLM 调用的不可变日志，包含：
- 时间戳。
- 用户 / 租户。
- 提示哈希（出于隐私不用原始提示）。
- 模型 + 版本。
- Token 数量。
- 成本。
- 响应哈希。
- 任何护栏触发。

按监管要求保留（SOC 2 1 年、HIPAA 6 年）。

### 2026 年 Vercel 事件

供应链攻击：被入侵的 CI/CD 凭据泄露了数千客户部署的环境变量。教训：CI/CD 凭据等同于生产凭据。存储在保管库中。范围狭窄。积极轮换。

### 你应该记住的数字

- 轮换策略：≤ 90 天。
- 每次提交扫描：TruffleHog / GitGuardian / Gitleaks。
- Vercel 2026：CI/CD 凭据被入侵 → 数千客户环境变量泄露。
- 审计日志保留：SOC 2 = 1 年、HIPAA = 6 年。

## 使用它

`code/main.py` 实现带一致分词的玩具 PII 清理器和仅追加的审计日志。

## 交付它

本课产出 `outputs/skill-llm-security-plan.md`。给定监管范围与当前状态，计划保管库迁移、清理器、出口、审计日志。

## 练习

1. 运行 `code/main.py`。发送两个引用相同 SSN 的提示。确认两个获得相同占位符。
2. 为调用 OpenAI + Anthropic + Weaviate 的 vLLM-on-EKS 部署设计网络出口策略。
3. 你在 git 历史中发现一个密钥（2 年前的）。正确的响应是什么 — 轮换密钥、清理历史、还是两者？论证。
4. 你的审计日志每天增长 10 GB。设计保留层（热层 30 天、温层 12 月、冷层 6 年）。
5. 论证反向分词（将真实值替换回 LLM 响应中）是否值得复杂性 vs 保持占位符可见。

## 核心术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Vault | "机密存储" | 集中化凭据管理服务 |
| IAM role | "基于身份认证" | 应用扮演的角色；返回短期凭据 |
| OIDC for CI/CD | "云颁发 token" | CI 中无静态密钥 — 通过 OIDC 身份 |
| TruffleHog / GitGuardian / Gitleaks | "机密扫描器" | 提交时机密检测 |
| RBAC / ABAC | "访问控制" | 基于角色 vs 基于属性 |
| PII scrubbing | "数据掩码" | 移除或分词化敏感实体 |
| Consistent tokenization | "稳定占位符" | 相同值 → 每次相同 token |
| Mesh approach | "Mesh 分词" | 语义保留的分词模式 |
| Egress whitelist | "出站白名单" | 仅允许的域名可达 |
| Audit log | "不可变历史" | 仅追加记录用于合规 |

## 进一步阅读

- [Doppler — 高级 LLM 安全](https://www.doppler.com/blog/advanced-llm-security)
- [Portkey — 用秘密引用管理 LLM API 密钥](https://portkey.ai/blog/secret-references-ai-api-key-management/)
- [Datadog — LLM 护栏最佳实践](https://www.datadoghq.com/blog/llm-guardrails-best-practices/)
- [JumpServer — 机密管理最佳实践 2026](https://www.jumpserver.com/blog/secret-management-best-practices-2026)
- [Microsoft Presidio](https://github.com/microsoft/presidio) — PII 检测和匿名化。
- [HashiCorp Vault 文档](https://developer.hashicorp.com/vault/docs)
