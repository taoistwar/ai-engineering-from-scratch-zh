# AI 的 SRE — 多代理事件响应、运行手册、预测检测

> AI SRE 使用基于基础设施数据（日志、运行手册、服务拓扑）通过 RAG 的 LLM 来自动化调查、文档和协调阶段。2026 年的架构模式是多代理编排 — 专业化代理（日志、指标、运行手册）由监督者协调；AI 提出假设和查询，人类批准判断。Datadog Bits AI 和 Azure SRE Agent 以托管产品形式提供。运行手册正在演化：NeuBird Hawkeye 使用对抗性评估（两个模型分析同一事件；一致 = 置信，不一致 = 不确定）；运维记忆在团队更替中保留。自动修复保持谨慎：AI 建议，人类批准。完全自主行动是狭窄的（重启 pod、回滚特定部署）具有严格护栏 — 任何卖"设置后忘记"的人都在过度推销。新兴前沿：事件前预测。MIT 研究报告一个在历史日志 + GPU 温度 + API 错误模式上训练的 LLM 预测了 89% 的中断，提前 10-15 分钟。预测：到 2026 年底，95% 的企业 LLM 拥有自动化故障转移。

**Type:** Learn
**Languages:** Python (stdlib, toy multi-agent incident triage simulator)
**Prerequisites:** Phase 17 · 13 (Observability), Phase 17 · 24 (Chaos Engineering)
**Time:** ~60 minutes

## 学习目标

- 绘制多代理 AI SRE 架构：监督者 + 专业化代理（日志、指标、运行手册）+ 人类批准门控。
- 解释为什么自动修复是狭窄的（重启 pod、回滚部署）而非广泛的（重新设计服务架构）。
- 说出对抗性评估模式（NeuBird Hawkeye）：两个模型一致 = 置信；不一致 = 升级。
- 引用 MIT 89% 早期检测结果和运维约束：没有执行的预测只是仪表板。

## 问题

一名值班工程师在凌晨 3 点被寻呼。"结账页面错误率高。"他们检查 Datadog、Loki、三本运行手册、部署日志。30 分钟后他们意识到根因是 KV 缓存尖峰导致的 vLLM OOM。他们重启 pod；错误清除。

2026 年，调查的前 20 分钟是可自动化的。按服务分组日志、关联到最近的部署、匹配运行手册 — 这些都是 RAG + 工具使用。在被监督的代理可以在人类打开 Datadog 之前完成初步分类并提出假设。

完全自主修复是不同的问题。重启 pod：安全。扩展 GPU 池：在策略允许下安全。重新设计服务架构：绝对不行。纪律在于划定狭窄的界限。

## 概念

### 多代理架构

```
          Incident
             │
             ▼
        Supervisor
        /    |    \
       ▼     ▼     ▼
  Log agent  Metric agent  Runbook agent
       │     │     │
       └─────┴─────┘
             │
             ▼
        Hypothesis + evidence
             │
             ▼
        Human approval
             │
             ▼
        Action (narrow set)
```

监督者将事件分解为子查询。专业化代理具有工具访问权限（日志搜索、PromQL、文档检索）。监督者综合，向人类提出假设 + 证据。人类批准或重定向。

### 自动修复范围

**安全（狭窄）**：重启 pod、回滚特定部署、在预批准边界内扩展池、启用预批准的功能标志。

**不安全（广泛）**：更改服务拓扑、修改资源限制、部署新代码、更改 IAM、修改数据库。

任何卖"设置后忘记"的人都在过度推销。随着 AI SRE 成熟，安全集合增长，但边界是真实的。

### 对抗性评估（NeuBird Hawkeye）

两个模型独立分析同一事件。如果它们对根因的意见一致，置信度很高。如果不一致，升级到人类，两个假设都可见。简单模式，有效过滤虚假根因。

### 运维记忆

团队更替是传统 SRE 的无声杀手 — 领域知识流失。AI SRE 将运行手册 + 事后分析存储在向量数据库中；代理在每个新事件上检索。当新工程师加入时，AI 拥有完整历史。

### 事件前预测

MIT 2025 年研究：在历史日志、GPU 温度、API 错误模式上训练的 LLM 在测试集上预测了 89% 的中断，提前 10-15 分钟。

现实检查：没有执行的预测只是仪表板。运维问题是"当我们预测时，我们做什么？"预排空？寻呼？自动扩展？答案是策略特定的。

### 2026 年产品

- **Datadog Bits AI** — Datadog 内部的托管 SRE 副驾驶。
- **Azure SRE Agent** — Azure 原生。
- **NeuBird Hawkeye** — 对抗性评估 + 运维记忆。
- **PagerDuty AIOps** — 分类 + 去重。
- **Incident.io Autopilot** — 事件指挥官 + 协调。

### 运行手册即代码

运行手册从 Confluence 页面演化为具有结构化部分的版本化 markdown（症状、假设、验证、行动）。结构化运行手册提供更好的 RAG 检索。从将非结构化运行手册转化为结构化开始任何 AI-SRE 部署。

### 你应该记住的数字

- MIT 早期检测：89% 的中断，10-15 分钟提前。
- 多代理分类：监督者 +（日志、指标、运行手册）+ 人类。
- 安全自动修复集合：重启 pod、回滚部署、在边界内扩展。
- 对抗性评估：两个模型独立；一致 = 置信。

## 使用它

`code/main.py` 模拟多代理分类：日志代理发现错误、指标代理发现 CPU 尖峰、运行手册代理匹配到已知问题。监督者排列假设。

## 交付它

本课产出 `outputs/skill-ai-sre-plan.md`。给定当前值班、事件量、团队成熟度，设计 AI SRE 部署。

## 练习

1. 运行 `code/main.py`。如果日志和指标代理意见不一致会怎样？监督者如何解决？
2. 为你的服务定义三个"安全"自动修复动作。论证每个。
3. 编写结构化运行手册模板：部分、必填字段、验证命令。
4. 预测检测在 12 分钟提前触发。你的策略是什么 — 寻呼、预排空、还是两者？
5. 论证一个 3 人团队应在 2026 年采用 AI SRE 还是等待。考虑成熟度、量、风险。

## 核心术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| AI SRE | "值班代理" | LLM 支持的事件调查 + 协调 |
| Supervisor agent | "编排器" | 将事件分解为子查询的顶级代理 |
| Specialized agent | "领域代理" | 具有工具访问权限的子代理（日志、指标、运行手册） |
| Auto-remediation | "AI 修复" | 狭窄的预批准动作；不是广泛的重新设计架构 |
| Operational memory | "向量运行手册" | 事后分析 + 运行手册在向量数据库中用于 RAG |
| Adversarial eval | "两模型检查" | 独立分析；一致 = 置信 |
| NeuBird Hawkeye | "对抗性那个" | 具有对抗性评估 + 记忆模式的产品 |
| Bits AI | "Datadog 的 SRE 代理" | Datadog 托管的 AI SRE |
| Pre-incident prediction | "早期检测" | 中断预测提前 10-15 分钟 |

## 进一步阅读

- [incident.io — AI SRE 完整指南 2026](https://incident.io/blog/what-is-ai-sre-complete-guide-2026)
- [InfoQ — 以人为中心的 AI SRE](https://www.infoq.com/news/2026/01/opsworker-ai-sre/)
- [DZone — SRE 中的 AI 2026](https://dzone.com/articles/ai-in-sre-whats-actually-coming-in-2026)
- [Datadog Bits AI](https://www.datadoghq.com/product/bits-ai/)
- [NeuBird Hawkeye](https://www.neubird.ai/)
- [awesome-ai-sre](https://github.com/agamm/awesome-ai-sre)
