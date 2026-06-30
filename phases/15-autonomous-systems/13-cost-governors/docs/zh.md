# 行动预算、迭代上限与成本控制器

> 一个中型电子商务智能体在团队启用"订单跟踪"技能后，其月LLM成本从1,200美元跃升至4,800美元。这不是定价Bug。这是一个找到了新循环并在其中持续花费的智能体。微软的Agent Governance Toolkit（2026年4月2日）编纂了针对此类的防御：每次请求 `max_tokens`、每个任务token和美元预算、每天/每月上限、迭代上限、分层模型路由、提示缓存、上下文窗口化、昂贵行动上的HITL检查点、预算超限的熔断开关。Anthropic的Claude Code Agent SDK以不同名称发布了相同的原语。财务速度限制——例如在10分钟内花费超过50美元即切断访问——比每月上限更快捕获循环。

**Type:** Learn
**Languages:** Python (stdlib, layered cost-governor simulator)
**Prerequisites:** Phase 15 · 10 (Permission modes), Phase 15 · 12 (Durable execution)
**Time:** ~60 minutes

## 问题

自主智能体在每个回合都花费真金白银。聊天机器人的不良输出是一个糟糕的回复；智能体的不良循环是一张账单。业界有记录的失败模式术语是"钱包拒绝服务"——智能体持续推理、持续调用工具、持续计费，没有任何东西阻止它，因为没有设计任何东西来阻止它。

修复方案不是一个数字。而是一个在不同时间尺度和粒度上的限制堆栈：每次请求、每个任务、每小时、每天、每月。一个设计良好的堆栈在几分钟内捕获失控循环，在几小时内捕获缓慢泄漏，在一天内捕获不良发布。当智能体是长周期且自主的时候，同一堆栈始终维持一个预算。

这是一堂工程课：数学是平凡的，纪律是团队失败的地方。下面的限制列表在微软Agent Governance Toolkit或Anthropic Claude Code Agent SDK文档中都有命名。

## 概念

### 成本控制器堆栈

1. **每次请求 `max_tokens`。** 简单。防止任何一次调用发出无限完成。
2. **每个任务token预算。** 在整个运行过程中，不超过N个token。在上限处硬停止。
3. **每个任务美元预算。** 与token相同但以货币计。Claude Code中的 `max_budget_usd`。
4. **每个工具调用上限。** 不超过N次 `WebFetch` 调用、N次 `shell_exec` 调用等。
5. **迭代上限（`max_turns`）。** 智能体循环总迭代次数；防止无限推理循环。
6. **每分钟/每小时/每天/每月上限。** 滚动窗口。在不同时间尺度捕获泄漏。
7. **财务速度限制。** 例如，"如果10分钟内支出超过50美元，切断访问。"在每月上限触发之前捕获基于循环的消耗。
8. **分层模型路由。** 默认使用较小的模型；仅当分类器判断任务需要时才升级到更大的模型。
9. **提示缓存。** 系统提示和稳定上下文存储在提供者缓存中；重新发送的token成本接近零。
10. **上下文窗口化。** 压缩/摘要以将活跃上下文保持在阈值以下；直接降低token成本。
11. **昂贵行动上的HITL检查点。** 在执行已知昂贵的行动（长工具调用、大下载、昂贵的模型升级）之前要求人工确认。
12. **预算超限的熔断开关。** 任何上限触发时，会话中止。上限被记录；需要单独的重新启用路径。

### 为什么是堆栈，而不是单一上限

单一每月上限只在钱包消失后才捕获失控智能体。单一每次请求上限在会话级别不捕获任何东西。不同的失败模式需要不同的时间尺度：

- **失控循环**（智能体陷入5秒重试）：由速度限制捕获。
- **缓慢泄漏**（智能体每个任务做约2倍于预期的操作）：由每日上限捕获。
- **不良发布**（新版本使用5倍token）：由每周/每月上限捕获。
- **合法激增**（真实需求而非Bug）：由带清晰日志的小时/天上限捕获。

### Claude Code的预算表面

Claude Code Agent SDK公开了（公开文档）：

- `max_turns`——迭代上限。
- `max_budget_usd`——美元上限；超限时会话中止。
- `allowed_tools` / `disallowed_tools`——工具允许列表和拒绝列表。
- 工具使用前的钩子点，用于自定义成本核算。

与权限模式阶梯（第10课）结合。一个没有 `max_budget_usd` 的 `autoMode` 会话是不受治理的自主性。Anthropic明确将自动模式框定为需要预算控制；分类器与成本是正交的。

### EU AI Act，OWASP Agentic Top 10

微软的Agent Governance Toolkit涵盖OWASP Agentic Top 10和EU AI Act第14条（人工监督）要求。对于在欧盟的生产环境，日志记录和上限执行不是可选的。

### 观察到的1,200→4,800美元案例

微软文档中的真实案例：一个电子商务智能体在添加新工具后月成本翻了三倍。该工具允许智能体在每次会话期间轮询订单状态。没有循环检测。没有每个工具上限。没有关于周与周增长的警报。修复方案是每个工具上限加上每日增长警报。这是一个模板：每个新工具表面都是一个潜在的新循环；每个新工具需要自己的上限和自己的警报。

## 运用

`code/main.py` 模拟智能体在有和没有分层成本控制器堆栈的情况下的运行。模拟的智能体在若干回合后漂移进入轮询循环；分层堆栈在速度窗口内捕获它，而单一每月上限要几天后才触发。

## 交付物

`outputs/skill-agent-budget-audit.md` 审计提议的智能体部署的成本控制器堆栈，并标记缺失的层。

## 练习

1. 运行 `code/main.py`。确认在轮询循环轨迹上速度限制在迭代上限之前触发。现在禁用速度限制，测量智能体在迭代上限捕获它之前"花费"了多少。

2. 为浏览器智能体设计每个工具上限集合（第11课）。哪个工具需要最严格的上限？哪个工具可以无限制运行而没有风险？

3. 阅读微软Agent Governance Toolkit文档。列出工具包命名的每种上限类型。将每种映射到一种失败模式（失控循环、缓慢泄漏、不良发布、激增）。

4. 为一个现实任务（例如"分类仓库中的50个Issue"）的隔夜无人值守运行定价。将 `max_budget_usd` 设置为你的点估计值的2倍。证明这个2倍。

5. Claude Code的 `max_budget_usd` 在会话累积成本上触发。设计一个你将在外部强制执行的补充速度限制。什么触发切断，重新启用是什么样子？

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|---|---|---|
| 钱包拒绝服务 | "失控账单" | 智能体循环产生支出而没有上限阻止它 |
| max_tokens | "每次请求上限" | 对单次完成的大小的限制 |
| max_turns | "迭代上限" | 会话中智能体循环迭代的限制 |
| max_budget_usd | "美元熔断开关" | 会话成本上限；超限时中止 |
| 速度限制 | "速率上限" | 短时间窗口内的支出限制（例如每10分钟50美元） |
| 分层路由 | "小模型优先" | 廉价模型默认；仅当分类器需要时才升级 |
| 提示缓存 | "缓存的系统提示" | 提供者端缓存将重新发送的token成本降至接近零 |
| HITL检查点 | "人工审批门" | 在昂贵行动前需要人工确认 |

## 进一步阅读

- [Anthropic Claude Code Agent SDK — agent loop and budgets](https://code.claude.com/docs/en/agent-sdk/agent-loop) — `max_turns`、`max_budget_usd`、工具允许列表。
- [Microsoft Agent Framework — human-in-the-loop and governance](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — 成本控制器检查点。
- [Anthropic — Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — 提供者端成本控制。
- [Anthropic — Prompt caching (Claude API docs)](https://platform.claude.com/docs/en/prompt-caching) — 缓存机制。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 长周期智能体的成本配置文件。
