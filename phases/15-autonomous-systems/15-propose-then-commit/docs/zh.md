# 人在回路中：提议然后提交

> 2026年关于HITL的共识是具体的。它不是"智能体询问，用户点击批准"。它是提议然后提交：提议的行动持久化到带有幂等键的持久存储中；以意图、数据沿袭、触及的权限、爆炸半径和回滚计划呈现给审阅者；仅在正面确认后提交；执行后验证以确认副作用确实发生。LangGraph的 `interrupt()` 搭配PostgreSQL检查点、Microsoft Agent Framework的 `RequestInfoEvent` 以及Cloudflare的 `waitForApproval()` 都实现了相同的形状。经典的失败模式是橡皮图章批准：不对"批准？"进行审查就点击。有记录的缓解措施是带有明确清单的挑战-响应。

**Type:** Learn
**Languages:** Python (stdlib, propose-then-commit state machine with idempotency)
**Prerequisites:** Phase 15 · 12 (Durable execution), Phase 15 · 14 (Tripwires)
**Time:** ~60 minutes

## 问题

一个智能体采取一项行动。用户必须决定：批准或不批准。如果决定是即时的，它很可能不是一次审查。如果决定是结构化的，它是慢的但值得信赖。工程问题是如何让结构化审查成为阻力最小的路径。

2023年时代的HITL模式是同步提示："智能体想要将邮件发送给X，正文为Y——批准？"用户点击批准。每个人都觉得系统是安全的。在实践中，这个表面被严重橡皮图章化：用户批准得很快，批准预测不了什么，当智能体出错时，审计追踪显示用户无法回忆的一系列批准历史。

2026年模式——提议然后提交——将HITL移动到持久基底上，附加结构化元数据，并要求正面提交。每个托管智能体SDK都发布了一个版本：LangGraph `interrupt()`、Microsoft Agent Framework `RequestInfoEvent`、Cloudflare `waitForApproval()`。API名称不同；形状相同。

## 概念

### 提议然后提交状态机

1. **提议。** 智能体生成一个提议行动。持久化到持久存储（PostgreSQL、Redis、Durable Object）。包括：
   - 意图（智能体为什么要这样做）
   - 数据沿袭（什么来源导致了此提议）
   - 触及的权限（哪些范围/文件/端点）
   - 爆炸半径（最坏情况是什么）
   - 回滚计划（如果提交了，我们如何撤销它）
   - 幂等键（每个提议唯一；重新提交返回相同记录）
2. **呈现。** 审阅者看到带有所有元数据的提议。审阅者是人（而非智能体审阅自己）。
3. **提交。** 正面确认。行动执行。
4. **验证。** 执行后，副作用被读回并确认。如果验证步骤失败，系统处于已知的不良状态，并启动警报。

### 幂等键

没有幂等键，一次瞬态失败后的重试可以双重执行已批准的行动。具体示例：用户批准"从A转账100美元到B"。网络短暂中断。工作流重试。用户批准了一次，但转账执行了两次。幂等键将批准与单一、唯一的副作用绑定；第二次执行是无操作。

这与Stripe和AWS API使用的幂等模式相同。在智能体批准上重用它，在Microsoft Agent Framework文档中有明确说明。

### 持久性：为什么批准要活得比进程长

批准等候室是智能体不拥有的一块状态。工作流被暂停（第12课）。当批准到达时，工作流从该确切点恢复。这就是为什么LangGraph将 `interrupt()` 与PostgreSQL检查点配对，而不仅使用内存状态——两天后的批准仍能发现工作流完好无损。

### 橡皮图章批准与挑战-响应缓解

HITL的默认UI（"批准"/"拒绝"按钮）产生快速批准而没有真正审查。有记录的缓解措施：一个挑战-响应清单，在启用批准按钮之前要求对特定问题做出正面回答。具体形状：

- "你理解这个行动触及了什么资源吗？[ ]"
- "你验证了爆炸半径是可接受的吗？[ ]"
- "如果失败，你有回滚计划吗？[ ]"

不是为官僚主义而官僚主义——而是一种强制功能。不能勾选方框的审阅者要么请求澄清（升级），要么拒绝（安全默认）。Anthropic的智能体安全研究明确引用了由清单驱动的HITL作为橡皮图章批准模式的缓解措施。

### 什么算作后果性的

并非每个行动都需要提议然后提交。2026年指导：

- **后果性行动**（始终HITL）：不可逆写入、金融交易、对外通信、生产数据库变更、破坏性文件系统操作。
- **可逆行动**（有时HITL）：对本地文件的编辑、暂存环境变更、具有明确回滚的可逆写入。
- **读取和检查**（从不HITL）：读取文件、列出资源、调用只读API。

### 行动后验证

"提交已运行"不等于"副作用已发生"。网络分区和竞态条件可能产生一个认为已成功的工作流，而后端并未持久化。验证步骤在提交后重新读取目标资源以确认。这与带有 `RETURNING` 子句的数据库事务或 `PutObject` 后的AWS `GetObject` 模式相同。

### EU AI Act第14条

第14条要求对欧盟高风险AI系统进行有效的人工监督。"有效"不是装饰性的。监管语言明确排除了橡皮图章模式。在Microsoft Agent Governance Toolkit合规文档中，带有挑战-响应的提议然后提交是能够经受第14条审查的形状。

## 运用

`code/main.py` 在标准库Python中实现提议然后提交状态机。持久存储是一个JSON文件。幂等键是（thread_id，action_signature）的哈希。驱动程序模拟三种情况：干净批准流程、瞬态失败后的重试（必须不双重执行），以及橡皮图章默认与挑战-响应流程的对比。

## 交付物

`outputs/skill-hitl-design.md` 审查一个提议的HITL工作流，检查其提议然后提交的形状，并标记缺失的元数据、幂等性、验证或挑战-响应层。

## 练习

1. 运行 `code/main.py`。确认对已批准提议的重试使用持久记录且不重新执行。现在更改幂等键以包含时间戳，展示重试双重执行。

2. 用 `rollback` 字段扩展提议记录。模拟一个验证步骤失败的执行。展示回滚自动触发。

3. 阅读Microsoft Agent Framework的 `RequestInfoEvent` 文档。确定API包含而玩具引擎缺失的一个元数据字段。添加它并解释它保护什么。

4. 为特定行动设计挑战-响应清单（例如"发布到公共Twitter账户"）。审阅者必须回答哪三个问题？为什么是这三个？

5. 选择一个同步"批准？"提示足够的情况（不需要持久存储）。解释为什么，并指出你正在接受的风险类别。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|---|---|---|
| 提议然后提交 | "两阶段批准" | 持久化提议 + 正面提交 + 验证 |
| 幂等键 | "重试安全令牌" | 每个提议唯一；第二次执行无操作 |
| 数据沿袭 | "它来自哪里" | 导致提议的具体来源内容 |
| 爆炸半径 | "最坏情况" | 行动出错时的效果范围 |
| 橡皮图章 | "快速批准" | 未经真正审查即点击"批准" |
| 挑战-响应 | "强制清单" | 审阅者必须正面承认特定问题 |
| RequestInfoEvent | "MS Agent Framework原语" | 带有结构化元数据的持久HITL请求 |
| `interrupt()` / `waitForApproval()` | "框架原语" | LangGraph / Cloudflare的相同形状的等价物 |

## 进一步阅读

- [Microsoft Agent Framework — Human in the loop](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — `RequestInfoEvent`、持久批准。
- [Cloudflare Agents — Human in the loop](https://developers.cloudflare.com/agents/concepts/human-in-the-loop/) — `waitForApproval()` 和 Durable Objects。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — HITL作为长周期风险的缓解措施。
- [EU AI Act — Article 14: Human oversight](https://artificialintelligenceact.eu/article/14/) — 高风险系统的监管基线。
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — 围绕监督的宪制框架。
