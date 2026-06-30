# 作为可执行约束的 Agent 指令

> 写成了散文的指令是愿望。写成了约束的指令是测试。工作台将每条规则转化为 agent 在运行时可以检查、审查者在事后可以验证的东西。

**类型：** 构建
**语言：** Python（标准库）
**前置课程：** 第 14 阶段 · 32（最小工作台）
**时间：** 约 50 分钟

## 学习目标

- 将路由散文与可操作规则分离。
- 将启动规则、禁止操作、完成的定义、不确定性处理以及审批边界表达为机器可检查的约束。
- 实现一个规则检查器，根据规则集对一次运行进行评分。
- 使规则集对 diff 友好，以便审查能看到变更内容。

## 问题

一份典型的 `AGENTS.md` 读起来像入职文档。它告诉 agent 要"小心"、"全面测试"、"不确定就问"。三天后，agent 提交了一个没有任何测试的更改，写入了禁止的目录，而且从未提问——因为它从来不知道边界在哪里。

指令在可操作时是强大的，在只是愿望时是脆弱的。解决方法是写下工作台可以解读、审查者可以评分的规则。

## 概念

规则放在 `docs/agent-rules.md` 中，与简短的路由文件分离。每条规则有一个名称、一个类别和一个检查。

```mermaid
flowchart LR
  Router[AGENTS.md] --> Rules[docs/agent-rules.md]
  Rules --> Checker[rule_checker.py]
  Checker --> Report[rule_report.json]
  Report --> Reviewer[审查者]
```

### 覆盖大多数规则的五种类别

| 类别 | 规则要回答的问题 | 示例 |
|------|----------------|------|
| 启动 | 工作开始前必须满足什么？ | "状态文件存在且是新的" |
| 禁止 | 什么绝对不能发生？ | "不要编辑 `scripts/release.sh`" |
| 完成的定义 | 什么证明任务已完？ | "pytest 以 0 退出，且验收行通过" |
| 不确定性 | agent 不确定时该怎么做？ | "打开一个问题记录而不是猜测" |
| 审批 | 什么需要人工审批？ | "任何新依赖、任何生产环境写入" |

不在这五类中的规则通常应该拆成两条规则。强制拆分。

### 规则是机器可读的

每条规则有一个短标签（slug）、一个类别、一行描述以及一个指向 `rule_checker.py` 中函数的 `check` 字段。添加一条规则意味着添加一个检查；检查器随工作台一起成长。

### 规则对 diff 友好

规则在一个单独的 markdown 文件中每条占一个标题。重命名在 diff 中可见。新规则放在其类别的最前面。过时的规则被删除，而不是被注释掉，因为工作台是真相的来源，而不是团队上个季度心情的聊天记录。

### 规则与框架护栏

框架护栏（OpenAI Agents SDK 的 guardrails、LangGraph 的 interrupts）在运行层强制执行规则。本课程中的规则集是这些护栏所实现的人类可读、可审查的契约。你需要两者：运行时在回合中捕获违规，规则集证明运行时在做正确的事。

### 渐进式披露：一张地图，而不是百科全书

`AGENTS.md` 不断膨胀的原因是每个事故添加一条规则，却没有事故删除一条规则。一年后，文件有两千行，agent 读了第一屏，注意力预算耗尽，只照着被告知内容的一小部分行动。一个巨大的指令文件失败的原因与一个四十页的入职文档失败的原因相同：读者扫一眼就再也不回到重要部分。

解决方案不是更短的文件，而是分层文件。根路由文件保持足够小，每次会话都能读完，只包含指针。深度内容放在主题文件中，agent 只在任务涉及它们时才加载。给 agent 一张地图，而不是整本百科全书，让它自己走到需要的页面。

```
AGENTS.md                  # 路由，< 50 行：这个仓库是什么，到哪里找，5 条硬规则
docs/
  agent-rules.md           # 完整的规则集（本课程）
  architecture.md          # 当任务触及模块边界时加载
  testing.md               # 当任务编写或运行测试时加载
  deploy.md                # 仅用于发布工作，由审批规则控制
feature_list.json          # 待办事项列表（第 14 阶段 · 36）
```

| 层级 | 所在文件 | 读取时机 | 大小预算 |
|------|----------|---------|---------|
| 路由 | `AGENTS.md` | 每次会话，始终 | 约 50 行以内 |
| 规则 | `docs/agent-rules.md` | 每次会话，启动时 | 每个类别一屏 |
| 主题文档 | `docs/<topic>.md` | 仅在任务涉及该主题时 | 按需深入 |

两个测试保持分层的真实性。可达性测试：agent 应该最多两跳从路由到达任何规则，因此路由必须通过路径链接每个主题文档，而不是用散文描述它。新鲜度测试：路由足够短，审查者在每个 PR 上都会重读它，这是唯一能阻止它悄悄膨胀回它所取代的百科全书的方法。一个不再解析的指针比丢失一条规则更糟糕，因此路由中的断链本身就是启动检查的违规。

## 构建

`code/main.py` 提供：

- 解析 `agent-rules.md` 并将规则加载为数据类的解析器。
- `rule_checker.py` 风格的检查函数，每个对应一个 `check` 引用。
- 一个违反两条规则的演示 agent 运行，以及一个捕获它们的检查通过。

运行：

```
python3 code/main.py
```

输出：解析后的规则集、运行轨迹、每条规则的通过/失败，以及脚本旁边保存的 `rule_report.json`。

## 真实生产中的模式

三种模式区分了一个能持续一个季度的规则集和一个一周内就会失效的规则集。

**写入时打上严重性标签。** 每条规则带有 `severity`：`block`、`warn` 或 `info`。检查器报告所有三种；运行时只在 `block` 上拒绝。大多数团队初期高估严重性，然后在截止日期压力下悄悄降低；写入时打标签会前置这次校准。配合验证门（第 14 阶段 · 38）一起使用，它会将对 `block` 规则的任何覆盖签入 `overrides.jsonl` 审计日志。

**规则过期作为强制函数。** 每条规则带有一个 `expires_at` 日期（默认从撰写起 90 天）。检查器在一条未过期规则连续 60 天零违规时发出警告；下一次季度审查要么证明它仍然合理，将其降级为 `info`，要么删除它。Cloudflare 的生产 AI 代码审查数据（2026 年 4 月，30 天内 5,169 个仓库中的 131,246 次审查运行）显示，具有显式过期期限的规则集保持在每仓库 30 条规则以下；而没有的规则集增长到 80 条以上，且大多数从未触发。

**Markdown 作为源，JSON 作为缓存。** `agent-rules.md` 是编写的文件；`agent-rules.lock.json` 是检查器在热路径中读取的缓存。锁文件由 pre-commit hook 重新生成。Markdown 差异可审查；JSON 解析不会出现在每个回合中。与 `package.json` / `package-lock.json` 以及 `Cargo.toml` / `Cargo.lock` 相同的模式。

## 使用

在生产中：

- Claude Code、Codex、Cursor 在会话启动时读取规则，并在拒绝操作时引用它们。检查器在 CI 中重新运行它们以捕获悄悄的偏离。
- OpenAI Agents SDK 的 guardrails 将相同的检查注册为输入和输出 guardrails。Markdown 是文档表层；SDK 是运行时表层。
- LangGraph 的 interrupts 在某个飞行节点违反规则时触发。中断处理程序读取规则，询问人类，然后恢复。

规则集在三个平台上都是可移植的，因为它只是 markdown 加上函数名称。

## 交付

`outputs/skill-rule-set-builder.md` 采访项目负责人，将其现有的散文指令归类到五个类别中，并发出一个带版本的 `agent-rules.md` 加一个检查器存根。

## 练习

1. 如果你的产品确实需要一个第六类别，请添加。论证为什么它不能被合并到五个类别中的一个。
2. 扩展检查器，使规则可以带有严重性（`block`、`warn`、`info`），报告相应汇总。
3. 将检查器接入 CI：如果最新 agent 运行中有 block 严重性规则失败，则构建也失败。
4. 为每条规则添加一个"过期"字段。90 天没有检查失败后，该规则进入审查。
5. 找一份真实的 `AGENTS.md`，将其重写为五类规则。其中有多少行是可操作的？又有多少只是愿望？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 可操作规则 | "一条真正的指令" | 工作台可以在运行时检查的规则 |
| 愿望式规则 | "要小心" | 一条没有检查的规则；要么删除，要么升级 |
| 完成的定义 | "验收" | 一个客观的、有文件支撑的任务完成证明 |
| Block 严重性 | "硬规则" | 违规会中断运行；不能在没有操作者的情况下静默处理 |
| 规则过期 | "陈旧规则清理" | 一条在 N 天内没有失败的规则可以退役 |

## 进一步阅读

- [OpenAI Agents SDK guardrails](https://platform.openai.com/docs/guides/agents-sdk/guardrails)
- [LangGraph interrupts](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/breakpoints/)
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)
- [Rick Hightower, Agent RuleZ: A Deterministic Policy Engine](https://medium.com/@richardhightower/agent-rulez-a-deterministic-policy-engine-for-ai-coding-agents-9489e0561edf) — 生产中的 block/warn/info 严重性
- [Cloudflare, Orchestrating AI Code Review at Scale](https://blog.cloudflare.com/ai-code-review/) — 131k 次审查运行，规则组合经验
- [microservices.io, GenAI development platform — part 1: guardrails](https://microservices.io/post/architecture/2026/03/09/genai-development-platform-part-1-development-guardrails.html) — 规则与 CI 之间的纵深防御
- [Type-Checked Compliance: Deterministic Guardrails (arXiv 2604.01483)](https://arxiv.org/pdf/2604.01483) — Lean 4 作为规则即检查的上限
- [logi-cmd/agent-guardrails](https://github.com/logi-cmd/agent-guardrails) — 合并门实现：范围、变异测试、违规预算
- 第 14 阶段 · 32 — 此规则集插入的最小工作台
- 第 14 阶段 · 38 — 消费规则报告的验证门
- 第 14 阶段 · 39 — 对规则合规性评分的审查 agent
