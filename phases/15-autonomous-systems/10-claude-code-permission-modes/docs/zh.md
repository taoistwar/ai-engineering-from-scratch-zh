# Claude Code作为自主智能体：权限模式与自动模式

> Claude Code公开了七种权限模式。"plan"在每次动作前询问，"default"仅对危险动作询问，"acceptEdits"自动批准文件写入但仍确认shell执行，而"bypassPermissions"批准一切。自动模式（2026年3月24日）用两阶段并行安全分类器取代了每次动作的审批：单token快速检查在每次动作上运行；被标记的动作触发思维链深度审查。动作预算通过 `max_turns` 和 `max_budget_usd` 强制执行。自动模式作为研究预览发布——Anthropic已明确声明该分类器单独是不足够的。

**Type:** Learn
**Languages:** Python (stdlib, two-stage classifier simulator)
**Prerequisites:** Phase 15 · 01 (Long-horizon agents), Phase 15 · 09 (Coding-agent landscape)
**Time:** ~45 minutes

## 问题

你机器上的自主编程智能体是一个独特的安全类别。攻击面是智能体可以触及的一切——文件系统、网络、凭据、剪贴板、任何浏览器标签页、任何打开的终端。Bruce Schneier等人已公开指出：计算机使用智能体不是聊天机器人的"功能更新"，它们是一种具有新类型风险概况的新工具。

Claude Code的权限系统是Anthropic的答案。不是一个单一的"自主/非自主"开关，而是七种模式跨越一个能力阶梯：plan → default → acceptEdits → … → bypassPermissions。每种模式是速度和每次动作审查之间不同的权衡。自动模式（2026年3月）添加了一个两阶段分类器，对于分类器判定安全的动作将审批从用户的关键路径上移开，同时为分类器标记的动作保留审查层。

工程问题是：这个系统捕获了什么，遗漏了什么，给定的任务实际上需要哪种模式？

## 概念

### 七种权限模式

| 模式 | 行为 | 何时使用 |
|---|---|---|
| `plan` | 智能体提出计划；用户批准整个计划；每个动作在执行前审查 | 不熟悉的任务；接近生产的代码；首次在仓库上使用智能体 |
| `default` | 智能体运行动作；对任何"危险"动作（shell执行、破坏性操作、网络调用）提示用户 | 大多数交互式编程会话 |
| `acceptEdits` | 文件写入自动批准；shell执行和网络调用仍需提示 | 跨多个文件的重构遍 |
| `acceptExec` | 在策展允许列表内的Shell命令自动批准；写入自动批准 | 每个shell命令都是 `npm test` 或类似命令的紧凑内循环 |
| `autoMode` | 两阶段安全分类器；被标记的动作提升到审查 | 在受限工作区中无人值守的长周期运行 |
| `yolo` | 跳过大多数提示；仍运行工具允许/拒绝列表 | 临时沙盒、CI作业、研究脚本 |
| `bypassPermissions` | 批准一切 | 文档注明"仅在您愿意丢弃的临时容器内使用" |

（上述名称匹配公开的Claude Code文档；"yolo"是宽松中间地带的非正式简称。）

### 自动模式一页纸

自动模式（2026年3月24日发布）是第一个将每次动作审批委托给模型的权限模式。结构：

1. **阶段1——单token分类器。** 在每次提议的动作上运行。成本低。返回"安全"或"标记"。设计为与主智能体循环并行运行，这样批准的动作不会在队列中等待。
2. **阶段2——辅助策略/安全审查。** 仅在标记的动作上运行。对该动作、会话的当前状态和声明的任务执行聚焦的策略和安全评估。如果置信度低，升级到用户HITL。

预算控制与分类器并列：

- `max_turns`——会话中的总迭代次数。
- `max_budget_usd`——中止会话的美元上限。
- 每个工具的动作次数限制（不超过N次 `WebFetch` 调用等）。

### 系统捕获什么

- 对工具输入的直白提示注入，其中注入的指令映射到已知危险的动作形状。
- 重复的工具循环——分类器可以看到第N+1个动作与第N个几乎相同，连续五次。
- 在其他方面仅进行文件编辑的会话中，明显超出范围的Shell命令。

### 系统可能遗漏什么

- **微妙的提示注入**，调节行为而不产生单一被标记的动作。间接提示注入不是一个可以完全修补的漏洞（OpenAI准备度负责人，2025年，关于浏览器智能体——见第11课）。
- **语义层面的不当行为。** 每个单独的动作看起来都是安全的，而组合起来的轨迹是有害的。分类器判断动作；它不重新推导用户的意图。
- **通过合法渠道的数据泄露。** 将数据写入你拥有的文件，然后 `git push` 到公共仓库，是一系列允许的动作，但其组合才是问题。

### 研究预览框架

Anthropic将自动模式作为研究预览发布。文档明确指出分类器是一层保护，而非解决方案：用户应将自动模式与预算、允许列表、隔离工作区和轨迹审计（第12-16课）结合使用。预览框架也反映了有记录的评估与部署差距（第1课）——一个通过离线评估的分类器可能在用户上下文模糊的真实会话中表现不同。

### 这个阶梯在你工作流程中的位置

- 不熟悉的任务：以 `plan` 开始。阅读计划比回滚一次糟糕运行的成本更低。
- 已知的重构：`acceptEdits` 节省大量确认点击。
- 无人值守的后台运行：仅在爆炸半径已被测量的工作区内使用 `autoMode`（无凭据、无生产挂载、未经你选择的外传流量）。
- 临时容器：`yolo` / `bypassPermissions` 当且仅当容器及其凭据是可丢弃的时可以接受。

```figure
autonomy-oversight
```

## 运用

`code/main.py` 模拟两阶段分类器。阶段1是在提议动作上的廉价关键词规则；阶段2是较慢的多规则审查器。驱动程序输入一个简短的合成轨迹（安全动作、一次提示注入尝试、一个重复循环），展示分类器在哪里捕获和在哪里遗漏。

## 交付物

`outputs/skill-permission-mode-picker.md` 将任务描述匹配到正确的权限模式、预算上限和所需隔离。

## 练习

1. 运行 `code/main.py`。哪种合成动作类型从未被阶段1标记，但总是被阶段2捕获？哪种被两者都遗漏？

2. 扩展阶段1规则集以捕获特定的已知不良形状（例如 `curl $ATTACKER/exfil`）。在良性动作样本上测量误报率。

3. 阅读Anthropic的"How the agent loop works"文档。列出在 `default` 模式下智能体默认触及的每种外部状态。在无人值守运行 `autoMode` 之前，你需要单独对哪些进行门控？

4. 设计一个24小时无人值守运行预算：`max_turns`、`max_budget_usd`、每个工具的上限、允许列表。为每个数字提供理由。

5. 描述一个轨迹，其中每个单独动作都被阶段1和阶段2批准，但组合行为是不对齐的。（第14课涵盖熔断开关和金丝雀令牌如何处理此问题。）

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|---|---|---|
| 权限模式 | "智能体能做多少" | 控制每次动作审批的七种命名策略之一 |
| plan模式 | "在任何事之前询问" | 智能体编写计划；用户在执行前批准 |
| acceptEdits | "让它写文件" | 文件写入自动批准；shell执行仍需提示 |
| autoMode | "自动审批" | 两阶段安全分类器；被标记的动作升级 |
| bypassPermissions | "完全YOLO" | 批准一切；旨在用于临时容器 |
| 阶段1分类器 | "快速token检查" | 对提议动作的单token规则；并行运行 |
| 阶段2分类器 | "深度审查" | 对被标记动作的思维链推理 |
| 研究预览 | "非GA" | Anthropic对失败模式仍被绘制中的功能的框架 |
## 进一步阅读

- [Anthropic — How the agent loop works](https://code.claude.com/docs/en/agent-sdk/agent-loop) — 权限模式、预算、动作格式。
- [Anthropic — Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — 托管服务执行模型。
- [Anthropic — Claude Code product page](https://www.anthropic.com/product/claude-code) — 功能面及自动模式公告。
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — 塑造分类器判断的基于推理的层。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 长周期权限设计的内部视角。
