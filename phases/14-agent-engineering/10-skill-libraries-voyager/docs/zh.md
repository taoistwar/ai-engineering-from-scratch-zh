# 技能库与终身学习（Voyager）

> Voyager（Wang et al., TMLR 2024）将可执行代码视为技能。技能是命名的、可检索的、可组合的，并通过环境反馈进行精炼。这是 Claude Agent SDK skills、skillkit 以及 2026 年技能库模式的参考架构。

**类型:** Build
**语言:** Python（标准库）
**前置要求:** Phase 14 · 07（MemGPT）, Phase 14 · 08（Letta 块）
**时间:** ~75 分钟

## 学习目标

- 说出 Voyager 的三个组件——自动课程、技能库、迭代提示——以及每个的作用。
- 解释为什么 Voyager 将动作空间设为代码而非原始命令。
- 实现一个标准库技能库，包含注册、检索、组合和失败驱动的精炼。
- 将 Voyager 的模式映射到 2026 年的 Claude Agent SDK skills 和 skillkit 生态系统。

## 问题

在每会话中从零重建每个能力的 agent 做错了三件事：

1. **浪费 token。** 每个任务重新激发相同的推理。
2. **丢失进展。** 在会话 A 中学到的修正没有转移到会话 B。
3. **在长周期组合上失败。** 复杂任务需要能力层次结构；一次性的提示无法表达它们。

Voyager 的答案：将每个可复用能力视为存储在库中的命名代码块，可通过相似性检索，可与其他技能组合，并通过执行反馈精炼。

## 概念

### 三个组件

Voyager（arXiv:2305.16291）围绕以下三个方面构建 agent：

1. **自动课程。** 一个好奇心驱动的提议者根据 agent 的当前技能集和环境状态选择下一个任务。探索是自底向上的。
2. **技能库。** 每个技能是可执行代码。新技能在任务成功时被添加。技能通过查询到描述的相似性来检索。
3. **迭代提示机制。** 失败时，agent 接收执行错误、环境反馈和自我验证输出，然后精炼技能。

Minecraft 评估（Wang et al., 2024）：比基线多 3.3 倍独特物品、8.5 倍快速制作石制工具、6.4 倍快速制作铁制工具、2.3 倍更长地图穿越。数字是 Minecraft 特定的，但模式是可转移的。

### 动作空间 = 代码

大多数 agent 发出原始命令。Voyager 发出 JavaScript 函数。一个技能是：

```
async function craftIronPickaxe(bot) {
  await mineIron(bot, 3);
  await mineStick(bot, 2);
  await placeCraftingTable(bot);
  await craft(bot, 'iron_pickaxe');
}
```

从子技能组合而成。以描述和嵌入为键存储。作为程序而非提示检索。

这就是 2026 年 Claude Agent SDK skill：一个命名的、可检索的代码块加上 agent 按需加载的指令。

### 技能检索

新任务"制作钻石镐"。Agent：

1. 嵌入任务描述。
2. 查询技能库中的 top-k 相似技能。
3. 检索 `craftIronPickaxe`、`mineDiamond`、`placeCraftingTable` 等。
4. 从检索到的原语 + 新逻辑组合新技能。

这是 MCP 资源（Phase 13）和 Agent SDK skills 实现的模式：对知识/代码面的检索，作用域限定到当前任务。

### 迭代精炼

Voyager 的反馈循环：

1. Agent 编写一个技能。
2. 技能在环境中运行。
3. 返回三种信号之一：`success`、`error`（带有堆栈跟踪）、`self-verification failure`。
4. Agent 使用该信号作为上下文重写技能。
5. 循环直到成功或达到最大轮数。

这是 Self-Refine（第 05 课）应用于代码生成，并配有环境锚定的验证。CRITIC（第 05 课）是同样的模式，用外部工具作为验证器。

### 课程与探索

Voyager 的课程模块根据 agent 已有的和尚未完成的事情提议任务，如"在湖边建造一个庇护所"。提议者使用环境状态 + 技能清单选择略高于当前能力的任务——探索的最佳点。

对生产级 agent 而言，这转化为一个"缺什么"操作符：给定当前技能库和一个领域，我们还缺哪些技能？团队通常手动实现这个作为课程审查。

### 此模式在哪些情况下会出错

- **技能库腐化。** 同一技能以略微不同的描述添加了 10 次。在写入时添加去重；检索只返回一个。
- **组合技能漂移。** 父技能依赖一个已被精炼的子技能。对技能版本化；固定到 v1 的父技能不会魔法般获得 v3。
- **检索质量。** 对技能描述的向量检索随库增长超过几百而退化。补充标签过滤和硬约束（"仅 `category=tooling` 的技能"）。

## Build It

`code/main.py` 实现一个标准库技能库：

- `Skill` — name、description、code（作为字符串）、version、tags、dependencies。
- `SkillLibrary` — register、search（token 重叠）、compose（依赖的拓扑排序）、refine（更新时版本号递增）。
- 一个脚本化 agent，注册三个原语技能，组合第四个，遇到一次失败，然后精炼。

运行它：

```
python3 code/main.py
```

追踪显示库写入、检索、组合、一次执行失败和一次 v2 精炼——Voyager 的完整循环端到端。

## Use It

- **Claude Agent SDK skills**（Anthropic） — 2026 年的参考：每个 skill 有描述、代码和指令；在 agent 会话期间按需加载。
- **skillkit**（npm: skillkit） — 跨 agent 技能管理，适用于 32+ AI 编码 agent。
- **自定义技能库** — 领域特定（数据 agent 的 SQL 技能、基础设施 agent 的 Terraform 技能）。Voyager 模式可向下扩展。
- **OpenAI Agents SDK `tools`** — 在低端；每个工具是一个轻量级技能。

## Ship It

`outputs/skill-skill-library.md` 为任何目标运行时生成 Voyager 形状的技能库，内置注册、检索、版本化和精炼。

## 练习

1. 为 `compose()` 添加一个依赖循环检测器。当技能 A 依赖 B 而 B 依赖 A 时会发生什么？错误还是警告？
2. 实现每个技能的版本固定。当父技能组合子技能 `crafting@1` 时，对 `crafting@2` 的精炼不能静默升级父技能。
3. 将 token 重叠检索替换为 sentence-transformers 嵌入（或 BM25 标准库实现）。在 50 技能的玩具库上测量 retrieval@5。
4. 添加一个"课程"agent：给定当前库和领域描述，提议 5 个缺失的技能。每周调用它。
5. 阅读 Anthropic 的 Claude Agent SDK skill 文档。将玩具库移植到 SDK 的 skill 模式。可发现性上有什么变化？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| 技能 | "可复用能力" | 命名的代码块 + 描述，可通过相似性检索 |
| 技能库 | "Agent 的操作手册记忆" | 技能的持久化存储，可搜索和可组合 |
| 课程 | "任务提议者" | 由当前能力差距驱动的自底向上目标生成器 |
| 组合 | "技能 DAG" | 技能调用技能；执行时拓扑排序 |
| 迭代精炼 | "自我修正循环" | 环境反馈 + 错误 + 自我验证折叠回下一个版本 |
| 动作空间即代码 | "程序化动作" | 发出函数而非原始命令，实现时间上扩展的行为 |
| 写入时去重 | "技能归并" | 接近重复的描述归并为一个规范技能 |

## 进一步阅读

- [Wang et al., Voyager (arXiv:2305.16291)](https://arxiv.org/abs/2305.16291) — 原始技能库论文
- [Claude Agent SDK 概览](https://platform.claude.com/docs/en/agent-sdk/overview) — skills 作为 2026 年的产品化
- [Anthropic, Building agents with the Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) — skills 和 subagent 的实践
- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) — Voyager 底层的精炼循环
