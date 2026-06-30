# 记忆块与睡眠时计算（Letta）

> MemGPT 在 2024 年变成了 Letta。2026 年的演进增加了两个思想：模型可以直接编辑的离散功能记忆块，以及一个在主 agent 空闲时异步整合记忆的睡眠时 agent。这就是你将记忆扩展到一次对话之外的方式。

**类型:** Build
**语言:** Python（标准库）
**前置要求:** Phase 14 · 07（MemGPT）
**时间:** ~75 分钟

## 学习目标

- 说出 Letta 使用的三个记忆层（core、recall、archival）以及每个的作用。
- 解释记忆块模式：Human 块、Persona 块和用户定义块作为一等类型对象。
- 描述什么是睡眠时计算，为什么它在关键路径之外，以及为什么它可以运行比主 agent 更强的模型。
- 实现一个脚本化的双 agent 循环，其中主 agent 提供响应，睡眠时 agent 在回合之间整合块。

## 问题

MemGPT（第 07 课）解决了虚拟内存的控制流。三个生产问题出现了：

1. **延迟。** 每个记忆操作都在关键路径上。如果 agent 必须在用户等待时修剪、总结或协调，尾部延迟会爆炸。
2. **记忆腐化。** 写入积累。矛盾的事实保留。检索被陈旧内容淹没。
3. **结构丢失。** 扁平档案存储无法表达"Human 块总是在提示中；Persona 块总是在提示中；Task 块按会话切换。"

Letta（letta.com）是 2026 年的重写。记忆块使结构显式化；睡眠时计算将整合移出关键路径。

## 概念

### 三层

| 层 | 范围 | 所在位置 | 写入者 |
|------|-------|----------------|------------|
| Core | 始终可见 | 在主提示内部 | Agent 工具调用 + 睡眠时重写 |
| Recall | 对话历史 | 可检索 | 自动回合记录 |
| Archival | 任意事实 | 向量 + KV + 图 | Agent 工具调用 + 睡眠时摄取 |

Core 是 MemGPT 的核心。Recall 是对话缓冲区及其被驱逐的尾部。Archival 是外部存储。此分离清理了 MemGPT 的两层过载。

### 记忆块

块是核心层的一个类型化、持久化、可编辑的节段。原始 MemGPT 论文定义了两类：

- **Human 块** — 关于用户的事实（姓名、角色、偏好、目标）。
- **Persona 块** — agent 的自我概念（身份、语气、约束）。

Letta 泛化到任意用户定义块：一个用于当前目标的 `Task` 块，一个用于代码库事实的 `Project` 块，一个用于硬约束的 `Safety` 块。每个块有 `id`、`label`、`value`、`limit`（字符上限）、`description`（这样模型知道何时编辑它）。

块通过工具面可编辑：

- `block_append(label, text)`
- `block_replace(label, old, new)`
- `block_read(label)`
- `block_summarize(label)` — 压缩接近其上限的块。

### 睡眠时计算

2025 年 Letta 的增加：在关键路径之外的后台运行第二个 agent。睡眠时 agent 处理对话转录和代码库上下文，将 `learned_context` 写入共享块，并整合或失效档案记录。

产生的属性：

- **无延迟成本。** 主响应不等待记忆操作。
- **允许使用更强的模型。** 睡眠时 agent 可以是更昂贵、更慢的模型，因为它不受延迟约束。
- **自然整合窗口。** 在用户不等待时去重、总结、失效矛盾的事实。

形状匹配人类的工作方式：你做任务，你睡一觉，长期记忆在一夜间安顿下来。

### Letta V1 与原生推理

Letta V1（`letta_v1_agent`，2026 年）废弃了 `send_message`/heartbeat 和内联 `Thought:` token，改为使用原生推理。Responses API（OpenAI）和带扩展思维的 Messages API（Anthropic）在单独通道上发出推理，跨回合传递（在生产中跨提供商加密）。控制循环仍然是 ReAct。思维轨迹是结构化的，而非提示形状的。

### 此模式在哪些情况下会出错

- **块膨胀。** 无限 `block_append` 迅速达到上限。在写入超过上限之前连接块摘要器。
- **静默漂移。** 睡眠时 agent 重写一个块，主 agent 从未注意到。对块进行版本化，并在追踪中显示差异。
- **中毒的整固。** 睡眠时 agent 将攻击者可接触的内容处理到核心中。第 27 课也适用于睡眠时面。

## Build It

`code/main.py` 实现：

- `Block` — id、label、value、limit、description。
- `BlockStore` — CRUD + `near_limit(label)` 辅助函数。
- 两个脚本化 agent —— `PrimaryAgent` 提供一个回合，`SleepTimeAgent` 在回合之间进行整固。
- 一个显示三轮对话的追踪，包含块写入，外加一个睡眠时通行，它总结一个块并使一个陈旧事实失效。

运行它：

```
python3 code/main.py
```

转录显示分离：主回合适快且产生原始写入；睡眠通行压缩和清理。

## Use It

- **Letta**（letta.com）用于参考实现。自托管或托管云。
- **Claude Agent SDK skills** 作为块形状的知识——skill 是一个命名的、版本化的、可检索的指令块，agent 按需加载。
- **自定义构建** 用于希望控制存储后端的团队。使用 Letta API 契约以便日后可以迁移。

## Ship It

`outputs/skill-memory-blocks.md` 为任何运行时生成 Letta 形状的块系统，带有睡眠时 hook，包括安全规则和引用连接。

## 练习

1. 添加 `block_summarize` 工具，在 `near_limit` 返回 true 时用模型生成的摘要替换块的值。哪个触发阈值能在最小化摘要调用和块溢出之间取得平衡？
2. 实现档案的睡眠时去重：文本有 >90% token 重叠的两条记录合并为一条。仅在睡眠通行中做，从不在关键路径上。
3. 版本化块。每次写入记录旧值和差异。暴露 `block_history(label)` 以便运维人员调试"为什么 agent 忘记了 X"。
4. 将睡眠时 agent 视为不可信的写入者。当它们触及 Persona 或 Safety 块时，在提交前要求第二个 agent 审查。
5. 将示例移植到使用 Letta API（`letta_v1_agent`）。块模式有什么变化？原生推理如何改变追踪形状？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| 记忆块 | "可编辑提示节段" | 核心记忆的类型化、持久化、LLM 可编辑的节段 |
| Human 块 | "用户记忆" | 关于用户的事实，固定在核心中 |
| Persona 块 | "Agent 身份" | 自我概念、语气、约束，固定在核心中 |
| 睡眠时计算 | "异步记忆工作" | 第二个 agent 在关键路径之外做整固 |
| Core / Recall / Archival | "层" | 三层记忆分离：始终可见 / 对话 / 外部 |
| 块上限 | "Cap" | 每个块的字符限制；强制摘要 |
| 原生推理 | "思维通道" | 提供者级别的推理输出，而非提示级别的 `Thought:` |
| 学习到的上下文 | "睡眠输出" | 睡眠时 agent 写入共享块的事实 |

## 进一步阅读

- [Letta, Memory Blocks 博客](https://www.letta.com/blog/memory-blocks) — 块模式
- [Letta, Sleep-time Compute 博客](https://www.letta.com/blog/sleep-time-compute) — 异步整固
- [Letta, 重构 Agent Loop](https://www.letta.com/blog/letta-v1-agent) — 原生推理重写
- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560) — 起源
