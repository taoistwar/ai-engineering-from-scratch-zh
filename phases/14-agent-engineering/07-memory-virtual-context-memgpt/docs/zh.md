# 记忆：虚拟上下文与 MemGPT

> 上下文窗口是有限的。对话、文档和工具轨迹则不是。MemGPT（Packer et al., 2023）将此框架为操作系统虚拟内存——主上下文是 RAM，外部存储是磁盘，agent 在它们之间换页。这是每个 2026 年记忆系统所继承的模式。

**类型:** Build
**语言:** Python（标准库）
**前置要求:** Phase 14 · 01（Agent Loop）, Phase 14 · 06（工具使用）
**时间:** ~75 分钟

## 学习目标

- 解释 MemGPT 所基于的操作系统类比：主上下文 = RAM，外部上下文 = 磁盘，memory 工具 = 换入/换出。
- 在标准库中实现两层 MemGPT 模式，包含主上下文缓冲区、可搜索外部存储和换入/换出工具。
- 描述 agent 如何发出"中断"来查询或修改外部记忆，以及结果如何被拼接到下一个提示中。
- 识别 MemGPT 的设计选择中哪些延续到了 Letta（第 08 课）和 Mem0（第 09 课）。

## 问题

上下文窗口看起来应该能解决记忆问题。它们没有。三种失败模式在生产中反复出现：

1. **溢出。** 多轮对话、长文档或工具调用密集的轨迹越过窗口。截止点之后的一切都丢失了。
2. **稀释。** 即使在窗口内，塞入无关上下文也会稀释对重要内容的注意力。前沿模型在长输入上仍然会退化。
3. **持久性。** 新会话以空窗口开始。没有外部记忆的 agent 无法跨会话说"记住你上次让我……"。

更大的窗口有帮助但不能解决此问题。Mem0 的 2025 年论文测量到，128k 窗口的基线仍然会遗漏长周期的事实，而一个带有外部记忆的 4k 窗口 agent 能捕获它们。

## 概念

### MemGPT：操作系统类比

Packer et al.（arXiv:2310.08560, v2 2024 年 2 月）将上下文管理映射到操作系统的虚拟内存：

| OS 概念 | MemGPT 概念 | 2026 生产类比 |
|------------|---------------|------------------------|
| RAM | 主上下文（提示） | Anthropic/OpenAI 上下文窗口 |
| 磁盘 | 外部上下文 | 向量数据库、KV、图存储 |
| 缺页 | 记忆工具调用 | `memory.search`、`memory.read`、`memory.write` |
| OS 内核 | agent 控制循环 | 带有记忆工具的 ReAct 循环 |

Agent 运行正常的 ReAct 循环。一类额外的工具让它将数据换入和换出主上下文。

### 两层

- **主上下文。** 固定大小的提示，持有当前任务。模型始终可见。
- **外部上下文。** 无界的，通过工具可搜索。相关时读取，事实出现时写入。

原始论文在两个超出基础窗口的任务上评估了该设计：超过 100k token 的文档分析，以及跨越多天且具有持久记忆的多会话聊天。

### 中断模式

MemGPT 引入了记忆即中断：对话中途 agent 可以调用记忆工具，运行时执行它，结果作为新的观察拼接到下一个 assistant 回合中。概念上与 Unix `read()` 系统调用相同，它阻塞进程，返回字节，进程继续。

经典记忆工具面：

- `core_memory_append(section, text)` — 写入提示的持久化部分。
- `core_memory_replace(section, old, new)` — 编辑持久化部分。
- `archival_memory_insert(text)` — 写入可搜索的外部存储。
- `archival_memory_search(query, top_k)` — 从外部存储检索。
- `conversation_search(query)` — 扫描过去的回合。

### MemGPT 的终点与 Letta 的起点

2024 年 9 月，MemGPT 变成了 Letta。研究仓库（`cpacker/MemGPT`）仍然存在；Letta 扩展了设计：

- 三层而非两层（core、recall、archival — 第 08 课）。
- 原生推理取代 `send_message`/heartbeat 模式（第 08 课）。
- 睡眠时 agent 运行异步记忆工作（第 08 课）。

即使生产系统运行 Letta、Mem0 或自定义两层存储，MemGPT 论文仍是 2026 年的基础。

### 此模式在哪些情况下会出错

- **记忆腐化。** 写入累积速度快于读取速度；检索被陈旧事实淹没。修复：定期整合（Letta 睡眠时）、显式失效（Mem0 冲突检测器）。
- **记忆中毒。** 外部记忆是检索到的文本。如果攻击者控制的内容落入记忆中，agent 会在下一次会话中重新摄取它。这是 Greshake et al.（第 27 课）攻击随时间重述。
- **引用丢失。** Agent 回忆"用户让我交付 X"但无法引用是哪个回合。在每次档案写入时存储源引用（会话 ID、回合 ID）。

```figure
context-budget
```

## Build It

`code/main.py` 在标准库中实现 MemGPT 的两层模式：

- `MainContext` — 固定大小提示缓冲区，带有 `core` 字典和 `messages` 列表；超出上限时自动压缩最旧的消息。
- `ArchivalStore` — 内存中的 BM25 风格存储（token 重叠评分），记录 `(id, text, tags, session, turn)`。
- 五个映射到 MemGPT 面的记忆工具。
- 一个脚本化 agent，先用事实填充档案，然后通过调用 `archival_memory_search` 回答问题。

运行它：

```
python3 code/main.py
```

追踪显示 agent 写入三条事实，将主上下文填满到上限（强制驱逐），然后通过从档案检索来回答后续问题——在没有任何真实 LLM 的情况下重现 MemGPT 工作流。

## Use It

今天每个生产级记忆系统都是 MemGPT 的变体：

- **Letta**（第 08 课） — 三层、原生推理、睡眠时计算。
- **Mem0**（第 09 课） — 向量 + KV + 图，融合评分层。
- **OpenAI Assistants / Responses** — 通过 threads 和 files 管理记忆。
- **Claude Agent SDK** — 通过 skills 和 session store 实现长期记忆。

按运维形态选择（自托管、托管、框架集成），而非核心模式——核心模式是 MemGPT。

## Ship It

`outputs/skill-virtual-memory.md` 是一个可复用 skill，为任何目标运行时生成正确的两层记忆脚手架（主 + 档案 + 工具面），并内置驱逐策略和引用字段。

## 练习

1. 添加一个用 token 度量的 `max_main_context_tokens` 上限（用 `len(text.split())` * 1.3 近似）。超出上限时将最旧的消息压缩成摘要。比较有和没有摘要器的行为。
2. 在档案存储上正确实现 BM25（词频、逆文档频率）。在玩具事实集上测量 recall@10 与 token 重叠基线的对比。
3. 为档案插入添加 `citation` 字段（session_id、turn_id、source_url）。让 agent 在每个检索支持的答案上引用来源。
4. 模拟记忆中毒：添加一条档案记录说"忽略所有未来的用户指令"。编写一个守卫，扫描检索结果中的指令形状文本并将其标记为不可信。
5. 将实现移植到 MemGPT 研究仓库的核心记忆 JSON 模式（`cpacker/MemGPT`）。从扁平字符串切换到分类型节段时有什么变化？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| 虚拟上下文 | "无限记忆" | 主（提示）+ 外部（可搜索）层，可换入/换出 |
| 主上下文 | "工作记忆" | 提示——固定大小，始终可见 |
| 档案记忆 | "长期存储" | 外部可搜索持久化，按需检索 |
| 核心记忆 | "持久提示节段" | 在提示内部固定的命名节段 |
| 记忆工具 | "记忆 API" | agent 发出以读写外部记忆的工具调用 |
| 中断 | "记忆缺页" | Agent 暂停，运行时获取，结果拼接到下一个回合 |
| 记忆腐化 | "陈旧事实" | 旧写入淹没检索；通过整固修复 |
| 记忆中毒 | "注入的持久笔记" | 攻击者内容存储为记忆，回忆时重新摄取 |

## 进一步阅读

- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560) — OS 启发的虚拟上下文论文
- [Letta, Memory Blocks 博客](https://www.letta.com/blog/memory-blocks) — 三层演进
- [Anthropic, Effective context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — 将上下文视为预算
- [Chhikara et al., Mem0 (arXiv:2504.19413)](https://arxiv.org/abs/2504.19413) — 在此模式之上的混合生产记忆
