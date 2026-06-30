# Claude Agent SDK：子 Agent 与会话存储

> Claude Agent SDK 是 Claude Code harness 的库形式。内置工具、用于上下文隔离的子 agent、钩子、W3C 追踪传播、会话存储对等。Claude Managed Agents 是用于长时间运行异步工作的托管替代方案。

**类型：** 学习 + 构建
**语言：** Python（标准库）
**前置条件：** 第 14 阶段 · 01（Agent 循环），第 14 阶段 · 10（技能库）
**时间：** ~75 分钟

## 学习目标

- 解释 Anthropic Client SDK（原始 API）与 Claude Agent SDK（harness 形态）之间的区别。
- 描述子 agent——并行化和上下文隔离——以及何时使用它们。
- 列举 Python SDK 的会话存储接口（`append`、`load`、`list_sessions`、`delete`、`list_subkeys`）以及 `--session-mirror` 的作用。
- 实现一个标准库 harness，具有内置工具、具有隔离上下文的子 agent 生成、生命周期钩子和会话存储。

## 问题

原始 LLM API 给你一次往返。生产 agent 需要工具执行、MCP 服务器、生命周期钩子、子 agent 生成、会话持久化、追踪传播。Claude Agent SDK 将此形态作为库提供——与 Claude Code 使用的相同 harness，公开用于自定义 agent。

## 概念

### Client SDK vs Agent SDK

- **Client SDK（`anthropic`）。** 原始 Messages API。你拥有循环、工具、状态。
- **Agent SDK（`claude-agent-sdk`）。** 内置工具执行、MCP 连接、钩子、子 agent 生成、会话存储。Claude Code 循环作为库。

### 内置工具

SDK 开箱即用提供 10+ 工具：文件读/写、shell、grep、glob、网页抓取等。自定义工具通过标准工具 schema 接口注册。

### 子 Agent

Anthropic 文档说明的两个目的：

1. **并行化。** 并发运行独立工作。"为这 20 个模块分别找到测试文件"是 20 个并行子 agent 任务。
2. **上下文隔离。** 子 agent 使用自己的上下文窗口；只有结果返回给编排者。编排者的预算得以保留。

Python SDK 最近新增：`list_subagents()`、`get_subagent_messages()` 用于读取子 agent 转录。

### 会话存储

与 TypeScript 的协议对等：

- `append(session_id, message)` — 添加一轮。
- `load(session_id)` — 恢复对话。
- `list_sessions()` — 枚举。
- `delete(session_id)` — 级联到子 agent 会话。
- `list_subkeys(session_id)` — 列出子 agent 键。

`--session-mirror`（CLI 标志）在流式传输时将转录镜像到外部文件，用于调试。

### 钩子

你可以注册的生命周期钩子：

- `PreToolUse`、`PostToolUse` — 门控或审计工具调用。
- `SessionStart`、`SessionEnd` — 设置和清理。
- `UserPromptSubmit` — 在模型看到之前对用户输入进行操作。
- `PreCompact` — 在上下文压缩之前运行。
- `Stop` — agent 退出时清理。
- `Notification` — 旁路告警。

钩子是 pro-workflow（第 14 阶段课程参考）和类似系统添加横切行为的方式。

### W3C 追踪上下文

调用方上的活跃 OTel span 通过 W3C 追踪上下文头传播到 CLI 子进程中。整个多进程追踪在后端显示为一个追踪。

### Claude Managed Agents

托管替代方案（beta 头 `managed-agents-2026-04-01`）。长时间运行异步工作，内置提示词缓存，内置压缩。用托管基础设施换取控制权。

### 这种模式可能出错的地方

- **子 agent 过度生成。** 为 100 个小任务生成 100 个子 agent。开销占主导。改为批量处理。
- **钩子蔓延。** 每个团队都添加钩子；启动时间膨胀。每季度审查钩子。
- **会话膨胀。** 会话累积；大小增长。使用 `list_sessions` + 过期策略。

## 构建它

`code/main.py` 在标准库中实现了 SDK 形态：

- `Tool`、`ToolRegistry` 具有内置 `read_file`、`write_file`、`list_dir`。
- `Subagent` — 私有上下文，隔离运行，返回结果。
- `SessionStore` — append、load、list、delete、list_subkeys。
- `Hooks` — `pre_tool_use`、`post_tool_use`、`session_start`、`session_end`。
- 演示：主 agent 并行生成 3 个子 agent（各自隔离），聚合结果，持久化会话。

运行它：

```
python3 code/main.py
```

跟踪显示子 agent 上下文隔离（编排者上下文大小保持有限）、钩子执行和会话持久化。

## 使用它

- **Claude Agent SDK** 用于想要 Claude Code harness 形态的 Claude 优先产品。
- **Claude Managed Agents** 用于托管的长运行异步工作。
- **OpenAI Agents SDK**（第 16 课）用于 OpenAI 优先的对应物。
- **LangGraph + 自定义工具** 如果你想要图形化状态机。

## 交付它

`outputs/skill-claude-agent-scaffold.md` 构建一个 Claude Agent SDK 应用，包含子 agent、钩子、会话存储、MCP 服务器挂载和 W3C 追踪传播。

## 练习

1. 添加一个子 agent 生成器，将 20 个任务批处理为每组 5 个的并行子 agent。测量编排者上下文大小与每个任务一个子 agent 的区别。
2. 实现一个 `PreToolUse` 钩子，对 `write_file` 调用进行速率限制（每个会话每分钟 5 次）。追踪行为。
3. 接入 `list_subkeys` 渲染子 agent 树。深层嵌套是什么样子？
4. 将玩具移植到真正的 `claude-agent-sdk` Python 包。工具注册有什么变化？
5. 阅读 Claude Managed Agents 文档。你何时会从自托管切换到托管？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| Agent SDK | "Claude Code 作为库" | Harness 形态：工具、MCP、钩子、子 agent、会话存储 |
| Subagent | "子 agent" | 独立上下文，自有预算；结果向上冒泡 |
| 会话存储（Session store） | "对话数据库" | 持久化、加载、列出、删除轮次，包含子 agent 级联 |
| Hook | "生命周期回调" | 前置/后置工具、会话、提示词提交、压缩、停止 |
| W3C 追踪上下文（W3C trace context） | "跨进程追踪" | 父 span 传播到 CLI 子进程 |
| Managed Agents | "托管 harness" | Anthropic 托管的长时间运行异步工作 |
| `--session-mirror` | "转录镜像" | 在流式传输时将会话轮次写入外部文件 |
| MCP 服务器 | "工具表面" | 挂载到 agent 的外部工具/资源源 |

## 进一步阅读

- [Claude Agent SDK 概览](https://platform.claude.com/docs/en/agent-sdk/overview) — Claude Code 的库形式
- [Anthropic, Building agents with the Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) — 生产模式
- [Claude Managed Agents 概览](https://platform.claude.com/docs/en/managed-agents/overview) — 托管替代方案
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — 对应物
