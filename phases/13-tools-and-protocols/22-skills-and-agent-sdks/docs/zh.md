# 技能和代理 SDK — Anthropic Skills、AGENTS.md、OpenAI Apps SDK

> MCP 说"存在什么工具"。Skills 说"如何执行一个任务"。2026 年的技术栈将两者分层叠加。Anthropic 的 Agent Skills（开放标准，2025 年 12 月）以 SKILL.md 形式发布，支持渐进式披露。OpenAI 的 Apps SDK 是 MCP 加上组件元数据。AGENTS.md（现存于 60,000 多个仓库中）位于仓库根目录，作为项目级代理上下文。本课程点明每种覆盖什么，并构建一个可在不同代理间移植的最小 SKILL.md + AGENTS.md 包。

**Type:** Learn
**Languages:** Python（stdlib，SKILL.md 解析器和加载器）
**Prerequisites:** Phase 13 · 07（MCP 服务器）
**Time:** ~45 分钟

## 学习目标

- 区分三个层次：AGENTS.md（项目上下文）、SKILL.md（可复用知识）、MCP（工具）。
- 编写包含 YAML 前置元数据和渐进式披露的 SKILL.md。
- 以文件系统风格将技能加载到代理运行时。
- 将技能与 MCP 服务器和 AGENTS.md 组合，使一个包能在 Claude Code、Cursor 和 Codex 中运行。

## 问题

一位工程师将发布说明编写工作流提炼为多步骤提示词："阅读最近合并的 PR。按领域分组。逐一总结。按照团队风格撰写变更日志条目。发布到 Slack 草稿频道。"他们将其放入团队的 Notion 文档。

现在他们想在 Claude Code、Cursor 和 Codex CLI 中使用此工作流。每个代理加载指令的方式都不同：Claude Code 的斜杠命令、Cursor 的规则、Codex 的 `.codex.md`。工程师将工作流复制三份并维护三份副本。

AGENTS.md 和 SKILL.md 一起解决了这个问题：

- **AGENTS.md** 位于仓库根目录。每个兼容的代理在会话启动时读取它。"这个项目如何运作？约定是什么？哪些命令运行测试？"
- **SKILL.md** 是一个可移植包：YAML 前置元数据（名称、描述）+ markdown 正文 + 可选资源。支持技能的代理按需按名称加载它们。
- **MCP**（第 13 阶段 · 06-14）处理技能需要调用的工具。

三个层次，一个可移植制品。

## 概念

### AGENTS.md

2025 年末推出，截至 2026 年 4 月被超过 60,000 个仓库采用。一个位于仓库根目录的文件。格式：

```markdown
# 项目：my-service

## 约定
- 使用严格模式的 TypeScript。
- Python 端使用 Pydantic 做模型。
- 用 `pnpm test` 运行测试。

## 构建和运行
- `pnpm dev` 启动本地开发服务器。
- `pnpm build` 构建生产包。
```

代理在会话启动时读取它，并使用它来为该项目校准行为。2026 年每个编码代理都支持 AGENTS.md：Claude Code、Cursor、Codex、Copilot Workspace、opencode、Windsurf、Zed。

### SKILL.md 格式

Anthropic 的 Agent Skills（作为开放标准于 2025 年 12 月发布）：

```markdown
---
name: release-notes-writer
description: 遵循此项目风格为最新合并的 PR 撰写变更日志条目。
---

# 发布说明编写器

被调用时，运行以下步骤：

1. 列出从上一个标签以来合并的 PR。使用 `gh pr list --base main --state merged`。
2. 按标签分组：feature、fix、chore、docs。
3. 对每组中的每个 PR，撰写一行：`- <标题> （#<编号>）`。
4. 草拟发布说明并将其暂存到 CHANGELOG.md。

如果用户说"ship"，运行 `git tag vX.Y.Z` 和 `gh release create`。

## 备注

- 绝不包含没有 PR 的提交。
- 在公开变更日志中跳过"chore"条目。
```

前置元数据声明技能的身份。正文是技能加载时显示给模型的提示词。

### 渐进式披露

技能可以引用代理仅在需要时才会获取的子资源。示例：

```
skills/
  release-notes-writer/
    SKILL.md
    style-guide.md
    template.md
    scripts/
      generate.sh
```

SKILL.md 说"关于风格规则，请参阅 style-guide.md"。代理仅在技能活跃运行时拉取 style-guide.md。这避免了用模型可能不需要的细节膨胀提示词。

### 文件系统发现

代理运行时扫描已知目录以寻找 SKILL.md 文件：

- `~/.anthropic/skills/*/SKILL.md`
- 项目 `./skills/*/SKILL.md`
- `~/.claude/skills/*/SKILL.md`

加载按文件夹名称和前置元数据的 `name` 进行。Claude Code、Anthropic Claude Agent SDK 和 SkillKit（跨代理）都遵循此模式。

### Anthropic Claude Agent SDK

`@anthropic-ai/claude-agent-sdk`（TypeScript）和 `claude-agent-sdk`（Python）在会话启动时加载技能，并将它们作为可调用的"agents"暴露在运行时中。代理循环在用户调用技能时分派给技能。

### OpenAI Apps SDK

2025 年 10 月推出；直接构建在 MCP 之上。将 OpenAI 之前的 Connectors 和 Custom GPT Actions 统一在一个开发界面之下。一个 Apps SDK 应用是：

- 一个 MCP 服务器（工具、资源、提示词）。
- 加上 ChatGPT UI 的组件元数据。
- 加上一个可选的 MCP Apps `ui://` 资源用于交互界面。

相同协议，更丰富的 UX。

### 通过 SkillKit 实现的跨代理可移植性

像 SkillKit 和类似的跨代理分发层这样的工具将单个 SKILL.md 翻译为 32 个以上 AI 代理的原生格式（Claude Code、Cursor、Codex、Gemini CLI、OpenCode 等）。一个信源；许多消费者。

### 三层技术栈

| 层次 | 文件 | 加载时机 | 目的 |
|------|------|----------|------|
| AGENTS.md | 仓库根目录 | 会话启动时 | 项目级约定 |
| SKILL.md | skills 目录 | 技能被调用时 | 可复用工作流 |
| MCP 服务器 | 外部进程 | 需要工具时 | 可调用操作 |

三者组合：代理在会话启动时读取 AGENTS.md，用户调用一个技能，技能指令包含 MCP 工具调用，代理通过 MCP 客户端分发。

## 使用

`code/main.py` 提供一个标准库 SKILL.md 解析器和加载器。它在 `./skills/` 下发现技能，解析 YAML 前置元数据和 markdown 正文，并生成按技能名称键控的字典。然后它模拟一个按名称调用 `release-notes-writer` 的代理循环。

关注要点：

- 使用最小标准库解析器解析 YAML 前置元数据（无 `pyyaml` 依赖）。
- 技能正文逐字存储；代理在调用时将其前置到系统提示词中。
- 通过 `read_subresource` 函数演示渐进式披露，该函数按需拉取引用的文件。

## 交付物

本课程产出 `outputs/skill-agent-bundle.md`。给定一个工作流，该技能生成组合的 SKILL.md + AGENTS.md + MCP 服务器蓝图包，可跨代理移植。

## 练习

1. 运行 `code/main.py`。在 `skills/` 下添加第二个技能并确认加载器将其拾取。

2. 为此课程仓库编写一个 AGENTS.md。包括测试命令、风格约定和第 13 阶段的心智模型。

3. 将你团队内部文档中的一个多步骤工作流移植到 SKILL.md。验证其在 Claude Code 中加载。

4. 手工将技能翻译为 Cursor 和 Codex 的原生规则格式。计算格式间差异——这就是 SkillKit 自动化的翻译界面。

5. 阅读 Anthropic Agent Skills 博客文章。找出本课程加载器未覆盖的 Claude Agent SDK 中的一个功能。（提示：代理子调用。）

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|----------|
| SKILL.md | "技能文件" | YAML 前置元数据加 markdown 正文，由代理运行时加载 |
| AGENTS.md | "仓库根目录代理上下文" | 会话启动时读取的项目级约定文件 |
| 渐进式披露 | "延迟加载子资源" | 技能正文引用仅在需要时拉取的文件 |
| 前置元数据 | "顶部的 YAML 块" | 位于 `---` 分隔符内的元数据（名称、描述） |
| Claude Agent SDK | "Anthropic 的技能运行时" | `@anthropic-ai/claude-agent-sdk`，加载技能并路由 |
| OpenAI Apps SDK | "MCP + 组件元数据" | 基于 MCP 的 OpenAI 开发界面，加上 ChatGPT UI 钩子 |
| 技能发现 | "文件系统扫描" | 遍历已知目录以寻找 SKILL.md，按名称键控 |
| 跨代理可移植性 | "一个技能，许多代理" | 通过 SkillKit 风格的工件将一个 SKILL.md 翻译到 32+ 个代理 |
| 代理技能 | "可移植知识" | MCP 工具概念之外的可复用任务模板 |
| Apps SDK | "MCP 加上 ChatGPT UI" | Connectors 和 Custom GPT 在 MCP 上统一 |

## 进一步阅读

- [Anthropic — Agent Skills 公告](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) — 2025 年 12 月发布
- [Anthropic — Agent Skills 文档](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview) — SKILL.md 格式参考
- [OpenAI — Apps SDK](https://developers.openai.com/apps-sdk) — 基于 MCP 的 ChatGPT 开发平台
- [agents.md](https://agents.md/) — AGENTS.md 格式和采用列表
- [Anthropic — anthropics/skills GitHub](https://github.com/anthropics/skills) — 官方技能示例
