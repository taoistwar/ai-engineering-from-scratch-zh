# 实践项目 01 — 终端原生编程智能体

> 到 2026 年，编程智能体的形态已经定型。一个 TUI 控制框架、一个带状态的计划、一个沙箱化的工具界面、以及一个"计划、执行、观察、恢复"的循环。Claude Code、Cursor 3 和 OpenCode 从远距离看起来完全一样。这个实践项目要求你从头到尾构建一个——CLI 输入，Pull Request 输出——并在 SWE-bench Pro 上与 mini-swe-agent 和 Live-SWE-agent 对比测量。你将学到为什么难点不在于模型调用，而在于工具循环、沙箱以及 50 轮运行的成本上限。

**类型:** 实践项目
**语言:** TypeScript / Bun（控制框架），Python（评估脚本）
**前置条件:** 阶段 11（LLM 工程），阶段 13（工具与协议），阶段 14（智能体），阶段 15（自主系统），阶段 17（基础设施）
**涉及的阶段:** P0 · P5 · P7 · P10 · P11 · P13 · P14 · P15 · P17 · P18
**时间:** 35 小时

## 问题

编程智能体在 2026 年成为占主导地位的 AI 应用类别。Claude Code（Anthropic）、Cursor 3 with Composer 2 and Agent Tabs（Cursor）、Amp（Sourcegraph）、OpenCode（112k stars）、Factory Droids 以及 Google Jules 都推出了相同架构的变体：一个终端控制框架、一个权限控制的工具界面、一个沙箱以及围绕前沿模型构建的"计划-执行-观察"循环。前沿很窄——Live-SWE-agent 在 SWE-bench Verified 上使用 Opus 4.5 达到了 79.2%——但工程工艺很广。大多数故障模式不是模型错误，而是工具循环的不稳定性、上下文污染、失控的 token 成本以及破坏性的文件系统操作。

你不能从外部推理这些智能体。你必须构建一个，观察循环在第 47 轮因为 ripgrep 返回 8MB 的匹配结果而崩溃，然后重建截断层。这就是这个实践项目的意义。

## 概念

控制框架有四个层面。**计划**维护一个类似于 TodoWrite 风格的状态对象，模型在每一轮重写它。**执行**分派工具调用（读取、编辑、运行、搜索、git）。**观察**捕获 stdout/stderr/退出码，截断并将摘要反馈回去。**恢复**处理工具错误，而不会撑爆上下文窗口或无限循环。2026 年的形态新增了一个东西：**钩子**。`PreToolUse`、`PostToolUse`、`SessionStart`、`SessionEnd`、`UserPromptSubmit`、`Notification`、`Stop` 和 `PreCompact`——可配置的扩展点，操作者可以将策略、遥测和护栏注入其中。

沙箱是 E2B 或 Daytona。每个任务在一个新的 devcontainer 中运行，带有一个以读写方式挂载的 git worktree。控制框架从不触碰主机文件系统。worktree 在成功或失败后被拆除。成本控制在三个层面执行：每轮 token 上限、每个会话的美元预算和硬性的轮次限制（通常为 50）。可观测性层面是带有 GenAI 语义约定的 OpenTelemetry span，发送到自托管的 Langfuse。

## 架构

```
  user CLI  ->  harness (Bun + Ink TUI)
                  |
                  v
           plan / act / observe loop  <--->  Claude Sonnet 4.7 / GPT-5.4-Codex / Gemini 3 Pro
                  |                          (via OpenRouter, model-agnostic)
                  v
           tool dispatcher (MCP StreamableHTTP client)
                  |
     +------------+------------+----------+
     v            v            v          v
  read/edit    ripgrep     tree-sitter   git/run
     |            |            |          |
     +------------+------------+----------+
                  |
                  v
           E2B / Daytona sandbox  (worktree isolated)
                  |
                  v
           hooks: Pre/Post, Session, Prompt, Compact
                  |
                  v
           OpenTelemetry -> Langfuse (spans, tokens, $)
                  |
                  v
           PR via GitHub app
```

## 技术栈

- 控制框架运行时: Bun 1.2 + Ink 5（终端中的 React）
- 模型访问: OpenRouter 统一 API，支持 Claude Sonnet 4.7、GPT-5.4-Codex、Gemini 3 Pro、Opus 4.5（用于最难的任务）
- 工具传输: Model Context Protocol StreamableHTTP（MCP 2026 修订版）
- 沙箱: E2B 沙箱（JS SDK）或 Daytona devcontainers
- 代码搜索: ripgrep 子进程，tree-sitter 解析器支持 17 种语言（预编译）
- 隔离: 每个任务使用 `git worktree add`，成功/失败后清理
- 评估框架: SWE-bench Pro（已验证子集）+ Terminal-Bench 2.0 + 你自己的 30 个留存任务
- 可观测性: OpenTelemetry SDK 使用 `gen_ai.*` 语义约定 → 自托管 Langfuse
- PR 发布: GitHub App 使用细粒度 token，范围限制在目标仓库

## 构建它

1. **TUI 和命令循环。** 使用 Ink 搭建一个 Bun 项目。接受 `agent run <repo> "<task>"`。打印分屏视图：计划窗格（顶部）、工具调用流（中间）、token 预算（底部）。在 Ctrl-C 上添加取消操作，在退出前触发 `SessionEnd` 钩子。

2. **计划状态。** 定义一个带类型的 TodoWrite 模式（pending/in_progress/done 项目以及备注）。模型每轮都将整个状态作为工具调用来重写——不要让它增量地变异。将计划持久化到 `.agent/state.json`，以便崩溃后可以恢复。

3. **工具界面。** 定义六个工具：`read_file`、`edit_file`（带差异预览）、`ripgrep`、`tree_sitter_symbols`、`run_shell`（带超时）、`git`（status/diff/commit/push）。通过 MCP StreamableHTTP 暴露，使控制框架与传输无关。每个工具返回截断的输出（每次调用上限 4k token）。

4. **沙箱包装。** 每个任务启动一个 E2B 沙箱。`git worktree add -b agent/$TASK_ID` 创建一个新分支。所有工具调用在沙箱内执行。主机文件系统不可访问。

5. **钩子。** 实现所有八个 2026 年钩子类型。至少连接四个用户编写的钩子：(a) `PreToolUse` 破坏性命令守卫，阻止 worktree 之外的 `rm -rf`，(b) `PostToolUse` token 核算，(c) `SessionStart` 预算初始化，(d) `Stop` 写入最终跟踪包。

6. **评估循环。** 克隆 SWE-bench Pro Python 的 30 个 issue 子集。将你的控制框架与 mini-swe-agent（最小基线）在 pass@1、每任务轮数和每任务美元成本上进行比较。将结果写入 `eval/results.jsonl`。

7. **成本控制。** 硬性上限：50 轮、200k 上下文、每任务 $5。`PreCompact` 钩子在 150k 标记处将早期轮次总结为 prior-state 块，为新观察腾出空间而不丢失计划。

8. **PR 发布。** 成功后，最后一步是 `git push` + 一个 GitHub API 调用，在正文中打开一个包含计划和差异摘要的 PR。

## 使用它

```
$ agent run ./my-repo "Fix the race condition in worker.rs"
[plan]  1 locate worker.rs and enumerate mutex uses
        2 identify shared state under contention
        3 propose fix, verify tests
[tool]  ripgrep mutex.*lock -t rust           (44 matches, truncated)
[tool]  read_file src/worker.rs 120..180
[tool]  edit_file src/worker.rs (+8 -3)
[tool]  run_shell cargo test worker::          (passed)
[plan]  1 done · 2 done · 3 done
[done]  PR opened: #482   turns=9   tokens=38k   cost=$0.41
```

## 交付它

可交付的技能位于 `outputs/skill-terminal-coding-agent.md`。给定一个仓库路径和任务描述，它在沙箱中运行完整的"计划-执行-观察"循环，并返回一个 PR URL 和一个跟踪包。本实践项目的评分标准：

| 权重 | 标准 | 如何衡量 |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 vs 基线 | 你的控制框架在 30 个匹配的 Python 任务上与 mini-swe-agent 对比 |
| 20 | 架构清晰度 | 计划/执行/观察的分离、钩子界面、工具模式——对照 Live-SWE-agent 布局审查 |
| 20 | 安全性 | 沙箱逃逸测试、权限提示、破坏性命令守卫通过红队测试 |
| 20 | 可观测性 | 跟踪完整性（100% 的工具调用有 span）、每轮 token 核算 |
| 15 | 开发者体验 | 冷启动 < 2s、崩溃恢复恢复计划、Ctrl-C 在工具运行中干净取消 |
| **100** | | |

## 练习

1. 将后端模型从 Claude Sonnet 4.7 替换为在 vLLM 上服务的 Qwen3-Coder-30B。比较 pass@1 和每任务美元成本。报告开放模型在哪些方面表现不佳。

2. 添加一个 `reviewer` 子智能体，在 PR 发布前阅读差异，并可以请求修订循环。衡量误报审查是否将 SWE-bench 通过率降至单智能体基线以下（提示：通常是会的）。

3. 对沙箱进行压力测试：编写一个尝试 `curl` 外部 URL 的任务和一个写入 worktree 之外的任务。确认两者都被 PreToolUse 钩子阻止。记录这些尝试。

4. 使用较小的模型（Haiku 4.5）实现 `PreCompact` 摘要。衡量在 3 倍压缩下计划保真度丢失了多少。

5. 将 MCP StreamableHTTP 传输替换为 stdio。对冷启动和每次调用延迟进行基准测试。为纯本地使用选择一个优胜方案。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| Harness | "智能体循环" | 围绕模型的代码，负责分派工具、维护计划状态和执行预算 |
| Hook | "智能体事件监听器" | 用户编写的脚本，由控制框架在八个生命周期事件之一上运行 |
| Worktree | "Git 沙箱" | 一个在单独路径下的链接 git 检出；可处置而不影响主克隆 |
| TodoWrite | "计划状态" | 一个带类型的 pending/in-progress/done 项目列表，模型每轮重写它 |
| StreamableHTTP | "MCP 传输" | 2026 年 MCP 修订版：长连接 HTTP 连接，带有双向流式传输；替代 SSE |
| Token ceiling | "上下文预算" | 每轮或每个会话的输入+输出 token 上限；触发压缩或终止 |
| pass@1 | "单次尝试通过率" | 首次运行且不重试或不窥探测试集的情况下解决的 SWE-bench 任务的比例 |

## 扩展阅读

- [Claude Code 文档](https://docs.anthropic.com/en/docs/claude-code) — Anthropic 的参考控制框架
- [Cursor 3 更新日志](https://cursor.com/changelog) — Agent Tabs 和 Composer 2 产品说明
- [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) — SWE-bench 控制框架比较的最小基线
- [Live-SWE-agent](https://github.com/OpenAutoCoder/live-swe-agent) — 使用 Opus 4.5 在 SWE-bench Verified 上达到 79.2%
- [OpenCode](https://opencode.ai) — 开放控制框架，112k stars
- [SWE-bench Pro 排行榜](https://www.swebench.com) — 本实践项目所针对的评估
- [Model Context Protocol 2026 路线图](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — StreamableHTTP、能力元数据
- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 工具调用和 token 使用的 span 模式
