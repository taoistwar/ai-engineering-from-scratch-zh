# 实践项目 16 — GitHub Issue 到 PR 的自主智能体

> AWS Remote SWE Agents、Cursor Background Agents、OpenAI Codex cloud 和 Google Jules 在 2026 年都推出了相同的产品形态：标记一个 issue，获得一个 PR。在云沙箱中运行智能体，验证测试通过，并发布一个带有理由的可评审 PR。难点在于自动复现仓库的构建环境、防止凭据泄漏、执行每个仓库的预算，以及确保智能体不能强制推送。这个实践项目构建自托管版本，并在成本和通过率上与托管替代方案进行比较。

**类型:** 实践项目
**语言:** Python（智能体），TypeScript（GitHub App），YAML（Actions）
**前置条件:** 阶段 11（LLM 工程），阶段 13（工具），阶段 14（智能体），阶段 15（自主），阶段 17（基础设施）
**涉及的阶段:** P11 · P13 · P14 · P15 · P17
**时间:** 30 小时

## 问题

异步云编程智能体是一个与交互式编程智能体（实践项目 01）不同的产品类别。用户体验是一个 GitHub 标签。你将一个 issue 标记为 `@agent fix this`，一个工作进程在云沙箱中启动，克隆仓库，运行测试，编辑文件，验证，并打开一个带有智能体理由的 PR。没有交互循环，没有终端。AWS Remote SWE Agents、Cursor Background Agents、OpenAI Codex cloud、Google Jules 和 Factory Droids 都汇聚于此。

工程挑战是具体的：环境复现（智能体必须从头构建仓库，没有缓存的开发映像）、片状测试（必须重新运行或隔离）、凭据范围界定（具有最小细粒度权限的 GitHub App）、每个仓库每天预算的执行、以及不强制推送策略。实践项目在与托管替代方案的比较中测量通过率、成本和安全。

## 概念

触发器是 GitHub webhook（issue 标签或 PR 评论）。调度器将工作排入 ECS Fargate 或 Lambda。工作进程将仓库拉入具有从仓库推断的通用 Dockerfile（语言、框架）的 Daytona 或 E2B 沙箱中。智能体在 Claude Opus 4.7 或 GPT-5.4-Codex 上运行 mini-swe-agent 或 SWE-agent v2 循环。它迭代：读取代码、提出修复、应用补丁、运行测试。

验证是门控步骤。在沙箱中 PR 打开之前，完整的 CI 必须通过。计算覆盖率差值；如果为负且超过阈值，PR 打开但被标记为 `needs-review`。智能体将理由作为 PR 描述发布，加上审查者可以提及以进行后续操作的 `@agent` 线程。

安全通过两个不同的 GitHub 面进行范围界定：App 提供一个短期安装 token，具有 `workflows: read` 和狭窄的仓库 contents/PR 作用域；分支保护（而非应用权限）强制执行"不直接写入 `main`"和"不强制推送"——应用永远不会被添加到绕过列表中。对 `.github/workflows` 的按路径只读访问不是真正的 GitHub App 原语，因此智能体对文件编辑的允许列表必须在工作进程处强制执行。每个仓库每天的预算上限在调度器处强制执行（例如，每个仓库每天最多 5 个 PR，每个 PR $20）。

## 架构

```
GitHub issue labeled `@agent fix` or PR comment
            |
            v
    GitHub App webhook -> AWS Lambda dispatcher
            |
            v
    ECS Fargate task (or GitHub Actions self-hosted runner)
       - pull repo
       - infer Dockerfile (language, package manager)
       - Daytona / E2B sandbox with target runtime
       - clone -> git worktree -> agent branch
            |
            v
    mini-swe-agent / SWE-agent v2 loop
       Claude Opus 4.7 or GPT-5.4-Codex
       tools: ripgrep, tree-sitter, read/edit, run_tests, git
            |
            v
    verify CI passes in-sandbox + coverage delta check
            |
            v (verified)
    git push + open PR via GitHub App
       PR body = rationale + diff summary + trace URL
       label: needs-review
            |
            v
    operator reviews; can @-mention agent for follow-ups
```

## 技术栈

- 触发器: GitHub App 带细粒度 token；通过 Lambda 或 Fly.io 的 webhook 接收器
- 工作进程: ECS Fargate 任务（或 GitHub Actions 自托管运行器）
- 沙箱: Daytona devcontainer 或每个任务的 E2B 沙箱
- 智能体循环: mini-swe-agent 基线或 SWE-agent v2 over Claude Opus 4.7 / GPT-5.4-Codex
- 检索: tree-sitter repo-map + ripgrep
- 验证: 沙箱内完整 CI + 覆盖率差值门
- 可观测性: Langfuse 带每个 PR 的追踪归档，从 PR 正文链接
- 预算: 每个仓库每天的美元上限；每个仓库每天的最大 PR 数

## 构建它

1. **GitHub App。** 细粒度安装 token：issues 读+写、pull_requests 写、contents 读+写、workflows 读。分支保护（唯一可以实现此目的的层面）强制执行"不直接推送到 `main`"和"不强制推送"；应用不在绕过列表中。工作进程将"不在 `.github/workflows` 下写入"作为对提议的差异的允许列表检查来强制执行，因为 GitHub App 权限不是按路径范围的。

2. **Webhook 接收器。** Lambda 函数接受 issue 标签 / PR 评论 webhooks。按标签 `@agent fix this` 过滤。排入 SQS。

3. **调度器。** 从 SQS 弹出任务。强制执行每个仓库每天的预算。使用仓库 URL、issue 正文和一个全新的 Daytona 沙箱启动一个 ECS Fargate 任务。

4. **环境推断。** 检测语言（Python、Node、Go、Rust）和包管理器（uv、pnpm、go mod、cargo）。如果不存在，即时生成 Dockerfile。

5. **智能体循环。** mini-swe-agent 或 SWE-agent v2 with Claude Opus 4.7。工具：ripgrep、tree-sitter repo-map、read_file、edit_file、run_tests、git。硬性限制：$20 成本、30 分钟墙钟时间、30 个智能体轮次。

6. **验证。** 循环结束后，在沙箱内运行完整测试套件。通过 jacoco / coverage.py 计算覆盖率差值。如果 CI 红色：停止，不打开 PR。如果覆盖率下降超过 2%：打开 PR 并标记 `needs-review`。

7. **PR 发布。** 推送智能体分支。通过 GitHub API 打开 PR，包含：标题、理由、差异摘要、追踪 URL、成本、轮次。

8. **凭据卫生。** 工作进程使用短期 GitHub App 安装 token 运行。日志在归档前脱敏秘密。

9. **评估。** 30 个不同难度级别的种子内部 issues。测量通过率、PR 质量（差异大小、风格、覆盖率）、成本、延迟。与 Cursor Background Agents 和 AWS Remote SWE Agents 在相同 issues 上进行比较。

## 使用它

```
# on github.com
  - user labels issue #842 with `@agent fix this`
  - PR #1903 appears 14 minutes later
  - body:
    > Fixed NPE in widget.dedupe() caused by null comparator entry.
    > Added regression test widget_test.go::TestDedupeNullComparator.
    > Coverage delta: +0.12%
    > Turns: 7  Cost: $1.80  Trace: langfuse:...
    > Label: needs-review
```

## 交付它

`outputs/skill-issue-to-pr.md` 是可交付成果。一个 GitHub App + 异步云工作进程，将标记的 issues 转化为具有有限成本和有限凭据的可评审 PR。

| 权重 | 标准 | 如何衡量 |
|:-:|---|---|
| 25 | 30 个 issues 的通过率 | 端到端成功（CI 绿色 + 覆盖率 OK） |
| 20 | PR 质量 | 差异大小、覆盖率差值、风格符合性 |
| 20 | 每个已解决问题的成本和延迟 | 每个 PR 的美元和墙钟时间 |
| 20 | 安全性 | 有范围的 token、每个仓库预算、不强制推送、凭据卫生 |
| 15 | 操作者体验 | 理由评论、重试能力、@-提及 后续 |
| **100** | | |

## 练习

1. 添加"修复片状测试"模式：标签 `@agent stabilize-flake TestX` 在沙箱中运行测试 50 次，并提出一个使其稳定的最小更改。

2. 在三个共享的 issues 上比较成本与 Cursor Background Agents。报告哪个工具在何处胜出。

3. 实现预算仪表盘：每个仓库每天的成本、每个用户的成本。在异常时告警。

4. 构建"演练"模式，打开一个草稿 PR 而不运行 CI，以便审查者可以廉价地检查计划。

5. 添加保留策略：超过 7 天未合并的 PR 分支自动删除。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| GitHub App | "有范围限制的机器人身份" | App 具有细粒度权限 + 短期安装 token |
| Async cloud agent | "后台智能体" | 在云沙箱中运行的非交互工作进程，而非终端 |
| Environment inference | "Dockerfile 合成" | 检测语言 + 包管理器，如果缺少则生成 Dockerfile |
| Verification | "沙箱内 CI" | 在工作进程内运行完整测试套件，然后才打开 PR |
| Coverage delta | "覆盖率保持" | 从基础到智能体分支的测试覆盖率 % 的变化 |
| Per-repo budget | "每日上限" | 在调度器处强制执行的美元和 PR 计数上限 |
| Rationale | "PR 正文解释" | 智能体对更改内容和原因的摘要；在 PR 正文中必需 |

## 扩展阅读

- [AWS Remote SWE Agents](https://github.com/aws-samples/remote-swe-agents) — 规范的异步云智能体参考
- [SWE-agent](https://github.com/SWE-agent/SWE-agent) — CLI 参考
- [Cursor Background Agents](https://docs.cursor.com/background-agent) — 商业替代方案
- [OpenAI Codex（cloud）](https://openai.com/codex) — 托管竞争对手
- [Google Jules](https://jules.google) — Google 的托管版本
- [Factory Droids](https://www.factory.ai) — 替代商业参考
- [GitHub App 文档](https://docs.github.com/en/apps) — 有范围限制的机器人身份
- [Daytona 云沙箱](https://daytona.io) — 参考沙箱
