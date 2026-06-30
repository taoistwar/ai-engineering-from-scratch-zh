# 实践项目 10 — 多智能体软件工程团队

> SWE-AF 的工厂架构、MetaGPT 的基于角色的提示、AutoGen 0.4 的类型化角色图、Cognition 的 Devin 和 Factory 的 Droids 在 2026 年都收敛于相同的形态：一位架构师做规划，N 个编码者在并行 worktree 中工作，一位评审者门控，一位测试者验证。并行 worktree 将墙钟时间转化为吞吐量。共享状态和交接协议成为故障表面。实践项目是构建这个团队，在 SWE-bench Pro 上评估，并报告哪些交接会断裂以及频率。

**类型:** 实践项目
**语言:** Python / TypeScript（智能体），Shell（worktree 脚本）
**前置条件:** 阶段 11（LLM 工程），阶段 13（工具），阶段 14（智能体），阶段 15（自主），阶段 16（多智能体），阶段 17（基础设施）
**涉及的阶段:** P11 · P13 · P14 · P15 · P16 · P17
**时间:** 40 小时

## 问题

单智能体编程控制框架在大型任务上碰到了天花板。不是因为任何单个智能体很弱，而是因为一个 200k token 的上下文无法容纳一个架构计划加上四个并行的代码库切片加上评审评论加上测试输出。多智能体工厂拆分问题：架构师拥有计划，编码者在并行 worktree 中拥有实现，评审者门控，测试者验证。SWE-AF 的"工厂"架构、MetaGPT 的角色、AutoGen 的类型化角色图——这三种框架都描述了相同的形态。

故障表面是交接。架构师计划了一些编码者无法实现的东西。编码者产生了冲突的差异。评审者批准了一个幻觉式的修复。测试者与仍在编写的编码者竞争。你将构建一个这样的团队，在 50 个 SWE-bench Pro 问题上运行它，追踪每个交接，并发布事后分析。

## 概念

角色是带类型的智能体。**架构师**（Claude Opus 4.7）读取 issue，编写计划，并将其分解为具有显式接口的子任务。**编码者**（Claude Sonnet 4.7，N 个并行实例，每个在 `git worktree` + Daytona 沙箱中）独立实现子任务。**评审者**（GPT-5.4）读取合并后的差异并批准或请求具体更改。**测试者**（Gemini 2.5 Pro）在隔离中运行测试套件，并报告通过/失败和产物。

通信通过共享任务板进行（基于文件或 Redis）。每个角色消费它被允许处理的任务。交接是 A2A 协议类型化消息。协调关注点：合并冲突解决（协调员角色或自动三方合并）、共享状态同步（计划在编码者开始后冻结；重新计划是单独事件）以及评审者门控（评审者不能批准自己的更改或它提出的更改）。

Token 放大是隐藏成本。每个角色边界增加了摘要提示和交接上下文。一个 40 轮的单智能体运行变成跨四个角色的总共 160 轮。评分标准特别权衡 token 效率与单智能体基线，因为问题不是"多智能体是否有效"而是"它是否在每美元上胜出"。

## 架构

```
GitHub issue URL
      |
      v
Architect (Opus 4.7)
   reads issue, produces plan with subtasks + interfaces
      |
      v
Task board (file / Redis)
      |
   +-- subtask 1 ---+-- subtask 2 ---+-- subtask 3 ---+-- subtask 4 ---+
   v                v                v                v                v
Coder A          Coder B          Coder C          Coder D          (4 parallel)
 (Sonnet)         (Sonnet)         (Sonnet)         (Sonnet)
 worktree A       worktree B       worktree C       worktree D
 Daytona          Daytona          Daytona          Daytona
      |                |                |                |
      +--------+-------+-------+--------+
               v
           merge coordinator  (three-way merge + conflict resolution)
               |
               v
           Reviewer (GPT-5.4)
               |
               v
           Tester  (Gemini 2.5 Pro)  -> passes? -> open PR
                                     -> fails?  -> route back to coder
```

## 技术栈

- 编排: LangGraph 带共享状态 + 每个智能体子图
- 消息: A2A 协议（Google 2025）用于类型化智能体间消息
- 模型: Opus 4.7（架构师）、Sonnet 4.7（编码者）、GPT-5.4（评审者）、Gemini 2.5 Pro（测试者）
- Worktree 隔离: 每个编码者使用 `git worktree add` + Daytona 沙箱
- 合并协调员: 自定义三方合并 + LLM 介导的冲突解决
- 评估: SWE-bench Pro（50 个 issues）、SWE-AF 场景、HumanEval++ 用于单元测试
- 可观测性: Langfuse 带角色标签的 spans，每个智能体的 token 核算
- 部署: K8s，每个角色作为独立的 Deployment + 积压上的 HPA

## 构建它

1. **任务板。** 基于文件的 JSONL 带类型化消息：`plan_request`、`subtask`、`diff_ready`、`review_needed`、`test_needed`、`approved`、`rejected`、`replan_needed`。智能体按标签订阅。

2. **架构师。** 读取 GitHub issue，使用 Opus 4.7 运行带有计划模板的、需要显式子任务接口（涉及的文件、公共函数、测试影响）的。发出一个带子任务 DAG 的 `plan_request`。

3. **编码者。** N 个并行工作器，每个从板上认领一个子任务。每个产生一个新的 `git worktree add` 分支加上一个 Daytona 沙箱。实现子任务。发出带有补丁 + 测试差异的 `diff_ready`。

4. **合并协调员。** 在所有编码者完成后，将 N 个分支三方合并到一个暂存分支。LLM 介导的冲突解决仅在文件级重叠存在时进行。

5. **评审者。** GPT-5.4 读取合并后的差异。不能批准它撰写的差异。发出 `approved`（无操作）或 `review_feedback` 带有具体更改请求，路由回相关编码者。

6. **测试者。** Gemini 2.5 Pro 在干净的沙箱中运行测试套件。捕获产物。发出 `test_passed` 或 `test_failed` 带堆栈跟踪。失败的测试循环回拥有失败子任务的编码者。

7. **交接核算。** 每个穿越角色边界的消息在 Langfuse 中获得一个 span，带有载荷大小和使用的模型。计算每个子任务的 token 放大（编码者_tokens + 评审者_tokens + 测试者_tokens + 架构师份额 / 编码者_tokens）。

8. **评估。** 在 50 个 SWE-bench Pro 问题上运行。与单智能体基线比较 pass@1 和每解决一个问题的 $（一个 Sonnet 4.7 在单个 worktree 中）。

9. **事后分析。** 对于每个失败的问题，识别断裂的交接（计划太模糊、合并冲突、评审者误批准、测试者片状）。生成交接故障直方图。

## 使用它

```
$ team run --issue https://github.com/acme/widget/issues/842
[architect] plan: 4 subtasks (parser, cache, api, migration)
[board]     dispatched to 4 coders in parallel worktrees
[coder-A]   subtask parser  -> 42 lines, tests pass locally
[coder-B]   subtask cache   -> 88 lines, tests pass locally
[coder-C]   subtask api     -> 31 lines, tests pass locally
[coder-D]   subtask migration -> 19 lines, tests pass locally
[merge]     3-way merge: 0 conflicts
[reviewer]  comments on cache (thread pool sizing); routed to coder-B
[coder-B]   revision: 92 lines; submits
[reviewer]  approved
[tester]    all 412 tests pass
[pr]        opened #3382   4 coders, 1 revision, $4.90, 18m
```

## 交付它

`outputs/skill-multi-agent-team.md` 是可交付成果。给定一个 issue URL 和并行级别，团队生成一个可合并的 PR，带有每个角色的 token 核算。

| 权重 | 标准 | 如何衡量 |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 | 匹配的 50 个问题子集，pass@1 |
| 20 | 并行加速 | 与单智能体基线的墙钟时间对比 |
| 20 | 评审质量 | 在注入错误探测上的误批准率 |
| 20 | Token 效率 | 每个已解决问题的总 token 数 vs 单智能体 |
| 15 | 协调工程 | 合并冲突解决、交接故障直方图 |
| **100** | | |

## 练习

1. 在运行中向差异注入一个明显的错误（在主体之前额外添加 `return None`）。测量评审者的误批准率。调整评审提示直到误批准率低于 5%。

2. 减少到两个编码者（架构师 + 编码者 + 评审者 + 测试者，编码者按顺序运行两个子任务）。比较墙钟时间和通过率。

3. 将合并协调员替换为单写入者约束（子任务触及不相交的文件集）。衡量架构师身上的计划负担。

4. 将评审者从 GPT-5.4 替换为 Claude Opus 4.7。测量误批准率和 token 成本差值。

5. 添加第五个角色：文档编写者（Haiku 4.5）。评审后，它生成一个更新日志条目。衡量文档质量是否证明额外的 token 花费是合理的。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| Parallel worktree | "隔离分支" | `git worktree add` 为每个编码者生成一个全新的工作树 |
| Task board | "共享消息总线" | 智能体订阅的类型化消息的文件或 Redis 存储 |
| Handoff | "角色边界" | 从一个角色的上下文穿越到另一个角色的上下文的任何消息 |
| Token amplification | "多智能体开销" | 跨角色的总 token 数 / 同一任务的单智能体 token 数 |
| A2A protocol | "智能体到智能体" | Google 2025 规范，用于类型化智能体间消息 |
| Merge coordinator | "集成器" | 运行三方合并并介导冲突的组件 |
| False approval | "评审者幻觉" | 评审者批准了带有已知错误的差异 |

## 扩展阅读

- [SWE-AF 工厂架构](https://github.com/Agent-Field/SWE-AF) — 参考 2026 年多智能体工厂
- [MetaGPT](https://github.com/FoundationAgents/MetaGPT) — 基于角色的多智能体框架
- [AutoGen v0.4](https://github.com/microsoft/autogen) — Microsoft 的类型化角色框架
- [Cognition AI (Devin)](https://cognition.ai) — 参考产品
- [Factory Droids](https://www.factory.ai) — 替代参考产品
- [Google A2A 协议](https://developers.google.com/agent-to-agent) — 智能体间消息规范
- [git worktree 文档](https://git-scm.com/docs/git-worktree) — 隔离基底
- [SWE-bench Pro](https://www.swebench.com) — 评估目标
