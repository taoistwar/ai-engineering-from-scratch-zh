# 实践项目 09 — 代码迁移智能体（仓库级语言/运行时升级）

> Amazon 的 MigrationBench（Java 8 到 17）和 Google 的 App Engine Py2-to-Py3 迁移器设定了 2026 年的标准。Moderne 的 OpenRewrite 在规模上执行确定性 AST 重写。Grit 使用类似 codemod 的 DSL 针对相同问题。生产模式结合了两者：一个用于安全重写的确定性基底加上一个用于模糊情况的智能体层，一个用于每个分支构建的沙箱，以及一个在 PR 打开前变绿的测试框架。实践项目是迁移 50 个真实仓库并发布带有故障分类的通过率。

**类型:** 实践项目
**语言:** Python（智能体），Java / Python（目标），TypeScript（仪表盘）
**前置条件:** 阶段 5（NLP），阶段 7（transformers），阶段 11（LLM 工程），阶段 13（工具），阶段 14（智能体），阶段 15（自主），阶段 17（基础设施）
**涉及的阶段:** P5 · P7 · P11 · P13 · P14 · P15 · P17
**时间:** 30 小时

## 问题

大规模代码迁移是 2026 年编程智能体最干净的生产应用之一。基准真相是明显的（迁移后测试套件是否通过？），回报是真实的（Java-8 舰队迁移是一个人头规模的项目），基准是公开的（MigrationBench 50 个仓库子集）。Moderne 的 OpenRewrite 处理确定性的一面。智能体层处理 OpenRewrite 配方无法处理的一切：模糊的重写、构建系统漂移、长尾语法、传递依赖断裂。

你将构建一个智能体，接受 Java 8 仓库（或 Python 2 仓库），生成一个绿色 CI 的迁移分支。你将测量通过率、测试覆盖率保持、每个仓库的成本，并构建一个故障分类。与纯确定性基线的并列对比告诉你智能体的价值实际在哪里。

## 概念

管道有两层。**确定性基底**（Java 的 OpenRewrite，Python 的 libcst）安全地运行大部分机械重写：导入、方法签名、空安全检查、try-with-resources、弃用的 API 替换。它快速且产生可审计的差异。**智能体层**（OpenAI Agents SDK 或 LangGraph over Claude Opus 4.7 和 GPT-5.4-Codex）处理配方无法处理的情况：构建文件升级（Maven/Gradle/pyproject）、传递依赖冲突、测试片状、自定义注解。

每个仓库获得一个预装了目标运行时的 Daytona 沙箱。智能体迭代：运行构建、分类失败、应用修复、重新运行。硬性限制：每个仓库 30 分钟、每个仓库 $8、20 个智能体轮次。如果所有测试通过且覆盖率差值不是负数，该分支打开一个 PR。如果不通过，仓库被归档到一个故障类下并附带证据。

故障分类是可交付成果。跨 50 个仓库，什么出了故障？传递依赖？自定义注解？构建工具版本？与迁移无关的测试片状？每个类别获得计数和示例差异。未来的配方作者可以针对前三名。

## 架构

```
target repo
      |
      v
OpenRewrite / libcst deterministic recipes
   (safe, fast, auditable, ~70-80% of fixes)
      |
      v
Daytona sandbox per branch
      |
      v
agent loop (Claude Opus 4.7 / GPT-5.4-Codex):
   - run build -> capture failures
   - classify failures (build, test, lint)
   - apply fix (patch or retry recipe)
   - rerun
   - budget: 30 min, $8, 20 turns
      |
      v
test + coverage delta gate
      |
      v (passed)
open PR
      |
      v (failed)
file under failure class + attach repro
```

## 技术栈

- 确定性基底: OpenRewrite（Java）或 libcst（Python）
- 智能体: OpenAI Agents SDK 或 LangGraph over Claude Opus 4.7 + GPT-5.4-Codex
- 沙箱: Daytona devcontainers 每个分支，预装目标运行时（Java 17 / Python 3.12）
- 构建系统: Maven、Gradle、uv（Python）
- 基准: Amazon MigrationBench 50 个仓库子集（Java 8 到 17），Google App Engine Py2-to-Py3 仓库
- 测试框架: 并行运行器，通过 Jacoco（Java）或 coverage.py（Python）的覆盖率
- 可观测性: Langfuse + 每个仓库带有每个差异块的追踪包
- 仪表盘: 故障分类仪表盘，带有每个类别的计数和示例差异

## 构建它

1. **配方传递。** 首先运行 OpenRewrite（Java）或 libcst（Python）配方。捕获 70-80% 的机械性迁移。作为"配方"提交进行提交。

2. **构建试验。** Daytona 沙箱：安装目标运行时，运行构建。如果绿色，跳到测试。如果红色，交给智能体。

3. **智能体循环。** LangGraph 带工具：`run_build`、`read_file`、`edit_file`、`run_test`、`git_diff`。智能体分类故障（依赖、语法、测试、构建工具）并应用有针对性的修复。重新运行。

4. **预算上限。** 每个仓库 30 分钟墙钟时间，$8 成本，20 个智能体轮次。任何违反都会中止并归档到"budget_exhausted"下并附带当前差异。

5. **测试 + 覆盖率门。** 构建变绿后，运行测试套件。将覆盖率与基础仓库进行比较。如果覆盖率下降超过 2%，归档到"coverage_regression"下。

6. **PR 打开。** 成功后，推送分支，用智能体撰写的差异和配方应用以及提交摘要打开 PR。

7. **故障分类。** 对于每个失败的仓库，标记一个类别：`dep_upgrade_required`、`build_tool_drift`、`custom_annotation`、`test_flake`、`syntax_edge_case`、`budget_exhausted`。构建仪表盘。

8. **50 个仓库运行。** 在 MigrationBench 子集上执行。报告每个类别的通过率、每个仓库的成本、覆盖率保持，以及与纯确定性基线的对比。

## 使用它

```
$ migrate legacy-java-service --target java17
[recipe]   27 rewrites applied (JUnit 4->5, HashMap initializer, try-with-resources)
[build]    FAIL: cannot find symbol sun.misc.BASE64Encoder
[agent]    turn 1 classify: removed_jdk_api
[agent]    turn 2 apply: sun.misc.BASE64Encoder -> java.util.Base64
[build]    OK
[tests]    412/412 passing; coverage 84.1% -> 84.3%
[pr]       opened #1841  cost=$3.20  turns=4
```

## 交付它

`outputs/skill-migration-agent.md` 是可交付成果。给定一个仓库，它执行确定性配方，然后执行智能体循环，生成一个绿色迁移分支，或将仓库归档到一个分类类别下。

| 权重 | 标准 | 如何衡量 |
|:-:|---|---|
| 25 | MigrationBench 通过率 | 50 个仓库子集的 pass@1 |
| 20 | 测试覆盖率保持 | 与基础仓库的平均覆盖率差值 |
| 20 | 每个迁移仓库的成本 | 通过运行中的 $/仓库 |
| 20 | 智能体/确定性工具集成 | OpenRewrite 处理 vs 智能体撰写的修复比例 |
| 15 | 故障分析撰写 | 分类完整性带示例 |
| **100** | | |

## 练习

1. 仅使用 OpenRewrite 运行迁移管道（无智能体）。将通过率与完整管道进行比较。识别只有智能体才能产生差异的案例。

2. 实现"lint-clean"检查：迁移后，运行样式 linter（Java 的 spotless，Python 的 ruff）。如果出现新的 lint 错误，使 PR 失败。测量覆盖率保持但样式回归的比率。

3. 添加"最小差异"优化器：在智能体的分支通过测试后，用第二次传递修剪不必要的更改。报告差异大小的缩减。

4. 扩展到第三种迁移：Node 18 到 Node 22。重用沙箱包装；将配方层替换为自定义 codemod。

5. 测量首次绿色构建时间（TTFGB）作为 UX 指标。目标：p50 低于 10 分钟。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| Deterministic substrate | "配方引擎" | OpenRewrite / libcst：带有安全保证的声明式 AST 重写 |
| Codemod | "代码修改程序" | 机械地更改源代码的重写规则 |
| Build drift | "工具版本偏差" | 主版本之间细微的 Maven / Gradle / uv 行为变化 |
| Failure class | "分类桶" | 仓库未能迁移的标签化原因：依赖、语法、测试、构建工具、预算 |
| Coverage delta | "覆盖率保持" | 从基础分支到迁移分支的测试覆盖率 % 的变化 |
| Agent turn | "工具调用轮次" | 智能体循环中的一个 计划 -> 执行 -> 观察 周期 |
| Budget exhaustion | "触达了上限" | 仓库消耗了其 30 分钟 / $8 / 20 轮次的限制而未能通过 |

## 扩展阅读

- [Amazon MigrationBench](https://aws.amazon.com/blogs/devops/amazon-introduces-two-benchmark-datasets-for-evaluating-ai-agents-ability-on-code-migration/) — 2026 年的规范基准
- [Moderne.io OpenRewrite 平台](https://www.moderne.io) — 确定性基底参考
- [OpenRewrite 文档](https://docs.openrewrite.org) — 配方编写
- [Grit.io](https://www.grit.io) — 替代 codemod DSL
- [OpenAI 沙箱迁移食谱](https://developers.openai.com/cookbook/examples/agents_sdk/sandboxed-code-migration/sandboxed_code_migration_agent) — Agents SDK 参考
- [Google App Engine Py2 to Py3 迁移器](https://cloud.google.com/appengine) — 替代迁移基准
- [libcst](https://github.com/Instagram/LibCST) — Python 确定性基底
- [Daytona 沙箱](https://daytona.io) — 参考每分支沙箱
