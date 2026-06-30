# 基准测试：SWE-bench、GAIA、AgentBench

> 2026 年有三个基准测试锚定 agent 评估。SWE-bench 测试代码修补。GAIA 测试通用工具使用。AgentBench 测试多环境推理。了解它们的构成、它们的污染情况以及它们不测量什么。

**类型：** 学习
**语言：** Python（标准库）
**前置条件：** 第 14 阶段 · 06（工具使用）
**时间：** ~60 分钟

## 学习目标

- 列举 SWE-bench 的测试 harness（FAIL_TO_PASS）并解释为什么它以单元测试为门控标准。
- 解释为什么 SWE-bench Verified（OpenAI，500 个任务）存在以及它移除了什么。
- 描述 GAIA 的设计：对人类简单，对 AI 困难；三个难度级别。
- 列举 AgentBench 的八个环境及其对开源 LLM 的主要障碍。
- 总结 SWE-bench+ 的污染发现及其影响。

## 问题

排行榜告诉你哪个模型在一个基准测试上获胜。它们不告诉你：

- 基准测试是否被污染（解决方案在训练数据中，测试泄露）。
- 基准测试是否测量你关心的内容（代码 vs 浏览 vs 通用）。
- 评估器是否稳健（AST 匹配、状态检查、人工审查）。

在引用一个数字之前，先了解三个锚定基准测试及其失败模式。

## 概念

### SWE-bench（Jimenez 等人，ICLR 2024 oral）

- 来自 12 个流行 Python 仓库的 2,294 个真实 GitHub issue。
- Agent 获得：修复前提交的代码库 + 自然语言 issue 描述。
- Agent 产生：一个补丁。
- 评估器：应用补丁，运行仓库的测试套件。补丁必须翻转 FAIL_TO_PASS 测试（之前失败，现在通过）而不破坏 PASS_TO_PASS 测试。

SWE-agent（Yang 等人，2024）在发布时达到 12.5%，强调 agent-计算机接口（文件编辑器命令、模型理解的搜索语法）。

### SWE-bench Verified

OpenAI，2024 年 8 月。人工策展的 500 任务子集。移除了模糊的 issue、不可靠的测试以及修复不明确的任务。用于"你的 agent 能否交付真正的补丁？"的主要基准测试。

### 污染

- 超过 94% 的 SWE-bench issue 早于大多数模型的截止日期。
- **SWE-bench+** 发现 32.67% 的成功补丁在 issue 文本中泄露了解决方案（模型在描述中看到了修复），31.08% 由于测试覆盖薄弱而可疑。
- Verified 更干净但并非无污染。

实际影响：一个在 SWE-bench 上得 50% 的模型可能在 SWE-bench+ 上得 35%。如果你声称 SWE-bench 性能，始终报告两者。

### GAIA（Mialon 等人，2023 年 11 月）

- 466 个问题；300 个保留用于 huggingface.co/gaia-benchmark 的私有排行榜。
- 设计哲学："对人类概念上简单（92%）但对 AI 困难（GPT-4 加插件：15%）。"
- 测试推理、多模态、网页、工具使用。
- 三个难度级别；Level 3 需要跨模态的长工具链。

GAIA 是你运行以衡量"通用能力"的测试。不要与特定代码基准测试混淆。

### AgentBench（Liu 等人，ICLR 2024）

- 8 个环境覆盖代码（Bash、DB、KG）、游戏（Alfworld、LTP）、网页（WebShop、Mind2Web）和开放式生成。
- 多轮，每个分割约 4k-13k 轮。
- 主要发现：长期推理、决策制定和指令遵循是开源 LLM 追赶商业模型的障碍。

### 这些不测量什么

- 真实世界的运营成本（token、墙钟时间）。
- 对抗条件下的安全行为。
- 在你领域上的表现（使用你自己的评估，第 30 课）。
- 尾部失败（基准测试取平均；生产运维人员关心最差的 1%）。

### 基准测试可能出错的地方

- **单一数字执念。** SWE-bench 50% 告诉你的比 P50/P75/P95 成本 + 步骤分布要少。
- **污染声明。** 报告 SWE-bench 而不提及 Verified 或 SWE-bench+ 是具有误导性的。
- **基准测试作为开发目标。** 为基准测试优化会偏离生产有用性。

## 构建它

`code/main.py` 实现了一个玩具 SWE-bench 风格的 harness：

- 合成 bug 修复任务（3 个任务）。
- 一个脚本化的"agent"，提出补丁。
- 一个测试运行器，检查 FAIL_TO_PASS（bug 现已修复）和 PASS_TO_PASS（没有破坏任何东西）。
- 一个基于问题分解深度的 GAIA 风格难度分类器。

运行它：

```
python3 code/main.py
```

输出显示每个任务的解决率 + 每个难度级别，并使评估器规则具体化。

## 使用它

- **SWE-bench Verified** 用于代码 agent。始终报告 Verified 分数。
- **GAIA** 用于通用 agent。使用私有排行榜分割。
- **AgentBench** 用于多环境比较。
- **自定义评估**（第 30 课）用于你产品的实际形态。

## 交付它

`outputs/skill-benchmark-harness.md` 为任何代码库-任务对构建一个 SWE-bench 风格的 harness，具有 FAIL_TO_PASS / PASS_TO_PASS 门控。

## 练习

1. 将玩具 harness 移植到在真实仓库上运行（选择你自己的一个）。为已知 bug 编写 3 个 FAIL_TO_PASS 测试。
2. 添加步骤计数指标。在你的 3 个任务上，每次解决需要多少 agent 步骤？
3. 阅读 SWE-bench+ 论文。实现一个解决方案泄露检查（将 issue 文本与 diff 进行模式匹配）。
4. 从公开分割下载一个 GAIA 问题。追踪 GPT-4 级别的 agent 会做什么。它需要什么工具？
5. 阅读 AgentBench 的按环境细分。哪个环境映射到你的产品表面？那里的"SOTA"是什么样子？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| SWE-bench | "代码 agent 基准测试" | 2,294 个 GitHub issue；补丁必须翻转 FAIL_TO_PASS 测试 |
| SWE-bench Verified | "干净的 SWE-bench" | 500 个人工策展任务，OpenAI |
| FAIL_TO_PASS | "修复门控" | 之前失败、应用补丁后必须通过的测试 |
| PASS_TO_PASS | "无回归门控" | 之前通过且必须仍然通过的测试 |
| GAIA | "通用基准测试" | 466 个人类容易 / AI 困难的多工具问题 |
| AgentBench | "多环境基准测试" | 8 个环境；长时间多轮 |
| 污染（Contamination） | "训练集泄露" | 基准测试任务出现在模型训练中 |
| SWE-bench+ | "污染审计" | 在成功的 SWE-bench 补丁中发现 32.67% 解决方案泄露 |

## 进一步阅读

- [Jimenez 等人，SWE-bench（arXiv:2310.06770）](https://arxiv.org/abs/2310.06770) — 原始基准测试
- [OpenAI，SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — 策展子集
- [Mialon 等人，GAIA（arXiv:2311.12983）](https://arxiv.org/abs/2311.12983) — 通用基准测试
- [Liu 等人，AgentBench（arXiv:2308.03688）](https://arxiv.org/abs/2308.03688) — 多环境套件
