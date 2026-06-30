# Self-Refine 与 CRITIC：迭代输出改进

> Self-Refine（Madaan et al., 2023）在循环中使用一个 LLM 扮演三个角色——生成、反馈、精炼。平均收益：7 个任务上 +20 绝对提升。CRITIC（Gou et al., 2023）通过将验证路由到外部工具来加强反馈步骤。2026 年，此模式在每个框架中作为 "evaluator-optimizer"（Anthropic）或 guardrail 循环（OpenAI Agents SDK）交付。

**类型:** Build
**语言:** Python（标准库）
**前置要求:** Phase 14 · 01（Agent Loop）, Phase 14 · 03（Reflexion）
**时间:** ~60 分钟

## 学习目标

- 陈述 Self-Refine 的三个提示（生成、反馈、精炼）并解释为什么历史对精炼提示很重要。
- 解释 CRITIC 的关键洞见：LLM 在没有外部锚定情况下进行自我验证是不可靠的。
- 实现一个标准库 Self-Refine 循环，带有历史记录和可选的外部验证器。
- 将此模式映射到 Anthropic 的 "evaluator-optimizer" 工作流和 OpenAI Agents SDK 的输出 guardrail。

## 问题

一个 agent 产生了一个几乎正确的答案。也许一行代码有语法错误。也许摘要太长。也许一个计划漏掉了一个边界情况。你想要的是：agent 批评自己的输出，然后修复它。

Self-Refine 展示这可以通过单个模型做到，无需训练数据，无需 RL。但有一个问题：LLM 在硬事实上不擅长自我验证。CRITIC 给出了解决办法——将验证步骤通过外部工具路由（搜索、代码解释器、计算器、测试运行器）。

这两篇论文共同定义了 2026 年迭代改进的默认做法：生成、验证（尽可能外部）、精炼、验证器通过时停止。

## 概念

### Self-Refine（Madaan et al., NeurIPS 2023）

一个 LLM，三个角色：

```
generate(task)            -> output_0
feedback(task, output_0)  -> critique_0
refine(task, output_0, critique_0, history) -> output_1
feedback(task, output_1)  -> critique_1
refine(task, output_1, critique_1, history) -> output_2
...
当 feedback 说"没有问题"或预算耗尽时停止。
```

关键细节：`refine` 看到完整历史——所有先前的输出和批评——这样它不会重复错误。论文消融了这一点：去掉历史，质量急剧下降。

头条：跨 7 个任务（数学、代码、缩写词、对话）平均 +20 绝对改进，包括 GPT-4。没有训练，没有外部工具，单个模型。

### CRITIC（Gou et al., arXiv:2305.11738, v4 2024 年 2 月）

Self-Refine 的弱点：反馈步骤是 LLM 对自己评分。对于事实性声明这是不可靠的（幻觉在产生它的模型看来往往很有说服力）。CRITIC 将 `feedback(task, output)` 替换为 `verify(task, output, tools)`，其中 `tools` 包括：

- 搜索引擎用于事实性声明。
- 代码解释器用于代码正确性。
- 计算器用于算术。
- 领域特定验证器（单元测试、类型检查器、linter）。

验证器产生一个锚定在工具结果中的结构化批评。精炼器然后依据此批评进行条件化。

头条：在事实性任务上 CRITIC 优于 Self-Refine，因为批评是有锚定的。在没有外部验证器的任务上（创意写作、格式化），CRITIC 退化为 Self-Refine。

### 停止条件

两种常见形态：

1. **验证器通过。** 外部测试返回成功。可用时首选（单元测试、类型检查器、guardrail 断言）。
2. **没有反馈发出。** 模型说"输出没问题。"更便宜但不可靠；配合最大迭代上限。

2026 年默认：结合它们。"如果验证器通过 OR 模型说没问题 AND 迭代次数 >=2 OR 迭代次数 >= max_iterations 则停止。"

### Evaluator-Optimizer（Anthropic, 2024）

Anthropic 2024 年 12 月的文章将此命名为五种工作流模式之一。两个角色：

- Evaluator：对输出评分并产生批评。
- Optimizer：根据批评修订输出。

循环直到评估器通过。这是 Anthropic 框架下的 Self-Refine/CRITIC。Anthropic 增加的关键工程细节：评估器和优化器提示应当显著不同，这样模型不只是橡皮图章式通过。

### OpenAI Agents SDK 输出 guardrail

OpenAI Agents SDK 将此模式作为"输出 guardrail"提供。Guardrail 是对 agent 最终输出运行的验证器。如果 guardrail 触发（引发 `OutputGuardrailTripwireTriggered`），输出被拒绝，agent 可以重试。Guardrail 可以调用工具（CRITIC 风格）或作为纯函数（Self-Refine 风格）。

### 2026 年的陷阱

- **橡皮图章循环。** 同一模型用相同的提示风格做生成和批评，收敛到"看起来不错"。使用结构上不同的提示，或使用更小更便宜的模型做批评。
- **过度精炼。** 每次精炼通行增加延迟和 token。预算 1-3 次通行；之后升级到人工审查。
- **在琐碎任务上使用 CRITIC。** 如果没有外部验证器，CRITIC 退化为 Self-Refine；不要为存根验证器支付延迟代价。

## Build It

`code/main.py` 在一个玩具任务上实现 Self-Refine 和 CRITIC：根据给定主题生成一个简短的要点列表。验证器检查格式（3 个要点，每个 60 个字符以内）。CRITIC 增加了一个外部"事实验证器"，惩罚已知的幻觉。

组件：

- `generate` — 脚本化生产者。
- `feedback` — LLM 风格自我批评。
- `verify_external` — CRITIC 风格锚定验证器。
- `refine` — 给定历史重写输出。
- 停止条件 — 验证器通过或最多 4 次迭代。

运行它：

```
python3 code/main.py
```

比较 Self-Refine 和 CRITIC 的运行。CRITIC 捕获了一个 Self-Refine 漏掉的事实性错误，因为外部验证器具有自我批评者不具备的锚定。

## Use It

Anthropic 的 evaluator-optimizer 是此模式在 Claude 友好语言中的形态。OpenAI Agents SDK 的输出 guardrail 是 CRITIC 形态的（guardrail 可以调用工具）。LangGraph 提供一个读起来像 Self-Refine 的反思节点。Google 的 Gemini 2.5 Computer Use 增加了一个每步安全评估器，这是 CRITIC 的一个变体：每个动作在提交前被验证。

## Ship It

`outputs/skill-refine-loop.md` 根据任务形状、验证器可用性和迭代预算配置 evaluator-optimizer 循环。发出生成器、评估器/验证器和优化器的提示，以及停止策略。

## 练习

1. 用 max_iterations=1 运行玩具。CRITIC 仍然有帮助吗？
2. 用有噪声的验证器替换外部验证器（随机 30% 误报）。循环会做什么？这是 2026 年大多数 guardrail 栈的现实。
3. 实现一个"不同模型上的生成器-批评者"变体：大模型生成，小模型批评。它能击败同模型吗？
4. 阅读 CRITIC 第 3 节（arXiv:2305.11738 v4）。说出三个验证工具类别并为每个举一个例子。
5. 将 OpenAI Agents SDK 的 `output_guardrails` 映射到 CRITIC 的验证器角色。SDK 什么做得对，什么做得不对？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| Self-Refine | "能自我修正的 LLM" | 在一个模型中生成 -> 反馈 -> 精炼的循环，带历史记录 |
| CRITIC | "工具锚定验证" | 用外部验证器替换反馈（搜索、代码、计算、测试） |
| Evaluator-Optimizer | "Anthropic 工作流模式" | 两个角色——评估器打分，优化器修订——循环至收敛 |
| 输出 guardrail | "事后检查" | OpenAI Agents SDK 验证器，在 agent 产生输出后运行 |
| 验证步骤 | "批评阶段" | 承重决策：锚定还是自我评分 |
| 精炼历史 | "模型已经尝试过的" | 先前输出 + 批评被前置到精炼提示中；去掉则质量崩溃 |
| 橡皮图章循环 | "自我同意失败" | 相同提示的批评返回"看起来不错"；用结构上不同的提示修复 |
| 停止条件 | "收敛测试" | 验证器通过 OR 无反馈 AND 迭代上限；永远不要单一条件 |

## 进一步阅读

- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) — 经典论文
- [Gou et al., CRITIC (arXiv:2305.11738)](https://arxiv.org/abs/2305.11738) — 工具锚定验证
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — evaluator-optimizer 工作流模式
- [OpenAI Agents SDK 文档](https://openai.github.io/openai-agents-python/) — 作为 CRITIC 形态验证器的输出 guardrail
