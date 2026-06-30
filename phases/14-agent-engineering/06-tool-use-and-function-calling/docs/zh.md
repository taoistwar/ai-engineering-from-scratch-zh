# 工具使用与函数调用

> Toolformer（Schick et al., 2023）启动了自监督工具标注。Berkeley Function Calling Leaderboard V4（Patil et al., 2025）设定了 2026 年的标准：40% agentic、30% 多轮、10% live、10% non-live、10% 幻觉。单轮已解决。记忆、动态决策和长周期工具链还没有。

**类型:** Build
**语言:** Python（标准库）
**前置要求:** Phase 14 · 01（Agent Loop）, Phase 13 · 01（Function Calling Deep Dive）
**时间:** ~60 分钟

## 学习目标

- 解释 Toolformer 的自监督训练信号：仅在工具执行降低下一个 token 的损失时保留工具标注。
- 说出 BFCL V4 的五个评估类别以及每个类别测量什么。
- 实现一个标准库工具注册表，包含模式验证、参数强制转换和执行沙盒。
- 诊断三个 2026 年的开放问题：长周期工具链、动态决策和记忆。

## 问题

早期的工具使用问：模型能预测正确的函数调用吗？现代工具使用问：模型能跨 40 步链式调用工具，带有记忆、部分可观测性、从工具失败中恢复，且不虚构不存在的工具吗？

Toolformer 建立了基线：模型可以通过自监督学习何时调用工具。BFCL V4 定义了 2026 年的评估目标。它们之间的差距是生产级 agent 所处的空间。

## 概念

### Toolformer（Schick et al., NeurIPS 2023）

想法：让模型用自己的预训练语料库标注候选的 API 调用。对每个候选，执行它。仅当包含工具结果降低下一个 token 的损失时才保留该标注。在过滤后的语料库上微调。

覆盖的工具：计算器、问答系统、搜索引擎、翻译器、日历。自监督信号纯粹是关于工具是否有助于预测文本——没有人工标签。

规模结果：工具使用在规模上涌现。更小的模型在工具标注下受损；更大的模型获益。这就是为什么 2026 年前沿模型有强大的内置工具使用能力，而大多数 7B 模型需要显式的工具使用微调才能可靠。

### Berkeley Function Calling Leaderboard V4（Patil et al., ICML 2025）

BFCL 是 2026 年事实上的评估标准。V4 构成：

- **Agentic（40%）** — 完整 agent 轨迹：记忆、多轮、动态决策。
- **Multi-Turn（30%）** — 带有工具链的交互对话。
- **Live（10%）** — 用户提交的真实提示（更困难的分布）。
- **Non-Live（10%）** — 合成测试用例。
- **Hallucination（10%）** — 检测何时不应调用工具。

V3 引入了基于状态的评估：在工具序列之后，检查 API 的实际状态（例如"文件是否已创建？"）而不是匹配工具调用的 AST。V4 添加了网页搜索、记忆和格式敏感类别。

2026 年的关键发现：单轮函数调用接近已解决。失败集中在记忆（跨轮传递上下文）、动态决策（根据先前结果选择工具）、长周期链（20+ 步后的漂移）和幻觉检测（没有合适工具时拒绝调用）。

### 工具模式

每个提供商都有一个模式。它们在细节上不同但共享相同的形状：

```
name: string
description: string（做什么，何时使用）
input_schema: JSON Schema（属性、必需字段、类型、枚举）
```

Anthropic 直接使用 `input_schema`。OpenAI 使用 `function.parameters`。两者都接受 JSON Schema。描述是承重的——模型读取它们来选择正确的工具。糟糕的工具描述是选错工具失败的首要根因。

### 参数验证

不信任任何工具调用。验证：

1. **类型强制转换。** 模型可能返回字符串 "5" 而模式说 int。如果无歧义则强制转换；如果有歧义则拒绝。
2. **枚举验证。** 如果模式说 `status in {"open", "closed"}` 而模型发出 `"in_progress"`，用描述性错误拒绝。
3. **必填字段。** 缺失必填字段 -> 立即返回错误观察给模型，而不是崩溃。
4. **格式验证。** 日期、邮箱、URL —— 用具体解析器验证，不用正则。

每个验证失败都应返回结构化观察，使模型可以用正确的形状重试。

### 并行工具调用

现代提供商支持在一个 assistant 回合中进行并行工具调用。循环：

1. 模型发出 3 个工具调用，带有不同的 `tool_use_id`。
2. 运行时执行它们（如果独立则并行）。
3. 每个结果作为 `tool_result` 块返回，通过 `tool_use_id` 关联。

工程规则：将关联 ID 视为承重的。交换它们会导致错误工具到错误结果的路由。

### 沙盒

工具执行是沙盒边界。详情见第 09 课。简短版本：每个工具应指定读/写面、网络访问、超时、内存上限。通用的 `run_shell(cmd)` 是红旗；具体的 `git_status()` 更安全。

```figure
tool-routing
```

## Build It

`code/main.py` 实现一个生产形态的工具注册表：

- JSON Schema 子集验证器（仅标准库）。
- 工具注册，包含描述、输入模式、超时和执行器。
- 参数强制转换和枚举验证。
- 带有关联 ID 的并行工具调度。
- 作为结构化字符串的错误观察。

运行它：

```
python3 code/main.py
```

追踪显示一个微型 agent 在一个回合中调用三个工具，其中一个故意格式错误的调用被带有描述性错误的拒绝，模型可以据此采取行动。

## Use It

每个提供商都有自己的工具模式 —— Anthropic、OpenAI、Gemini、Bedrock。如果你需要多提供商，使用一个翻译层（OpenAI Agents SDK、Vercel AI SDK、LangChain 工具适配器）。BFCL 是参考基准 —— 如果工具使用对产品至关重要，在发布前对你的 agent 运行它。

## Ship It

`outputs/skill-tool-registry.md` 为给定任务领域生成工具目录、模式和注册表。包括描述质量检查（每个工具的描述是否告诉模型何时使用它？）。

## 练习

1. 添加一个"no-op"工具，让模型显式拒绝使用任何其他工具。在类似 BFCL 的幻觉测试上测量。
2. 为 int-as-string 和 float-as-string 实现参数强制转换。强制转换从什么时候开始隐藏真正的 bug？
3. 为每个工具添加超时和断路器（三次连续失败后拒绝该工具 60 秒）。这改变了模型恢复的方式吗？
4. 阅读 BFCL V4 描述。选择一个类别（例如"multi-turn"）并通过你的 agent 运行 10 个示例提示。报告通过率。
5. 将标准库验证器移植到 Pydantic 或 Zod。Pydantic/Zod 捕获了玩具漏掉的什么？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| 函数调用 | "工具使用" | 带有验证模式的结构化输出工具调用 |
| Toolformer | "自监督工具标注" | Schick 2023 — 保留其结果能降低下一个 token 损失的工具调用 |
| BFCL | "Berkeley Function Calling Leaderboard" | 2026 基准：40% agentic、30% 多轮、10% live、10% non-live、10% 幻觉 |
| 工具模式 | "模型的功能签名" | name、description、参数的 JSON Schema |
| tool_use_id | "关联 ID" | 将工具调用绑定到其结果；并行调度必不可少 |
| 幻觉检测 | "知道何时不调用" | V4 类别：没有合适工具时拒绝调用 |
| 参数强制转换 | "字符串到整数修复" | 对可预测的模式不匹配的窄修复；有歧义时拒绝 |
| 沙盒 | "工具执行边界" | 每个工具的读/写面、网络、超时、内存上限 |

## 进一步阅读

- [Schick et al., Toolformer (arXiv:2302.04761)](https://arxiv.org/abs/2302.04761) — 自监督工具标注
- [Berkeley Function Calling Leaderboard (V4)](https://gorilla.cs.berkeley.edu/leaderboard.html) — 2026 评估基准
- [Anthropic, 工具使用文档](https://platform.claude.com/docs/en/agent-sdk/overview) — Claude Agent SDK 中的生产工具模式
- [OpenAI Agents SDK 文档](https://openai.github.io/openai-agents-python/) — 函数工具类型和 Guardrail
