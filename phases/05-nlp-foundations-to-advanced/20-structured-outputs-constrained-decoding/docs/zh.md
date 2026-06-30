# 结构化输出与约束解码

> 向 LLM 请求 JSON。大多数时候得到 JSON。在生产中，"大多数"就是问题。约束解码通过在采样前编辑 logits 将"大多数"变成"始终"。

**类型：** 构建
**语言：** Python
**先修要求：** 第五阶段 · 17（聊天机器人），第五阶段 · 19（子词 Tokenization）
**预计时间：** 约60分钟

## 问题

一个分类器提示 LLM："Return one of {positive, negative, neutral}。"模型返回"The sentiment is positive — this review is overwhelmingly favorable because the customer explicitly states that they ..."。你的解析器崩溃。你的分类器的 F1 是 0.0。

自由形式生成不是一个契约。它是一个建议。一个生产系统需要一个契约。

2026 年存在三个层次。

1. **提示。** 礼貌地请求。"Return only the JSON object."在前沿模型上大约 80% 有效，在较小的模型上效果较差。
2. **原生结构化输出 API。** OpenAI `response_format`、Anthropic tool use、Gemini JSON mode。对支持的 schema 可靠。供应商锁定。
3. **约束解码。** 在每个生成步骤修改 logits，使模型*不能*发出无效的 token。通过构造 100% 有效。适用于任何本地模型。

本课为三者建立直觉，并指出何时选用哪种。

## 概念

![Constrained decoding masking invalid tokens at each step](../assets/constrained-decoding.svg)

**约束解码如何工作。** 在每个生成步骤，LLM 在整个词汇表（约 100k token）上产生一个 logit 向量。一个*logit 处理器*位于模型和采样器之间。它计算给定目标文法中当前位置哪些 token 是有效的——JSON Schema、正则表达式、上下文无关文法——并将所有无效 token 的 logits 设置为负无穷。对剩余 logits 的 softmax 仅将概率质量放在有效延续上。

2026 年的实现：

- **Outlines。** 将 JSON Schema 或正则表达式编译为有限状态机。每个 token 获得 O(1) 的有效下一个 token 查找。基于 FSM，因此递归 schema 需要展开。
- **XGrammar / llguidance。** 上下文无关文法引擎。处理递归 JSON Schema。近乎零解码开销。OpenAI 在其 2025 年结构化输出实现中认可了 llguidance。
- **vLLM 引导解码。** 内置`guided_json`、`guided_regex`、`guided_choice`、`guided_grammar`，通过 Outlines、XGrammar 或 lm-format-enforcer 后端。
- **Instructor。** 基于 Pydantic 的包装器，适用于任何 LLM。验证失败时重试。跨提供商，但不修改 logits——它依赖重试 + 结构化输出感知的提示。

### 违反直觉的结果

约束解码通常比无线束生成*更快*。两个原因。首先，它缩小了下一个 token 搜索空间。其次，聪明的实现对于强制 token（如`{"name": "`的脚手架——每个字节都是确定的）完全跳过 token 生成。

### 让你付出代价的陷阱

字段顺序很重要。将`answer`放在`reasoning`之前，模型在思考之前就承诺了答案。JSON 是有效的。答案是错误的。没有验证能捕获它。

```json
// BAD
{"answer": "yes", "reasoning": "because ..."}

// GOOD
{"reasoning": "... therefore ...", "answer": "yes"}
```

Schema 字段顺序是逻辑，不是格式。

## 构建它

### 步骤 1：从零实现正则表达式约束生成

参见`code/main.py`获取独立的 FSM 实现。核心思想 30 行：

```python
def mask_logits(logits, valid_token_ids):
    mask = [float("-inf")] * len(logits)
    for tid in valid_token_ids:
        mask[tid] = logits[tid]
    return mask


def generate_constrained(model, tokenizer, prompt, fsm):
    ids = tokenizer.encode(prompt)
    state = fsm.initial_state
    while not fsm.is_accept(state):
        logits = model.next_token_logits(ids)
        valid = fsm.valid_tokens(state, tokenizer)
        logits = mask_logits(logits, valid)
        tok = sample(logits)
        ids.append(tok)
        state = fsm.transition(state, tok)
    return tokenizer.decode(ids)
```

FSM 跟踪到目前为止我们已满足本文法的哪些部分。`valid_tokens(state, tokenizer)`计算哪些词汇表 token 可以在不离开接受路径的情况下推进 FSM。

### 步骤 2：Outlines 用于 JSON Schema

```python
from pydantic import BaseModel
from typing import Literal
import outlines


class Review(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    evidence_span: str


model = outlines.models.transformers("meta-llama/Llama-3.2-3B-Instruct")
generator = outlines.generate.json(model, Review)

result = generator("Classify: 'The wait staff was attentive and the food arrived hot.'")
print(result)
# Review(sentiment='positive', confidence=0.93, evidence_span='attentive ... hot')
```

零验证错误。永远。FSM 使无效输出不可达。

### 步骤 3：Instructor 用于跨提供商的 Pydantic

```python
import instructor
from anthropic import Anthropic
from pydantic import BaseModel, Field


class Invoice(BaseModel):
    vendor: str
    total_usd: float = Field(ge=0)
    line_items: list[str]


client = instructor.from_anthropic(Anthropic())
invoice = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=1024,
    response_model=Invoice,
    messages=[{"role": "user", "content": "Extract from: 'Acme Corp $420. Widget, Gizmo.'"}],
)
```

不同的机制。Instructor 不触及 logits。它将 schema 格式化到提示中，解析输出，并在验证失败时重试（默认 3 次）。适用于任何提供商。重试增加延迟和成本。跨提供商可移植性是卖点。

### 步骤 4：原生供应商 API

```python
from openai import OpenAI

client = OpenAI()
response = client.responses.create(
    model="gpt-5",
    input=[{"role": "user", "content": "Classify: 'The food was cold.'"}],
    text={"format": {"type": "json_schema", "name": "sentiment",
          "schema": {"type": "object", "required": ["sentiment"],
                     "properties": {"sentiment": {"type": "string",
                                                  "enum": ["positive", "negative", "neutral"]}}}}},
)
print(response.output_parsed)
```

服务端约束解码。支持的 schema 在可靠性与 Outlines 持平。无需本地模型管理。将你锁定到供应商。

## 陷阱

- **递归 schema。** Outlines 将递归展开为固定深度。树状输出（嵌套评论、AST）需要 XGrammar 或 llguidance（基于 CFG）。
- **巨大枚举。** 10,000 选项的枚举编译缓慢或超时。切换到检索器：首先预测 top-k 候选，约束到这些。
- **文法过于严格。** 强制`date: "YYYY-MM-DD"`正则表达式，模型不能为缺失日期输出`"unknown"`。模型通过编造日期来补偿。允许`null`或一个哨兵值。
- **过早承诺。** 见上述字段顺序陷阱。始终先放推理。
- **不带 schema 的供应商 JSON 模式。** 纯 JSON 模式仅保证有效 JSON，不保证对你的用例*有效*。始终提供完整 schema。

## 使用它

2026 年技术栈：

| 场景 | 选择 |
|-----------|------|
| OpenAI/Anthropic/Google 模型，简单 schema | 原生供应商结构化输出 |
| 任何提供商，Pydantic 工作流，可容忍重试 | Instructor |
| 本地模型，需要 100% 有效性，平坦 schema | Outlines（FSM） |
| 本地模型，递归 schema | XGrammar 或 llguidance |
| 自托管推理服务器 | vLLM 引导解码 |
| 可接受重试的批处理 | Instructor + 最便宜的模型 |

## 交付它

保存为 `outputs/skill-structured-output-picker.md`：

```markdown
---
name: structured-output-picker
description: 选择结构化输出方法、schema 设计和验证计划。
version: 1.0.0
phase: 5
lesson: 20
tags: [nlp, llm, structured-output]
---

给定一个用例（提供商、延迟预算、schema 复杂度、失败容忍度），输出：

1. 机制。原生供应商结构化输出、Instructor 重试、Outlines FSM 或 XGrammar CFG。一句理由。
2. Schema 设计。字段顺序（推理第一，答案最后）、用于"未知"的可空字段、枚举 vs 正则表达式、必需字段。
3. 失败策略。最大重试次数、后退模型、优雅的`null`处理、分布外拒绝。
4. 验证计划。Schema 合规率（目标 100%）、语义有效性（LLM 评判）、字段覆盖率、延迟 p50/p99。

拒绝任何将`answer`或`decision`放在推理字段之前的设计。拒绝在没有 schema 的情况下使用裸 JSON 模式。标记仅 FSM 库后的递归 schema。
```

## 练习

1. **简单。** 在没有约束解码的情况下提示一个小型开放权重模型（例如 Llama-3.2-3B）用于`Review(sentiment, confidence, evidence_span)`。在 100 条评论上测量被解析为有效 JSON 的比例。
2. **中等。** 相同的语料库使用 Outlines JSON 模式。比较合规率、延迟和语义准确率。
3. **困难。** 从零为电话号码（`\d{3}-\d{3}-\d{4}`）实现一个正则表达式约束解码器。在 1000 个样本上验证 0 个无效输出。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| 约束解码 | 强制有效输出 | 在每个生成步骤掩盖无效 token 的 logits。 |
| Logit 处理器 | 约束者 | 函数：`(logits, state) -> masked_logits`。 |
| FSM | 有限状态机 | 编译的文法表示；O(1) 有效下一个 token 查找。 |
| CFG | 上下文无关文法 | 处理递归的文法；比 FSM 慢但更具表达力。 |
| Schema 字段顺序 | 重要吗？ | 是的——第一个字段承诺；始终将推理放在答案之前。 |
| 引导解码 | vLLM 对此的称呼 | 相同的概念，集成到推理服务器中。 |
| JSON 模式 | OpenAI 的早期版本 | 保证 JSON 语法；不保证 schema 匹配。 |

## 扩展阅读

- [Willard, Louf (2023). Efficient Guided Generation for LLMs](https://arxiv.org/abs/2307.09702) — Outlines 论文。
- [XGrammar paper (2024)](https://arxiv.org/abs/2411.15100) — 快速的基于 CFG 的约束解码。
- [vLLM — Structured Outputs](https://docs.vllm.ai/en/latest/features/structured_outputs.html) — 推理服务器集成。
- [OpenAI — Structured Outputs guide](https://platform.openai.com/docs/guides/structured-outputs) — API 参考 + 陷阱。
- [Instructor library](https://python.useinstructor.com/) — 跨提供商的 Pydantic + 重试。
- [JSONSchemaBench (2025)](https://arxiv.org/abs/2501.10868) — 对 6 个约束解码框架的基准测试。
