# 结构化输出——JSON Schema、Pydantic、Zod、受限解码

> "友好地要求模型返回 JSON" 即使在最前沿的模型上也有 5% 到 15% 的失败率。结构化输出通过受限解码弥合了差距：模型在字面上被阻止发出会违反 schema 的 token。OpenAI 的严格模式、Anthropic 的 schema 类型工具使用、Gemini 的 `responseSchema`、Pydantic AI 的 `output_type` 和 Zod 的 `.parse` 是同一想法的五种表面形式。本课构建 schema 验证器和学习者在每个生产提取流水线中将使用的严格模式契约。

**Type:** Build
**Languages:** Python (stdlib, JSON Schema 2020-12 subset)
**Prerequisites:** Phase 13 · 02 (function calling deep dive)
**Time:** ~75 minutes

## 学习目标

- 使用正确的约束（enum、min/max、required、pattern）为提取目标编写 JSON Schema 2020-12。
- 解释为什么严格模式和受限解码提供了与"生成后验证"不同的保证。
- 区分三种失败模式：解析错误、schema 违规、模型拒绝。
- 发布具有类型化修复和类型化拒绝处理的提取流水线。

## 问题

一个读取采购订单电子邮件的代理需要将自由文本转换为 `{customer, line_items, total_usd}`。三种方法。

**方法一：提示输出 JSON。** "以包含 customer、line_items、total_usd 字段的 JSON 格式回复。"在前沿模型上 85% 到 95% 的时间可行。有六种失败方式：缺失花括号、尾部逗号、错误类型、虚构字段、在 token 限制处截断、泄露的散文如"这是你的 JSON："。

**方法二：生成后验证。** 自由生成，解析，验证是否与 schema 一致，失败时重试。可靠但昂贵——你为每次重试付费，而截断错误每次发生需要额外的一次回合。

**方法三：受限解码。** 提供商在解码时强制 schema。无效 token 从采样分布中被掩码掉。输出保证可解析且保证通过验证。失败坍缩为一种模式：拒绝（模型判定输入不适合 schema）。

2026 年的每个前沿提供商都提供某种形式的方法三。

- **OpenAI。** `response_format: {type: "json_schema", strict: true}`，如果模型拒绝则响应中有 `refusal`。
- **Anthropic。** 对 `tool_use` 输入的 schema 强制；`stop_reason: "refusal"` 不是一回事，但 `end_turn` 不带工具调用是信号。
- **Gemini。** 请求级别的 `responseSchema`；在 2026 年 Gemini 对选定类型提供 token 级文法约束。
- **Pydantic AI。** `output_type=InvoiceModel` 发出类型化为 `InvoiceModel` 的结构化 `RunResult`。
- **Zod（TypeScript）。** 运行时解析器，根据 Zod schema 验证提供商的输出；与 OpenAI 的 `beta.chat.completions.parse` 配对。

共同线索：声明 schema 一次，端到端强制它。

## 概念

### JSON Schema 2020-12——通用语

每个提供商都接受 JSON Schema 2020-12。最常用的构造：

- `type`：`object`、`array`、`string`、`number`、`integer`、`boolean`、`null` 之一。
- `properties`：字段名到子 schema 的映射。
- `required`：必须出现的字段名列表。
- `enum`：允许值的封闭集合。
- `minimum` / `maximum`（数字），`minLength` / `maxLength` / `pattern`（字符串）。
- `items`：应用于每个数组元素的子 schema。
- `additionalProperties`：`false` 禁止额外字段（默认因模式而异）。

OpenAI 严格模式添加了三个要求：每个属性必须列在 `required` 中，到处 `additionalProperties: false`，且无未解析的 `$ref`。如果你违反了这些，API 在请求时返回 400。

### Pydantic，Python 绑定

Pydantic v2 通过 `model_json_schema()` 从 dataclass 形状的模型生成 JSON Schema。Pydantic AI 包装了这一点，所以你可以这样写：

```python
class Invoice(BaseModel):
    customer: str
    line_items: list[LineItem]
    total_usd: Decimal
```

代理框架在边缘将 schema 翻译为 OpenAI 严格模式、Anthropic `input_schema` 或 Gemini `responseSchema`。模型的输出作为类型化的 `Invoice` 实例返回。验证错误引发 `ValidationError`，带有类型化的错误路径。

### Zod，TypeScript 绑定

Zod（`z.object({customer: z.string(), ...})`）是 TS 等价物。OpenAI 的 Node SDK 暴露了 `zodResponseFormat(Invoice)`，将其翻译为 API 的 JSON Schema 载荷。

### 拒绝

严格模式不能强迫模型回答。如果输入无法适合 schema（"这封电子邮件是一首诗，不是发票"），模型发回一个包含原因的 `refusal` 字段。你的代码必须将其作为一等结果处理，而非作为失败。拒绝也可用作安全信号：被要求从受保护内容电子邮件中提取信用卡号的模型返回带有附加安全原因的拒绝。

### 开源中的受限解码

开源权重的实现使用三种技术。

1. **基于文法的解码**（`outlines`、`guidance`、`lm-format-enforcer`）：从 schema 构建确定性的有限状态自动机；每一步，掩码掉违反 FSM 的 token 的 logits。
2. **使用 JSON 解析器的 logit 掩码**：与模型同步运行流式 JSON 解析器；每一步，计算有效的下一个 token 集合。
3. **带验证器的推测解码**：廉价的草稿模型提出 token，验证器强制 schema。

商业提供商在幕后选择其中之一。2026 年最先进技术对于短结构化输出比普通生成更快，对于长输出速度大致相同。

### 三种失败模式

1. **解析错误。** 输出不是有效的 JSON。在严格模式下不可能发生。在非严格提供商上仍可能发生。
2. **Schema 违规。** 输出解析成功但违反 schema。在严格模式下不可能发生。在之外常见。
3. **拒绝。** 模型拒绝。必须作为类型化结果处理。

### 重试策略

当你在严格模式之外时（Anthropic 工具使用、非严格 OpenAI、较老的 Gemini），恢复模式是：

```
生成 -> 解析 -> 验证 -> 如果失败，注入错误并重试，最多 3 次
```

一次重试通常足够。三次重试能捕获弱模型的异常。超过三次是 schema 有问题的标志：模型对某些输入无法满足它，提示或 schema 需要修复。

### 小模型支持

受限解码在小模型上有效。一个 3B 参数的开源模型加上文法强制，在结构化任务上超越一个 70B 参数模型的原始提示。这是结构化输出对生产环境重要的主要原因：它将可靠性与模型大小解耦。

## 使用

`code/main.py` 发布了一个标准库中最小的 JSON Schema 2020-12 验证器（类型、required、enum、min/max、pattern、items、additionalProperties）。它包装一个 `Invoice` schema，并通过验证器运行假的 LLM 输出，演示解析错误、schema 违规和拒绝路径。在生产中可将假输出替换为任何提供商的真实响应。

需要注意的地方：

- 验证器返回带有路径和消息的类型化 `[ValidationError]` 列表。这是你希望暴露给重试提示的形状。
- 拒绝分支不重试。它记录并返回一个类型化的拒绝。Phase 14 · 09 将拒绝用作安全信号。
- `additionalProperties: false` 检查在对抗性测试输入上触发，展示了为什么严格模式关闭了虚构字段的大门。

## 产出

本课产出 `outputs/skill-structured-output-designer.md`。给定一个自由文本提取目标（发票、支持工单、简历等），该技能产生一个兼容严格模式的 JSON Schema 2020-12 和一个镜像它的 Pydantic 模型，并带有类型化的拒绝和重试处理骨架。

## 练习

1. 运行 `code/main.py`。添加一个 `total_usd` 为负数的第四个测试用例。确认验证器以 `minimum` 约束路径拒绝它。

2. 扩展验证器以支持带判别器的 `oneOf`。常见情况：`line_item` 是产品或服务，由 `kind` 标记。严格模式对此有微妙规则；查看 OpenAI 的结构化输出指南。

3. 编写相同的 Invoice schema 作为 Pydantic BaseModel，并将 `model_json_schema()` 输出与你手工编写的 schema 进行比较。识别手工编写版本省略的 Pydantic 默认设置的一个字段。

4. 测量拒绝率。构建十个不应可提取的输入（歌词、数学证明、空白电子邮件），并在真实提供商上以严格模式运行它们。计数拒绝 vs 虚构输出。这是你对拒绝感知重试的基准。

5. 从头到尾阅读 OpenAI 的结构化输出指南。识别它在严格模式下显式禁止但普通 JSON Schema 允许的一个构造。然后设计一个不完全需要该禁止构造的 schema，并重构它使其兼容严格模式。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| JSON Schema 2020-12 | "schema 规范" | 每个现代提供商都使用的 IETF 草案 schema 方言 |
| 严格模式 | "保证的 schema" | OpenAI 通过受限解码强制 schema 的标志 |
| 受限解码 | "Logit 掩码" | 解码时强制掩码无效的 next-token |
| 拒绝 | "模型拒绝" | 当输入无法适合 schema 时的类型化结果 |
| 解析错误 | "无效 JSON" | 输出未被解析为 JSON；在严格下不可能 |
| Schema 违规 | "错误的形状" | 已解析但违反类型/required/enum/范围 |
| `additionalProperties: false` | "不允许额外字段" | 禁止未知字段；OpenAI 严格要求 |
| Pydantic BaseModel | "类型化输出" | 发出和验证 JSON Schema 的 Python 类 |
| Zod schema | "TypeScript 输出类型" | 用于提供商输出验证的 TS 运行时 schema |
| 文法强制 | "开源受限解码" | 基于 FSM 的 logit 掩码，如 outlines / guidance |

## 拓展阅读

- [OpenAI — Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) —— 严格模式、拒绝和 schema 要求
- [OpenAI — Introducing structured outputs](https://openai.com/index/introducing-structured-outputs-in-the-api/) —— 2024 年 8 月发布文章，解释解码保证
- [Pydantic AI — Output](https://ai.pydantic.dev/output/) —— 序列化到每个提供商的类型化 output_type 绑定
- [JSON Schema — 2020-12 release notes](https://json-schema.org/draft/2020-12/release-notes) —— 权威规范
- [Microsoft — Structured outputs in Azure OpenAI](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/structured-outputs) —— 企业部署说明和严格模式注意事项
