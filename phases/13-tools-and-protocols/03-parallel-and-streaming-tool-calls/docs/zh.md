# 并行工具调用和带工具的流式传输

> 串行化三个独立的天气查询意味着三次往返。并行运行它们，总时间坍缩到最慢的单次调用。现在每个前沿提供商都在一个回合内发出多个工具调用。收益是实在的；管道是微妙的。本课讲解两半部分：并行扇出和流式参数重组，重点在 id 关联陷阱。

**Type:** Build
**Languages:** Python (stdlib, thread pool + streaming harness)
**Prerequisites:** Phase 13 · 02 (function calling deep dive)
**Time:** ~75 minutes

## 学习目标

- 解释为什么 `parallel_tool_calls: true` 存在以及何时禁用它。
- 在并行扇出期间将流式参数块关联到正确的工具调用 id。
- 将部分 `arguments` 字符串重组为完整 JSON，而不提前解析。
- 运行一个三城天气基准，展示顺序 vs 并行延迟。

## 问题

没有并行调用，回答"班加罗尔、东京和苏黎世的天气如何"的代理会这样做：

```
user -> LLM
LLM -> call get_weather(Bengaluru)
host -> 运行执行器，带回结果回复
LLM -> call get_weather(Tokyo)
host -> 运行执行器，带回结果回复
LLM -> call get_weather(Zurich)
host -> 运行执行器，带回结果回复
LLM -> 最终文本答案
```

三次 LLM 往返，每次还要支付执行器延迟。大约 4 倍于理想的墙上时钟时间。

有了并行调用：

```
user -> LLM
LLM -> call get_weather(Bengaluru); call get_weather(Tokyo); call get_weather(Zurich)
host -> 并发运行全部三个执行器，带回三个结果回复
LLM -> 最终文本答案
```

一次 LLM 往返。执行器时间是三者的最大值，而非总和。在 OpenAI、Anthropic 和 Gemini 上的生产基准测试显示，在扇出工作负载上减少了 60% 到 70% 的墙上时钟时间。

代价是关联复杂度。当三个调用乱序完成时，你的结果必须携带匹配的 `tool_call_id` 以便模型能对齐它们。当结果流式传输时，你必须将部分参数片段组装为完整 JSON 再执行。Gemini 3 添加唯一 id 部分是为了解决现实世界中两个并行调用到同一工具无法区分的问题。

## 概念

### 启用并行

- **OpenAI。** `parallel_tool_calls: true` 默认开启。设置为 `false` 强制串行。
- **Anthropic。** 通过 `disable_parallel_tool_use: false` 并行（Claude 3.5 及以上默认）。设置为 `true` 强制串行。
- **Gemini。** 始终支持并行；`tool_config.function_calling_config.mode = "AUTO"` 让模型决定。

当工具有顺序依赖（`create_file` 然后 `write_file`），或一个调用的输出通知另一个的输入，或速率限制器无法处理扇出时，禁用并行。

### Id 关联

模型发出的每个调用都有一个 `id`。宿主返回的每个结果必须包含相同的 id。没有它，结果是模糊的。

- **OpenAI。** 每个 tool-role 消息上的 `tool_call_id`。
- **Anthropic。** 每个 `tool_result` 块上的 `tool_use_id`。
- **Gemini。** 每个 `functionResponse` 上的 `id`（Gemini 3 及以上；Gemini 2 按名称匹配，这对相同名称的并行调用是破坏性的）。

### 并发运行调用

宿主在各自的线程、协程或远程工作器上运行每个调用的执行器。最简单的框架使用线程池；生产环境使用 `asyncio.gather` 或结构化并发。完成顺序不可预测——id 是标识符。

一个常见 bug：按调用列表顺序回复结果，而不是按完成顺序。这通常可行，因为模型只关心 `tool_call_id`，但如果结果是丢弃或重复的，乱序提交使调试更困难。优先使用带显式 id 的完成顺序回复。

### 流式工具调用

当模型流式传输时，`arguments` 以片段形式到达。三个并行调用的三个独立流式块在线路上交错。你需要每个 id 一个累加器。

按提供商区分形状：

- **OpenAI。** 每个块是 `choices[0].delta.tool_calls[i].function.arguments`（部分字符串）。块携带 `index`（在调用列表中的位置）。你按 index 累积，在它首次出现时读取 `id`，并在 `finish_reason = "tool_calls"` 时解析 JSON。
- **Anthropic。** 流事件是 `message_start`，然后每个块一个 `content_block_start`，类型为 `tool_use`（包含 id、name、空 input）。`content_block_delta` 事件携带 `input_json_delta` 块。`content_block_stop` 关闭每个块。
- **Gemini。** `streamFunctionCallArguments`（Gemini 3 及以上）携带一个 `functionCallId` 的块，因此调用可以干净地交错。Gemini 3 之前，流式传输一次返回一个完整调用。

### 部分 JSON 和提前解析陷阱

在 `arguments` 完整之前不能解析它。部分 JSON 如 `{"city": "Beng` 是无效的并会抛出异常。正确的门控是提供商的调用结束信号：OpenAI 的 `finish_reason = "tool_calls"`，Anthropic 的 `content_block_stop`，或 Gemini 的流结束事件。只有然后才尝试 `json.loads`。更健壮的方法是使用增量 JSON 解析器，在结构完成后产生事件；OpenAI 的流式指南对此推荐用于显示实时"思考"指示器的 UX。花括号计数作为完整性测试不可靠（引号字符串或转义内容内的花括号导致误报），只应用作非正式的调试启发式。

### 乱序完成

```
call_A: 快速 API，最先返回
call_B: 慢速 API，第二个返回
call_C: 中速 API，第三个返回
```

宿主回复仍必须引用 id：

```
[{role: "tool", tool_call_id: "call_A", content: ...},
 {role: "tool", tool_call_id: "call_B", content: ...},
 {role: "tool", tool_call_id: "call_C", content: ...}]
```

回复中的顺序对 OpenAI 或 Anthropic 的正确性无关紧要。只要 id 匹配，Gemini 也接受任何顺序。

### 基准：顺序 vs 并行

`code/main.py` 中的框架模拟三个执行器，延迟分别为 400、600 和 800 ms。顺序运行的总时间为 1800 ms。并行运行为 max(400, 600, 800) = 800 ms。差异是常数而非比例的，因此节省量随着工具数量增长而增长。

现实警示：并行调用对下游 API 施加压力。一个 10 路扇出到一个速率受限的服务将会失败。Phase 13 · 17 涵盖网关级背压；重试语义计划在未来的阶段中。

### 流式扇出墙上时钟

如果模型本身流式传输，你可以在一个调用的参数完成后立即开始执行，而非等待所有调用完成。这是 OpenAI 文档中提到但并非所有 SDK 都暴露的优化。本课的框架做到了：一旦模拟流产生一个完整的参数对象，宿主就启动该调用。

## 使用

`code/main.py` 有两部分。第一部分使用 `concurrent.futures.ThreadPoolExecutor` 顺序和并行运行三个模拟天气调用，并打印墙上时钟时间。第二部分重放一个假的流式响应——三个并行调用的 `arguments` 块在一个流上交错——并使用 `StreamAccumulator` 按 id 重组。没有 LLM，没有网络，只有重组逻辑。

需要注意的地方：

- 顺序计时器达到 1.8 秒。并行计时器在相同的假延迟下达到 0.8 秒。
- 累加器通过按 id 缓冲处理乱序到达的块，并仅在每个调用的 JSON 完成时解析。
- 一旦 id 的参数完成，执行器立即启动，而非在所有流结束后。

## 产出

本课产出 `outputs/skill-parallel-call-safety-check.md`。给定一个工具注册表，该技能审核哪些工具可以安全地并行化，哪些有顺序依赖，以及哪些会使下游速率限制不堪重负——返回一个带有每个工具 `parallel_safe` 标志的修订注册表。

## 练习

1. 运行 `code/main.py` 并变化模拟延迟。确认并行与顺序的比例大约是 `max/sum`（实际运行因线程调度、序列化和框架开销与理想值略有偏差）。在什么延迟分布下并行不再重要？

2. 扩展累加器以处理"调用在流中取消"的情况，通过丢弃其缓冲区并发出 `cancelled` 事件。哪个提供商明确记录了这种情况？检查 Anthropic 的 `content_block_stop` 语义和 OpenAI 的 `finish_reason: "length"` 行为。

3. 将线程池替换为 `asyncio.gather`。对两者进行基准测试。由于上下文切换成本较低，你应在 async 上看到小幅提升，但仅在执行器进行真正的 I/O 时才会出现。

4. 选择两个不应并行化的工具（例如 `create_file` 然后 `write_file`）。向注册表添加 `ordering_dependency` 图，并根据该图门控并行扇出。这是依赖感知调度的最小机制，将在未来的代理工程阶段中形式化。

5. 阅读 OpenAI 的并行 function-calling 部分和 Anthropic 的 `disable_parallel_tool_use` 文档。确定 Anthropic 推荐禁用并行的一种实际工具类型。（提示：对同一资源产生后果性变更。）

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 并行工具调用 | "一个回合中扇出" | 模型在一个助手消息中发出多个工具调用 |
| `parallel_tool_calls` | "OpenAI 的标志" | 启用或禁用多次调用发出 |
| `disable_parallel_tool_use` | "Anthropic 的逆标志" | 选择退出的标志；默认启用并行 |
| 工具调用 id | "关联句柄" | 每个调用的标识符，结果消息必须回显 |
| 累加器 | "流缓冲区" | 用于部分 `arguments` 块的按 id 字符串缓冲 |
| 乱序完成 | "最快的先完成" | 并行调用以不可预测的顺序完成；id 是纽带 |
| 依赖图 | "顺序约束" | 其输出馈入其他工具输入的工具；不能并行化 |
| 提前解析陷阱 | "JSON.parse 爆炸" | 尝试解析不完整的 `arguments` 字符串 |
| `streamFunctionCallArguments` | "Gemini 3 特性" | 流式参数块，每个调用带唯一 id |
| 完成顺序回复 | "不等所有完成" | 按到达顺序回复结果，按 id 键控 |

## 拓展阅读

- [OpenAI — Parallel function calling](https://platform.openai.com/docs/guides/function-calling#parallel-function-calling) —— 默认行为及选择退出标志
- [Anthropic — Tool use: implementing tool use](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implementing-tool-use) —— `disable_parallel_tool_use` 和结果批量处理
- [Google — Gemini function calling parallel section](https://ai.google.dev/gemini-api/docs/function-calling) —— 自 Gemini 3 起的 id 关联并行调用
- [OpenAI — Streaming responses with tools](https://platform.openai.com/docs/api-reference/responses-streaming) —— OpenAI 流的块化参数重组
- [Anthropic — Streaming messages](https://docs.anthropic.com/en/api/messages-streaming) —— 带 `input_json_delta` 的 `content_block_delta`
