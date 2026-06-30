# FIPA-ACL与言语行为的遗产

> 在MCP之前，在A2A之前，存在FIPA-ACL。2000年，IEEE智能物理智能体基金会批准了一种具有二十种施事行为、两种内容语言和一组交互协议的智能体通信语言——合同网、订阅/通知、条件请求。它因本体论开销对Web而言过重而从工业界淡出，但LLM对多智能体系统的复兴正在悄悄地重新实现相同的概念，只是没有形式化语义：JSON契约替代了施事行为，自然语言替代了本体论。本课认真阅读FIPA-ACL，以便你能看到哪些2026年的协议决策是重新发明，哪些是真正的创新，以及当前浪潮将在哪里重新发现2000年代已经解决的问题。

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 01 (Why Multi-Agent)
**Time:** ~60 minutes

## 问题

2026年的智能体协议格局十分繁忙：用于工具的MCP、用于智能体的A2A、用于企业审计的ACP、用于去中心化信任的ANP、用于自然语言内容的NLIP，加上CA-MCP和二十多个研究提案。每个规范都宣称自己是基础性的。

诚实的解读是，它们中的大多数正在重新发现一棵非常具体的、二十年前的决策树。Austin（1962年）和Searle（1969年）的言语行为理论给了我们"话语是行动"。KQML（1993年）将其转化为一种在线协议。FIPA-ACL（2000年批准）产生了参考标准化：二十种施事行为、内容语言SL0/SL1、用于合同网和订阅-通知的交互协议。JADE和JACK是Java参考平台。这项工作在2010年左右衰退，因为本体论开销过重且Web正在获胜。

当你看到MCP的 `tools/call`、A2A的任务生命周期或CA-MCP的共享上下文存储时，你实际上看到的是对FIPA决策的、更柔和的、JSON原生的重新演绎。了解遗产能告诉你两件事：哪些新的"创新"实际上是重新发明，哪些旧的失败模式新规范将会重新发现。

## 概念

### 言语行为，一段话概括

Austin注意到有些句子并不描述世界——它们改变世界。"我承诺。""我请求。""我宣布。"他将这些称为施事话语。Searle形式化了五个类别：断言、指令、承诺、表达、宣告。KQML（Finin等人，1993年）为软件智能体使之可操作：一条消息是施事行为（行动）加上内容（行动涉及什么）。FIPA-ACL清理了KQML的缺口，并围绕二十种施事行为进行了标准化。

### 二十种FIPA施事行为（部分列表）

| 施事行为 | 意图 |
|---|---|
| `inform` | "我告诉你P为真" |
| `request` | "我要求你做X" |
| `query-if` | "P是否为真？" |
| `query-ref` | "X的值是什么？" |
| `propose` | "我提议我们做X" |
| `accept-proposal` | "我接受提议" |
| `reject-proposal` | "我拒绝提议" |
| `agree` | "我同意做X" |
| `refuse` | "我拒绝做X" |
| `confirm` | "我确认P为真" |
| `disconfirm` | "我否认P" |
| `not-understood` | "你的消息无法解析" |
| `cfp` | "征集关于X的提案" |
| `subscribe` | "当X变化时通知我" |
| `cancel` | "取消进行中的X" |
| `failure` | "我尝试了X但失败了" |

完整列表在 `fipa00037.pdf` 中（FIPA ACL消息结构）。重点不是记忆它——重点是其中每一个都对应于一个LLM协议最终会重新添加的原语。

### 规范的FIPA-ACL消息

```
(inform
  :sender       agent1@platform
  :receiver     agent2@platform
  :content      "((price IBM 83))"
  :language     SL0
  :ontology     finance
  :protocol     fipa-request
  :conversation-id   conv-42
  :reply-with   msg-17
)
```

七个字段承载协议信封；一个字段（`content`）承载负载。其余字段恰好是你每次向JSON协议添加重试、线程和本体论时重新发明的东西。

### 两个遗留平台

**JADE**（Java智能体开发框架，1999-2020年代）是最常用的FIPA兼容运行时。智能体继承一个基类、交换ACL消息、运行在容器内，并使用"行为"进行协调。交互协议库随合同网、订阅-通知、条件请求和提议-接受一起发布。

**JACK**（Agent Oriented Software，商业）强调在FIPA消息之上的BDI（信念-愿望-意图）推理。更形式化，采用率较低。

两者都在Web技术栈吞并多智能体用例后衰退。MCP和A2A是2026年的运行时"容器"。

### 为什么FIPA衰退了

- **本体论开销。** FIPA需要共享本体论来解析 `content`。就本体论达成一致是一个多年标准进程。Web只是用HTTP + JSON。
- **无人使用的形式化语义。** SL（语义语言）给出了严格的真值条件，但大多数生产系统使用自由形式内容，忽略了形式主义。
- **工具锁定。** JADE仅限Java；JACK是商业的。多语言团队绕过了两者。
- **互联网赢得了技术栈。** REST，然后是JSON-RPC，然后是gRPC，取代了ACL的传输。

### LLM复兴是FIPA-lite

比较FIPA `request` 和MCP `tools/call`：

```
(request                                {
  :sender  agent1                         "jsonrpc": "2.0",
  :receiver tool-server                   "method":  "tools/call",
  :content "(lookup stock IBM)"           "params":  {"name":"lookup_stock",
  :ontology finance                                   "arguments":{"symbol":"IBM"}},
  :conversation-id c42                    "id": 42
)                                        }
```

相同的信封，不同的语法。两者都携带：谁、向谁、意图、负载、关联ID。两者互相之间都不是革命——它们是在同一设计上的不同权衡。

Liu等人2025年的调查（"A Survey of Agent Interoperability Protocols: MCP, ACP, A2A, ANP"，arXiv:2505.02279）明确地阐述了这一谱系：MCP对应工具使用的言语行为，A2A对应智能体对等的言语行为，ACP对应审计追踪的言语行为，ANP对应去中心化身份扩展。新规范是带有JSON语法和更松散语义的ACL后裔。

### 权衡，坦率陈述

**FIPA给你而现代规范放弃的：**

- 形式化语义——你可以证明 `inform` 蕴含发送者相信内容。
- 施事行为的规范目录——你不必重新争论"我们应该有 `cancel` 吗？"。
- 数十年的交互协议模式——合同网、订阅-通知、提议-接受——具有已知的正确性属性。

**现代规范给你而FIPA没有的：**

- 与每个现代工具兼容的JSON原生负载。
- LLM可以在没有手工编码本体论的情况下解释的自然语言内容。
- Web技术栈传输（HTTP、SSE、WebSocket）。
- 通过自描述文档的能力发现（MCP `listTools`、A2A Agent Card）。

更松散的意图语义以换取更简单的实现。这就是确切的交换。

### 值得移植的交互协议

FIPA发布了约15种交互协议。其中三种值得带入LLM多智能体系统：

1. **合同网协议（CNP）。** 管理者发出 `cfp`（征集提案）；投标者以 `propose` 回应；管理者接受/拒绝。这是规范的任务市场模式（第16阶段·第16课 协商）。
2. **订阅/通知。** 订阅者发送 `subscribe`；发布者每当主题变化时发送 `inform`。这是2026年的每个事件总线。
3. **条件请求。** "当条件Y成立时做X。"带有前置条件的延迟行动。2026年的类似物是持久工作流引擎中的延迟任务（第16阶段·第22课 生产扩展）。

每个都清晰地映射到现代消息队列、HTTP + 轮询或SSE流。

### 当你放弃本体论时什么会出问题

没有共享本体论，智能体从自然语言内容中推断含义。有记录的2026年失败模式是**语义漂移**：两个智能体对微妙不同的概念使用同一个词（`"customer"`），接收智能体根据错误解释采取行动，没有模式验证器捕获它。FIPA的本体论要求会在解析时就拒绝该消息。

不走完全本体论道路的缓解措施：

- `content` 上的JSON Schema——在线上拒绝结构性错误。
- 类型化产物（A2A）——拒绝错误模态。
- 信封中的显式施事行为——使意图即使在内容是自然语言时也不模糊。

### 2026年规范，映射到言语行为遗产

| 现代规范 | FIPA类似物 | 保留了 | 放弃了 |
|---|---|---|---|
| MCP `tools/call` | `request` | 显式意图、关联ID | 形式化语义、本体论 |
| MCP `resources/read` | `query-ref` | 显式意图、关联ID | 形式化语义 |
| A2A任务生命周期 | contract-net + request-when | 异步生命周期、状态转换 | 形式化完整性保证 |
| A2A流事件 | subscribe/notify | 异步推送 | 类型化谓词订阅 |
| CA-MCP共享上下文 | blackboard (Hayes-Roth 1985) | 多写者共享内存 | 逻辑一致性模型 |
| NLIP | natural-language content | LLM原生 | 模式 |

从上到下阅读表格，模式是：保留结构原语，放弃形式主义，让LLM弥补模糊性。

## 构建它

`code/main.py` 实现一个纯标准库的FIPA-ACL翻译器。它编码和解码规范ACL信封，并展示每条MCP / A2A消息形状如何归结为相同的七个字段。演示：

- 将五条MCP风格和A2A风格的消息编码为FIPA-ACL。
- 将FIPA-ACL解码回现代等价物。
- 使用 `cfp`、`propose`、`accept-proposal`、`reject-proposal` 在一个管理者和三个投标者之间运行玩具合同网协商。

运行：

```
python3 code/main.py
```

输出是并排跟踪，展示每条现代消息在其2026年JSON形式和其FIPA-ACL形式下的样子，然后是一次合同网投标的往返。相同的协议原语在往返后仍然存活；只有语法不同。

## 运用

`outputs/skill-fipa-mapper.md` 是一个技能，读取任何智能体协议规范并生成FIPA-ACL映射。在采用新协议之前使用它来回答："这真的是新的吗，还是只是带JSON语法的 `inform`？"

## 交付物

不要把FIPA-ACL带回来。带回它的检查清单：

- 每条消息的意图原语（施事行为）是什么？
- 是否有请求-响应和取消的关联ID？
- 是否有显式内容语言（JSON-RPC、纯文本、结构化类型化产物）？
- 交互协议是否一等公民，还是你正在从零开始重新实现合同网？
- 当两个智能体对内容含义（语义漂移）有分歧时会发生什么？

在将任何新协议交付到生产环境之前，为其记录这五个问题。

## 练习

1. 运行 `code/main.py`。观察往返编码。识别哪个FIPA施事行为对应于 `tools/call`、`resources/read` 和A2A任务创建。
2. 用 `cancel` 施事行为扩展合同网演示，让管理者在投标过程中撤回任务。`cancel` 解决了重试独自不能解决什么失败案例？
3. 阅读FIPA ACL消息结构（http://www.fipa.org/specs/fipa00037/）第4.1-4.3节。选择一个本课未涵盖的施事行为，描述其现代JSON-RPC类似物。
4. 阅读Liu等人，arXiv:2505.02279。针对MCP、A2A、ACP、ANP，列出它们保留和放弃的FIPA施事行为族。
5. 为你自己系统中的 `request` 施事行为的 `content` 字段设计一个极简JSON Schema。该模式给了你什么纯自然语言给不了的，以及它有什么成本？

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|----------------|------------------------|
| 言语行为 | "做某事的话语" | Austin/Searle：话语即行动。ACL的理论父体。 |
| FIPA | "那个旧的XML东西" | IEEE智能物理智能体基金会。2000年标准化了ACL。 |
| ACL | "智能体通信语言" | FIPA的信封格式：施事行为 + 内容 + 元数据。 |
| 施事行为 | "动词" | 一条消息的意图类别：`inform`、`request`、`propose`、`cfp` 等。 |
| KQML | "FIPA的前身" | 知识查询和操作语言（1993年）。更简单、范围更窄。 |
| 本体论 | "共享词汇" | 内容语言所谈论概念的正式定义。 |
| SL0 / SL1 | "FIPA内容语言" | 语义语言级别0和1——形式化内容语言族。 |
| 合同网 | "任务市场" | 管理者发出cfp；投标者提议；管理者接受。规范交互协议。 |
| 交互协议 | "消息模式" | 具有已知正确性的施事行为序列：条件请求、订阅-通知等。 |

## 进一步阅读

- [Liu et al. — A Survey of Agent Interoperability Protocols: MCP, ACP, A2A, ANP](https://arxiv.org/html/2505.02279v1) — 将现代规范与FIPA遗产连接的权威2025年调查
- [FIPA ACL Message Structure Specification (fipa00037)](http://www.fipa.org/specs/fipa00037/) — 2000年批准的信封格式
- [FIPA Communicative Act Library Specification (fipa00037)](http://www.fipa.org/specs/fipa00037/) — 完整的施事行为目录
- [MCP specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — `request`/`query-ref` 的现代工具使用等价物
- [A2A specification](https://a2a-protocol.org/latest/specification/) — contract-net和subscribe-notify的现代智能体对等等价物
