# 综合项目 86 — 宪章规则引擎

> 一条规则由一个名称、一个断言和一个解释组成。三者缺一不可，否则只是感觉，不是规则。

**类型：** 构建
**语言：** Python, YAML
**前置课程：** 第 18 阶段安全课程，第 19 阶段 A 系列课程 25-29
**时间：** 约 90 分钟

## 问题

分类器负责可识别的故障。规则引擎负责契约性的约束。一个正在编写编码助手的团队想要这样的约束："每个包含代码的回复，必须以可执行代码块或明确声明的前提假设结尾。"一个正在运行客服机器人的团队想要："每个拒绝回复都必须提供一个后续步骤。"这些约束不是分类器的自然目标。它们是对回复、对话和系统策略的断言，并且需要让非工程师也能读懂。

诚实的表示形式是一个声明式文件。一份宪章以 YAML 格式存放在代码旁边，纳入版本控制，并拥有独立的审查流程。每条规则都有一个 `name`（名称）、一个 `predicate`（断言）、一个 `severity`（严重级别）和一个 `explanation`（解释）模板。引擎加载该文件，针对候选输出评估每条规则，并为每条触发的规则返回一个结构化的 `Violation`（违规记录）。本综合项目中的规则引擎通过 `all_of`、`any_of` 和 `not_` 来组合断言，使单条规则就能表达"如果回复包含代码，则必须以可执行代码块结尾且不得引用仅供内部使用的库"。

本课的另一半是修订。一个只会拦截的规则引擎是不完整的。一个能提出修复建议的规则引擎才具有实际操作性：助手起草回复，引擎标记违规，修复器生成修订版回复，引擎确认修订版满足规则要求。本课交付一个最小化修复器（基于每条规则的正则替换）以及草稿版与修订版之间的结构化差异（diff）（逐行的新增、删除和编辑）。

## 概念

```mermaid
flowchart LR
  D[草稿回复] --> RE[规则引擎]
  RE -->|违规| F[修复器]
  F --> R[修订版回复]
  R --> RE2[规则引擎二次检查]
  RE2 -->|判定| OUT[接受或升级]
  D -.->|差异| R
```

一条规则的形态如下：

```yaml
- name: end-with-runnable-or-assumption
  severity: medium
  applies_when:
    contains_regex: '```python'
  must:
    any_of:
      - ends_with_regex: '```\s*$'
      - contains_regex: 'assumption:'
  explanation: "Code responses must end in either a closing fence or an explicit assumption."
  fix:
    append_if_missing: "\n\nAssumption: example inputs are valid."
```

原子断言包括：`contains_regex`、`not_contains_regex`、`ends_with_regex`、`starts_with_regex`、`max_words`、`min_words`。组合断言包括 `all_of`、`any_of`、`not_`。引擎首先评估 `applies_when`（适用条件）；如果规则不适用，违规记录将标记为 `not_applicable`（不适用）。否则，引擎评估 `must`（必须满足的条件），并得出 `pass`（通过）或 `violation`（违规）的结果。

严重级别包括 `low`（低）、`medium`（中）、`high`（高），与课程 85 保持一致。下游安全门（课程 87）将 `high` 级别的规则违规等同于 `high` 级别的分类器判定：拦截。

修复器是一组声明式操作：`append_if_missing`、`prepend_if_missing`、`replace_regex`。每个操作按名称将一条规则映射到一个转换。修复器有意限定于局部编辑；结构性重写属于单独的拒绝与帮助层，不在本课涵盖范围内。

差异是针对原始版本与修订版本计算得出的。它是一个 `Change`（变更）记录列表，每条记录包含 `op`（add 新增、remove 删除、edit 编辑）以及相关文本。下游安全门可以记录差异，以便人类评审员随时审查修复器的行为。

## 构建它

`code/rules.yml` 存储宪章。`code/main.py` 中的加载器接受 YAML 文件（当 PyYAML 可用时）或 JSON 文件（内置支持）。本课附带一个 `rules.yml` 文件，课程测试会通过两种代码路径解析它。`code/main.py` 定义了 `Engine`（引擎）类和 `Fixer`（修复器）类以及一个 `diff` 函数。组合断言通过递归求值，`any_of` 采用短路机制。

随附的宪章规则：

- `no-empty-refusal`（中）- 拒绝回复必须包含建议或重定向
- `end-with-runnable-or-assumption`（中）- 含代码的回复必须干净收尾
- `no-pii-in-examples`（高）- 示例数据不得包含邮箱地址或电话号码格式
- `cite-when-asserting-fact`（低）- 以"据……"开头的行必须包含括号引用
- `no-internal-library-leak`（高）- 输出中不得出现 `internal-only` 和 `policybot-internal` 字样
- `bounded-length`（低）- 回复不得超过 800 词

## 使用它

`python3 main.py`。演示程序将三份草稿回复送入引擎，打印违规信息，运行修复器，打印差异，并将结果写入 `outputs/rules_report.json`。其中一个测试用例包含一条不适用的规则（草稿中没有代码块），报告对该规则显示 `not_applicable`，以便团队看到引擎确实对其进行了显式评估。

## 交付它

`outputs/skill-constitutional-rules-engine.md` 记录了规则语法和修复器操作。

## 练习

1. 添加一条规则：当用户提示提及安全时，要求每个回复都包含"如果情况紧急"这样的措辞。使用组合断言。
2. 将基于正则的修复器替换为接受命名槽位的模板修复器。演示使用新设计重写一条规则。
3. 添加一个指标端点：给定一组草稿语料，返回每条规则的违规率，以便团队识别哪条规则触发过于频繁。

## 关键术语

| 术语 | 常见用法 | 精确定义 |
|---|---|---|
| 宪章（constitution） | 一份模糊的策略文档 | 一份包含规则（含断言、严重级别和解释）的 YAML 文件 |
| 断言（predicate） | 一项检查 | 一个从文本到布尔值的可调用对象，可以是原子形式，或通过 all_of/any_of/not_ 组合构成 |
| 违规（violation） | 一次失败 | 一条包含规则名称、严重级别、解释和匹配区间的结构化记录 |
| 修复器（fixer） | 模型微调 | 一个确定性的按规则转换，将草稿映射为修订版 |
| 差异（diff） | 字符串比较 | 一份结构化的新增、删除和编辑操作列表，对比草稿与修订版 |

## 扩展阅读

课程 87 将本引擎与输入端检测器和输出端分类器组合成一个统一的安全门。
