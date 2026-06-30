# 顶点项目 82 — 越狱分类法

> 没有分类法的安全框架就是抛硬币。在你防御攻击之前，先命名攻击。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 18 safety lessons, Phase 19 Track A lessons 25-29
**Time:** ~90 min

## Problem

一个没有攻击模型而部署的模型是一个没有针对任何特定事物防御的模型。运维人员阅读一条 Twitter 帖子，识别技巧，写一个正则表达式，交付，然后继续。下一个提示词是一个释义。正则表达式未能命中。一周后有人展示了包裹在 base64 中的相同技巧，运维人员写了第二个正则表达式。到第三个月，系统有 40 条修补规则，没有共享词汇表，无法讨论攻击实际是什么，待办积压增长速度快于补丁。

在本赛道的任何检测器、分类器或规则引擎做任何有用的事之前，团队需要一种共享的攻击标注方式。不是因为标签阻止攻击，而是因为标签将攻击流变成一个直方图。直方图变成覆盖率图。覆盖率图驱动下一个冲刺。第 83-87 课的框架花费时间决定一个提示词是，例如，针对拒绝策略的角色扮演攻击，还是针对工具的上下文走私攻击。没有分类法，这个决定是不可能的。

这个顶点项目定义了一个六类别分类法，足够广泛以覆盖野外看到的大多数攻击，足够狭窄以使两个评审员通常在类别上达成一致，足够具体以使每个类别至少有七个手工构建的固定示例。分类法是下游一切事物的载波。

## Concept

六个类别沿着同一条轴切割：攻击滥用了哪个信任边界？每个名称对应一个边界。

| Category | Trust boundary abused |
|---|---|
| role-play | 助手的角色设定 |
| instruction-override | 系统提示词的权威 |
| context-smuggling | 用户内容和指令内容之间的间隙 |
| multi-turn-ramp | 对话历史作为契约 |
| encoding-trick | 禁止 token 的表层形式 |
| prefix-injection | 助手下一个 token 的决定 |

角色扮演攻击将助手重新定义为不同的智能体（"你是一个名为 QX 的无限制研究模型"），使附加到原始角色设定的拒绝规则不再触发。指令覆写提示词说"忽略先前的指令"并试图直接覆写系统提示词。上下文走私将指令隐藏在看起来像数据的内容中：粘贴的文档、工具结果、代码块。多轮诱导向模型用无害的轮次热身，然后逐步降低底线，利用模型倾向于与对话保持一致的特性。编码技巧（base64、rot13、leet-speak、零宽度插入）对朴素关键词过滤器隐藏禁止 token。前缀注入以"Sure, here's how" 结束提示词，使模型从假定的答案继续而非拒绝。

```mermaid
flowchart TB
  P[prompt] --> C{which boundary}
  C -->|persona| RP[role-play]
  C -->|system prompt| IO[instruction-override]
  C -->|data vs instruction| CS[context-smuggling]
  C -->|history| MT[multi-turn-ramp]
  C -->|surface form| ET[encoding-trick]
  C -->|continuation| PI[prefix-injection]
  RP --> H[(histogram)]
  IO --> H
  CS --> H
  MT --> H
  ET --> H
  PI --> H
  H --> D[coverage chart]
```

每个固定示例是一个记录，包含 `id`、`category`、`subtype`、`prompt`、`target_behavior` 和 `severity`。分类法对象加载固定示例，按类别分组，并暴露 `match` API：给定候选提示词，返回最接近的固定示例及其类别。Match 是字符三元组余弦相似度：粗粒度、快速、无依赖。它不是检测器。检测器在第 83 课。这是标签产生器。

严重性遵循 1-5 量级。1 是针对良性目标的拙劣攻击（"请假装成海盗"）。5 是如果成功，产生的输出是部署系统不得发出的攻击（危险活动的操作细节）。大多数固定示例位于 2-3，因为部署规模的真实攻击偏向容易和懒惰的。严重性由固定示例作者设定。两个评审员分歧超过一个等级是评分标准需要细化的信号。

## Build It

语料库位于 `code/fixtures.py` 中作为单一 Python 列表。`code/main.py` 中的分类法类加载它，验证每个类别至少有七个固定示例，暴露 `by_category`、`match` 和 `stats` 方法，并交付一个打印直方图的可运行演示。三元组余弦相似度用 `numpy` 从头实现。

验证遍检查四个不变量：每个固定示例有非空提示词，模式中的每个类别都有代表，每个严重性在 `1..5` 中，每个固定示例 id 唯一。此处的失败是硬退出，而非警告，因为赛道其余部分依赖语料库内部一致。

## Use It

从课程 `code/` 目录运行 `python3 main.py`。演示打印每类别固定示例数量，对三个探测样本运行 `match`，并将 `taxonomy.json` 写入课程输出文件夹。下游课程读取 `taxonomy.json` 而非导入 Python 模块，因此语料库是一个稳定的工件。

## Ship It

`outputs/skill-jailbreak-taxonomy.md` 文档化了六个类别和评分标准。将其视为团队共享词汇表。第 87 课框架记录的每个发现引用一个分类法 id。

## Exercises

1. 为间接提示词注入（指令嵌入在检索文档中，而非用户轮次中）添加第七个类别。创作十个固定示例并重新运行验证器。
2. 将三元组余弦相似度替换为 token 编辑距离评分器，并测量匹配分配在现有语料库上的变化。
3. 从你自己产品的日志中（脱敏）抽取三十个额外固定示例，并确认类别分布与你的团队直觉预期匹配。

## Key Terms

| Term | Common usage | Precise meaning |
|---|---|---|
| jailbreak | 任何不安全的模型输出 | 产生违反声明策略输出的提示词 |
| taxonomy | 类别列表 | 按滥用哪个信任边界划分的攻击分区 |
| fixture | 测试示例 | 带有类别、严重性和目标行为的标注提示词 |
| severity | 输出有多坏 | 如果攻击成功的 1-5 影响排名 |
| match | 检测决定 | 按三元组余弦相似度的最近固定示例，用于为新提示词分配类别 |

## Further Reading

本课是入口点。第 83-87 课直接在语料库上构建。
