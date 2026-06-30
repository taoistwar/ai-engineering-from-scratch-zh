# 共指消解

> "She called him. He did not answer. The doctor was at lunch."三个指称指向两个人，没有人被点名。共指消解弄清楚了谁是谁。

**类型：** 学习
**语言：** Python
**先修要求：** 第五阶段 · 06（NER），第五阶段 · 07（POS 与句法分析）
**预计时间：** 约60分钟

## 问题

从一篇 300 词的报道中提取每次提及 Apple Inc。当报道说"Apple"时容易。当它说"the company"、"they"、"Cupertino's technology giant"或"Jobs's firm"时困难。如果不将这些提及消解到同一实体，你的 NER 流程会漏掉 60-80% 的提及。

共指消解将所有引用同一真实世界实体的表达链接到一个簇中。它是表层 NLP（NER、句法分析）和下游语义（IE、QA、摘要、KG）之间的粘合剂。

为什么它在 2026 年很重要：

- 摘要："The CEO announced..." vs "Tim Cook announced..."——摘要应该点名 CEO。
- 问答："Who did she call?"需要消解"she。"
- 信息提取：一个知识图谱具有"PER1 founded Apple"和"Jobs founded Apple"作为单独条目是错误的。
- 多文档 IE：跨报道合并且提及同一事件是跨文档共指。

## 概念

![Coreference clustering: mentions → entities](../assets/coref.svg)

**任务。** 输入：一个文档。输出：提及（跨度）的聚类，其中每个簇引用一个实体。

**提及类型。**

- **命名实体。** "Tim Cook"
- **名词性。** "the CEO"、"the company"
- **代词性。** "he"、"she"、"they"、"it"
- **同位语。** "Tim Cook, Apple's CEO,"

**架构。**

1. **基于规则的（Hobbs, 1978）。** 使用语法规则基于句法树进行代词消解。良好的基线。在代词上惊人地难以被击败。
2. **提及对分类器。** 对于每对提及（m_i, m_j），预测它们是否共指。通过传递闭包聚类。2016 年前的标准。
3. **提及排序。** 对于每个提及，对候选先行词（包括"无先行词"）进行排序。选择 top。
4. **基于跨度的端到端（Lee et al., 2017）。** Transformer 编码器。枚举所有候选跨度，直到长度上限。预测提及分数。预测每个跨度的先行词概率。贪心聚类。现代默认选择。
5. **生成式（2024+）。** 提示一个 LLM："列出此文本中的每个代词及其先行词。"在简单案例上效果良好，在长文档和罕见指称上困难。

**评估指标。** 五个标准指标（MUC、B³、CEAF、BLANC、LEA），因为没有单一指标能捕获聚类质量。报告前三个的平均值作为 CoNLL F1。2026 年 CoNLL-2012 上的最先进水平：约 83 F1。

**已知的困难案例。**

- 确定描述指涉到数页前引入的实体。
- 桥接回指（"the wheels" → 先前提到的汽车）。
- 中文和日语等语言中的零回指。
- 预指（代词在指称对象之前）："When **she** walked in, Mary smiled."

## 构建它

### 步骤 1：预训练神经共指消解（AllenNLP / spaCy-experimental）

```python
import spacy
nlp = spacy.load("en_coreference_web_trf")   # 实验性模型
doc = nlp("Apple announced new products. The company said they would ship soon.")
for cluster in doc._.coref_clusters:
    print(cluster, "->", [m.text for m in cluster])
```

在更长的文档上，你会得到类似：
- 簇 1：[Apple, The company, they]
- 簇 2：[new products]

### 步骤 2：基于规则的代词消解器（教学）

参见`code/main.py`获取仅标准库的实现：

1. 提取提及：命名实体（大写跨度）、代词（字典查找）、确定描述（"the X"）。
2. 对于每个代词，查看前 K 个提及并按以下评分：
   - 性别/数一致（启发式）
   - 最近性（更近胜出）
   - 句法角色（主语优先）
3. 链接得分最高的先行词。

不具有与神经模型的竞争力。但它展示了搜索空间以及端到端模型必须做出的决策。

### 步骤 3：使用 LLM 进行共指消解

```python
prompt = f"""Text: {text}

List every pronoun and noun phrase that refers to a person or company.
Cluster them by what they refer to. Output JSON:
[{{"entity": "Apple", "mentions": ["Apple", "the company", "it"]}}, ...]
"""
```

两个需要关注的失败模式。首先，LLM 过度合并（"him"和"her"指涉两个不同的人）。其次，LLM 在长文档中静默丢弃提及。始终用跨度偏移检查进行验证。

### 步骤 4：评估

标准的 conll-2012 脚本计算 MUC、B³、CEAF-φ4 并报告平均值。对于内部评估，从你的标注测试集上的跨度级精确率和召回率开始，然后添加提及链接 F1。

## 陷阱

- **单例爆炸。** 某些系统将每个提及报告为它自己的簇。B³ 是宽容的。MUC 惩罚这一点。始终检查所有三个指标。
- **长上下文中的代词。** 性能在超过 2,000 token 的文档上下降约 15 F1。仔细分块。
- **性别假设。** 硬编码的性别规则在非二元指称对象、组织、动物上失败。使用学习的模型或中性评分。
- **LLM 在长文档上的漂移。** 单次 API 调用无法可靠地跨 50+ 个段落对提及进行聚类。使用滑动窗口 + 合并。

## 使用它

2026 年技术栈：

| 场景 | 选择 |
|-----------|------|
| 英语、单文档 | `en_coreference_web_trf`（spaCy-experimental）或 AllenNLP neural coref |
| 多语言 | 在 OntoNotes 或 Multilingual CoNLL 上训练的 SpanBERT / XLM-R |
| 跨文档事件共指 | 专门的端到端模型（2025–26 SOTA） |
| 快速 LLM 基线 | GPT-4o / Claude 带结构化输出共指提示 |
| 生产对话系统 | 基于规则回退 + 神经主模型 + 关键槽位的人工审核 |

2026 年发布的集成模式：首先运行 NER，运行共指消解，将共指簇合并到 NER 实体中。下游任务看到每个簇一个实体，而不是每个提及一个实体。

## 交付它

保存为 `outputs/skill-coref-picker.md`：

```markdown
---
name: coref-picker
description: 选择共指消解方法、评估计划和集成策略。
version: 1.0.0
phase: 5
lesson: 24
tags: [nlp, coref, information-extraction]
---

给定一个用例（单文档 / 多文档、领域、语言），输出：

1. 方法。基于规则 / 神经基于跨度 / LLM 提示 / 混合。一句理由。
2. 模型。如果是神经的，命名检查点。
3. 集成。操作顺序：tokenize → NER → 共指 → 下游任务。
4. 评估。保留集上的 CoNLL F1（MUC + B³ + CEAF-φ4 平均）+ 在 20 个文档上的手动簇审核。

拒绝在超过 2,000 token 的文档上仅使用 LLM 共指而没有滑动窗口合并。拒绝任何运行共指而没有提及级精确率-召回率报告的流程。标记在人口统计学多样文本中部署的基于性别启发式的系统。
```

## 练习

1. **简单。** 在 5 个手工制作的段落上运行`code/main.py`中基于规则的消解器。测量相对于真实标签的提及链接准确率。
2. **中等。** 在一篇新闻文章上使用预训练神经共指模型。将簇与你自己的手动注释进行比较。它在何处失败了？
3. **困难。** 构建一个共指增强的 NER 流程：首先 NER，然后通过共指簇合并。在 100 篇文章上测量相对于仅 NER 的实体覆盖提升。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| 提及 | 一个指称 | 指向一个实体的一段文本（名称、代词、名词短语）。 |
| 先行词 | "it"所指的对象 | 后来的指称与之共指的更早提及。 |
| 簇 | 实体的提及 | 所有引用同一真实世界实体的提及集合。 |
| 回指 | 向后指称 | 后来的提及指涉更早的（"he" → "John"）。 |
| 预指 | 向前指称 | 更早的提及指涉后来的（"When he arrived, John..."）。 |
| 桥接 | 隐式指称 | "I bought a car. The wheels were bad."（那辆车的轮子。） |
| CoNLL F1 | 排行榜上的数字 | MUC、B³、CEAF-φ4 F1 分数的平均值。 |

## 扩展阅读

- [Jurafsky & Martin, SLP3 Ch. 26 — Coreference Resolution and Entity Linking](https://web.stanford.edu/~jurafsky/slp3/26.pdf) — 规范的教科书章节。
- [Lee et al. (2017). End-to-end Neural Coreference Resolution](https://arxiv.org/abs/1707.07045) — 基于跨度的端到端。
- [Joshi et al. (2020). SpanBERT](https://arxiv.org/abs/1907.10529) — 改善共指的预训练。
- [Pradhan et al. (2012). CoNLL-2012 Shared Task](https://aclanthology.org/W12-4501/) — 基准测试。
- [Hobbs (1978). Resolving Pronoun References](https://www.sciencedirect.com/science/article/pii/0024384178900064) — 基于规则的经典之作。
