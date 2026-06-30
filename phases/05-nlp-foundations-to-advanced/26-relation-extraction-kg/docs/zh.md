# 关系提取与知识图谱构建

> NER 找到了实体。实体链接锚定了它们。关系提取找到它们之间的边。知识图谱是节点、边及其来源的总和。

**类型：** 构建
**语言：** Python
**先修要求：** 第五阶段 · 06（NER），第五阶段 · 25（实体链接）
**预计时间：** 约60分钟

## 问题

一位分析师阅读："Tim Cook became CEO of Apple in 2011。"四个事实：

- `(Tim Cook, role, CEO)`
- `(Tim Cook, employer, Apple)`
- `(Tim Cook, start_date, 2011)`
- `(Apple, type, Organization)`

关系提取（RE）将自由文本转换为结构化三元组`(subject, relation, object)`。跨语料库聚合，你得到一个知识图谱。聚合并查询，你得到 RAG、分析或合规审计的推理基础。

2026 年的问题：LLM 热情地提取关系。太过热情了。它们幻觉出来的三元组源文本不支持。没有来源，你无法区分真实的三元组和看似合理的编造。2026 年的答案是 AEVS 式锚定-验证流程。

## 概念

![Text → triples → knowledge graph](../assets/relation-extraction.svg)

**三元组形式。** `(subject_entity, relation_type, object_entity)`。关系来自封闭本体论（Wikidata 属性、FIBO、UMLS）或开放集（OpenIE 式，一切皆可）。

**三种提取方法。**

1. **基于规则 / 模式的。** Hearst 模式："X such as Y" → `(Y, isA, X)`。加上手工编写的正则表达式。脆弱、精确、可解释。
2. **有监督分类器。** 给定一个句子中的两个实体提及，从固定集合中预测关系。在 TACRED、ACE、KBP 上训练。2015–2022 年的标准。
3. **生成式 LLM。** 提示模型发出三元组。开箱即用。需要来源，否则幻觉出看似合理的垃圾。

**AEVS（锚定-提取-验证-补充, 2026）。** 当前的幻觉缓解框架：

- **锚定。** 用精确位置识别每个实体跨度和关系短语跨度。
- **提取。** 生成链接到锚定跨度的三元组。
- **验证。** 将每个三元组元素匹配回源文本；拒绝任何不被支持的。
- **补充。** 覆盖传递确保没有锚定跨度被丢弃。

幻觉急剧下降。需要更多计算但可审计。

**开放与封闭的权衡。**

- **封闭本体论。** 固定属性列表（例如，Wikidata 的 11,000+ 属性）。可预测。可查询。难以编造。
- **开放 IE。** 任何动词短语成为一个关系。高召回率。低精确率。查询混乱。

生产知识图谱通常混合：开放 IE 用于发现，然后规范化关系到封闭本体论，再合并到主图。

## 构建它

### 步骤 1：基于模式的提取

```python
PATTERNS = [
    (r"(?P<s>[A-Z]\w+) (?:is|was) (?:a|an|the) (?P<o>[A-Z]?\w+)", "isA"),
    (r"(?P<s>[A-Z]\w+) (?:is|was) born in (?P<o>\w+)", "bornIn"),
    (r"(?P<s>[A-Z]\w+) works? (?:at|for) (?P<o>[A-Z]\w+)", "worksAt"),
    (r"(?P<s>[A-Z]\w+) founded (?P<o>[A-Z]\w+)", "founded"),
]
```

参见`code/main.py`获取完整的玩具提取器。Hearst 模式仍在领域特定流程中发布，因为它们是可调试的。

### 步骤 2：有监督关系分类

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSequenceClassification.from_pretrained("Babelscape/rebel-large")

text = "Tim Cook was born in Alabama. He later became CEO of Apple."
encoded = tok(text, return_tensors="pt", truncation=True)
output = model.generate(**encoded, max_length=200)
triples = tok.batch_decode(output, skip_special_tokens=False)
```

REBEL 是一个 seq2seq 关系提取器：文本输入，三元组输出，已经使用 Wikidata 属性 id。在远程监督数据上微调。标准的开放权重基线。

### 步骤 3：带锚定的 LLM 提示提取

```python
prompt = f"""Extract (subject, relation, object) triples from the text.
For each triple, include the exact character span in the source text.

Text: {text}

Output JSON:
[{{"subject": {{"text": "...", "span": [start, end]}},
   "relation": "...",
   "object": {{"text": "...", "span": [start, end]}}}}, ...]

Only include triples fully supported by the text. No inference beyond what is stated.
"""
```

针对源文本验证每个返回的跨度。拒绝任何`text[start:end] != triple_entity`的。这是 AEVS "验证"步骤的最小形式。

### 步骤 4：规范化到封闭本体论

```python
RELATION_MAP = {
    "is the CEO of": "P169",       # "chief executive officer"
    "was born in":   "P19",         # "place of birth"
    "founded":        "P112",       # "founded by" (主语/宾语倒置)
    "works at":       "P108",       # "employer"
}


def canonicalize(relation):
    rel_low = relation.lower().strip()
    if rel_low in RELATION_MAP:
        return RELATION_MAP[rel_low]
    return None   # 丢弃未映射的开放关系或路由到手动审核
```

规范化通常是 60-80% 的工程工作。为它留出预算。

### 步骤 5：构建一个小型图谱并查询

```python
triples = extract(text)
graph = {}
for s, r, o in triples:
    graph.setdefault(s, []).append((r, o))


def neighbors(node, relation=None):
    return [(r, o) for r, o in graph.get(node, []) if relation is None or r == relation]


print(neighbors("Tim Cook", relation="P108"))    # -> [(P108, Apple)]
```

这是每个基于 KG 的 RAG 系统的原子。用 RDF 三元组存储（Blazegraph、Virtuoso）、属性图（Neo4j）或向量增强的图存储来扩展它。

## 陷阱

- **关系提取前进行共指消解。** "He founded Apple" ——关系提取需要知道"he"是谁。首先运行共指消解（第 24 课）。
- **实体规范化。** "Apple Inc"和"Apple"必须消解到同一节点。先进行实体链接（第 25 课）。
- **幻觉三元组。** LLM 发出文本不支持的的三元组。强制执行跨度验证。
- **关系规范化漂移。** 开放 IE 的关系不一致（"was born in," "came from," "is a native of"）。坍缩到规范 id，否则图谱不可查询。
- **时间错误。** "Tim Cook is CEO of Apple"——现在为真，2005 年为假。许多关系是有时间边界的。使用限定符（Wikidata 中`P580`开始时间、`P582`结束时间）。
- **领域不匹配。** REBEL 在 Wikipedia 上训练。法律、医学和科学文本通常需要领域微调的关系提取模型。

## 使用它

2026 年技术栈：

| 场景 | 选择 |
|-----------|------|
| 快速生产、通用领域 | REBEL 或 LlamaPred 配合 Wikidata 规范化 |
| 领域特定（生物医学、法律）| SciREX 式领域微调 + 自定义本体论 |
| LLM 提示、可审计输出 | AEVS 流程：锚定 → 提取 → 验证 → 补充 |
| 高量新闻 IE | 基于模式 + 有监督混合 |
| 从零构建 KG | 开放 IE + 手动规范化遍次 |
| 时间知识图谱 | 带限定符提取（开始/结束时间、时间点）|

集成模式：NER → 共指 → 实体链接 → 关系提取 → 本体论映射 → 图谱加载。每个阶段都是一个潜在的质量把关。

## 交付它

保存为 `outputs/skill-re-designer.md`：

```markdown
---
name: re-designer
description: 设计带来源和规范化的关系提取流程。
version: 1.0.0
phase: 5
lesson: 26
tags: [nlp, relation-extraction, knowledge-graph]
---

给定一个语料库（领域、语言、量）和下游用途（KG-RAG、分析、合规），输出：

1. 提取器。基于模式 / 有监督 / LLM / AEVS 混合。与精确率 vs 召回率目标相关联的理由。
2. 本体论。封闭属性列表（Wikidata / 领域）或带规范化遍次的开放 IE。
3. 来源。每个三元组携带源字符跨度 + 文档 id。对于审计不可协商。
4. 合并策略。规范实体 id + 关系 id + 时间限定符；去重策略。
5. 评估。在 200 个人工标注三元组上的精确率 / 召回率 + LLM 提取样本上的幻觉率。

拒绝任何没有跨度验证（源来源）的基于 LLM 的关系提取流程。拒绝开放 IE 输出在没有规范化的情况下流入生产图谱。标记在对有时间边界关系（雇主、配偶、职位）的流程中没有时间限定符的。
```

## 练习

1. **简单。** 在 5 个新闻文章句子上运行`code/main.py`中的模式提取器。手动检查精确率。
2. **中等。** 在相同句子上使用 REBEL（或一个小型 LLM）。比较三元组。哪个提取器有更高精确率？更高召回率？
3. **困难。** 构建 AEVS 流程：用 LLM 提取 + 针对源文本验证跨度。在 50 个 Wikipedia 风格句子上测量验证步骤前后的幻觉率。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| 三元组 | 主语-关系-宾语 | `(s, r, o)`元组，KG 的原子单元。 |
| 开放 IE | 提取任何内容 | 开放词汇关系短语；高召回率，低精确率。 |
| 封闭本体论 | 固定模式 | 有界关系类型集合（Wikidata、UMLS、FIBO）。 |
| 规范化 | 归一化一切 | 将表面名称 / 关系映射到规范 id。 |
| AEVS | 接地提取 | 锚定-提取-验证-补充流程（2026）。 |
| 来源 | 真实来源链接 | 每个三元组携带一个文档 id + 字符跨度指向其源。 |
| 远程监督 | 廉价标签 | 将文本与现有 KG 对齐以创建训练数据。 |

## 扩展阅读

- [Mintz et al. (2009). Distant supervision for relation extraction without labeled data](https://www.aclweb.org/anthology/P09-1113.pdf) — 远程监督论文。
- [Huguet Cabot, Navigli (2021). REBEL: Relation Extraction By End-to-end Language generation](https://aclanthology.org/2021.findings-emnlp.204.pdf) — seq2seq 关系提取主力。
- [Wadden et al. (2019). Entity, Relation, and Event Extraction with Contextualized Span Representations (DyGIE++)](https://arxiv.org/abs/1909.03546) — 联合 IE。
- [AEVS — Anchor-Extraction-Verification-Supplement framework](https://www.mdpi.com/2073-431X/15/3/178) — 2026 幻觉缓解设计。
- [Wikidata SPARQL tutorial](https://www.wikidata.org/wiki/Wikidata:SPARQL_tutorial) — 规范图谱查询。
