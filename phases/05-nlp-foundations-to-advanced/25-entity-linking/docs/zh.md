# 实体链接与消歧

> NER 找到了"Paris。"实体链接决定：法国巴黎？Paris Hilton？得克萨斯州帕里斯？Paris（特洛伊王子）？没有链接，你的知识图谱保持模糊。

**类型：** 构建
**语言：** Python
**先修要求：** 第五阶段 · 06（NER），第五阶段 · 24（共指消解）
**预计时间：** 约60分钟

## 问题

一个句子写道："Jordan beat the press。"你的 NER 将"Jordan"标注为 PERSON。好。但是*哪个* Jordan？

- Michael Jordan（篮球）？
- Michael B. Jordan（演员）？
- Michael I. Jordan（伯克利 ML 教授——没错，这种混淆在 ML 论文中确实存在）？
- Jordan（国家）？
- Jordan（希伯来语名字）？

实体链接（EL）将每个提及消解到知识库中的唯一条目：Wikidata、Wikipedia、DBpedia 或你的领域 KB。两个子任务：

1. **候选生成。** 给定"Jordan"，哪些 KB 条目是可能的？
2. **消歧。** 给定上下文，哪个候选是正确的？

两个步骤都是可学习的。两者都被基准测试。组合流程已稳定了十年——变化的是消歧器的质量。

## 概念

![Entity linking pipeline: mention → candidates → disambiguated entity](../assets/entity-linking.svg)

**候选生成。** 给定提及的表面形式（"Jordan"），在别名索引中查找候选。Wikipedia 别名字典覆盖大多数命名实体："JFK" → John F. Kennedy、Jacqueline Kennedy、JFK 机场、JFK（电影）。典型索引每个提及返回 10-30 个候选。

**消歧：三种方法。**

1. **先验 + 上下文（Milne & Witten, 2008）。** `P(entity | mention) × context-similarity(entity, text)`。效果良好，快速，无需训练。
2. **基于嵌入的（ESS / REL / Blink）。** 编码提及 + 上下文。编码每个候选的描述。选择最大余弦。2020-2024 年的默认选择。
3. **生成式（GENRE, 2021；基于 LLM 的, 2023+）。** 逐 token 解码实体的规范名称。被约束到一个有效实体名称的 trie，使得输出保证是一个有效的 KB id。

**端到端 vs 流程。** 现代模型（ELQ、BLINK、ExtEnD、GENRE）在一次传递中运行 NER + 候选生成 + 消歧。流程系统在生产中仍占主导，因为你可以替换组件。

### 两个测量

- **提及召回率（候选生成）。** 正确 KB 条目出现在候选列表中的黄金提及占比。整个流程的底线。
- **消歧准确率 / F1。** 给定正确的候选，top-1 正确的频率。

始终报告两者。一个在 80% 候选召回率上有 99% 消歧率的系统是一个 80% 的流程。

## 构建它

### 步骤 1：从 Wikipedia 重定向构建别名索引

```python
alias_to_entities = {
    "jordan": ["Q41421 (Michael Jordan)", "Q810 (Jordan, country)", "Q254110 (Michael B. Jordan)"],
    "paris":  ["Q90 (Paris, France)", "Q663094 (Paris, Texas)", "Q55411 (Paris Hilton)"],
    "apple":  ["Q312 (Apple Inc.)", "Q89 (apple, fruit)"],
}
```

Wikipedia 别名数据：约 18M（别名、实体）对。从 Wikidata 转储下载。存储为倒排索引。

### 步骤 2：基于上下文的消歧

```python
def disambiguate(mention, context, alias_index, entity_desc):
    candidates = alias_index.get(mention.lower(), [])
    if not candidates:
        return None, 0.0
    context_words = set(tokenize(context))
    best, best_score = None, -1
    for entity_id in candidates:
        desc_words = set(tokenize(entity_desc[entity_id]))
        union = len(context_words | desc_words)
        score = len(context_words & desc_words) / union if union else 0.0
        if score > best_score:
            best, best_score = entity_id, score
    return best, best_score
```

Jaccard 重叠是玩具实现。替换为嵌入上的余弦相似度（参见`code/main.py`步骤 2 获取 transformer 版本）。

### 步骤 3：基于嵌入的（BLINK 风格）

```python
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_mention(text, mention_span):
    start, end = mention_span
    marked = f"{text[:start]} [MENTION] {text[start:end]} [/MENTION] {text[end:]}"
    return encoder.encode([marked], normalize_embeddings=True)[0]

def embed_entity(entity_id, description):
    return encoder.encode([f"{entity_id}: {description}"], normalize_embeddings=True)[0]
```

在索引时，一次嵌入每个 KB 实体。在查询时，一次嵌入提及 + 上下文，与候选池点积，选择最大值。

### 步骤 4：生成式实体链接（概念）

GENRE 逐字符解码实体的 Wikipedia 标题。约束解码（见第 20 课）确保只能输出有效标题。与 KB 支持的 trie 紧密集成。现代后代是 REL-GEN 和带结构化输出的 LLM 提示实体链接。

```python
prompt = f"""Text: {text}
Mention: {mention}
List the best Wikipedia title for this mention.
Respond with JSON: {{"title": "..."}}"""
```

结合白名单（Outlines `choice`），这是 2026 年可以发布的最简单实体链接流程。

### 步骤 5：在 AIDA-CoNLL 上评估

AIDA-CoNLL 是标准实体链接基准：1,393 篇 Reuters 文章，34k 提及，Wikipedia 实体。报告 KB 内准确率（`P@1`）和 KB 外 NIL 检测率。

## 陷阱

- **NIL 处理。** 某些提及不在 KB 中（新兴实体、不知名人物）。系统必须预测 NIL 而不是猜测错误的实体。单独测量。
- **提及边界错误。** 上游 NER 遗漏部分跨度（"Bank of America"只标注了"Bank"）。实体链接召回率下降。
- **流行度偏置。** 训练的系统过度预测频繁实体。ML 论文中对"Michael I. Jordan"的提及经常链接到篮球 Jordan。
- **跨语言实体链接。** 将中文文本中的提及映射到英文 Wikipedia 实体。需要多语言编码器或翻译步骤。
- **KB 过时。** 新公司、事件、人物不在去年的 Wikipedia 转储中。生产流程需要刷新循环。

## 使用它

2026 年技术栈：

| 场景 | 选择 |
|-----------|------|
| 通用英语 + Wikipedia | BLINK 或 REL |
| 跨语言、KB = Wikipedia | mGENRE |
| LLM 友好、每天少量提及 | 用候选列表 + 约束 JSON 提示 Claude/GPT-4 |
| 领域特定 KB（医学、法律） | 自定义 BERT 带 KB 感知检索 + 在领域 AIDA 风格集上微调 |
| 超低延迟 | 仅精确匹配先验（Milne-Witten 基线） |
| 研究最先进 | GENRE / ExtEnD / 生成式 LLM-EL |

2026 年发布的生产模式：NER → 共指 → 每个提及的实体链接 → 将簇坍缩为每个簇一个规范实体。输出：文档中每个实体一个 KB id，而不是每个提及一个。

## 交付它

保存为 `outputs/skill-entity-linker.md`：

```markdown
---
name: entity-linker
description: 设计实体链接流程——KB、候选生成器、消歧器、评估。
version: 1.0.0
phase: 5
lesson: 25
tags: [nlp, entity-linking, knowledge-graph]
---

给定一个用例（领域 KB、语言、量、延迟预算），输出：

1. 知识库。Wikidata / Wikipedia / 自定义 KB。版本日期。刷新频率。
2. 候选生成器。别名索引、嵌入或混合。目标提及召回率 @ K。
3. 消歧器。先验 + 上下文、基于嵌入、生成式或 LLM 提示。
4. NIL 策略。Top 分数阈值、分类器或显式 NIL 候选。
5. 评估。提及召回率 @ 30、top-1 准确率、保留集上的 NIL 检测 F1。

拒绝任何没有提及召回率基线的实体链接流程（在不知道候选生成是否显示正确实体的情况下，无法评估消歧器）。拒绝任何使用 LLM 提示实体链接而没有约束输出到有效 KB id 的流程。标记流行度偏置在没有领域微调的情况下影响少数实体（例如名称冲突）的系统。
```

## 练习

1. **简单。** 在 10 个模糊提及（Paris、Jordan、Apple）上实现`code/main.py`中的先验+上下文消歧器。手动标注正确的实体。测量准确率。
2. **中等。** 用句子 transformer 编码 50 个模糊提及。嵌入每个候选的描述。将基于嵌入的消歧与 Jaccard 上下文重叠进行比较。
3. **困难。** 构建一个 1k 实体的领域 KB（例如你公司的员工 + 产品）。实现端到端的 NER + 实体链接。在 100 个保留句子上测量精确率和召回率。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| 实体链接（EL）| 链接到 Wikipedia | 将提及映射到唯一的 KB 条目。 |
| 候选生成 | 可能是谁？ | 为提及返回一组可能的 KB 条目简短列表。 |
| 消歧 | 选择正确的一个 | 使用上下文对候选评分，挑选赢家。 |
| 别名索引 | 查找表 | 从表面形式 → 候选实体的映射。 |
| NIL | 不在 KB 中 | 明确预测没有 KB 条目匹配。 |
| KB | 知识库 | Wikidata、Wikipedia、DBpedia 或你的领域 KB。 |
| AIDA-CoNLL | 基准测试 | 1,393 篇带黄金实体链接的 Reuters 文章。 |

## 扩展阅读

- [Milne, Witten (2008). Learning to Link with Wikipedia](https://www.cs.waikato.ac.nz/~ihw/papers/08-DM-IHW-LearningToLinkWithWikipedia.pdf) — 基础的先验+上下文方法。
- [Wu et al. (2020). Zero-shot Entity Linking with Dense Entity Retrieval (BLINK)](https://arxiv.org/abs/1911.03814) — 基于嵌入的主力。
- [De Cao et al. (2021). Autoregressive Entity Retrieval (GENRE)](https://arxiv.org/abs/2010.00904) — 带约束解码的生成式实体链接。
- [Hoffart et al. (2011). Robust Disambiguation of Named Entities in Text (AIDA)](https://www.aclweb.org/anthology/D11-1072.pdf) — 基准测试论文。
- [REL: An Entity Linker Standing on the Shoulders of Giants (2020)](https://arxiv.org/abs/2006.01969) — 开放生产技术栈。
