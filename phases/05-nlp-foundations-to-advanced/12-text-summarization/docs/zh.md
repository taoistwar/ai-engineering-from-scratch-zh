# 文本摘要

> 提取式系统告诉你文档说了什么。生成式系统告诉你作者是什么意思。不同的任务，不同的陷阱。

**类型：** 构建
**语言：** Python
**先修要求：** 第五阶段 · 02（BoW + TF-IDF），第五阶段 · 11（机器翻译）
**预计时间：** 约75分钟

## 问题

一篇 2000 词的新闻文章出现在你的信息流中。你需要 120 词来捕获它。你可以从文章中挑选最重要的三个句子（提取式），或者用自己的话重写内容（生成式）。两者都称为摘要。但它们是完全不同的问题。

提取式摘要是一个排序问题。给每个句子打分，返回 top-`k`。输出总是符合语法，因为它是逐字提取的。风险是遗漏分布在整个文章中的内容。

生成式摘要是一个生成问题。一个 transformer 以输入为条件产生新文本。输出流畅且压缩，但可能幻觉出源中不存在的事实。风险是自信的编造。

本课构建两者，以及每种拥有的失败模式。

## 概念

![Extractive TextRank vs abstractive transformer](../assets/summarization.svg)

**提取式。** 将文章视为图，节点是句子，边是相似度。在图里运行 PageRank（或类似的东西）根据句子与所有其他句子的连接程度对其评分。得分最高的句子就是摘要。规范的实现是**TextRank**（Mihalcea and Tarau, 2004）。

**生成式。** 在文档-摘要对上微调一个 transformer 编码器-解码器（BART、T5、Pegasus）。在推理时，模型读取文档，通过交叉注意力逐 token 生成摘要。特别是 Pegasus 使用了一个间隔句子预训练目标，使其在无需太多微调的情况下擅长摘要。

用**ROUGE**（面向召回率的要点评估候补）评估。ROUGE-1 和 ROUGE-2 对单词和二元组重叠进行评分。ROUGE-L 对最长公共子序列进行评分。越高越好，但 40 ROUGE-L 是"良好"，50 是"卓越"。每篇论文都报告所有三个。使用`rouge-score`包。

## 构建它

### 步骤 1：TextRank（提取式）

```python
import math
import re
from collections import Counter


def sentence_split(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())


def similarity(s1, s2):
    w1 = Counter(s1.lower().split())
    w2 = Counter(s2.lower().split())
    intersection = sum((w1 & w2).values())
    denom = math.log(len(w1) + 1) + math.log(len(w2) + 1)
    if denom == 0:
        return 0.0
    return intersection / denom


def textrank(text, top_k=3, damping=0.85, iterations=50, epsilon=1e-4):
    sentences = sentence_split(text)
    n = len(sentences)
    if n <= top_k:
        return sentences

    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                sim[i][j] = similarity(sentences[i], sentences[j])

    scores = [1.0] * n
    for _ in range(iterations):
        new_scores = [1 - damping] * n
        for i in range(n):
            total_out = sum(sim[i]) or 1e-9
            for j in range(n):
                if sim[i][j] > 0:
                    new_scores[j] += damping * sim[i][j] / total_out * scores[i]
        if max(abs(s - ns) for s, ns in zip(scores, new_scores)) < epsilon:
            scores = new_scores
            break
        scores = new_scores

    ranked = sorted(range(n), key=lambda k: scores[k], reverse=True)[:top_k]
    ranked.sort()
    return [sentences[i] for i in ranked]
```

有两件事值得指出。相似度函数使用对数归一化的词重叠，这是原始 TextRank 变体。TF-IDF 向量的余弦相似度也可以。衰减因子 0.85 和迭代次数是 PageRank 的默认值。

### 步骤 2：使用 BART 的生成式摘要

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """(长新闻文章文本)"""

summary = summarizer(article, max_length=120, min_length=60, do_sample=False)
print(summary[0]["summary_text"])
```

BART-large-CNN 在 CNN/DailyMail 语料库上微调。它可以开箱即用生成新闻风格摘要。对于其他领域（科学论文、对话、法律），使用相应的 Pegasus 检查点或在目标数据上微调。

### 步骤 3：ROUGE 评估

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
scores = scorer.score(reference_summary, generated_summary)
print({k: round(v.fmeasure, 3) for k, v in scores.items()})
```

始终使用词干提取。没有它，"running"和"run"被视为不同的词，ROUGE 会低估。

### 超越 ROUGE（2026 年摘要评估）

ROUGE 二十年来一直是主导的摘要指标，但 2026 年它自身是不够的。一项对 NLG 论文的大规模元分析显示：

- **BERTScore**（上下文嵌入相似度）在 2023 年取得进展，现在在大多数摘要论文中与 ROUGE 一起报告。
- **BARTScore**将评估视为生成：通过一个预训练 BART 赋予摘要以源为条件的可能性来评分。
- **MoverScore**（在上下文嵌入上的推土机距离）在 2025 年摘要基准测试中登顶，因为它比 ROUGE 更好地捕获了语义重叠。
- **FactCC**和**基于 QA 的事实性**在 2021-2023 年很常见，现在经常被**G-Eval**（一个带有思维链推理的 GPT-4 提示链，对连贯性、一致性、流畅性、相关性进行评分）替代。
- **G-Eval**和类似的 LLM 评判者方法在评分标准设计良好时，大约 80% 的时间与人类判断一致。

生产建议：报告 ROUGE-L 用于遗留比较，BERTScore 用于语义重叠，G-Eval 用于连贯性和事实性。用 50-100 个人工标记摘要进行校准。

### 步骤 4：事实性问题

生成式摘要容易产生幻觉。提取式摘要的幻觉风险低得多，因为输出是从源逐字提取的，但如果源句子被脱离上下文、过时或乱序引用，它们仍然可能误导人。这是生产系统在合规相关内容方面仍然偏好提取式方法的最大单一原因。

需要命名的幻觉类型：

- **实体交换。** 源说"John Smith。"摘要说"John Brown。"
- **数字漂移。** 源说"25,000。"摘要说"25 million。"
- **极性翻转。** 源说"rejected the offer。"摘要说"accepted the offer。"
- **事实编造。** 源未提及 CEO。摘要说 CEO 批准了。

有效的评估方法：

- **FactCC。** 一个在源句子和摘要句子之间的蕴涵关系上训练的二分类器。预测事实/非事实。
- **基于 QA 的事实性。** 向 QA 模型提出问题，其答案在源中。如果摘要支持不同的答案，标记。
- **实体级 F1。** 比较源与摘要中的命名实体。仅出现在摘要中的实体是可疑的。

对于任何事实性重要的面向用户的内容（新闻、医学、法律、金融），提取式是更安全的默认选择。生成式需要在循环中有一个事实性检查。

## 使用它

2026 年技术栈：

| 用例 | 推荐 |
|---------|-------------|
| 新闻、3-5 句摘要、英语 | `facebook/bart-large-cnn` |
| 科学论文 | `google/pegasus-pubmed`或调优的 T5 |
| 多文档、长格式 | 任何具有 32k+ 上下文的 LLM，经提示 |
| 对话摘要 | `philschmid/bart-large-cnn-samsum` |
| 提取式，通过构造低幻觉风险 | TextRank 或`sumy`的 LSA / LexRank |

2026 年，当计算不是约束条件时，具有长上下文的 LLM 经常优于专门模型。权衡是成本和可复现性；专门的模型产生更一致的输出。

## 交付它

保存为 `outputs/skill-summary-picker.md`：

```markdown
---
name: summary-picker
description: 选择提取式或生成式，命名库，事实性检查。
version: 1.0.0
phase: 5
lesson: 12
tags: [nlp, summarization]
---

给定一个任务（文档类型、合规要求、长度、计算预算），输出：

1. 方法。提取式或生成式。用一句话解释为什么。
2. 起始模型 / 库。命名它。`sumy.TextRankSummarizer`、`facebook/bart-large-cnn`、`google/pegasus-pubmed`或一个 LLM 提示。
3. 评估计划。ROUGE-1、ROUGE-2、ROUGE-L（使用带词干提取的 rouge-score）。如果是生成式，加上事实性检查。
4. 需要探查的一个失败模式。实体交换是生成式新闻摘要中最常见的；标记源实体不出现在摘要中的样本。

拒绝为医学、法律、金融或受监管内容进行没有事实性把关的生成式摘要。标记超过模型上下文窗口的输入需要分块 map-reduce 摘要（不仅仅是截断）。
```

## 练习

1. **简单。** 在 5 篇新闻文章上运行 TextRank。将 top-3 句子与参考摘要进行比较。测量 ROUGE-L。你应该在 CNN/DailyMail 风格的文章上看到 30-45 ROUGE-L。
2. **中等。** 实现实体级事实性：从源和摘要中提取命名实体（spaCy），计算源实体在摘要中的召回率和摘要实体与源对照的精确率。高精确率和低召回率意味着安全但简略；低精确率意味着幻觉实体。
3. **困难。** 在 50 篇 CNN/DailyMail 文章上比较 BART-large-CNN 与 LLM（Claude 或 GPT-4）。报告 ROUGE-L、事实性（按实体 F1）和每个摘要的成本。记录每种方案在何处胜出。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| 提取式 | 挑选句子 | 从源逐字返回句子。永远不会产生幻觉。 |
| 生成式 | 重写 | 以源为条件生成新文本。可能产生幻觉。 |
| ROUGE | 摘要指标 | 系统输出与参考之间的 n-gram / LCS 重叠。 |
| TextRank | 基于图的提取式 | 在句子相似度图上的 PageRank。 |
| 事实性 | 是否正确 | 摘要的声明是否源自源。 |
| 幻觉 | 编造的内容 | 摘要中不被源支持的内容。 |

## 扩展阅读

- [Mihalcea and Tarau (2004). TextRank: Bringing Order into Texts](https://aclanthology.org/W04-3252/) — 提取式的经典论文。
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461) — BART 论文。
- [Zhang et al. (2019). PEGASUS: Pre-training with Extracted Gap-sentences](https://arxiv.org/abs/1912.08777) — Pegasus 和间隔句子目标。
- [Lin (2004). ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/) — ROUGE 论文。
- [Maynez et al. (2020). On Faithfulness and Factuality in Abstractive Summarization](https://arxiv.org/abs/2005.00661) — 事实性全景论文。
