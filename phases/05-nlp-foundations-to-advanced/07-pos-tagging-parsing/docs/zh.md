# 词性标注与句法分析

> 语法曾一度不受欢迎。然后每个 LLM 流程都需要验证结构化提取，它又回来了。

**类型：** 构建
**语言：** Python
**先修要求：** 第五阶段 · 01（文本处理），第二阶段 · 14（朴素贝叶斯）
**预计时间：** 约45分钟

## 问题

第 01 课承诺词形还原需要词性标签。不知道`running`是动词，词形还原器无法将其还原为`run`。不知道`better`是形容词，它无法还原为`good`。

那个承诺隐藏了一整个子领域。词性标注指定语法类别。句法分析恢复句子的树结构：哪个词修饰哪个，哪个动词支配哪些参数。经典 NLP 花了二十年完善这两者。然后深度学习将它们简化为预训练 transformer 之上的一个 token 分类任务，研究社区继续前进。

应用社区没有。每个结构化提取流程在底层仍然使用 POS 和依存树。LLM 生成的 JSON 根据语法约束进行验证。问答系统使用依存分析分解查询。机器翻译质量评估器检查分析树的对其。

值得了解。本课介绍标签集、基线，以及你停止从零实现而调用 spaCy 的节点。

## 概念

**POS 标注**为每个 token 分配一个语法类别。**宾州树库（PTB）**标签集是英语默认标准。36 个标签，其中包含让随意读者觉得小题大做的区分：`NN`单数名词，`NNS`复数名词，`NNP`专有名词单数，`VBD`动词过去式，`VBZ`动词第三人称单数现在式等等。**通用依存（UD）**标签集更粗粒度（17 个标签），且与语言无关；它成为跨语言工作的默认标准。

```
The/DET cats/NOUN were/AUX running/VERB at/ADP 3pm/NOUN ./PUNCT
```

**句法分析**产生一棵树。两种主要风格：

- **短语结构分析。** 名词短语、动词短语、介词短语嵌套在彼此内部。输出是一棵以词为叶子的非终结符类别（NP、VP、PP）树。
- **依存分析。** 每个词有一个它所依赖的单一中心词，标记以语法关系。输出是一棵树，其中每条边是一个（中心词，从属词，关系）三元组。

依存分析在 2010 年代胜出，因为它跨语言（尤其是自由语序语言）干净地泛化。

```
running is ROOT
cats is nsubj of running
were is aux of running
at is prep of running
3pm is pobj of at
```

## 构建它

### 步骤 1：最频繁标签基线

可行的最笨 POS 标注器。对于每个词，预测它在训练中最多出现的标签。

```python
from collections import Counter, defaultdict


def train_mft(train_examples):
    word_tag_counts = defaultdict(Counter)
    all_tags = Counter()
    for tokens, tags in train_examples:
        for token, tag in zip(tokens, tags):
            word_tag_counts[token.lower()][tag] += 1
            all_tags[tag] += 1
    word_best = {w: c.most_common(1)[0][0] for w, c in word_tag_counts.items()}
    default_tag = all_tags.most_common(1)[0][0]
    return word_best, default_tag


def predict_mft(tokens, word_best, default_tag):
    return [word_best.get(t.lower(), default_tag) for t in tokens]
```

在 Brown 语料库上，这个基线命中约 85% 的准确率。不够好，但这是任何严肃模型不应低于的底线。

### 步骤 2：二元组 HMM 标注器

对序列的联合概率进行建模：

```
P(tags, words) = prod P(tag_i | tag_{i-1}) * P(word_i | tag_i)
```

两个表：转移概率（给定前一个标签的标签概率）、发射概率（给定标签的词概率）。通过带拉普拉斯平滑的计数估计两者。用维特比算法解码（标签格上的动态规划）。

```python
import math


def train_hmm(train_examples, alpha=0.01):
    transitions = defaultdict(Counter)
    emissions = defaultdict(Counter)
    tags = set()
    vocab = set()

    for tokens, ts in train_examples:
        prev = "<BOS>"
        for token, tag in zip(tokens, ts):
            transitions[prev][tag] += 1
            emissions[tag][token.lower()] += 1
            tags.add(tag)
            vocab.add(token.lower())
            prev = tag
        transitions[prev]["<EOS>"] += 1

    return transitions, emissions, tags, vocab


def log_prob(table, given, key, smooth_denom, alpha):
    return math.log((table[given].get(key, 0) + alpha) / smooth_denom)


def viterbi(tokens, transitions, emissions, tags, vocab, alpha=0.01):
    tags_list = list(tags)
    n = len(tokens)
    V = [[0.0] * len(tags_list) for _ in range(n)]
    back = [[0] * len(tags_list) for _ in range(n)]

    for j, tag in enumerate(tags_list):
        em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
        tr_denom = sum(transitions["<BOS>"].values()) + alpha * (len(tags_list) + 1)
        tr = log_prob(transitions, "<BOS>", tag, tr_denom, alpha)
        em = log_prob(emissions, tag, tokens[0].lower(), em_denom, alpha)
        V[0][j] = tr + em
        back[0][j] = 0

    for i in range(1, n):
        for j, tag in enumerate(tags_list):
            em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
            em = log_prob(emissions, tag, tokens[i].lower(), em_denom, alpha)
            best_prev = 0
            best_score = -1e30
            for k, prev_tag in enumerate(tags_list):
                tr_denom = sum(transitions[prev_tag].values()) + alpha * (len(tags_list) + 1)
                tr = log_prob(transitions, prev_tag, tag, tr_denom, alpha)
                score = V[i - 1][k] + tr + em
                if score > best_score:
                    best_score = score
                    best_prev = k
            V[i][j] = best_score
            back[i][j] = best_prev

    last_best = max(range(len(tags_list)), key=lambda j: V[n - 1][j])
    path = [last_best]
    for i in range(n - 1, 0, -1):
        path.append(back[i][path[-1]])
    return [tags_list[j] for j in reversed(path)]
```

在 Brown 语料库上的二元组 HMM 命中约 93% 的准确率。从 85% 到 93% 的跳跃主要是转移概率——模型学到`DET NOUN`是常见的而`NOUN DET`是罕见的。

### 步骤 3：为什么现代标注器超越这个

转移 + 发射概率是局部的。它们无法捕捉到`saw`在"I bought a saw"中是名词而在"I saw the movie"中是动词。一个具有任意特征（后缀、词形、前后词、词本身）的 CRF 命中约 97%。BiLSTM-CRF 或 transformer 命中约 98%+。

这个任务的上限由标注者分歧设定。在宾州树库上，人工标注者大约 97% 的时间达成一致。超过 98% 的模型可能是在测试集上过拟合。

### 步骤 4：依存分析概述

完整的从零实现依存分析不在范围内；规范的教科书处理在 Jurafsky 和 Martin 的著作中。需要了解的两个经典系列：

- **基于转移的**分析器（arc-eager、arc-standard）表现得像移进-归约分析器：它们读取 token、将它们移入栈中，并应用创建弧的归约动作。贪心解码很快。经典实现是 MaltParser。现代神经版本：Chen 和 Manning 的基于转移的分析器。
- **基于图的**分析器（Eisner 算法、Dozat-Manning 双仿射）为每个可能的中心词-从属词边打分，并选择最大生成树。更慢但更准确。

对于大多数应用工作，调用 spaCy：

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running at 3pm.")
for token in doc:
    print(f"{token.text:10s} tag={token.tag_:5s} pos={token.pos_:6s} dep={token.dep_:10s} head={token.head.text}")
```

```
The        tag=DT    pos=DET    dep=det        head=cats
cats       tag=NNS   pos=NOUN   dep=nsubj      head=running
were       tag=VBD   pos=AUX    dep=aux        head=running
running    tag=VBG   pos=VERB   dep=ROOT       head=running
at         tag=IN    pos=ADP    dep=prep       head=running
3pm        tag=NN    pos=NOUN   dep=pobj       head=at
.          tag=.     pos=PUNCT  dep=punct      head=running
```

从下到上阅读`dep`列，句子的语法结构就自然呈现。

## 使用它

每个生产 NLP 库都作为标准流程的一部分提供 POS 和依存分析器。

- **spaCy**（`en_core_web_sm` / `md` / `lg` / `trf`）。快速、准确，与分词 + NER + 词形还原集成。`token.tag_`（Penn），`token.pos_`（UD），`token.dep_`（依存关系）。
- **Stanford NLP (stanza)**。Stanford 的 CoreNLP 继任者。60+ 种语言上最先进。
- **trankit**。基于 transformer，良好的 UD 准确率。
- **NLTK**。`pos_tag`。可用、较慢、较旧。适用于教学。

### 在 2026 年，这仍然重要在哪里

- **词形还原。** 第 01 课需要 POS 才能正确进行词形还原。始终如此。
- **从 LLM 输出进行结构化提取。** 验证生成的句子遵守语法约束（例如，主谓一致、必需的修饰语）。
- **基于方面的情感。** 依存分析告诉你哪个形容词修饰哪个名词。
- **查询理解。** "movies directed by Wes Anderson starring Bill Murray"通过分析分解为结构化约束。
- **跨语言迁移。** UD 标签和依存关系是语言无关的，使得对新语言进行零样本结构化分析成为可能。
- **低计算流程。** 如果你无法部署 transformer，POS + 依存分析 + 地名词典能让你走得出乎意料地远。

## 交付它

保存为 `outputs/skill-grammar-pipeline.md`：

```markdown
---
name: grammar-pipeline
description: 为下游 NLP 任务设计经典 POS + 依存分析流程。
version: 1.0.0
phase: 5
lesson: 07
tags: [nlp, pos, parsing]
---

给定一个下游任务（信息提取、重写验证、查询分解、词形还原），你输出：

1. 使用的标签集。仅英语的遗留流程使用宾州树库，多语言或跨语言使用通用依存。
2. 库。大多数生产环境用 spaCy，学术级多语言用 stanza，最高 UD 准确率用 trankit。命名具体的模型 ID。
3. 集成模式。展示调用库并消费所需属性（`.pos_`、`.dep_`、`.head`）的 3-5 行代码。
4. 需要测试的失败模式。名-动词歧义（`saw`、`book`、`can`）和 PP-附加歧义是经典陷阱。抽样 20 个输出并目视检查。

拒绝推荐自己编写分析器。从头构建分析器是研究项目，不是应用任务。标记任何消费 POS 标签而不处理大小写变体的流程为脆弱。
```

## 练习

1. **简单。** 在一个小型标注语料库（例如，NLTK 的 Brown 子集）上使用最频繁标签基线，在保留的句子上测量准确率。验证约 85% 的结果。
2. **中等。** 训练上述二元组 HMM 并报告每个标签的精确率/召回率。HMM 最常混淆哪些标签？
3. **困难。** 使用 spaCy 的依存分析从 1000 个句子的样本中提取主-谓-宾三元组。在 50 个手动标注的三元组上评估。记录提取在何处失败（通常是被动语态、并列结构以及省略主语）。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| POS 标签 | 词的类型 | 语法类别。PTB 有 36 个；UD 有 17 个。 |
| 宾州树库 | 标准标签集 | 英语专用。细粒度的动词时态和名词数。 |
| 通用依存 | 多语言标签集 | 比 PTB 粗粒度；语言中立；跨语言工作的默认标准。 |
| 依存分析 | 句子树 | 每个词有一个中心词，每条边有一个语法关系。 |
| 维特比 | 动态规划 | 在给定发射和转移概率的情况下找到最高概率的标签序列。 |

## 扩展阅读

- [Jurafsky and Martin — Speech and Language Processing, chapters 8 and 18](https://web.stanford.edu/~jurafsky/slp3/) — POS 和句法分析的规范教科书处理。
- [Universal Dependencies project](https://universaldependencies.org/) — 每个多语言分析器使用的跨语言标签集和树库集合。
- [spaCy linguistic features guide](https://spacy.io/usage/linguistic-features) — `Token`上每个暴露属性的实用参考。
- [Chen and Manning (2014). A Fast and Accurate Dependency Parser using Neural Networks](https://nlp.stanford.edu/pubs/emnlp2014-depparser.pdf) — 将神经分析器带入主流的论文。
