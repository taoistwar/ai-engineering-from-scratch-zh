# 文本处理 — 分词、词干提取、词形还原

> 语言是连续的。模型是离散的。预处理是两者之间的桥梁。

**类型：** 构建
**语言：** Python
**先修要求：** 第二阶段 · 14（朴素贝叶斯）
**预计时间：** 约45分钟

## 问题

模型无法读懂"The cats were running."。它只理解整数。

每个 NLP 系统都以同样的三个问题开始：词的边界在哪里？词的词根是什么？何时将"run"、"running"、"ran"视为同一个事物（当有帮助时），何时又将它们视为不同的事物（当没有帮助时）？

分词搞错了，模型就会从垃圾数据中学习。如果你的分词器将`don't`视为一个 token 而将`do n't`视为两个 token，那么训练分布就会分裂。如果你词干提取器将`organization`和`organ`归为同一个词干，那么主题建模就会失败。如果你的词形还原器需要词性上下文而你却没有提供，那么动词就会被当成名词处理。

本课将从零开始构建这三个预处理步骤，然后展示 NLTK 和 spaCy 如何完成同样的工作，以便你能看到其中的权衡。

## 概念

三个操作。每个操作都有一个职责和一个失败模式。

**分词**将字符串分割成 token。"Token"这个词故意含糊不清，因为合适的粒度取决于任务。经典 NLP 使用词级。Transformer 使用子词级。没有空格的语言使用字符级。

**词干提取**使用规则切掉后缀。速度快、激进、愚蠢。`running -> run`。`organization -> organ`。第二个就是失败模式。

**词形还原**利用语法知识将单词还原为词典形式。速度较慢、准确、需要查找表或形态分析器。`ran -> run`（需要知道"ran"是"run"的过去式）。`better -> good`（需要知道比较级形式）。

经验法则：当速度重要且你能容忍噪音时（搜索索引、粗略分类），使用词干提取。当含义重要时（问答、语义搜索、任何用户会看到的内容），使用词形还原。

```figure
edit-distance
```

## 构建它

### 步骤 1：一个正则表达式分词器

最简单的可用分词器在非字母数字字符处分割，同时将标点符号作为独立的 token 保留。不完美，也不是最终方案，但一行代码就能运行。

```python
import re

def tokenize(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]", text)
```

按优先级排列的三个模式。带有可选内部撇号的单词（`don't`、`it's`）。纯数字。任何单个非空白非字母数字字符作为独立 token（标点符号）。

```python
>>> tokenize("The cats weren't running at 3pm.")
['The', 'cats', "weren't", 'running', 'at', '3', 'pm', '.']
```

需要注意的失败模式。`3pm`被分割为`['3', 'pm']`，因为我们在字母序列和数字序列之间交替匹配。对大多数任务来说足够了。URL、电子邮件、话题标签全都会出错。在生产环境中，在通用模式之前添加专门的模式。

### 步骤 2：一个 Porter 词干提取器（仅步骤 1a）

完整的 Porter 算法有五个阶段的规则。仅步骤 1a 就能涵盖最常见的英语后缀，并展示其模式。

```python
def stem_step_1a(word):
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s") and len(word) > 1:
        return word[:-1]
    return word
```

```python
>>> [stem_step_1a(w) for w in ["caresses", "ponies", "caress", "cats"]]
['caress', 'poni', 'caress', 'cat']
```

从上到下读取规则。`ies -> i`规则就是为什么`ponies -> poni`而不是`pony`。真正的 Porter 算法有步骤 1b 可以修复这个问题。规则存在竞争。更早的规则优先。顺序比任何单独的规则都重要。

### 步骤 3：一个基于查找的词形还原器

真正的词形还原需要形态学知识。一个可教的教学版本使用一个小型词形还原表和一个回退策略。

```python
LEMMA_TABLE = {
    ("running", "VERB"): "run",
    ("ran", "VERB"): "run",
    ("runs", "VERB"): "run",
    ("better", "ADJ"): "good",
    ("best", "ADJ"): "good",
    ("cats", "NOUN"): "cat",
    ("cat", "NOUN"): "cat",
    ("were", "VERB"): "be",
    ("was", "VERB"): "be",
    ("is", "VERB"): "be",
}

def lemmatize(word, pos):
    key = (word.lower(), pos)
    if key in LEMMA_TABLE:
        return LEMMA_TABLE[key]
    if pos == "VERB" and word.endswith("ing"):
        return word[:-3]
    if pos == "NOUN" and word.endswith("s"):
        return word[:-1]
    return word.lower()
```

```python
>>> lemmatize("running", "VERB")
'run'
>>> lemmatize("cats", "NOUN")
'cat'
>>> lemmatize("better", "ADJ")
'good'
>>> lemmatize("watched", "VERB")
'watched'
```

最后一个例子是关键的教学时刻。`watched`不在我们的表中，而我们的回退策略只处理`ing`。真正的词形还原覆盖`ed`、不规则动词、比较级形容词、带有音变的复数（`children -> child`）。这就是为什么生产系统使用 WordNet、spaCy 的形态分析器或完整的形态分析工具。

### 步骤 4：将它们组合起来

```python
def preprocess(text, pos_tagger=None):
    tokens = tokenize(text)
    stems = [stem_step_1a(t.lower()) for t in tokens]
    tags = pos_tagger(tokens) if pos_tagger else [(t, "NOUN") for t in tokens]
    lemmas = [lemmatize(word, pos) for word, pos in tags]
    return {"tokens": tokens, "stems": stems, "lemmas": lemmas}
```

缺失的部分是一个 POS 标注器。第五阶段 · 07（词性标注）将构建一个。目前，将所有词默认为`NOUN`并承认这个局限性。

## 使用它

NLTK 和 spaCy 提供了生产版本。各只需几行代码。

### NLTK

```python
import nltk
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng")

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

text = "The cats were running."
tokens = word_tokenize(text)
stems = [PorterStemmer().stem(t) for t in tokens]
lemmatizer = WordNetLemmatizer()
tagged = pos_tag(tokens)


def nltk_pos_to_wordnet(tag):
    if tag.startswith("V"):
        return "v"
    if tag.startswith("J"):
        return "a"
    if tag.startswith("R"):
        return "r"
    return "n"


lemmas = [lemmatizer.lemmatize(t, nltk_pos_to_wordnet(tag)) for t, tag in tagged]
```

`word_tokenize`处理缩写、Unicode 以及你的正则表达式遗漏的边缘情况。`PorterStemmer`运行全部五个阶段。`WordNetLemmatizer`需要将 POS 标签从 NLTK 的 Penn Treebank 方案转换为 WordNet 的缩写集。上面那段转换连接代码是大多数教程跳过不讲的部分。

### spaCy

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running.")

for token in doc:
    print(token.text, token.lemma_, token.pos_)
```

```
The      the     DET
cats     cat     NOUN
were     be      AUX
running  run     VERB
.        .       PUNCT
```

spaCy 将整个流程隐藏在`nlp(text)`之后。分词、词性标注和词形还原全部运行。比 NLTK 在大规模上更快。开箱即用更准确。其代价是你难以轻松替换单个组件。

### 何时选择哪个

| 场景 | 选择 |
|-----------|------|
| 教学、研究、需要替换组件 | NLTK |
| 生产环境、多语言、速度重要 | spaCy |
| Transformer 流程（反正你会使用模型的 tokenizer 进行分词） | 使用 `tokenizers` / `transformers`，跳过经典预处理 |

### 两个没人警告你的失败模式

大多数教程教完算法就停了。有两件事会让真实的预处理流程出问题，而且几乎从未被涉及。

**可复现性漂移。** NLTK 和 spaCy 在不同版本之间会改变分词和词形还原的行为。在 spaCy 2.x 中产生`['do', "n't"]`的可能在 3.x 中产生`["don't"]`。你的模型是在一种分布上训练的。现在推理运行在另一种分布上。准确率悄悄下降，没有人知道为什么。在`requirements.txt`中固定库的版本。编写一个预处理回归测试，冻结 20 个样本句子的预期分词结果。在每次升级时运行它。

**训练/推理不匹配。** 训练时使用激进的预处理（小写化、停用词移除、词干提取），部署时在原始用户输入上运行，然后眼睁睁看着性能崩塌。这是最常见的生产 NLP 失败模式。如果你在训练期间进行预处理，那么你必须在推理期间运行完全相同的函数。将预处理作为模型包内部的函数发布，而不是作为服务团队重写的笔记本单元。

## 交付它

一个可复用的提示词，帮助工程师在不阅读三本教科书的情况下选择预处理策略。

保存为 `outputs/prompt-preprocessing-advisor.md`：

```markdown
---
name: preprocessing-advisor
description: 为 NLP 任务推荐分词、词干提取和词形还原设置。
phase: 5
lesson: 01
---

你为经典 NLP 预处理提供建议。给定任务描述，你输出：

1. 分词选择（正则表达式、NLTK word_tokenize、spaCy 或 transformer tokenizer）。解释原因。
2. 是否进行词干提取、词形还原、两者都做或两者都不做。解释原因。
3. 具体的库调用。命名函数。如果涉及 NLTK，引用 POS 标签转换代码。
4. 一个用户应该测试的失败模式。

拒绝为用户可见的文本推荐词干提取。拒绝在没有 POS 标签的情况下推荐词形还原。标记非英语输入需要不同的流程。
```

## 练习

1. **简单。** 扩展`tokenize`以将 URL 保留为单个 token。测试：`tokenize("Visit https://example.com today.")`应产生一个 URL token。
2. **中等。** 实现 Porter 步骤 1b。如果一个词包含元音且以`ed`或`ing`结尾，则去除它。处理双辅音规则（`hopping -> hop`，而不是`hopp`）。
3. **困难。** 构建一个使用 WordNet 作为查找表的词形还原器，但当 WordNet 没有条目时回退到你的 Porter 词干提取器。在标注语料库上测量准确率，与纯 WordNet 和纯 Porter 对比。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| Token | 一个词 | 模型消费的任意单位。可以是词、子词、字符或字节。 |
| 词干 | 词的词根 | 基于规则的后缀剥离结果。不总是一个真实的词。 |
| 词形 | 词典形式 | 你会查到的形式。需要语法上下文才能正确计算。 |
| POS 标签 | 词性 | 诸如 NOUN、VERB、ADJ 等类别。准确进行词形还原所必需。 |
| 形态学 | 词形变化规则 | 一个词如何根据时态、数、格改变形式。词形还原依赖于它。 |

## 扩展阅读

- [Porter, M. F. (1980). An algorithm for suffix stripping](https://tartarus.org/martin/PorterStemmer/def.txt) — 原始论文，五页，仍然是最清晰的解释。
- [spaCy 101 — linguistic features](https://spacy.io/usage/linguistic-features) — 真实流程是如何连接的。
- [NLTK book, chapter 3](https://www.nltk.org/book/ch03.html) — 你还没想到的分词边缘情况。
