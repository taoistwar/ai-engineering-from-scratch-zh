# GloVe、FastText 与子词嵌入

> Word2Vec 为每个词训练一个嵌入。GloVe 分解了共现矩阵。FastText 嵌入了各个部分。BPE 建立了通往 transformer 的桥梁。

**类型：** 构建
**语言：** Python
**先修要求：** 第五阶段 · 03（从零实现 Word2Vec）
**预计时间：** 约45分钟

## 问题

Word2Vec 留下了两个未解答的问题。

首先，有一条平行研究路线直接分解共现矩阵（LSA、HAL），而不是进行在线 skip-gram 更新。Word2Vec 的迭代方法在根本上更优吗，还是差异仅仅是两种方法处理计数的不同方式的产物？**GloVe** 回答了这个问题：使用深思熟虑选择的损失函数进行矩阵分解，其效果匹配或优于 Word2Vec，且训练成本更低。

其次，这两种方法都无法处理从未见过的词。`Zoomer-approved`、`dogecoin`、任何上周新造的名词、罕见词根的每种屈折形式。**FastText** 通过嵌入字符 n-gram 解决了这个问题：一个词是其各组成部分（包括词素）的和，因此即使是词表外的词也能得到一个合理的向量。

第三，一旦 transformer 出现，问题再次转变。词级词汇表最多约有一百万个条目；真实语言比这更开放。**字节对编码（BPE）** 及其同类通过学习覆盖一切的频繁子词单元词汇表解决了这个问题。每个现代 LLM 的每个现代 tokenizer 都是子词 tokenizer。

本课讲解这三者，然后解释何时选择哪种。

## 概念

**GloVe（全局向量）。** 构建词-词共现矩阵`X`，其中`X[i][j]`是词`j`在词`i`的上下文中出现的频率。训练向量使得`v_i · v_j + b_i + b_j ≈ log(X[i][j])`。对损失进行加权，使得频繁词对不占主导。完成。

**FastText。** 一个词是其字符 n-gram 加上词本身的和。`where`变为`<wh, whe, her, ere, re>, <where>`。词向量是那些组成向量的和。作为 Word2Vec 进行训练。好处：未见过的词（`whereupon`）从已知的 n-gram 组合出来。

**BPE（字节对编码）。** 从单个字节（或字符）的词汇表开始。统计语料库中每个相邻词对。将最频繁的词对合并为一个新的 token。重复`k`次迭代。结果：一个包含`k + 256`个 token 的词汇表，其中频繁序列（`ing`、`tion`、`the`）是单个 token，而罕见词被拆解为熟悉的片段。每个句子都能被 tokenize 成某些东西。

## 构建它

### GloVe：分解共现矩阵

```python
import numpy as np
from collections import Counter


def build_cooccurrence(docs, window=5):
    pair_counts = Counter()
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    for doc in docs:
        indexed = [vocab[t] for t in doc]
        for i, center in enumerate(indexed):
            for j in range(max(0, i - window), min(len(indexed), i + window + 1)):
                if i != j:
                    distance = abs(i - j)
                    pair_counts[(center, indexed[j])] += 1.0 / distance
    return vocab, pair_counts


def glove_train(vocab, pair_counts, dim=16, epochs=100, lr=0.05, x_max=100, alpha=0.75, seed=0):
    n = len(vocab)
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(n, dim))
    W_tilde = rng.normal(0, 0.1, size=(n, dim))
    b = np.zeros(n)
    b_tilde = np.zeros(n)

    for epoch in range(epochs):
        for (i, j), x_ij in pair_counts.items():
            weight = (x_ij / x_max) ** alpha if x_ij < x_max else 1.0
            diff = W[i] @ W_tilde[j] + b[i] + b_tilde[j] - np.log(x_ij)
            coef = weight * diff

            grad_W_i = coef * W_tilde[j]
            grad_W_tilde_j = coef * W[i]
            W[i] -= lr * grad_W_i
            W_tilde[j] -= lr * grad_W_tilde_j
            b[i] -= lr * coef
            b_tilde[j] -= lr * coef

    return W + W_tilde
```

有两个值得指出的关键点。加权函数`f(x) = (x/x_max)^alpha`降低非常频繁词对（如`(the, and)`）的权重，使它们不主导损失。最终嵌入是`W`（中心词）和`W_tilde`（上下文词）表的和。将两者求和是一个已发表技巧，通常优于仅使用其中之一。

### FastText：子词感知的嵌入

```python
def char_ngrams(word, n_min=3, n_max=6):
    wrapped = f"<{word}>"
    grams = {wrapped}
    for n in range(n_min, n_max + 1):
        for i in range(len(wrapped) - n + 1):
            grams.add(wrapped[i:i + n])
    return grams
```

```python
>>> char_ngrams("where")
{'<where>', '<wh', 'whe', 'her', 'ere', 're>', '<whe', 'wher', 'here', 'ere>', '<wher', 'where', 'here>'}
```

每个词由其 n-gram（通常为 3 到 6 个字符）集合表示。词嵌入是其 n-gram 嵌入的和。对于 skip-gram 训练，在 Word2Vec 使用单个向量的地方插入这个。

```python
def fasttext_vector(word, ngram_table):
    grams = char_ngrams(word)
    vecs = [ngram_table[g] for g in grams if g in ngram_table]
    if not vecs:
        return None
    return np.sum(vecs, axis=0)
```

对于一个未见过的词，只要它的一些 n-gram 是已知的，你仍然可以得到一个向量。`whereupon`与`where`共享`<wh`、`her`、`ere`和`<where`，因此两者会落在彼此附近。

### BPE：学习到的子词词汇表

```python
def learn_bpe(corpus, k_merges):
    vocab = Counter()
    for word, freq in corpus.items():
        tokens = tuple(word) + ("</w>",)
        vocab[tokens] = freq

    merges = []
    for _ in range(k_merges):
        pair_freq = Counter()
        for tokens, freq in vocab.items():
            for a, b in zip(tokens, tokens[1:]):
                pair_freq[(a, b)] += freq
        if not pair_freq:
            break
        best = pair_freq.most_common(1)[0][0]
        merges.append(best)

        new_vocab = Counter()
        for tokens, freq in vocab.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_vocab[tuple(new_tokens)] = freq
        vocab = new_vocab
    return merges


def apply_bpe(word, merges):
    tokens = list(word) + ["</w>"]
    for a, b in merges:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(a + b)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens
```

```python
>>> corpus = Counter({"low": 5, "lower": 2, "newest": 6, "widest": 3})
>>> merges = learn_bpe(corpus, k_merges=10)
>>> apply_bpe("lowest", merges)
['low', 'est</w>']
```

第一次迭代合并最常见的相邻词对。经过足够多次迭代后，频繁子串（`low`、`est`、`tion`）成为单个 token，罕见词则被干净地分解。

真正的 GPT / BERT / T5 tokenizer 学习 3 万到 10 万次合并。结果：任何文本都能被 tokenize 成一个有界长度的已知 ID 序列，永远不会有 OOV。

## 使用它

在实践中，你几乎从不需要自己训练这些。你加载预训练好的检查点。

```python
import fasttext.util
fasttext.util.download_model("en", if_exists="ignore")
ft = fasttext.load_model("cc.en.300.bin")
print(ft.get_word_vector("whereupon").shape)
print(ft.get_word_vector("zoomerapproved").shape)
```

对于 transformer 时代的 BPE 式子词 tokenization：

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
print(tok.tokenize("unbelievably tokenized"))
```

```
['un', 'bel', 'iev', 'ably', 'Ġtoken', 'ized']
```

`Ġ`前缀标记词边界（一个 GPT-2 的惯例）。每个现代 tokenizer 都是 BPE 变体、WordPiece（BERT）或 SentencePiece（T5、LLaMA）。

### 何时选择哪个

| 场景 | 选择 |
|-----------|------|
| 预训练的通用词向量，不需要 OOV 容忍 | GloVe 300d |
| 预训练的通用词向量，必须处理拼写错误 / 新词 / 形态丰富的语言 | FastText |
| 任何进入 transformer 的内容（训练或推理） | 模型自带的任何 tokenizer。永远不要换。 |
| 从零开始训练自己的语言模型 | 先在你的语料库上训练一个 BPE 或 SentencePiece tokenizer |
| 使用线性模型的生产文本分类 | 仍然是 TF-IDF。第 02 课。 |

## 交付它

保存为 `outputs/skill-embeddings-picker.md`：

```markdown
---
name: tokenizer-picker
description: 为新的语言模型或文本流程选择 tokenization 方法。
version: 1.0.0
phase: 5
lesson: 04
tags: [nlp, tokenization, embeddings]
---

给定一个任务和数据集描述，你输出：

1. Tokenization 策略（词级、BPE、WordPiece、SentencePiece、字节级）。一句理由。
2. 词汇表大小目标（例如，仅英语的 LM 32k，多语言 64k-100k）。
3. 带有确切训练命令的库调用。命名库。引用参数。
4. 一个可复现性陷阱。tokenizer-模型不匹配是最常见的沉默生产 bug；指出哪个对必须一起使用。

当用户正在微调预训练的 LLM 时，拒绝推荐训练自定义 tokenizer。拒绝为任何面向生产推理的模型推荐词级 tokenization。标记非英语/多文字语料库需要带字节回退的 SentencePiece。
```

## 练习

1. **简单。** 运行`char_ngrams("playing")`和`char_ngrams("played")`。计算两个 n-gram 集合的 Jaccard 重叠。你应该看到大量的共享片段（`pla`、`lay`、`play`），这就是为什么 FastText 能很好地跨形态变体迁移。
2. **中等。** 扩展`learn_bpe`以跟踪词汇表增长。绘制每个语料库字符的 token 数作为合并次数的函数。你应该看到最初快速压缩，渐近线在大约每 token 2-3 个字符处。
3. **困难。** 在莎士比亚的完整作品上训练一个 1k 次合并的 BPE。比较常见词与罕见专有名词的 tokenization。测量前后的每词平均 token 数。写下令你惊讶的地方。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| 共现矩阵 | 词-词频率表 | `X[i][j]` = 词`j`在词`i`周围的窗口中出现的频率。 |
| 子词 | 词的一部分 | 一个字符 n-gram（FastText）或学习到的 token（BPE/WordPiece/SentencePiece）。 |
| BPE | 字节对编码 | 迭代合并最频繁的相邻词对，直到词汇表达到目标大小。 |
| OOV | 词表外 | 模型从未见过的词。Word2Vec/GloVe 失败。FastText 和 BPE 可以处理。 |
| 字节级 BPE | 在原始字节上的 BPE | GPT-2 的方案。词汇表从 256 个字节开始，因此永远不会 OOV。 |

## 扩展阅读

- [Pennington, Socher, Manning (2014). GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf) — GloVe 论文，七页，仍然是对损失函数最好的推导。
- [Bojanowski et al. (2017). Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606) — FastText。
- [Sennrich, Haddow, Birch (2016). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — 将 BPE 引入现代 NLP 的论文。
- [Hugging Face tokenizer summary](https://huggingface.co/docs/transformers/tokenizer_summary) — BPE、WordPiece 和 SentencePiece 在实践中实际如何不同。
