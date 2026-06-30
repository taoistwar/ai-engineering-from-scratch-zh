# Transformer 之前的文本生成 — N-gram 语言模型

> 如果一个词令人惊讶，模型就糟糕。困惑度将惊讶量化。平滑使它保持有限。

**类型：** 构建
**语言：** Python
**先修要求：** 第五阶段 · 01（文本处理），第二阶段 · 14（朴素贝叶斯）
**预计时间：** 约45分钟

## 问题

在 transformer 之前，在 RNN 之前，在词嵌入之前，一个语言模型通过统计一个词跟随前`n-1`个词的频率来预测下一个词。统计"the cat" → "sat" 47 次，"the cat" → "jumped" 12 次，"the cat" → "refrigerator" 0 次。归一化得到概率分布。

这就是 n-gram 语言模型。它从 1980 年到 2015 年运行了每个语音识别器、每个拼写检查器和每个基于短语的机器翻译系统。当你在设备上需要廉价的语言建模时，它仍然在运行。

有趣的问题是如何处理未见过的 n-gram。一个基于原始计数的模型将零概率分配给任何它没见过的东西，这是灾难性的，因为句子很长，而几乎每个长句子包含至少一个未见过的序列。五十年的平滑研究解决了这个问题。Kneser-Ney 平滑是其成果，现代深度学习继承了其经验传统。

## 概念

![N-gram model: count, smooth, generate](../assets/ngram.svg)

**N-gram 概率：** `P(w_i | w_{i-n+1}, ..., w_{i-1})`。固定`n`（通常三元组用 3，四元组用 4）。从计数计算：

```text
P(w | context) = count(context, w) / count(context)
```

**零计数问题。** 任何训练中未见过的 n-gram 获得零概率。2007 年一项关于 Brown 语料库的研究发现，即使是 4-gram 模型在训练中也未见过 30% 的保留 4-gram。不在任何真实文本上进行平滑就无法评估。

**平滑方法，按复杂度排序：**

1. **拉普拉斯（加一）。** 给每个计数加 1。简单，在罕见事件上很差。
2. **Good-Turing。** 根据频率的频率将概率质量从较高频事件重新分配给未见过事件。
3. **插值。** 用可调权重结合 n-gram、(n-1)-gram 等估计。
4. **回退。** 如果 n-gram 的计数为零，回退到 (n-1)-gram。Katz 回退将此归一化。
5. **绝对折现。** 从所有计数中减去固定折现`D`，重新分配给未见过的。
6. **Kneser-Ney。** 绝对折现加上对低阶模型的巧妙选择：使用*延续概率*（一个词出现在多少种上下文中）而不是原始频率。

Kneser-Ney 的洞察很深。"San Francisco"是一个常见的二元组。单元组"Francisco"主要出现在"San"之后。朴素的绝对折现给"Francisco"高单元组概率（因为计数高）。Kneser-Ney 注意到"Francisco"只出现在一种上下文中，从而降低其延续概率。结果：以"Francisco"结尾的新二元组获得适当的低概率。

**评估：困惑度。** 在保留测试集上每个词的平均负对数似然的指数。越低越好。困惑度为 100 意味着模型的困惑程度如同在 100 个词中均匀选择。

```text
perplexity = exp(- (1/N) * Σ log P(w_i | context_i))
```

```figure
ngram-backoff
```

## 构建它

### 步骤 1：三元组计数

```python
from collections import Counter, defaultdict


def train_ngram(corpus_tokens, n=3):
    ngrams = Counter()
    contexts = Counter()
    for sentence in corpus_tokens:
        padded = ["<s>"] * (n - 1) + sentence + ["</s>"]
        for i in range(len(padded) - n + 1):
            ctx = tuple(padded[i:i + n - 1])
            word = padded[i + n - 1]
            ngrams[ctx + (word,)] += 1
            contexts[ctx] += 1
    return ngrams, contexts


def raw_probability(ngrams, contexts, context, word):
    ctx = tuple(context)
    if contexts.get(ctx, 0) == 0:
        return 0.0
    return ngrams.get(ctx + (word,), 0) / contexts[ctx]
```

输入是已分词的句子列表。输出是 n-gram 计数和上下文计数。`<s>`和`</s>`是句子边界。

### 步骤 2：拉普拉斯平滑

```python
def laplace_probability(ngrams, contexts, vocab_size, context, word):
    ctx = tuple(context)
    numerator = ngrams.get(ctx + (word,), 0) + 1
    denominator = contexts.get(ctx, 0) + vocab_size
    return numerator / denominator
```

给每个计数加 1。平滑但对未见过事件过度分配概率质量，也伤害了罕见已知事件。

### 步骤 3：Kneser-Ney（二元组，插值）

```python
def kneser_ney_bigram_model(corpus_tokens, discount=0.75):
    unigrams = Counter()
    bigrams = Counter()
    unigram_contexts = defaultdict(set)

    for sentence in corpus_tokens:
        padded = ["<s>"] + sentence + ["</s>"]
        for i, w in enumerate(padded):
            unigrams[w] += 1
            if i > 0:
                prev = padded[i - 1]
                bigrams[(prev, w)] += 1
                unigram_contexts[w].add(prev)

    total_unique_bigrams = sum(len(ctx_set) for ctx_set in unigram_contexts.values())
    continuation_prob = {
        w: len(ctx_set) / total_unique_bigrams for w, ctx_set in unigram_contexts.items()
    }

    context_totals = Counter()
    for (prev, w), count in bigrams.items():
        context_totals[prev] += count

    unique_follow = defaultdict(set)
    for (prev, w) in bigrams:
        unique_follow[prev].add(w)

    def prob(prev, w):
        count = bigrams.get((prev, w), 0)
        denom = context_totals.get(prev, 0)
        if denom == 0:
            return continuation_prob.get(w, 1e-9)
        first_term = max(count - discount, 0) / denom
        lambda_prev = discount * len(unique_follow[prev]) / denom
        return first_term + lambda_prev * continuation_prob.get(w, 1e-9)

    return prob
```

三个活动部分。`continuation_prob`捕获"这个词出现在多少种不同上下文中？"（Kneser-Ney 的创新）。`lambda_prev`是折现释放的质量，用于加权回退。最终概率是折现的主项加上加权的延续项。

### 步骤 4：使用采样生成文本

```python
import random


def generate(prob_fn, vocab, prefix, max_len=30, seed=0):
    rng = random.Random(seed)
    tokens = list(prefix)
    for _ in range(max_len):
        candidates = [(w, prob_fn(tokens[-1], w)) for w in vocab]
        total = sum(p for _, p in candidates)
        r = rng.random() * total
        acc = 0.0
        for w, p in candidates:
            acc += p
            if r <= acc:
                tokens.append(w)
                break
        if tokens[-1] == "</s>":
            break
    return tokens
```

按概率比例采样。每个种子总是给出不同的输出。对于类似束搜索的输出，在每一步选择 argmax（贪心）并添加小的随机性旋钮（温度）。

### 步骤 5：困惑度

```python
import math


def perplexity(prob_fn, sentences):
    total_log_prob = 0.0
    total_tokens = 0
    for sentence in sentences:
        padded = ["<s>"] + sentence + ["</s>"]
        for i in range(1, len(padded)):
            p = prob_fn(padded[i - 1], padded[i])
            total_log_prob += math.log(max(p, 1e-12))
            total_tokens += 1
    return math.exp(-total_log_prob / total_tokens)
```

越低越好。对于 Brown 语料库，一个调优良好的 4-gram KN 模型命中约 140 的困惑度。一个 transformer LM 在相同测试集上命中 15-30。差距大约是 10 倍。该差距就是该领域继续前进的原因。

## 使用它

- **经典 NLP 教学。** 你能得到的最清晰的平滑、MLE 和困惑度暴露。
- **KenLM。** 生产 n-gram 库。用于语音和 MT 系统中低延迟重要的重评分器。
- **设备端自动补全。** 键盘中的三元组模型。至今如此。
- **基线。** 在宣称你的神经 LM 良好之前，始终计算一个 n-gram LM 困惑度。如果你的 transformer 不是以大幅优势击败 KN，则有什么地方出了问题。

## 交付它

保存为 `outputs/prompt-lm-baseline.md`：

```markdown
---
name: lm-baseline
description: 在训练神经 LM 之前构建可复现的 n-gram 语言模型基线。
phase: 5
lesson: 16
---

给定一个语料库和目标用途（下一个词预测、重评分、困惑度基线），输出：

1. N-gram 阶数。通用英语用三元组，语料库大用四元组，语音重评分用五元组。
2. 平滑。修改的 Kneser-Ney 是默认；拉普拉斯仅用于教学。
3. 库。生产用`kenlm`，教学用`nltk.lm`，仅为了学习才自己实现。
4. 评估。在训练集和测试集之间具有一致 token 化的保留困惑度。

拒绝报告在用于比较的系统之间使用不同 token 化计算的困惑度——困惑度数字仅在相同 token 化下可比较。标记测试集中的 OOV 率；KN 处理 OOV 很差，除非在训练期间保留特殊的 <UNK> token。
```

## 练习

1. **简单。** 在 1,000 句莎十比亚语料库上训练三元组 LM。生成 20 个句子。它们在局部是合理但在全局是不连贯的。这是经典的演示。
2. **中等。** 在保留的莎十比亚分割上为你的 KN 模型实现困惑度。与拉普拉斯比较。你应该看到 KN 的困惑度低 30-50%。
3. **困难。** 构建一个三元组拼写校正器：给定一个拼写错误的词及其上下文，生成修正并就 LM 下的上下文概率排序。在公开的 Birkbeck 拼写语料库上评估。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| N-gram | 词序列 | `n`个连续 token 的序列。 |
| 平滑 | 避免零 | 重新分配概率质量，使未见过事件获得非零概率。 |
| 困惑度 | LM 质量指标 | 在保留数据上的`exp(-average log-prob)`。越低越好。 |
| 回退 | 回退到更短上下文 | 如果三元组计数为零，使用二元组。Katz 回退将其形式化。 |
| Kneser-Ney | n-gram 的最佳平滑 | 绝对折现 + 低阶模型的延续概率。 |
| 延续概率 | KN 特有 | 按`w`出现的上下文数而不是原始计数加权的`P(w)`。 |

## 扩展阅读

- [Jurafsky and Martin — Speech and Language Processing, Chapter 3 (2026 draft)](https://web.stanford.edu/~jurafsky/slp3/3.pdf) — n-gram LM 和平滑的规范处理。
- [Chen and Goodman (1998). An Empirical Study of Smoothing Techniques for Language Modeling](https://dash.harvard.edu/handle/1/25104739) — 将 Kneser-Ney 确定为最佳 n-gram 平滑器的论文。
- [Kneser and Ney (1995). Improved Backing-off for M-gram Language Modeling](https://ieeexplore.ieee.org/document/479394) — 原始的 KN 论文。
- [KenLM](https://kheafield.com/code/kenlm/) — 快速生产 n-gram LM，2026 年仍在延迟敏感应用中使用。
