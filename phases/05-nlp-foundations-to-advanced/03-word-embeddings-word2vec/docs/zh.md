# 词嵌入 — 从零实现 Word2Vec

> 一个词由它所处的环境定义。在这个思想上训练一个浅层网络，几何结构就会自然显现。

**类型：** 构建
**语言：** Python
**先修要求：** 第五阶段 · 02（BoW + TF-IDF），第三阶段 · 03（从零实现反向传播）
**预计时间：** 约75分钟

## 问题

TF-IDF 知道`dog`和`puppy`是不同的词。它不知道它们的意思几乎相同。一个在`dog`上训练的分类器无法泛化到关于`puppy`的评价。你可以通过列举同义词来弥补，但这对罕见词、领域术语以及你没有预料到的每种语言都会失败。

你希望有一种表示方法，让`dog`和`puppy`在空间中靠得很近。让`king - man + woman`接近`queen`。让在`dog`上训练的模型能够免费地将一些信号传递给`puppy`。

Word2Vec 给了我们那个空间。两层神经网络，万亿 token 级别的训练运行，于 2013 年发表。其架构简单得几乎令人尴尬。其结果重塑了 NLP 十年之久。

## 概念

**分布式假设**（Firth, 1957）："你可以通过一个词所处的环境来认识它。"如果两个词出现在相似的上下文中，它们可能表示相似的事物。

Word2Vec 有两种变体，都利用了那个思想。

- **Skip-gram。** 给定中心词，预测周围的词。`cat -> (the, sat, on)`，窗口大小为 2。
- **CBOW（连续词袋模型）。** 给定周围的词，预测中心词。`(the, sat, on) -> cat`。

Skip-gram 训练较慢，但能更好地处理罕见词。它成为了默认方案。

网络有一个隐藏层，没有非线性激活。输入是词汇表上的独热向量。输出是词汇表上的 softmax。训练完成后，你丢弃输出层。隐藏层的权重就是嵌入。

```
one-hot(center) ── W ──▶ hidden (d-dim) ── W' ──▶ softmax(vocab)
                          ^
                          这就是嵌入
```

技巧：10 万词汇上的 softmax 计算代价过高。Word2Vec 使用**负采样**将其转换为二分类任务。预测"这个上下文词是否出现在这个中心词附近，是或否"。为每个训练样本对采样少量负（不共现的）词，而不是对整个词汇表计算 softmax。

```figure
word-vector-arithmetic
```

## 构建它

### 步骤 1：从语料库生成训练样本对

```python
def skipgram_pairs(docs, window=2):
    pairs = []
    for doc in docs:
        for i, center in enumerate(doc):
            for j in range(max(0, i - window), min(len(doc), i + window + 1)):
                if i == j:
                    continue
                pairs.append((center, doc[j]))
    return pairs
```

```python
>>> skipgram_pairs([["the", "cat", "sat", "on", "mat"]], window=2)
[('the', 'cat'), ('the', 'sat'),
 ('cat', 'the'), ('cat', 'sat'), ('cat', 'on'),
 ('sat', 'the'), ('sat', 'cat'), ('sat', 'on'), ('sat', 'mat'),
 ...]
```

窗口内的每个 (中心词, 上下文词) 对都是一个正训练样本。

### 步骤 2：嵌入表

两个矩阵。`W` 是中心词嵌入表（你保留的那个）。`W'` 是上下文词表（通常被丢弃，有时与`W`取平均）。

```python
import numpy as np


def init_embeddings(vocab_size, dim, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(vocab_size, dim))
    W_prime = rng.normal(0, 0.1, size=(vocab_size, dim))
    return W, W_prime
```

小随机初始化。词汇表大小 1 万、维度 100 是现实的；用于教学，50 个词汇 x 16 维足以看到几何结构。

### 步骤 3：负采样目标函数

对于每个正样本对`(center, context)`，从词汇表中随机采样`k`个词作为负样本。训练模型使得正样本的点积`W[center] · W'[context]`高，负样本的点积低。

```python
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_pair(W, W_prime, center_idx, context_idx, negative_indices, lr):
    v_c = W[center_idx]
    u_pos = W_prime[context_idx]
    u_negs = W_prime[negative_indices]

    pos_score = sigmoid(v_c @ u_pos)
    neg_scores = sigmoid(u_negs @ v_c)

    grad_center = (pos_score - 1) * u_pos
    for i, u in enumerate(u_negs):
        grad_center += neg_scores[i] * u

    W[context_idx] = W[context_idx]
    W_prime[context_idx] -= lr * (pos_score - 1) * v_c
    for i, neg_idx in enumerate(negative_indices):
        W_prime[neg_idx] -= lr * neg_scores[i] * v_c
    W[center_idx] -= lr * grad_center
```

魔法公式：正样本对上的逻辑损失（希望 sigmoid 接近 1）加上负样本对上的逻辑损失（希望 sigmoid 接近 0）。梯度流向两个表。完整推导在原始论文中；如果想让它记住，用铅笔和纸手推一遍。

### 步骤 4：在小型语料库上训练

```python
def train(docs, dim=16, window=2, k_neg=5, epochs=100, lr=0.05, seed=0):
    vocab = build_vocab(docs)
    vocab_size = len(vocab)
    rng = np.random.default_rng(seed)
    W, W_prime = init_embeddings(vocab_size, dim, seed=seed)
    pairs = skipgram_pairs(docs, window=window)

    for epoch in range(epochs):
        rng.shuffle(pairs)
        for center, context in pairs:
            c_idx = vocab[center]
            ctx_idx = vocab[context]
            negs = rng.integers(0, vocab_size, size=k_neg)
            negs = [n for n in negs if n != ctx_idx and n != c_idx]
            train_pair(W, W_prime, c_idx, ctx_idx, negs, lr)
    return vocab, W
```

在大型语料库上经过足够多的 epoch 后，共享上下文的词具有相似的中心嵌入。在一个小型语料库上，你会隐隐约约看到这个效果。在数十亿 token 上，你会显著地看到它。

### 步骤 5：类比技巧

```python
def nearest(vocab, W, target_vec, topk=5, exclude=None):
    exclude = exclude or set()
    inv_vocab = {i: w for w, i in vocab.items()}
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-9
    W_norm = W / norms
    target = target_vec / (np.linalg.norm(target_vec) + 1e-9)
    sims = W_norm @ target
    order = np.argsort(-sims)
    out = []
    for i in order:
        if i in exclude:
            continue
        out.append((inv_vocab[i], float(sims[i])))
        if len(out) == topk:
            break
    return out


def analogy(vocab, W, a, b, c, topk=5):
    v = W[vocab[b]] - W[vocab[a]] + W[vocab[c]]
    return nearest(vocab, W, v, topk=topk, exclude={vocab[a], vocab[b], vocab[c]})
```

在预训练的 300d Google News 向量上：

```python
>>> analogy(vocab, W, "man", "king", "woman")
[('queen', 0.71), ('monarch', 0.62), ('princess', 0.59), ...]
```

`king - man + woman = queen`。不是因为模型知道皇室是什么。而是因为向量`(king - man)`捕捉了类似"皇室"的东西，将其加到`woman`上就落在了皇族-女性区域附近。

## 使用它

从零写 Word2Vec 是教学用途。生产 NLP 使用`gensim`。

```python
from gensim.models import Word2Vec

sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "ran", "across", "the", "room"],
]

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    negative=5,
    workers=4,
    epochs=30,
)

print(model.wv["cat"])
print(model.wv.most_similar("cat", topn=3))
```

对于实际工作，你几乎从不需要自己训练 Word2Vec。你下载预训练好的向量。

- **GloVe** — 斯坦福的共现矩阵分解方法。有 50d、100d、200d、300d 检查点。良好的通用覆盖。第 04 课专门介绍 GloVe。
- **fastText** — Facebook 的 Word2Vec 扩展，嵌入字符 n-gram。通过组合子词来处理词表外词汇。第 04 课。
- **Google News 上的预训练 Word2Vec** — 300d，300 万词汇表，2013 年发布。至今每天仍在被下载。

### Word2Vec 在 2026 年仍然胜出的场景

- 轻量级领域特定检索。在笔记本电脑上用一小时在医学摘要上训练，获得任何通用模型都无法捕获的专门化向量。
- 类比式特征工程。`gender_vector = mean(man - woman pairs)`。从其他词中减去它以获得性别中立的轴。仍在公平性研究中使用。
- 可解释性。100d 足够小，可以通过 PCA 或 t-SNE 绘图并实际看到聚类的形成。
- 任何推理需要在无 GPU 设备上运行的场景。Word2Vec 查找是单行取值操作。

### Word2Vec 失败的地方

多义性之墙。`bank`只有一个向量。`river bank`和`financial bank`共享它。`table`（电子表格 vs. 家具）共享它。下游分类器无法从向量中区分这些含义。

上下文嵌入（ELMo、BERT，以及之后的所有 transformer）通过基于周围上下文为每个词出现生成不同向量解决了这个问题。这就是从 Word2Vec 到 BERT 的飞跃：从静态到上下文相关。第七阶段涵盖 transformer 部分。

词表外问题是另一个失败。如果`Zoomer-approved`不在训练数据中，Word2Vec 从未见过它。没有回退方案。fastText 通过子词组合（第 04 课）解决了这个问题。

## 交付它

保存为 `outputs/skill-embedding-probe.md`：

```markdown
---
name: embedding-probe
description: 检查一个 word2vec 模型。运行类比测试、查找近邻、诊断质量。
version: 1.0.0
phase: 5
lesson: 03
tags: [nlp, embeddings, debugging]
---

你探查训练好的词嵌入以验证它们正在工作。给定一个`gensim.models.KeyedVectors`对象和一个词汇表，你运行：

1. 三个规范的类比测试。`king : man :: queen : woman`。`paris : france :: tokyo : japan`。`walking : walked :: swimming : ?`。报告 top-1 结果及其余弦值。
2. 针对用户提供的领域特定词，进行五个最近邻测试。打印 top-5 近邻及余弦值。
3. 一个对称性检查。在浮点精度内`similarity(a, b) == similarity(b, a)`。
4. 一个退化检查。如果任何嵌入的范数低于 0.01 或高于 100，模型存在训练 bug。标记它。

拒绝仅凭类比准确率就声明模型良好。类比基准是可被钻空子的，并且不会迁移到下游任务。推荐将内在评估和下游评估结合起来。
```

## 练习

1. **简单。** 在一个小型语料库（关于猫和狗的 20 个句子）上运行训练循环。200 个 epoch 后，验证`nearest(vocab, W, W[vocab["cat"]])`在其 top 3 中返回`dog`。如果没有，增加 epoch 或词汇表。
2. **中等。** 添加高频词的子采样。频率高于`10^-5`的词以与其频率成比例的概率从训练样本对中丢弃。衡量对罕见词相似度的影响。
3. **困难。** 在 20 Newsgroups 语料库上训练一个模型。计算两个偏置轴：`he - she`和`doctor - nurse`。将职业词投影到两个轴上。报告哪些职业具有最大的偏置差距。这就是公平性研究人员使用的那类探查。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| 词嵌入 | 词作为向量 | 一种从上下文中学习到的稠密、低维（通常 100-300）表示。 |
| Skip-gram | Word2Vec 技巧 | 从中心词预测上下文词。比 CBOW 慢，对罕见词更好。 |
| 负采样 | 训练捷径 | 用针对`k`个随机词的二分类替代全词汇表 softmax。 |
| 静态嵌入 | 每个词一个向量 | 无论上下文如何都相同的向量。对多义性失败。 |
| 上下文嵌入 | 上下文敏感的向量 | 根据周围词为每个词出现生成的不同向量。Transformer 产生的。 |
| OOV | 词表外 | 训练中未见过的词。Word2Vec 不能为这些词生成向量。 |

## 扩展阅读

- [Mikolov et al. (2013). Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) — 负采样论文。简短易读。
- [Rong, X. (2014). word2vec Parameter Learning Explained](https://arxiv.org/abs/1411.2738) — 梯度最清晰的推导，如果原始论文的数学感觉难懂的话。
- [gensim Word2Vec tutorial](https://radimrehurek.com/gensim/models/word2vec.html) — 实际有效的生产训练设置。
