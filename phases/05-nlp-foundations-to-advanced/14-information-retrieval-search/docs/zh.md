# 信息检索与搜索

> BM25 精确但脆弱。稠密方法覆盖面广但遗漏关键词。混合是 2026 年的默认方案。其他一切是调优。

**类型：** 构建
**语言：** Python
**先修要求：** 第五阶段 · 02（BoW + TF-IDF），第五阶段 · 04（GloVe、FastText、子词）
**预计时间：** 约75分钟

## 问题

用户输入"what happens if someone lies to get money"期望找到实际涵盖此内容的法律条文："Section 420 IPC。"关键词搜索完全遗漏它（没有共享词汇）。语义搜索如果嵌入不是在法律文本上训练的也会遗漏它。真正的搜索必须处理两者。

IR 是每个 RAG 系统、每个搜索栏、每个文档站点模糊查找底下的流程。2026 年在生产中有效的架构不是单一方法。它是互补方法的链条，每种方法捕获前一种方法的失败。

本课构建每一个部分并命名每种方法捕获哪些失败。

## 概念

![Hybrid retrieval: BM25 + dense + RRF + cross-encoder rerank](../assets/retrieval.svg)

四层。选择你需要的那些。

1. **稀疏检索（BM25）。** 快速，在精确匹配上精确，在语义上很差。在倒排索引上运行。在数百万文档上每查询不到 10 毫秒。正确处理法规引用、产品代码、错误消息、命名实体。
2. **稠密检索。** 将查询和文档编码为向量。最近邻搜索。捕获释义和语义相似度。遗漏相差一个字符的精确关键词匹配。使用 FAISS 或向量数据库每查询 50-200 毫秒。
3. **融合。** 合并来自稀疏和稠密的排序列表。倒数排序融合（RRF）是简单的默认方案，因为它忽略原始分数（它们在不同的尺度上）只使用排名位置。当你知道一种信号对你的领域占主导时，加权融合是一个选项。
4. **交叉编码器重排序。** 从融合中取 top-30。运行交叉编码器（查询 + 文档一起，对每一对评分）。保留 top-5。交叉编码器每对比双编码器更慢但准确得多。你通过仅在 top-30 上运行它们来分摊成本。

三元检索（BM25 + 稠密 + 学习稀疏如 SPLADE）在 2026 年基准测试中优于二元，但需要为学习稀疏索引提供基础设施。对于大多数团队，二元加交叉编码器重排序是最佳平衡点。

## 构建它

### 步骤 1：从零实现 BM25

```python
import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        if not corpus:
            raise ValueError("corpus must not be empty")
        self.corpus = [tokenize(d) for d in corpus]
        self.k1 = k1
        self.b = b
        self.n_docs = len(self.corpus)
        self.avg_dl = sum(len(d) for d in self.corpus) / self.n_docs
        self.df = Counter()
        for doc in self.corpus:
            for term in set(doc):
                self.df[term] += 1

    def idf(self, term):
        n = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))

    def score(self, query, doc_idx):
        q_tokens = tokenize(query)
        doc = self.corpus[doc_idx]
        dl = len(doc)
        freq = Counter(doc)
        score = 0.0
        for term in q_tokens:
            f = freq.get(term, 0)
            if f == 0:
                continue
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            score += self.idf(term) * numerator / denominator
        return score

    def rank(self, query, top_k=10):
        scored = [(self.score(query, i), i) for i in range(self.n_docs)]
        scored.sort(reverse=True)
        return scored[:top_k]
```

两个值得了解的参数。`k1=1.5`控制词频饱和；越高意味着对词重复给予更多权重。`b=0.75`控制长度归一化；0 忽略文档长度，1 完全归一化。默认值是 Robertson 在原始论文中的建议，很少需要调整。

### 步骤 2：使用双编码器的稠密检索

```python
from sentence_transformers import SentenceTransformer
import numpy as np


def build_dense_index(corpus, model_id="sentence-transformers/all-MiniLM-L6-v2"):
    encoder = SentenceTransformer(model_id)
    embeddings = encoder.encode(corpus, normalize_embeddings=True)
    return encoder, embeddings


def dense_search(encoder, embeddings, query, top_k=10):
    q_emb = encoder.encode([query], normalize_embeddings=True)
    sims = (embeddings @ q_emb.T).flatten()
    order = np.argsort(-sims)[:top_k]
    return [(float(sims[i]), int(i)) for i in order]
```

L2 归一化嵌入，使点积等于余弦相似度。`all-MiniLM-L6-v2`是 384 维，快速，对大多数英语检索足够强大。对于多语言工作，使用`paraphrase-multilingual-MiniLM-L12-v2`。对于最佳准确率，`bge-large-en-v1.5`或`e5-large-v2`。

### 步骤 3：倒数排序融合

```python
def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, (_, doc_idx) in enumerate(ranking):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(score, doc_idx) for doc_idx, score in fused]
```

`k=60`常量来自原始 RRF 论文。更高的`k`使排名差异的贡献趋于平坦；更低的`k`使顶部排名占主导。60 是已发布的默认值，很少需要调整。

### 步骤 4：混合搜索 + 重排序

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def hybrid_search(query, bm25, encoder, dense_embeddings, corpus, top_k=5, pool_size=30, reranker=reranker):
    sparse_ranking = bm25.rank(query, top_k=pool_size)
    dense_ranking = dense_search(encoder, dense_embeddings, query, top_k=pool_size)
    fused = reciprocal_rank_fusion([sparse_ranking, dense_ranking])[:pool_size]

    pairs = [(query, corpus[doc_idx]) for _, doc_idx in fused]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(scores, [doc_idx for _, doc_idx in fused]), reverse=True)
    return reranked[:top_k]
```

三个阶段组合。BM25 找到词汇匹配。稠密找到语义匹配。RRF 将两个排序合并，无需分数校准。交叉编码器使用查询-文档对一起重新评分 top-30，捕获双编码器遗漏的细粒度相关性。保留 top-5。

### 步骤 5：评估

| 指标 | 含义 |
|--------|---------|
| Recall@k | 在正确文档存在的查询中，它出现在 top-k 中的频率？ |
| MRR（平均倒数排名） | 第一个相关文档的排名倒数的平均值。 |
| nDCG@k | 考虑相关性梯度，而不仅仅是二元相关/不相关。 |

对于 RAG 特别重要的是，检索器的**Recall@k**是最重要的数字。如果正确的段落不在检索到的集合中，你的阅读器无法回答。

调试提示：对于失败的查询，比较稀疏和稠密的排序。如果一个找到了正确的文档而另一个没有，你有词汇不匹配（修复：添加缺失的那一半）或语义歧义（修复：更好的嵌入或重排序器）。

## 使用它

2026 年技术栈：

| 规模 | 技术栈 |
|-------|-------|
| 1k-100k 文档 | 内存内 BM25 + `all-MiniLM-L6-v2`嵌入 + RRF。无需单独数据库。 |
| 100k-10M 文档 | 稠密用 FAISS 或 pgvector + BM25 用 Elasticsearch / OpenSearch。并行运行。 |
| 10M+ 文档 | Qdrant / Weaviate / Vespa / Milvus 带混合支持。在 top-30 上交叉编码器重排序。 |
| 最佳质量前沿 | 三元（BM25 + 稠密 + SPLADE）+ ColBERT 后交互重排序 |

无论你选择什么，为评估留出预算。在基准测试端到端 RAG 准确率之前，基准测试检索召回率。阅读器无法修复检索器遗漏的内容。

### 2026 年生产 RAG 的来之不易教训

- **80% 的 RAG 失败可追溯到摄取和分块，而不是模型。** 团队花数周更换 LLM 和调优提示，而检索在每三个查询中静默返回错误的上下文。首先修复分块。
- **分块策略比块大小更重要。** 固定大小的分割破坏表格、代码和嵌套标题。句子感知是默认；对于技术文档和产品手册，语义或基于 LLM 的分块值得投入。
- **父文档模式。** 检索小"子"块以提高精确率。当同一父节的多个子块出现时，换入父块以保留上下文。这始终在不重新训练的情况下提升答案质量。
- **k_rerank=3 通常是最优的。** 超过此值的每个额外块增加 token 成本和生成延迟而不提升答案质量。如果 k=8 对你仍然优于 k=3，重排序器表现不佳。
- **HyDE / 查询扩展。** 从查询生成一个假设性答案，嵌入它，检索。弥合简短问题和长文档之间的措辞差距。无需训练的免费精确率提升。
- **上下文预算低于 8K token。** 在该限制处持续命中意味着重排序器阈值太宽松。
- **对一切进行版本管理。** 提示、分块规则、嵌入模型、重排序器。任何漂移静默地破坏答案质量。在忠实度、上下文精确率和未回答问题率上的 CI 门控在用户看到之前阻止退化。
- **三元检索（BM25 + 稠密 + 学习稀疏如 SPLADE）在 2026 年基准测试中优于二元，** 特别是对于混合专有名词和语义的查询。当基础设施支持 SPLADE 索引时部署它。

根据 2026 年的行业测量，正确的检索设计减少了 70-90% 的幻觉。大多数 RAG 性能增益来自更好的检索，而不是模型微调。

## 交付它

保存为 `outputs/skill-retrieval-picker.md`：

```markdown
---
name: retrieval-picker
description: 为给定的语料库和查询模式选择检索技术栈。
version: 1.0.0
phase: 5
lesson: 14
tags: [nlp, retrieval, rag, search]
---

给定需求（语料库大小、查询模式、延迟预算、质量门槛、基础设施约束），输出：

1. 技术栈。仅 BM25、仅稠密、混合（BM25 + 稠密 + RRF）、混合 + 交叉编码器重排序，或三元（BM25 + 稠密 + 学习稀疏）。
2. 稠密编码器。命名具体模型。匹配到语言、领域和上下文长度。
3. 重排序器。如果使用，命名具体的交叉编码器模型。标记重排序在 top-30 上增加 30-100 毫秒延迟。
4. 评估计划。Recall@10 是主要检索器指标。多答案用 MRR。首先建立基线，相对于基线测量增量改进。

拒绝为带有命名实体、错误代码或产品 SKU 的语料库推荐仅稠密方案，除非用户有证据表明稠密能处理精确匹配。拒绝为高风险检索（法律、医学）跳过重排序，这些情况下最终的 top-5 决定了用户的答案。
```

## 练习

1. **简单。** 在 500 文档语料库上实现上述`hybrid_search`。测试 20 个查询。比较仅 BM25、仅稠密和混合在 recall at 5 上的表现。
2. **中等。** 添加 MRR 计算。对于每个有已知正确文档的测试查询，找到正确文档在 BM25、稠密和混合排序中的排名。报告每种方法的 MRR。
3. **困难。** 使用 MultipleNegativesRankingLoss（Sentence Transformers）在你的领域上微调一个稠密编码器。从 500 个查询-文档对构建训练集。比较微调前后的召回率。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| BM25 | 关键词搜索 | Okapi BM25。按词频、IDF 和长度对文档评分。 |
| 稠密检索 | 向量搜索 | 将查询 + 文档编码为向量，找到最近邻。 |
| 双编码器 | 嵌入模型 | 独立编码查询和文档。查询时快速。 |
| 交叉编码器 | 重排序模型 | 一起编码查询 + 文档。慢但准确。 |
| RRF | 排名融合 | 通过求和`1/(k + rank)`合并两个排序。 |
| Recall@k | 检索指标 | 相关文档在 top-k 中的查询占比。 |

## 扩展阅读

- [Robertson and Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) — 权威的 BM25 处理。
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — DPR，规范的双编码器。
- [Formal et al. (2021). SPLADE: Sparse Lexical and Expansion Model](https://arxiv.org/abs/2107.05720) — 缩小与稠密方法差距的学习稀疏检索器。
- [Cormack, Clarke, Büttcher (2009). Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — RRF 论文。
- [Khattab and Zaharia (2020). ColBERT: Efficient and Effective Passage Search](https://arxiv.org/abs/2004.12832) — 后交互检索。
