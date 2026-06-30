# 嵌入模型 — 2026 深度剖析

> Word2Vec 给你每个词一个向量。现代嵌入模型给你每个段落一个向量，跨语言，具备稀疏、稠密和多向量视图，大小适合你的索引。选错了，你的 RAG 检索到错误的东西。

**类型：** 学习
**语言：** Python
**先修要求：** 第五阶段 · 03（Word2Vec），第五阶段 · 14（信息检索）
**预计时间：** 约60分钟

## 问题

你的 RAG 系统 40% 的时间检索到错误的段落。罪魁祸首很少是向量数据库或提示。是嵌入模型。

2026 年选择嵌入意味着跨越五个轴做出选择：

1. **稠密 vs 稀疏 vs 多向量。** 每个段落一个向量，或每个 token 一个，或稀疏加权词袋。
2. **语言覆盖。** 单语言英语模型在仅英语任务上仍然胜出。当语料库混合时，多语言模型胜出。
3. **上下文长度。** 512 token vs 8,192 vs 32,768——而真实的实际容量通常是广告最大值的 60-70%。
4. **维度预算。** 全精度的 3,072 个 float = 每个向量 12 KB。在 1 亿向量时，存储费每月 $1,300。Matryoshka 截断将其削减 4 倍。
5. **开放 vs 托管。** 开放权重意味着你控制技术栈和数据。托管意味着你用控制权换取始终最新。

本课指明权衡，使你能基于证据选择，而不是根据上个季度流行的任何东西。

## 概念

![Dense, sparse, and multi-vector embeddings](../assets/embedding-modes.svg)

**稠密嵌入。** 每个段落一个向量（通常 384-3,072 维）。余弦相似度按语义接近度排序段落。OpenAI `text-embedding-3-large`、BGE-M3 dense mode、Voyage-3。默认选择。

**稀疏嵌入。** SPLADE 风格。一个 transformer 为每个词汇表 token 预测权重，然后将大部分清零。结果是大小为 |vocab| 的稀疏向量。捕获词汇匹配（如 BM25），但具有学习到的词权重。在关键词重的查询上很强。

**多向量（后交互）。** ColBERTv2、Jina-ColBERT。每个 token 一个向量。用 MaxSim 评分：对于每个查询 token，找到最相似的文档 token，对分数求和。存储和评分更昂贵，但在长查询和领域特定语料库上胜出。

**BGE-M3：三者合一。** 单个模型同时输出稠密、稀疏和多向量表示。每种表示可以独立查询；分数通过加权和融合。2026 年当你希望从一个检查点获得灵活性的默认选择。

**Matryoshka 表示学习。** 训练使得向量的前 N 维形成一个有用的独立嵌入。将 1,536 维向量截断为 256 维，以约 1% 准确率为代价获得 6 倍存储节省。OpenAI text-3、Cohere v4、Voyage-4、Jina v5、Gemini Embedding 2、Nomic v1.5+ 支持。

### MTEB 排行榜讲了一个不完整的故事

大规模文本嵌入基准测试——启动时（2022）跨 8 种任务类型的 56 个任务，在 MTEB v2 中扩展到 100+ 任务。在 2026 年初，Gemini Embedding 2 在检索上领先（67.71 MTEB-R）。Cohere embed-v4 在通用上领先（65.2 MTEB）。BGE-M3 在开放权重的多语言上领先（63.0）。排行榜是必要的但不充分——始终在你的领域上基准测试。

### 三层模式

| 用例 | 模式 |
|----------|---------|
| 快速首遍 | 稠密双编码器（BGE-M3、text-3-small） |
| 召回提升 | 稀疏（SPLADE、BGE-M3 sparse）+ RRF 融合 |
| Top-50 上的精确率 | 多向量（ColBERTv2）或交叉编码器重排序器 |

大多数生产技术栈使用全部三层。

## 构建它

### 步骤 1：基线——用 Sentence-BERT 获取稠密嵌入

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
corpus = [
    "The first iPhone launched in 2007.",
    "Apple released the iPod in 2001.",
    "Android is an operating system from Google.",
]
emb = encoder.encode(corpus, normalize_embeddings=True)

query = "When was the iPhone released?"
q_emb = encoder.encode([query], normalize_embeddings=True)[0]
scores = emb @ q_emb
print(sorted(enumerate(scores), key=lambda x: -x[1]))
```

`normalize_embeddings=True`使点积等于余弦相似度。始终设置它。

### 步骤 2：Matryoshka 截断

```python
def truncate(vectors, dim):
    out = vectors[:, :dim]
    return out / np.linalg.norm(out, axis=1, keepdims=True)

emb_256 = truncate(emb, 256)
emb_128 = truncate(emb, 128)
```

截断后重新归一化。Nomic v1.5、OpenAI text-3 和 Voyage-4 被训练使得这对前几个级别是无损的。非 Matryoshka 模型（原始 Sentence-BERT）在被截断时急剧退化。

### 步骤 3：BGE-M3 多功能性

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

output = model.encode(
    corpus,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True,
)
# output["dense_vecs"]:    (n_docs, 1024)
# output["lexical_weights"]: list of dict {token_id: weight}
# output["colbert_vecs"]:  list of (n_tokens, 1024) arrays
```

三个索引，一次推理调用。分数融合：

```python
dense_score = ... # 在 dense_vecs 上的余弦
sparse_score = model.compute_lexical_matching_score(q_lex, d_lex)
colbert_score = model.colbert_score(q_col, d_col)
final = 0.4 * dense_score + 0.2 * sparse_score + 0.4 * colbert_score
```

在你的领域上调优权重。

### 步骤 4：在自定义任务上的 MTEB 评估

```python
from mteb import MTEB

tasks = ["ArguAna", "SciFact", "NFCorpus"]
evaluation = MTEB(tasks=tasks)
results = evaluation.run(encoder, output_folder="./mteb-results")
```

在*代表性*子集上运行你的候选模型。不要仅信任排行榜排名——你的领域很重要。

### 步骤 5：从零手写余弦

参见`code/main.py`。平均哈希技巧嵌入（仅标准库）。不具有与 transformer 嵌入的竞争力，但展示了形态：tokenize → 向量 → 归一化 → 点积。

## 陷阱

- **查询和文档用相同的模型。** 某些模型（Voyage、Jina-ColBERT）使用非对称编码——查询和文档通过不同的路径。始终检查模型卡片。
- **缺失前缀。** `bge-*` 模型需要在查询前加上 `"Represent this sentence for searching relevant passages: "`。如果忘记，召回率差 3-5 分。
- **过度修剪 Matryoshka。** 1,536 → 256 通常是安全的。1,536 → 64 不安全。在你的评估集上验证。
- **上下文截断。** 大多数模型静默截断超过其最大长度的输入。长文档需要分块（见第 23 课）。
- **忽略延迟尾部。** MTEB 分数隐藏了 p99 延迟。一个 600M 模型可能以 2 分优势击败 335M 模型，但每个查询成本高 3 倍。

## 使用它

2026 年技术栈：

| 场景 | 选择 |
|-----------|------|
| 仅英语、快速、API | `text-embedding-3-large`或`voyage-3-large` |
| 开放权重、英语 | `BAAI/bge-large-en-v1.5` |
| 开放权重、多语言 | `BAAI/bge-m3`或`Qwen3-Embedding-8B` |
| 长上下文（32k+） | Voyage-3-large、Cohere embed-v4、Qwen3-Embedding-8B |
| 仅 CPU 部署 | Nomic Embed v2（137M 参数、MoE） |
| 存储受限 | Matryoshka 截断 + int8 量化 |
| 关键词重的查询 | 添加 SPLADE 稀疏、RRF 与稠密融合 |

2026 年模式：从 BGE-M3 或 text-3-large 开始，用 MTEB 在你的领域上评估，如果领域特定模型以超过 3 分优势胜出则替换。

## 交付它

保存为 `outputs/skill-embedding-picker.md`：

```markdown
---
name: embedding-picker
description: 为给定的语料库和部署选择嵌入模型、维度和检索模式。
version: 1.0.0
phase: 5
lesson: 22
tags: [nlp, embeddings, retrieval]
---

给定一个语料库（大小、语言、领域、平均长度）、部署目标（云 / 边缘 / 本地）、延迟预算和存储预算，输出：

1. 模型。命名的检查点或 API。一句理由。
2. 维度。全尺寸 / Matryoshka 截断 / int8 量化。与存储预算相关联的理由。
3. 模式。稠密 / 稀疏 / 多向量 / 混合。理由。
4. 如果模型卡片要求查询前缀 / 模板。
5. 评估计划。与领域相关的 MTEB 任务 + 带 nDCG@10 的保留领域评估。

拒绝在没有领域验证的情况下将 Matryoshka 截断到 <64 维的建议。拒绝在 10k 段落以下的语料库上使用 ColBERTv2（开销不合理）。标记路由到 512 token 窗口模型的长文档语料库（>8k token）。
```

## 练习

1. **简单。** 用`bge-small-en-v1.5`在全维（384）上编码 100 个句子，然后在 Matryoshka 128 上。在 10 个查询上测量 MRR 下降。
2. **中等。** 在你领域的 500 个段落上比较 BGE-M3 稠密、稀疏和 colbert。在 recall@10 上哪种胜出？RRF 融合是否击败了最佳单一模式？
3. **困难。** 在你的 top-2 个领域任务上对三个候选模型运行 MTEB。报告 MTEB 分数、100 查询批次上的 p99 延迟以及每百万查询成本。选择帕累托最优的那个。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| 稠密嵌入 | 向量 | 每个文本一个固定大小向量。余弦相似度排序。 |
| 稀疏嵌入 | 学习到的 BM25 | 每个词汇表 token 一个权重；大部分为零；端到端训练。 |
| 多向量 | ColBERT 风格 | 每个 token 一个向量；MaxSim 评分；更大的索引，更好的召回率。 |
| Matryoshka | 俄罗斯套娃技巧 | 前 N 维自身是一个有效的较小嵌入。 |
| MTEB | 基准测试 | 大规模文本嵌入基准——启动时 56 个任务，v2 中 100+。 |
| BEIR | 检索基准测试 | 18 个零样本检索任务；经常被引用用于跨领域鲁棒性。 |
| 非对称编码 | 查询 ≠ 文档路径 | 模型对查询和文档使用不同的投影。 |

## 扩展阅读

- [Reimers, Gurevych (2019). Sentence-BERT](https://arxiv.org/abs/1908.10084) — 双编码器论文。
- [Muennighoff et al. (2022). MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316) — 排行榜论文。
- [Chen et al. (2024). BGE-M3: Multi-lingual, Multi-functionality, Multi-granularity](https://arxiv.org/abs/2402.03216) — 统一三模式模型。
- [Kusupati et al. (2022). Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) — 维度阶梯训练目标。
- [Santhanam et al. (2022). ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://arxiv.org/abs/2112.01488) — 生产中的后交互。
- [MTEB leaderboard on Hugging Face](https://huggingface.co/spaces/mteb/leaderboard) — 实时排名。
