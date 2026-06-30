# 分块策略对比

> 分块决定了你的检索器是否能找到任何内容。如果边界错误，下游的任何嵌入模型、任何重排器、任何 LLM 都无法修复损坏。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 11 lessons 04 (embeddings), 06 (RAG), 07 (advanced RAG); Phase 19 Track B foundations (lessons 20-29)
**Time:** ~90 minutes

## Learning Objectives
- 从零实现五种分块策略：固定窗口、句子拆分、递归分割、语义聚类和结构化 markdown 标题。
- 在带有金标答案跨度的 fixture 语料库上测量 recall@k，并解释为什么一种策略在散文上胜出，另一种策略在技术文档上胜出。
- 阅读分块长度分布，识别每种策略注入的故障模式：孤儿句子、符号中间截断、仅有标题的分块、语义漂移。
- 通过检查三个属性为新语料库选择一个默认策略，而无需运行基准测试：文档类型、平均段落长度以及格式是否携带显式结构。

## The Problem

每个 RAG 管线都从将源文档切割为足够小以便嵌入模型容纳、足够大以便每块携带独立想法的小块开始。选择在哪里切割不是一个超参数。它是检索器能返回什么的上限。

一个询问"预算中止阈值是什么样的"的查询只有在包含中止阈值的分块可被找到时才能成功。如果固定窗口分割器将阈值从其周围上下文中切出，嵌入就会移动到不同的聚类，BM25 分数下降，重排器看到噪音，LLM 生成的答案是错误的。2024 年的论文 "LongRAG: Enhancing Retrieval-Augmented Generation with Long-context LLMs" 测得仅因分块选择就有 35 个百分点的检索召回率绝对摆动。2025 年关于上下文分块标题的后续工作缩小了差距但并未弥合。

本课并排构建五种策略，在带有金标答案跨度的 fixture 语料库上运行它们，让你亲自阅读召回率数字。

## The Concept

```mermaid
flowchart LR
  Doc[Source Document] --> S1[Fixed Window]
  Doc --> S2[Sentence]
  Doc --> S3[Recursive Split]
  Doc --> S4[Semantic Cluster]
  Doc --> S5[Structural Markdown]
  S1 --> Chunks1[Chunks]
  S2 --> Chunks2[Chunks]
  S3 --> Chunks3[Chunks]
  S4 --> Chunks4[Chunks]
  S5 --> Chunks5[Chunks]
  Chunks1 --> Index[Embedding Index]
  Chunks2 --> Index
  Chunks3 --> Index
  Chunks4 --> Index
  Chunks5 --> Index
  Index --> Eval[Recall@k vs Gold Spans]
```

### 固定窗口

暴力基线。每个 N 个字符切割一次。可选重叠，使得在位置 N 截断的句子完整地出现在从位置 N - overlap 开始的分块中。快速、确定性、边界极差。将其用作对照，而非默认。

### 句子拆分

用正则表达式或简单状态机在句子边界上分割。将一个或多个句子打包进一个不超过目标字符预算的分块中。不会在词语中间截断。但仍然会在段落中间和章节中间截断。许多早期 RAG 管线的默认选择，对没有其他结构的散文来说是合理的选择。

### 递归分割

由 2023 时代的库流行起来的层次策略。先尝试最强的分隔符（双换行，段落），回退到次强的（单换行），然后是句子，然后是字符。当分块适合预算时递归终止。在结构不一致的文档上很强，因为它按区域适配。

### 语义聚类

嵌入每个句子。对共享一个主题质心的连续句子进行聚类。每当与质心的运行相似度低于阈值时切割。边界反映含义，而非字符。构建较慢且依赖嵌入模型，但对段落内切换主题的文档具有韧性。

### 结构化 markdown 标题

对于携带显式结构的文档（markdown、reStructuredText、RFC 风格编号章节），在标题边界上切割。每个分块变成标题加上其下方直到相同或更高级别的下一个标题的所有内容。每个主题的最小分块，但仅在语料库格式良好时可用。

### recall@k 如何衡量边界选择

一个金标查询携带源文档内答案跨度的精确字符偏移量。分块后，你问：检索器返回的前 k 个分块中是否有任何一个与金标跨度重叠？如果有，该查询的 recall@k 为 1。如果没有，则为 0。在查询集上求平均。对每种策略运行相同的评估，结果差异告诉你哪种边界策略能适应你的语料库。

## Build It

`code/main.py` implements:

- `fixed_window(text, size, overlap)` - the baseline.
- `sentence_chunks(text, target)` - simple sentence packer.
- `recursive_split(text, separators, target)` - hierarchical recursion.
- `semantic_chunks(text, similarity_threshold)` - centroid-based clustering on top of a deterministic mock embedding.
- `structural_markdown(text)` - header-aware splitter.
- `mock_embed(text, dim)` - a hash-based embedding so the loop runs offline.
- `DenseIndex` - the same shape used in Phase 19 Track B's hybrid retrieval lesson.
- `eval_recall(strategy, corpus, queries, k)` - the comparison loop.
- A `main()` that runs every strategy on the fixture corpus and prints a recall@k table.

Run it:

```bash
python3 code/main.py
```

输出是一个小表格，每行一种策略，每列一个 k。句子拆分在结构化 fixture 上落败。结构化 markdown 在 markdown fixture 上胜出。递归分割在混合 fixture 上表现良好，因为递归可以适配。语义聚类在没有有用结构线索的散文 fixture 上胜出。

## 表格不会隐藏的故障模式

**孤儿句子。** 句子打包产生遗漏主题句的分块。嵌入随后指向错误的聚类。

**符号中间截断。** 固定窗口在代码或 YAML 内部会将一个标识符截成两半。两半部分嵌入到噪音中。

**仅有标题的分块。** 结构化 markdown 会发出一个只包含 `## Title` 的分块。过滤掉这些或将下一块的第一个段落附加过来。

**语义漂移。** 当语料库在主题上是均匀的时，语义聚类会欠切割。一个 5000 字符的分块将许多特定答案打包成一个弥漫的嵌入。将语义与硬字符上限结合。

**陈旧的嵌入。** 语义聚类使用嵌入模型。如果你更换模型，你也更换了分块。将分块模型与检索模型分开固定，或一起重建索引。

## 无需运行基准测试选择默认策略

三个属性决定新语料库的默认分块器。

| Property | Value | Default |
|----------|-------|---------|
| Document type | Prose with no structure | Recursive split, target 800 |
| Document type | Markdown / RFC / API docs | Structural markdown |
| Document type | Code | AST-aware (out of scope; see Phase 19 lesson 02) |
| Paragraph length | Long, single topic | Sentence, target 500 |
| Paragraph length | Short, mixed topics | Semantic, threshold 0.6 |

When in doubt, pick recursive split. It is the strongest single-strategy baseline.

## Use It

生产模式：

- 在发布新管线之前运行评估；不要信任你的库默认的策略。
- 当你更换嵌入模型或语料库混合时重新运行评估；胜者是取决于语料库的。
- 将策略名称持久化在每个分块的元数据中，以便稍后归因回退。

## Ship It

第 69 课的 Track F 端到端 RAG 系统使用此处选择的分块器作为第一阶段。第 68 课的评估工具从本课中 `eval_recall` 返回的相同形状读取 recall@k。选择在你的语料库上胜出的策略并将其投入生产。

## Exercises

1. Add a sixth strategy: token-window using `tiktoken` instead of character counts. Compare against fixed-window on the same fixture.
2. Inject a 30 percent fraction of code blocks into the prose fixture. Re-run the table. Explain why every strategy except structural markdown loses recall.
3. Replace the deterministic embedding with the one from your project's real provider. Measure the semantic-clustering recall delta. Report whether the spread between strategies widens or narrows.
4. Add a `summary` field per chunk: a one-sentence centroid description. Re-run the eval with the summary appended to the chunk body. Measure the recall lift.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Recall@k | "Did we get the right chunk?" | Fraction of queries where any of the top-k chunks overlaps the gold answer span |
| Chunk overlap | "Sliding window" | Re-include the last N characters of the previous chunk in the next chunk |
| Structural splitter | "Header-aware chunks" | Cut at H1/H2/H3 boundaries; the heading text is part of the chunk |
| Semantic chunker | "Topic-aware chunks" | Embed sentences, cluster by centroid similarity, cut on drift |
| Centroid drift | "Topic shift" | Cosine similarity between the running mean and the next sentence drops past a threshold |

## Further Reading

- [LongRAG: Enhancing Retrieval-Augmented Generation with Long-context LLMs (arXiv 2406.15319)](https://arxiv.org/abs/2406.15319)
- [Anthropic, Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [LlamaIndex, Chunking strategies for production RAG](https://docs.llamaindex.ai/en/stable/optimizing/production_rag/)
- Phase 11 lesson 06 - RAG fundamentals
- Phase 11 lesson 07 - advanced RAG
- Phase 19 lesson 65 - hybrid retrieval that ranks the chunks produced here
- Phase 19 lesson 68 - the eval harness that scores the strategy choice in production
