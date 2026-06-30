# RAG 的分块策略

> 分块配置对检索质量的影响与嵌入模型的选择一样大（Vectara NAACL 2025）。分块搞错了，再多的重排序也救不了你。

**类型：** 构建
**语言：** Python
**先修要求：** 第五阶段 · 14（信息检索），第五阶段 · 22（嵌入模型）
**预计时间：** 约60分钟

## 问题

你将一份 50 页的合同放入 RAG 系统。用户问："终止条款是什么？"检索器返回封面页。为什么？因为模型是在 512 token 的分块上训练的，而终止条款坐在 20 页深处，跨页分割，没有局部关键词将其与查询关联起来。

修复方案不是"买个更好的嵌入模型。"修复方案是分块。多大？重叠？在哪里分割？带周围上下文？

2026 年 2 月的基准测试显示了令人惊讶的结果：

- Vectara 的 2026 年研究：递归 512 token 分块以 69% → 54% 的准确率击败了语义分块。
- SPLADE + Mistral-8B 在 Natural Questions 上：重叠提供了零可量化收益。
- 上下文悬崖：回答质量在大约 2,500 token 上下文处急剧下降。

"显然的"答案（语义分块、20% 重叠、1000 token）往往是错误的。本课为六种策略建立直觉，并告诉你何时选择哪种。

## 概念

![Six chunking strategies visualized on one passage](../assets/chunking.svg)

**固定分块。** 每 N 个字符或 token 分割。最简单的基线。破句分割。好的压缩，差的连贯性。

**递归分块。** LangChain 的`RecursiveCharacterTextSplitter`。首先尝试按`\n\n`分割，然后`\n`，然后`.`，然后空格。干净地回退。2026 年的默认选择。

**语义分块。** 嵌入每个句子。计算相邻句子之间的余弦相似度。在相似度低于阈值处分割。保留主题连贯性。更慢；有时产生伤害检索的微小 40 token 片段。

**句子分块。** 在句子边界上分割。每个分块一句或 N 句的窗口。以极低的成本匹配高达约 5k token 的语义分块。

**父文档分块。** 存储小"子"分块用于检索*以及*较大的"父"分块用于上下文。按子分块检索；返回父分块。优雅退化：糟糕的子分块仍返回合理父分块。

**后分块（2024）。** 首先在 token 级别嵌入整个文档，然后将 token 嵌入池化为分块嵌入。保留跨分块上下文。适用于长上下文嵌入器（BGE-M3、Jina v3）。计算成本更高。

**上下文检索（Anthropic，2024）。** 在每个分块前加上 LLM 生成的关于其在文档中位置的摘要（"此分块是终止条款的第 3.2 节..."）。在 Anthropic 自己的基准测试中检索提升 35-50%。索引成本高。

### 击败每个默认值的规则

将分块大小与查询类型匹配：

| 查询类型 | 分块大小 |
|------------|-----------|
| 事实类（"CEO 叫什么名字？"） | 256-512 token |
| 分析性 / 多跳 | 512-1024 token |
| 整节理解 | 1024-2048 token |

NVIDIA 2026 年基准测试。分块应足够大以包含答案加局部上下文，足够小使检索器的 top-K 返回集中在答案上而不是上下文噪声上。

## 构建它

### 步骤 1：固定和递归分块

```python
def chunk_fixed(text, size=512, overlap=0):
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step)]


def chunk_recursive(text, size=512, seps=("\n\n", "\n", ". ", " ")):
    if len(text) <= size:
        return [text]
    for sep in seps:
        if sep not in text:
            continue
        parts = text.split(sep)
        chunks = []
        buf = ""
        for p in parts:
            if len(p) > size:
                if buf:
                    chunks.append(buf)
                    buf = ""
                chunks.extend(chunk_recursive(p, size=size, seps=seps[1:] or (" ",)))
                continue
            candidate = buf + sep + p if buf else p
            if len(candidate) <= size:
                buf = candidate
            else:
                if buf:
                    chunks.append(buf)
                buf = p
        if buf:
            chunks.append(buf)
        return [c for c in chunks if c.strip()]
    return chunk_fixed(text, size)
```

### 步骤 2：语义分块

```python
def chunk_semantic(text, encoder, threshold=0.6, min_chars=200, max_chars=2048):
    sentences = split_sentences(text)
    if not sentences:
        return []
    embs = encoder.encode(sentences, normalize_embeddings=True)
    chunks = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = float(embs[i] @ embs[i - 1])
        current_len = sum(len(s) for s in chunks[-1])
        if sim < threshold and current_len >= min_chars:
            chunks.append([sentences[i]])
        else:
            chunks[-1].append(sentences[i])

    result = []
    for group in chunks:
        text_group = " ".join(group)
        if len(text_group) > max_chars:
            result.extend(chunk_recursive(text_group, size=max_chars))
        else:
            result.append(text_group)
    return result
```

在你的领域上调优`threshold`。太高 → 碎片。太低 → 一个巨大的分块。

### 步骤 3：父文档分块

```python
def chunk_parent_child(text, parent_size=2048, child_size=256):
    parents = chunk_recursive(text, size=parent_size)
    mapping = []
    for p_idx, parent in enumerate(parents):
        children = chunk_recursive(parent, size=child_size)
        for child in children:
            mapping.append({"child": child, "parent_idx": p_idx, "parent": parent})
    return mapping


def retrieve_parent(child_query, mapping, encoder, top_k=3):
    child_embs = encoder.encode([m["child"] for m in mapping], normalize_embeddings=True)
    q_emb = encoder.encode([child_query], normalize_embeddings=True)[0]
    scores = child_embs @ q_emb
    top = np.argsort(-scores)[:top_k]
    seen, parents = set(), []
    for i in top:
        if mapping[i]["parent_idx"] not in seen:
            parents.append(mapping[i]["parent"])
            seen.add(mapping[i]["parent_idx"])
    return parents
```

关键见解：去重父分块。多个子分块可以映射到相同的父分块；全部返回会浪费上下文。

### 步骤 4：上下文检索（Anthropic 模式）

```python
def contextualize_chunks(document, chunks, llm):
    context_prompts = [
        f"""<document>{document}</document>
Here is the chunk to situate: <chunk>{c}</chunk>
Write 50-100 words placing this chunk in the document's context."""
        for c in chunks
    ]
    contexts = llm.batch(context_prompts)
    return [f"{ctx}\n\n{c}" for ctx, c in zip(contexts, chunks)]
```

索引添加上下文的分块。在查询时，检索从额外的周围信号中受益。

### 步骤 5：评估

```python
def recall_at_k(queries, corpus_chunks, encoder, k=5):
    chunk_embs = encoder.encode(corpus_chunks, normalize_embeddings=True)
    hits = 0
    for q_text, gold_idxs in queries:
        q_emb = encoder.encode([q_text], normalize_embeddings=True)[0]
        top = np.argsort(-(chunk_embs @ q_emb))[:k]
        if any(i in gold_idxs for i in top):
            hits += 1
    return hits / len(queries)
```

始终基准测试。你的语料库的"最佳"策略可能与任何博客文章不匹配。

## 陷阱

- **仅在事实类查询上评估分块。** 多跳查询揭示了非常不同的胜者。使用按查询类型分层的评估集。
- **没有最小大小的语义分块。** 产生损害检索的 40 token 碎片。始终强制执行`min_tokens`。
- **重叠作为迷信。** 2026 年研究发现重叠通常提供零收益并且使索引成本加倍。测量，不要假设。
- **没有最小/最大强制执行。** 5 token 或 5000 token 的分块都会破坏检索。夹紧。
- **跨文档分块。** 永远不要让分块跨越两个文档。始终按文档分块，然后合并。

## 使用它

2026 年技术栈：

| 场景 | 策略 |
|-----------|----------|
| 首次构建，未知语料库 | 递归，512 token，无重叠 |
| 事实类 QA | 递归，256-512 token |
| 分析性 / 多跳 | 递归，512-1024 token + 父文档 |
| 重度交叉引用（合同、论文） | 后分块或上下文检索 |
| 对话 / 对话语料库 | 轮次级分块 + 说话者元数据 |
| 短话语（推文、评论） | 一个文档 = 一个分块 |

从递归 512 开始。在 50 查询评估集上测量 recall@5。从那里调优。

## 交付它

保存为 `outputs/skill-chunker.md`：

```markdown
---
name: chunker
description: 为给定语料库和查询分布选择分块策略、大小和重叠。
version: 1.0.0
phase: 5
lesson: 23
tags: [nlp, rag, chunking]
---

给定一个语料库（文档类型、平均长度、领域）和查询分布（事实类 / 分析性 / 多跳），输出：

1. 策略。递归 / 句子 / 语义 / 父文档 / 后分块 / 上下文。理由。
2. 分块大小。Token 数。与查询类型相关联的理由。
3. 重叠。默认 0；如果 >0 则证明。
4. 最小/最大强制执行。`min_tokens`、`max_tokens`护栏。
5. 评估计划。在 50 查询分层评估集（事实类、分析性、多跳）上的 Recall@5。

拒绝没有最小/最大分块大小强制执行的任何分块策略。拒绝在没有消融显示其有效的情况下超过 20% 的重叠。标记没有最小 token 底线的语义分块建议。
```

## 练习

1. **简单。** 用固定（512, 0）、递归（512, 0）和递归（512, 100）分块一篇 20 页的文档。比较分块数和边界质量。
2. **中等。** 在 5 个文档上构建 30 查询评估集。测量递归、语义和父文档的 recall@5。哪种胜出？它与博客文章匹配吗？
3. **困难。** 实现上下文检索。相对于基线递归测量 MRR 提升。报告索引成本（LLM 调用）与准确率增益。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| 分块 | 文档的一个片段 | 被嵌入、索引和检索的子文档单元。 |
| 重叠 | 安全边界 | 相邻分块之间共享的 N 个 token；在 2026 年基准测试中通常无用。 |
| 语义分块 | 智能分块 | 在相邻句子嵌入相似度下降处分割。 |
| 父文档 | 两级检索 | 检索小子分块，返回较大父分块。 |
| 后分块 | 嵌入后分块 | 在 token 级别嵌入整个文档，池化为分块向量。 |
| 上下文检索 | Anthropic 的技巧 | 在索引前在每个分块前加上 LLM 生成的摘要。 |
| 上下文悬崖 | 2500 token 墙 | 在 RAG 中约 2.5k 上下文 token 处观察到的质量下降（2026 年 1 月）。 |

## 扩展阅读

- [Yepes et al. / LangChain — Recursive Character Splitting docs](https://python.langchain.com/docs/how_to/recursive_text_splitter/) — 生产中的默认选择。
- [Vectara (2024, NAACL 2025). Chunking configurations analysis](https://arxiv.org/abs/2410.13070) — 分块与嵌入选择一样重要。
- [Jina AI — Late Chunking in Long-Context Embedding Models (2024)](https://jina.ai/news/late-chunking-in-long-context-embedding-models/) — 后分块论文。
- [Anthropic — Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) — 使用 LLM 生成上下文前缀获得 35-50% 检索提升。
- [NVIDIA 2026 chunk-size benchmark — Premai summary](https://blog.premai.io/rag-chunking-strategies-the-2026-benchmark-guide/) — 按查询类型的分块大小。
