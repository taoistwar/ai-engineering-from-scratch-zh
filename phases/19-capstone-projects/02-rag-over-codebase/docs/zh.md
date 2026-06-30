# 实践项目 02 — 代码库 RAG（跨仓库语义搜索）

> 2026 年，每个严肃的工程组织都运行着一个能理解含义而不仅仅是字符串的内部代码搜索。Sourcegraph Amp、Cursor 的代码库问答、Augment 的企业图谱、Aider 的 repomap、Pinterest 的内部 MCP——形态相同。摄入多个仓库，用 tree-sitter 解析，将函数和类级别的块嵌入，混合搜索，重排序，用引文回答。这个实践项目要求你构建一个，能跨 10 个仓库处理 200 万行代码，并在每次 git push 时能在增量重建索引中存活。

**类型:** 实践项目
**语言:** Python（摄入），TypeScript（API + UI）
**前置条件:** 阶段 5（NLP 基础），阶段 7（transformers），阶段 11（LLM 工程），阶段 13（工具），阶段 17（基础设施）
**涉及的阶段:** P5 · P7 · P11 · P13 · P17
**时间:** 30 小时

## 问题

到 2026 年，每个前沿编程智能体都配备了代码库检索引擎，因为仅靠上下文窗口并不能解决跨仓库的问题。Claude 的 1M token 上下文有所帮助，但并不能消除对排序检索的需求。对原始块进行简单的余弦搜索会在生成的代码、单一仓库复制品和很少被导入的长尾符号上产生有毒的结果。生产级的解决方案是对 AST 感知的块进行混合（稠密 + BM25）搜索，并配有重排序器，以符号引用图谱作为支撑。

你通过索引一个真实的代码舰队来学习——而不是一个教程仓库——并测量 MRR@10、引文忠实度和增量新鲜度。故障模式是基础设施层面的：一个 10 万文件的单一仓库，一个触及一半文件的推送，一个需要跨越四个仓库才能正确回答的查询。

## 概念

一个 AST 感知的摄入管道用 tree-sitter 解析每个文件，提取函数和类节点，并在节点边界处而非固定的 token 窗口处进行分块。每个块获得三种表示：稠密嵌入（Voyage-code-3 或 nomic-embed-code）、稀疏 BM25 词项和简短的摘要。摘要增加了第三种可检索的模态——用户问"X 是如何被授权的"，摘要提到"authz"，即使代码中只有 `check_permission`。

检索是混合的。一个查询同时触发稠密和 BM25 搜索，合并 top-k，并将联合结果交给交叉编码器重排序器（Cohere rerank-3 或 bge-reranker-v2-gemma-2b）。重排序后的列表交给一个长上下文合成器（Claude Sonnet 4.7，带提示缓存，或自托管的 Llama 3.3 70B），并指示用文件和行范围引用每个声明。没有引文的答案会被后置过滤器拒绝。

增量新鲜度是基础设施问题。Git push 触发差异分析：哪些文件变了，哪些符号变了。只有受影响的块被重新嵌入。受影响的跨文件符号边（导入、方法调用）被重新计算。索引在每次提交时保持一致性，无需重新处理 200 万行。

## 架构

```
git push --> webhook --> ingest worker (LlamaIndex Workflow)
                           |
                           v
             tree-sitter parse + AST chunk
                           |
            +--------------+----------------+
            v              v                v
          dense        BM25 index       summary (LLM)
        (Voyage / bge)  (Tantivy)        (Haiku 4.5)
            |              |                |
            +------> Qdrant / pgvector <----+
                            |
                            v
                      symbol graph (Neo4j / kuzu)
                            |
  query --> LangGraph agent (retrieve -> rerank -> synth)
                            |
                            v
                 Claude Sonnet 4.7 1M context
                            |
                            v
                 answer + file:line citations
```

## 技术栈

- 解析: tree-sitter，支持 17 种语言语法（Python、TS、Rust、Go、Java、C++ 等）
- 稠密嵌入: Voyage-code-3（托管）或 nomic-embed-code-v1.5（自托管），bge-code-v1 备用
- 稀疏索引: Tantivy（Rust），使用 BM25F，对符号名称 vs 正文进行字段加权
- 向量数据库: Qdrant 1.12 支持混合搜索，或 pgvector + pgvectorscale（适用于低于 50M 向量的团队）
- 块摘要模型: Claude Haiku 4.5 或 Gemini 2.5 Flash，带提示缓存
- 重排序器: Cohere rerank-3 或 bge-reranker-v2-gemma-2b 自托管
- 编排: LlamaIndex Workflows 用于摄入，LangGraph 用于查询智能体
- 合成器: Claude Sonnet 4.7（1M 上下文），带提示缓存
- 符号图谱: Neo4j（托管）或 kuzu（嵌入式），用于导入和调用边
- 可观测性: Langfuse spans 针对每个检索 + 合成步骤

## 构建它

1. **摄入遍历器。** 在每次推送钩子上迭代 git 历史记录。收集更改的文件。对于每个文件，用 tree-sitter 解析，提取函数和类节点及其完整源代码范围。发出块记录 `{repo, path, start_line, end_line, symbol, body}`。

2. **块摘要器。** 将块批量送入 Haiku 4.5 调用，并在系统前言上使用提示缓存。提示词："用一句话总结这个函数，说明其公共契约和副作用。"将摘要与块一起存储。

3. **嵌入池。** 两个并行队列：稠密（Voyage-code-3 批量 128）和摘要（相同模型，但在摘要字符串上）。将向量写入 Qdrant，载荷为 `{repo, path, start_line, end_line, symbol, kind}`。

4. **BM25 索引。** 字段加权的 Tantivy 索引：符号名称权重 4，符号正文权重 1，摘要权重 2。既能实现"查找名为 X 的函数"的查询，也能实现"查找执行 X 操作的函数"的查询。

5. **符号图谱。** 对于每个块，记录边：导入（此文件使用来自仓库 Z 的符号 Y）、调用（此函数调用类 C 上的方法 M）、继承。存储在 kuzu 中。在查询时用于跨越仓库边界扩展检索。

6. **查询智能体。** LangGraph 包含三个节点。`retrieve` 并行触发稠密 + BM25，按 (repo, path, symbol) 去重。`rerank` 对 top-50 运行交叉编码器并保留 top-10。`synth` 调用 Claude Sonnet 4.7，将重排序后的块放在上下文中，缓存系统提示，要求 file:line 引文。

7. **引文执行。** 解析模型输出；任何没有 `(repo/path:start-end)` 锚点的声明都会标记为需要重新询问或被丢弃。只返回有引用的答案给用户。

8. **增量重建索引。** 在每个 webhook 上，计算符号级别的差异。只对文本发生变化的块重新嵌入。对其导入发生变化的块重新计算符号边。衡量：在 2M LOC 代码舰队上，50 个文件的推送在 60 秒内完成重建索引。

9. **评估。** 标注 100 个跨仓库问题，带有黄金标准的 file:line 答案。测量 MRR@10、nDCG@10、引文忠实度（带有可验证锚点的声明比例）以及 p50/p99 延迟。

## 使用它

```
$ code-rag ask "how is S3 multipart abort wired into our retry budget?"
[retrieve]  12 chunks dense + 7 chunks bm25, 16 unique after dedup
[rerank]    top-5 kept (cohere rerank-3)
[synth]     claude-sonnet-4.7, cache hit rate 68%, 2.1s
answer:
  Multipart aborts are triggered by `AbortMultipartOnFail` in
  services/uploader/retry.go:122-148, which decrements the per-bucket
  retry budget defined in config/budgets.yaml:34-51 ...
  citations: [services/uploader/retry.go:122-148, config/budgets.yaml:34-51,
              libs/s3client/multipart.ts:44-61]
```

## 交付它

可交付的技能是 `outputs/skill-codebase-rag.md`。给定一个仓库语料库，它搭建摄入管道、混合索引和查询智能体，并为任何跨仓库问题返回带有引文的答案。评分标准：

| 权重 | 标准 | 如何衡量 |
|:-:|---|---|
| 25 | 检索质量 | 在 100 个问题的留存集上的 MRR@10 和 nDCG@10 |
| 20 | 引文忠实度 | 带有可验证 file:line 锚点的答案声明比例 |
| 20 | 延迟和规模 | 在已索引语料库大小上以 10k QPS 的 p95 查询延迟 |
| 20 | 增量索引正确性 | 在 50 个文件的提交上，从 git push 到可搜索的时间 |
| 15 | 用户体验和答案格式 | 引文可点击性、片段预览、追问能力 |
| **100** | | |

## 练习

1. 将 Voyage-code-3 替换为自托管的 nomic-embed-code。测量 MRR@10 的差值。报告在启用重排序的情况下差距是否缩小。

2. 将 20% 的生成代码（LLM 生成的样板代码）注入语料库并重新评估。观察检索中毒。将"generated"标志添加到载荷中并降权这些命中。

3. 在你的语料库大小上对 Qdrant 混合搜索 vs pgvector + pgvectorscale 进行基准测试。报告批量为 1 时的 p99。

4. 添加基于抽样的漂移检查：每周重新运行 100 个问题的评估。在 MRR@10 下降 > 5% 时报警。

5. 扩展到跨语言符号解析：一个调用 gRPC Go 服务的 Python 函数。使用符号图谱将它们链接起来。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| AST-aware chunking | "函数级拆分" | 在 tree-sitter 节点边界处切割代码，而非固定 token 窗口 |
| Hybrid search | "稠密 + 稀疏" | 并行运行 BM25 和向量搜索，合并 top-k，重排序 |
| Cross-encoder rerank | "第二阶段排序" | 模型同时为每个（查询，候选）对评分，比余弦相似度更准确 |
| Prompt caching | "缓存的系统提示" | 2026 年 Claude/OpenAI 功能，对重复的前缀 token 提供高达 90% 的折扣 |
| Symbol graph | "代码图谱" | 跨文件和仓库的导入、调用、继承的边 |
| Citation faithfulness | "有根据的答案率" | 用户可以通过点击锚点并阅读引用的范围来验证的声明比例 |
| Incremental re-index | "推送至可搜索时间" | 从 git push 到已更改符号可查询的墙钟时间 |

## 扩展阅读

- [Sourcegraph Amp](https://ampcode.com) — 生产级跨仓库代码智能
- [Sourcegraph Cody RAG 架构](https://sourcegraph.com/blog/how-cody-understands-your-codebase) — 本实践项目的参考深入解析
- [Aider repo-map](https://aider.chat/docs/repomap.html) — tree-sitter 排序的仓库视图
- [Augment Code 企业图谱](https://www.augmentcode.com) — 商业符号图谱 RAG
- [Qdrant 混合搜索文档](https://qdrant.tech/documentation/concepts/hybrid-queries/) — 参考实现
- [Voyage AI 代码嵌入](https://docs.voyageai.com/docs/embeddings) — Voyage-code-3 详情
- [Cohere rerank-3](https://docs.cohere.com/reference/rerank) — 交叉编码器参考
- [Pinterest MCP 内部搜索](https://medium.com/pinterest-engineering) — 内部平台参考
