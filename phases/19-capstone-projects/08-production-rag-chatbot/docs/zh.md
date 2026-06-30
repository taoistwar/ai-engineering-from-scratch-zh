# 实践项目 08 — 面向受监管垂直领域的生产级 RAG 聊天机器人

> Harvey、Glean、Mendable 和 LlamaCloud 在 2026 年都运行着相同的生产形态。使用 docling 或 Unstructured 和 ColPali 进行视觉内容的摄入。混合搜索。使用 bge-reranker-v2-gemma 重排序。使用 Claude Sonnet 4.7 合成，提示缓存命中率达到 60-80%。使用 Llama Guard 4 和 NeMo Guardrails 进行守护。使用 Langfuse 和 Phoenix 进行监控。在 200 个问题的黄金标准集上使用 RAGAS 评分。在受监管领域（法律、临床、保险）构建一个，实践项目是通过黄金标准集、红队测试和漂移仪表盘。

**类型:** 实践项目
**语言:** Python（管道 + API），TypeScript（聊天 UI）
**前置条件:** 阶段 5（NLP），阶段 7（transformers），阶段 11（LLM 工程），阶段 12（多模态），阶段 17（基础设施），阶段 18（安全）
**涉及的阶段:** P5 · P7 · P11 · P12 · P17 · P18
**时间:** 30 小时

## 问题

受监管领域的 RAG（法律合同、临床试验方案、保险单）是 2026 年发货最多的生产形态，因为 ROI 显而易见，利害关系是具体的。Harvey（Allen & Overy）为法律构建了它。Mendable 提供开发者文档版本。Glean 涵盖企业搜索。模式是：高保真摄入，带重排序的混合检索，带引文执行和提示缓存的合成，多层安全防护，以及持续漂移监控。

难点不在于模型。难点在于司法管辖权合规（HIPAA、GDPR、SOC2）、引文级别的可审计性、成本控制（当命中率高时提示缓存可带来 60-90% 的折扣）、通过 RAGAS 忠实度进行的幻觉检测，以及当源文档更新但索引未跟上时的漂移检测。这个实践项目要求你在一个 200 个问题的黄金标准集上交付所有这些，并附带红队套件。

## 概念

管道有两侧。**摄入**：docling 或 Unstructured 解析结构化文档；ColPali 处理视觉丰富的文档；块获得摘要、标签和基于角色的访问标签。向量进入 pgvector + pgvectorscale（低于 50M 向量）或 Qdrant Cloud；稀疏 BM25 并行运行。**对话**：LangGraph 处理记忆和多轮对话；每个查询运行混合检索，使用 bge-reranker-v2-gemma-2b 重排序，使用 Claude Sonnet 4.7（提示缓存）合成，通过 Llama Guard 4 和 NeMo Guardrails 传递输出，并发出带有引文锚点的响应。

评估栈有四层。**黄金标准集**（200 个带有引文的标注 Q/A）用于正确性。**红队**（越狱、PII 提取尝试、领域外问题）用于安全性。**RAGAS** 用于每个轮次的忠实度/答案相关性/上下文精度。**漂移仪表盘**（Arize Phoenix）每周监控检索质量和幻觉分数。

提示缓存是成本杠杆。Claude 4.5+ 和 GPT-5+ 支持缓存系统提示 + 检索到的上下文。在 60-80% 的命中率下，每次查询的成本下降 3-5 倍。管道必须设计为具有稳定的前缀（系统提示 + 先重排序后的上下文），以实现高缓存命中率。

## 架构

```
documents (contracts, protocols, policies)
      |
      v
docling / Unstructured parse + ColPali for visuals
      |
      v
chunks + summaries + role-labels + jurisdiction tags
      |
      v
pgvector + pgvectorscale  +  BM25 (Tantivy)
      |
query + role + jurisdiction
      |
      v
LangGraph conversational agent
   +--- retrieve (hybrid)
   +--- filter by role + jurisdiction
   +--- rerank (bge-reranker-v2-gemma-2b or Voyage rerank-2)
   +--- synthesize (Claude Sonnet 4.7, prompt cached)
   +--- guard (Llama Guard 4 + NeMo Guardrails + Presidio output PII scrub)
   +--- cite + return
      |
      v
eval:
  RAGAS faithfulness / answer_relevance / context_precision (online)
  Langfuse annotation queue (sampled)
  Arize Phoenix drift (weekly)
  red team suite (pre-release)
```

## 技术栈

- 摄入: Unstructured.io 或 docling 用于结构化文档；ColPali 用于视觉丰富的 PDF
- 向量数据库: pgvector + pgvectorscale（低于 50M 向量）；否则 Qdrant Cloud
- 稀疏: Tantivy BM25 带字段权重
- 编排: LlamaIndex Workflows（摄入）+ LangGraph（对话）
- 重排序器: bge-reranker-v2-gemma-2b 自托管或 Voyage rerank-2 托管
- LLM: Claude Sonnet 4.7 带提示缓存；自托管的 Llama 3.3 70B 作为备用
- 评估: RAGAS 0.2 在线，DeepEval 用于幻觉和越狱套件
- 可观测性: Langfuse 自托管带标注队列；Arize Phoenix 用于漂移
- 护栏: Llama Guard 4 输入/输出分类器，NeMo Guardrails v0.12 策略，Presidio PII 脱敏
- 合规: 块上的基于角色的访问标签；针对 GDPR/HIPAA 的司法管辖权标签

```figure
canary-rollout
```

## 构建它

1. **摄入。** 使用 Unstructured 或 docling 解析你的语料库（对于严肃的构建，至少 1000-10000 个文档）。对于扫描/视觉密集的页面，通过 ColPali 路由。生成带有摘要、角色标签、司法管辖权标签的块。

2. **索引。** 稠密嵌入（Voyage-3 或 Nomic-embed-v2）进入 pgvector + pgvectorscale。通过 Tantivy 的 BM25 侧索引。角色和司法管辖权过滤器作为载荷。

3. **混合检索。** 首先按角色+司法管辖权过滤；然后并行稠密 + BM25；使用倒数排名融合合并；top-20 到重排序器；top-5 到合成器。

4. **带提示缓存的合成。** 系统提示 + 静态策略在缓存头中；重排序后的上下文作为缓存扩展；用户问题作为未缓存的后缀。目标是在稳定状态下达到 60-80% 的缓存命中率。

5. **护栏。** 输入上的 Llama Guard 4；NeMo Guardrails 轨道阻止领域外问题或策略禁止的主题；Presidio 脱敏输出中的意外 PII；引文执行后置过滤器。

6. **黄金标准集。** 由领域专家标注的 200 个 Q/A 对，带有 (answer, citations)。按精确引文匹配、答案正确性、忠实度（RAGAS）对智能体评分。

7. **红队。** 50 个对抗性提示：越狱（PAIR、TAP）、PII 外传尝试、领域外、跨司法管辖泄漏。用通过/失败和严重性评分。

8. **漂移仪表盘。** Arize Phoenix 每周追踪检索质量（nDCG、引文忠实度）。在 5% 下降时报警。

9. **成本报告。** Langfuse：提示缓存命中率、每查询 token 数、按阶段的 $/查询 分解。

## 使用它

```
$ chat --role=analyst --jurisdiction=GDPR
> what is the data-retention obligation for EU user profiles under our contract?
[retrieve]  hybrid top-20 filtered to GDPR + analyst-role
[rerank]    top-5 kept
[synth]     claude-sonnet-4.7, cache hit 74%, 0.8s
answer:
  The contract (Section 12.4, Master Services Agreement dated 2024-03-11)
  obligates EU user profile deletion within 30 days of termination per GDPR
  Article 17. The DPA amendment (DPA-v2.1, Section 5) extends this to 14 days
  for "restricted" category data.
  citations: [MSA-2024-03-11 s12.4, DPA-v2.1 s5]
```

## 交付它

`outputs/skill-production-rag.md` 描述了可交付成果。一个部署了合规标签、通过了评分标准、并使用实时漂移监控进行观察的受监管领域聊天机器人。

| 权重 | 标准 | 如何衡量 |
|:-:|---|---|
| 25 | RAGAS 忠实度 + 答案相关性 | 黄金标准集上的在线分数（200 Q/A） |
| 20 | 引文正确性 | 带有可验证源锚点的答案比例 |
| 20 | 护栏覆盖率 | Llama Guard 4 通过率 + 越狱套件结果 |
| 20 | 成本/延迟工程 | 提示缓存命中率、p95 延迟、$/查询 |
| 15 | 漂移监控仪表盘 | Phoenix 实时仪表盘，每周检索质量趋势 |
| **100** | | |

## 练习

1. 在不同司法管辖区（例如，HIPAA 与 GDPR 并存）下构建第二个语料库切片。在 20 个问题的跨司法管辖探测上展示角色+司法管辖过滤如何防止跨泄漏。

2. 测量一周生产流量下的提示缓存命中率。识别哪些查询破坏了缓存前缀。重新构建。

3. 添加带 10k token 摘要缓冲区的多轮记忆。测量忠实度是否随着对话增长而下降。

4. 将 Claude Sonnet 4.7 替换为自托管的 Llama 3.3 70B。测量 $/查询 和忠实度的差值。

5. 添加"不确定"模式：如果最高的重排序分数低于某个阈值，智能体说"我没有自信的引文"而不是回答。测量虚假置信度下降。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| Prompt caching | "缓存的系统 + 上下文" | Claude/OpenAI 功能：缓存的前缀 token 在命中时打 60-90% 的折扣 |
| RAGAS | "RAG 评估器" | 自动化忠实度、答案相关性、上下文精度的评分 |
| Golden set | "标注的评估集" | 200+ 专业标注的 Q/A 带引文；是基准真相 |
| Jurisdiction tag | "合规标签" | 附加到块上的 GDPR/HIPAA/SOC2 范围；由检索过滤器强制执行 |
| Citation faithfulness | "有根据的答案率" | 由可检索源范围支持的声明比例 |
| Drift | "检索质量衰减" | nDCG 或引文分数的每周变化；报警阈值 5% |
| Red team | "对抗性评估" | 发布前的越狱、PII 提取、领域外探测 |

## 扩展阅读

- [Harvey AI](https://www.harvey.ai) — 参考法律生产栈
- [Glean 企业搜索](https://www.glean.com) — 参考企业级 RAG
- [Mendable 文档](https://mendable.ai) — 开发者文档 RAG 参考
- [LlamaCloud Parse + Index](https://docs.llamaindex.ai/en/stable/examples/llama_cloud/llama_parse/) — 托管摄入
- [Anthropic 提示缓存](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — 成本杠杆参考
- [RAGAS 0.2 文档](https://docs.ragas.io/) — 规范的 RAG 评估框架
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — 参考漂移可观测性
- [Llama Guard 4](https://ai.meta.com/research/publications/llama-guard-4/) — 2026 年安全分类器
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) — 策略轨道框架
