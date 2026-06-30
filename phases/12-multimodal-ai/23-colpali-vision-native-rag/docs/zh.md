# ColPali 与视觉原生文档 RAG

> 传统 RAG 将 PDF 解析为文本，拆分为块，嵌入块，存储向量。每一步都损失信号：OCR 丢弃图表数据，分块打断表格行，文本嵌入忽略图形。ColPali（Faysse 等人，2024 年 7 月）提出了一个更简单的问题：为什么要提取文本？直接通过 PaliGemma 嵌入页面图像，使用 ColBERT 风格的后期交互进行检索，并保留文档所携带的所有布局、图形、字体和格式信号。已发表基准：在视觉丰富的文档上端到端准确率比文本 RAG 高 20-40%。ColQwen2、ColSmol 和 VisRAG 扩展了该模式。本课阅读视觉原生 RAG 的论点，并构建一个微型 ColPali 风格索引器。

**Type:** Build
**Languages:** Python (stdlib, multi-vector indexer + MaxSim scorer)
**Prerequisites:** Phase 11 (LLM Engineering — RAG basics), Phase 12 · 05 (LLaVA)
**Time:** ~180 minutes

## 学习目标

- 解释双编码器检索（每个文档一个向量）与后期交互检索（每个文档多个向量）的区别。
- 描述 ColBERT 的 MaxSim 操作以及 ColPali 如何将其从文本 token 推广到图像 patch。
- 构建一个微型 ColPali 风格索引器：页面 → patch 嵌入 → 查询词嵌入上的 MaxSim → top-k 页面。
- 在发票/财务报告用例上比较 ColPali + Qwen2.5-VL 生成器 vs 文本 RAG + GPT-4。

## 问题

对 PDF 的文本 RAG 丢弃了文档的大部分。财务报告的 Q3 收入增长通常在图表中；医疗报告的发现存在于标注图像中；法律合同的签名块是布局事实，而非文本事实。

文本 RAG 流水线：

1. PDF → 通过 OCR / pdftotext 得到文本。
2. 文本 → 300-500 个 token 的块。
3. 块 → 双编码器嵌入（一个向量）。
4. 用户查询 → 嵌入 → 余弦相似度 → top-k 块。
5. 块 + 查询 → LLM。

五个有损步骤。图表未被捕获。表格跨块被拆散。多列布局被摊平。图形标注消失。

ColPali 的修复：跳过 OCR，直接嵌入页面图像。使用 ColBERT 风格的后期交互进行检索，以便模型在查询时可以关注细粒度 patch。

## 概念

### ColBERT（2020）

ColBERT（Khattab & Zaharia，arXiv:2004.12832）是一种文本检索方法。不是每个文档一个向量，而是每个 token 产生一个向量。在查询时：

- 查询 token 获得自己的嵌入（N_q 个向量）。
- 文档 token 获得嵌入（N_d 个向量，通常缓存）。
- 分数 = 查询 token 上对文档 token 余弦相似度最大值的求和：Σ_i max_j cos(q_i, d_j)。

这就是 MaxSim 操作。每个查询 token"挑选"其最佳匹配的文档 token。最终分数是总和。

优点：强召回，处理词级语义。缺点：每个文档 N_d 个向量，存储昂贵。

### ColPali

ColPali（Faysse 等人，arXiv:2407.01449）将 ColBERT 模式应用于图像。

- 每个页面由 PaliGemma（ViT + 语言）编码为 patch 嵌入：每页 N_p 个向量。
- 每个用户查询（文本）被编码为查询 token 嵌入：N_q 个向量。
- 分数 = Σ_i max_j cos(q_i, p_j)，即查询文本 token 与页面图像 patch 上的 MaxSim。
- 按总分检索 top-k 页面。

在文档摄入时：使用 PaliGemma 嵌入每个页面，存储所有 patch 嵌入。在查询时：嵌入查询 token，针对所有已存储的页面嵌入计算 MaxSim，返回 top-k 页面。

优点：在视觉丰富的文档上端到端击败文本 RAG 20-40%。每个 patch 向量捕获局部布局和内容。

缺点：每页 N_p 个 patch × 4 字节浮点 × D 维向量 = 存储快速增长。通过 PQ / OPQ 量化缓解。

### ColQwen2 与 ColSmol

ColQwen2（illuin-tech，2024-2025）将 PaliGemma 换为 Qwen2-VL。更好的基础编码器，更好的检索。

ColSmol 是用于本地/边缘场景的较小规模变体。约 1B 参数的 ColSmol 检索器可在消费级 GPU 上运行。

### VisRAG

VisRAG（Yu 等人，arXiv:2410.10594）是一种不同的变体：不是对 patch 使用 MaxSim，而是使用 VLM 将每个页面池化为单个向量，然后使用双编码器检索。索引更快 + 存储更小，召回更弱。

质量 vs 成本的权衡：ColPali 用于质量，VisRAG 用于规模。

### M3DocRAG

M3DocRAG（Cho 等人，arXiv:2411.04952）将多模态检索扩展到多页面多文档推理。跨文档检索页面，为 VLM 组成多页面上下文。

### ViDoRe——基准

ColPali 的伴随基准。视觉文档检索评估。任务包括财务报告、科学论文、行政文件、医疗记录、手册。指标：nDCG@5。

ColPali-v1 在 ViDoRe 上得分约 80% nDCG@5；相同文档上的文本 RAG 得分约 50-60%。

### 端到端 RAG 流水线

对于视觉原生 RAG：

1. 摄入：PDF → 页面图像 → PaliGemma 编码 → 存储所有 patch 嵌入。
2. 查询：用户文本 → 查询 token 嵌入 → 对所有已索引页面的 MaxSim → top-k 页面。
3. 生成：top-k 页面图像 + 查询 → VLM（Qwen2.5-VL 或 Claude）→ 答案。

没有任何 OCR。图形、图表、字体、布局全部流入答案。

### 存储数学

一份 50 页的财务报告，每页 729 个 patch，128 维嵌入：

- ColPali：50 * 729 * 128 * 4 字节 = 约 18 MB 原始，PQ 后约 4 MB。
- 文本 RAG：50 块 * 768-dim * 4 字节 = 约 150 kB。

ColPali 每文档存储约 30 倍。大规模下，OPQ / PQ 将其降至约 5-10 倍，通常可接受。

### 文本 RAG 仍胜出的场景

- 没有布局信号的纯文本文档（百科文章、聊天日志）。文本 RAG 更简单且存储更便宜。
- 存储主导成本的百万页级档案。
- 严格要求可提取 OCR 文本与检索并列的严格监管要求。

对于 2026 年的其他一切——财务报告、科学论文、法律合同、医疗记录、UX 文档——视觉原生 RAG 胜出。

## 使用

`code/main.py`：

- 玩具 patch 编码器：将"页面"（小型特征向量网格）映射为 patch 嵌入数组。
- MaxSim 评分器：计算查询 token 嵌入集与页面 patch 集之间的 ColBERT 风格分数。
- 索引 5 个玩具页面，运行 3 个查询，返回带分数的 top-k。

## 产出

本课产出 `outputs/skill-vision-rag-designer.md`。给定一个文档 RAG 项目，选择 ColPali / ColQwen2 / VisRAG / 文本 RAG，并确定存储规模。

## 练习

1. 一份 200 页的年报，每页 729 个 patch，128 维嵌入，4 字节浮点。计算原始存储和 PQ 压缩（8x）后的存储。

2. MaxSim 是 Σ_i max_j cos(q_i, p_j)。这个求和捕获了什么简单均值相似度无法捕获的信息？

3. ColPali 将页面索引为 patch 集。如果我们改为在词级别索引（如 ColBERT 那样）会有什么变化？权衡是什么？

4. 为具有 500ms 每查询延迟预算的 1M 页语料设计端到端流水线。选择 ColQwen2 / VisRAG 并论证。

5. 阅读 M3DocRAG（arXiv:2411.04952）。描述多页面注意力模式以及它与单页面 ColPali 检索有何不同。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 后期交互 | "ColBERT 风格" | 使用每 token 或每 patch 嵌入 + MaxSim 的检索，而非单个文档向量 |
| MaxSim | "patch 上的最大值" | 对每个查询 token，选择最高相似度的文档 token；跨查询求和 |
| 双编码器 | "单向量" | 每个文档一个向量；更快但会丢失粒度 |
| 多向量 | "每个文档多个向量" | 每个文档/页面存储 N_p 个向量；存储成本增加但召回改善 |
| Patch 嵌入 | "页面特征" | 来自 VLM 编码器的每图像 patch 一个向量，按页面缓存 |
| ViDoRe | "视觉文档基准" | ColPali 用于视觉文档检索的基准套件 |
| PQ 量化 | "乘积量化" | 在维持向量相似度的同时将存储压缩约 8 倍 |

## 拓展阅读

- [Faysse et al. — ColPali (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449)
- [Khattab & Zaharia — ColBERT (arXiv:2004.12832)](https://arxiv.org/abs/2004.12832)
- [Yu et al. — VisRAG (arXiv:2410.10594)](https://arxiv.org/abs/2410.10594)
- [Cho et al. — M3DocRAG (arXiv:2411.04952)](https://arxiv.org/abs/2411.04952)
- [illuin-tech/colpali GitHub](https://github.com/illuin-tech/colpali)
