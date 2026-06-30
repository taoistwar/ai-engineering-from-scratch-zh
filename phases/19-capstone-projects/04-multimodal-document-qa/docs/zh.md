# 实践项目 04 — 多模态文档问答（视觉优先的 PDF、表格、图表）

> 2026 年的文档问答前沿已从"先 OCR 再文本处理"转向"视觉优先的后期交互"。ColPali、ColQwen2.5 和 ColQwen3-omni 将每个 PDF 页面视为图像，使用多向量后期交互嵌入，并让查询直接关注补丁。在财务 10-K 表格、科学论文和手写笔记上，这种模式以大幅优势击败了"先 OCR 再文本处理"。在 10k 页面上端到端构建管道，并发布与"先 OCR 再文本处理"的对比结果。

**类型:** 实践项目
**语言:** Python（管道），TypeScript（查看器 UI）
**前置条件:** 阶段 4（计算机视觉），阶段 5（NLP），阶段 7（transformers），阶段 11（LLM 工程），阶段 12（多模态），阶段 17（基础设施）
**涉及的阶段:** P4 · P5 · P7 · P11 · P12 · P17
**时间:** 30 小时

## 问题

企业坐拥大量被 OCR 管道搞得面目全非的 PDF：带有旋转表格的扫描版 10-K 表格、布满公式的密集科学论文、只有作为图像才有意义的图表、手写批注。将它们作为文本优先处理意味着丢失一半的信号。2026 年的答案是针对原始页面图像的后期交互多向量检索。ColPali（Illuin Tech）引入了它；ColQwen2.5-v0.2 和 ColQwen3-omni 推动了准确度。在 ViDoRe v3 上，视觉优先的检索分数以有意义的差距高于"先 OCR 再文本处理"——而且在图表、表格和手写体上差距更大。

权衡在于存储和延迟。一个 ColQwen 嵌入大约是每页约 2048 个补丁向量，而不是单一的 1024 维向量。原始存储会膨胀。DocPruner（2026 年）在几乎没有可测量的准确度损失的情况下带来 50% 的剪枝。你将索引 10k 页，测量 ViDoRe v3 nDCG@5，在 2 秒内提供答案，并与"先 OCR 再文本处理"基线直接比较。

## 概念

后期交互意味着每个查询 token 与每个补丁 token 进行评分，并对每个查询 token 的最大评分求和。你获得了细粒度的匹配，而不需要单一的汇聚向量。多向量索引（Vespa、Qdrant 多向量或 AstraDB）存储每个补丁的嵌入，并在检索时运行 MaxSim。

回答器是一个视觉语言模型，它接受查询加上 top-k 检索到的页面作为图像，并写出带有证据区域（边界框或页面引用）的答案。Qwen3-VL-30B、Gemini 2.5 Pro 和 InternVL3 是 2026 年的前沿选择。对于公式和科学记号，OCR 回退（Nougat、dots.ocr）作为可选的文本通道被拼接进来。

评估是一个二维矩阵。一轴：内容类型（纯文本段落、密集表格、条形/折线图、手写笔记、公式）。另一轴：检索方法（视觉优先后期交互 vs 先 OCR 再文本处理 vs 混合）。每个单元格得到 nDCG@5 和答案准确度。报告就是可交付成果。

## 架构

```
PDFs -> page renderer (PyMuPDF, 180 DPI)
           |
           v
  ColQwen2.5-v0.2 embed (multi-vector per page, ~2048 patches)
           |
           +------> DocPruner 50% compression
           |
           v
   multi-vector index (Vespa or Qdrant multi-vector)
           |
query ----+----> retrieve top-k pages (MaxSim)
           |
           v
  VLM answerer: Qwen3-VL-30B | Gemini 2.5 Pro | InternVL3
    inputs: query + top-k page images + optional OCR text
           |
           v
  answer with cited page numbers + evidence regions
           |
           v
  Streamlit / Next.js viewer: highlighted boxes on source page
```

## 技术栈

- 页面渲染: PyMuPDF（fitz），180 DPI，纵向归一化
- 后期交互模型: ColQwen2.5-v0.2 或 ColQwen3-omni（Hugging Face 上的 vidore 团队）
- 索引: Vespa 多向量字段，或 Qdrant 多向量，或 AstraDB with MaxSim
- 剪枝: DocPruner 2026 策略（保留高方差补丁，50% 压缩，准确度损失 < 0.5%）
- OCR 回退（公式/密集表格）: dots.ocr 或 Nougat
- VLM 回答器: 自托管的 Qwen3-VL-30B 或托管的 Gemini 2.5 Pro；InternVL3 作为备用
- 评估: ViDoRe v3 基准，M3DocVQA 用于多页推理
- 查看器 UI: Next.js 15，带 canvas 覆盖层用于证据区域

## 构建它

1. **摄入。** 遍历 10k 个 PDF 页面的语料库，涵盖 10-K 表格、科学论文和扫描文档。将每个页面渲染为 1536x2048 的 PNG。持久化 `{doc_id, page_num, image_path}`。

2. **嵌入。** 在每个页面图像上运行 ColQwen2.5-v0.2。输出形状约为 2048 个 dim 128 的补丁嵌入。应用 DocPruner 保留最高信号的一半。写入 Vespa 多向量字段或 Qdrant 多向量。

3. **查询。** 对于每个传入查询，使用查询塔（token 级别嵌入）进行嵌入。对索引运行 MaxSim：对于每个查询 token，取页面补丁嵌入的最大点积，求和。返回 top-k 页面。

4. **合成。** 使用查询和 top-5 页面图像调用 Qwen3-VL-30B。提示词："只使用提供的页面来回答。用 (doc_id, page) 引用每条声明，并命名区域（图、表、段落）。"

5. **证据区域。** 后处理答案以提取引用的区域。如果 VLM 发出边界框（Qwen3-VL 可以），在查看器中将它们渲染为叠加层。

6. **OCR 回退。** 对于被识别为公式密集的页面（基于图像方差的启发式），运行 Nougat 或 dots.ocr，并将 OCR 文本作为与图像并列的额外通道传入。

7. **评估。** 运行 ViDoRe v3（检索 nDCG@5）和 M3DocVQA（多页 QA 准确度）。也在同一语料库上使用相同的合成器运行"先 OCR 再文本处理"管道。生成内容类型 × 方法的矩阵。

8. **UI。** 首先做 Streamlit 原型；Next.js 15 生产查看器，带有逐页证据区域叠加层。

## 使用它

```
$ doc-qa ask "what was the 2024 operating margin change for segment EMEA?"
[retrieve]   top-5 pages in 320ms (ColQwen2.5, MaxSim, Vespa)
[synth]      qwen3-vl-30b, 1.4s, cited (form-10k-2024, p. 88) + (..., p. 92)
answer:
  EMEA operating margin moved from 18.2% to 16.8%, a 140bp decline.
  cited: 10-K-2024.pdf p.88 (Table 4, Segment Operating Margin)
         10-K-2024.pdf p.92 (MD&A, Operating Performance)
[viewer]     open with highlighted bounding boxes overlaid on p.88 Table 4
```

## 交付它

`outputs/skill-doc-qa.md` 描述了可交付成果：一个视觉优先的多模态文档问答系统，针对特定语料库调优，并在 ViDoRe v3 上与"先 OCR 再文本处理"基线进行评估。

| 权重 | 标准 | 如何衡量 |
|:-:|---|---|
| 25 | ViDoRe v3 / M3DocVQA 准确度 | 与 OCR-文本基线和已发布排行榜的基准对比数字 |
| 20 | 证据区域依据 | 实际包含答案范围的引用区域比例 |
| 20 | 存储和延迟工程 | DocPruner 压缩率、索引 p95、答案 p95 |
| 20 | 多页推理 | 在手动标注的 100 个多页问题集上的准确度 |
| 15 | 源检查 UX | 查看器清晰度、叠加保真度、并列比较工具 |
| **100** | | |

## 练习

1. 在同一语料库上测量 ColQwen2.5-v0.2 vs ColQwen3-omni。哪个对哪些页面是正确的而另一个是错误的？向索引添加一个"内容类"标签以按类型路由。

2. 激进地剪枝嵌入（75%，90%）。找到压缩悬崖：ViDoRe nDCG@5 降到 OCR 基线以下的那个点。

3. 构建一个混合方案：并行运行"先 OCR 再文本处理"和 ColQwen，使用 RRF 融合，用交叉编码器重排序。混合方案是否胜过任一单独方案？它在哪里帮助最大？

4. 将 Qwen3-VL-30B 替换为较小的 VLM（Qwen2.5-VL-7B）。测量准确度-美元曲线。

5. 添加手写笔记支持。渲染手写语料库，用 ColQwen 嵌入，测量检索。与手写 OCR 管道比较。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| Late interaction | "ColPali 式检索" | 查询 token 与页面补丁独立评分；MaxSim 聚合 |
| Multi-vector | "每补丁嵌入" | 每个文档有许多向量，而不是一个汇聚向量 |
| MaxSim | "后期交互评分" | 对于每个查询 token，取文档向量的最大相似度；求和 |
| DocPruner | "补丁压缩" | 2026 年剪枝方法，保留 50% 的补丁，准确度损失可忽略 |
| ViDoRe v3 | "文档检索基准" | 2026 年衡量视觉文档检索的标准 |
| Evidence region | "引用的边界框" | 源页面上定位答案范围的边界框 |
| OCR fallback | "公式通道" | 与视觉管道并排使用的文本管道，用于公式或表格密集的页面 |

## 扩展阅读

- [ColPali (Illuin Tech) 仓库](https://github.com/illuin-tech/colpali) — 参考后期交互文档检索
- [ColPali 论文 (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449) — 基础方法论文
- [Hugging Face 上的 ColQwen 系列](https://huggingface.co/vidore) — 生产就绪的检查点
- [M3DocRAG (Adobe)](https://arxiv.org/abs/2411.04952) — 多页多模态 RAG 基线
- [Vespa 多向量教程](https://docs.vespa.ai/en/colpali.html) — 参考服务栈
- [Qdrant 多向量支持](https://qdrant.tech/documentation/concepts/vectors/#multivectors) — 替代索引
- [AstraDB 多向量](https://docs.datastax.com/en/astra-db-serverless/databases/vector-search.html) — 替代托管索引
- [Nougat OCR](https://github.com/facebookresearch/nougat) — 支持公式的 OCR 回退
