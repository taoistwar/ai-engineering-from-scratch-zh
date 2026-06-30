# 文档与图表理解

> 文档不是照片。一份 PDF、科学论文、发票或手写表格具有布局、表格、图表、脚注、标题和语义结构，这是普通图像理解无法捕获的。VLM 之前的技术栈是一个流水线：Tesseract OCR + LayoutLMv3 + 表格提取启发式。VLM 浪潮用无需 OCR 的模型——Donut（2022）、Nougat（2023）、DocLLM（2023）——替换了它，这些模型直接输出结构化标记。到 2026 年，前沿只是"将页面图像以 2576px 原生分辨率喂给 Claude Opus 4.7"，结构化标记输出随之而来。本课阅读文档 AI 的三个时代弧线。

**Type:** Build
**Languages:** Python (stdlib, layout-aware document parser skeleton)
**Prerequisites:** Phase 12 · 05 (LLaVA), Phase 5 (NLP)
**Time:** ~180 minutes

## 学习目标

- 解释文档 AI 的三个时代：OCR 流水线、无需 OCR、VLM 原生。
- 描述 LayoutLMv3 的三条输入流：文本、布局（bbox）、图像 patch，带统一掩码。
- 比较 Donut（无需 OCR，图像 → 标记）、Nougat（科学论文 → LaTeX）、DocLLM（布局感知生成式）、PaliGemma 2（VLM 原生）。
- 为新任务（发票、科学论文、手写表格、中文收据）选择文档模型。

## 问题

"理解这份 PDF" 看似简单，实则困难。信息存在于：

- 文本内容（90% 的信号）。
- 布局（标题、脚注、侧栏、双列格式）。
- 表格（行、列、合并单元格）。
- 图形和图表。
- 手写批注。
- 字体和排版（标题 vs 正文）。

原始 OCR 转储文本并丢失其余信息。一个关心发票的系统需要知道"总计：¥1,245"来自右下角，而不是脚注。

## 概念

### 时代 1——OCR 流水线（2021 年前）

经典技术栈：

1. PDF → 每页图像。
2. Tesseract（或商业 OCR）提取文本及每词边界框。
3. 布局分析器识别块（标题、表格、段落）。
4. 表格结构识别器解析表格。
5. 领域规则 + 正则表达式提取字段。

适用于清晰印刷文本。在手写、倾斜扫描、复杂表格、非英语文字上失败。每种失败模式都需要自定义异常路径。

### TrOCR（2021）

TrOCR（Li 等人，arXiv:2109.10282）用训练在合成 + 真实文本图像上的 Transformer 编码器-解码器替换了 Tesseract 的经典 CNN-CTC。在手写和多语言文本上取得了干净的胜利。仍然是流水线（检测器然后 TrOCR 然后布局），但 OCR 步骤大幅改进。

### 时代 2——无需 OCR（2022-2023）

第一个无需 OCR 的模型说：完全跳过检测，直接将图像像素映射到结构化输出。

Donut（Kim 等人，arXiv:2111.15664）：
- 编码器-解码器 Transformer，编码器是 Swin-B。
- 输出是用于表单理解的 JSON，用于摘要的 markdown，或任何任务特定的 schema。
- 无需 OCR，无需布局，无需检测。

Nougat（Blecher 等人，arXiv:2308.13418）：
- 专门在科学论文上训练。
- 输出是 LaTeX / markdown。
- 处理方程式、多列布局、图形。
- 每个 arXiv 解析器都调用的模型。

这些都是专家，而非通才。Donut 在科学论文上失败；Nougat 在发票上失败。

### LayoutLMv3（2022）

一条不同的轨道。LayoutLMv3（Huang 等人，arXiv:2204.08387）保留了 OCR，但增加了布局理解：

- 三条输入流：OCR 文本 token、每 token 的 2D 边界框、图像 patch。
- 跨所有三种模态的统一掩码训练目标（掩码文本、掩码 patch、掩码布局）。
- 下游：分类、实体提取、表格 QA。

LayoutLMv3 是基于 OCR 的文档理解的巅峰。在表单和发票上表现出色。需要上游 OCR。标准化文档基准上的前 VLM 最佳准确率。

### DocLLM（2023）

DocLLM（Wang 等人，arXiv:2401.00908）是 LayoutLM 的生成式兄弟。以布局 token 为条件生成自由形式答案。更适合文档的 QA；仍依赖 OCR 输入。

### 时代 3——VLM 原生（2024+）

2024 年，VLM 变得足够好，可以完全替代流水线。将完整页面图像以高分辨率馈送给 VLM，提问，得到答案。

- LLaVA-NeXT 336-tile AnyRes 适用于小型文档。
- Qwen2.5-VL 动态分辨率原生处理 2048+ 像素。
- Claude Opus 4.7 支持 2576px 文档。
- PaliGemma 2（2025 年 4 月）专门针对文档 + 手写进行训练。

VLM 原生与 OCR 流水线之间的差距迅速缩小。到 2026 年，VLM 原生在以下方面胜出：

- 场景文本（手写 + 印刷，混合文字）。
- 带有合并单元格的复杂表格。
- 嵌入文本中的数学方程。
- 带有文本标注的图形。

OCR 流水线在以下方面仍然胜出：

- 大规模纯扫描工作负载，每页延迟至关重要。
- 流水线可靠性（确定性失败 vs VLM 幻觉）。
- 需要可审计 OCR 输出的受监管环境。

### Claude 4.7 / GPT-5 前沿

在 2576 像素原生输入下，前沿 VLM 以接近人类准确率进行文档理解。2026 年初的基准数据：

- DocVQA：Claude 4.7 ~95.1，PaliGemma 2 ~88.4，Nougat ~77.3，流水线 LayoutLMv3 ~83。
- ChartQA：Claude 4.7 ~92.2，GPT-4V ~78。
- VisualMRC：Claude 4.7 ~94。

闭源模型差距主要在于分辨率和基础 LLM 规模。7B 开源模型落后几分但正在追赶。

### 数学方程与 LaTeX 输出

科学论文需要精确的 LaTeX 输出用于方程式。Nougat 就是为此训练的。使用 LaTeX 目标训练的 VLM（Qwen2.5-VL-Math、Nougat 衍生品）产生可用的 LaTeX。没有显式 LaTeX 训练的 VLM 会产生可读但不精确的转录。

对于 2026 年的科学论文流水线：在 PDF 上链式运行 Nougat，然后在棘手页面上使用 VLM。

### 手写

仍然是最难的子任务。混合的印刷 + 手写（医生笔记、填写的表单）是 OCR 流水线在成本上仍击败 VLM 的地方。纯手写 VLM 正在改进（Claude 4.7、PaliGemma 2）。

### 2026 配方

对于新的文档 AI 项目：

- 大规模纯印刷发票：LayoutLMv3 + 规则，成本效益高。
- 混合文档（科学 + 手写 + 表单）：VLM 原生（PaliGemma 2 或 Qwen2.5-VL）。
- 完整 arXiv 摄入：Nougat 用于数学，VLM 用于图形。
- 受监管：OCR 流水线 + VLM 验证器进行交叉检查。

## 使用

`code/main.py`：

- 一个玩具布局感知分词器：给定（文本，bbox）对，产生 LayoutLMv3 风格的输入。
- 一个 Donut 风格的任务 schema 生成器：用于表单的 JSON 模板。
- 每页在 OCR 流水线、Donut、Nougat 和 VLM 原生下的 token 预算比较。

## 产出

本课产出 `outputs/skill-document-ai-stack-picker.md`。给定一个文档 AI 项目（领域、规模、质量、监管），在 OCR 流水线、无需 OCR 专家和 VLM 原生之间做出选择。

## 练习

1. 你的项目是每天 10M 张发票。哪种技术栈在不损失准确率的情况下最小化每页成本？

2. 为什么 LayoutLMv3 在表单 QA 上优于纯 CLIP-VLM，但在场景文本上表现不佳？bbox 流放弃了什么？

3. Nougat 生成 LaTeX。提出一个 VLM 原生输出在 LaTeX 保真度上击败 Nougat 的测试用例，以及一个 Nougat 胜出的用例。

4. 阅读 PaliGemma 2 论文（Google，2024）。提升文档准确率的关键训练数据添加是什么（相比 PaliGemma 1）？

5. 设计一个监管安全的混合方案：OCR 流水线作为主要，VLM 作为次要交叉检查。如何解决分歧？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| OCR 流水线 | "Tesseract 风格" | 阶段性堆叠：检测 → OCR → 布局 → 规则；确定性，脆弱 |
| 无需 OCR | "Donut 风格" | 跳过显式 OCR 的图像到输出 Transformer；单一模型 |
| 布局感知 | "LayoutLM" | 输入包括每 token 的 bbox 坐标；跨模态统一掩码 |
| VLM 原生 | "前沿 VLM" | 以高分辨率直接将页面图像馈送给 Claude/GPT/Qwen VLM；无流水线 |
| DocVQA | "文档基准" | 文档 VQA 标准；引用最多的分数 |
| 标记输出 | "LaTeX / MD" | 代替自由形式文本的结构化输出格式；支持下游自动化 |

## 拓展阅读

- [Li et al. — TrOCR (arXiv:2109.10282)](https://arxiv.org/abs/2109.10282)
- [Blecher et al. — Nougat (arXiv:2308.13418)](https://arxiv.org/abs/2308.13418)
- [Huang et al. — LayoutLMv3 (arXiv:2204.08387)](https://arxiv.org/abs/2204.08387)
- [Kim et al. — Donut (arXiv:2111.15664)](https://arxiv.org/abs/2111.15664)
- [Wang et al. — DocLLM (arXiv:2401.00908)](https://arxiv.org/abs/2401.00908)
