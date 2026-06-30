# 问答系统

> 三个系统塑造了现代 QA。提取式寻找文本跨度。检索增强式将其建立在文档上。生成式产生答案。每个现代 AI 助手都是这三者的混合体。

**类型：** 构建
**语言：** Python
**先修要求：** 第五阶段 · 11（机器翻译），第五阶段 · 10（注意力机制）
**预计时间：** 约75分钟

## 问题

用户输入"When did the first iPhone launch?"期望"June 29, 2007。"不是"Apple's history is long and varied。"也不是孤立的"2007"没有句子。一个直接的、有根据的、正确的答案。

三种架构在过去十年主导了 QA。

- **提取式 QA。** 给定一个问题和一个已知包含答案的段落，在段落中找到答案跨度的起始和结束索引。SQuAD 是规范基准。
- **开放域 QA。** 段落未给出。首先检索相关段落，然后提取或生成答案。这是今天每个 RAG 流程的基石。
- **生成式 / 闭卷 QA。** 一个大语言模型从其参数记忆中回答。无检索。推理最快，对事实最不可靠。

2026 年的趋势是混合的：检索最佳的几个段落，然后提示一个生成模型以那些段落为基础回答。那就是 RAG，第 14 课深入涵盖检索部分。本课构建 QA 部分。

## 概念

![QA architectures: extractive, retrieval-augmented, generative](../assets/qa.svg)

**提取式。** 用 transformer（BERT 系列）一起编码问题和段落。训练两个头，预测答案跨度的起始和结束 token 索引。损失是在有效位置上的交叉熵。输出是段落中的一个跨度。永远不会产生幻觉（按构造），永远不能回答段落无法回答的问题（按构造）。

**检索增强式（RAG）。** 两个阶段。首先，检索器从语料库中找到 top-`k`个段落。其次，阅读器（提取式或生成式）使用这些段落产生答案。检索器-阅读器分离使每个部分能独立训练和评估。现代 RAG 经常在两者之间添加重排序器。

**生成式。** 一个仅解码器 LLM（GPT、Claude、Llama）从学习的权重中回答。无检索步骤。对常见知识出色，对罕见或近期事实灾难性。幻觉率与预训练数据中的事实频率成反相关。

## 构建它

### 步骤 1：使用预训练模型进行提取式 QA

```python
from transformers import pipeline

qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

passage = (
    "Apple Inc. released the first iPhone on June 29, 2007. "
    "The device was announced by Steve Jobs at Macworld in January 2007."
)
question = "When was the first iPhone released?"

answer = qa(question=question, context=passage)
print(answer)
```

```python
{'score': 0.98, 'start': 57, 'end': 70, 'answer': 'June 29, 2007'}
```

`deepset/roberta-base-squad2`在 SQuAD 2.0 上训练，该数据集包含不可回答的问题。默认情况下，`question-answering`流程即使模型的空分数获胜也返回得分最高的跨度——它*不会*自动返回空答案。要获得显式的"无法回答"行为，向流程调用传递`handle_impossible_answer=True`：流程仅在空分数超过每个跨度分数时返回空答案。无论哪种方式，始终检查`score`字段。

### 步骤 2：检索增强流程（概述）

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

corpus = [
    "Apple Inc. released the first iPhone on June 29, 2007.",
    "Macworld 2007 featured the iPhone announcement by Steve Jobs.",
    "Android launched in 2008 as Google's mobile operating system.",
    "The first iPod was released in 2001.",
]
corpus_embeddings = encoder.encode(corpus, normalize_embeddings=True)


def retrieve(question, top_k=2):
    q_emb = encoder.encode([question], normalize_embeddings=True)
    sims = (corpus_embeddings @ q_emb.T).squeeze()
    order = np.argsort(-sims)[:top_k]
    return [corpus[i] for i in order]


def answer(question):
    passages = retrieve(question, top_k=2)
    combined = " ".join(passages)
    return qa(question=question, context=combined)


print(answer("When was the first iPhone released?"))
```

两阶段流程。稠密检索器（Sentence-BERT）通过语义相似度找到相关段落。提取式阅读器（RoBERTa-SQuAD）从组合的 top 段落中提取答案跨度。适用于小型语料库。对于百万文档语料库，使用 FAISS 或向量数据库。

### 步骤 3：使用 RAG 的生成式

```python
def rag_generate(question, llm):
    passages = retrieve(question, top_k=3)
    prompt = f"""Context:
{chr(10).join('- ' + p for p in passages)}

Question: {question}

Answer using only the context above. If the context does not contain the answer, say "I don't know."
"""
    return llm(prompt)
```

提示模式很重要。明确告诉模型以上下文为基础，并在上下文不足时返回"I don't know"，相对于幼稚的提示可以减少 40-60% 的幻觉率。更精细的模式添加引用、置信度分数和结构化提取。

### 步骤 4：反映真实世界的评估

SQuAD 使用**精确匹配（EM）**和**token 级 F1**。EM 是归一化后的严格匹配（小写、去除标点、移除冠词）——要么预测完全匹配，要么得 0 分。F1 在预测与参考之间的 token 重叠上计算，给予部分分数。两者都对释义低估："June 29, 2007"与"June 29th, 2007"通常得到 0 EM（序数词破坏了归一化），但仍能从重叠 token 中获得可观的 F1。

对于生产 QA：

- **答案准确率**（由 LLM 评判或人工评判，因为指标不捕获语义等价性）。
- **引用准确率。** 被引用的段落是否实际支持答案？通过生成的引用与检索到的段落之间的字符串匹配易于自动检查。
- **拒答校准。** 当答案不在检索到的段落中时，系统是否正确地说"I don't know"？测量错误置信率。
- **检索召回率。** 在评估阅读器之前，测量检索器是否将正确段落放入 top-`k`。阅读器无法修复一个缺失的段落。

### RAGAS：2026 年生产评估框架

`RAGAS`专为 RAG 系统构建，是 2026 年的发布默认标准。它从四个维度评分而不需要黄金参考答案：

- **忠实度。** 答案中的每个声明是否来自检索到的上下文？通过基于 NLI 的蕴涵关系测量。你的主要幻觉指标。
- **答案相关性。** 答案是否回应了问题？通过从答案生成假设性问题并与真实问题比较来测量。
- **上下文精确率。** 在检索到的块中，哪些部分实际相关？低精确率 = 提示中的噪声。
- **上下文召回率。** 检索到的集合是否包含所有需要的信息？低召回率 = 阅读器无法成功。

无参考评分让你在实时生产流量上评估，无需精心策划的黄金答案。在精确匹配指标无用的开放式问题上，在上面再加 LLM 作为评判者。

`pip install ragas`。插入你的检索器 + 阅读器。每个查询获得四个标量。对退化发出警报。

## 使用它

2026 年技术栈。

| 用例 | 推荐 |
|---------|-------------|
| 给定段落，找到答案跨度 | `deepset/roberta-base-squad2` |
| 在固定语料库上，闭卷不可接受 | RAG：稠密检索器 + LLM 阅读器 |
| 在文档存储上的实时 | RAG 带混合（BM25 + 稠密）检索器 + 重排序器（第 14 课） |
| 对话式 QA（后续问题） | LLM 带对话历史 + 每轮 RAG |
| 高度事实性、受监管领域 | 在权威语料库上的提取式；永远不要仅使用生成式 |

提取式 QA 在 2026 年不受欢迎，因为带 LLM 的 RAG 处理更多情况。它仍然在需要字面引用的上下文中发布：法律研究、监管合规、审计工具。

## 交付它

保存为 `outputs/skill-qa-architect.md`：

```markdown
---
name: qa-architect
description: 选择 QA 架构、检索策略和评估计划。
version: 1.0.0
phase: 5
lesson: 13
tags: [nlp, qa, rag]
---

给定需求（语料库大小、问题类型、事实性约束、延迟预算），输出：

1. 架构。提取式、带提取式阅读器的 RAG、带生成式阅读器的 RAG，或闭卷 LLM。一句理由。
2. 检索器。无、BM25、稠密（命名编码器）或混合。
3. 阅读器。SQuAD 调优的模型、按名称的 LLM，或"领域微调的 DistilBERT。"
4. 评估。提取式基准用 EM + F1；生产用答案准确率 + 引用准确率 + 拒答校准。命名你正在测量什么以及如何测量。

拒绝为监管或合规敏感问题提供闭卷 LLM 答案。拒绝任何没有检索召回率基线的 QA 系统（在不知道检索器是否显示正确段落的情况下，无法评估阅读器）。标记需要多跳推理的问题为需要专门的多跳检索器，如 HotpotQA 训练的系统。
```

## 练习

1. **简单。** 在 10 个 Wikipedia 段落上设置上述 SQuAD 提取式流程。手工制作 10 个问题。测量答案正确的频率。如果段落和问题干净，你应该看到 7-9 个正确。
2. **中等。** 添加一个拒答分类器。当 top 检索分数低于阈值（例如 0.3 余弦）时，返回"I don't know"而不是调用阅读器。在保留集上调优阈值。
3. **困难。** 在你选择的 10,000 文档语料库上构建一个 RAG 流程。实现带 RRF 融合的混合检索（BM25 + 稠密）（见第 14 课）。测量有无混合步骤的答案准确率。记录哪种问题类型受益最多。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| 提取式 QA | 找到答案跨度 | 预测给定段落内答案的起始和结束索引。 |
| 开放域 QA | 在语料库上的 QA | 未给定段落；必须检索然后回答。 |
| RAG | 检索然后生成 | 检索增强生成。检索器 + 阅读器流程。 |
| SQuAD | 规范基准 | 斯坦福问答数据集。EM + F1 指标。 |
| 幻觉 | 编造的答案 | 不受检索上下文支持的阅读器输出。 |
| 拒答校准 | 知道何时闭嘴 | 系统在无法回答时正确地说"I don't know"。 |

## 扩展阅读

- [Rajpurkar et al. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250) — 基准论文。
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — DPR，QA 的规范稠密检索器。
- [Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — 命名 RAG 的论文。
- [Gao et al. (2023). Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) — 全面的 RAG 综述。
