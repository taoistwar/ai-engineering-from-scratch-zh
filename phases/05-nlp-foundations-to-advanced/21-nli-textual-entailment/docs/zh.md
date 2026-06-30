# 自然语言推理 — 文本蕴涵

> "t 蕴含 h"意味着阅读 t 的人类读者会得出结论 h 为真。NLI 是预测蕴涵 / 矛盾 / 中性（无关）的任务。表面无聊，在生产中承载负荷。

**类型：** 学习
**语言：** Python
**先修要求：** 第五阶段 · 05（情感分析），第五阶段 · 13（问答）
**预计时间：** 约60分钟

## 问题

你构建了一个摘要器。它生成了一个摘要。你怎么知道摘要不包含幻觉？

你构建了一个聊天机器人。它回答了"yes。"你怎么知道答案被检索到的段落支持？

你需要按主题分类 10,000 篇新闻文章。你没有训练标签。你能复用一个模型吗？

所有三个问题都归结为自然语言推理。NLI 问：给定一个前提`t`和一个假设`h`，`h`是被`t`蕴含、矛盾还是中性（无关）？

- **幻觉检查：** `t` = 源文档，`h` = 摘要声明。非蕴涵 = 幻觉。
- **接地 QA：** `t` = 检索到的段落，`h` = 生成的答案。非蕴涵 = 编造。
- **零样本分类：** `t` = 文档，`h` = 语言化的标签（"This is about sports"）。蕴涵 = 预测标签。

一个任务，三种生产用途。这就是为什么每个 RAG 评估框架在底层都搭载了一个 NLI 模型。

## 概念

![NLI: three-way classification, premise vs hypothesis](../assets/nli.svg)

**三个标签。**

- **蕴涵。** `t` → `h`。"The cat is on the mat"蕴含"There is a cat."
- **矛盾。** `t` → ¬`h`。"The cat is on the mat"矛盾"There is no cat."
- **中性。** 任何方向都没有推理。"The cat is on the mat"与"The cat is hungry."中性。

**不是逻辑蕴涵。** NLI 是*自然*语言推理——一个典型的人类读者会推断出什么，而不是严格的逻辑。"John walked his dog"在 NLI 中蕴含"John has a dog"，但严格的一阶逻辑只有在你公理化拥有关系时才接受它。

**数据集。**

- **SNLI**（2015）。570k 人工标注对，图像标题作为前提。领域窄。
- **MultiNLI**（2017）。433k 对，跨 10 种体裁。2026 年的标准训练语料库。
- **ANLI**（2019）。对抗性 NLI。人类专门写了旨在破坏现有模型的示例。更难。
- **DocNLI、ConTRoL**（2020–21）。文档长度前提。测试多跳和长程推理。

**架构。** 一个 transformer 编码器（BERT、RoBERTa、DeBERTa）读取`[CLS] premise [SEP] hypothesis [SEP]`。`[CLS]`表示输入 3-way softmax。在 MNLI 上训练，在保留基准上评估，在分布内对上获得 90%+ 的准确率。

**通过 NLI 进行零样本。** 给定一个文档和候选标签，将每个标签转换为假设（"This text is about sports"）。计算每个标签的蕴涵概率。选择最大值。这是 Hugging Face `zero-shot-classification`流程背后的机制。

## 构建它

### 步骤 1：运行预训练的 NLI 模型

```python
from transformers import pipeline

nli = pipeline("text-classification",
               model="facebook/bart-large-mnli",
               top_k=None)  # 返回所有标签；替代已弃用的 return_all_scores=True

premise = "The cat is sleeping on the couch."
hypothesis = "There is a cat in the room."

result = nli({"text": premise, "text_pair": hypothesis})[0]
print(result)
# [{'label': 'entailment', 'score': 0.97},
#  {'label': 'neutral', 'score': 0.02},
#  {'label': 'contradiction', 'score': 0.01}]
```

对于生产 NLI，`facebook/bart-large-mnli`和`microsoft/deberta-v3-large-mnli`是开放的默认选择。DeBERTa-v3 在排行榜上领先。

### 步骤 2：零样本分类

```python
zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "The stock market rallied after the central bank cut interest rates."
labels = ["finance", "sports", "politics", "technology"]

result = zs(text, candidate_labels=labels)
print(result)
# {'labels': ['finance', 'politics', 'technology', 'sports'],
#  'scores': [0.92, 0.05, 0.02, 0.01]}
```

模板默认是"This example is about {label}。"用`hypothesis_template`自定义。无需训练数据。无需微调。开箱即用。

### 步骤 3：RAG 的忠实度检查

```python
def is_faithful(answer, context, threshold=0.5):
    result = nli({"text": context, "text_pair": answer})[0]
    entail = next(s for s in result if s["label"] == "entailment")
    return entail["score"] > threshold
```

这是 RAGAS 忠实度的核心。将生成的答案拆分为原子声明。针对检索到的上下文检查每个声明。报告被蕴含的比例。

### 步骤 4：手写 NLI 分类器（概念性）

参见`code/main.py`获取仅标准库的玩具实现：前提和假设通过词汇重叠 + 否定检测进行比较。不具有与 transformer 模型的竞争力——但它展示了任务的形态：两个文本输入，3-way 标签输出，损失 = `{entail, contradict, neutral}`上的交叉熵。

## 陷阱

- **仅假设捷径。** 模型可以仅从假设预测标签，在 SNLI 上约 60%，因为"not"、"nobody"、"never"与矛盾相关。检测标签泄漏的强基线。
- **词汇重叠启发式。** 子序列启发式（"每个子序列被蕴含"）通过了 SNLI 但未通过 HANS/ANLI。使用对抗性基准。
- **文档长度退化。** 单句 NLI 模型在文档长度前提下下降 20+ F1。对长上下文使用 DocNLI 训练的模型。
- **零样本模板敏感性。** "This example is about {label}" vs "{label}" vs "The topic is {label}"可以摆动 10+ 分准确率。调优模板。
- **领域不匹配。** MNLI 在通用英语上训练。法律、医学和科学文本需要领域特定的 NLI 模型（例如，SciNLI、MedNLI）。

## 使用它

2026 年技术栈：

| 用例 | 模型 |
|---------|-------|
| 通用 NLI | `microsoft/deberta-v3-large-mnli` |
| 快速 / 边缘 | `cross-encoder/nli-deberta-v3-base` |
| 零样本分类（轻量） | `facebook/bart-large-mnli` |
| 文档级 NLI | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` |
| 多语言 | `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` |
| RAG 中的幻觉检测 | RAGAS / DeepEval 内部的 NLI 层 |

2026 年的元模式：NLI 是文本理解的万能胶。每当你需要"A 是否支持 B？"或"A 是否矛盾 B？"——在用另一个 LLM 调用之前先使用 NLI。

## 交付它

保存为 `outputs/skill-nli-picker.md`：

```markdown
---
name: nli-picker
description: 为分类 / 忠实度 / 零样本任务选择 NLI 模型、标签模板和评估设置。
version: 1.0.0
phase: 5
lesson: 21
tags: [nlp, nli, zero-shot]
---

给定一个用例（忠实度检查、零样本分类、文档级推理），输出：

1. 模型。命名的 NLI 检查点。与领域、长度、语言相关联的理由。
2. 模板（如果为零样本）。语言化模式。示例。
3. 阈值。决策规则的蕴涵截止值。基于校准的理由。
4. 评估。保留标注集上的准确率、仅假设基线、对抗性子集。

拒绝在没有 100 样本标注合理性检查的情况下发布零样本分类。拒绝在文档长度前提下使用句子级 NLI 模型。标记任何声称 NLI 解决幻觉的说法——它减少了幻觉；它不能消除幻觉。
```

## 练习

1. **简单。** 在 20 个手工制作的覆盖所有三个类的（前提、假设、标签）三元组上运行`facebook/bart-large-mnli`。测量准确率。添加对抗性"子序列启发式"陷阱（"I did not eat the cake" vs "I ate the cake"）并看它是否被破坏。
2. **中等。** 在 100 条 AG News 标题上比较零样本模板`"This text is about {label}"`与`"The topic is {label}"`和`"{label}"`。报告准确率摆动。
3. **困难。** 构建一个 RAG 忠实度检查器：原子声明分解 + 每个声明的 NLI。在 50 个 RAG 生成的答案上评估，带有黄金上下文。测量相对于手动标签的假阳性率和假阴性率。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| NLI | 自然语言推理 | 前提-假设关系的 3-way 分类。 |
| RTE | 识别文本蕴涵 | NLI 的更早名称；相同的任务。 |
| 蕴涵 | "t 蕴含 h" | 典型读者会在给定 t 的情况下结论 h 为真。 |
| 矛盾 | "t 排除 h" | 典型读者会在给定 t 的情况下结论 h 为假。 |
| 中性 | "未决定" | 从 t 到 h 任何方向都没有推理。 |
| 零样本分类 | NLI 作为分类器 | 将标签语言化为假设，选择最大蕴涵。 |
| 忠实度 | 答案是否被支持？ | 在（检索到的上下文、生成的答案）上的 NLI。 |

## 扩展阅读

- [Bowman et al. (2015). A large annotated corpus for learning natural language inference](https://arxiv.org/abs/1508.05326) — SNLI。
- [Williams, Nangia, Bowman (2017). A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference](https://arxiv.org/abs/1704.05426) — MultiNLI。
- [Nie et al. (2019). Adversarial NLI](https://arxiv.org/abs/1910.14599) — ANLI 基准。
- [Yin, Hay, Roth (2019). Benchmarking Zero-shot Text Classification](https://arxiv.org/abs/1909.00161) — NLI 作为分类器。
- [He et al. (2021). DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654) — 2026 年的 NLI 主力。
