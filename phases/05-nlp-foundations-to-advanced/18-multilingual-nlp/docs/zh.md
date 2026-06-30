# 多语言 NLP

> 一个模型，100+ 种语言，对其中大多数没有训练数据。跨语言迁移是 2020 年代实用化的奇迹。

**类型：** 学习
**语言：** Python
**先修要求：** 第五阶段 · 04（GloVe、FastText、子词），第五阶段 · 11（机器翻译）
**预计时间：** 约45分钟

## 问题

英语有数十亿标注样本。乌尔都语有数千个。迈蒂利语几乎没有。任何服务全球受众的实用 NLP 系统都必须在没有特定任务训练数据的长尾语言上工作。

多语言模型通过在多种语言上同时训练一个模型来解决这个问题。共享表示让模型将在高资源语言上学到的技能迁移到低资源语言。在英语情感分析上微调模型，在乌尔都语上开箱即用产生出奇好的情感预测。这就是零样本跨语言迁移，它重塑了 NLP 如何向世界发布。

本课命名权衡、规范模型，以及一个让新接触多语言工作的团队栽跟头的决定：为迁移选择源语言。

## 概念

![Cross-lingual transfer via shared multilingual embedding space](../assets/multilingual.svg)

**共享词汇表。** 多语言模型使用在所有目标语言的文本上训练的 SentencePiece 或 WordPiece tokenizer。词汇表是共享的：相同的子词单元在相关语言中表示相同的词素。英语和意大利语中的`anti-`获得相同的 token。

**共享表示。** 跨多种语言以掩码语言建模预训练的 transformer 学到，不同语言中语义相似的句子产生相似的隐藏状态。mBERT、XLM-R 和 NLLB 都表现出此特性。英语中"cat"的嵌入在法语"chat"和西班牙语"gato"附近聚类，完整句子的嵌入也是如此。

**零样本迁移。** 在一种语言（通常是英语）的标注数据上微调模型。在推理时，在模型支持的任何其他语言上运行。不需要目标语言标签。对于类型学相关的语言结果很强，对于远离的语言结果较弱。

**少样本微调。** 在目标语言中添加 100-500 个标注样本。分类任务上准确率跃升至英语基线的 95-98%。这是多语言 NLP 中性价比最高的单一杠杆。

## 模型

| 模型 | 年份 | 覆盖范围 | 备注 |
|-------|------|----------|-------|
| mBERT | 2018 | 104 种语言 | 在 Wikipedia 上训练。第一个实用的多语言 LM。低资源语言上较弱。 |
| XLM-R | 2019 | 100 种语言 | 在 CommonCrawl（远大于 Wikipedia）上训练。设定跨语言基线。Base 270M，Large 550M。 |
| XLM-V | 2023 | 100 种语言 | XLM-R 带 1M token 词汇表（对比 250k）。在低资源语言上更好。 |
| mT5 | 2020 | 101 种语言 | 用于多语言生成的 T5 架构。 |
| NLLB-200 | 2022 | 200 种语言 | Meta 的翻译模型；包括 55 种低资源语言。 |
| BLOOM | 2022 | 46 种语言 + 13 种编程语言 | 多语言训练的开放 176B LLM。 |
| Aya-23 | 2024 | 23 种语言 | Cohere 的多语言 LLM。在阿拉伯语、印地语、斯瓦希里语上很强。 |

按用例选择。分类以 XLM-R-base 作为合理的默认效果良好。生成任务根据翻译与开放式生成调用 mT5 或 NLLB。LLM 式工作与 Aya-23 或使用显式的多语言提示的 Claude 搭配。

## 源语言决定（2026 年研究）

大多数团队默认使用英语作为微调源语言。最近的研究（2026）表明这通常是错误的。

语言相似度比原始语料库大小更能预测迁移质量。对于斯拉夫语系目标语言，德语或俄语通常优于英语。对于印度语系目标语言，印地语通常优于英语。**qWALS**相似度指标（2026 年，基于世界语言结构图集特征）量化了这一点。**LANGRANK**（Lin et al., ACL 2019）是一个独立的、更早的方法，从语言相似度、语料库大小和谱系相关性的组合中对候选源语言进行排名。

实用规则：如果你的目标语言有一个类型学上接近的高资源语言亲属，首先尝试在该语言上微调，然后与英语微调比较。

## 构建它

### 步骤 1：零样本跨语言分类

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tok = AutoTokenizer.from_pretrained("joeddav/xlm-roberta-large-xnli")
model = AutoModelForSequenceClassification.from_pretrained("joeddav/xlm-roberta-large-xnli")


def classify(text, candidate_labels, hypothesis_template="This text is about {}."):
    scores = {}
    for label in candidate_labels:
        hypothesis = hypothesis_template.format(label)
        inputs = tok(text, hypothesis, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        entail_score = torch.softmax(logits, dim=-1)[2].item()
        scores[label] = entail_score
    return dict(sorted(scores.items(), key=lambda x: -x[1]))


print(classify("I love this product!", ["positive", "negative", "neutral"]))
print(classify("मुझे यह उत्पाद पसंद है!", ["positive", "negative", "neutral"]))
print(classify("J'adore ce produit !", ["positive", "negative", "neutral"]))
```

一个模型，三种语言，相同的 API。在 NLI 数据上训练的 XLM-R 通过蕴涵技巧良好地迁移到分类。

### 步骤 2：多语言嵌入空间

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

pairs = [
    ("The cat is sleeping.", "Le chat dort."),
    ("The cat is sleeping.", "El gato está durmiendo."),
    ("The cat is sleeping.", "Die Katze schläft."),
    ("The cat is sleeping.", "The dog is barking."),
]

for eng, other in pairs:
    emb_eng = model.encode([eng], normalize_embeddings=True)[0]
    emb_other = model.encode([other], normalize_embeddings=True)[0]
    sim = float(np.dot(emb_eng, emb_other))
    print(f"  {eng!r} <-> {other!r}: cos={sim:.3f}")
```

翻译在嵌入空间中接近。不同的英语句子落得更远。这就是跨语言检索、聚类和相似度能够工作的原因。

### 步骤 3：少样本微调策略

```python
from transformers import TrainingArguments, Trainer
from datasets import Dataset


def few_shot_finetune(base_model, base_tokenizer, examples):
    ds = Dataset.from_list(examples)

    def tokenize_fn(ex):
        out = base_tokenizer(ex["text"], truncation=True, max_length=128)
        out["labels"] = ex["label"]
        return out

    ds = ds.map(tokenize_fn)
    args = TrainingArguments(
        output_dir="out",
        per_device_train_batch_size=8,
        num_train_epochs=5,
        learning_rate=2e-5,
        save_strategy="no",
    )
    trainer = Trainer(model=base_model, args=args, train_dataset=ds)
    trainer.train()
    return base_model
```

对于 100-500 个目标语言样本，`num_train_epochs=5`和`learning_rate=2e-5`是安全的默认值。更高的学习率导致多语言对齐崩塌，你会得到一个仅英语的模型。

## 实际有效的评估

- **在保留集上的每种语言准确率。** 不是聚合的。聚合掩盖了长尾。
- **与单语言基线的基准对比。** 对于有足够数据的语言，从头训练的单语言模型有时优于多语言模型。测试。
- **实体级测试。** 目标语言中的命名实体。多语言模型在远离拉丁字母的文字上 tokenization 通常较弱。
- **跨语言一致性。** 两种语言中相同的含义应产生相同的预测。测量差距。

## 使用它

2026 年技术栈：

| 任务 | 推荐 |
|-----|-------------|
| 分类，100 种语言 | XLM-R-base（约 270M）微调 |
| 零样本文本分类 | `joeddav/xlm-roberta-large-xnli` |
| 多语言句子嵌入 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| 翻译，200 种语言 | `facebook/nllb-200-distilled-600M`（见第 11 课） |
| 生成式多语言 | Claude、GPT-4、Aya-23、mT5-XXL |
| 低资源语言 NLP | XLM-V 或在相关高资源语言上的领域特定微调 |

如果性能重要，始终为在目标语言上微调留出预算。零样本是起点，而不是最终答案。

### Tokenization 税（低资源语言出什么问题）

多语言模型在所有语言之间共享一个 tokenizer。该词汇表在由英语、法语、西班牙语、中文、德语主导的语料库上训练。对于任何在主导集合之外的语言，三种税收静默叠加：

- **繁殖税。** 低资源语言文本每个词的 token 数远多于英语。一个印地语句子可能需要英语等价句子的 3-5 倍 token 数。那 3-5 倍消耗了你的上下文窗口、训练效率和延迟。
- **变体恢复税。** 每个拼写错误、变音符号变体、Unicode 归一化不匹配或大小写变化在嵌入空间中变为一个冷启动不相关序列。模型无法学习母语者认为显然的正字法对应。
- **容量溢出税。** 税 1 和税 2 消耗上下文位置、层深度和嵌入维度。留给实际推理的剩余容量系统地小于同模型给高资源语言的容量。

实际症状：你的模型在印地语上正常训练，损失曲线看起来对，评估困惑度看起来合理，生产输出微妙地错误。形态学在句子中间崩塌。罕见屈折保持不可恢复。**你不能用数据规模来修复一个有问题的 tokenizer。**

缓解措施：选择一个对你的目标语言覆盖良好的 tokenizer（XLM-V 的 1M token 词汇表是直接修复）；在训练前验证保留目标文本上的 tokenization 繁殖率；对真正的长尾文字使用字节级回退（SentencePiece `byte_fallback=True`、GPT-2 式字节级 BPE），使得任何内容永远不会 OOV。

## 交付它

保存为 `outputs/skill-multilingual-picker.md`：

```markdown
---
name: multilingual-picker
description: 为多语言 NLP 任务选择源语言、目标模型和评估计划。
version: 1.0.0
phase: 5
lesson: 18
tags: [nlp, multilingual, cross-lingual]
---

给定需求（目标语言、任务类型、每种语言可用的标注数据），输出：

1. 微调的源语言。默认英语；如果目标语言有类型学上接近的高资源语言，检查 LANGRANK 或 qWALS。
2. 基础模型。XLM-R（分类）、mT5（生成）、NLLB（翻译）、Aya-23（生成式 LLM）。
3. 少样本预算。如果可用，从 100-500 个目标语言样本开始。仅在标注不可行时才零样本。
4. 评估计划。每种语言准确率（不是聚合的）、跨语言一致性、非拉丁文字上的实体级 F1。

拒绝在没有每种语言评估的情况下发布多语言模型——聚合指标隐藏长尾失败。标记 tokenization 覆盖率低的文字（阿姆哈拉语、提格里尼亚语、许多非洲语言）为需要带字节回退的模型（带 byte_fallback=True 的 SentencePiece，或像 GPT-2 这样的字节级 tokenizer）。
```

## 练习

1. **简单。** 在英语、法语、印地语和阿拉伯语每语言 10 个句子上运行零样本分类流程。报告每种语言上的准确率。你应该看到法语很强、印地语不错、阿拉伯语有变化。
2. **中等。** 使用`paraphrase-multilingual-MiniLM-L12-v2`在一个小型混合语言语料库上构建跨语言检索器。用英语查询，以任何语言检索文档。测量 recall@5。
3. **困难。** 比较印地语分类任务的英语源和印地语源微调。在两种方案下使用 500 个目标语言样本进行少样本微调。报告哪种源产生更好的印地语准确率以及好多少。这是 LANGRANK 论点的小型版。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| 多语言模型 | 一个模型，多种语言 | 跨语言共享的词汇表和参数。 |
| 跨语言迁移 | 在一种语言上训练，在另一种上运行 | 在源语言上微调，在目标上评估，无需目标语言标签。 |
| 零样本 | 没有目标语言标签 | 不经在目标语言上微调的迁移。 |
| 少样本 | 少量目标标签 | 用于微调的 100-500 个目标语言样本。 |
| mBERT | 第一个多语言 LM | 104 语言 BERT，在 Wikipedia 上预训练。 |
| XLM-R | 标准跨语言基线 | 100 语言 RoBERTa，在 CommonCrawl 上预训练。 |
| NLLB | Meta 的 200 语言 MT | 不让任何语言掉队。包括 55 种低资源语言。 |

## 扩展阅读

- [Conneau et al. (2019). Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116) — XLM-R 论文。
- [Pires, Schlinger, Garrette (2019). How Multilingual is Multilingual BERT?](https://arxiv.org/abs/1906.01502) — 开始了跨语言迁移研究线的分析论文。
- [Costa-jussà et al. (2022). No Language Left Behind](https://arxiv.org/abs/2207.04672) — NLLB-200 论文。
- [Üstün et al. (2024). Aya Model: An Instruction Finetuned Open-Access Multilingual Language Model](https://arxiv.org/abs/2402.07827) — Aya，Cohere 的多语言 LLM。
- [Language Similarity Predicts Cross-Lingual Transfer Learning Performance (2026)](https://www.mdpi.com/2504-4990/8/3/65) — qWALS / LANGRANK 源语言论文。
