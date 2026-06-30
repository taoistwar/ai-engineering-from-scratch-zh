# BERT — 掩码语言建模

> GPT 预测下一个词。BERT 预测一个缺失的词。一句话的区别——以及半个十年的所有与嵌入相关的东西。

**类型：** 构建
**语言：** Python
**前置条件：** 第七阶段 · 05（完整 Transformer），第五阶段 · 02（文本表示）
**时间：** 约 45 分钟

## 问题

2018 年，每个 NLP 任务——情感分析、命名实体识别、问答、蕴含——都在自己标注的数据上从头训练自己的模型。没有预训练的"理解英语"检查点可以微调。ELMo（2018）展示了你可以用双向 LSTM 预训练上下文嵌入；它有帮助，但没有泛化。

BERT（Devlin 等人 2018）问道：如果我们取一个 transformer 编码器，在互联网上的每个句子上训练它，并强迫它从两侧的上下文中预测缺失的词，会怎样？然后你在你的下游任务上微调一个头。参数效率是一个启示。

结果：在 18 个月内，BERT 及其变体（RoBERTa、ALBERT、ELECTRA）主导了所有存在的 NLP 排行榜。到 2020 年，地球上每个搜索引擎、内容审核流水线和语义搜索系统内部都有一个 BERT。

到 2026 年，仅编码器模型仍然是分类、检索和结构化提取的正确工具——它们每个 token 的运行速度比解码器快 5–10 倍，它们的嵌入是每个现代检索技术栈的骨干。ModernBERT（2024 年 12 月）通过 Flash Attention + RoPE + GeGLU 将该架构推到了 8K 上下文。

## 概念

![掩码语言建模：选择 token，掩码它们，预测原始](../assets/bert-mlm.svg)

### 训练信号

取一个句子：`the quick brown fox jumps over the lazy dog`。

随机掩码 15% 的 token：

```
input:  the [MASK] brown fox jumps [MASK] the lazy dog
target: the  quick brown fox jumps  over  the lazy dog
```

训练模型预测掩码位置的原始 token。因为编码器是双向的，预测位置 1 的 `[MASK]` 可以使用位置 2+ 的 `brown fox jumps`。这是 GPT 不能做的事。

### BERT 掩码规则

在选出的 15% 的 token 中：

- 80% 替换为 `[MASK]`。
- 10% 替换为随机 token。
- 10% 保持不变。

为什么不是总是 `[MASK]`？因为 `[MASK]` 在推理时永远不会出现。在 100% 的掩码位置训练模型期望 `[MASK]` 会在预训练和微调之间产生分布偏移。10% 随机 + 10% 不变使模型保持诚实。

### 下一句预测（NSP）——以及为什么它被放弃了

原始 BERT 还在 NSP 上进行了训练：给定两个句子 A 和 B，预测 B 是否在 A 之后。RoBERTa（2019）对其进行了消融研究，并表明 NSP 有害无益。现代编码器跳过了它。

### 2026 年的变化：ModernBERT

2024 年 ModernBERT 论文用 2026 年的原语重建了模块：

| 组件 | 原始 BERT（2018） | ModernBERT（2024） |
|-----------|----------------------|-------------------|
| 位置 | 可学习绝对位置 | RoPE |
| 激活 | GELU | GeGLU |
| 归一化 | LayerNorm | Pre-norm RMSNorm |
| 注意力 | 完整密集 | 交替局部（128）+ 全局 |
| 上下文长度 | 512 | 8192 |
| 分词器 | WordPiece | BPE |

与 2018 年的技术栈不同，它原生支持 Flash Attention。在序列长度为 8K 时，推理速度比 DeBERTa-v3 快 2–3 倍，并具有更好的 GLUE 分数。

### 2026 年仍然选择编码器的用例

| 任务 | 为什么编码器优于解码器 |
|------|---------------------------|
| 检索 / 语义搜索嵌入 | 双向上下文 = 每个 token 更好的嵌入质量 |
| 分类（情感、意图、毒性） | 一次前向传播；无生成开销 |
| NER / token 标注 | 逐位置输出，原生双向 |
| 零样本蕴含（NLI） | 编码器顶部分类头 |
| RAG 的重排序器 | 交叉编码器评分，比 LLM 重排序器快 10 倍 |

```figure
transformer-residual
```

## 动手构建

### 步骤 1：掩码逻辑

参见 `code/main.py`。函数 `create_mlm_batch` 接收 token ID 列表、词汇表大小和掩码概率。返回输入 ID（已应用掩码）和标签（仅在掩码位置，其他地方为 -100——PyTorch 的忽略索引约定）。

```python
def create_mlm_batch(tokens, vocab_size, mask_prob=0.15, rng=None):
    input_ids = list(tokens)
    labels = [-100] * len(tokens)
    for i, t in enumerate(tokens):
        if rng.random() < mask_prob:
            labels[i] = t
            r = rng.random()
            if r < 0.8:
                input_ids[i] = MASK_ID
            elif r < 0.9:
                input_ids[i] = rng.randrange(vocab_size)
            # 否则：保持原始
    return input_ids, labels
```

### 步骤 2：在微型语料库上运行 MLM 预测

在 20 个词的词汇表、200 个句子上训练一个 2 层编码器 + MLM 头。没有梯度——我们做前向传播的健全性检查。完整训练需要 PyTorch。

### 步骤 3：比较掩码类型

展示三路规则如何使模型在没有 `[MASK]` 时仍然可用。在无掩码句子和掩码句子上预测。两者都应该产生合理的 token 分布，因为模型在训练中看到了两种模式。

### 步骤 4：微调头

在玩具情感数据集上用分类头替换 MLM 头。只有头在训练；编码器被冻结。这是每个 BERT 应用程序遵循的模式。

## 使用它

```python
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")

text = "Attention is all you need."
inputs = tok(text, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, N, 768)
```

**嵌入模型是微调过的 BERT。** 像 `all-MiniLM-L6-v2` 这样的 `sentence-transformers` 模型是用对比损失训练的 BERT。编码器是相同的。损失变了。

**交叉编码器重排序器也是微调过的 BERT。** 对 `[CLS] query [SEP] doc [SEP]` 进行配对分类。query 和 doc 之间的双向注意力正是赋予交叉编码器相对于双编码器的质量优势的原因。

**2026 年何时不选择 BERT。** 任何生成性的东西。编码器没有合理的方式来自回归地产生 token。另外：任何低于 1B 参数的情况下，一个小型解码器可以以更多灵活性匹配质量（Phi-3-Mini、Qwen2-1.5B）。

## 交付成果

参见 `outputs/skill-bert-finetuner.md`。该技能为新的分类或提取任务界定 BERT 微调的范围（骨干选择、头规范、数据、评估、停止条件）。

## 练习

1. **简单。** 运行 `code/main.py` 并打印 10,000 个 token 上的掩码分布。确认约 15% 被选中，其中约 80% 变成 `[MASK]`。
2. **中等。** 实现全词掩码：如果一个词被分词为子词，要么全部掩码，要么全部不掩码。测量这在 500 个句子的语料库上是否提高 MLM 准确率。
3. **困难。** 在来自公共数据集的 10,000 个句子上训练一个微型（2 层，d=64）BERT。微调 `[CLS]` token 进行 SST-2 情感分析。与在匹配参数下的仅解码器基线进行比较——哪个胜出？

## 关键术语

| 术语 | 人们怎么说 | 它实际上的含义 |
|------|-----------------|-----------------------|
| MLM | "掩码语言建模" | 训练信号：随机将 15% 的 token 替换为 `[MASK]`，预测原始值。 |
| 双向 | "能同时看两边" | 编码器注意力没有因果掩码——每个位置看到所有其他位置。 |
| `[CLS]` | "池化 token" | 放在每个序列开头的特殊 token；其最终嵌入用作句子级表示。 |
| `[SEP]` | "段落分隔符" | 分隔配对序列（例如 query/doc、句子 A/B）。 |
| NSP | "下一句预测" | BERT 的第二个预训练任务；在 RoBERTa 中被证明无用，2019 年后被放弃。 |
| 微调 | "适应任务" | 保持编码器大部分冻结；在其上训练一个小头用于下游任务。 |
| 交叉编码器 | "重排序器" | 将 query 和 doc 都作为输入，输出相关性分数的 BERT。 |
| ModernBERT | "2024 年刷新" | 用 RoPE、RMSNorm、GeGLU、交替局部/全局注意力、8K 上下文重建的编码器。 |

## 延伸阅读

- [Devlin 等人 (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) — 原始论文。
- [Liu 等人 (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) — 如何正确训练 BERT；终结了 NSP。
- [Clark 等人 (2020). ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555) — 替换 token 检测在匹配计算量下优于 MLM。
- [Warner 等人 (2024). Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder](https://arxiv.org/abs/2412.13663) — ModernBERT 论文。
- [HuggingFace `modeling_bert.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py) — 规范编码器参考实现。
