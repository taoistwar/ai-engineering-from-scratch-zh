# T5、BART — 编码器-解码器模型

> 编码器理解。解码器生成。把它们重新组合在一起，你就得到了一个为输入 → 输出任务而构建的模型：翻译、摘要、重写、转录。

**类型：** 学习
**语言：** Python
**前置条件：** 第七阶段 · 05（完整 Transformer），第七阶段 · 06（BERT），第七阶段 · 07（GPT）
**时间：** 约 45 分钟

## 问题

仅解码器 GPT 和仅编码器 BERT 分别精简了 2017 年的架构以达到不同的目标。但许多任务本质上是输入-输出：

- 翻译：英语 → 法语。
- 摘要：5,000 token 文章 → 200 token 摘要。
- 语音识别：音频 token → 文本 token。
- 结构化提取：散文 → JSON。

对于这些，编码器-解码器是最干净的匹配。编码器产生源的密集表示。解码器生成输出，在每一步交叉关注那个表示。训练在输出侧偏移一位。与 GPT 相同的损失，只是在编码器输出上进行条件化。

两篇论文定义了现代剧本：

1. **T5**（Raffel 等人 2019）。"Text-to-Text Transfer Transformer。"每个 NLP 任务重新框架化为文本输入、文本输出。单一架构、单一词汇表、单一损失。在掩码跨度预测上预训练（破坏输入中的跨度，在输出中解码它们）。
2. **BART**（Lewis 等人 2019）。"Bidirectional and Auto-Regressive Transformer。"去噪自编码器：以多种方式破坏输入（打乱、掩码、删除、旋转），要求解码器重建原始文本。

到 2026 年，编码器-解码器格式在输入结构重要的地方继续存在：

- Whisper（语音 → 文本）。
- Google 的翻译技术栈。
- 一些具有不同上下文和编辑结构的代码补全/修复模型。
- Flan-T5 及其变体用于结构化推理任务。

仅解码器赢得了聚光灯，但编码器-解码器从未消失。

## 概念

![带交叉注意力的编码器-解码器](../assets/encoder-decoder.svg)

### 前向循环

```
source tokens ─▶ encoder ─▶ (N_src, d_model)  ──┐
                                                 │
target tokens ─▶ decoder block                   │
                 ├─▶ masked self-attention       │
                 ├─▶ cross-attention ◀───────────┘
                 └─▶ FFN
                ↓
              next-token logits
```

至关重要的是，编码器每个输入运行一次。解码器自回归运行，但在每一步交叉关注*相同的*编码器输出。缓存编码器输出对于长输入来说是免费的加速。

### T5 预训练 — 跨度破坏

在输入中选择随机跨度（平均长度 3 个 token，总计 15%）。用唯一的哨兵 token 替换每个跨度：`<extra_id_0>`、`<extra_id_1>` 等。解码器仅输出带有哨兵前缀的被破坏的跨度：

```
source: The quick <extra_id_0> fox jumps <extra_id_1> dog
target: <extra_id_0> brown <extra_id_1> over the lazy
```

比预测整个序列更便宜的信号。在 T5 论文的消融研究中，与 MLM（BERT）和 prefix-LM（UniLM）竞争力相当。

### BART 预训练 — 多噪声去噪

BART 尝试了五种加噪函数：

1. Token 掩码。
2. Token 删除。
3. 文本填充（掩码一个跨度，解码器插入正确长度）。
4. 句子排列。
5. 文档旋转。

结合文本填充 + 句子排列产生了最好的下游数字。解码器总是重建原始文本。BART 的输出是完整序列，而不仅仅是破坏的跨度——所以预训练计算量高于 T5。

### 推理

与 GPT 相同的自回归生成。贪心 / 束搜索 / top-p 采样适用。束搜索（宽度 4–5）是翻译和摘要的标准做法，因为输出分布比聊天更窄。

### 2026 年何时选择每个变体

| 任务 | 编码器-解码器？ | 为什么 |
|------|------------------|-----|
| 翻译 | 是，通常 | 清晰的源序列；固定的输出分布；束搜索有效 |
| 语音转文本 | 是（Whisper） | 输入模态与输出不同；编码器塑造音频特征 |
| 聊天 / 推理 | 否，仅解码器 | 没有持久的"输入"——对话就是序列 |
| 代码补全 | 通常否 | 具有长上下文的仅解码器胜出；像 Qwen 2.5 Coder 这样的代码模型是仅解码器 |
| 摘要 | 两者都可以 | BART、PEGASUS 击败早期仅解码器基线；现代仅解码器 LLM 匹敌它们 |
| 结构化提取 | 两者都可以 | T5 干净，因为"文本 → 文本"吸收任何输出格式 |

自 2022 年左右的趋势：仅解码器接管了编码器-解码器曾经拥有的任务，因为 (a) 指令微调的仅解码器 LLM 通过提示泛化到任何事物，(b) 一个架构比两个更容易扩展，(c) RLHF 假设为解码器。编码器-解码器在输入模态不同（语音、图像）或束搜索质量重要的地方坚守。

## 动手构建

参见 `code/main.py`。我们为玩具语料库实现 T5 风格的跨度破坏——这是本课最有用的单个片段，因为它出现在此后的每个编码器-解码器预训练配方中。

### 步骤 1：跨度破坏

```python
def corrupt_spans(tokens, mask_rate=0.15, mean_span=3.0, rng=None):
    """选择跨度，其总和约为 mask_rate 的 token。返回 (corrupted_input, target)。"""
    n = len(tokens)
    n_mask = max(1, int(n * mask_rate))
    n_spans = max(1, int(round(n_mask / mean_span)))
    ...
```

目标格式是 T5 约定：`<sent0> span0 <sent1> span1 ...`。被破坏的输入在跨度位置用哨兵 token 交错放置未改变的 token。

### 步骤 2：验证往返

给定被破坏的输入和目标，重建原始句子。如果你的破坏是可逆的，前向传播就是良定义的。这是健全性检查——真实训练从不这样做，但测试是廉价的，可以捕获跨度簿记中的 off-by-one 错误。

### 步骤 3：BART 加噪

五个函数：`token_mask`、`token_delete`、`text_infill`、`sentence_permute`、`document_rotate`。组合其中两个并展示结果。

## 使用它

HuggingFace 参考：

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
tok = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

inputs = tok("translate English to French: Attention is all you need.", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=32)
print(tok.decode(out[0], skip_special_tokens=True))
```

T5 的技巧：任务名称放入输入文本。同一个模型处理数十个任务，因为每个任务都是文本输入、文本输出。在 2026 年，这个模式已被指令微调的仅解码器模型泛化，但 T5 首先将其编纂成文。

## 交付成果

参见 `outputs/skill-seq2seq-picker.md`。该技能根据输入-输出结构、延迟和质量目标，为新的任务在编码器-解码器和仅解码器之间进行选择。

## 练习

1. **简单。** 运行 `code/main.py`，将跨度破坏应用于 30 token 的句子，验证将非哨兵源 token 与解码出的目标跨度拼接可以重现原始文本。
2. **中等。** 实现 BART 的 `text_infill` 噪声：用单个 `<mask>` token 替换随机跨度，解码器必须推断正确的跨度长度和内容。展示一个例子。
3. **困难。** 在微型英语 → pig-Latin 语料库（200 对）上微调 `flan-t5-small`。在保留的 50 对集合上测量 BLEU。与在相同数据上用相同计算量微调的 `Llama-3.2-1B` 进行比较。

## 关键术语

| 术语 | 人们怎么说 | 它实际上的含义 |
|------|-----------------|-----------------------|
| 编码器-解码器 | "Seq2seq transformer" | 两个堆叠：用于输入的双向编码器，用于输出的带有交叉注意力的因果解码器。 |
| 交叉注意力 | "源与目标对话的地方" | 解码器的 Q × 编码器的 K/V。编码器信息进入解码器的唯一位置。 |
| 跨度破坏 | "T5 的预训练技巧" | 用哨兵 token 替换随机跨度；解码器输出跨度。 |
| 去噪目标 | "BART 的游戏" | 对输入应用噪声函数，训练解码器重建干净序列。 |
| 哨兵 token | "`<extra_id_N>` 占位符" | 在源中标记破坏跨度并在目标中重新标记它们的特殊 token。 |
| Flan | "指令微调的 T5" | 在超过 1,800 个任务上微调的 T5；使编码器-解码器在指令遵循方面具有竞争力。 |
| 束搜索 | "解码策略" | 每步保留 top-k 个部分序列；翻译/摘要的标准做法。 |
| 教师强制 | "训练时输入" | 训练期间，将真实的先前输出 token 输入解码器，而不是采样得到的。 |

## 延伸阅读

- [Raffel 等人 (2019). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) — T5。
- [Lewis 等人 (2019). BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461) — BART。
- [Chung 等人 (2022). Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416) — Flan-T5。
- [Radford 等人 (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Whisper，2026 年规范的编码器-解码器。
- [HuggingFace `modeling_t5.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py) — 参考实现。
