# 序列到序列模型

> 两个 RNN 假装成翻译器。它们遇到的瓶颈就是注意力存在的原因。

**类型：** 构建
**语言：** Python
**先修要求：** 第五阶段 · 08（文本上的 CNN + RNN），第三阶段 · 11（PyTorch 入门）
**预计时间：** 约75分钟

## 问题

分类将变长序列映射到单个标签。翻译将变长序列映射到另一个变长序列。输入和输出位于不同的词汇表中，可能是不同语言，且不能保证长度相等。

seq2seq 架构（Sutskever, Vinyals, Le, 2014）用一个故意简单的配方破解了这个问题。两个 RNN。一个读取源句子并产生一个固定大小的上下文向量。另一个读取该向量并逐 token 生成目标句子。与你在第 08 课编写的代码相同，只是连接方式不同。

这值得研究有两个原因。首先，上下文向量瓶颈是 NLP 中最具教学价值的失败。它促使了注意力和 transformer 擅长的所有内容。其次，训练配方（教师强制、计划抽样、推理时的束搜索）仍然适用于包括 LLM 在内的每个现代生成系统。

## 概念

**编码器。** 一个读取源句子的 RNN。其最终隐藏状态是**上下文向量**——整个输入的固定大小摘要。理应不丢失除源语言外的任何信息。

**解码器。** 另一个从上下文向量初始化的 RNN。每一步它取先前生成的 token 作为输入，并产生目标词汇表上的分布。采样或 argmax 选择下一个 token。将其反馈回去。重复直到产生`<EOS>` token 或达到最大长度。

**训练：** 每个解码器步骤的交叉熵损失，在序列上求和。通过两个网络的标准时间反向传播。

**教师强制。** 在训练期间，解码器在步骤`t`的输入是位置`t-1`处的*真实* token，而不是解码器自己之前的预测。这稳定了训练；没有它，早期错误会级联，模型永远学不会。在推理时，你必须使用模型自己的预测，因此总存在训练/推理分布差距。该差距称为**曝光偏差**。

**瓶颈。** 编码器学到的关于源语言的一切都必须压缩到那一个上下文向量中。长句子丢失细节。罕见词被模糊化。重新排序（chat noir vs. black cat）必须被记忆，而不是被计算。

注意力（第 10 课）通过让解码器查看编码器的*每个*隐藏状态（而不仅仅是最后一个）来解决这个问题。这就是全部亮点。

```figure
lstm-gates
```

## 构建它

### 步骤 1：编码器

```python
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        e = self.embed(src)
        outputs, hidden = self.gru(e)
        return outputs, hidden
```

`outputs`形状为`[batch, seq_len, hidden_dim]`——每个输入位置一个隐藏状态。`hidden`形状为`[1, batch, hidden_dim]`——最后一步。第 08 课说"在 outputs 上池化用于分类"。这里我们将最后一个隐藏状态保留为上下文向量，忽略每步的 outputs。

### 步骤 2：解码器

```python
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self, token, hidden):
        e = self.embed(token)
        out, hidden = self.gru(e, hidden)
        logits = self.fc(out)
        return logits, hidden
```

解码器一次调用一步。输入：一批单个 token 和当前隐藏状态。输出：下一个 token 的词汇表 logits 和更新后的隐藏状态。

### 步骤 3：带教师强制的训练循环

```python
def train_batch(encoder, decoder, src, tgt, bos_id, optimizer, teacher_forcing_ratio=0.9):
    optimizer.zero_grad()
    _, hidden = encoder(src)
    batch_size, tgt_len = tgt.shape
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    loss = 0.0
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for t in range(tgt_len):
        logits, hidden = decoder(input_token, hidden)
        step_loss = loss_fn(logits.squeeze(1), tgt[:, t])
        loss += step_loss
        use_teacher = torch.rand(1).item() < teacher_forcing_ratio
        if use_teacher:
            input_token = tgt[:, t].unsqueeze(1)
        else:
            input_token = logits.argmax(dim=-1)

    loss.backward()
    optimizer.step()
    return loss.item() / tgt_len
```

有两个值得指出的旋钮。`ignore_index=0`跳过 padding token 上的损失。`teacher_forcing_ratio`是每一步使用真实 token 相对于模型预测的概率。从 1.0（完全教师强制）开始，在训练过程中逐步退火到约 0.5，以缩小曝光偏差差距。

### 步骤 4：推理循环（贪心）

```python
@torch.no_grad()
def greedy_decode(encoder, decoder, src, bos_id, eos_id, max_len=50):
    _, hidden = encoder(src)
    batch_size = src.shape[0]
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    output_ids = []
    for _ in range(max_len):
        logits, hidden = decoder(input_token, hidden)
        next_token = logits.argmax(dim=-1)
        output_ids.append(next_token)
        input_token = next_token
        if (next_token == eos_id).all():
            break
    return torch.cat(output_ids, dim=1)
```

贪心解码在每一步选择概率最高的 token。它可能走偏：一旦你选定了一个 token，就无法收回。**束搜索**保留 top-`k`个部分序列存活，并在最后选择得分最高的完整序列。束宽 3-5 是标准的。

### 步骤 5：瓶颈演示

在一个玩具复制任务上训练模型：源`[a, b, c, d, e]`，目标`[a, b, c, d, e]`。增加序列长度。观察准确率。

```
seq_len=5   copy accuracy: 98%
seq_len=10  copy accuracy: 91%
seq_len=20  copy accuracy: 62%
seq_len=40  copy accuracy: 23%
```

单个 GRU 隐藏状态无法无损地记忆 40 个 token 的输入。信息在每个编码器步骤都存在，但解码器只看到最后一个状态。注意力直接修复了这个问题。

## 使用它

PyTorch 有`nn.Transformer`和基于`nn.LSTM`的 seq2seq 模板。Hugging Face 的`transformers`库提供完整的编码器-解码器模型（BART、T5、mBART、NLLB），在数十亿 token 上训练。

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

src = tok("Translate this to French: Hello, how are you?", return_tensors="pt")
out = model.generate(**src, max_new_tokens=50, num_beams=4)
print(tok.decode(out[0], skip_special_tokens=True))
```

现代编码器-解码器已用 transformer 替换了 RNN。高层形状（编码器、解码器、逐 token 生成）与 2014 年的 seq2seq 论文完全相同。每个块内部的机制不同。

### 何时仍然使用基于 RNN 的 seq2seq

几乎从不，对于新项目。具体例外：

- 使用有限内存逐 token 消费输入的流式翻译。
- 设备端文本生成，transformer 的内存开销过高。
- 教学。理解编码器-解码器瓶颈是理解为什么 transformer 胜出的最快路径。

### 曝光偏差及其缓解

- **计划抽样。** 在训练期间退火教师强制比率，使模型学会从自己的错误中恢复。
- **最小风险训练。** 以句子级 BLEU 分数而不是 token 级交叉熵进行训练。更接近你实际想要的。
- **强化学习微调。** 用一个指标奖励序列生成器。在现代 LLM RLHF 中使用。

所有三种方法仍然适用于基于 transformer 的生成。

## 交付它

保存为 `outputs/prompt-seq2seq-design.md`：

```markdown
---
name: seq2seq-design
description: 为给定任务设计序列到序列流程。
phase: 5
lesson: 09
---

给定一个任务（翻译、摘要、释义、问题改写），输出：

1. 架构。预训练 transformer 编码器-解码器（BART、T5、mBART、NLLB）是默认选择。基于 RNN 的 seq2seq 仅用于特定约束条件。
2. 起始检查点。命名它（`facebook/bart-base`、`google/flan-t5-base`、`facebook/nllb-200-distilled-600M`）。将检查点与任务和语言覆盖匹配。
3. 解码策略。贪心用于确定性输出，束搜索（宽度 4-5）用于质量，带温度的采样用于多样性。一句证明理由。
4. 发布前需要验证的一个失败模式。曝光偏差表现为更长输出上的生成漂移；在 90 分位数长度处抽样 20 个输出并目视检查。

拒绝推荐在不足一百万个平行样本的情况下从零训练 seq2seq。标记任何对用户可见内容使用贪心解码的流程为脆弱（贪心解码重复和循环）。
```

## 练习

1. **简单。** 实现玩具复制任务。在目标等于源的输入-输出对上训练一个 GRU seq2seq。在长度 5、10、20 处测量准确率。复现瓶颈。
2. **中等。** 添加束宽为 3 的束搜索解码。在一个小型平行语料库上相对于贪心解码测量 BLEU。记录束搜索在何处胜出（通常是最后几个 token）以及在何处没有差异。
3. **困难。** 在 1 万个样本对的释义数据集上微调`facebook/bart-base`。将微调后模型的 4-束输出与基础模型在保留输入上的输出进行比较。报告 BLEU，并选取 10 个定性示例。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| 编码器 | 输入 RNN | 读取源序列。产生每步隐藏状态和最终上下文向量。 |
| 解码器 | 输出 RNN | 从上下文向量初始化。逐次生成目标 token。 |
| 上下文向量 | 摘要 | 编码器的最终隐藏状态。固定大小。注意力解决的瓶颈。 |
| 教师强制 | 使用真实 token | 在训练时输入真实的前一个 token。稳定学习。 |
| 曝光偏差 | 训练/测试差距 | 在真实 token 上训练的模型从未练习过从自身错误中恢复。 |
| 束搜索 | 更好的解码 | 每一步保留 top-k 个部分序列，而不是贪心地选定。 |

## 扩展阅读

- [Sutskever, Vinyals, Le (2014). Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) — 原创 seq2seq 论文。四页。
- [Cho et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) — 引入了 GRU 和编码器-解码器框架。
- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — 注意力论文。本课后立即阅读。
- [PyTorch NLP from Scratch tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) — 可构建的 seq2seq + 注意力代码。
