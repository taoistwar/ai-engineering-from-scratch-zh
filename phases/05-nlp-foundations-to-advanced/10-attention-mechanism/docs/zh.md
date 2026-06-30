# 注意力机制 — 突破

> 解码器不再眯着眼看压缩后的摘要，而是开始查看完整的源序列。此后的一切都是注意力加工程学。

**类型：** 构建
**语言：** Python
**先修要求：** 第五阶段 · 09（序列到序列模型）
**预计时间：** 约45分钟

## 问题

第 09 课以一个可量化的失败结束。一个在玩具复制任务上训练的 GRU 编码器-解码器在长度 5 时准确率 89%，在长度 80 时接近随机水平。原因是结构性的，而不是训练 bug：编码器收集的每一点信息都必须装入一个固定大小的隐藏状态，而解码器从未看到任何其他东西。

Bahdanau、Cho 和 Bengio 在 2014 年发表了三行修复方案。不再只给解码器最终编码器状态，而是保留每个编码器状态。在每个解码器步骤，计算编码器状态的加权平均，其中权重表示"解码器现在需要在多大程度上关注编码器位置`i`？"该加权平均就是上下文，并且在每个解码器步骤都会变化。

这就是全部思想。Transformer 将其扩展。自注意力将其应用于单个序列。多头注意力并行运行。但 2014 年版已经打破了瓶颈，一旦你掌握了它，转向 transformer 就是工程学问题，而非概念性问题。

## 概念

![Bahdanau attention: decoder queries all encoder states](../assets/attention.svg)

在每个解码器步骤`t`：

1. 使用先前的解码器隐藏状态`s_{t-1}`作为**查询**。
2. 将其与每个编码器隐藏状态`h_1, ..., h_T`评分。每个编码器位置一个标量。
3. 对分数进行 Softmax 处理，得到和为 1 的注意力权重`α_{t,1}, ..., α_{t,T}`。
4. 上下文向量`c_t = Σ α_{t,i} * h_i`。编码器状态的加权平均。
5. 解码器取`c_t`加上先前的输出 token，产生下一个 token。

加权平均是关键。当解码器需要将"Je"翻译为"I"时，它给"Je"上的编码器状态高权重，给其他低权重。当需要"not"时，给"pas"高权重。上下文向量在每个步骤重新成形。

## 形状（让每个人出错的地方）

这就是每个注意力实现在第一次出错的地方。慢慢阅读。

| 事物 | 形状 | 备注 |
|-------|-------|-------|
| 编码器隐藏状态`H` | `(T_enc, d_h)` | 如果 BiLSTM，`d_h = 2 * d_hidden` |
| 解码器隐藏状态`s_{t-1}` | `(d_s,)` | 一个向量 |
| 注意力分数`e_{t,i}` | 标量 | 每个编码器位置一个 |
| 注意力权重`α_{t,i}` | 标量 | 在所有`i`上的 softmax 后 |
| 上下文向量`c_t` | `(d_h,)` | 与编码器状态形状相同 |

**Bahdanau（加性）分数。** `e_{t,i} = v_α^T * tanh(W_a * s_{t-1} + U_a * h_i)`。

- `s_{t-1}`形状为`(d_s,)`，`h_i`形状为`(d_h,)`。
- `W_a`形状为`(d_attn, d_s)`。`U_a`形状为`(d_attn, d_h)`。
- 它们在 tanh 内部的和形状为`(d_attn,)`。
- `v_α`形状为`(d_attn,)`。与`v_α`的内积压缩为标量。**这就是`v_α`的作用。** 这不是魔法。它是将注意力维向量转变为标量分数的投影。

**Luong（乘性）分数。** 三种变体：

- `dot`：`e_{t,i} = s_t^T * h_i`。需要`d_s == d_h`。硬约束。如果你的编码器是双向的，则跳过。
- `general`：`e_{t,i} = s_t^T * W * h_i`，`W`形状为`(d_s, d_h)`。消除了等维约束。
- `concat`：本质上是 Bahdanau 形式。因为前两者更便宜，很少使用。

**一个值得指出的 Bahdanau / Luong 陷阱。** Bahdanau 使用`s_{t-1}`（生成当前词*之前*的解码器状态）。Luong 使用`s_t`（*之后*的状态）。混淆它们会产生微妙的错误梯度，极难调试。选一篇论文并坚持其约定。

```figure
attention-heatmap
```

## 构建它

### 步骤 1：加性（Bahdanau）注意力

```python
import numpy as np


def additive_attention(decoder_state, encoder_states, W_a, U_a, v_a):
    projected_dec = W_a @ decoder_state
    projected_enc = encoder_states @ U_a.T
    combined = np.tanh(projected_enc + projected_dec)
    scores = combined @ v_a
    weights = softmax(scores)
    context = weights @ encoder_states
    return context, weights


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()
```

用上表检查你的形状。`encoder_states`形状为`(T_enc, d_h)`。`projected_enc`形状为`(T_enc, d_attn)`。`projected_dec`形状为`(d_attn,)`并进行广播。`combined`形状为`(T_enc, d_attn)`。`scores`形状为`(T_enc,)`。`weights`形状为`(T_enc,)`。`context`形状为`(d_h,)`。完成。

### 步骤 2：Luong dot 和 general

```python
def dot_attention(decoder_state, encoder_states):
    scores = encoder_states @ decoder_state
    weights = softmax(scores)
    return weights @ encoder_states, weights


def general_attention(decoder_state, encoder_states, W):
    projected = W.T @ decoder_state
    scores = encoder_states @ projected
    weights = softmax(scores)
    return weights @ encoder_states, weights
```

各三行。这就是为什么 Luong 的论文落地了。在大多数任务上相同的准确率，少得多的代码。

### 步骤 3：一个具体的数值示例

给定三个编码器状态（大致为"cat"、"sat"、"mat"）和一个与第一个最对齐的解码器状态，注意力分布集中在位置 0。如果解码器转移为与最后一个对齐，注意力移动到位置 2。上下文向量跟踪。

```python
H = np.array([
    [1.0, 0.0, 0.2],
    [0.5, 0.5, 0.1],
    [0.1, 0.9, 0.3],
])

s_close_to_cat = np.array([0.9, 0.1, 0.2])
ctx, w = dot_attention(s_close_to_cat, H)
print("weights:", w.round(3))
```

```
weights: [0.464 0.305 0.231]
```

第一行胜出。然后将解码器状态移向更靠近第三个编码器状态，观察权重的移动。就是这样。注意力就是显式对齐。

### 步骤 4：为什么这是通往 transformer 的桥梁

将上述语言翻译为 Q/K/V：

- **查询** = 解码器状态`s_{t-1}`
- **键** = 编码器状态（我们评分的依据）
- **值** = 编码器状态（我们加权求和的内容）

在经典注意力中，键和值是同一事物。自注意力将它们分开：你可以用不同的 K 和 V 学习投影来查询一个序列的自身。多头注意力用不同的学习投影并行运行。Transformer 多次堆叠整个阶段并丢弃 RNN。

数学是相同的。形状是相同的。从 Bahdanau 注意力到缩放点积注意力的教学跳跃主要是符号上的。

## 使用它

PyTorch 和 TensorFlow 直接提供注意力。

```python
import torch
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)
query = torch.randn(2, 5, 128)
key = torch.randn(2, 10, 128)
value = torch.randn(2, 10, 128)

output, weights = mha(query, key, value)
print(output.shape, weights.shape)
```

```
torch.Size([2, 5, 128]) torch.Size([2, 5, 10])
```

那是一个 transformer 注意力层。5 个位置的查询批次，10 个位置的键/值批次，各 128 维，8 个头。`output`是新的上下文增强查询。`weights`是你可以可视化的 5x10 对齐矩阵。

### 经典注意力仍然重要之处

- 教学。单头、单层、基于 RNN 的版本使每个概念可见。
- 设备端序列任务，transformer 不适合。
- 任何 2014-2017 年的论文。不了解 Bahdanau 的约定，你会误读它。
- MT 中的细粒度对齐分析。原始注意力权重即使在 transformer 模型上也是一个可解释性工具，读取它们需要知道它们是什么。

### 注意力权重作为解释的陷阱

注意力权重看起来是可解释的。它们是跨位置和为 1 的权重；你可以绘制它们；高表示"关注了这个"。审稿人喜欢它们。

它们不像看起来那样可解释。Jain 和 Wallace（2019）表明，注意力分布可以被置换并被任意替代方案替换，而不改变某些任务的模型预测。在没有消融或反事实检查的情况下，永远不要将注意力权重作为推理的证据报告。

## 交付它

保存为 `outputs/prompt-attention-shapes.md`：

```markdown
---
name: attention-shapes
description: 调试注意力实现中的形状 bug。
phase: 5
lesson: 10
---

给定一个破损的注意力实现，你识别形状不匹配。输出：

1. 哪个矩阵形状错误。命名张量。
2. 它应有的形状，从 (d_s, d_h, d_attn, T_enc, T_dec, batch_size) 推导。
3. 一行修复。转置、重塑或投影。
4. 一个用于捕获回归的测试。通常：断言`output.shape == (batch, T_dec, d_h)`和`weights.shape == (batch, T_dec, T_enc)`且`weights.sum(dim=-1) close to 1`。

拒绝推荐会静默广播的修复。广播隐藏的 bug 后期表现为静默的准确率下降，这是最糟糕的注意力 bug。

对于 Bahdanau 混淆，坚持解码器输入是`s_{t-1}`（步骤前状态）。对于 Luong，是`s_t`（步骤后状态）。对于点积，标记查询和键之间的维度不匹配为最常见的首次错误。
```

## 练习

1. **简单。** 实现`softmax`遮罩，使编码器中的 padding token 获得零注意力权重。在具有变长序列的批次上测试。
2. **中等。** 在 Luong `general`形式上添加多头注意力。将`d_h`分割为`n_heads`组，每个头运行注意力，拼接。验证单头情况与之前的实现匹配。
3. **困难。** 在来自第 09 课的玩具复制任务上训练一个带 Bahdanau 注意力的 GRU 编码器-解码器。绘制准确率与序列长度的关系。与无注意力基线比较。随着长度增长，你应该看到差距扩大，证实注意力提升了瓶颈。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| 注意力 | 关注事物 | 值序列的加权平均，权重由查询-键相似度计算。 |
| 查询、键、值 | QKV | 三个投影：Q 提问，K 匹配什么，V 返回什么。 |
| 加性注意力 | Bahdanau | 前馈分数：`v^T tanh(W q + U k)`。 |
| 乘性注意力 | Luong dot / general | 分数为`q^T k`或`q^T W k`。更便宜，大多数任务相同准确率。 |
| 对齐矩阵 | 漂亮的图片 | 注意力权重作为`(T_dec, T_enc)`网格。读取它以查看模型关注了什么。 |

## 扩展阅读

- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — 原始论文。
- [Luong, Pham, Manning (2015). Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025) — 三种分数变体及其比较。
- [Jain and Wallace (2019). Attention is not Explanation](https://arxiv.org/abs/1902.10186) — 可解释性警告。
- [Dive into Deep Learning — Bahdanau Attention](https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html) — 带有 PyTorch 的可运行演示。
