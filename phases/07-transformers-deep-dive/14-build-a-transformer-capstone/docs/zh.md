# 从零开始构建 Transformer — 毕业项目

> 十三课。一个模型。没有捷径。

**类型：** 构建
**语言：** Python
**前置条件：** 第七阶段 · 01 至 13。不要跳过。
**时间：** 约 120 分钟

## 问题

你已经读了每篇论文。你已经实现了注意力、多头拆分、位置编码、编码器和解码器模块、BERT 和 GPT 损失、MoE、KV 缓存。现在让它们在一个真实任务上协同工作。

毕业项目：在字符级语言建模任务上端到端训练一个小型仅解码器 transformer。它读莎士比亚。它生成新的莎士比亚。它足够小，可以在笔记本电脑上 10 分钟内训练。它足够正确，替换为更大的数据集和更长的训练就能得到一个真正的 LM。

这是课程中的"nanoGPT"。它不是原创的——Karpathy 2023 年的 nanoGPT 教程是每个学生至少写一次的参考实现。我们借用其形状并围绕我们涵盖的内容重新装备。

## 概念

![从零开始的 Transformer 模块图](../assets/capstone.svg)

带注释的架构：

```
input tokens (B, N)
   │
   ▼
token embedding + positional embedding  ◀── 第 04 课（RoPE 选项）
   │
   ▼
┌──── block × L ────────────────────┐
│  RMSNorm                          │  ◀── 第 05 课
│  MultiHeadAttention (causal)      │  ◀── 第 03 + 07 课（因果掩码）
│  residual                         │
│  RMSNorm                          │
│  SwiGLU FFN                       │  ◀── 第 05 课
│  residual                         │
└────────────────────────────────── ┘
   │
   ▼
final RMSNorm
   │
   ▼
lm_head（与 token embedding 绑定）
   │
   ▼
logits (B, N, V)
   │
   ▼
shift-by-one cross-entropy            ◀── 第 07 课
```

### 我们交付的内容

- `GPTConfig` — 一个配置所有超参数的地方。
- `MultiHeadAttention` — 因果、批量，带有可选 Flash 风格路径（PyTorch 的 `scaled_dot_product_attention`）。
- `SwiGLUFFN` — 现代 FFN。
- `Block` — pre-norm，残差包裹的注意力 + FFN。
- `GPT` — 嵌入、堆叠模块、LM head、generate()。
- 带有 AdamW、余弦 LR、梯度裁剪的训练循环。
- 在莎士比亚文本上的字符级分词器。

### 我们不交付的内容

- RoPE — 在第 04 课中概念性地实现。这里我们为简单起见使用可学习的位置嵌入。练习要求你替换为 RoPE。
- 生成期间的 KV 缓存 — 每个生成步骤在整个前缀上重新计算注意力。更慢但更简单。练习要求你添加 KV 缓存。
- Flash Attention — PyTorch 2.0+ 如果输入匹配则自动调度；我们使用 `F.scaled_dot_product_attention`。
- MoE — 每个模块一个 FFN。你在第 11 课中看到了 MoE。

### 目标指标

在 Mac M2 笔记本电脑上，4 层、4 头、d_model=128 的 GPT 在 `tinyshakespeare.txt` 上训练 2,000 步：

- 训练损失在约 6 分钟内从 ~4.2（随机）收敛到 ~1.5。
- 采样输出看起来是莎士比亚形状：古词、换行、人名如 "ROMEO:" 涌现。
- 验证损失（保留的最后 10% 文本）紧密跟踪训练损失；在此大小/预算下无过拟合。

## 动手构建

本课使用 PyTorch。安装 `torch`（CPU 构建就可以）。参见 `code/main.py`。脚本处理：

- 如果缺失则下载 `tinyshakespeare.txt`（或读取本地副本）。
- 字节级字符分词器。
- 训练/验证按 90/10 分割。
- 在支持的硬件上使用 bf16 自动混合精度的训练循环。
- 训练完成后采样。

### 步骤 1：数据

```python
text = open("tinyshakespeare.txt").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda xs: "".join(itos[x] for x in xs)
```

65 个唯一字符。微小的词汇表。适合 4 字节 vocab_size。没有 BPE，没有分词器麻烦。

### 步骤 2：模型

参见 `code/main.py`。模块是第 05 课的教科书式——pre-norm、RMSNorm、SwiGLU、因果 MHA。4/4/128 的参数数量：~800K。

### 步骤 3：训练循环

获取随机批次的 length-256 token 窗口。前向传播。偏移一位交叉熵。反向传播。AdamW 步进。记录。重复。

```python
for step in range(max_steps):
    x, y = get_batch("train")
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
```

### 步骤 4：采样

给定一个提示，重复前向传播，从 top-p logits 采样，追加，并继续。在 500 个 token 后停止。

### 步骤 5：阅读输出

2,000 步后：

```
ROMEO:
Away and mild will not thy friend, that thou shalt wit:
The chief that well shame and hath been his friends,
...
```

不是莎士比亚。但是莎士比亚形状。对于 ~800K 参数和笔记本电脑上的 6 分钟训练来说是一个明显的胜利。

## 使用它

这个毕业项目是一个参考架构。三个扩展以将其用于真正的工作：

1. **替换分词器。** 使用 BPE（例如 `tiktoken.get_encoding("cl100k_base")`）。词汇表大小从 65 跃升到 ~50,000。模型容量需要扩展以补偿。
2. **在更大的语料库上训练。** 使用 `OpenWebText` 或 `fineweb-edu`（HuggingFace）。在单张 A100 上，10B token 训练一个 125M 参数的 GPT 大约需要 24 小时。
3. **添加 RoPE + KV 缓存 + Flash Attention。** 下面的练习指导你完成每个。

这最终成为一个生成流畅英语的 125M 参数 GPT。不是前沿模型。但相同的代码路径——只是更大——正是 Karpathy、EleutherAI 和 Allen Institute 在 2026 年用来训练研究检查点的方法。

## 交付成果

参见 `outputs/skill-transformer-review.md`。该技能对照之前 13 课检查从零开始的 transformer 实现的正确性。

## 练习

1. **简单。** 运行 `code/main.py`。验证你训练的模型最终步骤验证损失低于 2.0。将 `max_steps` 从 2,000 改为 5,000——验证损失是否继续改善？
2. **中等。** 用 RoPE 替换可学习的位置嵌入。在 `MultiHeadAttention` 内部对 Q 和 K 应用旋转。训练并验证验证损失至少一样低。
3. **中等。** 在采样循环中实现 KV 缓存。在有缓存和无缓存的情况下生成 500 个 token。挂钟时间在笔记本电脑上应提高 5–20 倍。
4. **困难。** 向模型添加第二个头，预测下一个+一个 token（MTP — 多 token 预测，来自 DeepSeek-V3）。联合训练。有帮助吗？
5. **困难。** 将每个模块的单个 FFN 替换为 4 个专家的 MoE。路由器 + top-2 路由。看在匹配活跃参数下验证损失如何变化。

## 关键术语

| 术语 | 人们怎么说 | 它实际上的含义 |
|------|-----------------|-----------------------|
| nanoGPT | "Karpathy 的教程仓库" | 最小的仅解码器 transformer 训练代码，约 300 行；规范参考。 |
| tinyshakespeare | "标准玩具语料库" | ~1.1 MB 文本；自 2015 年以来每个字符 LM 教程都使用它。 |
| 绑定嵌入 | "共享输入/输出矩阵" | LM head 权重 = token 嵌入矩阵的转置；节省参数，提高质量。 |
| bf16 自动混合精度 | "训练精度技巧" | 在 bf16 中运行前向/反向传播，将优化器状态保持在 fp32 中；自 2021 年起为标准做法。 |
| 梯度裁剪 | "阻止激增" | 将全局梯度范数限制在 1.0；防止训练爆炸。 |
| 余弦 LR 调度 | "2020+ 默认方案" | LR 线性上升（热身），然后余弦衰减到峰值的 10%。 |
| MFU | "模型 FLOP 利用率" | 实现的 FLOPs / 理论峰值；在 2026 年，密集模型 40%、MoE 30% 是强劲的。 |
| 验证损失 | "保留损失" | 模型从未见过的数据上的交叉熵；过拟合检测器。 |

## 延伸阅读

- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) — 经典注释实现。
