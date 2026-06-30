# 完整 Transformer — 编码器 + 解码器

> 注意力是明星。其他所有东西——残差连接、归一化、前馈网络、交叉注意力——是让你能够将其堆叠得很深的脚手架。

**类型：** 构建
**语言：** Python
**前置条件：** 第七阶段 · 02（自注意力），第七阶段 · 03（多头注意力），第七阶段 · 04（位置编码）
**时间：** 约 75 分钟

## 问题

单层注意力是特征提取器，不是模型。每层一次矩阵乘法对于语言来说不够容量。你需要深度——而深度在没有正确管道的情况下会崩溃。

2017 年 Vaswani 论文打包了六项设计决策，将一层注意力变成了可堆叠的模块。之后的每一个 transformer——仅编码器（BERT）、仅解码器（GPT）、编码器-解码器（T5）——都继承了相同的骨架。到 2026 年，这些模块已被精炼（RMSNorm、SwiGLU、pre-norm、RoPE），但骨架是相同的。

本课是骨架。后续课程将其专门化——06 用于编码器，07 用于解码器，08 用于编码器-解码器。

## 概念

![编码器和解码器模块内部结构，已接线](../assets/full-transformer.svg)

### 六个组件

1. **嵌入 + 位置信号。** Token → 向量。通过 RoPE（现代）或正弦（经典）注入位置。
2. **自注意力。** 每个位置关注所有其他位置。在解码器中进行掩码。
3. **前馈网络（FFN）。** 逐位置的两层 MLP：`W_2 · activation(W_1 · x)`。默认扩展比为 4×。
4. **残差连接。** `x + sublayer(x)`。没有它，梯度在大约 6 层后就消失了。
5. **层归一化。** `LayerNorm` 或 `RMSNorm`（现代）。稳定残差流。
6. **交叉注意力（仅解码器）。** Queries 来自解码器，keys 和 values 来自编码器输出。

观察一个向量流过一个模块：注意力在位置之间混合，残差将其向前传递，FFN 对其进行变换，norm 使流保持稳定。

```figure
transformer-block
```

### 编码器模块（BERT、T5 编码器使用）

```
x → LN → MHA(self) → + → LN → FFN → + → out
                     ^              ^
                     |              |
                     └── residual ──┘
```

编码器是双向的。没有掩码。所有位置看到所有位置。

### 解码器模块（GPT、T5 解码器使用）

```
x → LN → MHA(masked self) → + → LN → MHA(cross to encoder) → + → LN → FFN → + → out
```

解码器每个模块有三个子层。中间那个——交叉注意力——是信息从编码器流向解码器的唯一位置。在纯仅解码器架构（GPT）中，交叉注意力被省略，你只有掩码自注意力 + FFN。

### Pre-norm 与 post-norm

原始论文：`x + sublayer(LN(x))` vs `LN(x + sublayer(x))`。Post-norm 在 2019 年左右失宠——没有仔细的热身很难训练得更深。Pre-norm（`LN` 在子层*之前*）是 2026 年的默认方案：Llama、Qwen、GPT-3+、Mistral 都使用它。

### 2026 年现代化模块

Vaswani 2017 提供了 LayerNorm + ReLU。现代技术栈替换了两者。生产模块实际的样子：

| 组件 | 2017 | 2026 |
|-----------|------|------|
| 归一化 | LayerNorm | RMSNorm |
| FFN 激活 | ReLU | SwiGLU |
| FFN 扩展 | 4× | 2.6×（SwiGLU 使用三个矩阵，总参数匹配） |
| 位置 | 正弦绝对 | RoPE |
| 注意力 | 完整 MHA | GQA（或 MLA） |
| 偏置项 | 是 | 否 |

RMSNorm 去掉了 LayerNorm 的均值中心化（少了一次减法），节省了计算量，并且在经验上至少同样稳定。SwiGLU（`Swish(W1 x) ⊙ W3 x`）在 Llama、PaLM 和 Qwen 论文中始终比 ReLU/GELU FFN 好约 0.5 点困惑度。

### 参数计数

对于具有 `d_model = d` 和 FFN 扩展 `r` 的一个模块：

- MHA：`4 · d²`（Q、K、V、O 投影）
- FFN (SwiGLU)：`3 · d · (r · d)` ≈ `3rd²`
- Norms：可忽略不计

在 `d = 4096, r = 2.6, layers = 32`（大致为 Llama 3 8B），总计：`32 · (4·4096² + 3·2.6·4096²) ≈ 32 · (16 + 32) M = ~1.5B 每层参数 × 32 ≈ 7B`（加上嵌入和 head）。与公布的计数匹配。

## 动手构建

### 步骤 1：构建模块

使用第 03 课的微型 `Matrix` 类（为保持独立性而复制到此文件）：

- `layer_norm(x, eps=1e-5)` — 减去均值，除以标准差。
- `rms_norm(x, eps=1e-6)` — 除以 RMS。不减去均值。
- `gelu(x)` 和 `silu(x) * W3 x`（SwiGLU）。
- `ffn_swiglu(x, W1, W2, W3)`。
- `encoder_block(x, params)` 和 `decoder_block(x, enc_out, params)`。

完整接线参见 `code/main.py`。

### 步骤 2：连接一个 2 层编码器和一个 2 层解码器

堆叠它们。将编码器输出传递到每个解码器交叉注意力中。在输出投影之前添加最终的 LN。

```python
def encode(tokens, params):
    x = embed(tokens, params.emb) + sinusoidal(len(tokens), params.d)
    for block in params.encoder_blocks:
        x = encoder_block(x, block)
    return x

def decode(target_tokens, encoder_out, params):
    x = embed(target_tokens, params.emb) + sinusoidal(len(target_tokens), params.d)
    for block in params.decoder_blocks:
        x = decoder_block(x, encoder_out, block)
    return x
```

### 步骤 3：在玩具示例上运行前向传播

输入一个 6-token 源和一个 5-token 目标。验证输出形状是 `(5, vocab)`。没有训练——本课是关于架构，而不是损失。

### 步骤 4：替换为 RMSNorm + SwiGLU

用 RMSNorm 和 SwiGLU 替换 LayerNorm 和 ReLU-FFN。确认形状仍然匹配。这就是通过一次函数替换实现的 2026 年现代化。

## 使用它

PyTorch/TF 参考实现：`nn.TransformerEncoderLayer`、`nn.TransformerDecoderLayer`。但大多数 2026 年生产代码自己编写模块，因为：

- Flash Attention 是在注意力内部调用的，不是通过 `nn.MultiheadAttention`。
- GQA / MLA 不在标准库参考中。
- RoPE、RMSNorm、SwiGLU 不是 PyTorch 的默认值。

HF `transformers` 有干净的参考模块你应该阅读：`modeling_llama.py` 是 2026 年规范的仅解码器模块。大约 500 行代码，值得通读一遍。

**编码器 vs 解码器 vs 编码器-解码器 — 何时选择：**

| 需求 | 选择 | 示例 |
|------|------|---------|
| 分类、嵌入、文本问答 | 仅编码器 | BERT、DeBERTa、ModernBERT |
| 文本生成、聊天、代码、推理 | 仅解码器 | GPT、Llama、Claude、Qwen |
| 结构化输入 → 结构化输出（翻译、摘要） | 编码器-解码器 | T5、BART、Whisper |

仅解码器在语言上胜出，因为它扩展最干净，并且同时处理理解和生成。当输入具有明确的"源序列"标识时（翻译、语音识别、结构化任务），编码器-解码器仍然是最好的。

## 交付成果

参见 `outputs/skill-transformer-block-reviewer.md`。该技能对照 2026 年默认值审查新的 transformer 模块实现，并标记缺失的部分（pre-norm、RoPE、RMSNorm、GQA、FFN 扩展比）。

## 练习

1. **简单。** 计算你的 encoder_block 在 `d_model=512, n_heads=8, ffn_expansion=4, swiglu=True` 时的参数数量。通过实现模块并使用 `sum(p.numel() for p in block.parameters())` 进行验证。
2. **中等。** 从 post-norm 切换到 pre-norm。初始化两者，并测量在随机输入上堆叠 12 层后的激活范数。Post-norm 的激活应该爆炸；pre-norm 的应该保持有界。
3. **困难。** 在玩具复制任务（复制 `x` 的反转）上实现一个 4 层编码器-解码器。训练 100 步。报告损失。替换为 RMSNorm + SwiGLU + RoPE——损失是否下降？

## 关键术语

| 术语 | 人们怎么说 | 它实际上的含义 |
|------|-----------------|-----------------------|
| 模块 | "一个 transformer 层" | 由 norm + attention + norm + FFN 组成的堆叠，包裹在残差连接中。 |
| 残差 | "跳跃连接" | `x + f(x)` 输出；使得梯度能够流经深层堆叠。 |
| Pre-norm | "在前面归一化，而不是后面" | 现代：`x + sublayer(LN(x))`。无需复杂的热身即可更深度地训练。 |
| RMSNorm | "没有均值的 LayerNorm" | 除以 RMS；少一次运算，相同的经验稳定性。 |
| SwiGLU | "每个人切换到的 FFN" | `Swish(W1 x) ⊙ W3 x → W2`。在 LM 困惑度上击败 ReLU/GELU。 |
| 交叉注意力 | "解码器如何看到编码器" | 使用解码器的 Q 和编码器输出的 K/V 的 MHA。 |
| FFN 扩展 | "中间 MLP 有多宽" | 隐藏大小与 d_model 的比率，通常为 4（LayerNorm）或 2.6（SwiGLU）。 |
| 无偏置 | "去掉 +b 项" | 现代技术栈在线性层中省略偏置；略微改善困惑度，更小的模型。 |

## 延伸阅读

- [Vaswani 等人 (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 原始模块规范。
- [Xiong 等人 (2020). On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) — 为什么 pre-norm 在深层上优于 post-norm。
- [Zhang、Sennrich (2019). Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) — RMSNorm。
- [Shazeer (2020). GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) — SwiGLU 论文。
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — 规范的 2026 年仅解码器模块。
