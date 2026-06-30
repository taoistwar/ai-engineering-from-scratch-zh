# 位置编码 — 正弦、RoPE、ALiBi

> 注意力是置换不变的。"猫坐在垫子上"和"垫子上在坐猫"在没有位置信号的情况下产生相同的输出。三种算法修复了这个问题——每种算法对"位置"的含义有不同的赌注。

**类型：** 构建
**语言：** Python
**前置条件：** 第七阶段 · 02（自注意力），第七阶段 · 03（多头注意力）
**时间：** 约 45 分钟

## 问题

缩放点积注意力对顺序是无感的。注意力矩阵 `softmax(Q K^T / √d) V` 是通过成对相似度计算的。打乱 `X` 的行，输出的行也会以相同方式被打乱。注意力内部没有任何东西关心位置。

这在词袋模型中不是 bug。对于语言、代码、音频、视频——任何顺序承载含义的东西——这是致命的。

修复方法是以某种方式将位置注入到嵌入中。三个时代的答案：

1. **绝对正弦**（Vaswani 2017）。将 `sin/cos` 按位置加到嵌入上。简单、无需学习参数、超出训练长度时外推能力差。
2. **RoPE — 旋转位置嵌入**（Su 2021）。按与位置成正比的角度旋转 Q 和 K 向量。在点积中直接编码*相对*位置。2026 年占主导地位。
3. **ALiBi — 线性偏置注意力**（Press 2022）。完全跳过嵌入；根据距离向注意力分数添加每个头的线性惩罚。优秀的长序列外推能力。

截至 2026 年，几乎所有前沿开源模型都使用 RoPE：Llama 2/3/4、Qwen 2/3、Mistral、Mixtral、DeepSeek-V3、Kimi。少数长上下文模型使用 ALiBi 或其现代变体。绝对正弦已成为历史。

## 概念

![正弦绝对 vs RoPE 旋转 vs ALiBi 距离偏置](../assets/positional-encoding.svg)

### 绝对正弦

预计算一个形状为 `(max_len, d_model)` 的固定矩阵 `PE`：

```
PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))
```

然后在注意力之前 `X' = X + PE[:N]`。每个维度是不同频率的正弦波。模型学会从相位模式中读取位置。在 `max_len` 之外失败：当模型只看到位置 0–2047 时，没有告诉它位置 2048 会发生什么。

### RoPE

旋转 Q 和 K 向量（不是嵌入）。对于一对维度 `(2i, 2i+1)`：

```
[q'_2i    ]   [ cos(pos·θ_i)  -sin(pos·θ_i) ] [q_2i   ]
[q'_2i+1  ] = [ sin(pos·θ_i)   cos(pos·θ_i) ] [q_2i+1 ]

θ_i = base^(-2i / d_head),  base = 默认 10000
```

对 key 应用相同的旋转，位置为 `pos_k`。点积 `q'_m · k'_n` 变成仅关于 `(m - n)` 的函数。也就是说：**注意力分数仅依赖于相对距离**，尽管旋转是基于绝对位置的。美妙的技巧。

扩展 RoPE：可以缩放 `base`（NTK-aware、YaRN、LongRoPE）以在不重新训练的情况下外推到更长的上下文。Llama 3 就是这样将上下文从 8K 扩展到 128K 的。

### ALiBi

跳过嵌入技巧。直接偏置注意力分数：

```
attn_score[i, j] = (q_i · k_j) / √d  -  m_h · |i - j|
```

其中 `m_h` 是特定于头的斜率（例如 `1 / 2^(8·h/H)`）。更近的 token 获得增强；远的 token 受到惩罚。无训练时成本。论文显示长度外推能力超过正弦，并在其原始训练长度上与 RoPE 匹敌。

### 2026 年该选哪个

| 变体 | 外推能力 | 训练成本 | 使用方 |
|---------|---------------|---------------|---------|
| 绝对正弦 | 差 | 免费 | 原始 transformer、早期 BERT |
| 可学习绝对 | 无 | 微小 | GPT-2、GPT-3 |
| RoPE | 配合缩放良好 | 免费 | Llama 2/3/4、Qwen 2/3、Mistral、DeepSeek-V3、Kimi |
| RoPE + YaRN | 极好 | 微调阶段 | Qwen2-1M、Llama 3.1 128K |
| ALiBi | 极好 | 免费 | BLOOM、MPT、Baichuan |

RoPE 胜出是因为它在不改变架构的情况下嵌入注意力，编码相对位置，并且其 `base` 超参数为长上下文微调提供了一个干净的控制旋钮。

```figure
rope-explorer
```

## 动手构建

### 步骤 1：正弦编码

参见 `code/main.py`。一个 4 行的计算：

```python
def sinusoidal(N, d):
    pe = [[0.0] * d for _ in range(N)]
    for pos in range(N):
        for i in range(d // 2):
            theta = pos / (10000 ** (2 * i / d))
            pe[pos][2 * i]     = math.sin(theta)
            pe[pos][2 * i + 1] = math.cos(theta)
    return pe
```

在第一个注意力层之前将其加到嵌入矩阵上。

### 步骤 2：对 Q、K 应用 RoPE

RoPE 原地操作于 Q 和 K。对于每对维度：

```python
def apply_rope(x, pos, base=10000):
    d = len(x)
    out = list(x)
    for i in range(d // 2):
        theta = pos / (base ** (2 * i / d))
        c, s = math.cos(theta), math.sin(theta)
        a, b = x[2 * i], x[2 * i + 1]
        out[2 * i]     = a * c - b * s
        out[2 * i + 1] = a * s + b * c
    return out
```

关键：将相同的函数应用于位置 `m` 处的 Q 和位置 `n` 处的 K。它们的点积在每个坐标对上获得一个 `cos((m-n)·θ_i)` 因子。注意力免费获得了相对位置。

### 步骤 3：ALiBi 斜率和偏置

```python
def alibi_bias(n_heads, seq_len):
    # slope_h = 2 ** (-8 * h / n_heads) 对于 h = 1..n_heads
    slopes = [2 ** (-8 * (h + 1) / n_heads) for h in range(n_heads)]
    bias = []
    for m in slopes:
        row = [[-m * abs(i - j) for j in range(seq_len)] for i in range(seq_len)]
        bias.append(row)
    return bias  # 在 softmax 之前加到注意力分数上
```

将 `bias[h]` 加到头 `h` 的 `(seq_len, seq_len)` 注意力分数矩阵上，然后 softmax。

### 步骤 4：验证 RoPE 的相对距离性质

选择两个随机向量 `a, b`。用 `(pos_a, pos_b)` 旋转它们。然后用 `(pos_a + k, pos_b + k)` 旋转。两个点积必须在浮点误差范围内匹配。这个性质是 RoPE 的全部意义——它对绝对偏移是不变的，只有相对差距重要。

## 使用它

PyTorch 2.5+ 在 `torch.nn.functional` 中提供了 RoPE 实用程序。大多数生产代码使用 `flash_attn` 或 `xformers`，其中 RoPE 在注意力内核内部应用。

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("meta-llama/Llama-3.2-3B")
# model.config.rope_scaling → {"type": "yarn", "factor": 32.0, "original_max_position_embeddings": 8192}
```

**2026 年的长上下文技巧：**

- **NTK-aware 插值。** 当从 4K 扩展到 16K+ 时，将 `base` 重缩放为 `base * (scale_factor)^(d/(d-2))`。
- **YaRN。** 更智能的插值，在长上下文中保持注意力熵。Llama 3.1 128K 使用了它。
- **LongRoPE。** 微软 2024 年的方法，使用进化搜索为每个维度选择缩放因子。Phi-3-Long 使用了它。
- **位置插值 + 微调。** 只需按扩展因子收缩位置，并在 1–5B token 上进行微调。出人意料地有效。

## 交付成果

参见 `outputs/skill-positional-encoding-picker.md`。该技能根据目标上下文长度、外推需求和训练预算，为新的模型选择编码策略。

## 练习

1. **简单。** 将正弦 `PE` 矩阵绘制为 `max_len=512, d=128` 的热力图。确认"条纹随着维度索引增加而变宽"的模式。
2. **中等。** 实现 NTK-aware RoPE 缩放。在长度为 256 的序列上训练一个小型 LM，然后在长度为 1024 上测试，有缩放和没有缩放。测量困惑度。
3. **困难。** 在同一个注意力模块中实现 ALiBi 和 RoPE。在复制任务上训练一个 4 层 transformer，序列长度为 512。在测试时外推到 2048。比较退化程度。

## 关键术语

| 术语 | 人们怎么说 | 它实际上的含义 |
|------|-----------------|-----------------------|
| 位置编码 | "告诉注意力关于顺序" | 添加到嵌入或注意力中的任何信号，用于编码位置。 |
| 正弦 | "原始的那个" | 以几何频率将 `sin/cos` 加到嵌入上；不能外推。 |
| RoPE | "旋转嵌入" | 按与位置相关的角度旋转 Q、K；点积编码相对距离。 |
| ALiBi | "线性偏置技巧" | 将 `-m·|i-j|` 加到注意力分数上；无需嵌入，优秀的长度外推能力。 |
| base | "RoPE 的旋钮" | RoPE 中的频率缩放器；在推理时增加以扩展上下文。 |
| NTK-aware | "一种 RoPE 缩放技巧" | 重缩放 `base` 以使高频维度在上下文扩展时不被压缩。 |
| YaRN | "花哨的那个" | 保持注意力熵的逐维度插值+外推。 |
| 外推 | "在训练长度之外工作" | 位置方案能否在训练中见过的 `max_len` 之外提供正确的输出？ |

## 延伸阅读

- [Vaswani 等人 (2017). Attention Is All You Need §3.5](https://arxiv.org/abs/1706.03762) — 原始正弦。
- [Su 等人 (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — RoPE 论文。
- [Press、Smith、Lewis (2021). Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409) — ALiBi。
- [Peng 等人 (2023). YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) — 最先进的 RoPE 缩放。
- [Chen 等人 (2023). Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595) — Meta 的 Llama 2 长上下文论文。
- [Ding 等人 (2024). LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13753) — 微软方法，被 Phi-3-Long 使用并在"使用它"部分引用。
- [HuggingFace Transformers — `modeling_rope_utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py) — 每种 RoPE 缩放方案的生产级实现（default、linear、dynamic、YaRN、LongRoPE、Llama-3）。
