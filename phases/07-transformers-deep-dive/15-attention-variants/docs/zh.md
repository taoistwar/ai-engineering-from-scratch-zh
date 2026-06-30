# 注意力变体 — 滑动窗口、稀疏、差分

> 完整注意力是一个圆。每个 token 看到每个 token，内存为此付出代价。四种变体弯曲了圆的形状并收复了一半的成本。

**类型：** 构建
**语言：** Python
**前置条件：** 第七阶段 · 02（自注意力），第七阶段 · 03（多头），第七阶段 · 12（KV 缓存 / Flash Attention）
**时间：** 约 60 分钟

## 问题

完整注意力在序列长度上花费 `O(N²)` 内存和 `O(N²)` 计算。对于一个 128K 上下文的 Llama 3 70B，每层 160 亿个注意力条目，乘以 80 层。Flash Attention（第 12 课）隐藏了 `O(N²)` 激活内存，但不改变算术成本——每个 token 仍然关注每个其他 token。

三类变体改变了注意力矩阵本身的拓扑结构：

1. **滑动窗口注意力（SWA）。** 每个 token 关注一个固定的邻居窗口，而不是完整前缀。内存和计算降到 `O(N · W)`，其中 `W` 是窗口大小。Gemma 2/3、Mistral 7B 的第一层、Phi-3-Long。
2. **稀疏 / 块注意力。** 只有选定的 `(i, j)` 对被评分；其余被强制为零权重。Longformer、BigBird、OpenAI 稀疏 transformer。
3. **差分注意力。** 使用单独的 Q/K 投影计算两个注意力图，一个减去另一个。消除将权重倾泻到前几个 token 的"注意力汇"。微软的 DIFF Transformer（2024）。

这些共存。2026 年的前沿模型经常混合使用它们：大多数层是 SWA-1024，每五层是全局完整注意力，少数是清理检索的差分层。Gemma 3 的 5:1 SWA 与全局比率是当前教科书默认值。

## 概念

### 滑动窗口注意力（SWA）

位置 `i` 的每个查询只关注 `[i - W, i]`（因果 SWA）或 `[i - W/2, i + W/2]`（双向）中的位置。窗口外的 token 在分数矩阵中获得 `-inf`。

```
完整因果：              滑动窗口（W=4）：
位置 0-7                 位置 0-7, W=4
    0 1 2 3 4 5 6 7        0 1 2 3 4 5 6 7
0 | x                0 |  x
1 | x x              1 |  x x
2 | x x x            2 |  x x x
3 | x x x x          3 |  x x x x
4 | x x x x x        4 |    x x x x
5 | x x x x x x      5 |      x x x x
6 | x x x x x x x    6 |        x x x x
7 | x x x x x x x x  7 |          x x x x
```

对于 `N = 8192` 和 `W = 1024`，分数矩阵预期有 1024 × 8192 个非零行——减少了 8 倍。

**KV 缓存随 SWA 缩小。** 每层只需要保留最后 `W` 个 token 的 K 和 V。对于 Gemma-3 风格的配置（1024 窗口，128K 上下文），KV 缓存减少 128 倍。

**质量成本。** 纯 SWA transformer 在长距离检索上有困难。修复方法：将 SWA 层与完整注意力层交错。Gemma 3 使用 5:1 SWA:全局。Mistral 7B 使用了因果 SWA 堆叠，信息通过重叠窗口"向前流动"——每层将有效感受野扩展 `W`，经过 `L` 层后模型可以关注回 `L × W` 个 token。

### 稀疏 / 块注意力

提前选择 `N × N` 稀疏模式。三种规范形状：

- **局部 + 步长（OpenAI 稀疏 transformer）。** 关注最后 `W` 个 token 加上在此之前每 `stride` 个 token。以 `O(N · sqrt(N))` 的计算量捕获局部和长距离。
- **Longformer / BigBird。** 局部窗口 + 一小部分全局 token（例如 `[CLS]`），它们关注所有人并被所有人关注 + 随机稀疏链接。在匹配质量下经验性 2× 上下文。
- **原生稀疏注意力（DeepSeek，2025）。** 学习哪些 `(Q, K)` 块重要；在内核级别跳过零块。兼容 FlashAttention。

稀疏注意力是一个内核工程故事。数学很简单（掩码分数矩阵）；胜利来自从不将零条目加载到 SRAM 中。FlashAttention-3 和 2026 年的 FlexAttention API 使自定义稀疏模式成为 PyTorch 的一等公民。

### 差分注意力（DIFF Transformer，2024）

常规注意力有一个"注意力汇"问题：softmax 强制每行求和为 1，所以不想关注任何特定内容的 token 将权重倾泻到第一个 token（或前几个）。这窃取了本应流给真实内容的容量。

差分注意力通过计算**两个**注意力图并相减来修复此问题：

```
A1 = softmax(Q1 K1^T / √d)
A2 = softmax(Q2 K2^T / √d)
DiffAttn = (A1 - λ · A2) V
```

其中 `λ` 是一个学习的标量（通常 0.5–0.8）。A1 捕获真实内容权重；A2 捕获汇。减法抵消了汇，将权重重新分配给相关 token。

报告结果（微软 2024）：5–10% 更低的困惑度，在相同训练长度下 1.5–2× 更长的有效上下文，更清晰的针在干草堆检索。

### 变体比较

| 变体 | 计算量 | KV 缓存 | 质量 vs 完整 | 生产使用 |
|---------|---------|----------|-----------------|----------------|
| 完整注意力 | O(N²) | 每层 O(N) | 基线 | 每个模型的默认层 |
| SWA（窗口 1024） | O(N·W) | 每层 O(W) | -0.1 ppl，配合全局层良好 | Gemma 2/3、Phi-3-Long |
| 局部 + 步长稀疏 | O(N·√N) | 混合 | 类似 SWA | OpenAI 稀疏 transformer、Longformer |
| BigBird（局部 + 全局 + 随机） | 约 O(N) | 混合 | 在 2× 上下文时匹配完整 | 早期长上下文 BERT |
| 原生稀疏（DeepSeek-V3.2） | O(N · 活跃分数) | O(N) | 在 0.05 ppl 以内 | DeepSeek-V3.2，2025 |
| 差分 | O(2·N²) | O(2N) | -5 到 -10% ppl | DIFF Transformer，2026 年初模型 |

```figure
gqa-kv-sharing
```

## 动手构建

参见 `code/main.py`。我们实现一个因果掩码比较器，在玩具序列上并排显示完整、SWA、局部+步长和差分注意力。

### 步骤 1：完整因果掩码（基线）

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

第 07 课的基线。下三角；对角线以上为零权重。

### 步骤 2：滑动窗口因果掩码

```python
def swa_mask(n, window):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
    return M
```

一个参数 — `window`。对于 `window >= n`，你恢复到完整因果注意力。对于 `window = 1`，每个 token 只关注自己。

### 步骤 3：局部 + 步长稀疏掩码

```python
def strided_mask(n, window, stride):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
        for j in range(0, i + 1, stride):
            M[i][j] = 0.0
    return M
```

密集局部窗口加上每隔 `stride` 个 token 返回到序列开头。感受野随着额外层以对数步长增长。

### 步骤 4：差分注意力

```python
def diff_attention(Q1, K1, Q2, K2, V, lam):
    A1 = softmax_causal(Q1 @ K1.T / sqrt_d)
    A2 = softmax_causal(Q2 @ K2.T / sqrt_d)
    return (A1 - lam * A2) @ V
```

两次注意力传递，用学习的混合系数减去。在代码中，我们比较单一 vs 差分的注意力汇热力图，并观察汇的崩溃。

### 步骤 5：KV 缓存大小

打印在 `N = 131072` 时每种变体的每层缓存大小。SWA 和稀疏变体下降 10–100 倍。差分加倍。有意识地支付你的内存账单。

## 使用它

2026 年生产模式：

```python
from transformers import AutoModelForCausalLM
# Gemma 3 以 5:1 混合 SWA（window=1024）和全局层。
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-27b-it")
# print(model.config.sliding_window, model.config.layer_types)
```

PyTorch 2.5+ 中的 FlexAttention 接受掩码函数：

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def swa_pattern(b, h, q_idx, kv_idx):
    return (q_idx - kv_idx < 1024) & (q_idx >= kv_idx)

mask = create_block_mask(swa_pattern, B=batch, H=heads, Q_LEN=n, KV_LEN=n)
out = flex_attention(q, k, v, block_mask=mask)
```

这会编译为自定义 Triton 内核。对于常见模式，速度在 FlashAttention-3 的 10% 以内，且掩码函数是一个 Python 可调用对象。

**何时选择每种：**

- **纯完整注意力** — 每个层最多 ~16K 上下文，或当检索质量至关重要时。
- **SWA + 全局混合** — 长上下文（>32K），训练和推理内存受限。2026 年 32K 以上的默认方案。
- **稀疏块注意力** — 自定义内核，自定义模式。预留给专门的工作负载（检索、音频）。
- **差分注意力** — 任何注意力汇污染有害的工作负载（长上下文 RAG、针在干草堆）。

## 交付成果

参见 `outputs/skill-attention-variant-picker.md`。该技能根据目标上下文长度、检索需求和训练/推理计算概况，为新的模型选择注意力拓扑。

## 练习

1. **简单。** 运行 `code/main.py`。验证 SWA 在 `window=4` 时每行最后 4 个 token 之外全是零。验证 `window=n` 位对位地重现完整因果注意力。
2. **中等。** 在第 07 课的毕业项目之上实现 `window=1024` 的因果 SWA。在 tinyshakespeare 上训练 1,000 步。验证损失相比完整注意力下降多少？峰值内存下降多少？
3. **困难。** 在毕业项目模型中实现 Gemma-3 风格的 5:1 层混合（5 SWA，1 全局）。在匹配参数下比较损失、内存和生成质量与纯 SWA 和纯全局基线。
4. **困难。** 实现每个头部具有学习 `λ` 的差分注意力。在合成检索任务（一根针，2,000 个干扰项）上训练。在匹配参数下测量检索准确率与单一注意力基线。

## 关键术语

| 术语 | 人们怎么说 | 它实际上的含义 |
|------|-----------------|-----------------------|
| 滑动窗口注意力（SWA） | "局部注意力" | 每个查询关注其最后 `W` 个 token；KV 缓存缩小到 `O(W)`。 |
| 有效感受野 | "模型往后看多远" | 在具有窗口 `W` 的 `L` 层 SWA 堆叠中，最多 `L × W` 个 token。 |
| Longformer / BigBird | "局部 + 全局 + 随机" | 带有少数始终关注的全局 token 的稀疏模式；早期长上下文方法。 |
| 原生稀疏注意力 | "DeepSeek 的内核技巧" | 学习块级稀疏性；在内核级别跳过零块同时保持质量。 |
| 差分注意力 | "两个图，一个减去" | DIFF Transformer：从第一个注意力图中减去学习的 `λ` 乘以第二个以抵消注意力汇。 |
| 注意力汇 | "权重倾泻到 token 0" | Softmax 归一化强制行求和为 1；无信息的查询将权重倾泻到位置 0。 |
| FlexAttention | "掩码即 Python" | PyTorch 2.5+ API，将任意掩码函数编译为 FlashAttention 形状的内核。 |
| 层类型混合 | "5:1 SWA 与全局" | 在堆叠中交错稀疏和完整注意力层，以较低内存保持质量。 |

## 延伸阅读

- [Beltagy、Peters、Cohan (2020). Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) — 规范的滑动窗口 + 全局 token 论文。
- [Zaheer 等人 (2020). Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) — 局部 + 全局 + 随机。
- [Child 等人 (2019). Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509) — OpenAI 的局部+步长模式。
- [Gemma Team (2024). Gemma 2: Improving Open Language Models at a Practical Size](https://arxiv.org/abs/2408.00118) — 1:1 SWA:全局混合。
- [Gemma Team (2025). Gemma 3 technical report](https://arxiv.org/abs/2503.19786) — 带有 window=1024 的 5:1 混合，现在是教科书默认方案。
- [Ye 等人 (2024). Differential Transformer](https://arxiv.org/abs/2410.05258) — DIFF Transformer 论文。
- [Yuan 等人 (2025). Native Sparse Attention](https://arxiv.org/abs/2502.11089) — DeepSeek-V3.2 的学习稀疏性注意力。
- [PyTorch — FlexAttention 博客和文档](https://pytorch.org/blog/flexattention/) — "使用它"部分中掩码即可调用的模式的 API 参考。
