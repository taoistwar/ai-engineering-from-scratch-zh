# 差分注意力（V2）

> Softmax 注意力将少量概率分散到每个不匹配的 token 上。在 100k token 上，这些噪声累积起来会淹没信号。差分 Transformer（Ye 等人，ICLR 2025）通过将注意力计算为两个 softmax 之差，减去共享的噪声基底来解决这个问题。DIFF V2（Microsoft，2026 年 1 月）是生产栈重写版：匹配基准 Transformer 的解码延迟，无需自定义 kernel，与 FlashAttention 兼容。本课是从 V1 到 V2 的端到端讲解，包含一个你可以在 stdlib Python 中运行的可工作玩具实现。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 7 · 02 (self-attention), Phase 7 · 15 (attention variants), Phase 10 · 14 (architecture walkthrough)
**Time:** ~60 minutes

## 学习目标

- 精确说明为什么 softmax 注意力有一个噪声基底，以及为什么它会随着上下文长度增长而增长。
- 推导差分注意力公式，并解释为什么减去操作能够消除共享噪声成分同时保留信号。
- 走过 V1 到 V2 的差异：什么变快了、什么变简单了、什么变稳定了，以及每个变化为什么是生产预训练所必需的。
- 在纯 Python 中从零实现差分注意力，并实验验证合成信号加噪声查询上的噪声消除特性。

## 问题

标准 softmax 注意力有一个数学属性，在大规模下变成了操作上的头痛。对于查询 `q`，注意力权重是 `softmax(qK^T / sqrt(d))`。Softmax 永远不能产生精确的零——每个不匹配的 token 获得一些正的质量。这些残余质量是噪声，并且随上下文长度增长而增长。在 128k token 上，即使每个不匹配 token 只获得 0.001% 的概率，其中 127,999 个 token 加起来贡献约 12% 的总量。模型必须学会绕过这个随上下文增长的噪声基底。

从经验上看，这表现为注意力头干扰：长上下文 RAG 中幻觉化的引用、100k token 检索任务中的 lost-in-the-middle 失败，以及超过 32k 的 needle-in-haystack 基准上的微妙精度下降。差分 Transformer 论文（arXiv:2410.05258，ICLR 2025）测量了这个差距：DIFF Transformer 在相同大小的基线上取得了更低的困惑度、更高的长上下文精度和更少的幻觉。

DIFF V1 有三个问题使其无法进入前沿预训练流水线。它的值缓存必须在每个解码步骤加载两次，需要自定义 CUDA kernel 破坏了 FlashAttention 兼容性，并且其每头 RMSNorm 在 70B+ 规模上会破坏长期训练的稳定性。DIFF V2（Microsoft unilm blog，2026 年 1 月 20 日）修复了这三个问题。本课讲解两个版本，构建差值操作符，并在玩具查询上基准测试噪声消除。

## 概念

### softmax 的噪声基底

对于查询 `q` 和键 `K = [k_1, ..., k_N]`，注意力权重为：

```
w_i = exp(q . k_i / sqrt(d)) / sum_j exp(q . k_j / sqrt(d))
```

没有 `w_i` 会是零。如果 `k_i` 与 `q` 完全无关，得分 `q . k_i` 不是 0——它在零附近波动，方差为 `||q||^2 / d`。在 softmax 归一化后，每个无关 token 仍然贡献 `O(1/N)` 的加权和。无关 token 的总贡献是 `O((N-1)/N) = O(1)`——不是一个小的量。

模型想要的是类似硬 top-k 的东西：匹配 token 上权重高，其他地方接近零。Softmax 太光滑了，无法直接做到这一点。

### 差分思想

将每个头的 Q 和 K 投影分成两个：Q = (Q_1, Q_2) 和 K = (K_1, K_2)。计算两个注意力图：

```
A_1 = softmax(Q_1 K_1^T / sqrt(d))
A_2 = softmax(Q_2 K_2^T / sqrt(d))
```

输出：

```
DiffAttn = (A_1 - lambda * A_2) V
```

相减操作消除了两个图共享的任何噪声分布。如果两个图在 127k 无关 token 上具有大致均匀的权重（在随机初始化时确实如此），这些权重会抵消。信号——在少数实际相关 token 上集中的权重——只有当它同时出现在两个图中相同数量级时才会抵消，而一旦模型训练后就不会出现这种情况。

`lambda` 是每个头可学习的标量，参数化为 `lambda = exp(lambda_q1 dot lambda_k1) - exp(lambda_q2 dot lambda_k2) + lambda_init`。它可以为负值。`lambda_init` 默认为一个小正数，如 0.8。

### 为什么这匹配头噪声消除

想象两个有噪声的麦克风录制同一个声音。两者都拾取说话者加上相关的背景噪声。将一个从另一个中减去，共享的噪声就消失了。声音存活下来，因为两个信号的相位或幅度差异足够大，阻止了完全抵消。每个头的 `lambda` 精确学习了这种平衡。

### V1 vs V2：差异

V1 保持参数数量与基线 Transformer 相等。为了每头获得两个查询，它将头维度减半。这牺牲了头表达能力，并且——更痛苦的是——每头将值缓存减半。解码必须每步加载值缓存两次（每个 softmax 分支一次）。结果：尽管参数量匹配，解码比基线慢。

V2 将查询头的数量加倍并保持 KV 头不变（从上投影借用参数）。头维度与基线相同。相减后，额外的维度被投影回去以匹配基线 Transformer 的 O_W 投影。三件事同时发生：

1. 解码速度匹配基线（KV 缓存加载一次）。
2. FlashAttention 原样运行（无需自定义 kernel）。
3. 解码时的算术强度提高（从 HBM 每字节加载更多计算）。

V2 还移除了 V1 用于稳定减法操作的每头 RMSNorm。在 70B 级预训练规模上，该 RMSNorm 会破坏后期训练的稳定性。V2 用一个更简单的初始化方案替代它，在没有额外模块的情况下保持训练稳定。

### 何时使用

| 工作负载 | 收益 |
|----------|---------|
| 长上下文 RAG（64k+） | 更清晰的注意力图，更少幻觉引用 |
| Needle-in-haystack 基准 | 32k 以上精度大幅提升 |
| 多文档 QA | 减少跨文档干扰 |
| 8k 代码补全 | 边际收益，不值得构架变更 |
| 短对话（< 4k） | 与基线基本无差异 |

价值随上下文长度增长。在 4k token 上，噪声基底足够小，标准注意力就已足够。在 128k 上，它在伤害你。

### 与其他 2026 年旋钮的兼容性

| 特性 | 与 DIFF V2 兼容？ |
|---------|------------------------|
| GQA | 是（V2 增加 Q 头数，不增加 KV 头数） |
| MLA（DeepSeek） | 原则上是，尚无已发表的组合论文 |
| MoE | 是（注意力独立于 MLP 块） |
| RoPE | 是（无变化） |
| YaRN / 长上下文缩放 | 是（正是 DIFF 帮助最大的场景） |
| FlashAttention | V2 中是（V1 中否） |
| 推测解码 | 是（注意力变化对 spec-decode 循环透明） |

```figure
differential-attention
```

## 构建它

`code/main.py` 在纯 Python 中实现差分注意力。一个具有已知信号加噪声结构的玩具查询让你可以直接测量噪声消除比率。

### 步骤 1：标准 softmax 注意力

标准库矩阵操作：列表的列表，手动 matmul，使用减去最大值的数值稳定 softmax。

```python
def softmax(row):
    m = max(row)
    exps = [math.exp(x - m) for x in row]
    s = sum(exps)
    return [e / s for e in exps]
```

### 步骤 2：将 Q、K 分成两半

V1 风格：将头维度减半。V2 风格：保持头维度并将头数加倍。玩具实现使用 V1 以求教学清晰——数学相同，只有记账不同。

### 步骤 3：两个 softmax 分支 + 相减

```python
A1 = [softmax([dot(q1, k) / scale for k in K1]) for q1 in Q1]
A2 = [softmax([dot(q2, k) / scale for k in K2]) for q2 in Q2]
diff_weights = [[a1 - lam * a2 for a1, a2 in zip(r1, r2)] for r1, r2 in zip(A1, A2)]
out = [[sum(w * v[j] for w, v in zip(row, V)) for j in range(d_v)] for row in diff_weights]
```

注意：输出权重可以是负数。这没问题——值缓存仍然能处理有符号的贡献。后续的 V 投影吸收符号。

### 步骤 4：噪声消除测量

构建一个长度为 1024 的合成序列。将信号 token 放在已知位置，其余填充噪声。计算 (a) 信号位置上的标准 softmax 注意力权重和 (b) 差分注意力权重。测量每个的信号噪声比。根据两个分支被训练得差异的程度，DIFF 注意力可靠地产生高出 3 倍到 10 倍的信号噪声比。

### 步骤 5：V1 vs V2 参数统计

给定一个配置（hidden=4096, heads=32, d_head=128），打印：

- 基线 Transformer：Q、K、V 每个大小 `hidden * hidden`，MLP 为 4 * hidden。
- DIFF V1：Q、K 每个大小 `hidden * hidden`，V 大小 `hidden * hidden`（不变），内部头维度减半。添加每头 `lambda` 参数（O(heads * d_head)）。
- DIFF V2：Q 大小 `2 * hidden * hidden`，K 大小 `hidden * hidden`，V 大小 `hidden * hidden`。额外的维度在 O_W 之前被投影回去。添加相同的 `lambda` 参数。

玩具程序测量 V2 的额外参数成本（每个注意力块大约额外 `hidden * hidden`），并打印出来。

## 使用它

截至 2026 年 4 月，DIFF V2 尚未在每个生产推理服务器中搭载，但集成正在 vLLM 和 SGLang 中进行。同时，该模式出现在：

- Microsoft 内部长上下文生产模型。
- 多个目标 256k+ 上下文的开放模型训练运行中的研究复现。
- 在交替层上结合 DIFF 注意力和滑动窗口注意力的混合架构。

在 2026 年何时使用它：

- 从零训练一个目标 64k+ 有效上下文的新模型。从一开始就加入差分注意力；后期再训练代价高昂。
- 微调一个长上下文模型，其中 lost-in-the-middle 失败主导你的评估。在 Q 投影上的 LoRA 可以近似 DIFF 结构。

何时不用：

- 你正在部署一个具有稳定长上下文性能的预训练密集模型。对现有权重重新训练的成本很少能收回。
- 你的上下文始终在 16k 以下。噪声基底可以忽略。

## 产出

本课产出 `outputs/skill-diff-attention-integrator.md`。给定一个模型架构、目标上下文长度、幻觉概况和训练预算，它产生一个向新的预训练运行或 LoRA 微调中添加差分注意力的集成计划。

## 练习

1. 运行 `code/main.py`。验证差分注意力在合成查询上报告的信号噪声比高于标准 softmax 注意力。改变噪声幅度，展示标准注意力变得不可用的交叉点。

2. 为一个 7B 级别模型（hidden=4096, heads=32, d_head=128, 32 layers）计算从基线到 DIFF V1 以及从基线到 DIFF V2 的参数量增量。展示哪些组件增加了参数，哪些保持不变。

3. 阅读 DIFF V1 论文的第 3 节（arXiv:2410.05258）和 DIFF V2 Hugging Face blog 的第 2 节。用两句话解释为什么需要 V1 的每头 RMSNorm，以及为什么 V2 可以在不导致训练发散的情况下移除它。

4. 实现一项消融：用 `lambda = 0`（纯第一个 softmax）和 `lambda = 1`（完全相减）计算差分注意力。在合成查询上，测量信号噪声比在扫描中的变化。识别最大化信号噪声比的 `lambda`。

5. 将玩具扩展到 GQA + DIFF V2。选择 8 个 KV 头和 32 个 Q 头。展示 KV 缓存大小与具有相同（8, 32）配置的基线 GQA 模型匹配。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| 差分注意力 | "两个 softmax 相减" | 将 Q、K 分成两半，计算两个 softmax 图，用第一个减去第二个（按 lambda 缩放），然后乘以 V |
| 噪声基底 | "softmax 的非零尾部" | Softmax 在每个无关 token 上放置的 O(1/N) 权重，在长上下文中总和为 O(1) |
| lambda | "减法比例" | 每头可学习标量，参数化为 `exp(lq1.lk1) - exp(lq2.lk2) + lambda_init`；可以为负值 |
| DIFF V1 | "ICLR 2025 版本" | 原始差分 Transformer；将头维度减半以保持参数量，需要自定义 kernel，解码较慢 |
| DIFF V2 | "2026 年 1 月修正版" | 将 Q 头数加倍，KV 头数不变；匹配基线解码速度，可与 FlashAttention 配合 |
| 每头 RMSNorm | "V1 稳定器" | V1 在差分后应用的额外 norm；V2 移除了它以防止后期训练不稳定 |
| 信号噪声比 | "多少注意力被浪费了" | 真实信号位置上的权重与无关位置上平均权重的比率 |
| Lost in the middle | "长上下文失败模式" | 检索精度在长上下文中间文档上下降的经验现象——差分注意力减少了这种影响 |
| 算术强度 | "每字节加载的 FLOPs" | V2 在解码时通过每个 KV 加载加倍查询数来提高的比率；对内存受限的解码很重要 |

## 延伸阅读

- [Ye et al. — Differential Transformer (arXiv:2410.05258, ICLR 2025)](https://arxiv.org/abs/2410.05258) —— 原始论文，含噪声消除理论和长上下文消融
- [Microsoft unilm — Differential Transformer V2 (Hugging Face blog, January 2026)](https://huggingface.co/blog/microsoft/diff-attn-v2) —— 生产栈重写版，匹配基线解码速度，FlashAttention 兼容
- [Understanding Differential Transformer Unchains Pretrained Self-Attentions (arXiv:2505.16333)](https://arxiv.org/abs/2505.16333) —— 为什么相减能恢复预训练注意力结构的理论分析
- [Shared DIFF Transformer (arXiv:2501.17900)](https://arxiv.org/html/2501.17900) —— 参数共享变体
- [Vaswani et al. — Attention Is All You Need (arXiv:1706.03762)](https://arxiv.org/abs/1706.03762) —— DIFF 从中减去的基础 Transformer
- [Liu et al. — Lost in the Middle (arXiv:2307.03172)](https://arxiv.org/abs/2307.03172) —— DIFF 注意力所针对的长上下文基准
