# Jamba — 混合 SSM-Transformer

> 状态空间模型（SSMs）和 transformer 各有所需。Transformer 以二次方成本通过注意力换取质量。SSM 通过递推以线性时间推理和恒定内存换取，但质量稍逊。AI21 的 Jamba（2024 年 3 月）和 Jamba 1.5（2024 年 8 月）将它们放进同一个模型：每 7 个 Mamba 层配 1 个 Transformer 层，每隔一个块使用 MoE，以及一个能放入单张 80GB GPU 的 256k 上下文窗口。Mamba-3（ICLR 2026）用复值状态空间和 MIMO 投影优化了 SSM 端。本课端到端阅读两种架构，并解释为什么混合配方在纯 SSM 和纯 Transformer 长上下文尝试都失败的情况下存活了三年的扩展。

**Type:** Learn
**Languages:** Python (stdlib, layer-mix calculator)
**Prerequisites:** Phase 10 · 14 (open-model architectures), Phase 10 · 17 (native sparse attention)
**Time:** ~60 minutes

## 学习目标

- 解释 Jamba 块中的三个原语——Transformer 层、Mamba 层、MoE——以及 1:7:even 的交错配方。
- 在高层次陈述 SSM 的递推长什么样，以及为什么它能实现恒定内存推理。
- 计算 Jamba 模型在 256k 上下文下的 KV 缓存占用，并与纯 Transformer 模型需要的做比较。
- 命名三个 Mamba-3 创新（指数梯形离散化、复值状态更新、MIMO）以及每个针对的问题。

## 问题

注意力是序列长度的二次方。状态空间模型是线性的。这个差异会放大：在 256k token 上，一个 Transformer 注意力图每个头是 65B 条目；一个 SSM 的递推状态是固定大小的，无论序列长度。

纯 SSM 模型（Mamba、Mamba-2）在小规模上匹配 Transformer 困惑度，但在状态追踪任务上落后，并在某些类别的上下文检索中失败。直觉：SSM 将历史压缩为固定状态，当历史很长时，信息会泄漏。注意力精确记忆一切，但付出二次方成本。

明显的修复：两者都用。将 Transformer 层放在需要精确记忆的地方。其他地方使用 SSM 层。调整比例。Jamba 是第一个生产级模型以规模化搭载这种混合配方（52B 总量，12B 激活，256k 上下文，单张 80GB GPU）。Jamba 1.5 将家族扩展到 398B 总量 / 94B 激活。Mamba-3（ICLR 2026）是当前最佳的纯 SSM 基线，混合模型可以围绕它重建。

本课阅读所有三篇论文，并产生"选择合适的比例"的思维模型。

## 概念

### 一页纸看懂 SSM

一个状态空间模型通过固定大小的状态 `h` 处理序列 `x_1, ..., x_N`：

```
h_t = A h_{t-1} + B x_t
y_t = C h_t
```

每步状态通过线性动力学 `A` 演化，接受输入 `B x_t`，并发出输出 `C h_t`。`A, B, C` 可以被学习。注意关键属性：计算 `y_t` 只需要 `h_{t-1}` 和 `x_t`，不需要任何更早的 `x`。内存是恒定的。推理是每个 token O(1)。

建模质量的诀窍是 `A` 的结构。S4（Gu 2021）使用了一个在训练时可以有效评估为长卷积的高度结构化矩阵。Mamba（Gu, Dao 2023）将固定的 `A, B, C` 替换为数据依赖的（"选择性"部分）。Mamba-2（2024）进一步简化了结构。Mamba-3（2026）在特定地方重新添加了复杂性。

关键属性：对于一个 decoder LLM，SSM 层是注意力层的即插即用替代，具有每层固定大小的状态而非增长的 KV 缓存。

### Jamba 块

一个 Jamba 块根据两个数字交错层：

- `l`：注意力与 Mamba 的比率。Jamba 使用 `l = 8`，意味着每 7 个 Mamba 层配 1 个 Transformer 层（每组 7 Mamba + 1 Attention = 8 层）。
- `e`：MoE 频率。Jamba 使用 `e = 2`，意味着每隔一层应用 MoE。

一个块内的层序列：

```
M  M  M  M  M  M  M  A    (7 Mamba + 1 Attention)
|  M  |  M  |  M  |  M    (其中 | 标记 MoE 应用处)
```

每个 Jamba 块是 8 层。在 4 个块深（总共 32 层）时，得到 28 个 Mamba 和 4 个 Attention 层。其中 16 个使用 MoE。

### 为什么是 1:7 比率

AI21 运行了消融：注意力与 Mamba 的什么比率能在其长上下文评估上提供最佳的每参数困惑度和上下文召回？

- 注意力过多（1:1）：质量上升但内存和速度下降。
- 注意力过少（1:15）：内存很好但上下文检索失败。
- 最佳点：1:7 或 1:8。

直觉：Transformer 层处理精确回忆和状态追踪。Mamba 层处理廉价的大宗处理。

### 位置编码

Mamba 层自身是位置感知的（通过递推）。原始基于 Mamba 的混合模型中的注意力层不使用 RoPE——SSM 层提供位置信息。Jamba 1.5 基于经验长上下文评估，向注意力层添加 RoPE 用于更长上下文的泛化，这是一个事后精炼。

### 内存预算

对于一个 Jamba-1 形状（32 层：28 Mamba + 4 Attention，hidden 4096，32 注意力头）：

- KV 缓存（仅注意力层）：`2 * 4 * 32 * 128 * 256k * 2 = 8.4 GB`，在 256k BF16 下。只有 4 个注意力层贡献。
- SSM 状态：每个 token 前缀 `28 * hidden * state_size`，但这是每个层固定的，不随序列长度增长。典型 Mamba 状态每特征 16，hidden 4096：`28 * 4096 * 16 * 2 = 3.7 MB` 总计。

与 32 层、相同 hidden、32 头全 MHA 的纯 Transformer 比较：`2 * 32 * 32 * 128 * 256k * 2 = 128 GB`，在 256k BF16 下。KV 缓存减少 8 倍。即使与大多数 2024 模型使用的 GQA(8) 基线对比（`2 * 32 * 8 * 128 * 256k * 2 = 32 GB`），Jamba 的 1:7 混合在 16 GB 时仍然小 2 倍。

这就是 AI21 所说的"256k 上下文在单张 80GB GPU 上"。全 MHA 纯 Transformer 的 KV 缓存放不下；即使是 GQA 基线也没有给权重和激活留下空间；Jamba 则可以。

### Mamba-3：2026 年纯 SSM 基线

Mamba-3（ICLR 2026，arXiv:2603.15569）在纯 SSM 端引入了三个创新：

1. **指数梯形离散化。** 将 Mamba-2 中的欧拉方法离散化替换为更具表现力的递推。在核心递推内对状态输入应用类卷积操作，而非作为 `x_t` 上的外部卷积。

2. **复值状态更新。** 之前的 Mamba 将状态矩阵从复数（S4）简化为实对角（Mamba）再到缩放恒等（Mamba-2）。Mamba-3 重新添加复数——等价于状态上的数据依赖的旋转嵌入。这恢复了之前实值简化所牺牲的状态追踪能力。

3. **多输入多输出（MIMO）投影。** 使用矩阵值投影而非每特征标量投影。在不增加解码延迟的情况下提高建模能力和推理时硬件利用率。

在 1.5B 参数上，Mamba-3 将平均下游精度比 Gated DeltaNet 提高 0.6 分；MIMO 变体额外增加 1.2 分，总计 1.8 分增益。在相同状态大小下，Mamba-3 以一半的状态匹配 Mamba-2。

Mamba-3 尚未在大规模生产混合模型中搭载——但它显然是下一代 Jamba 类模型 SSM 端的顺手候选。

### 何时使用混合模型

混合模型胜出当：

- 上下文足够长使纯 Transformer KV 缓存变得痛苦（64k+）。
- 任务混合短程结构（SSM 擅长）与长程召回（需要 Transformer）。
- 你想在单 GPU 内存预算上部署，其中 Transformer KV 缓存单独放不下。

混合模型失败当：

- 上下文较短（16k 以下）。SSM 开销被浪费；纯 Transformer 已足够。
- 任务需要处处到处的注意力（深度推理、多文档交叉引用）。混合模型中注意力层的稀疏性造成伤害。
- 你正在扩展到万亿参数前沿模型。纯 Transformer + MLA + MoE（DeepSeek-V3 风格）目前正在赢得能力竞赛。

### 竞争格局

| 模型 | 家族 | 规模 | 独特声明 |
|-------|--------|------|-------------|
| Mamba-2 | pure SSM | 3B | 线性时间，恒定内存 |
| Jamba | hybrid | 52B/12B | 256k on 80GB |
| Jamba 1.5 Large | hybrid | 398B/94B | 企业级长上下文 |
| Mamba-3 | pure SSM | 1.5B (论文) | 状态追踪恢复 |
| DeepSeek-V3 | pure Transformer + MoE | 671B/37B | 前沿能力 |

2026 年格局：纯 Transformer MoE 主导前沿，但混合模型占据了 256k+ 上下文的利基。Mamba-3 的状态追踪胜利可能推动下一代中混合比率降低（更多 SSM，更少注意力）。

```figure
swiglu-ffn
```

## 使用它

`code/main.py` 是混合架构的内存计算器。给定一个 SSM-Transformer 比率和一个 hidden-size / layer-count 配置，它计算：

- 目标上下文下的 KV 缓存。
- SSM 状态内存。
- 在上下文 N 下多种模型形状的总内存。

计算器支持：

- 纯 Transformer 基线（KV 缓存随 N 增长）。
- Jamba 风格 1:7 混合。
- 纯 SSM（完全没有 KV 缓存）。

数字直接来自 Jamba-1 和 Jamba-1.5 论文对其发布形状的数据，并对假设变体进行外推。

真实部署中的集成考虑：

- 大多数生产推理服务器（vLLM，SGLang）支持 Jamba 和 Mamba。检查具体版本。
- 在 256k 上下文下，Jamba 的内存优势展现在并发请求吞吐量上。在相同 VRAM 上，你容纳的 Jamba 序列比 Transformer 序列多。
- Mamba-3 作为独立模型尚未在生产中搭载——1.5B 的研究预览。

## 产出

本课产出 `outputs/skill-hybrid-picker.md`。给定一个工作负载规格（上下文长度概况、任务混合、内存预算），它在纯 Transformer、Jamba 风格混合和纯 SSM 之间推荐，并明确阐述内存和质量权衡。

## 练习

1. 运行 `code/main.py` 计算 32 层纯 Transformer（hidden 4096, 32 heads）和相同形状的 Jamba-1 混合模型在 256k 上下文下的 KV 缓存。验证 AI21 论文声明的约 8× 内存减少。

2. 修改计算器以模拟 1:3 混合（4 Mamba : 1 Attention）和 1:15 混合（14 Mamba : 1 Attention）。绘制 KV 缓存与比率。在什么比率下 KV 缓存等于 SSM 状态内存？

3. 阅读 Jamba 论文（arXiv:2403.19887）的第 3 节。解释为什么 AI21 使用 Mamba-1 而非 Mamba-2，尽管 Mamba-2 更快。提示：混合消融节记录了这一点。

4. 计算 Jamba 1.5 Large（398B 总量，94B 激活）中间隔一层 MoE 的参数开销。将激活比率与 DeepSeek-V3（37B/671B）比较，解释为什么 Jamba 的架构推动激活比率更高。

5. 阅读 Mamba-3 论文（arXiv:2603.15569）的第 3 节。用三句话解释为什么复值状态更新等价于数据依赖的旋转嵌入。将答案与第 7 阶段 · 第 04 课的 RoPE 推导联系起来。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| 状态空间模型（SSM）| "具有固定状态的递推" | 一个具有学习递推的层 `h_t = A h_{t-1} + B x_t`；每个 token 恒定内存 |
| 选择性 SSM | "Mamba 的技巧" | 数据依赖的 A、B、C 参数，给模型类似门控的选择性并保持线性时间 |
| 注意力与 Mamba 比率 | "多少注意力层" | 在 Jamba 中，`l = 8` 意味着每 7 个 Mamba 层配 1 个注意力层 |
| Jamba 块 | "8 层组" | 一个注意力 + 七个 Mamba + 交替位置的 MoE |
| SSM 状态 | "隐藏缓冲区" | 为 Mamba 层替换 KV 缓存的每层固定大小的状态 |
| 256k 上下文 | "Jamba 的标志性数字" | Jamba-1 能放入单张 80GB GPU 的序列长度；纯 Transformer 在该大小做不到 |
| Mamba-3 | "2026 纯 SSM" | 当前最佳纯 SSM 架构，具有复数状态 + MIMO；混合模型围绕其重建的基线 |
| MIMO | "多输入多输出" | Mamba-3 创新，使用矩阵值投影代替每特征标量 |
| 指数梯形离散化 | "Mamba-3 的递推" | 更具表现力的递推，包含 Mamba-2 的欧拉方法离散化 |
| 混合架构 | "混合注意力和 SSM" | 任何交错 Transformer 和 SSM 层的模型；Jamba 是生产原型的代表 |

## 延伸阅读

- [Lieber et al. — Jamba: A Hybrid Transformer-Mamba Language Model (arXiv:2403.19887)](https://arxiv.org/abs/2403.19887) —— 原始 Jamba 论文，比率消融，256k 上下文声明
- [AI21 — Jamba 1.5: Hybrid Transformer-Mamba at Scale (arXiv:2408.12570)](https://arxiv.org/abs/2408.12570) —— 扩展后的家族，398B/94B 和 12B/52B 公开发布
- [Gu, Dao — Mamba: Linear-Time Sequence Modeling with Selective State Spaces (arXiv:2312.00752)](https://arxiv.org/abs/2312.00752) —— Jamba 构建于其上的选择性 SSM 论文
- [Dao, Gu — Mamba-2 (arXiv:2405.21060)](https://arxiv.org/abs/2405.21060) —— 简化的结构化状态空间后继
- [Lahoti et al. — Mamba-3 (arXiv:2603.15569, ICLR 2026)](https://arxiv.org/abs/2603.15569) —— 复值状态、MIMO，2026 年纯 SSM 前沿
- [Gu et al. — Efficiently Modeling Long Sequences with Structured State Spaces (arXiv:2111.00396)](https://arxiv.org/abs/2111.00396) —— S4 论文，LLM 的 SSM 谱系起点
