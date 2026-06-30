# 任意分辨率视觉：Patch-n'-Pack 与 NaFlex

> 真实图像不是 224x224 的正方形。收据是 9:16，图表是 16:9，医疗扫描可能是 4096x4096，手机截图是 9:19.5。2024 年之前的 VLM 答案——将所有内容缩放到固定正方形——抛弃了使 OCR、文档理解和高度场景解析成为可能的信号。NaViT（Google，2023）表明可以将可变分辨率的 patch 打包到单个 Transformer batch 中，并使用块对角掩码。Qwen2-VL 的 M-RoPE（2024）完全放弃了绝对位置表。LLaVA-NeXT 的 AnyRes 将高分辨率图像平铺成基础 + 子图像。SigLIP 2 的 NaFlex 变体（2025）现在是希望单一检查点服务所有宽高比的开源 VLM 的默认编码器。本课从头到尾实现 patch-n'-pack。

**Type:** Build
**Languages:** Python (stdlib, patch packer + block-diagonal mask)
**Prerequisites:** Phase 12 · 01 (ViT patches), Phase 12 · 05 (LLaVA)
**Time:** ~120 minutes

## 学习目标

- 将一个 batch 中可变分辨率图像的 patch 打包到一个序列中，并构建块对角注意力掩码。
- 在 AnyRes 平铺（LLaVA-NeXT）、NaFlex（SigLIP 2）和 M-RoPE（Qwen2-VL）之间为给定任务做出选择。
- 在不改变尺寸的情况下计算 OCR、图表和照片的 token 预算。
- 列举正方形尺寸调整的三种失败模式：文本变形、内容裁剪、用 padding 浪费 token。

## 问题

Transformers 期望输入的序列。一个 batch 是相同长度序列的堆叠。如果你的图像是 224x224，你每次都得到 196 个 patch token，不需要填充，任务完成。在 224 分辨率上训练，在 224 分辨率上推理，永远不必再想分辨率的问题。

但世界并不配合。文档是竖屏的（8.5x11 英寸，约 2:3）。图表截图是横屏的（16:9）。收据又高又窄（1:3）。医学图像以 2048x2048 或更大尺寸交付。移动设备截图为 1170x2532（0.46:1）。

2024 年之前的三个选项及各自的失败原因：

1. 缩放至固定正方形（224x224 或 336x336）。扭曲导致文本和人脸变形。缩小破坏了图表标签和 OCR 内容。LLaVA-1.5 之前的标准做法。
2. 裁剪至固定宽高比。丢弃图像的大部分内容，而选择裁剪位置本身就是一个视觉问题。
3. 填充至最长边。修复了变形，但在竖屏图像上浪费了 50% 以上的 token，且二次方的注意力成本消耗在所有这些填充 token 上。

2024-2025 年的答案：让 Transformer 以图像原生分辨率消化 patch，并想出一种方法将异构的 batch 打包到一个序列中而不浪费计算。

## 概念

### NaViT 与 Patch-n'-Pack

NaViT（Dehghani 等人，2023）是展示这种大规模可行性的论文。其思想是机械性的：

1. 对于 batch 中的每张图像，在选定的 patch 大小（比如 14）下计算其原生 patch 网格。
2. 将每张图像的 patch 展平到其自己的可变长度序列中。
3. 将所有图像的 patch 连接成一个用于 batch 的长序列。
4. 构建一个块对角注意力掩码，使图像 A 的 patch 仅在图像 A 内部进行关注。
5. 携带每个 patch 的位置信息（2D RoPE 或分数位置嵌入）。

三张图像（336x336 的 576 个 token、224x224 的 256 个 token、448x336 的 768 个 token）组成一个 1600 个 token 的序列，带有一个 1600x1600 的块对角掩码。没有填充。没有浪费计算。Transformer 处理任意宽高比。

NaViT 还引入了训练期间的分数 patch 丢弃——在 batch 中随机丢弃 50% 的 patch——这既提供了正则化效果又加速了训练。SigLIP 2 继承了这一点。

### AnyRes（LLaVA-NeXT）

LLaVA-NeXT 的 AnyRes 是实用的替代方案。给定一张高分辨率图像和一个固定编码器（CLIP 或 SigLIP @ 336），对图像进行平铺：

1. 从预定义的集合中选择最适合图像宽高比的网格布局——(1x1)、(1x2)、(2x1)、(1x3)、(3x1)、(2x2) 等。
2. 将完整图像平铺到该网格中；每个平铺成为一个 336x336 的裁剪。
3. 同时生成一个缩略图：将整个图像缩放到 336x336 作为全局上下文 token。
4. 通过冻结的 336 编码器编码每个平铺。拼接平铺 token + 缩略图 token。

对于一张 672x672 图像使用 2x2 网格加缩略图：4 * 576 + 576 = 2880 个视觉 token。昂贵但有效——LLM 同时看到局部细节和全局上下文。

AnyRes 是当你的编码器冻结且只支持单一分辨率时的首选路线。它会使大图像的 token 数爆炸（一张 1344x1344 图像在 4x4 网格下是 9216 + 576 ≈ 9800 个 token，填满大部分 8k LLM 上下文）。

### M-RoPE（Qwen2-VL）

Qwen2-VL 引入了多模态旋转位置嵌入（M-RoPE，Multimodal Rotary Position Embedding）。不是使用 NaViT 的分数位置或 AnyRes 的平铺与缩略图，每个 patch 携带一个 3D 位置（时间，高度，宽度）。查询/键旋转处理任意的 H、W 和时间长度。

M-RoPE 支持原生动态分辨率而无需重新训练。在推理时，你输入任意 HxW 的图像，patch 嵌入器生成 H/14 x W/14 个 token，每个 token 获得其 (t=0, r=row, c=col) 位置，RoPE 使用正确的频率旋转注意力，完成。Qwen2.5-VL 和 Qwen3-VL 继续使用此方案。InternVL3 的 V2PE 是相同的理念，每种模态有可变的编码。

与 AnyRes 不同，M-RoPE 在原生分辨率下是 O(H x W / P^2) token——没有乘性的平铺开销。与 NaViT 不同，它仍然期望每次前向传播处理一张图像。跨分辨率的 batch 仍然需要在顶部叠加 patch-n'-pack。

### NaFlex（SigLIP 2）

NaFlex 是 SigLIP 2 检查点的原生灵活模式。单个模型在推理时支持多种序列长度（256、729、1024 个 token）。内部它在训练期间使用 NaViT 风格的 patch-n'-pack，并使用每个 patch 的绝对分数位置。卖点：一个检查点，根据任务在推理时选择你的 token 预算。

对于语义任务（分类、检索），256 个 token。对于 OCR 或图表理解，1024 个 token。无需重新训练。

### 打包掩码

块对角掩码是大多数实现会出错的地方。对于一个覆盖图像 `i=0..B-1`、长度 `n_i` 的打包序列，总长度 `N_total`，形状为 `(N_total, N_total)` 的掩码 `M`：如果两个索引落在同一图像块内则为 1，否则为 0。你可以从累积长度列表构建：

```
offsets = [0, n_0, n_0+n_1, ..., N_total]
M[i, j] = 1 iff 存在 b 使得 offsets[b] <= i < offsets[b+1] 且 offsets[b] <= j < offsets[b+1]
```

这在 PyTorch 中可以通过 `torch.block_diag` 或显式的 gather 操作用一行代码实现。FlashAttention 的可变长度路径（`cu_seqlens`）完全跳过掩码，使用累积长度张量直接在序列内进行注意力——对于典型 batch，比密集掩码快约 10 倍。

### Token 预算

按任务选择策略：

- OCR / 文档：1024-4096 个 token。SigLIP 2 NaFlex 在 1024，或 AnyRes 3x3 + 缩略图。
- 图表和 UI：384-448 原生分辨率下 729-1024 个 token。Qwen2.5-VL 动态分辨率并带最大像素上限。
- 自然照片：256-576 个 token 就足够了。下游 LLM 看到足够的信息。在内容密度高的地方为 token 付费。
- 视频：空间池化后每帧 64-128 个 token，2-8 FPS。第 12.17 课介绍相关内容。

2026 年的生产规则：选择每个任务的最大像素上限，在该上限下按原生宽高比编码，打包 batch，跳过填充。Qwen2.5-VL 暴露 `min_pixels` 和 `max_pixels` 正是为此提供旋钮。

## 使用

`code/main.py` 为具有整数像素坐标的异构图像 batch 实现 patch-n'-pack。它：

- 接受一个 (H, W) 图像尺寸的列表。
- 在 patch 大小 14 下计算每张图像的 patch 序列长度。
- 将它们打包到一个总长度为 `sum(n_i)` 的序列中。
- 构建块对角注意力掩码（为清晰起见使用密集形式）。
- 将打包成本与正方形尺寸调整和 AnyRes 平铺进行比较。
- 打印混合 batch（收据、图表、截图、照片）的 token 预算表。

运行它。得出的数字正是每个 2026 年开源 VLM 使用 patch-n'-pack 的原因。

## 产出

本课产出 `outputs/skill-resolution-budget-planner.md`。给定一个混合宽高比的工作负载（OCR、图表、照片、视频帧）和总 token 预算，它选择正确的策略（NaFlex、AnyRes、M-RoPE 或固定正方形）并发出每个请求的配置。在为你产品选择 VLM 大小时使用此技能——它能防止无声的 10 倍 token 膨胀摧毁延迟预算。

## 练习

1. 一张收据是 600x1500（1:2.5）。在 patch 大小 14 下，原生分辨率有多少 token？正方形缩放至 336 后有多少？在实际中哪种方式会损失更多 OCR 准确率？

2. 为一个包含四张图像（长度分别为 256、576、729、1024）的 batch 构建块对角掩码。验证注意力矩阵是 2585x2585，且恰好有 `256^2 + 576^2 + 729^2 + 1024^2` 个非零条目。

3. 对于一张 1792x896 的 patch 14 图像，比较：(a) 正方形缩放至 336 然后编码，(b) AnyRes 2x1 + 缩略图，(c) 原生分辨率下的 M-RoPE。哪种使用的 token 最少？哪种保留的细节最多？

4. 实现分数 patch 丢弃：给定一个打包序列，均匀随机丢弃 50% 的 token，并相应更新块对角掩码。测量掩码稀疏度的变化。

5. 阅读 Qwen2-VL 论文（arXiv:2409.12191）第 3.2 节。用两句话描述 `min_pixels` 和 `max_pixels` 控制什么，以及为什么两个边界都重要。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Patch-n'-pack | "NaViT 风格打包" | 将不同图像的可变长度 patch 序列拼接到一个 batch 维度中 |
| 块对角掩码 | "打包掩码" | 注意力掩码，限制每张图像的 patch 只关注自身，而不关注包中的相邻图像 |
| AnyRes | "LLaVA-NeXT 平铺" | 将高分辨率图像分割成固定尺寸平铺网格外加一个全局缩略图；用固定编码器编码每个平铺 |
| NaFlex | "SigLIP 2 原生灵活" | 单个 SigLIP 2 检查点在推理时支持 256/729/1024 token 预算，无需重新训练 |
| M-RoPE | "多模态 RoPE" | 三维旋转位置编码（时间，行，列），处理任意 H、W、T 而不需要位置表 |
| cu_seqlens | "FlashAttention 打包" | FlashAttention varlen 路径使用的累积长度张量，替代密集块对角掩码 |
| min_pixels / max_pixels | "分辨率边界" | Qwen2.5-VL 的每个请求旋钮，限制极小或极大输入的 token 数 |
| 视觉 token 预算 | "每张图像多少个 token" | 每张图像发出的 patch token 大致数量；设置 LLM 的提示预算和注意力成本 |

## 拓展阅读

- [Dehghani et al. — Patch n' Pack: NaViT (arXiv:2307.06304)](https://arxiv.org/abs/2307.06304)
- [Wang et al. — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
- [Laurençon et al. — What matters when building vision-language models? (Idefics2, arXiv:2405.02246)](https://arxiv.org/abs/2405.02246)
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786)
- [Qwen Team — Qwen2.5-VL Technical Report (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
