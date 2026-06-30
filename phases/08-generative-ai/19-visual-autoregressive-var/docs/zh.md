# 视觉自回归建模（VAR）：下一尺度预测

> 扩散模型在时间上迭代采样（去噪步骤）。VAR 在尺度上迭代采样——它预测 1x1 token，然后 2x2，然后 4x4，直到最终分辨率，每个尺度条件化于前一个。2024 年的论文展示了 VAR 在图像生成上匹配 GPT 风格的扩展律，并在相同计算预算下击败了 DiT。本课构建核心机制。

**类型：** 构建
**语言：** Python（使用 PyTorch）
**前置条件：** 第七阶段第 03 课（多头注意力），第八阶段第 06 课（DDPM）
**时间：** 约 90 分钟

## 问题

自回归生成主导语言建模是因为其扩展是可预测的：更多计算，更多参数，更低的困惑度，更好的输出。在 2024 年之前，图像生成有两次主要的 AR 尝试：PixelRNN/PixelCNN（逐像素）和 DALL-E 1 / Parti / MuseGAN（在 VQ-VAE 编码上逐 token）。

两者都遭受了生成顺序问题。像素和 token 排列在 2D 网格中，但 AR 模型必须以 1D 光栅顺序访问它们。早期角落像素不知道图像最终会变成什么。生成质量的扩展比 GPT-on-text 差，从未在匹配计算量下达到扩散模型质量。

VAR 通过改变被生成的内容来修复生成顺序问题。VAR 不是逐个在空间中预测图像 token，而是以递增分辨率预测整个图像。步骤 1：预测 1x1 token（整体图像"摘要"）。步骤 2：预测 2x2 token 网格（较粗糙的特征）。步骤 3：预测 4x4 网格。步骤 K：预测最终 (H/8)x(W/8) 网格。

每个尺度都关注所有先前尺度（在"尺度顺序"中因果地），并且在其自身尺度内并行。顺序问题消失了：尺度 k 的整个图像在一次 transformer 传递中产生。

## 概念

### VQ-VAE 多尺度分词器

VAR 需要一个**多尺度离散分词器**。对于图像 x，它产生一系列渐进更高分辨率的 token 网格：

```
x -> encoder -> latent f
f -> tokenize at 1x1: token grid z_1 of shape (1, 1)
f -> tokenize at 2x2: token grid z_2 of shape (2, 2)
...
f -> tokenize at (H/p)x(W/p): token grid z_K of shape (H/p, W/p)
```

每个 z_k 使用相同的码本（典型大小 4096-16384）。每个尺度的分词不是独立的——它被训练使得在每个尺度的残差求和重建 f：

```
f ≈ upsample(embed(z_1), target_size) + ... + upsample(embed(z_K), target_size)
```

这是**残差 VQ** 变体。尺度 k 捕获尺度 1..k-1 遗漏的内容。解码器取所有尺度嵌入的和并产生图像。

多尺度 VQ 分词器训练一次（像 VQGAN）然后冻结。所有生成工作由顶部的自回归模型完成。

### 下一尺度预测

生成模型是一个看到所有先前尺度的 token 并预测下一尺度 token 的 transformer。

输入序列结构：
```
[START, z_1 tokens, z_2 tokens, z_3 tokens, ..., z_K tokens]
```

位置嵌入编码尺度索引和尺度内的空间位置。注意力在尺度顺序上是因果的：尺度 k、位置 (i, j) 处的 token 可以关注尺度 1..k 处的所有 token 以及尺度 k 自身内在使用的任何尺度内顺序中更早的 token（VAR 使用固定位置注意力，无尺度内因果性——一个尺度内的所有位置并行预测）。

训练损失：在每个尺度 k，给定所有先前尺度 token 预测 token z_k。在离散 VQ 编码上的交叉熵损失。与 GPT 相同的结构，只是"序列"现在是尺度结构的。

### 生成

在推理时：
```
generate z_1 = sample from p(z_1)                    # 1 token
generate z_2 = sample from p(z_2 | z_1)              # 4 tokens in parallel
generate z_3 = sample from p(z_3 | z_1, z_2)         # 16 tokens in parallel
...
decode: f = sum of embed-and-upsample scales 1..K
image = VAE_decoder(f)
```

对于 K = 10 个尺度，生成是 10 次 transformer 前向传播。每次传递并行产生其整个尺度——一个尺度内没有逐 token 自回归。对于 256x256 图像，这大约是 10 次传递 vs DiT 的 28-50 次。

### 为什么下一尺度胜过下一 token

三方面结构性胜利：
1. **粗糙到精细与自然图像统计对齐。** 人类视觉感知和图像数据集都表现出依赖尺度的规律性：低频结构是稳定且可预测的；高频细节以低频内容为条件。下一尺度预测利用这一点。
2. **尺度内并行生成。** 与 GPT 风格 token AR 不同，VAR 一步生成一个尺度内的所有 token。有效生成长度是对数尺度而非线性。
3. **无生成顺序偏差。** 尺度 k 的 token 看到尺度 k-1 的全部；没有"左"或"上"偏差强制早期 token 在后期上下文可用之前提交。

### 扩展律

Tian 等人演示了 VAR 遵循 ImageNet 上 FID 的幂律扩展曲线——就像 GPT 对困惑度所做的那样。加倍参数或计算可靠地减半误差。这是首个像语言模型一样干净地展示这种扩展行为的图像生成模型。结果是 VAR 规模预测可以基于计算，而不是每个架构的经验猜测。

### 与扩散的关系

VAR 和扩散共享相同的数据压缩故事：两者都将生成问题分解为更简单子问题的序列。

- 扩散：逐渐添加噪声，学习撤销一步。
- VAR：逐渐添加分辨率，学习预测下一尺度。

它们是通过问题的不同轴。两者都产生可处理的条件分布。经验上 VAR 在推理时更快（更少的传递，尺度内全部并行），并在类别条件 ImageNet 上匹配或击败 DiT。文本条件 VAR（VARclip、HART）是一个活跃的研究方向。

## 动手构建

在 `code/main.py` 中，你将：
1. 在合成"图像"数据（2D 高斯环）上构建微型**多尺度 VQ 分词器**。
2. 训练一个 **VAR 风格 transformer** 进行下一尺度预测。
3. 通过调用 transformer 4 次（4 个尺度）并解码进行采样。
4. 验证尺度顺序训练使尺度内生成并行化。

这是一个玩具实现。重点是看尺度结构的注意力掩码和尺度内并行生成实际工作。

## 交付成果

本课产出 `outputs/skill-var-tokenizer-designer.md`——设计多尺度分词器的技能：尺度数量、尺度比率、码本大小、残差共享、解码器架构。

## 练习

1. **尺度数量消融。** 用 4、6、8、10 个尺度训练 VAR。测量重建质量 vs 自回归传递次数。更多尺度 = 更精细残差 = 更好的质量但更多传递。

2. **码本大小。** 用 512、4096、16384 的码本大小训练分词器。更大的码本给出更好的重建但更难的预测。找到拐点。

3. **尺度内并行检查。** 对于训练好的 VAR，显式地测量注意力模式。在尺度 k 内，模型是否关注跨尺度位置而非尺度内位置？验证掩码实现。

4. **VAR vs DiT 扩展。** 对于相同的 ImageNet 类别条件任务，在匹配的参数预算下训练 VAR 和 DiT（例如 33M、130M、458M）。绘制 FID vs 计算。VAR 应在每个大小上超过 DiT——在小规模上重现论文的结果。

5. **文本条件化。** 扩展 VAR 以通过 adaLN 将文本嵌入（CLIP pooled）作为额外条件输入。这是 HART 配方。在文本对齐采样上 FID 提高了多少？

## 关键术语

| 术语 | 人们怎么说 | 它实际上的含义 |
|------|----------------|----------------------|
| VAR | "Visual AutoRegressive" | 通过在 VQ token 网格金字塔上进行下一尺度预测的图像生成 |
| 下一尺度预测 | "先预测较粗，再较细" | 模型以递增分辨率尺度预测 token，条件化于所有先前尺度 |
| 多尺度 VQ 分词器 | "残差 VQ" | 产生 K 个递增分辨率 token 网格的 VQ-VAE，解码器求和所有尺度 |
| 尺度 k | "金字塔级别 k" | K 个分辨率级别之一，从 k=1 的 1x1 到 k=K 的 (H/p)x(W/p) |
| 尺度内并行 | "每个尺度一次前向传播" | 尺度 k 的所有 token 在一次 transformer 传递中预测，非自回归地 |
| 跨尺度因果 | "尺度顺序注意力" | 尺度 k 的 token 可以关注尺度 1..k 的所有内容，但不能关注尺度 k+1..K |
| 残差 VQ | "加性分词" | 每个尺度的 token 编码较低尺度遗留的残差；解码器求和所有尺度嵌入 |
| VAR 扩展律 | "图像 GPT 扩展" | FID 在计算上遵循可预测的幂律，就像语言模型的困惑度 |
| HART | "混合 VAR + 文本" | 结合 MaskGIT 风格迭代解码与 VAR 尺度结构的文本条件 VAR 变体 |
| 尺度位置嵌入 | "(scale, row, col) 三元组" | 位置编码携带尺度索引和尺度内的空间坐标 |

## 延伸阅读

- [Tian 等人，2024 — "Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction"](https://arxiv.org/abs/2404.02905) — VAR 论文，规范参考
- [Peebles 和 Xie，2022 — "Scalable Diffusion Models with Transformers"](https://arxiv.org/abs/2212.09748) — DiT，扩散比较基线
- [Esser 等人，2021 — "Taming Transformers for High-Resolution Image Synthesis"](https://arxiv.org/abs/2012.09841) — VQGAN，VAR 多尺度分词器扩展的分词器家族
- [van den Oord 等人，2017 — "Neural Discrete Representation Learning"](https://arxiv.org/abs/1711.00937) — VQ-VAE，离散图像分词的基础
- [Tang 等人，2024 — "HART: Efficient Visual Generation with Hybrid Autoregressive Transformer"](https://arxiv.org/abs/2410.10812) — 文本条件 VAR
