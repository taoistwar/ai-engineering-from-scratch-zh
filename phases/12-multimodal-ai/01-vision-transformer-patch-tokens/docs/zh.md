# 视觉 Transformer 与 Patch-Token 原语

> 在任何多模态之前，图像必须先变成 Transformer 能够消化的 token 序列。2020 年的 ViT 论文用 16x16 像素块、线性投影和位置嵌入回答了这个问题。五年后，每一款 2026 年前沿模型（Claude Opus 4.7 原生 2576px、Gemini 3.1 Pro、Qwen3.5-Omni）仍然以此为基础——编码器从 ViT 演进到了 DINOv2 再到 SigLIP 2，加入了 register token，位置方案变成了 2D-RoPE，但这个原语始终保留。本课从头到尾阅读 patch-token 流水线，并用纯 Python 标准库实现它，为 Phase 12 的后续内容建立一个关于"视觉 token"的具体心智模型。

**Type:** Learn
**Languages:** Python (stdlib, patch tokenizer + geometry calculator)
**Prerequisites:** Phase 7 (Transformers), Phase 4 (Computer Vision)
**Time:** ~120 minutes

## 学习目标

- 将 HxWx3 图像转换为带有正确位置编码的 patch token 序列。
- 计算给定（patch 大小、分辨率、隐藏维度、深度）的 ViT 的序列长度、参数量和 FLOPs。
- 列举从 2020 年研究到 2026 年生产环境 ViT 经历的三个升级：自监督预训练（DINO / MAE）、register token 和原生分辨率打包。
- 为下游任务在 CLS 池化、均值池化和 register token 之间做出选择。

## 问题

Transformer 操作的是向量序列。文本天然就是序列（字节或 token）。图像是一个具有三个颜色通道的二维像素网格——不是序列。如果将每个像素展开，一张 224x224 的 RGB 图像会变成 150,528 个 token，而这种长度的自注意力是无法使用的（与序列长度呈二次方关系）。

2020 年之前的方法是在前端加一个 CNN 特征提取器：ResNet 产生一个 7x7 的 2048 维向量特征图，将这 49 个 token 输入 Transformer。这能工作，但继承了 CNN 的偏置（平移等变性、局部感受野），并丧失了 Transformer 对规模的适应性。

Dosovitskiy 等人（2020）提出了一个直白的问题：如果跳过 CNN 会怎样？将图像分割成固定大小的 patch（比如 16x16 像素），将每个 patch 线性投影为一个向量，加上位置嵌入，然后将序列输入一个普通 Transformer。这在当时是异端——没有卷积的视觉。但有了足够的数据（JFT-300M，然后是 LAION），它在 ImageNet 上击败了 ResNet 并持续提升。

到 2026 年，ViT 原语已是无可争议的基础。每一款开源 VLM 的视觉塔都是其某种后代（DINOv2、SigLIP 2、CLIP、EVA、InternViT）。问题不再是要不要用 patch，而是用什么 patch 大小、什么分辨率策略、什么预训练目标、什么位置编码。

## 概念

### Patch 作为 Token

给定一张形状为 `(H, W, 3)` 的图像 `x` 和一个 patch 大小 `P`，你将图像划分成一个 `(H/P) x (W/P)` 的互不重叠的 patch 网格。每个 patch 是一个 `P x P x 3` 的像素立方体。将每个立方体展平为一个 `3 P^2` 向量。应用一个共享的线性投影 `W_E`（形状为 `(3 P^2, D)`）将每个 patch 映射到模型的隐藏维度 `D`。

对于 ViT-B/16 的典型配置：
- 分辨率 224，patch 大小 16 → 14x14 网格 → 196 个 patch token。
- 每个 patch 是 `16 x 16 x 3 = 768` 个像素值，投影到 `D = 768`。
- 添加一个可学习的 `[CLS]` token → 序列长度 197。

patch 投影在数学上与核大小为 `P`、步长为 `P`、输出通道数为 `D` 的 2D 卷积完全相同。这就是生产代码实际实现的方式——`nn.Conv2d(3, D, kernel_size=P, stride=P)`。"线性投影"是概念上的表述；卷积核形式才是高效的实现。

### 位置嵌入

Patch 没有内在顺序——Transformer 把它们当作一个"袋子"。早期 ViT 添加了一个可学习的 1D 位置嵌入（每个位置一个 768 维向量，共 197 个）。这能工作，但将模型与训练分辨率绑定：在推理时如果改变网格，就必须插值位置表。

现代视觉骨干网络使用 2D-RoPE（Qwen2-VL 的 M-RoPE、SigLIP 2 的默认方式）或因式分解的 2D 位置编码。2D-RoPE 基于 patch 的（行，列）索引旋转查询和键向量，使模型从旋转角度推断相对 2D 位置。没有位置表。模型可以在推理时处理任意网格大小。

### CLS Token、池化输出与 Register Token

图像的表示是什么？有三种共存的选择：

1. `[CLS]` token。在 patch 序列前添加一个可学习向量。经过所有 Transformer 块后，CLS token 的隐藏状态就是图像表示。继承自 BERT。原始 ViT、CLIP 使用。
2. 均值池化。对 patch token 的输出隐藏状态取平均。SigLIP、DINOv2、大多数现代 VLM 使用。
3. Register token。Darcet 等人（2023）观察到，在没有显式汇 token 的情况下训练的 ViT 会产生高范数"伪影"patch，劫持自注意力。添加 4-16 个可学习的 register token 吸收了这部分负载，并提升了密集预测（分割、深度估计）的质量。DINOv2 和 SigLIP 2 都配备了 register。

这种选择对下游任务很重要。CLS 对分类任务足够。对于将 patch token 馈入 LLM 的 VLM，你完全跳过池化——每个 patch 都成为 LLM 输入 token。Register 在传递给 LLM 之前被丢弃（它们是脚手架，不是内容）。

### 预训练：有监督、对比、掩码、自蒸馏

2020 年的 ViT 使用 JFT-300M 上的有监督分类进行预训练。很快被以下方法取代：

- CLIP（2021）：在 400M 对数据上的对比图像-文本训练。见第 12.02 课。
- MAE（2021，He 等人）：掩码 75% 的 patch，重建像素。自监督，仅需纯图像数据。
- DINO（2021）/ DINOv2（2023）：学生-教师自蒸馏，无需标签、无需标题。2023 年的 DINOv2 ViT-g/14 是最强的纯视觉骨干网络，也是"密集特征"用例的默认选择。
- SigLIP / SigLIP 2（2023，2025）：带有 sigmoid 损失和 NaFlex（原生宽高比）的 CLIP。是 2026 年开源 VLM（Qwen、Idefics2、LLaVA-OneVision）的主流视觉塔。

选择哪种预训练方式决定了骨干网络擅长什么：CLIP/SigLIP 用于与文本的语义匹配，DINOv2 用于密集视觉特征，MAE 作为下游微调的起点。

### 规模定律

ViT 的规模定律（Zhai 等人，2022）确定了 ViT 的质量遵循模型大小、数据规模和计算量方面的可预测规律。在固定计算量下：
- 更大的模型 + 更多数据 → 更好的质量。
- Patch 大小是序列长度与保真度的杠杆。Patch 14（DINOv2/SigLIP SO400m 的典型选择）每张图像比 patch 16 产生更多 token；对 OCR 和密集任务更好，但对速度不利。
- 分辨率是另一个重要的杠杆。从 224 到 384 再到 512 几乎总是有帮助，但 FLOPs 成本是平方级增长的。

ViT-g/14（1B 参数，patch 14，分辨率 224 → 256 token）和 SigLIP SO400m/14（400M 参数，patch 14）是 2026 年开源 VLM 的两款主力编码器。

### ViT 的参数计数

完整计算在 `code/main.py` 中。对于 224 分辨率的 ViT-B/16：

```
patch_embed = 3 * 16 * 16 * 768 + 768  =  591k
cls + pos    = 768 + 197 * 768          =  152k
block        = 4 * 768^2 (QKVO) + 2 * 4 * 768^2 (MLP) + 2 * 2*768 (LN)
             = 12 * 768^2 + 3k          =  7.1M
12 blocks    = 85M
final LN    = 1.5k
total       ≈ 86M
```

在加载检查点之前，都可以这样快速估算每个 ViT。骨干网络的大小决定了任何下游 VLM 的显存下限。

### 2026 年生产配置

2026 年大多数开源 VLM 配备的编码器是原生分辨率（NaFlex）下的 SigLIP 2 SO400m/14。它具有：
- 400M 参数。
- Patch 大小 14，默认分辨率 384 → 每张图像 729 个 patch token。
- 图像级任务使用均值池化；VQA 时全部 729 个 patch 流入 LLM。
- 4 个 register token，传递给 LLM 前丢弃。
- 2D-RoPE 配合图像级缩放以支持原生宽高比。

该配置中的每一项决策都可以追溯到一篇可读的论文。

```figure
image-patch-tokens
```

## 使用

`code/main.py` 是一个 patch token 生成器和几何参数计算器。它接收（图像 H、W、patch P、隐藏维度 D、深度 L）并报告：

- patch 化后的网格形状和序列长度。
- 一张合成的 8x8 像素玩具图像的 token 序列（逐步展示展平 + 投影路径）。
- 按 patch 嵌入、位置嵌入、Transformer 块和头部分类器拆分的参数计数。
- 目标分辨率下每次前向传播的 FLOPs。
- ViT-B/16 @ 224、ViT-L/14 @ 336、DINOv2 ViT-g/14 @ 224、SigLIP SO400m/14 @ 384 的对比表。

运行它。将参数计数与已发布的数据进行比对。调整 patch 大小和分辨率，感受 token 数量的成本。

## 产出

本课产出 `outputs/skill-patch-geometry-reader.md`。给定一个 ViT 配置（patch 大小、分辨率、隐藏维度、深度），它会给出 token 数、参数数和带有理由的显存估算。当为 VLM 选择视觉骨干网络时使用此技能——它能防止"token 爆炸导致 LLM 上下文耗尽"的意外。

## 练习

1. 计算 Qwen2.5-VL 在原生 1280x720 输入下、patch 大小为 14 时的 patch-token 序列长度。与仅使用 CLS 的表示相比如何？

2. 一帧 1080p 画面（1920x1080）在 patch 14 下产生多少个 token？以 30 FPS 计算一段 5 分钟视频的总视觉 token 数。以下哪种方式最能节省成本：池化、帧采样还是 token 合并？

3. 用纯 Python 实现对 patch token 的均值池化。验证对 196 个 token 进行均值池化的结果是否与 DINOv2 模型的 `forward` 方法在请求池化嵌入时返回的结果一致。

4. 阅读 "Vision Transformers Need Registers"（arXiv:2309.16588）第 3 节。用两句话描述 register 吸收了哪些伪影，以及为什么这对下游密集预测很重要。

5. 修改 `code/main.py` 以支持 patch-n'-pack：给定一个包含不同分辨率图像的列表，产生一个打包后的序列和块对角注意力掩码。在学到第 12.06 课时进行验证。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Patch | "16x16 像素方块" | 输入图像中固定大小、互不重叠的区域；成为一个 token |
| Patch 嵌入 | "线性投影" | 一个共享的可学习矩阵（或步长=P 的 Conv2d），将展平的 patch 像素映射为 D 维向量 |
| CLS token | "类 token" | 前置的可学习向量，其最终隐藏状态表示整张图像；2026 年为可选项 |
| Register token | "汇 token" | 额外的可学习 token，吸收 ViT 在预训练期间产生的高范数注意力伪影 |
| 位置嵌入 | "位置信息" | 每个位置的向量或旋转，使序列具有顺序感知能力；2D-RoPE 是现代默认选择 |
| 网格 | "Patch 网格" | 给定分辨率和 patch 大小下的 (H/P) x (W/P) 二维 patch 阵列 |
| NaFlex | "原生灵活分辨率" | SigLIP 2 的一项特性：单个模型支持多种宽高比和分辨率，无需重新训练 |
| 骨干网络 | "视觉塔" | 预训练的图像编码器，其 patch-token 输出馈入 VLM 中的 LLM |
| 池化 | "图像级摘要" | 将 patch token 转化为单个向量的策略：CLS、均值池化、注意力池化或基于 register 的方式 |
| Patch 14 vs 16 | "更细 vs 更粗的网格" | Patch 14 每张图像产生更多 token，对 OCR 保真度更好但更慢；patch 16 是经典默认值 |

## 拓展阅读

- [Dosovitskiy et al. — An Image is Worth 16x16 Words (arXiv:2010.11929)](https://arxiv.org/abs/2010.11929) —— 原始 ViT。
- [He et al. — Masked Autoencoders Are Scalable Vision Learners (arXiv:2111.06377)](https://arxiv.org/abs/2111.06377) —— MAE，自监督预训练。
- [Oquab et al. — DINOv2 (arXiv:2304.07193)](https://arxiv.org/abs/2304.07193) —— 大规模自蒸馏，无需标签。
- [Darcet et al. — Vision Transformers Need Registers (arXiv:2309.16588)](https://arxiv.org/abs/2309.16588) —— register token 与伪影分析。
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) —— 2026 年默认视觉塔。
- [Zhai et al. — Scaling Vision Transformers (arXiv:2106.04560)](https://arxiv.org/abs/2106.04560) —— 经验规模定律。
