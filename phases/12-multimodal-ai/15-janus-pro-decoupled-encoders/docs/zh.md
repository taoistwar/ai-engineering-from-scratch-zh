# Janus-Pro：统一多模态模型的解耦编码器

> 统一多模态模型存在一种不可避免的张力。理解需要语义特征——SigLIP 或 DINOv2 输出富含概念级别信息的向量。生成需要适合重建的编码——能组合回清晰像素的 VQ token。这两个目标在单个编码器中是不兼容的。Janus（DeepSeek，2024 年 10 月）和 Janus-Pro（DeepSeek，2025 年 1 月）认为解决方法是停止尝试：解耦两个编码器。在任务之间共享 Transformer 主体，但通过 SigLIP 路由理解，通过 VQ 分词器路由生成。在 7B 参数下，Janus-Pro 在 GenEval 上击败了 DALL-E 3，同时在 MMMU 上匹配 LLaVA。本课阅读为什么两个编码器在一个失败的地方有效。

**Type:** Build
**Languages:** Python (stdlib, dual-encoder routing + shared-body signal)
**Prerequisites:** Phase 12 · 13 (Transfusion), Phase 12 · 14 (Show-o)
**Time:** ~120 minutes

## 学习目标

- 解释为什么单个共享编码器会损害理解或生成质量。
- 描述 Janus-Pro 的路由：输入侧 SigLIP 特征用于理解，输入和输出侧 VQ token 用于生成。
- 追踪数据混合扩展如何使 Janus-Pro 在 Janus 失败的地方成功。
- 比较解耦（Janus-Pro）、耦合连续（Transfusion）和耦合离散（Show-o）架构。

## 问题

统一模型在理解和生成之间共享 Transformer 主体。之前的尝试（Chameleon、Show-o、Transfusion）都对两个方向使用一个视觉分词器。该分词器是一个妥协：

- 为重建（生成）优化：VQ-VAE 捕获细粒度像素细节，但产生的 token 语义一致性较弱。
- 为语义（理解）优化：SigLIP 嵌入将"猫"的图像分组在"猫"的 token 附近，但不允许好的重建。

Show-o 和 Transfusion 为此在一个方向上付出了可见的质量代价。Janus-Pro 问：当任务有不同的需求时，为什么要求一个分词器？

## 概念

### 解耦的视觉编码

Janus-Pro 的架构分离了两个编码器：

- 理解路径。输入图像 → SigLIP-SO400m → 2 层 MLP → Transformer 主体。
- 生成路径。输入图像（如果以现有图像为条件）→ VQ 分词器 → token ID → Transformer 主体。
- 输出生成。由 Transformer 预测的图像 token → VQ 解码器 → 像素。

Transformer 主体是共享的。主体上游和下游的一切都是任务特定的。

输入通过提示格式区分：`<understand>` 标签通过 SigLIP 路由；`<generate>` 通过 VQ 路由。或者路由从任务中隐式确定。

### 为什么这有效

理解损失获得 SigLIP 特征，CLIP 风格预训练已经将其调整为语义相似性。模型的感知基准比 Show-o / Transfusion 有所提升，因为输入特征对任务更优。

生成损失获得 VQ token，分词器已将其调整为重建效果。图像质量比 Show-o 有所提升，因为 VQ 编码能干净地组合回像素。

共享的 Transformer 主体看到两种输入分布（SigLIP 和 VQ）并学会同时处理两者。声称：足够的数据 + 足够的参数，主体可以吸收切换。

### 数据扩展——Janus vs Janus-Pro

Janus（原始版，arXiv 2410.13848）引入了解耦但在小规模下（1.3B 参数，有限数据）。Janus-Pro（arXiv 2501.17811）扩展了：

- 7B 参数（vs 1.3B）。
- 阶段 1（对齐）用了 90M 图像-文本对，高于 72M。
- 阶段 2（统一）用了 72M，高于 26M。
- 阶段 3 增加了 200k 图像生成指令样本。

结论：Janus-Pro-7B 在 MMMU 上匹配 LLaVA（60.3 vs ~58），在 GenEval 上击败 DALL-E 3（0.80 vs 0.67）。一个开源模型，在统一谱系的两端都有竞争力。

### JanusFlow——整流流变体

JanusFlow（arXiv 2411.07975）将 VQ 生成路径替换为整流流生成路径（连续）。拆分变为 SigLIP 用于理解 + 整流流用于生成。质量上限进一步提升。架构保持解耦编码器共享主体的模式。

### 共享主体的工作

Transformer 主体处理一个统一序列，但具有两种输入分布。它的工作是：

- 对于理解：消费 SigLIP 特征 + 文本 token → 自回归输出文本。
- 对于生成：消费文本 token +（可选图像 VQ token）→ 自回归输出图像 VQ token。

主体在每个块没有模态特定的权重。它就是你期望在 Qwen 或 Llama 内部找到的文本风格 Transformer，加上两个输入适配器。

有趣的是，这意味着 Janus-Pro 的主体可以从预训练的 LLM 初始化。Janus-Pro 确实从 DeepSeek-MoE-7B 初始化。这个选择很重要：LLM 贡献了纯从头训练的统一模型难以达到的推理能力。

### 与 InternVL-U 的比较

InternVL-U（第 12.10 课）是 2026 年的后续版本。它结合了：

- 原生多模态预训练（InternVL3 骨干网络）。
- 解耦编码器路由（SigLIP 输入，VQ + 扩散头输出）。
- 统一理解 + 生成 + 编辑。

InternVL-U 将 Janus-Pro 的架构选择纳入一个更大的框架中。解耦编码器的理念现在是大规模统一模型的默认选择。

### 局限

解耦编码器增加了架构复杂性。需要训练两个分词器，维护两条输入路径，有两组失败模式。对于不需要生成的产品，Janus-Pro 是过度设计的——选择一个 LLaVA 家族的理解模型。

对于不需要理解的产品，Janus-Pro 是过度资格的——选择一个 Stable Diffusion 3 / Flux 模型。

对于两者都需要的产品，Janus-Pro 现在是参考开源架构。

## 使用

`code/main.py` 模拟 Janus-Pro 路由：

- 两个模拟编码器：SigLIP 风格（产生 256 维语义向量）和 VQ 风格（产生整数编码）。
- 一个提示路由器，根据任务标签选择编码器。
- 一个共享主体（占位符），处理 token 序列无论哪个编码器产生的。
- 一个从阶段 1（对齐）到阶段 3（指令微调）的加权样本调度切换。

打印 3 个示例的路由路径：图像 QA、T2I、图像编辑。

## 产出

本课产出 `outputs/skill-decoupled-encoder-picker.md`。给定一个希望在前沿质量上统一生成 + 理解的产品，它选择 Janus-Pro、JanusFlow 或 InternVL-U，并给出具体的数据规模建议。

## 练习

1. Janus-Pro-7B 在 GenEval 上击败 DALL-E 3。解释为什么一个 7B 开源模型能在生成上匹配前沿专有模型，但在理解上不行。

2. 实现一个路由器函数：给定提示文本，分类为 `understand` 或 `generate`。如何处理"先描述再草绘"这样的模糊提示？

3. JanusFlow 将 VQ 路径替换为整流流。Transformer 主体现在输出什么，损失函数中有什么变化？

4. 提出 Janus-Pro 架构通过增加一个解耦编码器可以处理的第四种任务。示例：图像分割（DINO 风格）、深度估计（MiDaS 风格）。

5. 阅读 Janus-Pro 第 4.2 节关于数据扩展的内容。哪个数据阶段对 T2I 质量提升（相比 Janus）贡献最大？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 解耦编码 | "两个视觉编码器" | 每个方向独立的分词器或编码器：语义编码器用于理解，重建编码器用于生成 |
| 共享主体 | "一个 Transformer" | 单个 Transformer 处理任一编码器的输出；没有模态特定的权重 |
| SigLIP 用于理解 | "语义特征" | CLIP 家族视觉塔，提供丰富的概念特征但重建效果差 |
| VQ 用于生成 | "重建编码" | 向量量化 token，能干净地解码回像素 |
| JanusFlow | "整流流变体" | Janus-Pro 使用连续流匹配生成头而非 VQ |
| 路由标签 | "任务标签" | 提示标记（`<understand>` / `<generate>`），选择输入编码器 |

## 拓展阅读

- [Wu et al. — Janus (arXiv:2410.13848)](https://arxiv.org/abs/2410.13848)
- [Chen et al. — Janus-Pro (arXiv:2501.17811)](https://arxiv.org/abs/2501.17811)
- [Ma et al. — JanusFlow (arXiv:2411.07975)](https://arxiv.org/abs/2411.07975)
- [InternVL-U (arXiv:2603.09877)](https://arxiv.org/abs/2603.09877)
- [Dong et al. — DreamLLM (arXiv:2309.11499)](https://arxiv.org/abs/2309.11499)
