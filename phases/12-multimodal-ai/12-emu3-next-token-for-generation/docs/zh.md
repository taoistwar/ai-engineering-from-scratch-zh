# Emu3：用下一 Token 预测实现图像和视频生成

> BAAI 的 Emu3（Wang 等人，2024 年 9 月）是 2024 年本应终结扩散 vs 自回归辩论的成果。一个单一的 Llama 风格仅解码器 Transformer，仅在下一 token 预测目标上训练，跨统一词表中的文本 + VQ 图像 token + 3D VQ 视频 token，在图像生成上击败了 SDXL，在感知上击败了 LLaVA-1.6。没有 CLIP 损失。没有扩散调度。在推理时使用无分类器引导以提高质量，但核心训练目标是使用教师强制（teacher forcing）的下一 token 预测。发表于 Nature。本课阅读 Emu3 的论点——为什么一个更好的分词器加上规模就是全部所需——并与扩散方法进行对比。

**Type:** Learn
**Languages:** Python (stdlib, 3D video tokenizer math + autoregressive sampler skeleton)
**Prerequisites:** Phase 12 · 11 (Chameleon)
**Time:** ~120 minutes

## 学习目标

- 解释为什么 Emu3 的单一损失下一 token 目标有效，尽管长期以来认为扩散是图像质量所需的假设。
- 描述 3D 视频分词器：时空 VQ 码本是什么样子的，为什么 patch 跨时间跨度。
- 比较 Emu3 与 Stable Diffusion XL 在（训练计算、推理成本、质量上限）上的差异。
- 列举同一个 Emu3 模型扮演的三种角色：Emu3-Gen（图像生成）、Emu3-Chat（感知）、Emu3-Stage2（视频生成）。

## 问题

直到 2024 年的传统智慧：图像生成需要扩散。论点：离散图像 token 损失的信息太多，无法重建细节，而自回归采样会跨数千个 token 累积误差。Stable Diffusion、DALL-E 3、Imagen、Midjourney 都使用某种形式的扩散。Chameleon（第 12.11 课）在小规模上部分推翻了这一点，但未能在质量上匹配 SDXL。

Emu3 正面攻击了这一论点。其主张：更好的视觉分词器 + 足够的规模 + 下一 token 损失 = 在同一个既能做感知又能做生成的模型中实现超越扩散的图像生成。

这个赌注在发布时是有争议的。两年后，开源统一生成家族（Emu3、Show-o、Janus-Pro、Transfusion）是研究的默认路径；生产前沿模型似乎使用了某种变体。

## 概念

### Emu3 分词器

关键成分是视觉分词器。Emu3 训练了一个自定义的 IBQ 类分词器（逆瓶颈量化器，SBER-MoVQGAN 家族），每个 token 进行 8x8 分辨率缩减。一张 512x512 图像变为 64x64 = 4096 个 token，码本大小为 32768。

这比 Chameleon 每张 512x512 的 1024 个 token 在 K=8192 下更大，但每个 token 更便宜（更小的码本查找，更简单的编解码器）。关键指标：重建 PSNR 为 30.5 dB，与 Stable Diffusion 连续潜在空间的 32 dB 有竞争力。

对于视频：一个 3D VQ 分词器将时空 patch（4x4x4 像素）编码为一个整数。一段以 8 FPS 播放的 4 秒片段有 32 帧；在 256x256、4x 空间和 4x 时间压缩下，token 数为 (256/4) * (256/4) * (32/4) = 64 * 64 * 8 = 32,768 个 token。

分词器质量是上限。Emu3 的贡献部分在于"我们训练了一个非常好的分词器"。

### 单一损失训练

Emu3 使用一个目标：在跨文本 token、2D 图像 token 和 3D 视频 token 的共享词表上预测下一 token。训练期间权重乘以模态特定因子以平衡贡献，但损失函数完全相同。

在以下混合上训练：
- 图像生成：`<text caption> <image> image_tokens </image>`
- 图像感知：`<image> image_tokens </image> <question> text_tokens`
- 视频生成：`<text caption> <video> video_tokens </video>`
- 视频感知：类似。
- 纯文本：标准 NTP。

模型从数据分布中学习何时发出图像 token 与文本 token。生成能力源于模型预测 `<image>` 标签后的图像 token。

### 无分类器引导与温度

自回归图像生成在推理时使用无分类器引导（CFG）效果显著更好。Emu3 使用它：生成两次，一次使用完整标题，一次使用空标题，使用引导权重（典型值 3.0-7.0）混合 logits。这与扩散中使用的 CFG 技巧相同，被借用到自回归设置中。

温度很重要：太高，会产生伪影；太低，会出现模式坍缩。Emu3 推荐的温度是感知为 1.0，图像生成为 0.8。

### 三种角色，一个模型

Emu3 以三个功能上不同的 API 发布，但底层是同一套权重：

- Emu3-Gen。图像生成。输入文本，输出图像 token。
- Emu3-Chat。VQA 和标题生成。输入图像（token），输出文本。
- Emu3-Stage2。视频生成和视频 VQA。输入文本或视频，输出文本或视频。

没有任务特定的头。只有不同的提示模板。相同的检查点。

### 基准

来自 Emu3 论文（2024 年 9 月）：

- 图像生成：在 MJHQ-30K FID 上击败 SDXL（5.4 vs 5.6），GenEval 总体（0.54 vs 0.55——统计平局），Deep-Eval 综合得分持平。
- 图像感知：在 VQAv2 上击败 LLaVA-1.6（75.1 vs 72.4），在 MMMU 上大致匹配。
- 视频生成：4 秒片段质量在 FVD 上与 Sora 时代公开基准模型有竞争力。

数字并不总是获胜——Emu3 在这里交换一点，在那里交换一点——但"下一 token 预测就是全部所需"的主张跨模态是可辩护的。

### 计算成本

Emu3 在约 3000 亿多模态 token 上训练，使用 7B 参数模型。GPU 小时大致与 Llama-2-7B 预训练相当（A100 级硅上 2k-4k GPU 年）。扩散模型如 Stable Diffusion 3 在类似预算下训练，但需要独立的文本编码器和更复杂的流水线。

在推理时，Emu3 每张图像比 SDXL 慢：4096 个图像 token 以 30 tok/s 的速度约需 2 分钟生成一张 512x512 图像，而 SDXL 只需 2-5 秒。推测解码和 KV 缓存优化缩小了差距，但未能关闭。自回归图像生成计算量大；这是存在的权衡。

### 为什么重要

Emu3 的深层贡献是概念性的。如果下一 token 预测可以扩展到匹配扩散在图像生成上的表现，那么统一模型路径（一个损失、一个骨干网络、任何模态）是可行的。未来模型不需要独立的文本编码器、独立的扩散调度器、独立的 VAE。一个 Transformer，每种模态一个分词器，规模。

Show-o、Janus-Pro 和 InternVL-U 都建立在此论点基础上或对其提出挑战。中国实验室（BAAI、DeepSeek）在 2025 年期间比美国实验室在此方向上更积极发布成果。

## 使用

`code/main.py` 构建了两个玩具部分：

- 一个 2D vs 3D VQ 分词器计数计算器：给定（分辨率、patch、片段长度、FPS），计算图像 vs 视频的 token 数。
- 一个带温度的无分类器引导自回归图像 token 采样器。

CFG 实现匹配 Emu3 的配方——使用引导权重混合条件和非条件 logits。

## 产出

本课产出 `outputs/skill-token-gen-cost-analyzer.md`。给定生成产品规格（图像或视频、目标分辨率、质量层级、延迟预算），计算 token 数、推理成本，并在 Emu3 家族和扩散之间做出选择。

## 练习

1. Emu3 在 8x8 缩减下每张 512x512 图像产生 4096 个 token。计算 1024x1024 和 2048x2048 的等效数值。推理延迟会怎样？

2. 阅读 Emu3 第 3.3 节关于视频分词器的内容。描述 3D VQ patch 形状，以及为什么它是 4x4x4 而不是 8x8x1。

3. 无分类器引导权重 5.0 vs 3.0：视觉效果是什么？在 `code/main.py` 中追踪数学。

4. 计算 Emu3-7B 在 300B token 上的训练 FLOPs，并与 Stable Diffusion 3 进行比较。哪个训练成本更高？

5. Emu3 在 FID 上击败 SDXL，但在 VQAv2 上没有击败专门的 VLM。解释为什么统一损失方法在不同基准上相对于专门模型表现出不同的强项。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 下一 token 预测 | "NTP" | 标准自回归损失：给定 token[0..i] 预测 token[i+1]；分词后适用于任何模态 |
| IBQ 分词器 | "逆瓶颈量化器" | 一类 VQ-VAE，比 Chameleon 的码本更大（32768+），重建质量更好 |
| 3D VQ | "时空量化器" | 按（时间，行，列）索引的码本；一个 token 覆盖一个 4x4x4 像素立方体 |
| 无分类器引导 | "CFG" | 使用权重 gamma 混合条件和非条件 logits；在推理时提升图像质量 |
| 统一词表 | "共享 token" | 文本 + 图像 + 视频都来自相同的整数空间；模型预测接下来是什么模态 |
| MJHQ-30K | "图像生成基准" | Midjourney 质量基准，30k 提示词；Emu3 在此报告 FID |

## 拓展阅读

- [Wang et al. — Emu3: Next-Token Prediction is All You Need (arXiv:2409.18869)](https://arxiv.org/abs/2409.18869)
- [Sun et al. — Emu: Generative Pretraining in Multimodality (arXiv:2307.05222)](https://arxiv.org/abs/2307.05222)
- [Liu et al. — LWM (arXiv:2402.08268)](https://arxiv.org/abs/2402.08268)
- [Yu et al. — MAGVIT-v2 (arXiv:2310.05737)](https://arxiv.org/abs/2310.05737)
- [Tian et al. — VAR (arXiv:2404.02905)](https://arxiv.org/abs/2404.02905)
