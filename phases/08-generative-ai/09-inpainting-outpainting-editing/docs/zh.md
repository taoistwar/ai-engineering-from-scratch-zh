# 图像修复、外延与图像编辑

> 文本到图像创造新东西。图像修复修复旧东西。在生产中，70% 的可计费图像工作是编辑——交换背景、移除徽标、扩展画布、重新生成一只手。图像修复是扩散赚取利润的地方。

**类型：** 构建
**语言：** Python
**前置条件：** 第八阶段 · 07（潜在扩散），第八阶段 · 08（ControlNet 和 LoRA）
**时间：** 约 75 分钟

## 问题

客户发送了一张完美产品照片，背景中有一个分散注意力的标志。你想擦除标志并保持其他所有内容像素相同。你不能从头运行文本到图像——结果将有不同的颜色、不同的光照、不同的产品角度。你想*仅*重新生成掩码区域，并且希望重新生成尊重周围的上下文。

这就是图像修复。变体：

- **图像修复。** 在掩码内部重新生成，保留外部像素。
- **图像外延。** 在掩码外部（或画布之外）重新生成，保留内部。
- **图像编辑。** 重新生成整个图像，但保持对原始图像的语义或结构保真度（SDEdit、InstructPix2Pix）。

2026 年的每个扩散流水线都发布了图像修复模式。Flux.1-Fill、Stable Diffusion Inpaint、SDXL-Inpaint、DALL-E 3 Edit。它们基于相同的原理工作。

## 概念

![图像修复：带掩码的去噪与上下文保留重新注入](../assets/inpainting.svg)

### 朴素方法（以及为什么它是错的）

使用掩码运行标准文本到图像。在每个采样步，用正向扩散的干净图像替换无掩码区域的带噪声潜在向量。它有效……但不佳。边界伪影渗出，因为模型没有关于掩码区域内有什么的信息。

### 适当的图像修复模型

训练一个修改后的 U-Net，它接收 9 个输入通道而不是 4 个：

```
input = concat([ noisy_latent (4ch), encoded_image (4ch), mask (1ch) ], dim=channel)
```

额外的通道是 VAE 编码的源图像副本加上单通道掩码。在训练时，你随机掩码图像区域并训练模型仅对掩码区域去噪，而无掩码区域作为干净的条件信号给出。在推理时，模型可以"看到"掩码区域周围的内容并产生连贯的补全。

SD-Inpaint、SDXL-Inpaint、Flux-Fill 都使用这个 9 通道（或类似）输入。diffusers 的 `StableDiffusionInpaintPipeline`、`FluxFillPipeline`。

### SDEdit（Meng 等人，2022）——免费编辑

向源图像添加噪声直到某个中间 `t`，然后用新提示从 `t` 向下运行反向链到 0。无需重新训练。起始 `t` 的选择用保真度换取创意自由度：

- `t/T = 0.3` → 几乎与源相同，小幅风格变化
- `t/T = 0.6` → 适度编辑，保留粗略结构
- `t/T = 0.9` → 从近噪声生成，最小源保留

### InstructPix2Pix（Brooks 等人，2023）

在 `(input_image, instruction, output_image)` 三元组上微调扩散模型。在推理时，同时条件化于输入图像和文本指令（"make it sunset"、"add a dragon"）。两个 CFG 尺度：图像尺度和文本尺度。

### RePaint（Lugmayr 等人，2022）

保持标准无条件扩散模型。在每个反向步，重新采样——偶尔跳回更嘈杂的状态并重新生成。避免边界伪影。当你没有训练过的图像修复模型时使用。

## 动手构建

`code/main.py` 在 5 维数据上实现玩具 1-D 图像修复方案。我们在 5-D 混合数据上训练 DDPM，其中每个样本是来自两个簇之一的 5 个浮点数。在推理时，我们"掩码" 5 个维度中的 2 个，在每一步注入无掩码三维的正向噪声版本，并仅重新生成掩码维度。

### 步骤 1：5-D DDPM 数据

```python
def sample_data(rng):
    cluster = rng.choice([0, 1])
    center = [-1.0] * 5 if cluster == 0 else [1.0] * 5
    return [c + rng.gauss(0, 0.2) for c in center], cluster
```

### 步骤 2：在所有 5 维上训练去噪器

标准 DDPM。网络输出 5-D 噪声预测对 5-D 带噪声输入。

### 步骤 3：在推理时，掩码感知反向传播

```python
def inpaint_step(x_t, mask, clean_image, alpha_bars, t, rng):
    # 用干净源的新鲜噪声版本替换无掩码维度
    a_bar = alpha_bars[t]
    for i in range(len(x_t)):
        if not mask[i]:
            x_t[i] = math.sqrt(a_bar) * clean_image[i] + math.sqrt(1 - a_bar) * rng.gauss(0, 1)
    # ... 然后在 x_t 上运行正常的反向步骤
```

这就是朴素方法，它在玩具 1-D 数据上有效。真实图像修复使用 9 通道输入，因为纹理相干性更重要。

### 步骤 4：图像外延

图像外延是掩码反转的图像修复：掩码新的（先前不存在）画布，用原始图像填充其余部分。相同的训练目标。

## 缺陷陷阱

- **接缝。** 朴素方法留下可见边界，因为梯度信息不跨掩码流动。修复：将掩码膨胀 8-16 像素，或使用适当的图像修复模型。
- **掩码泄漏。** 如果条件图像的无掩码区域质量低或嘈杂，它会污染掩码内部的生成。稍微去噪或模糊。
- **CFG 与掩码大小交互。** 小掩码上的高 CFG = 饱和补丁。对小型编辑降低 CFG。
- **SDEdit 保真度悬崖。** 从 `t/T = 0.5` 到 `t/T = 0.6` 可能会丢失主题的身份。扫描并设置检查点。
- **提示不匹配。** 提示应该描述*整个*图像，而不仅仅是新内容。"A cat sitting on a chair" 而不是 "a cat"。

## 使用它

| 任务 | 流水线 |
|------|----------|
| 移除物体，小掩码 | SD-Inpaint 或 Flux-Fill，标准提示 |
| 替换天空 | SD-Inpaint + "blue sky at sunset" |
| 扩展画布 | SDXL outpaint 模式（8px 羽化）或带 outpaint 掩码的 Flux-Fill |
| 重新生成手 / 脸 | SD-Inpaint 带重新描述主题的提示 + ControlNet-Openpose |
| 改变一个区域的风格 | 在掩码区域上以 `t/T=0.5` 进行 SDEdit |
| "让它变成日落" | InstructPix2Pix 或 Flux-Kontext |
| 背景替换 | SAM 掩码 → SD-Inpaint |
| 超高保真度 | Flux-Fill 或 GPT-Image（托管）用于最困难情况 |

SAM（Meta 的 Segment Anything，2023）+ 扩散图像修复是 2026 年背景移除流水线。SAM 2（2024）适用于视频。

## 交付成果

保存 `outputs/skill-editing-pipeline.md`。技能接收原始图像 + 编辑描述 + 可选掩码（或 SAM 提示）并输出：掩码生成方法、基础模型、CFG 尺度（图像 + 文本）、SDEdit-t 或图像修复模式、以及 QA 检查清单。

## 练习

1. **简单。** 在 `code/main.py` 中，将从 0.2 到 0.8 变化的被掩码维度比例变化。在什么比例下图像修复质量（掩码维度中的残差）等于无条件生成？
2. **中等。** 实现 RePaint：每 10 个反向步，跳回 5 步（添加噪声）并重新去噪。测量它是否减少掩码边缘的边界残差。
3. **困难。** 使用 Hugging Face diffusers 比较：SD 1.5 Inpaint + ControlNet-Openpose vs Flux.1-Fill 在 20 个人脸再生任务上。分别评分姿态遵循度和身份保留。

## 关键术语

| 术语 | 人们怎么说 | 它实际上的含义 |
|------|-----------------|-----------------------|
| 图像修复 | "填充洞" | 在掩码内重新生成；保留外部像素。 |
| 图像外延 | "扩展画布" | 在画布外重新生成；保留内部。 |
| 9 通道 U-Net | "适当的图像修复模型" | U-Net 以 `noisy | encoded-source | mask` 作为输入。 |
| SDEdit | "带噪声级别的 Img2img" | 加噪到时间 `t`，用新提示去噪。 |
| InstructPix2Pix | "纯文本编辑" | 在（图像、指令、输出）三元组上微调的扩散模型。 |
| RePaint | "无需重新训练" | 在反向过程中定期重新加噪声以减少接缝。 |
| SAM | "Segment Anything" | 通过点击或框生成掩码；与图像修复配对。 |
| Flux-Kontext | "带上下文编辑" | 接受参考图像 + 指令进行编辑的 Flux 变体。 |

## 生产笔记：编辑流水线对延迟敏感

编辑图像的用户期望亚 5 秒的往返。在 L4 上以 1024² 进行 30 步 SDXL-Inpaint 需要 3-4 秒，加上 SAM 掩码生成（~200 ms）和 VAE 编码/解码（合计 ~500 ms）。在生产框架中，这是 TTFT 受限而非吞吐量受限——1 个批次，低并发，最小化每个阶段：

- **SAM-H 是慢的那个。** SAM-H 在 1024² 下约为 200 ms；SAM-ViT-B 约为 40 ms，质量损失轻微。SAM 2（视频）增加了时间开销；不要将其用于单张图像编辑。
- **在可能时跳过编码。** `pipe.image_processor.preprocess(img)` 编码为潜在向量。如果你有来自上一代的潜在向量（在迭代编辑 UI 中典型），通过 `latents=...` 直接传递以跳过一次 VAE 编码。
- **掩码膨胀对吞吐量也重要。** 小掩码意味着大部分 U-Net 前向传播被浪费（无掩码像素无论如何都被钳位）。`diffusers` 的 `StableDiffusionInpaintPipeline` 无论如何都运行完整 U-Net；只有 9 通道适当的图像修复变体利用掩码计算。
- **Flux-Kontext 是 2025 年的答案。** 在 `(source_image, instruction)` 上的单次前向传播——无单独的掩码，无 SDEdit 噪声扫描。在 H100 上它在约 1.5 秒内完成编辑。架构上的教训：折叠阶段。

## 延伸阅读

- [Lugmayr 等人 (2022). RePaint: Inpainting using Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2201.09865) — 无需训练的图像修复。
- [Meng 等人 (2022). SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations](https://arxiv.org/abs/2108.01073) — SDEdit。
- [Brooks、Holynski、Efros (2023). InstructPix2Pix](https://arxiv.org/abs/2211.09800) — 文本指令编辑。
- [Kirillov 等人 (2023). Segment Anything](https://arxiv.org/abs/2304.02643) — SAM，掩码源。
- [Ravi 等人 (2024). SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714) — 视频 SAM。
- [Hertz 等人 (2022). Prompt-to-Prompt Image Editing with Cross-Attention Control](https://arxiv.org/abs/2208.01626) — 注意力级编辑。
- [Black Forest Labs (2024). Flux.1-Fill and Flux.1-Kontext](https://blackforestlabs.ai/flux-1-tools/) — 2024 工具集。
