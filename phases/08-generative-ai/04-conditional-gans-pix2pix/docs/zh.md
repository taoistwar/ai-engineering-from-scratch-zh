# 条件 GAN 与 Pix2Pix

> 2014-2017 年的第一个大解锁是控制 GAN 制作什么。附加一个标签，或一张图像，或一个句子。Pix2Pix 做了图像版本，它在狭窄的图像到图像任务上仍然击败了每个通用文本到图像模型。

**类型：** 构建
**语言：** Python
**前置条件：** 第八阶段 · 03（GAN），第四阶段 · 06（U-Net），第三阶段 · 07（CNN）
**时间：** 约 75 分钟

## 问题

无条件 GAN 采样任意人脸。对演示有用，在生产中无用。你想要：*将草图映射到照片*，*将地图映射到航拍照片*，*将白天场景映射到夜晚*，*为灰度图像上色*。在所有这些中，你给定一个输入图像 `x`，必须输出具有某种语义对应的 `y`。每个 `x` 有许多合理的 `y`。均方误差将它们压成糊状。对抗损失不会，因为"看起来真实"是锐利的。

条件 GAN（Mirza 和 Osindero，2014）添加条件 `c` 作为 `G` 和 `D` 的输入。Pix2Pix（Isola 等人，2017）对此进行了专门化：条件是完整输入图像，生成器是 U-Net，判别器是基于*补丁*的分类器（PatchGAN），损失是对抗 + L1。即使在 2026 年，该配方在狭窄的图像到图像领域上也优于从头开始的文本到图像模型，因为它是基于*配对数据*训练的——你拥有恰好需要的信号。

## 概念

![Pix2Pix：U-Net 生成器，PatchGAN 判别器](../assets/pix2pix.svg)

**条件 G。** `G(x, z) → y`。在 Pix2Pix 中，`z` 是 G 内部的 dropout（无输入噪声——Isola 发现显式噪声被忽略了）。

**条件 D。** `D(x, y) → [0, 1]`。输入是*配对*（条件，输出）。这是关键区别：D 必须判断 `y` 是否与 `x` 一致，而不仅仅是 `y` 是否看起来真实。

**U-Net 生成器。** 具有跨越瓶颈的跳跃连接的编码器-解码器。对于输入和输出共享低级结构（边缘、轮廓）的任务至关重要。没有跳跃连接，高频细节消失。

**PatchGAN 判别器。** D 不输出单一真实/赝品分数，而是输出一个 `N×N` 网格，其中每个单元格判断大约 70×70 像素的感受野。取平均值。这是马尔可夫随机场假设：真实性是局部的。训练快得多，参数更少，输出更锐利。

**损失。**

```
loss_G = -log D(x, G(x)) + λ · ||y - G(x)||_1
loss_D = -log D(x, y) - log (1 - D(x, G(x)))
```

L1 项稳定训练并将 G 推向已知目标。L1 给出比 L2 更锐利的边缘（中位数，不是均值）。`λ = 100` 是 Pix2Pix 的默认值。

## CycleGAN — 当你没有配对时

Pix2Pix 需要配对的 `(x, y)` 数据。CycleGAN（Zhu 等人，2017）以增加额外损失为代价放弃了这个要求：*循环一致性*损失。两个生成器 `G: X → Y` 和 `F: Y → X`。训练它们使 `F(G(x)) ≈ x` 且 `G(F(y)) ≈ y`。这让你能够将马转换为斑马，夏天转换为冬天，无需配对示例。

在 2026 年，无配对图像到图像主要通过扩散（ControlNet、IP-Adapter）而非 CycleGAN 完成，但循环一致性思想在几乎每篇无配对域适应论文中存活。

## 动手构建

`code/main.py` 在 1-D 数据上实现微型条件 GAN。条件 `c` 是类别标签（0 或 1）。任务：从给定类别的条件分布中产生样本。

### 步骤 1：将条件附加到 G 和 D 的输入

```python
def G(z, c, params):
    return mlp(concat([z, one_hot(c)]), params)

def D(x, c, params):
    return mlp(concat([x, one_hot(c)]), params)
```

One-hot 编码是最简单的方法。更大的模型使用学习的嵌入、FiLM 调制或交叉注意力。

### 步骤 2：条件训练

```python
for step in range(steps):
    x, c = sample_real_conditional()
    noise = sample_noise()
    update_D(x_real=x, x_fake=G(noise, c), c=c)
    update_G(noise, c)
```

生成器必须匹配给定条件的真实分布，而不是边缘分布。

### 步骤 3：验证每类输出

```python
for c in [0, 1]:
    samples = [G(noise, c) for noise in batch]
    mean_c = mean(samples)
    assert_near(mean_c, real_mean_for_class_c)
```

## 缺陷陷阱

- **条件被忽略。** G 学会边缘化，D 从不惩罚因为条件信号弱。修复：更激进地条件化 D（早期层，不仅是后期层），使用投影判别器（Miyato 和 Koyama 2018）。
- **L1 权重太低。** G 漂移到任意的看起来真实的输出，而不是忠实的。对于 Pix2Pix 风格的任务从 λ≈100 开始。
- **L1 权重太高。** G 产生模糊的输出，因为 L1 仍然是 L_p 范数。一旦训练稳定就退火降低。
- **D 中的真实标签泄漏。** 将 `(x, y)` 拼接为 D 输入，而不仅仅是 `y`。没有这个 D 无法检查一致性。
- **每类模式崩溃。** 每个类别可以独立崩溃。运行类别条件多样性检查。

## 使用它

2026 年图像到图像任务的现状：

| 任务 | 最佳方法 |
|------|---------------|
| 草图 → 照片，同一领域，配对数据 | Pix2Pix / Pix2PixHD（仍然快速，仍然锐利） |
| 草图 → 照片，无配对 | 使用 Scribble 条件模型的 ControlNet |
| 语义分割 → 照片 | SPADE / GauGAN2 或 SD + ControlNet-Seg |
| 风格迁移 | 使用 IP-Adapter 或 LoRA 的扩散；GAN 方法是遗留的 |
| 深度 → 照片 | 在 Stable Diffusion 上的 ControlNet-Depth |
| 超分辨率 | Real-ESRGAN (GAN)、ESRGAN-Plus 或 SD-Upscale（扩散） |
| 上色 | ColTran、基于扩散的上色器或 Pix2Pix-color |
| 白天 → 夜晚、季节、天气 | CycleGAN 或基于 ControlNet 的 |

当 (a) 你有数千个配对示例，(b) 任务狭窄且可重复，(c) 你需要快速推理时，Pix2Pix 仍然是正确的工具。在通用开放领域任务上，扩散获胜。

## 交付成果

保存 `outputs/skill-img2img-chooser.md`。技能接收任务描述、数据可用性（配对 vs 无配对，N 样本）和延迟/质量预算，然后输出：方法（Pix2Pix、CycleGAN、ControlNet 变体、SDXL + IP-Adapter）、训练数据要求、推理成本和评估协议（LPIPS、FID、任务特定）。

## 练习

1. **简单。** 修改 `code/main.py` 添加第三类。确认 G 仍然将每个类的噪声映射到正确模式。
2. **中等。** 在 1-D 设置中用感知风格损失替换 L1（例如作为特征提取器的小型冻结 D）。它是否改变了条件分布的锐度？
3. **困难。** 在 1-D 设置中勾勒 CycleGAN：两个分布，两个生成器，循环损失。展示它在无配对数据下学习它们之间的映射。

## 关键术语

| 术语 | 人们怎么说 | 它实际上的含义 |
|------|-----------------|-----------------------|
| 条件 GAN | "带标签的 GAN" | G(z, c)，D(x, c)。两个网络都看到条件。 |
| Pix2Pix | "图像到图像 GAN" | 配对 cGAN，带有 U-Net G 和 PatchGAN D + L1 损失。 |
| U-Net | "带跳跃连接的编码器-解码器" | 对称 conv 网络；跳跃连接保留高频。 |
| PatchGAN | "局部真实性分类器" | D 输出每补丁分数而不是全局分数。 |
| CycleGAN | "无配对图像转换" | 两个 G + 循环一致性损失；无配对数据。 |
| SPADE | "GauGAN" | 使用语义图归一化中间激活；分割到图像。 |
| FiLM | "特征级线性调制" | 来自条件的每特征仿射变换；廉价的条件化。 |

## 生产笔记：Pix2Pix 作为延迟受限基线

当你有配对数据和狭窄任务（草图 → 渲染、语义图 → 照片、白天 → 夜晚）时，Pix2Pix 的一次性推理在延迟上比扩散高出一个数量级。生产比较通常是：

| 路径 | 步骤 | 在单张 L4 上 512² 的典型延迟 |
|------|-------|----------------------------------------|
| Pix2Pix（U-Net 前向传播） | 1 | ~30 ms |
| SD-Inpaint 或 SD-Img2Img | 20 | ~1.2 s |
| SDXL-Turbo Img2Img | 1-4 | ~0.15-0.35 s |
| ControlNet + SDXL 基座 | 20-30 | ~3-5 s |

Pix2Pix 在静态批次上赢得吞吐量（每个请求是相同 FLOPs）。扩散在质量和泛化上获胜。现代玩法通常是为狭窄任务发布 Pix2Pix 风格的蒸馏模型，并为尾部输入使用扩散后备方案。

## 延伸阅读

- [Mirza 和 Osindero (2014). Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784) — cGAN 论文。
- [Isola 等人 (2017). Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) — Pix2Pix。
- [Zhu 等人 (2017). Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) — CycleGAN。
- [Wang 等人 (2018). High-Resolution Image Synthesis with Conditional GANs](https://arxiv.org/abs/1711.11585) — Pix2PixHD。
- [Park 等人 (2019). Semantic Image Synthesis with Spatially-Adaptive Normalization](https://arxiv.org/abs/1903.07291) — SPADE / GauGAN。
- [Miyato 和 Koyama (2018). cGANs with Projection Discriminator](https://arxiv.org/abs/1802.05637) — 投影 D。
