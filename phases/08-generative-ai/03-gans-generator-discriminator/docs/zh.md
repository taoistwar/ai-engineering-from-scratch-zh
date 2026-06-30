# GAN — 生成器 vs 判别器

> Goodfellow 2014 年的技巧是完全跳过密度。两个网络。一个制造赝品。一个抓住它们。它们斗争直到赝品与真实无法区分。这不应该有效。它经常不。当它有效时，对于狭窄领域的样本，文献中仍然是最锐利的。

**类型：** 构建
**语言：** Python
**前置条件：** 第三阶段 · 02（反向传播），第三阶段 · 08（优化器），第八阶段 · 02（VAE）
**时间：** 约 75 分钟

## 问题

VAE 产生模糊的样本，因为它们的 MSE 解码器损失对于*均值*图像是贝叶斯最优的——而许多合理数字的均值是一个模糊的数字。你想要一个奖励*合理性*的损失，而不是像素级接近任何一个目标。合理性没有闭式。你必须学习它。

Goodfellow 的想法：训练一个分类器 `D(x)` 来区分真实图像和赝品。训练一个生成器 `G(z)` 来欺骗 `D`。`G` 的损失信号是 `D` 当前认为使某些东西看起来真实的东西。这个信号随着 `G` 的改进而更新，追逐一个移动的目标。如果两个网络都收敛，`G` 已经学会了数据分布，而不需要写下 `log p(x)`。

这就是对抗训练。数学是一个极小极大博弈：

```
min_G max_D  E_real[log D(x)] + E_fake[log(1 - D(G(z)))]
```

在 2026 年，GAN 不再是 SOTA 生成器（扩散和流匹配吃了那个王冠）。但 StyleGAN 2/3 仍然是发布过的最锐利的人脸模型，GAN 判别器在扩散训练中被用作*感知损失*，对抗训练驱动了让你能够发布实时扩散的快速 1 步蒸馏（SDXL-Turbo、SD3-Turbo、LCM）。

## 概念

![GAN 训练：极小极大中的生成器和判别器](../assets/gan.svg)

**生成器 `G(z)`。** 将噪声向量 `z ~ N(0, I)` 映射到样本 `x̂`。解码器形状的网络（密集或转置卷积）。

**判别器 `D(x)`。** 将样本映射到标量概率（或分数）。真实 → 1，赝品 → 0。

**损失。** 两个交替更新：

- **训练 `D`：** `loss_D = -[ log D(x) + log(1 - D(G(z))) ]`。对真实=1、赝品=0 的二元交叉熵。
- **训练 `G`：** `loss_G = -log D(G(z))`。这是 Goodfellow 使用的*非饱和*形式（原始的 `log(1 - D(G(z)))` 在 `D` 自信时会饱和并杀死梯度）。

**训练循环。** 一步 `D`，一步 `G`。重复。

**为什么有效。** 如果 `G` 完美匹配 `p_data`，则 `D` 无法比随机做得更好，并处处输出 0.5；`G` 没有更多梯度。均衡。

**为什么崩溃。** 模式崩溃（`G` 找到一个 `D` 无法分类的模式并永远铸造它）、梯度消失（`D` 学习太快且 `log D` 饱和）、训练不稳定（学习率、批量大小、任何东西）。

## 使 GAN 有效的变体

| 年份 | 创新 | 修复 |
|------|------------|-----|
| 2015 | DCGAN | Conv/deconv、batch norm、LeakyReLU——第一个稳定架构。 |
| 2017 | WGAN、WGAN-GP | 用 Wasserstein 距离 + 梯度惩罚替换 BCE。修复梯度消失。 |
| 2017 | 谱归一化 | Lipschitz 约束判别器。2026 年判别器中仍在使用。 |
| 2018 | Progressive GAN | 先训练低分辨率，添加层。第一个百万像素结果。 |
| 2019 | StyleGAN / StyleGAN2 | 映射网络 + 自适应实例归一化。固定领域照片写实的 SOTA。 |
| 2021 | StyleGAN3 | 无混叠，平移等变——2026 年仍然是人脸黄金标准。 |
| 2022 | StyleGAN-XL | 条件化，类感知，更大规模。 |
| 2024 | R3GAN | 以更强的正则化重新品牌化；在无技巧下在 1024² 上工作。 |

```figure
gan-minimax
```

## 动手构建

`code/main.py` 在 1-D 数据上训练一个微型 GAN：两个高斯的混合。生成器和判别器是单隐藏层 MLP。我们手动实现前向、反向和极小极大循环。目标是观察两个关键失败模式（模式崩溃 + 梯度消失）的发生。

### 步骤 1：非饱和损失

标准 Goodfellow 损失 `log(1 - D(G(z)))` 在 D 以高置信度将 G 的赝品分类为赝品时趋于 0。在那时 G 的梯度基本为零——G 无法改进。非饱和形式 `-log D(G(z))` 有相反的渐近线：当 D 自信时它炸开，给 G 一个强信号。

```python
def g_loss(d_fake):
    # 最大化 log D(G(z))  <=>  最小化 -log D(G(z))
    return -sum(math.log(max(p, 1e-8)) for p in d_fake) / len(d_fake)
```

### 步骤 2：每个生成器步对应的一个判别器步

```python
for step in range(steps):
    # 训练 D
    real_batch = sample_real(batch_size)
    fake_batch = [G(z) for z in sample_noise(batch_size)]
    update_D(real_batch, fake_batch)

    # 训练 G
    fake_batch = [G(z) for z in sample_noise(batch_size)]  # 新鲜的赝品
    update_G(fake_batch)
```

为 G 使用新鲜的赝品，否则梯度是过时的。

### 步骤 3：观察模式崩溃

```python
if step % 200 == 0:
    samples = [G(z) for z in sample_noise(500)]
    mode_a = sum(1 for s in samples if s < 0)
    mode_b = 500 - mode_a
    if min(mode_a, mode_b) < 50:
        print("  [!] 模式崩溃：一个模式被饿死了")
```

规范症状：两个真实模式之一停止被生成。判别器停止纠正它，因为它从未被当作赝品看到。

## 缺陷陷阱

- **判别器太强。** 将 D 的学习率降低 2-5 倍，或添加实例/层噪声。如果 D 达到 >95% 准确率，G 就死了。
- **生成器记忆了一个模式。** 向 D 输入添加噪声，使用 minibatch-discriminator 层，或切换到 WGAN-GP。
- **Batch norm 泄漏统计。** 真实批次 + 赝品批次流经同一 BN 层会混合它们的统计。使用 instance norm 或 spectral norm 替代。
- **Inception 分数博弈。** FID 和 IS 在低样本计数时是嘈杂的。评估时使用 ≥10k 样本。
- **一次性采样对于条件任务是谎言。** 你仍然需要 CFG 尺度、截断技巧和重采样来获得可用的输出。

## 使用它

2026 年 GAN 技术栈：

| 情况 | 选择 |
|-----------|------|
| 照片写实人脸，固定姿态 | StyleGAN3（最锐利，最小） |
| 动漫 / 风格化人脸 | StyleGAN-XL 或 Stable Diffusion LoRA |
| 图像到图像转换 | Pix2Pix / CycleGAN（第八阶段 · 04）或 ControlNet（第八阶段 · 08） |
| 快速 1 步文本到图像 | 扩散的对抗蒸馏（SDXL-Turbo、SD3-Turbo） |
| 扩散训练器内部的感知损失 | 图像裁剪上的小型 GAN 判别器 |
| 任何多模态、开放性的 | 不要——使用扩散或流匹配 |

GAN 锐利但狭窄。一旦你的领域打开——照片、任意文本提示、视频——切换到扩散。对抗技巧作为一个组件（感知损失、蒸馏）生存下来，而不是独立的生成器。

## 交付成果

保存 `outputs/skill-gan-debugger.md`。技能接收一个失败的 GAN 运行（损失曲线、样本网格、数据集大小）并输出可能原因的排序列表、单行修复和重新运行协议。

## 练习

1. **简单。** 使用默认设置运行 `code/main.py`。然后设置 `D_LR = 5 * G_LR` 并重新运行。G 的损失多快崩溃为一个常数？
2. **中等。** 用 WGAN 损失替换 Goodfellow BCE 损失：`loss_D = E[D(fake)] - E[D(real)]`、`loss_G = -E[D(fake)]`，并将 D 的权重裁剪到 `[-0.01, 0.01]`。训练更稳定吗？比较挂钟收敛。
3. **困难。** 将 1-D 示例扩展到 2-D 数据（环上 8 个高斯的混合）。在步骤 1k、5k、10k 跟踪生成器捕获了 8 个模式中的多少。实现 minibatch 判别并重新测量。

## 关键术语

| 术语 | 人们怎么说 | 它实际上的含义 |
|------|-----------------|-----------------------|
| 生成器 | "G" | 噪声到样本网络，`G: z → x̂`。 |
| 判别器 | "D" | 分类器 `D: x → [0, 1]`，真实 vs 赝品。 |
| 极小极大 | "博弈" | 联合目标的 `min_G max_D`。 |
| 非饱和损失 | "修复" | 对 G 使用 `-log D(G(z))` 而不是 `log(1 - D(G(z)))`。 |
| 模式崩溃 | "G 记住了一件事" | 尽管数据多样，生成器产生很少的独特输出。 |
| WGAN | "Wasserstein" | 用推土机距离 + 梯度惩罚替换 BCE；更平滑的梯度。 |
| 谱归一化 | "Lipschitz 技巧" | 约束 D 的权重范数以限制其斜率；稳定训练。 |
| StyleGAN | "那个有效的" | 映射网络 + AdaIN；对于人脸仍然是最佳，2026 年仍然如此。 |

## 生产笔记：一次性推理是 GAN 的持久优势

GAN 在开放领域生成的样本质量上不再胜出，但在推理成本上仍然胜出。在推理文献词汇中，GAN 有：

- **无预填充、无解码阶段。** 单一的 `G(z)` 前向传播。TTFT ≈ 总延迟。
- **无 KV 缓存压力。** 唯一的状态是权重。批量大小受激活内存限制，而不是缓存。
- **简单连续批处理。** 由于每个请求花费相同的固定 FLOPs，服务器目标占用的静态批次通常是最优的。不需要飞行中调度器。

这就是 GAN 蒸馏（SDXL-Turbo、SD3-Turbo、ADD、LCM）在 2026 年是快速文本到图像主导技术的原因：它将 20-50 步的扩散流水线折叠为 1-4 次 GAN 式前向传播，同时保持扩散基础的分布。对抗损失作为将慢速生成器变快的训练时旋钮而生存。

## 延伸阅读

- [Goodfellow 等人 (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) — 原始 GAN 论文。
- [Radford 等人 (2015). Unsupervised Representation Learning with DCGAN](https://arxiv.org/abs/1511.06434) — 第一个稳定架构。
- [Arjovsky、Chintala、Bottou (2017). Wasserstein GAN](https://arxiv.org/abs/1701.07875) — WGAN。
- [Miyato 等人 (2018). Spectral Normalization for GANs](https://arxiv.org/abs/1802.05957) — SN。
- [Karras 等人 (2020). Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) — StyleGAN2。
- [Karras 等人 (2021). Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423) — StyleGAN3。
- [Sauer 等人 (2023). Adversarial Diffusion Distillation](https://arxiv.org/abs/2311.17042) — SDXL-Turbo。
