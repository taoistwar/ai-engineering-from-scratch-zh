# 自编码器与变分自编码器（VAE）

> 普通的自编码器压缩然后重建。它记忆。它不生成。添加一个技巧——强制编码看起来像高斯分布——你就得到了一个采样器。那个单一技巧，`z = μ + σ·ε` 的重参数化，正是你在 2026 年使用的每个潜在扩散和流匹配图像模型在输入端都有一个 VAE 的原因。

**类型：** 构建
**语言：** Python
**前置条件：** 第三阶段 · 02（反向传播），第三阶段 · 07（CNN），第八阶段 · 01（分类）
**时间：** 约 75 分钟

## 问题

将 784 像素的 MNIST 数字压缩为 16 个数字的编码，然后重建。普通自编码器将完美完成重建 MSE，但编码空间是一团乱麻。在编码空间中取一个随机点，解码它，你会得到噪声。它没有采样器。它是一种装扮起来的压缩模型。

你真正想要的是：(a) 编码空间是一个干净、平滑的你可以采样的分布——比如各向同性高斯 `N(0, I)`，(b) 解码任何样本产生合理的数字，以及 (c) 编码器和解码器仍然压缩良好。三个目标，一个架构，一个损失。

Kingma 2013 年的 VAE 通过训练编码器输出一个*分布* `q(z|x) = N(μ(x), σ(x)²)`，通过 KL 惩罚将该分布拉向先验 `N(0, I)`，然后在解码之前从 `q(z|x)` 采样 `z` 来解决此问题。在推理时，丢弃编码器，采样 `z ~ N(0, I)`，解码。KL 惩罚是强制编码空间结构化的原因。

在 2026 年，VAE 很少单独使用——它们在原始图像质量上已被扩散超越——但它们是每个潜在扩散模型（SD 1/2/XL/3、Flux、AudioCraft）的首选编码器。学会 VAE，你就学会了你使用的每个图像流水线看不见的第一层。

## 概念

![自编码器 vs VAE：重参数化技巧](../assets/vae.svg)

**自编码器。** `z = encoder(x)`，`x̂ = decoder(z)`，损失 = `||x - x̂||²`。编码空间无结构。

**VAE 编码器。** 输出两个向量：`μ(x)` 和 `log σ²(x)`。它们定义 `q(z|x) = N(μ, diag(σ²))`。

**重参数化技巧。** 从 `q(z|x)` 采样是不可微的。将采样重写为 `z = μ + σ·ε`，其中 `ε ~ N(0, I)`。现在 `z` 是 `(μ, σ)` 加上非参数噪声的确定性函数——梯度流过 `μ` 和 `σ`。

**损失。** 证据下界（ELBO），两项：

```
loss = reconstruction + β · KL[q(z|x) || N(0, I)]
     = ||x - x̂||²  + β · Σ_i ( σ_i² + μ_i² - log σ_i² - 1 ) / 2
```

重建推 `x̂` 向 `x`。KL 推 `q(z|x)` 向先验。它们权衡。小 β (<1) = 更锐利的样本，编码空间不那么高斯。大 β (>1) = 更干净的编码空间，更模糊的样本。β-VAE（Higgins 2017）使这个旋钮闻名并开启了解耦研究。

**采样。** 在推理时：抽取 `z ~ N(0, I)`，通过解码器前向传播。一次前向传播——不像扩散那样需要迭代采样。

```figure
vae-latent-grid
```

## 动手构建

`code/main.py` 实现一个没有 numpy 或 torch 的微型 VAE。输入是从 8 维中 2 成分高斯混合中抽取的 8 维合成数据。编码器和解码器是单隐藏层 MLP。我们实现 tanh 激活、前向传播、损失和手写的反向传播。不适用于生产——教学用途。

### 步骤 1：编码器前向传播

```python
def encode(x, enc):
    h = tanh(add(matmul(enc["W1"], x), enc["b1"]))
    mu = add(matmul(enc["W_mu"], h), enc["b_mu"])
    log_sigma2 = add(matmul(enc["W_sig"], h), enc["b_sig"])
    return mu, log_sigma2
```

`log σ²` 而不是 `σ`，这样网络输出是无约束的（σ 的 softplus 是个陷阱——梯度在 σ ≈ 0 时消失）。

### 步骤 2：重参数化和解码

```python
def reparameterize(mu, log_sigma2, rng):
    eps = [rng.gauss(0, 1) for _ in mu]
    sigma = [math.exp(0.5 * lv) for lv in log_sigma2]
    return [m + s * e for m, s, e in zip(mu, sigma, eps)]

def decode(z, dec):
    h = tanh(add(matmul(dec["W1"], z), dec["b1"]))
    return add(matmul(dec["W_out"], h), dec["b_out"])
```

### 步骤 3：ELBO

```python
def elbo(x, x_hat, mu, log_sigma2, beta=1.0):
    recon = sum((a - b) ** 2 for a, b in zip(x, x_hat))
    kl = 0.5 * sum(math.exp(lv) + m * m - lv - 1 for m, lv in zip(mu, log_sigma2))
    return recon + beta * kl, recon, kl
```

精确闭式 KL，因为两个分布都是高斯的。不要数值积分。人们在 2026 年仍然发布带有蒙特卡洛 KL 估计的代码——这毫无理由地慢 3 倍。

### 步骤 4：生成

```python
def sample(dec, z_dim, rng):
    z = [rng.gauss(0, 1) for _ in range(z_dim)]
    return decode(z, dec)
```

这就是生成模型。五行代码。

## 缺陷陷阱

- **后验崩溃。** KL 项如此激进地将 `q(z|x) → N(0, I)` 使得 `z` 不携带关于 `x` 的信息。修复：β 退火（从 β=0 开始，递增到 1）、free bits，或跳过不活跃维度的 KL。
- **模糊样本。** 高斯解码器似然意味着 MSE 重建，这对于 L2（均值）是贝叶斯最优的——一组合理数字的均值是一个模糊的数字。修复：离散解码器（VQ-VAE、NVAE），或仅将 VAE 用作编码器并将扩散堆叠在潜在向量上（这就是 Stable Diffusion 所做的）。
- **β 太大、太早。** 参见后验崩溃。从 β≈0.01 开始并增加。
- **潜在维度太小。** 16-D 适用于 MNIST，256-D 适用于 ImageNet 256²，2048-D 适用于 ImageNet 1024²。Stable Diffusion 的 VAE 压缩 512×512×3 → 64×64×4（空间面积 32 倍下采样，通道 32 倍压缩）。

## 使用它

2026 年 VAE 技术栈：

| 情况 | 选择 |
|-----------|------|
| 扩散的图像潜在编码器 | Stable Diffusion VAE (`sd-vae-ft-ema`) 或 Flux VAE |
| 音频潜在编码器 | Encodec (Meta)、SoundStream 或 DAC (Descript) |
| 视频潜在向量 | Sora 的时空补丁、Latte VAE、WAN VAE |
| 解耦表示学习 | β-VAE、FactorVAE、TCVAE |
| 离散潜在向量（用于 transformer 建模） | VQ-VAE、RVQ (ResidualVQ) |
| 用于生成的连续潜在向量 | 普通 VAE，然后在该潜在空间中条件化一个流/扩散模型 |

潜在扩散模型是一个 VAE，在编码器和解码器之间生活着一个扩散模型。VAE 做粗略压缩，扩散模型做繁重工作。视频的模式相同（VAE + 视频扩散 DiT）和音频（Encodec + MusicGen transformer）。

## 交付成果

保存 `outputs/skill-vae-trainer.md`。

技能接收：数据集概况 + 潜在维度目标 + 下游用途（重建、采样或潜在扩散输入），并输出：架构选择（plain/β/VQ/RVQ）、β 调度、潜在维度、解码器似然（高斯 vs 类别）、以及评估计划（重建 MSE、每维 KL、`q(z|x)` 与 `N(0, I)` 之间的 Fréchet 距离）。

## 练习

1. **简单。** 将 `code/main.py` 中的 `β` 改为 `0.01`、`0.1`、`1.0`、`5.0`。记录最终重建 MSE 和 KL。对于你的合成数据，哪个 β 是帕累托最优的？
2. **中等。** 用 Bernoulli 似然（交叉熵损失）替换高斯解码器似然。在相同合成数据的二值化版本上比较样本质量。
3. **困难。** 将 `code/main.py` 扩展为迷你 VQ-VAE：用 K=32 条目的码本中的最近邻查找替换连续 `z`。比较重建 MSE 并报告有多少码本条目标被使用（码本崩溃是真实的）。

## 关键术语

| 术语 | 人们怎么说 | 它实际上的含义 |
|------|-----------------|-----------------------|
| 自编码器 | 编码-解码网络 | `x → z → x̂`，学习 MSE。不生成。 |
| VAE | 带采样器的 AE | 编码器输出分布，KL 惩罚塑造编码空间。 |
| ELBO | 证据下界 | `log p(x) ≥ recon - KL[q(z|x) || p(z)]`；当 `q = p(z|x)` 时紧致。 |
| 重参数化 | `z = μ + σ·ε` | 将随机节点重写为确定性 + 纯噪声。使得能够通过采样反向传播。 |
| 先验 | `p(z)` | 潜在向量的目标分布，通常为 `N(0, I)`。 |
| 后验崩溃 | "KL 项获胜" | 编码器忽略 `x`，输出先验；解码器必须幻觉。 |
| β-VAE | 可调 KL 权重 | `loss = recon + β·KL`。更高 β = 更解耦但更模糊。 |
| VQ-VAE | 离散潜在向量 | 用最近码本向量替换连续 `z`；使 transformer 建模成为可能。 |

## 生产笔记：VAE 是扩散服务器中最热路径

在 Stable Diffusion / Flux / SD3 流水线中，VAE 每个请求被调用两次——一次编码（如果做 img2img / inpainting）和一次解码。在 1024² 时，解码器传递通常是整个流水线中最大的单次激活内存峰值，因为它将 `128×128×16` 潜在向量上采样回 `1024×1024×3`。两个实际后果：

- **切片或平铺解码。** `diffusers` 暴露 `pipe.vae.enable_slicing()` 和 `pipe.vae.enable_tiling()`。平铺将小的接缝伪影换成 `O(tile²)` 内存而不是 `O(H·W)`。对于消费级 GPU 上的 1024²+ 至关重要。
- **bf16 解码器，最终调整大小时使用 fp32 数值精度。** SD 1.x VAE 是以 fp32 发布的，当转为 fp16 时在 1024²+ *静默产生 NaN*。SDXL 提供 `madebyollin/sdxl-vae-fp16-fix`——始终偏好 fp16-fix 变体或使用 bf16。

## 延伸阅读

- [Kingma 和 Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — VAE 论文。
- [Higgins 等人 (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl) — 解耦 β-VAE。
- [van den Oord 等人 (2017). Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) — VQ-VAE。
- [Vahdat 和 Kautz (2021). NVAE: A Deep Hierarchical Variational Autoencoder](https://arxiv.org/abs/2007.03898) — 最先进的图像 VAE。
- [Rombach 等人 (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Stable Diffusion；VAE 作为编码器。
- [Défossez 等人 (2022). High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — Encodec，音频 VAE 标准。
