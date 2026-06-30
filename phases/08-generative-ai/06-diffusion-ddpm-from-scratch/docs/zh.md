# 扩散模型 — 从零开始的 DDPM

> Ho、Jain、Abbeel（2020）给了该领域一个它无法放弃的配方。经过一千个小步骤用噪声破坏数据。训练一个神经网络来预测噪声。在推理时反转该过程。今天每个主流图像、视频、3D 和音乐模型都在这个循环上运行，可能上面还加了流匹配或一致性技巧。

**类型：** 构建
**语言：** Python
**前置条件：** 第三阶段 · 02（反向传播），第八阶段 · 02（VAE）
**时间：** 约 75 分钟

## 问题

你想要一个 `p_data(x)` 的采样器。GAN 玩一个经常发散的极小极大博弈。VAE 从高斯解码器产生模糊样本。你真正想要的是一个训练目标，它是 (a) 单一的稳定损失（无鞍点，无极小极大），(b) `log p(x)` 的下界（这样你就有似然），(c) 匹配 SOTA 质量的样本。

Sohl-Dickstein 等人（2015）有一个理论答案：定义一个逐渐添加高斯噪声的马尔可夫链 `q(x_t | x_{t-1})`，并训练一个反向链 `p_θ(x_{t-1} | x_t)` 进行去噪。Ho、Jain、Abbeel（2020）表明损失可以简化为一行——预测噪声——并清理了数学。在 2020 年这是一个好奇。在 2021 年它产生了最先进的样本。在 2022 年它成为了 Stable Diffusion。在 2026 年它是基础。

## 概念

![DDPM：前向噪声，反向去噪](../assets/ddpm.svg)

**前向过程 `q`。** 在 `T` 个小步骤中添加高斯噪声。闭式形式——数学可处理的原因——是累积步骤也是高斯的：

```
q(x_t | x_0) = N( sqrt(α̅_t) · x_0,  (1 - α̅_t) · I )
```

其中 `α̅_t = ∏_{s=1..t} (1 - β_s)` 对于 `β_t` 的调度。在 T=1000 步上从 1e-4 到 0.02 线性选择 `β_t`，则 `x_T` 近似为 `N(0, I)`。

**反向过程 `p_θ`。** 学习一个神经网络 `ε_θ(x_t, t)`，预测已添加的噪声。给定 `x_t`，通过以下方式去噪：

```
x_{t-1} = (1 / sqrt(α_t)) · ( x_t - (β_t / sqrt(1 - α̅_t)) · ε_θ(x_t, t) )  +  σ_t · z
```

其中 `σ_t` 要么是 `sqrt(β_t)` 要么是学习到的方差。表达式很丑，但它只是代数——给定后验 `q(x_{t-1} | x_t, x_0)` 求解 `x_{t-1}` 并用其噪声预测估计替换 `x_0`。

**训练损失。**

```
L_simple = E_{x_0, t, ε} [ || ε - ε_θ( sqrt(α̅_t) · x_0 + sqrt(1 - α̅_t) · ε,  t ) ||² ]
```

从数据中采样 `x_0`，选择随机 `t`，采样 `ε ~ N(0, I)`，通过闭式形式一次性计算带噪声的 `x_t`，并在噪声上进行回归。一个损失，无极小极大，无 KL，无重参数化技巧。

**采样。** 开始 `x_T ~ N(0, I)`。从 `t = T` 迭代反向步骤到 `1`。完成。

## 为什么有效

三种直觉：

1. **去噪容易；生成难。** 在 `t=T` 时，数据是纯噪声——网络只需解决一个平凡问题。在 `t=0` 时，网络只需清理少量像素。在中间 `t` 时，问题很难但网络有来自每个噪声级别的许多梯度流过相同的权重。

2. **伪装的分数匹配。** Vincent（2011）证明预测噪声等效于估计 `∇_x log q(x_t | x_0)`，即*分数*。反向 SDE 使用这个分数沿着密度梯度上升——向高概率区域的引导随机游走。

3. **ELBO 简化为简单 MSE。** 完整变分下界在每个时间步有一个 KL 项。使用 DDPM 的参数化，那些 KL 项简化为噪声预测上的 MSE，带有特定系数；Ho 丢弃了系数（称其为"简单"损失），且质量*改善*了。

```figure
diffusion-denoise
```

## 动手构建

`code/main.py` 实现 1-D DDPM。数据是双模式混合。"网络"是一个微型 MLP，接收 `(x_t, t)` 并输出预测噪声。训练是一行损失。采样迭代反向链。

### 步骤 1：前向调度（闭式）

```python
betas = [1e-4 + (0.02 - 1e-4) * t / (T - 1) for t in range(T)]
alphas = [1 - b for b in betas]
alpha_bars = []
cum = 1.0
for a in alphas:
    cum *= a
    alpha_bars.append(cum)
```

### 步骤 2：一次性采样 `x_t`

```python
def forward_sample(x0, t, alpha_bars, rng):
    a_bar = alpha_bars[t]
    eps = rng.gauss(0, 1)
    x_t = math.sqrt(a_bar) * x0 + math.sqrt(1 - a_bar) * eps
    return x_t, eps
```

### 步骤 3：一个训练步

```python
def train_step(x0, model, alpha_bars, rng):
    t = rng.randrange(T)
    x_t, eps = forward_sample(x0, t, alpha_bars, rng)
    eps_hat = model_forward(model, x_t, t)
    loss = (eps - eps_hat) ** 2
    return loss, gradient_step(model, ...)
```

### 步骤 4：反向采样

```python
def sample(model, alpha_bars, T, rng):
    x = rng.gauss(0, 1)
    for t in range(T - 1, -1, -1):
        eps_hat = model_forward(model, x, t)
        beta_t = 1 - alphas[t]
        x = (x - beta_t / math.sqrt(1 - alpha_bars[t]) * eps_hat) / math.sqrt(alphas[t])
        if t > 0:
            x += math.sqrt(beta_t) * rng.gauss(0, 1)
    return x
```

对于具有 40 个时间步和 24 单元 MLP 的 1-D 问题，这在大约 200 个 epoch 内学习了双模式混合。

## 时间条件化

网络需要知道它正在去噪哪个时间步。两种标准选项：

- **正弦嵌入。** 类似 Transformer 位置编码。`embed(t) = [sin(t/ω_0), cos(t/ω_0), sin(t/ω_1), ...]`。通过 MLP 传递，广播到网络中。
- **FiLM / 组归一化条件化。** 在每个块将嵌入投影到每通道 scale/bias (FiLM)。

我们的玩具代码使用正弦 → 拼接。生产 U-Net 使用 FiLM。

## 缺陷陷阱

- **调度非常重要。** 线性 `β` 是 DDPM 默认值，但余弦调度（Nichol 和 Dhariwal，2021）在相同计算量下给出更好的 FID。如果质量停滞则切换调度。
- **时间步嵌入是脆弱的。** 将原始 `t` 作为浮点数传递对玩具 1-D 有效但对图像失败；始终使用适当的嵌入。
- **V 预测 vs ε 预测。** 对于狭窄体制（非常小或非常大的 t），`ε` 具有差的信噪比。V 预测（`v = α·ε - σ·x`）更稳定；SDXL、SD3 和 Flux 使用它。
- **无分类器引导。** 在推理时，计算条件和非条件的 `ε`，然后 `ε_cfg = (1 + w) · ε_cond - w · ε_uncond`，其中 `w ≈ 3-7`。在第 08 课中介绍。
- **1000 步很多。** 生产中使用 DDIM（20-50 步）、DPM-Solver（10-20 步）或蒸馏（1-4 步）。参见第 12 课。

## 使用它

| 角色 | 2026 年典型技术栈 |
|------|-----------------------|
| 图像像素空间扩散（小型、玩具） | DDPM + U-Net |
| 图像潜在扩散 | VAE 编码器 + U-Net 或 DiT（第 07 课） |
| 视频潜在扩散 | 时空 DiT（Sora、Veo、WAN） |
| 音频潜在扩散 | Encodec + diffusion transformer |
| 科学（分子、蛋白质、物理） | 等变扩散（EDM、RFdiffusion、AlphaFold3） |

扩散是通用生成骨干。流匹配（第 13 课）是 2024-2026 年的竞争对手，通常在相同质量下在推理速度上胜出。

## 交付成果

保存 `outputs/skill-diffusion-trainer.md`。技能接收数据集 + 计算预算并输出：调度（linear/cosine/sigmoid）、预测目标（ε/v/x）、步数、引导尺度、采样器家族和评估协议。

## 练习

1. **简单。** 在 `code/main.py` 中将 T 从 40 改为 10。样本质量（输出的视觉直方图）如何退化？在什么 T 下双模式结构崩溃？
2. **中等。** 从 ε 预测切换到 v 预测。重新推导反向步骤。比较最终样本质量。
3. **困难。** 添加无分类器引导。以类别标签 `c ∈ {0, 1}` 为条件，在训练时 10% 的情况下丢弃它，并在采样时使用 `ε = (1+w)·ε_cond - w·ε_uncond`。在 `w = 0, 1, 3, 7` 时测量条件模式命中率。

## 关键术语

| 术语 | 人们怎么说 | 它实际上的含义 |
|------|-----------------|-----------------------|
| 前向过程 | "添加噪声" | 固定马尔可夫链 `q(x_t | x_{t-1})`，破坏数据。 |
| 反向过程 | "去噪" | 学习到的链 `p_θ(x_{t-1} | x_t)`，重建数据。 |
| β 调度 | "噪声阶梯" | 每步方差；线性、余弦或 sigmoid。 |
| α̅ | "Alpha bar" | 累积乘积 `∏(1 - β)`；给出从 `x_0` 的闭式 `x_t`。 |
| 简单损失 | "噪声上的 MSE" | `||ε - ε_θ(x_t, t)||²`；所有变分推导都塌缩为此。 |
| ε 预测 | "预测噪声" | 输出是添加的噪声；标准 DDPM。 |
| V 预测 | "预测速度" | 输出是 `α·ε - σ·x`；跨 t 的更好条件化。 |
| DDPM | "那篇论文" | Ho 等人 2020；线性 β，1000 步，U-Net。 |
| DDIM | "确定性采样器" | 非马尔可夫采样器，20-50 步，相同训练目标。 |
| 无分类器引导 | "CFG" | 混合条件和非条件噪声预测以放大条件化。 |

## 生产笔记：扩散推理是一个步数问题

DDPM 论文运行 T=1000 反向步。没人在生产中发布那个。每个真实推理技术栈选择三种策略之一——每种清晰映射到"延迟来自哪里"的生产框架：

1. **更快的采样器，相同模型。** DDIM（20-50 步）、DPM-Solver++（10-20）、UniPC（8-16）。反向循环的即插即用替换；训练好的 `ε_θ` 权重不变。延迟削减 20-50×。
2. **蒸馏。** 训练学生模型以更少步数匹配教师：渐进蒸馏（2 → 1）、一致性模型（任意 → 1-4）、LCM、SDXL-Turbo、SD3-Turbo。延迟再削减 5-10×，需要重新训练。
3. **缓存和编译。** `torch.compile(unet, mode="reduce-overhead")`、TensorRT-LLM 的扩散后端、`xformers`/SDPA 注意力、bf16 权重。每步延迟削减约 2×。与 (1) 和 (2) 叠加。

对于生产扩散服务器，预算对话与文献对 LLM 描述的一样：延迟是 `num_steps × step_cost + VAE_decode`，吞吐量是 `batch_size × (num_steps × step_cost)^-1`。TTFT 很小（一步）；TPOT 等效是完整响应时间，因为从用户角度看图像生成是"一次性全部"的。

## 延伸阅读

- [Sohl-Dickstein 等人 (2015). Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585) — 扩散论文，超前于时代。
- [Ho、Jain、Abbeel (2020). Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — DDPM。
- [Song、Meng、Ermon (2021). Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) — DDIM，更少步数。
- [Nichol 和 Dhariwal (2021). Improved DDPM](https://arxiv.org/abs/2102.09672) — 余弦调度，学习的方差。
- [Dhariwal 和 Nichol (2021). Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) — 分类器引导。
- [Ho 和 Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — CFG。
- [Karras 等人 (2022). Elucidating the Design Space of Diffusion-Based Generative Models (EDM)](https://arxiv.org/abs/2206.00364) — 统一符号，最干净的配方。
