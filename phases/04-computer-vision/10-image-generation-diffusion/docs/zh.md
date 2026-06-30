# 图像生成 — 扩散模型

> 扩散模型学习去噪。训练它从带噪图像中去除一丁点噪声，把这个过程反向重复一千次，你就得到一个图像生成器。

**类型：** Build
**语言：** Python
**先修要求：** Phase 4 Lesson 07 (U-Net), Phase 1 Lesson 06 (Probability), Phase 3 Lesson 06 (Optimizers)
**时间：** ~75 分钟

## 学习目标

- 推导前向加噪过程 `x_0 -> x_1 -> ... -> x_T`，并解释为什么闭式 `q(x_t | x_0)` 对任意 t 都成立
- 实现一个 DDPM 风格的训练目标，回归每一步添加的噪声，以及一个从纯噪声逆向走回图像的采样器
- 构建一个小型时间条件 U-Net（足够在 CPU 上训练），为任意时间步预测噪声
- 解释 DDPM 和 DDIM 采样之间的区别，以及各自适用的场景（Lesson 23 深入探讨 flow matching 和 rectified flow）

## 问题

GAN 是一次性生成的：噪声进，图像出，一次前向传播。它们很快但很难训练。扩散模型迭代生成：从纯噪声开始，以小步去噪，图像慢慢浮现。它们很慢但很容易训练。在过去五年中，后一种性质占据了主导地位：任何小团队都可以训练一个扩散模型并得到合理的样本；而 GAN 训练是你在多年失败运行中学会的一门手艺。

除训练稳定性外，扩散模型的迭代结构开启了现代图像生成能做的一切：文本条件、修复、图像编辑、超分辨率、可控风格。采样循环的每一步都是注入新约束的地方。这个钩子是为什么 Stable Diffusion、Imagen、DALL-E 3、Midjourney 以及你将要使用的每个可控图像模型都是基于扩散的原因。

本课构建最小的 DDPM：前向加噪、反向去噪、训练循环。下一课（Stable Diffusion）将它接入一个生产系统，配备 VAE、文本编码器和无分类器引导。

## 概念

### 前向过程

取一张图像 `x_0`。加少量高斯噪声得到 `x_1`。再加少量得到 `x_2`。持续 T 步直到 `x_T` 几乎与纯高斯噪声无法区分。

```
q(x_t | x_{t-1}) = N(x_t; sqrt(1 - beta_t) * x_{t-1},  beta_t * I)
```

`beta_t` 是一个小的方差调度（variance schedule），通常在 T=1000 步内从 0.0001 线性增加到 0.02。每一步略微缩小信号并注入新噪声。

### 闭式跳跃

逐步加噪是一个 Markov 链，但数学可以折叠：你可以一步直接从 `x_0` 采样 `x_t`。

```
定义 alpha_t = 1 - beta_t
定义 alpha_bar_t = prod_{s=1..t} alpha_s

那么：
  q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0,  (1 - alpha_bar_t) * I)

等价地：
  x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
  其中 epsilon ~ N(0, I)
```

这一个方程是整个扩散模型变得实用的原因。训练时你随机挑选一个 `t`，从 `x_0` 直接采样 `x_t`，然后一步训练——无需模拟完整的 Markov 链。

### 反向过程

前向过程是固定的。反向过程 `p(x_{t-1} | x_t)` 是神经网络要学习的。扩散模型不直接预测 `x_{t-1}`；它们预测在第 t 步添加的噪声 `epsilon`，然后由数学从中推导出 `x_{t-1}`。

```mermaid
flowchart LR
    X0["x_0<br/>(干净图像)"] --> Q1["q(x_t|x_0)<br/>加噪声"]
    Q1 --> XT["x_t<br/>(带噪)"]
    XT --> MODEL["model(x_t, t)"]
    MODEL --> EPS["预测的 epsilon"]
    EPS --> LOSS["与真实 epsilon<br/>的 MSE"]

    XT -.->|采样| STEP["p(x_{t-1}|x_t)"]
    STEP -.-> XT1["x_{t-1}"]
    XT1 -.->|重复 1000 次| X0S["x_0（采样结果）"]

    style X0 fill:#dcfce7,stroke:#16a34a
    style MODEL fill:#fef3c7,stroke:#d97706
    style LOSS fill:#fecaca,stroke:#dc2626
    style X0S fill:#dbeafe,stroke:#2563eb
```

### 训练损失

每一个训练步骤：

1. 采样一张真实图像 `x_0`。
2. 从 [1, T] 均匀采样时间步 `t`。
3. 采样噪声 `epsilon ~ N(0, I)`。
4. 计算 `x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon`。
5. 用网络预测 `epsilon_theta(x_t, t)`。
6. 最小化 `|| epsilon - epsilon_theta(x_t, t) ||^2`。

就是这样。神经网络学习在任意时间步预测噪声。损失是 MSE。没有对抗博弈，没有模式崩溃，没有振荡。

### 采样器（DDPM）

生成时：从 `x_T ~ N(0, I)` 出发，逐步反向走。

```
for t = T, T-1, ..., 1:
    eps = model(x_t, t)
    x_{t-1} = (1 / sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * eps) + sqrt(beta_t) * z
    其中 z ~ N(0, I) 当 t > 1，否则为 0
return x_0
```

关键在于，尽管一般情况下反向条件分布在闭式中不可知，但对于这个特定的高斯前向过程，它是可知的。那些看起来丑陋的系数正是贝叶斯规则给出的结果。

### 为什么是 1000 步

前向噪声调度的选择使得每一步添加的噪声刚好足够让反向步骤接近高斯分布。步数太少，反向步骤远离高斯分布，网络无法很好地建模。步数太多，采样变得昂贵且收益递减。T=1000、线性调度是 DDPM 的默认设置。

### DDIM：20 倍更快的采样

训练是一样的。采样变了。DDIM（Song 等，2020）定义了一个确定性反向过程，无需重新训练即可跳过时间步。用 DDIM 在 50 步内采样可以达到接近 DDPM 1000 步的质量。每个生产系统都使用 DDIM 或更快的变体（DPM-Solver、Euler ancestral）。

### 时间条件

网络 `epsilon_theta(x_t, t)` 需要知道它正在对哪个时间步去噪。现代扩散模型通过正弦时间嵌入（与 Transformer 中位置编码相同的思路）注入 `t`，这些嵌入被加到每个 U-Net 级别的特征图中。

```
t_embedding = sinusoidal(t)
feature_map += MLP(t_embedding)
```

没有时间条件，网络必须从图像本身猜测噪声水平——这也能工作，但样本效率低得多。

## Build It

### 步骤 1：噪声调度

```python
import torch

def linear_beta_schedule(T=1000, beta_start=1e-4, beta_end=2e-2):
    return torch.linspace(beta_start, beta_end, T)


def precompute_schedule(betas):
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
        "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod),
        "sqrt_recip_alphas": torch.sqrt(1.0 / alphas),
    }

schedule = precompute_schedule(linear_beta_schedule(T=1000))
```

一次预计算，在训练和采样时按索引取用。

### 步骤 2：前向扩散（q_sample）

```python
def q_sample(x0, t, noise, schedule):
    sqrt_a = schedule["sqrt_alphas_cumprod"][t].view(-1, 1, 1, 1)
    sqrt_one_minus_a = schedule["sqrt_one_minus_alphas_cumprod"][t].view(-1, 1, 1, 1)
    return sqrt_a * x0 + sqrt_one_minus_a * noise
```

一行闭式。`t` 是一个时间步批次，批次中每张图像一个。

### 步骤 3：一个微型时间条件 U-Net

```python
import torch.nn as nn
import torch.nn.functional as F
import math

def timestep_embedding(t, dim=64):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([args.sin(), args.cos()], dim=-1)
    return emb


class TinyUNet(nn.Module):
    def __init__(self, img_channels=3, base=32, t_dim=64):
        super().__init__()
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, base * 4),
            nn.SiLU(),
            nn.Linear(base * 4, base * 4),
        )
        self.t_dim = t_dim
        self.enc1 = nn.Conv2d(img_channels, base, 3, padding=1)
        self.enc2 = nn.Conv2d(base, base * 2, 4, stride=2, padding=1)
        self.mid = nn.Conv2d(base * 2, base * 2, 3, padding=1)
        self.dec1 = nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1)
        self.dec2 = nn.Conv2d(base * 2, img_channels, 3, padding=1)
        self.time_proj = nn.Linear(base * 4, base * 2)

    def forward(self, x, t):
        t_emb = timestep_embedding(t, self.t_dim)
        t_emb = self.t_mlp(t_emb)
        t_proj = self.time_proj(t_emb)[:, :, None, None]

        h1 = F.silu(self.enc1(x))
        h2 = F.silu(self.enc2(h1)) + t_proj
        h3 = F.silu(self.mid(h2))
        d1 = F.silu(self.dec1(h3))
        d2 = torch.cat([d1, h1], dim=1)
        return self.dec2(d2)
```

两级 U-Net，在瓶颈处注入时间条件。对于真实图像，增加深度和宽度。

### 步骤 4：训练循环

```python
def train_step(model, x0, schedule, optimizer, device, T=1000):
    model.train()
    x0 = x0.to(device)
    bs = x0.size(0)
    t = torch.randint(0, T, (bs,), device=device)
    noise = torch.randn_like(x0)
    x_t = q_sample(x0, t, noise, schedule)
    pred = model(x_t, t)
    loss = F.mse_loss(pred, noise)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
```

这就是整个训练循环。没有 GAN 博弈，没有专门的损失，只有一次 MSE 调用。

### 步骤 5：采样器（DDPM）

```python
@torch.no_grad()
def sample(model, schedule, shape, T=1000, device="cpu"):
    model.eval()
    x = torch.randn(shape, device=device)
    betas = schedule["betas"].to(device)
    sqrt_one_minus_a = schedule["sqrt_one_minus_alphas_cumprod"].to(device)
    sqrt_recip_alphas = schedule["sqrt_recip_alphas"].to(device)

    for t in reversed(range(T)):
        t_batch = torch.full((shape[0],), t, dtype=torch.long, device=device)
        eps = model(x, t_batch)
        coef = betas[t] / sqrt_one_minus_a[t]
        mean = sqrt_recip_alphas[t] * (x - coef * eps)
        if t > 0:
            x = mean + torch.sqrt(betas[t]) * torch.randn_like(x)
        else:
            x = mean
    return x
```

1000 次前向传播来生成一批样本。在实际代码中你会用 DDIM 50 步采样器替换它。

### 步骤 6：DDIM 采样器（确定性，约 20 倍更快）

```python
@torch.no_grad()
def sample_ddim(model, schedule, shape, steps=50, T=1000, device="cpu", eta=0.0):
    model.eval()
    x = torch.randn(shape, device=device)
    alphas_cumprod = schedule["alphas_cumprod"].to(device)

    ts = torch.linspace(T - 1, 0, steps + 1).long()
    for i in range(steps):
        t = ts[i]
        t_prev = ts[i + 1]
        t_batch = torch.full((shape[0],), t, dtype=torch.long, device=device)
        eps = model(x, t_batch)
        a_t = alphas_cumprod[t]
        a_prev = alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)
        x0_pred = (x - torch.sqrt(1 - a_t) * eps) / torch.sqrt(a_t)
        sigma = eta * torch.sqrt((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev))
        dir_xt = torch.sqrt(1 - a_prev - sigma ** 2) * eps
        noise = sigma * torch.randn_like(x) if eta > 0 else 0
        x = torch.sqrt(a_prev) * x0_pred + dir_xt + noise
    return x
```

`eta=0` 是完全确定性的（相同噪声输入总是产生相同输出）。`eta=1` 恢复 DDPM。

## Use It

对于生产工作，使用 `diffusers`：

```python
from diffusers import DDPMScheduler, UNet2DModel

unet = UNet2DModel(sample_size=32, in_channels=3, out_channels=3, layers_per_block=2)
scheduler = DDPMScheduler(num_train_timesteps=1000)
```

该库提供了现成的调度器（DDPM、DDIM、DPM-Solver、Euler、Heun）、可配置的 U-Net、文本到图像和图像到图像的 pipeline，以及 LoRA 微调辅助工具。

对于研究，`k-diffusion`（Katherine Crowson）拥有最忠实的参考实现和最佳的采样变体。

## Ship It

本课产出：

- `outputs/prompt-diffusion-sampler-picker.md`——一个根据质量目标、延迟预算和条件类型选择 DDPM / DDIM / DPM-Solver / Euler 的 prompt。
- `outputs/skill-noise-schedule-designer.md`——一个根据 T 和目标损坏水平生成线性、余弦或 sigmoid beta 调度，以及信噪比随时间变化的诊断图表的 skill。

## 练习

1. **（简单）** 可视化前向过程：取一张图像，绘制 `t in [0, 100, 250, 500, 750, 1000]` 时的 `x_t`。验证 `x_1000` 看起来像纯高斯噪声。
2. **（中等）** 在合成圆形数据集上训练 TinyUNet，共 20 个 epoch，采样 16 个圆形。比较 DDPM（1000 步）和 DDIM（50 步）采样——从相同噪声种子出发，它们是否产生相似的图像？
3. **（困难）** 实现余弦噪声调度（Nichol & Dhariwal, 2021）：`alpha_bar_t = cos^2((t/T + s) / (1 + s) * pi / 2)`。用线性和余弦调度训练同一模型，展示余弦在低步数下产生更好的样本。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|----------------------|
| 前向过程 | "随时间加噪" | 固定的 Markov 链，在 T 步内将图像破坏为高斯噪声 |
| 反向过程 | "逐步去噪" | 从噪声走回图像的学习分布 |
| Epsilon 预测 | "预测噪声" | 训练目标：`epsilon_theta(x_t, t)` 预测在步骤 t 添加的噪声 |
| Beta 调度 | "噪声量" | T 个小的方差序列，定义每步输入多少噪声 |
| alpha_bar_t | "累积保留因子" | 截至时间 t 的 (1 - beta_s) 乘积；t 越大意味着剩余信号越少 |
| DDPM 采样器 | "ancestral, 随机" | 从条件高斯分布采样每个 x_{t-1}；1000 步 |
| DDIM 采样器 | "确定性, 快" | 将采样重写为确定性 ODE；20-100 步具有相似的质量 |
| 时间条件 | "告诉模型是哪个 t" | 注入 U-Net 的 t 正弦嵌入，使其知道噪声水平 |

## 延伸阅读

- [Denoising Diffusion Probabilistic Models (Ho et al., 2020)](https://arxiv.org/abs/2006.11239)——使扩散模型实用并在 FID 上击败 GAN 的论文
- [Improved DDPM (Nichol & Dhariwal, 2021)](https://arxiv.org/abs/2102.09672)——余弦调度和 v-参数化
- [DDIM (Song, Meng, Ermon, 2020)](https://arxiv.org/abs/2010.02502)——使实时推理成为可能的确定性采样器
- [Elucidating the Design Space of Diffusion (Karras et al., 2022)](https://arxiv.org/abs/2206.00364)——每个扩散设计选择的统一视图；当前最佳参考
