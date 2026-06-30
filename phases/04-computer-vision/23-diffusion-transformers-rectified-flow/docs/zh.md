# 扩散 Transformer 与 Rectified Flow

> 用 Transformer 替换 U-Net 作为去噪骨架。将扩散公式从 SDE 重新定义为整流 ODE。这是每一个现代图像生成模型的内核。

**类型：** Build
**语言：** Python
**先修要求：** Phase 4 Lesson 10 (Diffusion), Phase 4 Lesson 14 (ViT), Phase 7 Lesson 02 (Self-Attention)
**时间：** ~75 分钟

## 学习目标

- 从 U-Net 扩散过渡到基于 Transformer 的扩散（DiT）：针对空间 Token 序列的自注意力 + AdaLN 时间条件
- 解释 Rectified Flow 以及为什么它与 DDPM 不同，但使用相同的 epsilon-prediction 目标
- 实现一个极小的 DiT（适配 CPU），在合成数据上训练，并比较 DDPM vs Rectified Flow 时间表
- 区分 SD3、FLUX、HunyuanDiT——在 2026 年何时使用每个

## 问题

扩散模型自 2020 年以来使用 U-Net。这个架构在视觉上很高效但有两个限制：它不能很好地扩展到超高分辨率（每个 U-Net 层固定在特定空间尺度），并且它不重用为 NLP 开发的 Transformer 硬件优化（FlashAttention、序列并行）。

DiT（Peebles & Xie, 2022）表明：可以用标准的 ViT Transformer 替代 U-Net，只要保持三个模式——"patchify"图像，通过 AdaLN 注入时间，并用标准扩散损失训练。SD3（Stability AI, 2024）将其扩展到 80 亿参数，使用多模态 Transformer。FLUX（Black Forest Labs, 2024）进一步推进到 120 亿参数。

与此同时，Rectified Flow（Liu 等, 2022）将去噪重新定义为在两个分布之间的直线路径上学习一个速度场。它与扩散几乎相同，但使用略简单的时间调度，并在较少的采样步骤下训练得稍好。SD3 和 FLUX 都使用 Rectified Flow 而非 DDPM。即使你调用 `diffusers.DDPMScheduler`，在 SD3 下它也正运行 Rectified Flow。

## 概念

### 从 U-Net 到 Transformer

U-Net 在空间上压缩然后膨胀；Transformer 将图像展开为 Token 并全部到全部地注意。

```
U-Net 扩散：    图像 (C, H, W) -> U-Net -> 噪声预测 (C, H, W)
DiT 扩散：      图像 (C, H, W) -> patchify -> Token (N, D) -> N 个 Transformer 块 -> 噪声 token (N, D) -> unpatchify -> (C, H, W)
```

DiT 中不存在空间下采样。每个 Token 在每个块中被平等处理。这消除了 U-Net 中最大的架构决策（在每个 scale 中有多少层）并用数量替代——更多层和维度。

### Rectified Flow 一句话

DDPM 训练模型预测在每个扩散时间步 `t` 添加的噪声：`epsilon_theta(x_t, t)`。Rectified Flow 训练模型预测速度 `v = d x_t / dt`，其中 `x_t` 沿 `x_0`（真实图像）和 `x_1`（纯噪声）之间的直线路径。

```
x_t = (1 - t) * x_0 + t * epsilon

Rectified Flow 目标：v_theta(x_t, t) 预测 epsilon - x_0
```

在推理时，通过 Euler 步积分速度场：`x_{t - dt} = x_t - dt * v_theta(x_t, t)`。

在实践中相同：两者训练 Transformer 预测一个张量。区别在于时间路径（直线 vs 弯曲）、目标符号以及采样的确定性程度。Rectified Flow 在约 10-20 步内样本稍好；在 50 步以上相同。

### AdaLN 条件

U-Net 将时间 `t` 作为每个分辨率的附加张量注入。DiT 使用自适应层归一化（AdaLN）：`t` 通过 MLP 确定每个 Transformer 块的比例和偏置系数。

```
scale, shift, gate = t_mlp(t_emb)
x = x + gate * attn(scale * layer_norm(x) + shift)
```

与常规层归一化相同的形状，但 `scale` 和 `shift` 是依赖于 `t` 的向量。每个 DiT 层都有 AdaLN。

### SD3 和 FLUX 中的文本编码器

SD3 和 FLUX 使用两种文本表示：
- **CLIP-L**（或 SigLIP）——用于整体语义和风格的池化嵌入。
- **T5-XXL**——用于逐 Token 细节的序列级嵌入。

两者通过交叉注意力注入到 DiT 中。T5 带来参数计数激增，但提供了主导结果的字幕遵循性。FLUX Pro 使用 T5-XXL（11B 参数文本编码器，用于 12B 参数 DiT）。

### 无分类器引导仍然成立

CFG（Lesson 11）的工作方式完全相同：联合训练条件和无条件去噪，10% dropout 在文本上，在 `w=3.5-7.5` 处推理（DiT 通常优选比 U-Net 更低的 `w`）。

### Consistency、Turbo、Schnell、LCM

蒸馏技术将扩散模型减少到 1-4 步：
- **一致性模型**（Song 等, 2023）——训练模型映射任意噪声水平到干净输出。
- **LCM**（Luo 等, 2023）——用于 Stable Diffusion U-Net 的 LoRA 适配器，1-4 步采样。
- **Turbo**（FLUX）——1 步 DiT 推理。由 FLUX Schnell / FLUX Dev 使用。
- **Schnell**——FLUX 的蒸馏 4 步变体，开源。

质量降级约为标准质量的 10-20%，但延迟为亚秒级。权衡：质量或速度，如常。

### 2026 年模型景观

- **FLUX.2 / FLUX Pro**（Black Forest Labs）——120 亿参数 DiT，Rectified Flow，SOTA 文本到图像。闭源。
- **SD3.5 Large / Medium**（Stability AI）——80 亿 / 20 亿参数 DiT，开权重。
- **HunyuanDiT**（Tencent）——开源 130 亿参数 DiT。"普通话原生"文本编码器。
- **PixArt-Alpha / Sigma**（Huawei）——更小（6 亿参数），开源，快速。好基线。
- **Wan-DiT**（Alibaba）——视频 DiT，开源，Rectified Flow。

2026 年新项目的默认是 SD3.5 Medium 用于微调，FLUX.2 用于最高质量，HunyuanDiT 用于中文或其他非英语。

### 为什么这个 Phase 转变重要

从 Phase 4 纯视觉学习者的视角来看，DiT 是集成点：U-Net（Lesson 7）与 Transformer 块（Lesson 14）相遇，时间条件（Lesson 10）与交叉注意力（Lesson 11）相遇，与从 NLP 借用的训练模式相遇。一旦你训练了 DiT，你就同时理解了图像和语言 Transformer。

## Build It

### 步骤 1：带 AdaLN 的 DiT 块

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaLNZero(nn.Module):
    def __init__(self, dim, t_dim=128):
        super().__init__()
        self.t_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_dim, dim * 6),
        )

    def forward(self, t_emb):
        chunks = self.t_mlp(t_emb).chunk(6, dim=-1)
        return dict(zip(["shift_msa", "scale_msa", "gate_msa",
                         "shift_mlp", "scale_mlp", "gate_mlp"], chunks))


class DiTBlock(nn.Module):
    def __init__(self, dim=64, num_heads=2, t_dim=128):
        super().__init__()
        self.adaln = AdaLNZero(dim, t_dim)
        self.ln1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim),
        )

    def forward(self, x, t_emb):
        adaln = self.adaln(t_emb)
        # MSA
        ms = adaln["scale_msa"][:, None, :]
        sh = adaln["shift_msa"][:, None, :]
        a, _ = self.attn(self.ln1(x) * (1 + ms) + sh,
                         self.ln1(x) * (1 + ms) + sh,
                         self.ln1(x) * (1 + ms) + sh,
                         need_weights=False)
        x = x + adaln["gate_msa"][:, None, :] * a
        # MLP
        ms = adaln["scale_mlp"][:, None, :]
        sh = adaln["shift_mlp"][:, None, :]
        b = self.mlp(self.ln2(x) * (1 + ms) + sh)
        x = x + adaln["gate_mlp"][:, None, :] * b
        return x
```

AdaLN 遵循 SD3 论文的零初始化版本：gate 参数初始化为零，使 DiT 在第一次前向传播时成为恒等映射。

### 步骤 2：微型 DiT

```python
def timestep_embedding(t, dim=128):
    half = dim // 2
    freqs = torch.exp(-torch.log(torch.tensor(10000.0)) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=-1)


class TinyDiT(nn.Module):
    def __init__(self, in_channels=3, dim=64, depth=4, heads=4, patch=4, img=32):
        super().__init__()
        self.patch = patch
        self.img = img
        grid = img // patch
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, grid * grid, dim))
        self.blocks = nn.ModuleList([DiTBlock(dim, heads) for _ in range(depth)])
        self.head = nn.Linear(dim, in_channels * patch * patch)

    def forward(self, x, t):
        x = self.proj(x).flatten(2).transpose(1, 2)  # (N, P, dim)
        x = x + self.pos
        t_emb = timestep_embedding(t)
        for blk in self.blocks:
            x = blk(x, t_emb)
        x = self.head(x)  # (N, P, dim_out)
        return self._unpatchify(x)

    def _unpatchify(self, x):
        n, p, _ = x.shape
        return x.view(n, self.img // self.patch, self.img // self.patch, self.patch, self.patch, -1).permute(0, 5, 1, 3, 2, 4).reshape(n, -1, self.img, self.img)
```

模型有约 900k 参数。在 32x32 输入上进行十秒 CPU 训练即可。

### 步骤 3：Rectified Flow 训练

```python
def rectified_flow_train_step(model, x0, optimizer, device):
    model.train()
    x0 = x0.to(device)
    bs = x0.size(0)
    t = torch.rand(bs, device=device)
    noise = torch.randn_like(x0)
    x_t = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * noise
    v_target = noise - x0
    v_pred = model(x_t, t)
    loss = F.mse_loss(v_pred, v_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
```

`t` 从 Uniform(0, 1) 采样而非 DDPM 中的离散 `0..T`。目标是速度 `noise - x0` 而非噪声本身。

### 步骤 4：Euler 采样器

```python
@torch.no_grad()
def rectified_flow_sample(model, shape, steps=10, device="cpu"):
    model.eval()
    x = torch.randn(shape, device=device)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((shape[0],), i * dt, device=device)
        v = model(x, t)
        x = x - dt * v
    return x
```

10 步 Euler 给出第一个可识别样本。50 步接近全质量。

### 步骤 5：端到端冒烟测试

```python
def synthetic_blobs(size=32, blobs=3, seed=0):
    rng = torch.Generator().manual_seed(seed)
    img = torch.zeros(1, 3, size, size)
    for _ in range(blobs):
        x = torch.randint(size // 4, 3 * size // 4, (1,), generator=rng)
        y = torch.randint(size // 4, 3 * size // 4, (1,), generator=rng)
        c = torch.rand(3, generator=rng)
        img[:, :, y - 4:y + 4, x - 4:x + 4] = c.view(3, 1, 1)
    return img

data = torch.cat([synthetic_blobs() for _ in range(300)])
model = TinyDiT()
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
for step in range(500):
    idx = torch.randint(0, len(data), (16,))
    loss = rectified_flow_train_step(model, data[idx], opt, "cpu")
    if step % 100 == 0:
        print(f"步骤 {step:4d}  损失 {loss:.4f}")
samples = rectified_flow_sample(model, (4, 3, 32, 32), steps=20)
```

在仅 CPU 的合成斑点上的 500 步应产生带有合理斑点状纹理的噪声图像——不太逼真，但在数学上是正确的。

## Use It

2026 年生产使用模式：

- **FLUX.2 Pro API**（Black Forest Labs）——最高质量文本到图像。非开源。
- **SD3.5**（Stability AI）——开源 DiT 权重，社区微调。
- **HunyuanDiT**——开源，普通话原生，125 亿参数。
- **PixArt-Sigma**——6 亿参数 DiT，学术/小型项目基线。

微调路径：Hugging Face `diffusers` 对 SD3.5 的支持已成熟，包括 LoRA、ControlNet 适配器。典型的自定义 DiT 微调在 1 张 A100 上消耗约 10-20 GB 的 VRAM，batch size 为 1，开启梯度检查点。

## Ship It

本课产出：

- `outputs/prompt-dit-variant-picker.md`——根据分辨率、文本条件和 compute budget 在 U-Net 扩散 / DiT / Rectified Flow 之间选择的 prompt。
- `outputs/skill-ffl-sampling-script.md`——编写 Rectified Flow + Euler 采样在任意扩散检查点之上的脚本的 skill。

## 练习

1. **（简单）** 在合成 blob 数据上将 TinyDiT 训练 200 步 DDPM 和 200 步 Rectified Flow。报告每个的最终 MSE。哪个在低步骤下产生更好的样本？
2. **（中等）** 移除 AdaLN 并用每个块的静态可学习增益/偏置替换。比较有无 AdaLN 的收敛曲线。
3. **（困难）** 从 Hugging Face 加载 SD3.5-M 检查点，用 `diffusers` 分析其 DiT 结构，并在 20 个图像上比较 DDIM 50 步 vs Rectified Flow 10 步采样。在不同提示上逐图像报告质量差异。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|----------------------|
| DiT | "扩散 Transformer" | Transformer 驱动的扩散模型；patchify，AdaLN，标准自注意力 |
| Rectified Flow | "直线路径" | 扩散重写为沿线性路径的常速 ODE；更简单的调度，更快的采样 |
| AdaLN | "自适应层归一化" | 时间条件 `t` 通过 MLP 确定每层的 scale 和 shift；在所有 DiT 块中 |
| SD3 | "Stability Diffusion 3" | 2024 年开权重 DiT；使用 Rectified Flow + T5 文本编码器 |
| FLUX | "Black Forest Labs DiT" | 2024 年 120 亿参数 DiT；当前 SOTA 文本到图像质量 |
| HunyuanDiT | "腾讯 DiT" | 2024 年开源 130 亿参数 DiT，普通话原生 |
| Consistency / Turbo | "1-4 步采样" | 蒸馏技术将 DiT 采样从 50 步减少到 1-4 步；±20% 质量 |

## 延伸阅读

- [DiT: Scalable Diffusion Models with Transformers (Peebles & Xie, 2023)](https://arxiv.org/abs/2212.09748)
- [SD3 paper (Esser et al., 2024)](https://arxiv.org/abs/2403.03206)
- [FLUX announcement (Black Forest Labs, 2024)](https://blackforestlabs.ai/announcing-black-forest-labs/)
- [Flow Matching for Generative Modeling (Lipman et al., 2023)](https://arxiv.org/abs/2210.02747)——Rectified Flow 的理论基础
