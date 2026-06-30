# 视觉 Transformer (ViT)

> 将图像切成 patch，将每个 patch 视为一个词，运行标准 Transformer。别回头。

**类型：** Build
**语言：** Python
**先修要求：** Phase 7 Lesson 02 (Self-Attention), Phase 4 Lesson 04 (Image Classification)
**时间：** ~45 分钟

## 学习目标

- 从头实现 patch 嵌入、可学习位置嵌入、类别 Token 和 Transformer 编码器块来构建最小 ViT
- 解释为什么 ViT 曾被认为需要海量预训练数据，直到 DeiT 和 MAE 证明了相反
- 在架构先验（无、局部窗口注意力、卷积骨干）上比较 ViT、Swin 和 ConvNeXt
- 使用 `timm` 和标准线性探测/微调配方在小规模数据集上微调预训练 ViT

## 问题

十年来，卷积与计算机视觉是同义词。CNN 具有强大的归纳偏置——局部性、平移等变性——没有人认为能被替代。然后 Dosovitskiy 等（2020）展示了将纯 Transformer 应用于展平的图像 patch，完全不用卷积机制，可以在大规模上匹敌甚至击败最佳 CNN。

问题是"在大规模上"。ImageNet-1k 上的 ViT 输给了 ResNet。在 ImageNet-21k 或 JFT-300M 上预训练然后在 ImageNet-1k 上微调的 ViT 击败了它。结论是 Transformer 缺乏有用的先验，但能从足够多的数据中学习它们。后续工作（DeiT、MAE、DINO）表明，使用正确的训练配方——强力数据增强、自监督预训练、蒸馏——ViT 在小数据上也能训练得很好。

到 2026 年，纯 CNN 在边缘设备上仍然有竞争力（ConvNeXt 最强），但 Transformer 主导了其他所有领域：分割（Mask2Former、SegFormer）、检测（DETR、RT-DETR）、多模态（CLIP、SigLIP）、视频（VideoMAE、VJEPA）。ViT 块结构是你需要了解的。

## 概念

### Pipeline

```mermaid
flowchart LR
    IMG["图像<br/>(3, 224, 224)"] --> PATCH["Patch 嵌入<br/>conv 16x16 s=16<br/>-> (768, 14, 14)"]
    PATCH --> FLAT["展平到<br/>(196, 768) Token"]
    FLAT --> CAT["前置<br/>[CLS] Token"]
    CAT --> POS["添加可学习<br/>位置嵌入"]
    POS --> ENC["N 个 Transformer<br/>编码器块"]
    ENC --> CLS["取 [CLS]<br/>Token 输出"]
    CLS --> HEAD["MLP 分类器"]

    style PATCH fill:#dbeafe,stroke:#2563eb
    style ENC fill:#fef3c7,stroke:#d97706
    style HEAD fill:#dcfce7,stroke:#16a34a
```

七个步骤。Patches -> Token -> 注意力 -> 分类器。每个变体（DeiT、Swin、ConvNeXt、MAE 预训练）改变其中一两个，其余保持不变。

### Patch 嵌入

第一个卷积是秘密。核大小为 16，步长为 16，所以一张 224x224 图像变成一个 14x14 网格的 16x16 patch，每个投影为 768 维嵌入。这一个卷积同时完成了 patch 化和线性投影。

```
输入:  (3, 224, 224)
卷积 (3 -> 768, k=16, s=16, 无填充):
输出: (768, 14, 14)
展平空间: (196, 768)
```

196 个 patches = 196 个 Token。每个 Token 的特征维度是 768（ViT-B）、1024（ViT-L）或 1280（ViT-H）。

### 类别 Token

一个前置到序列的单一可学习向量：

```
tokens = [CLS; patch_1; patch_2; ...; patch_196]   形状 (197, 768)
```

经过 N 个 Transformer 块后，`[CLS]` 输出是全局图像表示。分类头只读取这一个向量。

### 位置嵌入

Transformer 没有内置的空间位置概念。给每个 Token 添加一个可学习向量：

```
tokens = tokens + learned_pos_embedding   (同样形状 (197, 768))
```

嵌入是模型的参数；基于梯度的训练使其适应 2D 图像结构。正弦 2D 替代方案存在但实践中很少使用。

### Transformer 编码器块

标准。多头自注意力、MLP、残差连接、前置 LayerNorm。

```
x = x + MSA(LN(x))
x = x + MLP(LN(x))

MLP 是两层带 GELU：Linear(d -> 4d) -> GELU -> Linear(4d -> d)
```

ViT-B/16 堆叠 12 个这样的块，每个块有 12 个注意头，总计 86M 参数。

### 为什么前置 LN

早期 Transformer 使用后置 LN（`x = LN(x + sublayer(x))`）并且在没有 warmup 的情况下难以训练超过 6-8 层。前置 LN（`x = x + sublayer(LN(x))`）无需 warmup 即可稳定地训练更深的网络。每个 ViT 和每个现代 LLM 都使用前置 LN。

### Patch 大小权衡

- 16x16 patches -> 196 个 Token，标准。
- 32x32 patches -> 49 个 Token，更快但分辨率更低。
- 8x8 patches -> 784 个 Token，更精细但 O(n^2) 注意力成本扩展得很差。

更大的 patches = 更少的 Token = 更快但更少的空间细节。SwinV2 在层次化窗口中使用 4x4 patches。

### DeiT 在 ImageNet-1k 上训练 ViT 的配方

原始 ViT 需要 JFT-300M 来击败 CNN。DeiT（Touvron 等，2020）仅在 ImageNet-1k 上将 ViT-B 训练到 81.8% top-1，做了四个改变：

1. 强力数据增强：RandAugment、Mixup、CutMix、Random Erasing。
2. 随机深度（训练时随机丢弃整个块）。
3. 重复增强（同一张图像在同一个 batch 中采样 3 次）。
4. 从 CNN 教师蒸馏（可选，进一步提升准确率）。

每个现代 ViT 训练配方都源自 DeiT。

### Swin vs ConvNeXt

- **Swin**（Liu 等，2021）——基于窗口的注意力。每个块在局部窗口内做注意力；交替块移动窗口以在窗口之间混合信息。在保持注意力算子的同时重新引入类似 CNN 的局部性先验。
- **ConvNeXt**（Liu 等，2022）——重新设计 CNN 以匹配 Swin 的架构选择（深度可分离卷积、LayerNorm、GELU、倒置瓶颈）。证明了差距不是"注意力 vs 卷积"而是"现代训练配方 + 架构"。

到 2026 年，ConvNeXt-V2 和 Swin-V2 都是生产级；正确的选择取决于你的推理技术栈（ConvNeXt 对边缘编译更好）和预训练语料。

### MAE 预训练

掩码自编码器（He 等，2022）：随机掩码 75% 的 patches，训练编码器只处理可见的 25%，训练一个小解码器从编码器输出重建被掩码的 patches。预训练后，丢弃解码器并微调编码器。

MAE 使 ViT 可仅在 ImageNet-1k 上训练，达到 SOTA，是当前默认的自监督配方。

## Build It

### 步骤 1：Patch 嵌入

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, dim=192, image_size=64):
        super().__init__()
        assert image_size % patch_size == 0
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.num_patches = num_patches

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)
```

一个卷积，一次展平，一次转置。这就是整个图像到 Token 的步骤。

### 步骤 2：Transformer 块

前置 LN、多头自注意力、带 GELU 的 MLP、残差连接。

```python
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x
```

`nn.MultiheadAttention` 处理头拆分、缩放点积和输出投影。`batch_first=True` 使形状为 `(N, seq, dim)`。

### 步驟 3：ViT

```python
class ViT(nn.Module):
    def __init__(self, image_size=64, patch_size=16, in_channels=3,
                 num_classes=10, dim=192, depth=6, num_heads=3, mlp_ratio=4):
        super().__init__()
        self.patch = PatchEmbedding(in_channels, patch_size, dim, image_size)
        num_patches = self.patch.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        x = self.patch(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.ln(x[:, 0])
        return self.head(x)

vit = ViT(image_size=64, patch_size=16, num_classes=10, dim=192, depth=6, num_heads=3)
x = torch.randn(2, 3, 64, 64)
print(f"output: {vit(x).shape}")
print(f"params: {sum(p.numel() for p in vit.parameters()):,}")
```

约 2.8M 参数——一个在 CPU 上可行的微型 ViT。真实的 ViT-B 是 86M；类定义相同，但 `dim=768, depth=12, num_heads=12`。

### 步骤 4：健全检查——单张图像推理

```python
logits = vit(torch.randn(1, 3, 64, 64))
print(f"logits: {logits}")
print(f"probs:  {logits.softmax(-1)}")
```

应该运行无错误。概率和为 1。

## Use It

`timm` 提供每个 ViT 变体，搭载 ImageNet 预训练权重。一行：

```python
import timm

model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=10)
```

`timm` 是 2026 年视觉 Transformer 的生产默认库。支持 ViT、DeiT、Swin、Swin-V2、ConvNeXt、ConvNeXt-V2、MaxViT、MViT、EfficientFormer 以及数十个其他模型，使用相同 API。

对于多模态工作（图像 + 文本），`transformers` 提供 CLIP、SigLIP、BLIP-2、LLaVA。所有这些中的图像编码器都是 ViT 变体。

## Ship It

本课产出：

- `outputs/prompt-vit-vs-cnn-picker.md`——一个基于数据集大小、计算和推理技术栈在 ViT、ConvNeXt 或 Swin 之间选择的 prompt。
- `outputs/skill-vit-patch-and-pos-embed-inspector.md`——一个验证 ViT 的 patch 嵌入和位置嵌入形状与模型预期序列长度匹配，捕捉最常见移植错误的 skill。

## 练习

1. **（简单）** 打印上述微型 ViT 一次前向传播中每个中间张量的形状。确认：输入 `(N, 3, 64, 64)` -> patches `(N, 16, 192)` -> 加 CLS `(N, 17, 192)` -> 分类器输入 `(N, 192)` -> 输出 `(N, num_classes)`。
2. **（中等）** 在与 Lesson 4 相同的合成 CIFAR 数据集上微调预训练 `timm` ViT-S/16。与在相同数据上微调 ResNet-18 比较。报告训练时间和最终准确率。
3. **（困难）** 为微型 ViT 实现 MAE 预训练：掩码 75% 的 patches，训练编码器 + 小解码器重建被掩码的 patches。评估预训练前后合成数据上的线性探测准确率。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|----------------------|
| Patch 嵌入 | "第一个卷积" | 核大小 = 步长 = patch 大小的卷积；将图像转为 Token 嵌入网格 |
| 类别 Token | "[CLS]" | 前置到 Token 序列的可学习向量；其最终输出是全局图像表示 |
| 位置嵌入 | "可学习 pos" | 添加到每个 Token 的可学习向量，使 Transformer 知道每个 patch 来自哪里 |
| 前置 LN | "子层之前的 LayerNorm" | 稳定的 Transformer 变体：`x + sublayer(LN(x))` 而非 `LN(x + sublayer(x))` |
| 多头注意力 | "并行注意力" | 标准 Transformer 注意力拆分为 num_heads 个独立子空间，之后拼接 |
| ViT-B/16 | "Base, patch 16" | 标准尺寸：dim=768, depth=12, heads=12, patch_size=16, image=224；~86M 参数 |
| DeiT | "数据高效 ViT" | 仅在 ImageNet-1k 上用强力增强训练的 ViT；证明大规模预训练数据集不是严格必需的 |
| MAE | "掩码自编码器" | 自监督预训练：掩码 75% 的 patches，重建；主导的 ViT 预训练配方 |

## 延伸阅读

- [An Image is Worth 16x16 Words (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929)——ViT 论文
- [DeiT: Data-efficient Image Transformers (Touvron et al., 2020)](https://arxiv.org/abs/2012.12877)——如何仅在 ImageNet-1k 上训练 ViT
- [Masked Autoencoders are Scalable Vision Learners (He et al., 2022)](https://arxiv.org/abs/2111.06377)——MAE 预训练
- [timm 文档](https://huggingface.co/docs/timm)——你在生产中将要用到的每个视觉 Transformer 的参考文档
