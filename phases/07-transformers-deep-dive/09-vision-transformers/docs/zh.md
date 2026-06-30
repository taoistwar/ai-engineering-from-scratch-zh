# 视觉 Transformer（ViT）

> 图像是补丁的网格。句子是 token 的网格。同一个 transformer 两者都能处理。

**类型：** 构建
**语言：** Python
**前置条件：** 第七阶段 · 05（完整 Transformer），第四阶段 · 03（CNN），第四阶段 · 14（视觉 Transformer 导论）
**时间：** 约 45 分钟

## 问题

2020 年之前，计算机视觉意味着卷积。ImageNet、COCO 和检测基准上的每个 SOTA 都使用 CNN 骨干。Transformer 是用于语言的。

Dosovitskiy 等人（2020）——"An Image is Worth 16x16 Words"——表明你可以完全抛弃卷积。将图像切成固定大小的补丁，将每个补丁线性投影为嵌入，将序列输入普通的 transformer 编码器。在足够的规模下（ImageNet-21k 预训练或更大），ViT 匹敌或击败基于 ResNet 的模型。

ViT 是 2026 年更广泛模式的开始：一种架构，多种模态。Whisper 将音频分词化。ViT 将图像分词化。动作 token 用于机器人学。像素 token 用于视频。transformer 不关心——输入一个序列，它就会学习。

到 2026 年，ViT 及其后代（DeiT、Swin、DINOv2、ViT-22B、SAM 3）主导了大部分视觉领域。CNN 在边缘设备和延迟敏感任务上仍然胜出。其他一切都在技术栈的某个位置有一个 ViT。

## 概念

![图像 → 补丁 → token → transformer](../assets/vit.svg)

### 步骤 1 — 补丁化

将 `H × W × C` 图像分割为 `N × (P·P·C)` 扁平补丁序列。典型设置：`224 × 224` 图像，`16 × 16` 补丁 → 196 个补丁，每个 768 个值。

```
image (224, 224, 3) → 14 × 14 网格的 16x16x3 补丁 → 196 个长度为 768 的向量
```

补丁大小是杠杆。更小的补丁 = 更多的 token，更好的分辨率，二次注意力成本。更大的补丁 = 更粗糙，更便宜。

### 步骤 2 — 线性嵌入

一个单一的学习矩阵将每个扁平补丁投影到 `d_model`。等效于内核大小为 `P` 且步幅为 `P` 的卷积。在 PyTorch 中，这实际上是 `nn.Conv2d(C, d_model, kernel_size=P, stride=P)`——一个 2 行的实现。

### 步骤 3 — 前置 `[CLS]` token，添加位置嵌入

- 前置一个可学习的 `[CLS]` token。其最终隐藏状态是用于分类的图像表示。
- 添加可学习的位置嵌入（ViT 原始）或正弦 2D（后来的变体）。
- 在 2024+ 中，RoPE 扩展到 2D 用于位置，有时无需显式嵌入。

### 步骤 4 — 标准 transformer 编码器

堆叠 L 个 `LayerNorm → Self-Attention → + → LayerNorm → MLP → +` 的模块。与 BERT 相同。没有视觉特定的层。这是该论文教学上的点睛之笔。

### 步骤 5 — 头

对于分类：取 `[CLS]` 隐藏状态 → 线性 → softmax。对于 DINOv2 或 SAM，丢弃 `[CLS]`，直接使用补丁嵌入。

### 重要的变体

| 模型 | 年份 | 变化 |
|-------|------|--------|
| ViT | 2020 | 原始版本。固定补丁大小，完整的全局注意力。 |
| DeiT | 2021 | 蒸馏；仅在 ImageNet-1k 上可训练。 |
| Swin | 2021 | 具有移位窗口的层次结构。固定了次二次方成本。 |
| DINOv2 | 2023 | 自监督（无标签）。最佳通用视觉特征。 |
| ViT-22B | 2023 | 22B 参数；扩展律适用。 |
| SigLIP | 2023 | ViT + 语言对，sigmoid 对比损失。 |
| SAM 3 | 2025 | 分割一切；ViT-Large + 可提示掩码解码器。 |

### 为什么花了一些时间

ViT 需要*大量*数据才能匹敌 CNN，因为它没有 CNN 的归纳偏置（平移不变性、局部性）。没有 >1 亿标注图像或强大的自监督预训练，CNN 在匹配计算量下仍然胜出。DeiT 在 2021 年通过蒸馏技巧修复了这一点；DINOv2 在 2023 年通过自监督永久修复了这一点。

## 动手构建

参见 `code/main.py`。纯标准库补丁化 + 线性嵌入 + 健全性检查。没有训练——任何现实规模的 ViT 都需要 PyTorch 和数小时的 GPU 时间。

### 步骤 1：伪造图像

一个 24 × 24 RGB 图像，作为 `(R, G, B)` 元组行的列表。我们使用 6×6 补丁 → 16 个补丁，每个 108 维嵌入向量。

### 步骤 2：补丁化

```python
def patchify(image, P):
    H = len(image)
    W = len(image[0])
    patches = []
    for i in range(0, H, P):
        for j in range(0, W, P):
            patch = []
            for di in range(P):
                for dj in range(P):
                    patch.extend(image[i + di][j + dj])
            patches.append(patch)
    return patches
```

光栅顺序：网格上的行主序。每个 ViT 都使用这种顺序。

### 步骤 3：线性嵌入

将每个扁平补丁乘以随机 `(patch_flat_size, d_model)` 矩阵。验证在添加 `[CLS]` 后输出形状为 `(N_patches + 1, d_model)`。

### 步骤 4：计算现实 ViT 的参数数量

打印 ViT-Base 的参数计数：12 层，12 头，d=768，patch=16。与 ResNet-50（~25M）比较。ViT-Base 大约 86M。ViT-Large ~307M。ViT-Huge ~632M。

## 使用它

```python
from transformers import ViTImageProcessor, ViTModel
import torch
from PIL import Image

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

img = Image.open("cat.jpg")
inputs = processor(img, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, 197, 768): [CLS] + 196 patches
cls_emb = out[:, 0]                       # 图像表示
```

**DINOv2 嵌入是 2026 年图像特征的默认选择。** 冻结骨干，训练一个小头。适用于分类、检索、检测、描述。Meta 的 DINOv2 检查点在每个非文本视觉任务上优于 CLIP。

**补丁大小选择。** 小型模型使用 16×16（ViT-B/16）。密集预测（分割）使用 8×8 或 14×14（SAM、DINOv2）。非常大的模型使用 14×14。

## 交付成果

参见 `outputs/skill-vit-configurator.md`。该技能根据数据集大小、分辨率和计算预算，为新的视觉任务选择 ViT 变体和补丁大小。

## 练习

1. **简单。** 运行 `code/main.py`。验证补丁数量等于 `(H/P) * (W/P)` 且扁平补丁维度等于 `P*P*C`。
2. **中等。** 实现 2D 正弦位置嵌入——每个补丁的 `row` 和 `col` 有两个独立的正弦编码，拼接在一起。将它们输入一个微型 PyTorch ViT，并在 CIFAR-10 上比较准确率与可学习位置嵌入。
3. **困难。** 构建一个 3 层 ViT（PyTorch），在 1,000 张 MNIST 图像上以 4×4 补丁进行训练。测量测试准确率。现在在相同的 1,000 张图像上添加 DINOv2 预训练（简化版：仅训练编码器从掩码补丁中预测补丁嵌入）。准确率提高了吗？

## 关键术语

| 术语 | 人们怎么说 | 它实际上的含义 |
|------|-----------------|-----------------------|
| 补丁 | "视觉 transformer token" | 图像的 `P × P × C` 区域的扁平像素值向量。 |
| 补丁化 | "切割 + 展平" | 将图像切成不重叠的补丁，将每个展平为向量。 |
| `[CLS]` token | "图像摘要" | 前置的可学习 token；其最终嵌入是图像表示。 |
| 归纳偏置 | "模型假设了什么" | ViT 比 CNN 具有更少的先验；需要更多数据来弥补差距。 |
| DINOv2 | "自监督 ViT" | 使用图像增强 + 动量教师进行无标签训练。2026 年最佳通用图像特征。 |
| SigLIP | "CLIP 的后继者" | 使用 sigmoid 对比损失训练的 ViT + 文本编码器；在匹配计算量下优于 CLIP。 |
| Swin | "窗口化 ViT" | 具有局部注意力 + 移位窗口的层次化 ViT；次二次方复杂度。 |
| 寄存器 token | "2023 年的技巧" | 一些额外的可学习 token 吸收注意力汇；改进了 DINOv2 特征。 |

## 延伸阅读

- [Dosovitskiy 等人 (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) — ViT 论文。
- [Touvron 等人 (2021). Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) — DeiT。
- [Liu 等人 (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) — Swin。
- [Oquab 等人 (2023). DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) — DINOv2。
- [Darcet 等人 (2023). Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588) — DINOv2 的寄存器 token 修复。
