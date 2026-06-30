# 视觉编码器：Patch

> 一个读取像素的视觉模型需要一个像素的分词器。Patch embedding 就是那个分词器。将图像切割为方块网格，展平每个方块，通过一个线性层投影，然后添加 2D 位置信号，使 transformer 知道每个方块在原始图像中的位置。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 19 lessons 30-37 (Track B foundations)
**Time:** ~90 minutes

## Learning Objectives

- 将图像分词为固定长度的 patch embedding 序列。
- 实现基于 `Conv2d` 的 patch 投影，在数学上与 unfold-then-linear 匹配。
- 构建确定性 2D 正弦位置嵌入，使 token 顺序编码空间位置。
- 在合成 fixture 上验证 patch 数量、嵌入形状和 Conv2d/unfold 等价性。

## The Problem

transformer 消耗的是一个向量序列。一张图像是一个 3 通道网格。将每个像素作为 token 读取会使序列长度爆炸：一张 224x224 的 RGB 图像是 150,528 个 token，一个 12 层 transformer 在注意力上无法承受。将图像作为一个巨大的平坦向量读取则丢弃了局部性，注意力层无法从中恢复。编码器前端的工作是将像素网格压缩为几百个 token，每个 token 总结一个方形区域。

Patch embedding 通过一次线性投影解决这个问题。一张 224x224 的图像按 16x16 的 patch 切割产生一个 14x14 网格的 196 个 patch。每个 patch 从 `(3, 16, 16) = 768` 个像素值展平为一个向量，然后一个线性层将其映射到模型的隐藏维度。transformer 看到 196 个维度为 `hidden`（通常为 768）的 token 加上一个 CLS token。这是一个网络其余部分可以处理的序列。

## The Concept

```mermaid
flowchart LR
  Image[224x224x3 image] --> Cut[cut into 16x16 patches]
  Cut --> Grid[14x14 grid of patches]
  Grid --> Flatten[flatten each patch]
  Flatten --> Proj[linear projection]
  Proj --> Tokens[196 tokens of dim hidden]
  Tokens --> Pos[add 2D sinusoidal position]
  Pos --> Out[final token sequence]
```

### 为什么是 patch，而不是像素

注意力在序列长度上是二次的。一个 196 token 序列每头每层耗费 `196 * 196 = 38,416` 个注意力分数；一个 150,528 token 序列耗费 `150,528 * 150,528 = 226 亿`。Patch 带来了注意力计算 590,000 倍的减少，而一个 16x16 的区域包含了足够的高级视觉任务信号。代价是丢失了一个 patch 内部的细粒度空间细节，这就是为什么下游多模态栈在精细定位重要时经常运行第二个高分辨率分支。

### 为什么一个线性投影就够了

每个 patch 被视为独立向量。投影学习一个基：边缘检测器、颜色过滤器、简单纹理。单个线性层很小（ViT-Base 的 `768 * 768 = 589,824` 个参数）且训练快速。存在更深的卷积茎（"混合" ViT），但平坦线性投影是标准，大多数现代开放权重编码器都以此形状发布。

### `Conv2d` 技巧

一个 `Conv2d(in_channels=3, out_channels=hidden, kernel_size=patch_size, stride=patch_size)` 不带填充给出与 unfold-then-linear 相同的数值结果，因为每个输出位置将 patch 像素与一个滤波器进行点积。卷积就是 patch 投影，大多数生产代码库都这样发布，因为它在 GPU 上更快且少用一次 reshape。

### 位置嵌入

Token 在投影之后不携带顺序。2D 正弦嵌入给每个 token 一个固定信号，编码其 `(row, col)` 位置。嵌入维度的一半用多个频率的 sin/cos 编码行位置；另一半编码列位置。编码是确定性的，因此你可以在不重新训练的情况下交换分辨率，并且它可以干净地插值到模型在训练时从未见过的网格尺寸。

| Component | Shape | Parameters |
|-----------|-------|------------|
| Patch projection (`Conv2d`) | `(hidden, 3, patch, patch)` | `3 * P * P * hidden + hidden` |
| Position embedding (fixed) | `(num_patches, hidden)` | 0 (computed, not learned) |
| CLS token (learned) | `(1, hidden)` | `hidden` |

对于 224 分辨率的 ViT-Base/16：投影中 590,592 个参数，CLS token 中 768 个，正弦位置为零。下一课（59）在此前端之上堆叠一个 12 层 transformer。

### 等价性作为健全检查

patch 步骤有两种写法：`Conv2d` 投影和显式的 unfold-then-linear。对于相同的权重，它们必须产生相同的输出。如果不是，那么 unfold 数学是错误的，编码器的其余部分就建立在沙子上。本课的测试验证这种等价性。

## Build It

`code/main.py` implements:

- `PatchEmbed`, an `nn.Module` wrapping `Conv2d` for patch projection.
- `sinusoidal_2d(grid_h, grid_w, dim)`, a stateless function that builds the 2D position table.
- `VisionFrontEnd`, which composes patch embedding, CLS prepend, and position addition into one forward pass.
- A `synthesize_image(seed)` helper that builds a deterministic 224x224x3 fixture from `numpy.random`.
- A demo that runs one fixture image through the front end and prints the output shape, the CLS token norm, and one row of the position embedding.

Run it:

```bash
python3 code/main.py
```

Output: the 224x224 fixture is tokenized to a sequence of shape `(1, 197, 768)`. The first token is the CLS; the next 196 are patch tokens. The position embedding norms are uniform within a row, which is the sinusoidal signature.

## Use It

The same patch front end shows up in every modern vision-language model: CLIP ViT-L/14, SigLIP, DINOv2, the Qwen-VL family, and the InternVL stack all start from a `Conv2d` patch projection plus a position signal. Differences across families live downstream (CLS vs no-CLS pooling, register tokens, varying patch sizes 14 vs 16, dynamic resolution via interpolated positions). The frontend in this lesson is the substrate every one of those models stands on.

## Tests

`code/test_main.py` covers:

- patch count matches `(image_size / patch_size) ** 2`
- output shape matches `(batch, num_patches + 1, hidden)`
- the `Conv2d` projection equals manual unfold-then-linear on a small fixture
- sinusoidal position table is deterministic across calls
- CLS token broadcasts across batch dim without leakage

Run them:

```bash
python3 -m unittest code/test_main.py
```

## Exercises

1. Replace the sinusoidal position with a learned `nn.Parameter` and compare the first-epoch loss on a tiny synthetic classification task. Learned positions win at fixed resolution; sinusoidal wins when you change resolution after training.

2. Swap the `Conv2d` for an explicit `nn.Unfold` plus `nn.Linear` and assert the outputs match to within float tolerance. Same math, two ways to spell it.

3. Add support for non-square patch sizes (e.g. 32x16 for wide-aspect inputs) and verify the position table handles non-square grids.

4. Profile the patch step at batch sizes 1, 8, 64. The patch projection is rarely the bottleneck; the attention layers downstream dominate.

5. Train the front end as a frozen feature extractor on a 4-class synthetic shape dataset (circles, squares, triangles, stars). The CLS token output should linearly separate.

## Key Terms

| Term | What it means |
|------|---------------|
| Patch | 图像的一个方形子区域，通常为 14x14 或 16x16 |
| Patch embedding | 将一个展平的 patch 线性投影到隐藏维度 |
| Sequence length | Patch 分词后的 token 数量，通常加上 CLS |
| Sinusoidal position | 编码 2D 网格坐标的固定 sin/cos 信号 |
| CLS token | 前置到序列上的可学习向量，用作池化头 |

## Further Reading

- An Image is Worth 16x16 Words (ViT, 2021) for the original patch-embed framing.
- Attention Is All You Need (2017) for the sinusoidal position formula adapted here to 2D.
- DINOv2 paper for register tokens, an extension you can add as exercise 6.
