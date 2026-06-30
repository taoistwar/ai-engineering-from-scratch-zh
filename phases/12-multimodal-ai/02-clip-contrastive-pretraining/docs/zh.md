# CLIP 与对比视觉-语言预训练

> OpenAI 的 CLIP（2021）证明了一个足以驱动接下来五年的核心思想：仅使用网络上嘈杂的图像-标题配对和对比损失，将图像编码器与文本编码器对齐到同一向量空间中。零有监督标签。400M 配对。由此产生的嵌入空间可以进行零样本分类、图文检索，并接入每一款 2026 年 VLM 中作为其视觉塔。SigLIP 2（2025）将 softmax 替换为 sigmoid，以更低的成本超越了 CLIP。本课将从 InfoNCE 到 sigmoid 逐对损失的数学推导，并用 Python 标准库实现训练步骤。

**Type:** Build
**Languages:** Python (stdlib, InfoNCE + sigmoid loss implementations)
**Prerequisites:** Phase 12 · 01 (ViT patches), Phase 7 (Transformers)
**Time:** ~180 minutes

## 学习目标

- 从互信息推导 InfoNCE 损失，并实现数值稳定的向量化版本。
- 解释为什么 sigmoid 逐对损失（SigLIP）能扩展到 batch 32768+ 规模，而没有 softmax 所需的 all-gather 开销。
- 通过构造文本模板（`a photo of a {class}`）并对余弦相似度取 argmax 来实现零样本 ImageNet 分类。
- 列举 CLIP / SigLIP 预训练提供的四个杠杆：batch 大小、温度、提示模板、数据质量。

## 问题

CLIP 之前的视觉是有监督的。收集有标签数据集（ImageNet：1.2M 图像，1000 类），训练 CNN，发布。标签很昂贵，标签会偏向标注者能达成一致的内容，而且不经过微调就无法迁移到新任务。

网络上的图文对提供了超过十亿的免费、松散的标注配对。一张金毛寻回犬的照片，配以替代文本 "my dog Max in the park"，承载着监督信号——文本描述了图像。问题在于：能否将其转化为有用的训练？

CLIP 的答案：将图文对视为一个匹配任务。给定一个包含 N 张图像和 N 条标题的 batch，学习将每张图像与其自身的标题匹配，而对 N-1 个干扰项不匹配。监督信号是"这两者属于一起；其余 N-1 对不是"。没有类别标签。没有人工标注。只有一个对比损失。

由此产生的嵌入空间能做到超出 CLIP 训练目标的事情。ImageNet 零样本分类之所以有效，是因为 "a photo of a cat" 的嵌入接近那些从未被显式标记为猫的猫的照片。这是催生了每个 2026 年 VLM 的赌注。

## 概念

### 双编码器

CLIP 有两个塔：

- 图像编码器 `f`：ViT 或 ResNet，每张图像输出一个 D 维向量。
- 文本编码器 `g`：小型 Transformer，每条标题输出一个 D 维向量。

两个塔都将输出归一化为单位长度。相似度为 `cos(f(x), g(y)) = f(x)^T g(y)`，因为两者都是单位范数。

对于包含 N 对（图像，标题）的 batch，构建形状为 `(N, N)` 的相似度矩阵 `S`：

```
S[i, j] = cos(f(x_i), g(y_j)) / tau
```

其中 `tau` 是一个可学习的温度参数（CLIP 初始化为 0.07；在 log 空间中学习）。

### InfoNCE 损失

CLIP 使用行和列上的对称交叉熵：

```
loss_i2t = CE(S, labels=identity)     # 每张图像的正样本是其自身的标题
loss_t2i = CE(S^T, labels=identity)   # 每条标题的正样本是其自身的图像
loss = (loss_i2t + loss_t2i) / 2
```

这就是 InfoNCE。交叉熵中的 softmax 强制每张图像比 batch 中的其他所有标题更匹配自己的标题。"负样本"是 batch 中的所有其他项。更大的 batch = 更多负样本 = 更强的信号。CLIP 以 batch 32k 进行训练；规模很重要。

### 温度

`tau` 控制 softmax 的锐度。低 tau → 尖锐分布，硬负样本挖掘效果。高 tau → 平滑，所有样本都有贡献。CLIP 学习 `log(1/tau)`，并裁剪以防止坍缩。SigLIP 2 使用固定的初始 tau，并改用可学习的偏置。

### 为什么 Sigmoid 扩展性更好（SigLIP）

Softmax 需要整个相似度矩阵同步。在分布式训练中，你必须将所有嵌入 all-gather 到每个副本，然后执行 softmax。这带来了与规模二次方相关的通信开销。

SigLIP 用逐元素 sigmoid 替换 softmax：对于每对 `(i, j)`，损失是一个二分类问题——"这两者是否匹配？"正样本类标签在对角线上，其他所有都是负样本。损失为：

```
L = -1/N sum over (i, j) [ y_ij log sigmoid(S[i,j]) + (1-y_ij) log sigmoid(-S[i,j]) ]
```

`y_ij = 1` 当 `i == j` 时，否则为 0。每对的损失是独立的。不需要 all-gather。每个 GPU 计算自己的局部块并求和。SigLIP 2 可以廉价地扩展到 batch 32k-512k，而 CLIP 需要按比例增加通信量。

### 零样本分类

给定 N 个类名，为每个类构建一个文本模板：

```
"a photo of a {class}"
```

用文本编码器嵌入每个模板。用图像编码器嵌入你的图像。余弦相似度的 argmax = 预测类别。无需在目标类别上进行训练。

提示模板很重要。CLIP 原始论文每类使用了 80 个模板（普通、艺术、照片、绘画等）并平均其嵌入，提升了 3 个 ImageNet 百分点。现代用法通常选择一两个模板。

### 线性探测与微调

零样本是基线。线性探测（在冻结的 CLIP 特征之上训练一个线性层用于目标类别）在域内任务上优于零样本。全微调在域内任务上优于线性探测，但可能损害零样本迁移能力。三种方案，三种权衡。

### SigLIP 2：NaFlex 与密集特征

SigLIP 2（2025）新增了：
- NaFlex：单个模型处理可变的宽高比和分辨率。
- 更好的密集特征用于分割和深度估计，目标是在 VLM 中作为冻结骨干网络使用。
- 多语言：在 100+ 种语言上训练，而 CLIP 仅支持英语。
- 1B 参数规模，而 CLIP 最高为 400M。

在 2026 年的开源 VLM 中，SigLIP 2 SO400m/14 是默认的视觉塔。CLIP 仍然是纯图像-文本检索的默认选择，当特定的 LAION-2B 训练分布与你的查询模式匹配时。

### ALIGN、BASIC、OpenCLIP、EVA-CLIP

ALIGN（Google，2021）：与 CLIP 相同的想法，1.8B 配对规模，90% 噪音。证明了噪音数据的扩展性。OpenCLIP（LAION）：在 LAION-400M / 2B 上对 CLIP 的开源复现，多规模，是首选的开源检查点。EVA-CLIP：从掩码图像建模初始化；是 VLM 的强骨干网络。BASIC：Google 的 CLIP+ALIGN 混合体。都是同一家族，区别在于数据与调参。

### 零样本天花板

CLIP 类模型在 ImageNet 零样本任务上大约达到 76%（CLIP-G，OpenCLIP-G）。超过这个水平需要更大规模的数据（SigLIP 2 达到 80%+）或架构变更（有监督头部、更多参数）。基准正在饱和；真正的价值在于下游 VLM 使用的嵌入空间。

```figure
multimodal-fusion
```

## 使用

`code/main.py` 实现了：

1. 一个玩具双编码器（基于哈希的图像特征，文本字符特征），让你在不使用 numpy 的情况下也能看清 InfoNCE 的结构。
2. 纯 Python 实现的 InfoNCE 损失（通过 log-sum-exp 保持数值稳定性）。
3. 用于对比的 sigmoid 逐对损失。
4. 一个零样本分类程序：计算与一组文本提示的余弦相似度，取 argmax 作为预测。

运行它，观察损失曲线。绝对数值是玩具级的；但曲线形状与真实 CLIP 训练器输出的形状一致。

## 产出

本课产出 `outputs/skill-clip-zero-shot.md`。给定一组图像（通过路径）和一个目标类别列表，它使用 CLIP 模板构建文本提示，用指定的检查点（例如 `openai/clip-vit-large-patch14`）嵌入双方，并返回 top-1 / top-5 预测及相似度分数。该技能拒绝为不在提示列表中的类别做出声明。

## 练习

1. 手工实现 4 对数据的 InfoNCE。构建 4x4 相似度矩阵，执行 softmax，提取对角线，计算交叉熵。将你的 Python 实现与此手工计算进行验证。

2. SigLIP 除了温度参数外还使用了一个偏置参数 `b`：`S'[i,j] = S[i,j]/tau + b`。当 batch 中存在严重的类别不平衡（每行负样本远多于正样本）时，`b` 扮演什么角色？阅读 SigLIP 第 3 节（arXiv:2303.15343）。

3. 构建一个猫 vs 狗零样本分类器。尝试两种提示模板：`a photo of a {class}` 和 `a picture of a {class}`。在 100 张测试图像上测量准确率。模板集成是否优于单个模板？

4. 对于在 batch 32k 上运行的 512 个 GPU，计算 softmax InfoNCE vs sigmoid 逐对损失的通信成本。哪个按 O(N) 扩展，哪个按 O(N^2) 扩展？引用 SigLIP 第 4 节。

5. 阅读 OpenCLIP 规模定律论文（arXiv:2212.07143，Cherti 等人）。从图中的数据扩展结论复现他们的发现：在固定模型大小下，ImageNet 零样本准确率与训练数据规模之间是什么对数线性关系？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| InfoNCE | "对比损失" | 在一个 batch 的相似度矩阵上的交叉熵；每项的"正样本"是其配对的项，"负样本"是其他所有项 |
| Sigmoid 损失 | "SigLIP 损失" | 逐对二分类交叉熵；无需 softmax，无需 all-gather，在分布式训练中扩展廉价 |
| 温度 | "tau" | 在 softmax/sigmoid 之前对 logits 的缩放标量；控制分布的锐度 |
| 零样本 | "无需微调的分类" | 使用文本提示构造类别嵌入，通过余弦相似度进行分类；无需在目标类别上训练 |
| 提示模板 | "a photo of a ..." | 围绕类别名称的文本框架；影响零样本准确率 1-5 个百分点 |
| 双编码器 | "双塔" | 一个图像编码器 + 一个文本编码器，输出在共享的 D 维空间中 |
| 硬负样本 | "难区分的干扰项" | 一个与正样本足够相似的负样本，使模型必须努力将它们分开 |
| 线性探测 | "冻结 + 一层" | 仅在冻结特征之上训练一个线性分类器；衡量特征质量 |
| NaFlex | "原生灵活分辨率" | SigLIP 2 的能力：在不改变尺寸的情况下输入任意宽高比和分辨率的图像 |
| 温度缩放 | "log 参数化 tau" | CLIP 参数化 `log(1/tau)` 使梯度行为良好；裁剪以防止坍缩到接近零的 tau |

## 拓展阅读

- [Radford et al. — Learning Transferable Visual Models From Natural Language Supervision (arXiv:2103.00020)](https://arxiv.org/abs/2103.00020) —— CLIP 论文。
- [Zhai et al. — Sigmoid Loss for Language Image Pre-Training (arXiv:2303.15343)](https://arxiv.org/abs/2303.15343) —— SigLIP。
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) —— 多语言 + NaFlex。
- [Jia et al. — ALIGN (arXiv:2102.05918)](https://arxiv.org/abs/2102.05918) —— 使用嘈杂网络数据扩展。
- [Cherti et al. — Reproducible scaling laws for contrastive language-image learning (arXiv:2212.07143)](https://arxiv.org/abs/2212.07143) —— OpenCLIP 规模定律。
