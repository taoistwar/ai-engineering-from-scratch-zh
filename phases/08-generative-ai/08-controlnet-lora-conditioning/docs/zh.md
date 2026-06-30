# ControlNet、LoRA 与条件化

> 仅有文本是一个笨拙的控制信号。ControlNet 让你克隆预训练的扩散模型并用深度图、姿态骨架、涂鸦或边缘图像来操控它。LoRA 让你通过训练一千万参数来微调一个 2B 参数的模型。它们一起将 Stable Diffusion 从玩具变成了每个机构都在发布的 2026 年图像流水线。

**类型：** 构建
**语言：** Python
**前置条件：** 第八阶段 · 07（潜在扩散），第十阶段（从零开始的 LLM——为 LoRA 基础）
**时间：** 约 75 分钟

## 问题

像"一个女人穿着红裙子在繁忙的街道上遛狗"这样的提示没有给模型关于狗*在哪里*、女人处于*什么姿态*或街道的*视角*的信息。文本固定了你需要指定一幅图像所需内容的约 10%。其余的是视觉性的，无法高效地用文字描述。

为每个信号（姿态、深度、canny、分割）从头训练新的条件模型是禁止性的。你想要保持 2.6B 参数的 SDXL 骨干冻结，附加一个读取条件的小型侧网络，并让它推动骨干的中间特征。这就是 ControlNet。

你还想教模型新概念（你的脸、你的产品、你的风格）而不重新训练完整模型。你想要一个 100 倍更小的增量。这就是 LoRA——插入现有注意力权重的低秩适配器。

ControlNet + LoRA + 文本 = 2026 年实践者的工具包。大多数生产图像流水线在 SDXL / SD3 / Flux 基础上叠加 2-5 个 LoRA、1-3 个 ControlNet 和一个 IP-Adapter。

## 概念

![ControlNet 克隆编码器；LoRA 添加低秩增量](../assets/controlnet-lora.svg)

### ControlNet（Zhang 等人，2023）

取一个预训练的 SD。*克隆* U-Net 的编码器部分。冻结原始部分。训练克隆以接受额外的条件输入（边缘、深度、姿态）。用*零卷积*跳跃连接（初始化为零的 1×1 卷积——作为无操作开始，学习增量）将克隆连接回原始的编码器部分。

```
SD U-Net 解码器： ... ← orig_enc_features + zero_conv(controlnet_enc(condition))
```

零卷积初始化意味着 ControlNet 作为恒等映射开始——即使训练前也无害。在 100 万对（提示、条件、图像）三元组上使用标准扩散损失进行训练。

每种模态的 ControlNet 作为小型侧模型发布（SDXL 约 360M，SD 1.5 约 70M）。你可以在推理时组合它们：

```
features += weight_a * control_a(depth) + weight_b * control_b(pose)
```

### LoRA（Hu 等人，2021）

对于模型中任何线性层 `W ∈ R^{d×d}`，冻结 `W` 并添加低秩增量：

```
W' = W + ΔW,  ΔW = B @ A,  A ∈ R^{r×d},  B ∈ R^{d×r}
```

其中 `r << d`。注意力标准是秩 4-16，重度微调是秩 64-128。新参数数量：`2 · d · r` 而不是 `d²`。对于 `d=640`、`r=16` 的 SDXL 注意力：每个适配器 20k 参数而不是 410k——20 倍减少。在整个模型上：一个 LoRA 通常是 20-200MB 对比基础模型的 5GB。

在推理时你可以缩放 LoRA：`W' = W + α · B @ A`。`α = 0.5-1.5` 是正常的。多个 LoRA 可以累加堆叠（带有它们以非线性方式相互作用的常见注意事项）。

### IP-Adapter（Ye 等人，2023）

一个接受*图像*作为条件（与文本并列）的微型适配器。使用 CLIP 图像编码器产生图像 token，与文本 token 一起注入到交叉注意力中。每个基础模型约 20MB。让你能够做"生成一张这个参考风格的图像"而无需 LoRA。

## 可组合性矩阵

| 工具 | 它控制什么 | 大小 | 何时使用 |
|------|------------------|------|-------------|
| ControlNet | 空间结构（姿态、深度、边缘） | 70-360MB | 精确布局、构图 |
| LoRA | 风格、主题、概念 | 20-200MB | 个性化、风格 |
| IP-Adapter | 来自参考图像的风格或主题 | 20MB | 无文本能描述外观 |
| Textual Inversion | 作为新 token 的单个概念 | 10KB | 遗留，大多被 LoRA 取代 |
| DreamBooth | 主题上的完整微调 | 2-5GB | 强身份，高计算量 |
| T2I-Adapter | 更轻量级的 ControlNet 替代 | 70MB | 边缘设备，推理预算 |

ControlNet ≈ 空间。LoRA ≈ 语义。两者都用。

## 动手构建

`code/main.py` 在 1-D 上模拟两种机制：

1. **LoRA。** 一个预训练的线性层 `W`。冻结它。训练低秩 `B @ A` 使得 `W + BA` 匹配目标线性层。展示 `r = 1` 足以完美学习秩为 1 的修正。

2. **ControlNet-lite。** 一个"冻结基础"预测器和一个读取额外信号的"侧网络"。侧网络的输出由初始化为零的可学习标量门控（我们的零卷积版本）。训练并观察门的上升。

### 步骤 1：LoRA 数学

```python
def lora(W, A, B, x, alpha=1.0):
    # W 被冻结；A、B 是可训练的低秩因子。
    return [W[i][j] * x[j] for i, j in ...] + alpha * (B @ (A @ x))
```

### 步骤 2：零初始化侧网络

```python
side_out = control_net(x, condition)
gated = gate * side_out  # gate 初始化为 0
h = base(x) + gated
```

在第 0 步，输出与基础相同。早期训练缓慢更新 `gate`——无灾难性漂移。

## 缺陷陷阱

- **LoRA 缩放过度。** `α = 2` 或 `α = 3` 是常见的"让它更强"的黑客手段，产生过度风格化/破坏的输出。保持 `α ≤ 1.5`。
- **ControlNet 权重冲突。** 以权重 1.0 使用姿态 ControlNet 和权重 1.0 使用深度 ControlNet 通常会过冲。权重之和 ≈ 1.0 是安全默认值。
- **LoRA 在错误的基础上。** SDXL LoRA 在 SD 1.5 上静默无操作，因为注意力维度不匹配。diffusers 在 0.30+ 中会警告。
- **Textual Inversion 漂移。** 在一个检查点上训练的 token 在另一个上严重漂移。LoRA 更可移植。
- **LoRA 权重合并和存储。** 你可以将 LoRA 烘焙到基础模型权重中以进行更快的推理（无需运行时加法），但会失去在运行时缩放 `α` 的能力。保留两个版本。

## 使用它

| 目标 | 2026 年流水线 |
|------|---------------|
| 再现品牌的艺术风格 | 在 ~30 张策划图像上以秩 32 训练的 LoRA |
| 将我的脸放入生成的图像中 | DreamBooth 或 LoRA + IP-Adapter-FaceID |
| 特定姿态 + 提示 | ControlNet-Openpose + SDXL + 文本 |
| 深度感知构图 | ControlNet-Depth + SD3 |
| 参考 + 提示 | IP-Adapter + 文本 |
| 精确布局 | ControlNet-Scribble 或 ControlNet-Canny |
| 背景替换 | ControlNet-Seg + Inpainting（第 09 课） |
| 快速 1 步风格 | SDXL-Turbo 上的 LCM-LoRA |

## 交付成果

保存 `outputs/skill-sd-toolkit-composer.md`。技能接收任务（输入资产：提示，可选参考图像，可选姿态，可选深度，可选涂鸦）并输出：工具栈、权重和可重现的种子协议。

## 练习

1. **简单。** 在 `code/main.py` 中，将 LoRA 秩 `r` 从 1 变为 4。在什么秩下 LoRA 精确匹配秩为 2 的目标增量？
2. **中等。** 在两个目标变换上训练两个单独的 LoRA。一起加载它们并展示它们的叠加交互。什么时候交互破坏线性？
3. **困难。** 使用 diffusers 叠加：SDXL-base + Canny-ControlNet（权重 0.8）+ 风格 LoRA（α 0.8）+ IP-Adapter（权重 0.6）。随着叠加权重的变化测量 FID 与提示遵循度的权衡。

## 关键术语

| 术语 | 人们怎么说 | 它实际上的含义 |
|------|-----------------|-----------------------|
| ControlNet | "空间控制" | 克隆的编码器 + 零卷积跳跃连接；读取条件图像。 |
| 零卷积 | "从恒等映射开始" | 初始化为零的 1×1 卷积；ControlNet 作为无操作开始。 |
| LoRA | "低秩适配器" | `W + B @ A`，`r << d`；比完整微调少 100 倍参数。 |
| 秩 r | "旋钮" | LoRA 压缩；通常 4-16，对于重度个性化 64+。 |
| α | "LoRA 强度" | LoRA 增量的运行时缩放。 |
| IP-Adapter | "参考图像" | 通过 CLIP 图像 token 的小型图像条件适配器。 |
| DreamBooth | "完整主题微调" | 在 ~30 张主题图像上训练完整模型。 |
| Textual Inversion | "新 token" | 仅学习新的词嵌入；遗留，大多被取代。 |

## 生产笔记：LoRA 交换、ControlNet 通道、多租户服务

一个真实的文本到图像 SaaS 在相同基础检查点上服务数百个 LoRA 和十几个 ControlNet。服务问题看起来很像 LLM 多租户（生产文献在连续批处理和 LoRAX / S-LoRA 下涵盖 LLM 案例）：

- **热交换 LoRA，不要合并。** 将 `W' = W + α·B·A` 合并到基础模型中可获得约 3-5% 更快的每步推理，但会冻结 `α` 和基础。将 LoRA 作为秩 r 增量热保留在 VRAM 中；diffusers 暴露 `pipe.load_lora_weights()` + `pipe.set_adapters([...], adapter_weights=[...])` 用于按请求激活。交换成本是 `2 · d · r · num_layers` 权重——MB 级，亚秒级。
- **ControlNet 作为第二注意力通道。** 克隆的编码器与基础并行运行。两个 ControlNet 各权重 1.0 = 每步两次额外前向传播，不是一次合并传递。批量大小余量二次方下降。预算每个活跃 ControlNet 约 1.5× 步成本。
- **LoRA 也要量化。** 如果你量化了基础（参见第 07 课，8GB 上的 Flux），LoRA 增量也可干净地量化为 8 位或 4 位。QLoRA 风格加载让你在 4 位 Flux 基础之上叠加 5-10 个 LoRA 而不爆内存。

Flux 特定：Niels 的 Flux-on-8GB notebook 将基础量化为 4 位；在该量化基础之上叠加风格 LoRA（`pipe.load_lora_weights("user/style-lora")`）在 `weight_name="pytorch_lora_weights.safetensors"` 仍然有效。这是 2026 年大多数 SaaS 机构发布的配方。

## 延伸阅读

- [Zhang、Rao、Agrawala (2023). Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) — ControlNet。
- [Hu 等人 (2021). LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — LoRA（最初用于 LLM；移植到扩散）。
- [Ye 等人 (2023). IP-Adapter: Text Compatible Image Prompt Adapter](https://arxiv.org/abs/2308.06721) — IP-Adapter。
- [Mou 等人 (2023). T2I-Adapter: Learning Adapters to Dig Out More Controllable Ability](https://arxiv.org/abs/2302.08453) — ControlNet 的轻量级替代。
- [Ruiz 等人 (2023). DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242) — DreamBooth。
- [HuggingFace Diffusers — ControlNet / LoRA / IP-Adapter 文档](https://huggingface.co/docs/diffusers/training/controlnet) — 参考流水线。
