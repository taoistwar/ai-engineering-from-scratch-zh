# StyleGAN

> 大多数生成器同时将 `z` 搅拌到每一层。StyleGAN 将其分开：首先将 `z` 映射到中间 `w`，然后通过 AdaIN 在每个分辨率级别*注入* `w`。这个单一的改变解缠了潜在空间，并使照片写实人脸成为连续七年已解决的问题。

**类型：** 构建
**语言：** Python
**前置条件：** 第八阶段 · 03（GAN），第四阶段 · 08（归一化），第三阶段 · 07（CNN）
**时间：** 约 45 分钟

## 问题

DCGAN 通过转置卷积堆叠将 `z` 映射到图像。问题是：`z` 控制一切——姿态、光照、身份、背景——纠缠在一起。沿 `z` 的一个轴移动，四个都变。你不能要求模型"同一个人，不同姿态"，因为表示不按那种方式因式分解。

Karras 等人（2019，NVIDIA）提出：停止将 `z` 直接输入卷积层。输入一个常数 `4×4×512` 张量作为网络输入。学习一个 8 层 MLP，将 `z ∈ Z → w ∈ W`。通过*自适应实例归一化*（AdaIN）在每个分辨率注入 `w`：归一化每个 conv 特征图，然后通过 `w` 的仿射投影进行缩放和移位。添加每层噪声以获取随机细节（皮肤毛孔、发丝）。

结果：`W` 具有大致正交的轴，分别对应"高层风格"（姿态、身份）vs"细粒度风格"（光照、颜色）。你可以通过使用图像 A 的 `w` 用于低分辨率级别和图像 B 的 `w` 用于高分辨率级别来交换两个图像之间的风格。这解锁了编辑、跨域风格化和整个"StyleGAN 反转"研究线。

## 概念

![StyleGAN：映射网络 + AdaIN + 每层噪声](../assets/stylegan.svg)

**映射网络。** `f: Z → W`，8 层 MLP。`Z = N(0, I)^512`。`W` 不强制为高斯——它学习数据适应的形状。

**合成网络。** 从学习的常数 `4×4×512` 开始。每个分辨率块：`upsample → conv → AdaIN(w_i) → noise → conv → AdaIN(w_i) → noise`。分辨率翻倍：4、8、16、32、64、128、256、512、1024。

**AdaIN。**

```
AdaIN(x, y) = y_scale · (x - mean(x)) / std(x) + y_bias
```

其中 `y_scale` 和 `y_bias` 来自 `w` 的仿射投影。按特征图归一化，然后重新塑造风格。这里的"风格"是特征图的一阶和二阶统计量。

**每层噪声。** 添加到每个特征图的单通道高斯噪声，由学习的每通道因子缩放。控制随机细节而不影响全局结构。

**截断技巧。** 在推理时，采样 `z`，计算 `w = mapping(z)`，然后 `w' = ŵ + ψ·(w - ŵ)` 其中 `ŵ` 是许多样本上的均值 `w`。`ψ < 1` 用多样性换取质量。几乎每个 StyleGAN 演示都使用 `ψ ≈ 0.7`。

## StyleGAN 1 → 2 → 3

| 版本 | 年份 | 创新 |
|---------|------|------------|
| StyleGAN | 2019 | 映射网络 + AdaIN + 噪声 + 逐级增长。 |
| StyleGAN2 | 2020 | 权重解调替换 AdaIN（修复水滴伪影）；跳跃/残差架构；路径长度正则化。 |
| StyleGAN3 | 2021 | 无混叠卷积 + 等变核；消除纹理粘附到像素网格。 |
| StyleGAN-XL | 2022 | 类别条件，1024²，ImageNet。 |
| R3GAN | 2024 | 以更强正则化重新品牌化；以 20 倍更少参数在 FFHQ-1024 上接近扩散。 |

在 2026 年，StyleGAN3 仍然是 (a) 高 FPS 的狭窄领域照片写实，(b) few-shot 域适应（用 100 张图像在新数据集上训练，冻结映射），(c) 基于反转的编辑（找到重建真实照片的 `w`，然后编辑该 `w`）的默认方案。对于开放领域文本到图像，它不是工具——扩散才是。

## 动手构建

`code/main.py` 在 1-D 中实现了一个玩具"style-GAN lite"：一个映射 MLP，一个合成函数，接收学习的常量向量并用 `w` 衍生的 scale/bias 对其进行调制，以及每层噪声。它表明通过仿射调制注入 `w` 匹敌或优于将 `z` 拼接到生成器的输入中。

### 步骤 1：映射网络

```python
def mapping(z, M):
    h = z
    for i in range(num_layers):
        h = leaky_relu(add(matmul(M[f"W{i}"], h), M[f"b{i}"]))
    return h
```

### 步骤 2：自适应实例归一化

```python
def adain(x, w_scale, w_bias):
    mu = mean(x)
    sd = std(x)
    x_norm = [(xi - mu) / (sd + 1e-8) for xi in x]
    return [w_scale * xi + w_bias for xi in x_norm]
```

每特征图的 scale 和 bias 通过线性投影来自 `w`。

### 步骤 3：每层噪声

```python
def add_noise(x, sigma, rng):
    return [xi + sigma * rng.gauss(0, 1) for xi in x]
```

每通道的 sigma 是可学习的。

## 缺陷陷阱

- **水滴伪影。** StyleGAN 1 在特征图中产生了斑点水滴，因为 AdaIN 将均值置零。StyleGAN 2 的权重解调通过缩放卷积权重来修复。
- **纹理粘附。** StyleGAN 1 和 2 的纹理跟随像素坐标，而不是物体坐标（插值时可见）。StyleGAN 3 的无混叠卷积用加窗 sinc 滤波器修复。
- **模式覆盖。** 截断 `ψ < 0.7` 看起来干净但从狭窄锥中采样；如果你需要多样性使用 `ψ = 1.0`。
- **反转是有损的。** 将真实照片反转到 `W` 通常通过优化或编码器（e4e、ReStyle、HyperStyle）完成。结果在许多迭代中漂移。

## 使用它

| 用例 | 方法 |
|----------|----------|
| 照片写实人脸（动漫、产品、窄领域） | StyleGAN3 FFHQ / 自定义微调 |
| 从照片进行人脸编辑 | e4e 反转 + StyleSpace / InterFaceGAN 方向 |
| 换脸 / 重现 | StyleGAN + 编码器 + 混合 |
| 头像流水线 | StyleGAN3 w/ ADA 用于低数据微调 |
| 从少量图像进行域适应 | 冻结映射网络，微调合成 |
| 多模态或文本条件生成 | 不要——使用扩散 |

对于答案就是"一张人脸照片"的产品级演示，StyleGAN 在推理成本（单次前向传播，4090 上 <10ms）和相同质量栏下的锐度上击败扩散。

## 交付成果

保存 `outputs/skill-stylegan-inversion.md`。技能接收真实照片并输出：反转方法（e4e / ReStyle / HyperStyle）、预期潜在损失、编辑预算（在出现伪影之前你可以在 `W` 中走多远）、以及已知良好编辑方向列表（年龄、表情、姿态）。

## 练习

1. **简单。** 运行 `code/main.py`，设置 `adain_on=True` 和 `adain_on=False`。比较固定潜在向量 vs 扰动潜在向量的输出分布。
2. **中等。** 实现混合正则化：对于训练批次，计算 `w_a`、`w_b`，对合成的前半部分应用 `w_a`，后半部分应用 `w_b`。解码器是否学到了解缠的风格？
3. **困难。** 取一个预训练的 StyleGAN3 FFHQ 模型（ffhq-1024.pkl）。通过在标注样本上训练 SVM 找到控制"微笑"的 `w` 方向；报告在身份漂移之前你可以推多远。

## 关键术语

| 术语 | 人们怎么说 | 它实际上的含义 |
|------|-----------------|-----------------------|
| 映射网络 | "MLP" | `f: Z → W`，8 层，将潜在几何与数据统计解耦。 |
| W 空间 | "风格空间" | 映射网络的输出；大致解缠。 |
| AdaIN | "自适应实例归一化" | 归一化特征图，然后通过 `w` 投影进行缩放 + 移位。 |
| 截断技巧 | "Psi" | `w = mean + ψ·(w - mean)`，ψ<1 用多样性换取质量。 |
| 路径长度正则化 | "PL reg" | 惩罚每单位 `w` 变化的大图像变化；使 `W` 更平滑。 |
| 权重解调 | "StyleGAN2 修复" | 归一化 conv 权重而不是激活；消除水滴伪影。 |
| 无混叠 | "StyleGAN3 的技巧" | 加窗 sinc 滤波器；消除纹理粘附到像素网格。 |
| 反转 | "为真实图像找到 w" | 优化或编码 `x → w` 使得 `G(w) ≈ x`。 |

## 生产笔记：为什么 StyleGAN 在 2026 年仍然发布

StyleGAN3 在 4090 上生成 1024² FFHQ 人脸只需不到 10 ms——`num_steps = 1`，无 VAE 解码，无交叉注意力传递。在生产术语中，这是任何图像生成器的延迟下限。相同分辨率下的 50 步 SDXL + VAE 解码流水线约 3 秒。那是 **300 倍的差距**，对于狭窄领域产品（头像服务、身份证件流水线、库存人脸生成）它在 TCO 上获胜。

两个操作后果：

- **无调度器，无批处理器。** 目标占用的静态批次是最优的。连续批处理（对 LLM 和扩散至关重要）提供零收益，因为每个请求花费相同的 FLOPs。
- **截断 `ψ` 是安全旋钮。** `ψ < 0.7` 从映射网络范围的狭窄锥中采样。这是服务层对样本方差的唯一杠杆。在峰值负载时降低 `ψ`，为高级用户提高它。

## 延伸阅读

- [Karras 等人 (2019). A Style-Based Generator Architecture for GANs](https://arxiv.org/abs/1812.04948) — StyleGAN。
- [Karras 等人 (2020). Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) — StyleGAN2。
- [Karras 等人 (2021). Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423) — StyleGAN3。
- [Tov 等人 (2021). Designing an Encoder for StyleGAN Image Manipulation](https://arxiv.org/abs/2102.02766) — e4e 反转。
- [Sauer 等人 (2022). StyleGAN-XL: Scaling StyleGAN to Large Diverse Datasets](https://arxiv.org/abs/2202.00273) — StyleGAN-XL。
- [Huang 等人 (2024). R3GAN: The GAN is dead; long live the GAN!](https://arxiv.org/abs/2501.05441) — 现代极简 GAN 配方。
