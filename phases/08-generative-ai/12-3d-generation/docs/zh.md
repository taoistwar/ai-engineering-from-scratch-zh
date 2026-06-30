# 3D 生成

> 3D 是 2D 到 3D 杠杆作用最强的模态。2023 年的突破是 3D 高斯溅射。2024-2026 年的生成推动在其上叠加多视图扩散 + 3D 重建，以从单个提示或照片产生物体和场景。

**类型：** 学习
**语言：** Python
**前置条件：** 第四阶段（视觉），第八阶段 · 07（潜在扩散）
**时间：** 约 45 分钟

## 问题

3D 内容令人痛苦：

- **表示。** 网格、点云、体素网格、有符号距离场（SDF）、神经辐射场（NeRF）、3D 高斯函数。每种都有权衡。
- **数据稀缺。** ImageNet 有 1400 万张图像。最大的干净 3D 数据集（Objaverse-XL，2023）约有 1000 万个物体，大多数质量低下。
- **内存。** 512³ 体素网格是 1.28 亿个体素；一个有用的场景 NeRF 需要 100 万个样本/光线。生成比重建更困难。
- **监督。** 对于 2D 图像你有像素。对于 3D，你通常只有少量 2D 视图，必须提升到 3D。

2026 年技术栈分离了两个问题。首先，用扩散模型生成*2D 多视图图像*。其次，将*3D 表示*（通常是高斯溅射）拟合到这些图像。

## 概念

![3D 生成：多视图扩散 + 3D 重建](../assets/3d-generation.svg)

### 表示：3D 高斯溅射（Kerbl 等人，2023）

将场景表示为约 100 万个 3D 高斯的云。每个有 59 个参数：位置（3）、协方差（6，或四元数 4 + 缩放 3）、不透明度（1）、球谐颜色（3 阶 48，0 阶 3）。

渲染 = 投影 + alpha 合成。快速（4090 上 1080p 约 100 fps）。可微。通过对地面真实照片进行梯度下降拟合。一个场景在消费级 GPU 上 5-30 分钟内拟合。

顶部两项 2023-2024 年创新：
- **生成式高斯溅射。** 像 LGM、LRM、InstantMesh 这样的模型直接从一张或少量图像预测高斯云。
- **4D 高斯溅射。** 带有每帧偏移的高斯用于动态场景。

### 多视图扩散

微调预训练的图像扩散模型以从文本提示或单张图像生成同一物体的多个一致视图。Zero123（Liu 等人，2023）、MVDream（Shi 等人，2023）、SV3D（Stability，2024）、CAT3D（Google，2024）。通常输出围绕物体的 4-16 个视图，通过高斯溅射或 NeRF 提升到 3D。

### 文本到 3D 流水线

| 模型 | 输入 | 输出 | 时间 |
|-------|-------|--------|------|
| DreamFusion (2022) | text | 通过 SDS 的 NeRF | 每个资产约 1 小时 |
| Magic3D | text | mesh + texture | 约 40 分钟 |
| Shap-E (OpenAI, 2023) | text | implicit 3D | 约 1 分钟 |
| SJC / ProlificDreamer | text | NeRF / mesh | 约 30 分钟 |
| LRM (Meta, 2023) | image | triplane | 约 5 秒 |
| InstantMesh (2024) | image | mesh | 约 10 秒 |
| SV3D (Stability, 2024) | image | novel views | 约 2 分钟 |
| CAT3D (Google, 2024) | 1-64 images | 3D NeRF | 约 1 分钟 |
| TripoSR (2024) | image | mesh | 约 1 秒 |
| Meshy 4 (2025) | text + image | PBR mesh | 约 30 秒 |
| Rodin Gen-1.5 (2025) | text + image | PBR mesh | 约 60 秒 |
| Tencent Hunyuan3D 2.0 (2025) | image | mesh | 约 30 秒 |

2025-2026 年方向：具有适合游戏引擎的 PBR 材质的直接文本到网格模型。多视图扩散中间步骤仍然是一般物体的最佳性能配方。

### NeRF（作背景）

神经辐射场（Mildenhall 等人，2020）。微小的 MLP 接受 `(x, y, z, view direction)` 并输出 `(color, density)`。通过沿射线积分渲染。在质量上击败基于网格的新视角合成，但渲染慢 100-1000 倍。大多数实时使用已被高斯溅射取代，但在研究中仍占主导。

## 动手构建

`code/main.py` 实现玩具 2D "高斯溅射"拟合：将合成目标图像（平滑渐变）表示为 2D 高斯溅射的和。通过梯度下降优化位置、颜色和协方差以匹配目标。你看到两个核心操作：前向渲染（溅射 + alpha 合成）和通过梯度下降拟合。

### 步骤 1：2D 高斯溅射

```python
def gaussian_at(x, y, gaussian):
    px, py = gaussian["pos"]
    sigma = gaussian["sigma"]
    d2 = (x - px) ** 2 + (y - py) ** 2
    return math.exp(-d2 / (2 * sigma * sigma))
```

### 步骤 2：通过求和溅射渲染

```python
def render(image_size, gaussians):
    img = [[0.0] * image_size for _ in range(image_size)]
    for g in gaussians:
        for y in range(image_size):
            for x in range(image_size):
                img[y][x] += g["color"] * gaussian_at(x, y, g)
    return img
```

真实 3D 高斯溅射按深度排序高斯并按顺序进行 alpha 合成。我们的 2D 玩具只求和。

### 步骤 3：通过梯度下降拟合

```python
for step in range(steps):
    pred = render(size, gaussians)
    loss = mse(pred, target)
    gradients = compute_grads(pred, target, gaussians)
    update(gaussians, gradients, lr)
```

## 缺陷陷阱

- **视图不一致。** 如果你独立生成 4 个视图并且它们在物体结构上不一致，3D 拟合是模糊的。修复：带有共享注意力的多视图扩散。
- **背面幻觉。** 单张图像 → 3D 必须发明看不见的一面。质量差异巨大。
- **高斯溅射爆炸。** 无约束训练增长到 1000 万个溅射并过拟合。增密 + 修剪启发式（来自 3D-GS 原始论文）至关重要。
- **拓扑问题。** 来自隐式场（SDF）的网格通常有孔或自交。在发布前运行 remesher（例如 blender 的 voxel remesh）。
- **训练数据许可。** Objaverse 有混合许可；商业用途因模型而异。

## 使用它

| 任务 | 2026 年选择 |
|------|-----------|
| 从照片重建场景 | 高斯溅射（3DGS、Gsplat、Scaniverse） |
| 文本到 3D 物体用于游戏 | Meshy 4 或 Rodin Gen-1.5（PBR 输出） |
| 图像到 3D | Hunyuan3D 2.0、TripoSR、InstantMesh |
| 从少量图像进行新视角合成 | CAT3D、SV3D |
| 动态场景重建 | 4D 高斯溅射 |
| 头像 / 穿着衣服的人 | Gaussian Avatar、HUGS |
| 研究 / SOTA | 任何上周刚发布的东西 |

对于在游戏或电商流水线中发布生产 3D：Meshy 4 或 Rodin Gen-1.5 输出 PBR 网格，可直接进入 Unity / Unreal。

## 交付成果

保存 `outputs/skill-3d-pipeline.md`。技能接收 3D 需求（输入：text / 一张图像 / 少量图像；输出：mesh / splat / NeRF；使用：渲染 / 游戏 / VR）并输出：流水线（多视图扩散 + 拟合，或直接网格模型）、基础模型、迭代预算、拓扑后处理、所需材质通道。

## 练习

1. **简单。** 用 4、16、64 个高斯运行 `code/main.py`。报告最终 MSE vs 目标。
2. **中等。** 扩展到颜色高斯（RGB）。确认重建匹配目标颜色模式。
3. **困难。** 使用 gsplat 或 Nerfstudio，从 50 张照片捕捉中重建真实物体。报告拟合时间和保留视图上的最终 SSIM。

## 关键术语

| 术语 | 人们怎么说 | 它实际上的含义 |
|------|-----------------|-----------------------|
| 3D 高斯溅射 | "3DGS" | 场景作为 3D 高斯的云；可微 alpha 合成渲染。 |
| NeRF | "神经辐射场" | 在 3D 点输出颜色 + 密度的 MLP；通过射线积分渲染。 |
| Triplane | "三个 2-D 平面" | 将 3D 因式分解为三个 2-D 轴对齐特征网格；比体积更便宜。 |
| SDS | "分数蒸馏采样" | 通过使用 2D 扩散分数作为伪梯度训练 3D 模型。 |
| 多视图扩散 | "一次许多视图" | 输出一批一致相机视图的扩散模型。 |
| PBR | "基于物理的渲染" | 具有反照率、粗糙度、金属度、法线通道的材质。 |
| 增密 | "增长溅射" | 3DGS 训练启发式：在高梯度区域分割/克隆溅射。 |

## 生产笔记：3D 尚未有共享基质

与图像（潜在扩散 + DiT）和视频（时空 DiT）不同，3D 在 2026 年没有单一主导运行时。生产决策树在表示上分叉：

- **NeRF / triplane。** 推理是每次样本的射线行进 + MLP 前向传播。512² 渲染需要数百万次 MLP 前向传播。积极批处理射线样本；SDPA/xformers 适用。
- **多视图扩散 + LRM 重建。** 两阶段流水线。阶段 1（多视图 DiT）就像一个扩散服务器。阶段 2（LRM transformer）是在视图上的单次前向传播。总体延迟概况是"扩散 + 单次"——相应地选择每个阶段的服务原语。
- **SDS / DreamFusion。** 每个资产的优化，而非推理。构建作业，不是请求处理器。

对于大多数 2026 年产品，正确答案是"按请求运行多视图扩散模型，异步重建为 3DGS，提供 3DGS 用于实时查看"。这干净地将工作负载分解为 GPU 推理服务器（快）和离线优化器（慢）。

## 延伸阅读

- [Mildenhall 等人 (2020). NeRF: Representing Scenes as Neural Radiance Fields](https://arxiv.org/abs/2003.08934) — NeRF。
- [Kerbl 等人 (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079) — 3DGS。
- [Poole 等人 (2022). DreamFusion: Text-to-3D using 2D Diffusion](https://arxiv.org/abs/2209.14988) — SDS。
- [Liu 等人 (2023). Zero-1-to-3: Zero-shot One Image to 3D Object](https://arxiv.org/abs/2303.11328) — Zero123。
- [Shi 等人 (2023). MVDream](https://arxiv.org/abs/2308.16512) — 多视图扩散。
- [Hong 等人 (2023). LRM: Large Reconstruction Model for Single Image to 3D](https://arxiv.org/abs/2311.04400) — LRM。
- [Gao 等人 (2024). CAT3D: Create Anything in 3D with Multi-View Diffusion Models](https://arxiv.org/abs/2405.10314) — CAT3D。
- [Stability AI (2024). Stable Video 3D (SV3D)](https://stability.ai/research/sv3d) — SV3D。
