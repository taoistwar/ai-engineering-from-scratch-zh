# 单目深度与几何估计

> 深度图是单通道图像，其中每个像素是到相机的距离。从单个 RGB 帧预测它，在没有立体视觉或 LiDAR 的情况下曾被认为不可能。到 2026 年，冻结的 ViT 编码器加一个轻量头可以达到真值的几个百分点以内。

**类型：** Build + Use
**语言：** Python
**先修要求：** Phase 4 Lesson 14 (ViT), Phase 4 Lesson 17 (Self-Supervised Vision), Phase 4 Lesson 07 (U-Net)
**时间：** ~60 分钟

## 学习目标

- 区分相对深度和度量深度，并说明每个生产模型（MiDaS、Marigold、Depth Anything V3、ZoeDepth）解决哪一个
- 使用 Depth Anything V3（DINOv2 骨干）预测任意单张图像的深度，无需标定
- 解释为什么单目深度从单张图像就能工作（透视线索、纹理梯度、已学习的先验）以及它不能恢复什么（绝对尺度、被遮挡的几何）
- 使用深度图和针孔相机内参将 2D 检测提升到 3D 点

## 问题

深度是 2D 计算机视觉中缺失的轴。给定 RGB，你知道事物在图像平面中出现的位置；你不知道它们有多远。深度传感器（立体视觉装置、LiDAR、飞行时间）直接解决这个问题，但昂贵、脆弱且范围有限。

单目深度估计——从单个 RGB 帧预测深度——曾经产生模糊、不可靠的输出。到 2026 年，大型预训练编码器改变了这一点：Depth Anything V3 使用冻结的 DINOv2 骨干，产生跨室内、室外、医疗和卫星域泛化的深度图。Marigold 将深度重新表述为条件扩散问题。ZoeDepth 回归真实的度量距离。

深度也是 2D 检测和 3D 理解之间的桥梁：将检测到的框的像素乘以深度，你就将 2D 对象提升到 3D 点云。这是每个 AR 遮挡系统、每个避障 pipeline 和每个"拿起杯子"机器人的核心。

## 概念

### 相对 vs 度量深度

- **相对深度**——有序的 `z` 值，没有真实世界单位。"像素 A 比像素 B 更近，但距离的比率没有用米锁定。"
- **度量深度**——从相机以米为单位的绝对距离。需要模型已学习图像线索与真实距离之间的统计关系。

MiDaS 和 Depth Anything V3 产生相对深度。Marigold 产生相对深度。ZoeDepth、UniDepth 和 Metric3D 产生度量深度。度量模型对相机内参敏感；相对模型则不然。

### 编码器-解码器模式

```mermaid
flowchart LR
    IMG["图像 (H x W x 3)"] --> ENC["冻结 ViT 编码器<br/>(DINOv2 / DINOv3)"]
    ENC --> FEATS["稠密特征<br/>(H/14, W/14, d)"]
    FEATS --> DEC["深度解码器<br/>(conv upsampler,<br/>DPT 风格)"]
    DEC --> DEPTH["深度图<br/>(H, W, 1)"]

    style ENC fill:#dbeafe,stroke:#2563eb
    style DEC fill:#fef3c7,stroke:#d97706
    style DEPTH fill:#dcfce7,stroke:#16a34a
```

Depth Anything V3 冻结编码器，只训练 DPT 风格解码器。编码器提供丰富的特征；解码器将它们插值回图像分辨率并回归深度。

### 为什么单张图像能产生深度

2D 图像包含许多与深度相关的单目线索：

- **透视**——3D 平行线在 2D 中汇聚。
- **纹理梯度**——远处的表面有更小、更密集的纹理。
- **遮挡顺序**——更近的物体遮挡更远的物体。
- **大小恒常性**——已知物体（汽车、人类）给出近似的尺度。
- **大气透视**——室外场景中远距离物体显得更模糊、更蓝。

在数十亿图像上训练的 ViT 内化了这些线索。有足够的数据和强大的骨干，单目深度可以达到合理的准确率，无需任何显式 3D 监督。

### 单目深度不能做到什么

- **绝对度量尺度**，没有内参或场景中的已知物体。网络可以预测"杯子比勺子远两倍"，但不知道杯子是 1 m 还是 10 m 远。
- **被遮挡的几何**——椅子的背面不可见，无法可靠推断。
- **真正无纹理 / 镜面反射 / 反射表面**——镜子、玻璃、均匀墙壁。网络报告看似合理但错误的深度。

### 2026 年的 Depth Anything V3

- Vanilla DINOv2 ViT-L/14 作为编码器（冻结）。
- DPT 解码器。
- 在多样化来源的位姿图像对上训练（仅光度一致性就足够，无需显式深度监督）。
- 从**任意数量的视觉输入**预测空间一致几何，**有或没有已知相机位姿**。
- SOTA 跨单目深度、任意视角几何、视觉渲染、相机位姿估计。

这是 2026 年当你需要深度时调用的即插即用模型。

### Marigold — 用于深度的扩散

Marigold（Ke 等，CVPR 2024）将深度估计重新表述为条件图像到图像扩散。条件：RGB。目标：深度图。使用预训练 Stable Diffusion 2 U-Net 作为骨干。输出深度图在物体边界异常锐利。权衡：推理比前馈模型慢（10-50 去噪步）。

### 内参与针孔相机

要将像素 `(u, v)` 与深度 `d` 提升到相机坐标系中的 3D 点 `(X, Y, Z)`：

```
fx, fy, cx, cy = 相机内参
X = (u - cx) * d / fx
Y = (v - cy) * d / fy
Z = d
```

内参来自 EXIF 元数据、标定模式或单目内参估计器（Perspective Fields、UniDepth）。没有内参，你仍然可以通过假设 60-70° FOV 和中等分辨率的主点渲染点云——对可视化可用，不用于测量。

### 评估

两个标准指标：

- **AbsRel**（绝对相对误差）：`mean(|d_pred - d_gt| / d_gt)`。越低越好。生产模型 0.05-0.1。
- **delta < 1.25**（阈值准确率）：`max(d_pred/d_gt, d_gt/d_pred) < 1.25` 的像素比例。越高越好。SOTA 为 0.9+。

对于相对深度（Depth Anything V3、MiDaS），评估使用两个指标的尺度和偏移不变版本。

## Build It

### 步骤 1：深度指标

```python
import torch

def abs_rel_error(pred, target, mask=None):
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    return (torch.abs(pred - target) / target.clamp(min=1e-6)).mean().item()


def delta_accuracy(pred, target, threshold=1.25, mask=None):
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    ratio = torch.maximum(pred / target.clamp(min=1e-6), target / pred.clamp(min=1e-6))
    return (ratio < threshold).float().mean().item()
```

评估前始终遮罩无效深度像素（零、NaN、饱和）。

### 步骤 2：尺度和偏移对齐

对于相对深度模型，在计算指标前将预测对齐到真值。`a * pred + b = target` 的最小二乘拟合：

```python
def align_scale_shift(pred, target, mask=None):
    if mask is not None:
        p = pred[mask]
        t = target[mask]
    else:
        p = pred.flatten()
        t = target.flatten()
    A = torch.stack([p, torch.ones_like(p)], dim=1)
    coeffs, *_ = torch.linalg.lstsq(A, t.unsqueeze(-1))
    a, b = coeffs[:2, 0]
    return a * pred + b
```

评估 MiDaS / Depth Anything 时在 `abs_rel_error` 之前运行 `align_scale_shift`。

### 步骤 3：将深度提升为点云

```python
import numpy as np

def depth_to_point_cloud(depth, intrinsics):
    H, W = depth.shape
    fx, fy, cx, cy = intrinsics
    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack([x, y, z], axis=-1)


depth = np.random.uniform(0.5, 4.0, (240, 320))
intr = (320.0, 320.0, 160.0, 120.0)
pc = depth_to_point_cloud(depth, intr)
print(f"点云形状: {pc.shape}  (H, W, 3)")
```

一个函数，每个将 3D 提升的应用。将点云导出为 `.ply` 并在 MeshLab 或 CloudCompare 中打开。

### 步骤 4：合成深度场景的冒烟测试

```python
def synthetic_depth(size=96):
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    # 地面：从近（顶部）到远（底部）的线性梯度
    depth = 1.0 + (yy / size) * 4.0
    # 中间的盒子：更近
    mask = (np.abs(xx - size / 2) < size / 6) & (np.abs(yy - size * 0.6) < size / 6)
    depth[mask] = 2.0
    return depth.astype(np.float32)


gt = torch.from_numpy(synthetic_depth(96))
pred = gt + 0.3 * torch.randn_like(gt)  # 模拟预测
aligned = align_scale_shift(pred, gt)
print(f"对齐前  absRel = {abs_rel_error(pred, gt):.3f}")
print(f"对齐后   absRel = {abs_rel_error(aligned, gt):.3f}")
```

### 步骤 5：Depth Anything V3 用法（参考）

```python
import torch
from transformers import pipeline
from PIL import Image

pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-v2-large")

image = Image.open("street.jpg").convert("RGB")
out = pipe(image)
depth_np = np.array(out["depth"])
```

三行。`out["depth"]` 是 PIL 灰度图；转换为 numpy 用于数学计算。对于 Depth Anything V3 具体来说，一旦发布就交换模型 id；API 不变。

## Use It

- **Depth Anything V3**（Meta AI / ByteDance, 2024-2026）——相对深度的默认选择。生产中最快的 ViT-large 骨干模型。
- **Marigold**（ETH, 2024）——最高视觉质量，慢推理。
- **UniDepth**（ETH, 2024）——带相机内参估计的度量深度。
- **ZoeDepth**（Intel, 2023）——度量深度；较老，仍然可靠。
- **MiDaS v3.1**——旧版但稳定；好的比较基线。

典型集成模式：

1. RGB 帧到达。
2. 深度模型产生深度图。
3. 检测器产生框。
4. 通过深度将框中心提升为 3D；如果可用，与点云合并。
5. 下游：AR 遮挡、路径规划、物体大小估计、立体视觉替代。

对于实时使用，Depth Anything V2 Small（INT8 量化）在消费级 GPU 上以 518x518 达到约 30 fps。

## Ship It

本课产出：

- `outputs/prompt-depth-model-picker.md`——根据延迟、度量 vs 相对需求和场景类型在 Depth Anything V3、Marigold、UniDepth、MiDaS 之间选择。
- `outputs/skill-depth-to-pointcloud.md`——一个构建具有正确处理内参和导出到 `.ply` 的从深度图到点云的 skill。

## 练习

1. **（简单）** 在任何 10 张你的桌子的图像上运行 Depth Anything V2。将深度保存为灰度 PNG 并检查。找出一个预测深度看起来错误的物体并解释为什么单目线索失败了。
2. **（中等）** 给定来自 Depth Anything V2 的 RGB + 深度，提升为点云并用 `open3d` 渲染。比较两个场景（室内 / 室外）并注意哪个看起来更可信。
3. **（困难）** 取五对仅在已知物体位置上有差异的图像（例如瓶子移近 30 厘米）。使用 UniDepth 预测两张的度量深度。报告预测的距离差 vs 真实 30 cm。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|----------------------|
| 单目深度 | "单图像深度" | 从一个 RGB 帧估计深度，无立体视觉或 LiDAR |
| 相对深度 | "有序深度" | 有序 z 值，无真实世界单位 |
| 度量深度 | "绝对距离" | 以米为单位的深度；需要标定或具有度量监督训练的模型 |
| AbsRel | "绝对相对误差" | |d_pred - d_gt| / d_gt 的均值；标准深度指标 |
| Delta 准确率 | "delta < 1.25" | 预测在真值 25% 以内的像素比例 |
| 针孔相机 | "fx, fy, cx, cy" | 用于将 (u, v, d) 提升到 (X, Y, Z) 的相机模型 |
| DPT | "Dense Prediction Transformer" | 在冻结 ViT 编码器之上用于深度的基于卷积的解码器 |
| DINOv2 骨干 | "它能工作的原因" | 自监督特征，跨域泛化，无需深度标签 |

## 延伸阅读

- [Depth Anything V3 论文页面](https://depth-anything.github.io/)——带 DINOv2 编码器的 SOTA 单目深度
- [Marigold (Ke et al., CVPR 2024)](https://marigoldmonodepth.github.io/)——基于扩散的深度估计
- [UniDepth (Piccinelli et al., 2024)](https://arxiv.org/abs/2403.18913)——带内参的度量深度
- [MiDaS v3.1 (Intel ISL)](https://github.com/isl-org/MiDaS)——经典的相对深度基线
- [DINOv3 博客文章 (Meta)](https://ai.meta.com/blog/dinov3-self-supervised-vision-model/)——提升深度准确率的编码器系列
