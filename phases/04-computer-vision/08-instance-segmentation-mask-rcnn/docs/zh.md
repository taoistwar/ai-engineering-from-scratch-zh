# 实例分割——Mask R-CNN

> 在 Faster R-CNN 检测器上添加一个微小的掩码分支，你就有了实例分割。困难的部分是 RoIAlign，这比看起来更难。

**类型：** 构建 + 学习
**语言：** Python
**前置条件：** 第四阶段第 06 课（YOLO），第四阶段第 07 课（U-Net）
**时间：** 约 75 分钟

## 学习目标

- 端到端追溯 Mask R-CNN 架构：骨干、FPN、RPN、RoIAlign、框头、掩码头
- 从零实现 RoIAlign 并解释为什么不再使用 RoIPool
- 使用 torchvision `maskrcnn_resnet50_fpn_v2` 预训练模型进行生产质量的实例掩码并正确读取其输出格式
- 通过替换框和掩码头并保持骨干冻结，在小型自定义数据集上微调 Mask R-CNN

## 问题

语义分割为每个类别提供一个掩码。实例分割为每个物体提供一个掩码，即使两个物体共享同一类别也是如此。计数个体、跨帧追踪和测量事物（墙中每块砖的包围框、显微镜图像中的每个细胞）都要求实例分割。

Mask R-CNN (He et al., 2017) 通过在 Faster R-CNN 之上添加一个并行的小型 FCN 掩码预测分支扩展了 Faster R-CNN。关键创新是 RoIAlign——RoIPool 的浮点精度替换——它消除了由粗糙量化引起的像素级错位，并将掩码 AP 提升了 10-50%。

## 概念

### RoIAlign、FPN 和 Mask R-CNN 架构

（完整的概念部分涵盖 Mask R-CNN 的端到端架构——骨干、特征金字塔网络（FPN）、区域提议网络（RPN）、RoIAlign（双线性插值替代粗糙量化）以及并行的框回归和掩码预测头——详见 `docs/en.md` 概念部分）

（完整的构建部分包含自定义 RoIAlign 实现、torchvision 预训练模型的使用以及自定义数据集上的微调——详见 `code/main.py`）

## 使用它

torchvision 直接内置了生产就绪的 Mask R-CNN：

```python
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2

model = maskrcnn_resnet50_fpn_v2(weights="DEFAULT")
model.eval()
```

预测输出包含 `boxes`、`labels`、`scores` 和 `masks`——每个检测到的实例的逐像素掩码。

## 交付物

本课产出：
- 实例分割管道配置和微调提示词

## 练习

1. 实现 RoIAlign 并与 `torchvision.ops.roi_align` 比较。验证浮点精度差异。
2. 在 COCO 的 5 类子集上微调预训练 Mask R-CNN。比较冻结骨干 vs 解冻最后 2 个阶段的性能。
3. 添加关键点预测分支（扩展 Mask R-CNN 到 Keypoint R-CNN）。在人体姿态数据集上测试。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|----------|
| 实例分割 | "每个物体的掩码" | 分离同一类的不同实例，每个都有自己的二进制掩码 |
| RoIAlign | "精确 RoI 池化" | 使用双线性插值从特征图中提取固定大小特征，消除粗糙量化 |
| RoIPool | "粗糙 RoI 池化" | 将浮点坐标量化为整数，导致错位；已被 RoIAlign 取代 |
| FPN | "特征金字塔" | 从多尺度骨干特征构建特征金字塔以检测不同大小的物体 |
| RPN | "区域提议网络" | 预测物体性分数和初始框提议的轻量级网络 |

## 延伸阅读

- [Mask R-CNN (He et al., 2017)](https://arxiv.org/abs/1703.06870) — 原始论文
- [Feature Pyramid Networks (Lin et al., 2017)](https://arxiv.org/abs/1612.03144) — FPN 论文
- torchvision detection docs — 官方 PyTorch 检测/分割模型参考
