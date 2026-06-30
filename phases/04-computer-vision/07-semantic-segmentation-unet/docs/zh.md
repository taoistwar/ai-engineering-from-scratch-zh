# 语义分割——U-Net

> 分割是在每个像素上的分类。U-Net 通过将下采样编码器与上采样解码器配对并在它们之间连接跳跃连接来使其工作。

**类型：** 构建
**语言：** Python
**前置条件：** 第四阶段第 03 课（CNN），第四阶段第 04 课（图像分类）
**时间：** 约 75 分钟

## 学习目标

- 区分语义分割、实例分割和全景分割，并为给定问题选择正确的任务
- 在 PyTorch 中从零构建 U-Net，包括编码器块、瓶颈、带转置卷积的解码器和跳跃连接
- 实现逐像素交叉熵、Dice 损失和组合损失（当前医学和工业分割的默认值）
- 阅读每类 IoU 和 Dice 指标，诊断低分数来自小物体召回率、边界准确率还是类别不平衡

## 问题

分类为每张图像输出一个标签。检测为每张图像输出少量框。分割为每个像素输出一个标签。对于大小为 `H x W` 的输入，输出是形状为 `H x W`（语义）或 `H x W x N_instances`（实例）的张量。那是每张图像数百万个预测，而不是一个。

分割是医学影像（肿瘤边界）、自动驾驶（可行驶区域）和卫星分析（土地利用）等精确边界要求严格的任务的基本操作。U-Net 架构（Ronneberger 等人 2015）以其对称的编码器-解码器设计和跳跃连接成为了标准——不仅用于生物医学，而是用于任何需要精确保留边界的密集预测任务。

## 概念

### 编码器-解码器设计

U-Net 有两个对称的一半。编码器通过重复的 conv-bn-relu-pool 块逐步降低空间分辨率，同时增加特征通道。解码器通过转置卷积或最近邻上采样后跟卷积来恢复分辨率。编码器和解码器匹配层之间的跳跃连接将低级空间细节直接连接到上采样路径。

### Dice 损失、逐像素交叉熵和 IoU 指标

（完整的概念部分涵盖分割任务类型、U-Net 架构设计、损失函数和评估指标——详见 `docs/en.md` 概念部分）

（完整的构建部分包含合成圆/方数据集、U-Net 编码器/解码器实现、组合损失的训练循环——详见 `code/main.py`）

## 使用它

对于生产，`segmentation_models_pytorch` ("smp") 用任何 torchvision 或 timm 骨干包装每个标准分割架构。三行代码：

```python
import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=3,
)
```

同样值得了解用于实际工作的：
- **DeepLabV3+** 用扩张卷积替换最大池化下采样，使瓶颈保持分辨率
- **SegFormer** 将卷积编码器替换为分层 transformer；当前多个基准上的 SOTA
- **Mask2Former** / **OneFormer** 在单一架构中统一语义、实例和全景分割

## 交付物

本课产出：
- `outputs/prompt-segmentation-task-picker.md`——分割任务选择的提示词
- `outputs/skill-segmentation-mask-inspector.md`——分割掩码检查的技能

## 练习

1. **（简单）** 为二分类分割任务（前景 vs 背景）实现 `bce_dice_loss`。验证组合损失在前景仅占 5% 像素时比单独 BCE 收敛更快。
2. **（中等）** 用 `nn.ConvTranspose2d` 上采样块替换 `nn.Upsample + conv`。在合成数据集上训练两者并比较 mIoU。观察转置卷积版本中出现棋盘伪影的位置。
3. **（困难）** 取一个真实分割数据集（Oxford-IIIT Pets、Cityscapes mini split 或医学子集）训练 U-Net 达到 `smp.Unet` 参考的 2 IoU 点以内。报告每类 IoU 并确定哪些类别从添加 Dice 到损失中获益最多。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|----------|
| 语义分割 | "标记每个像素" | 在每个像素上进行 C 个类别分类；同一类的实例会合并 |
| 实例分割 | "标记每个物体" | 分离同一类的不同实例；仅前景 |
| 全景分割 | "语义 + 实例" | 每个像素有一个类别；每个事物实例也有唯一 ID |
| 跳跃连接 | "U-Net 桥" | 将编码器特征连接到匹配分辨率解码器特征；保留高频细节 |
| 转置卷积 | "反卷积" | 可学习的上采样；可能产生棋盘伪影 |
| Dice 损失 | "重叠损失" | 1 - 2|A ∩ B| / (|A| + |B|)；直接优化掩码重叠，对类别不平衡鲁棒 |
| mIoU | "平均交并比" | 按类别平均的 IoU；社区标准的分割指标 |
| Boundary F1 | "边界准确率" | 仅在边界像素上计算的 F1 分数；对精度关键任务很重要 |

## 延伸阅读

- [U-Net: Convolutional Networks for Biomedical Image Segmentation (Ronneberger et al., 2015)](https://arxiv.org/abs/1505.04597) — 原始论文
- [Fully Convolutional Networks (Long et al., 2015)](https://arxiv.org/abs/1411.4038) — 首先将分割变成端到端卷积问题的论文
- [segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch) — 生产分割的参考
- [Lessons learned from training SOTA segmentation](https://www.kaggle.com/code/iafoss/carvana-unet-pytorch) — 关于 TTA、伪标签和类别权重在真实数据上重要性的实战攻略
