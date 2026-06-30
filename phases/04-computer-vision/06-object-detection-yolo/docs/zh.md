# 目标检测——从零实现 YOLO

> 检测是分类加回归，在特征图的每个位置运行，然后用非极大值抑制清理。

**类型：** 构建
**语言：** Python
**前置条件：** 第四阶段第 03 课（CNN），第四阶段第 04 课（图像分类），第四阶段第 05 课（迁移学习）
**时间：** 约 75 分钟

## 学习目标

- 解释将检测转变为密集预测问题的网格和锚框设计，并说明输出张量中每个数字的含义
- 计算框之间的交并比，并从零开始实现非极大值抑制
- 在预训练骨干上构建最小的 YOLO 风格检测头，包括分类、物体性和框回归损失
- 阅读检测指标行（precision@0.5、recall、mAP@0.5、mAP@0.5:0.95）并选择下一个要调整的方向

## 问题

分类说"这张图像是一只狗"。检测说"在像素 (112, 40, 280, 210) 处有一只狗，在 (400, 180, 560, 310) 处有一只猫，帧中没有其他东西"。这一结构性变化——预测可变数量的标记框而不是每张图像一个标签——是每个自主系统、每个监控产品、每个文档布局解析器和每条工厂视觉线所依赖的。

检测也是视觉中所有工程权衡同时出现的地方。你想要准确的框（回归头），你想要每个框的正确类别（分类头），你想要模型知道什么时候没有东西可检测（物体性分数），你想要每个真实物体恰好一个预测（非极大值抑制）。错过其中任何一个，管道要么漏检物体，要么报告幻觉框，要么在稍微不同的位置预测同一物体十五次。

YOLO（You Only Look Once，Redmon 等人 2016）是通过卷积网络的单次前向传播使所有这些实时运行的设计，相同的结构决定仍然是现代检测器（YOLOv8、YOLOv9、YOLO-NAS、RT-DETR）的骨干。学习核心，每个变体都成为相同部件的重新排列。

## 概念

### 检测作为密集预测

YOLO 风格检测器为每张图像输出 `(S x S x (5 + C))` 个数字，其中 S 是空间网格大小。每个 S*S 网格单元预测 B 个框。对于每个框：4 个数字描述几何形状（tx, ty, tw, th），1 个数字是物体性分数，C 个数字是类别概率。每单元总计：B * (5 + C)。

### 为什么用网格和锚框

网格通过将每个真实框分配给其中心所在的网格单元来解决空间锚定问题。锚框解决了第二个问题：预定义 B 个先验框形状，预测与每个锚框的小偏移，而不是从零回归。

### 解码预测、IoU、NMS 和 YOLO 损失

（完整的概念部分涵盖框解码（sigmoid + exp）、IoU 计算、NMS 算法的贪婪实现，以及组合物体性/分类/框回归损失——详见 `docs/en.md` 概念部分）

### 步骤 7：推理管道

解码原始检测头输出，应用 sigmoid/exp，按物体性阈值过滤，然后 NMS。

```python
def postprocess(pred_tensor, anchors, stride, img_size, conf_threshold=0.25, iou_threshold=0.45):
    pred = pred_tensor.detach().cpu().numpy()
    grid_h, grid_w = pred.shape[1], pred.shape[2]
    num_anchors = len(anchors)

    boxes, scores, classes = [], [], []
    for gy in range(grid_h):
        for gx in range(grid_w):
            for a in range(num_anchors):
                tx, ty, tw, th, obj, *cls = pred[0, gy, gx, a]
                score = sigmoid(obj) * sigmoid(np.array(cls)).max()
                if score < conf_threshold:
                    continue
                cls_idx = int(np.argmax(cls))
                cx = (sigmoid(tx) + gx) * stride
                cy = (sigmoid(ty) + gy) * stride
                w = anchors[a][0] * np.exp(tw)
                h = anchors[a][1] * np.exp(th)
                boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
                scores.append(float(score))
                classes.append(cls_idx)

    if not boxes:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=int)
    boxes = np.array(boxes)
    scores = np.array(scores)
    classes = np.array(classes)
    keep = nms(boxes, scores, iou_threshold)
    return boxes[keep], scores[keep], classes[keep]
```

这是完整的评估路径：检测头 -> 解码 -> 阈值过滤 -> NMS。

## 使用它

`torchvision.models.detection` 以与上述相同概念结构的内置生产检测器。

```python
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
model.eval()
with torch.no_grad():
    predictions = model([torch.randn(3, 400, 600)])
```

对于实时推理管道，`ultralytics`（YOLOv8/v9）是标准：`from ultralytics import YOLO; model = YOLO('yolov8n.pt'); model(img)`。

## 交付物

本课产出：
- `outputs/prompt-detection-metric-reader.md`——将 mAP 行转化为诊断的提示词
- `outputs/skill-anchor-designer.md`——数据集锚框设计的技能

## 练习

1. **（简单）** 实现 `box_iou` 并与 `torchvision.ops.box_iou` 在 1,000 对随机框上进行测试。验证最大绝对差低于 `1e-6`。
2. **（中等）** 将 `yolo_loss` 移植到使用 `CIoU` 框损失代替 MSE 的版本。展示 CIoU 在相同 epoch 数内收敛到更好的最终 mAP。
3. **（困难）** 实现多尺度推理：以三种分辨率将相同图像输入模型，合并框预测，并在最后运行单个 NMS。测量与单尺度推理相比的 mAP 提升。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|----------|
| Anchor | "框先验" | 每个网格单元的预定义框形状，网络从中预测偏移而不是绝对坐标 |
| IoU | "重叠" | 两个框的交并比；检测中的通用相似度度量 |
| NMS | "去重" | 贪心算法，保留最高分数预测并删除高于阈值的重叠预测 |
| Objectness | "这里有什么东西吗" | 每个锚框、每个单元的标量，预测物体是否中心在该单元 |
| Grid stride | "下采样因子" | 每个网格单元的像素数；416px 输入加 13 网格头有 stride 32 |
| mAP | "平均精度均值" | 精度-召回曲线下面积的平均值，按类别和（对 COCO）IoU 阈值平均 |
| AP@0.5 | "PASCAL VOC AP" | IoU 阈值 0.5 的平均精度 |
| mAP@0.5:0.95 | "COCO AP" | 在 IoU 阈值 0.5..0.95 步长 0.05 上的平均值；严格版本和当前社区标准 |

## 延伸阅读

- [YOLOv1: You Only Look Once (Redmon et al., 2016)](https://arxiv.org/abs/1506.02640) — 创始论文
- [YOLOv3 (Redmon & Farhadi, 2018)](https://arxiv.org/abs/1804.02767) — 引入多尺度 FPN 风格头的论文
- [Ultralytics YOLOv8 docs](https://docs.ultralytics.com) — 当前生产参考
- [The Illustrated Guide to Object Detection (Jonathan Hui)](https://jonathan-hui.medium.com/object-detection-series-24d03a12f904) — 整个检测器系列的最佳通俗解释
