# 音频分类 — 从 MFCC 上的 k-NN 到 AST 和 BEATs

> 从"狗叫 vs 警笛"到"这是什么语言"的一切都是音频分类。特征是 Mel。架构每十年移动。评估保持 AUC、F1 和每类召回率。

**类型：** 构建
**语言：** Python
**先修要求：** 第六阶段 · 02（频谱图与 Mel），第三阶段 · 06（CNN），第五阶段 · 08（文本上的 CNN 和 RNN）
**预计时间：** 约75分钟

## 问题

你获得一个 10 秒片段。你想知道："它是什么？"城市声音（警笛、电钻、狗）、语音命令（yes/no/stop）、语言 ID（en/es/ar）、说话人情绪（愤怒/中性），或环境声音（室内/室外、嘈杂人声）。所有这些都是*音频分类*，在 2026 年基线架构已经成熟：对数 Mel → CNN 或 Transformer → softmax。

核心困难不是网络。是数据。音频数据集有严重的类别不均衡、强大的领域转移（干净 vs 嘈杂）和标签噪声（谁决定了"urban babble"和"restaurant noise"？）。80% 的问题是策划、增强和评估，而不是将 CNN 换成 Transformer。

## 概念

![Audio classification ladder: k-NN on MFCCs to AST to BEATs](../assets/audio-classification.svg)

**MFCC 上的 k-NN（1990 年代基线）。** 展平每个片段的 MFCC，计算与标注库的余弦相似度，返回 top K 的多数投票。在干净的小型数据集（Speech Commands、ESC-50）上惊人地强大。无需 GPU 运行。

**对数 Mel 上的 2D CNN（2015-2019）。** 将 `(T, n_mels)` 对数 Mel 视为图像。应用 ResNet-18 或 VGG 风格。对时间轴进行全局均值池化。在类别上 softmax。在大多数 2026 kaggle 比赛中仍然是基线。

**Audio Spectrogram Transformer，AST（2021-2024）。** 将对数 Mel 分块（例如 16×16 块）、添加位置嵌入、输入 ViT。在有监督学习的 AudioSet（mAP 0.485）上是最先进的。

**BEATs 和 WavLM-base（2024-2026）。** 在数百万小时上的自监督预训练。用你所需监督数据的 1-10% 在你的任务上微调。在 2026 年，这是非语音音频的默认起点。BEATs-iter3 在 AudioSet 上以 1/4 的计算量比 AST 高出 1-2 mAP。

**Whisper 编码器作为冻结骨干（2024）。** 取 Whisper 的编码器、丢弃解码器、附加线性分类器。在没有音频增强的情况下，在语言 ID 和简单事件分类上接近最先进。"免费午餐"基线。

### 类别不均衡是真正的挑战

ESC-50：50 类，每类 40 个片段——均衡、简单。UrbanSound8K：10 类，10:1 不均衡。AudioSet：632 类，具有 100,000:1 的长尾。有效的技术：

- 在训练期间（而不是评估期间）进行均衡采样。
- Mixup：线性插值两个片段（及其标签）作为增强。
- SpecAugment：掩盖随机时间和频率带。简单；关键。

### 评估

- 互斥多类（Speech Commands）：top-1 准确率、top-5 准确率。
- 多类多标签（AudioSet、UrbanSound 风格）：平均精度均值（mAP）。
- 严重不均衡：每类召回率 + 宏 F1。

你应该知道的 2026 年数字：

| 基准 | 基线 | 2026 最先进 | 来源 |
|-----------|----------|-----------|--------|
| ESC-50 | 82%（AST） | 97.0%（BEATs-iter3） | BEATs 论文（2024） |
| AudioSet mAP | 0.485（AST） | 0.548（BEATs-iter3） | HEAR 排行榜 2026 |
| Speech Commands v2 | 98%（CNN） | 99.0%（Audio-MAE） | HEAR v2 结果 |

## 构建它

### 步骤 1：特征化

```python
def featurize_mfcc(signal, sr, n_mfcc=13, n_mels=40, frame_len=400, hop=160):
    mag = stft_magnitude(signal, frame_len, hop)
    fb = mel_filterbank(n_mels, frame_len, sr)
    mels = apply_filterbank(mag, fb)
    log = log_transform(mels)
    return [dct_ii(frame, n_mfcc) for frame in log]
```

### 步骤 2：固定长度摘要

```python
def summarize(mfcc_frames):
    n = len(mfcc_frames[0])
    mean = [sum(f[i] for f in mfcc_frames) / len(mfcc_frames) for i in range(n)]
    var = [
        sum((f[i] - mean[i]) ** 2 for f in mfcc_frames) / len(mfcc_frames) for i in range(n)
    ]
    return mean + var
```

简单但强大：时间均值 + 方差给出 13 系数 MFCC 的 26 维固定嵌入。立即运行。在 2017 年于 ESC-50 上击败了最先进的 NN 基线。

### 步骤 3：k-NN

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(x * x for x in b)) or 1e-12
    return dot / (na * nb)

def knn_classify(q, bank, labels, k=5):
    sims = sorted(range(len(bank)), key=lambda i: -cosine(q, bank[i]))[:k]
    votes = Counter(labels[i] for i in sims)
    return votes.most_common(1)[0][0]
```

### 步骤 4：升级到对数 Mel 上的 CNN

在 PyTorch 中：

```python
import torch.nn as nn

class AudioCNN(nn.Module):
    def __init__(self, n_mels=80, n_classes=50):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):  # x: (B, 1, T, n_mels)
        return self.head(self.body(x).flatten(1))
```

300 万参数。在单个 RTX 4090 上约 10 分钟训练 ESC-50。80%+ 准确率。

### 步骤 5：2026 默认——微调 BEATs

```python
from transformers import ASTFeatureExtractor, ASTForAudioClassification

ext = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=50,
    ignore_mismatched_sizes=True,
)

inputs = ext(audio, sampling_rate=16000, return_tensors="pt")
logits = model(**inputs).logits
```

对于 BEATs，通过 `beats` 库使用 `microsoft/BEATs-base`；transformers API 的形状相同。

## 使用它

2026 年技术栈：

| 场景 | 起始点 |
|-----------|-----------|
| 微小数据集（<1000 片段） | MFCC 均值上的 k-NN（你的基线）+ 音频增强 |
| 中等数据集（1K–100K） | BEATs 或 AST 微调 |
| 大数据集（>100K） | 从零训练或微调 Whisper 编码器 |
| 实时、边缘 | 40-MFCC CNN，量化为 int8（KWS 风格） |
| 多标签（AudioSet） | 带 BCE 损失 + mixup + SpecAugment 的 BEATs-iter3 |
| 语言 ID | MMS-LID、SpeechBrain VoxLingua107 基线 |

决策规则：**从冻结骨干开始，而不是新模型。** 微调 BEATs 头在数小时而不是数周内让你达到 95% 的最先进。

## 交付它

保存为 `outputs/skill-classifier-designer.md`。为给定的音频分类任务选择架构、增强、类别均衡策略和评估指标。

## 练习

1. **简单。** 运行 `code/main.py`。它在 4 类合成数据集（不同音高的纯音）上训练 k-NN MFCC 基线。报告混淆矩阵。
2. **中等。** 将 `summarize` 替换为 [mean, var, skew, kurtosis]。4 阶矩池化在相同合成数据集上是否优于均值+方差？
3. **困难。** 使用 `torchaudio` 在 ESC-50 fold 1 上训练 2D CNN。报告 5 折交叉验证准确率。添加 SpecAugment（time mask = 20、freq mask = 10）并报告增量。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| AudioSet | 音频的 ImageNet | Google 的 2M 片段、632 类弱标注 YouTube 数据集。 |
| ESC-50 | 小型分类基准 | 50 类 × 40 片段的环境声音。 |
| AST | 音频频谱图 Transformer | 在对数 Mel 块上的 ViT；2021 最先进。 |
| BEATs | 自监督音频 | 微软模型，iter3 截至 2026 年在 AudioSet 领先。 |
| Mixup | 对增强 | `x = λ·x1 + (1-λ)·x2; y = λ·y1 + (1-λ)·y2`。 |
| SpecAugment | 基于掩模的增强 | 将频谱图的随机时间和频率带置零。 |
| mAP | 主要多标签指标 | 跨类别和阈值的平均精度均值。 |

## 扩展阅读

- [Gong, Chung, Glass (2021). AST: Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778) — 2021–2024 年的记录架构。
- [Chen et al. (2022, rev. 2024). BEATs: Audio Pre-Training with Acoustic Tokenizers](https://arxiv.org/abs/2212.09058) — 2024+ 默认。
- [Park et al. (2019). SpecAugment](https://arxiv.org/abs/1904.08779) — 主导音频增强方法。
- [Piczak (2015). ESC-50 dataset](https://github.com/karolpiczak/ESC-50) — 持续存在的 50 类基准。
- [Gemmeke et al. (2017). AudioSet](https://research.google.com/audioset/) — 632 类 YouTube 分类法；仍然是黄金标准。
