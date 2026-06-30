# 说话人识别与验证

> ASR 问"他们说了什么？"说话人识别问"谁说的？"数学看起来相同——嵌入加余弦——但每个生产决定都取决于一个单一的 EER 数字。

**类型：** 构建
**语言：** Python
**先修要求：** 第六阶段 · 02（频谱图与 Mel），第五阶段 · 22（嵌入模型）
**预计时间：** 约45分钟

## 问题

一个用户说出一句口令。你想知道：这是他们自称的那个人吗（*验证*，1:1），还是你的注册库中的第一个人（*识别*，1:N）？或都不是——这是一个未知的说话人（*开放集*）？

2018 年前：GMM-UBM + i-vectors。合理的 EER 但对信道转移（电话 vs 笔记本）和情绪脆弱。2018–2022：x-vectors（用角度边际训练的 TDNN 骨干）。2022+：ECAPA-TDNN 和 WavLM-large 嵌入。到 2026 年该领域由三个模型和一个指标主导。

指标是 **EER** — 等错误率。设置你的决策阈值使得误接受率 = 误拒绝率。交叉点就是 EER。用于每篇论文、每个排行榜、每个采购电话。

## 概念

![Enrollment + verification pipeline with embedding + cosine + EER](../assets/speaker-verification.svg)

**流程。** 注册：录制 5–30 秒目标说话人音频；计算固定维度的嵌入（ECAPA-TDNN 为 192-d，WavLM-large 为 256-d）。验证：获取测试话语嵌入；计算余弦相似度；与阈值比较。

**ECAPA-TDNN（2020，2026 年仍主导）。** 强调通道注意力、传播和聚合 - 时延神经网络。带挤压-激励的 1D 卷积块、多头注意力池化，后接线性层到 192-d。在 VoxCeleb 1+2（2,700 说话人、1.1M 话语）上以附加角度边际损失（AAM-softmax）训练。

**WavLM-SV（2022+）。** 用 AAM 损失微调预训练的 WavLM-large SSL 骨干。更高质量但更慢——300+ MB vs 15 MB。

**x-vector（基线）。** TDNN + 统计池化。经典；在 CPU / 边缘上仍然有用。

**AAM-softmax。** 在角度空间中带有附加边际 `m` 的标准 softmax：正确类别的 `cos(θ + m)`。强制类间角度分离。典型 `m=0.2`，缩放 `s=30`。

### 评分

- **余弦** 在注册和测试嵌入之间。基于阈值的决策。
- **PLDA（概率 LDA）。** 将嵌入投影到潜在空间，其中相同说话人与不同说话人具有闭式似然比。添加在余弦之上以减少 +10–20% 的 EER。2020 年前的标准；现在仅用于封闭集设置。
- **分数归一化。** `S-norm` 或 `AS-norm`：相对于冒名顶替者均值和标准差的队列归一化每个分数。对跨域评估至关重要。

### 你应该知道的数字（2026）

| 模型 | VoxCeleb1-O EER | 参数 | 吞吐量（A100） |
|-------|-----------------|--------|-------------------|
| x-vector（经典） | 3.10% | 5 M | 400× RT |
| ECAPA-TDNN | 0.87% | 15 M | 200× RT |
| WavLM-SV large | 0.42% | 316 M | 20× RT |
| Pyannote 3.1 segmentation + embedding | 0.65% | 6 M | 100× RT |
| ReDimNet（2024） | 0.39% | 24 M | 100× RT |

### 说话人分割

多说话人片段中"谁在何时说话"。流程：VAD → 分段 → 嵌入每个分段 → 聚类（凝聚式或谱聚类）→ 平滑边界。现代技术栈：`pyannote.audio` 3.1，它将说话人分割 + 嵌入 + 聚类打包在一个调用后。2026 年 AMI 上的最先进 DER 约为 15%（从 2022 年的 23% 下降）。

## 构建它

### 步骤 1：从 MFCC 统计量的玩具嵌入

```python
def embed_mfcc_stats(signal, sr):
    frames = featurize_mfcc(signal, sr, n_mfcc=13)
    mean = [sum(f[i] for f in frames) / len(frames) for i in range(13)]
    std = [
        math.sqrt(sum((f[i] - mean[i]) ** 2 for f in frames) / len(frames))
        for i in range(13)
    ]
    return mean + std  # 26-d
```

离最先进差很远——仅用于教学。`code/main.py` 在合成说话人数据上将其用作概念验证。

### 步骤 2：余弦相似度 + 阈值

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0

def verify(enroll, test, threshold=0.75):
    return cosine(enroll, test) >= threshold
```

### 步骤 3：从相似度对计算 EER

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 1.0, 0.0)  # (fa, fr, threshold)
    for t in thresholds:
        fr = sum(1 for s in same_scores if s < t) / len(same_scores)
        fa = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        if abs(fa - fr) < abs(best[0] - best[1]):
            best = (fa, fr, t)
    return (best[0] + best[1]) / 2, best[2]
```

返回 (eer, threshold_at_eer)。报告两者。

### 步骤 4：用 SpeechBrain 进行生产

```python
from speechbrain.pretrained import EncoderClassifier

clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# enroll: 平均 3-5 个干净样本的嵌入
enroll = torch.stack([clf.encode_batch(load(x)) for x in enrollment_clips]).mean(0)
# verify
score = clf.similarity(enroll, clf.encode_batch(load("test.wav"))).item()
verdict = score > 0.25   # ECAPA 典型阈值；根据你的数据调优
```

### 步骤 5：用 pyannote 进行说话人分割

```python
from pyannote.audio import Pipeline

pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
diarization = pipe("meeting.wav", num_speakers=None)
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:.1f}–{turn.end:.1f}  {speaker}")
```

## 使用它

2026 年技术栈：

| 场景 | 选择 |
|-----------|------|
| 封闭集 1:1 验证、边缘 | ECAPA-TDNN + 余弦阈值 |
| 开放集验证、云 | WavLM-SV + AS-norm |
| 说话人分割（会议、播客） | `pyannote/speaker-diarization-3.1` |
| 反欺骗（重放 / 深度伪造检测） | AASIST 或 RawNet2 |
| 微型嵌入式（KWS + 注册） | Titanet-Small（NeMo） |

## 陷阱

- **信道不匹配。** 在 VoxCeleb（网络视频）上训练的模型 ≠ 电话呼叫音频。始终在目标信道上评估。
- **短话语。** EER 在低于 3 秒的测试音频下急剧下降。
- **带噪声注册。** 一个噪声注册样本污染了锚点。使用 ≥3 个干净样本并取平均。
- **跨条件固定阈值。** 始终在目标领域的保留开发集上调优阈值。
- **在非归一化嵌入上的余弦。** 首先进行 L2 归一化；否则幅度占主导。

## 交付它

保存为 `outputs/skill-speaker-verifier.md`。选择模型、注册协议、阈值调优计划和欺诈防护。

## 练习

1. **简单。** 运行 `code/main.py`。构建合成"说话人"（不同音调特征），注册，在 100 个对测试列表上计算 EER。
2. **中等。** 在 30 个 VoxCeleb1 话语（5 说话人 × 每人 6 个）上使用 SpeechBrain ECAPA。计算余弦 vs PLDA 的 EER。
3. **困难。** 用 `pyannote.audio` 构建完整的注册 → 说话人分割 → 验证流程。在 AMI dev 集上评估 DER。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| EER | 主要指标 | 误接受 = 误拒绝时的阈值。 |
| 验证 | 1:1 | "这是 Alice 吗？" |
| 识别 | 1:N | "谁在说话？" |
| 开放集 | 可能有未知者 | 测试集可以包含未注册的说话人。 |
| 注册 | 登记 | 计算说话人的参考嵌入。 |
| AAM-softmax | 损失函数 | 带附加角度边际的 softmax；强制簇分离。 |
| PLDA | 经典评分 | 概率 LDA；在嵌入之上的似然比评分。 |
| DER | 说话人分割指标 | 分割错误率——遗漏 + 误报 + 混淆。 |

## 扩展阅读

- [Snyder et al. (2018). X-Vectors: Robust DNN Embeddings for Speaker Recognition](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf) — 经典深度嵌入论文。
- [Desplanques et al. (2020). ECAPA-TDNN](https://arxiv.org/abs/2005.07143) — 2020–2026 主导架构。
- [Chen et al. (2022). WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://arxiv.org/abs/2110.13900) — SV 和说话人分割的 SSL 骨干。
- [Bredin et al. (2023). pyannote.audio 3.1](https://github.com/pyannote/pyannote-audio) — 生产说话人分割 + 嵌入技术栈。
- [VoxCeleb leaderboard (updated 2026)](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) — 当前跨模型的 EER 排名。
