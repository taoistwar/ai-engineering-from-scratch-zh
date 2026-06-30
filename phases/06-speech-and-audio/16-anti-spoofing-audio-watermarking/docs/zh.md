# 语音反欺骗与音频水印 — ASVspoof 5、AudioSeal、WaveVerify

> 声音克隆的发布速度快于防御。2026 年生产语音系统需要两样东西：一个分类真实 vs 虚假语音的检测器（AASIST、RawNet2），以及一个能在压缩和编辑后存活的水印（AudioSeal）。同时发布两者，否则不要发布声音克隆。

**类型：** 构建
**语言：** Python
**先修要求：** 第六阶段 · 06（说话人识别），第六阶段 · 08（声音克隆）
**预计时间：** 约75分钟

## 问题

三种相关的防御：

1. **反欺骗 / 深度伪造检测。** 给定一个音频片段，它是合成的还是真实的？ASVspoof 基准测试（ASVspoof 2019 → 2021 → 5）是黄金标准。
2. **音频水印。** 在生成的音频中嵌入一个不可感知的信号，检测器之后可以提取。AudioSeal（Meta）和 WavMark 是开源选项。
3. **认证来源。** 音频文件 + 元数据的密码学签名。C2PA / Content Authenticity Initiative。

检测处理不合作的对手。水印处理合规——AI 生成的音频应可识为此类。两者在 2026 年都是必需的。

## 概念

![Anti-spoofing vs watermarking vs provenance — three defense layers](../assets/spoofing-watermark.svg)

### ASVspoof 5 —— 2024-2025 基准

与先前版本最大的变化：

- **众包数据**（非录音棚干净）——真实条件。
- **约 2000 名说话人**（之前约 100 名）。
- **32 种攻击算法。** TTS + 语音转换 + 对抗性扰动。
- **两条赛道。** 对策（CM）独立检测；欺骗鲁棒 ASV（SASV）用于生物识别系统。

ASVspoof 5 上的最先进水平：约 7.23% EER。在较旧的 ASVspoof 2019 LA 上：0.42% EER。真实世界部署：在野外片段上预期 5-10% EER。

### AASIST 和 RawNet2 —— 检测模型族

**AASIST**（2021，持续更新至 2026）。频谱特征上的图注意力。ASVspoof 5 对策任务上的当前最先进。

**RawNet2。** 原始波形上的卷积前端 + TDNN 骨干。更简单的基线；微调后仍具竞争力。

**NeXt-TDNN + SSL 特征。** 2025 变体：ECAPA 风格 + WavLM 特征 + focal loss。在 ASVspoof 2019 LA 上达到 0.42% EER。

### AudioSeal —— 2024 年水印默认选择

Meta 的 **AudioSeal**（2024 年 1 月，v0.2 2024 年 12 月）。关键设计：

- **本地化。** 以 16 kHz 样本分辨率（1/16000 s）检测每帧水印。
- **生成器 + 检测器联合训练。** 生成器学习嵌入听不见的信号；检测器学习通过增强找到它。
- **鲁棒。** 能在 MP3 / AAC 压缩、EQ、±10% 变速、+10 dB SNR 噪声混合后存活。
- **快速。** 检测器以 485 倍实时运行；比 WavMark 快 1000 倍。
- **容量。** 16 位有效载荷（可编码模型 ID、生成时间戳、用户 ID），可在每个话语中嵌入。

### WavMark

AudioSeal 之前的开源基线。可逆神经网络，32 bits/sec。问题：

- 同步暴力破解慢。
- 能被高斯噪声或 MP3 压缩移除。
- 不实时友好。

### WaveVerify（2025 年 7 月）

解决 AudioSeal 的弱点——特别是时间操纵（反转、变速）。使用基于 FiLM 的生成器 + 专家混合检测器。在标准攻击上可与 AudioSeal 竞争；处理时间编辑。

### 对手利用的差距

来自 AudioMarkBench："在音高偏移下，所有水印的比特恢复准确率低于 0.6，表明接近完全移除。"**音高偏移是通用攻击。** 没有 2026 年水印对激进的音高修改是完全鲁棒的。这就是为什么你需要检测器（AASIST）伴随水印。

### C2PA / Content Authenticity Initiative

不是 ML 技术——一种清单格式。音频文件携带关于创建工具、作者、日期的密码学签名元数据。Audobox / Seamless 使用它。对来源有好处；如果不良行为者重新编码并剥离元数据则什么也做不了。

## 构建它

### 步骤 1：一个简单的频谱特征检测器（玩具）

```python
def spectral_rolloff(spec, percentile=0.85):
    cum = 0
    total = sum(spec)
    if total == 0:
        return 0
    threshold = total * percentile
    for k, v in enumerate(spec):
        cum += v
        if cum >= threshold:
            return k
    return len(spec) - 1

def is_suspicious(audio):
    spec = magnitude_spectrum(audio)
    rolloff = spectral_rolloff(spec)
    return rolloff / len(spec) > 0.92
```

合成语音通常有异常平坦的高频能量。生产检测器使用 AASIST，而不是这个。但直觉成立。

### 步骤 2：AudioSeal 嵌入 + 检测

```python
from audioseal import AudioSeal
import torch

generator = AudioSeal.load_generator("audioseal_wm_16bits")
detector = AudioSeal.load_detector("audioseal_detector_16bits")

audio = load_wav("generated.wav", sr=16000)[None, None, :]
payload = torch.tensor([[1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0]])
watermark = generator.get_watermark(audio, sample_rate=16000, message=payload)
watermarked = audio + watermark

result, decoded_payload = detector.detect_watermark(watermarked, sample_rate=16000)
# result: [0, 1] 中的 float — 水印存在的概率
# decoded_payload: 16 bits; 与嵌入的有效载荷匹配
```

### 步骤 3：评估 —— EER

```python
def eer(real_scores, fake_scores):
    thresholds = sorted(set(real_scores + fake_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in fake_scores if s >= t) / len(fake_scores)
        frr = sum(1 for s in real_scores if s < t) / len(real_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

### 步骤 4：生产集成

```python
def safe_tts(text, voice, clone_reference=None):
    if clone_reference is not None:
        verify_consent(user_id, clone_reference)
    audio = tts_model.synthesize(text, voice)
    audio_with_wm = audioseal_embed(audio, payload=build_payload(user_id, model_id))
    manifest = c2pa_sign(audio_with_wm, user_id, timestamp=now())
    return audio_with_wm, manifest
```

每次生成发布：(1) 水印、(2) 签名清单、(3) 保留策略合规的审计日志。

## 使用它

| 用例 | 防御 |
|----------|---------|
| 发布 TTS / 声音克隆 | 在每个输出上嵌入 AudioSeal（不可协商） |
| 生物识别语音解锁 | AASIST + ECAPA 集成；活跃度挑战 |
| 呼叫中心欺诈检测 | 在 20% 来电样本上的 AASIST |
| 播客真实性 | 上传时 C2PA 签名，如果 AI 生成则 AudioSeal |
| 研究 / 训练检测器 | ASVspoof 5 train/dev/eval 集 |

## 陷阱

- **水印但没有检测器运行。** 毫无意义。在 CI 中发布检测器。
- **检测器没有校准。** 在 ASVspoof LA 上训练的 AASIST 过拟合；真实世界准确率下降。在你的领域上校准。
- **音高偏移差距。** 激进的音高偏移移除大多数水印。有一个检测后备。
- **元数据剥离和重新托管。** C2PA 可通过重新编码轻易绕过。始终将密码学 + 感知（水印）防御一起添加。
- **活跃度作为检测。** 要求用户说一个随机短语。防止重放攻击但不防止实时克隆。

## 交付它

保存为 `outputs/skill-spoof-defender.md`。为语音生成部署选择检测模型、水印、来源清单和操作手册。

## 练习

1. **简单。** 运行 `code/main.py`。玩具检测器 + 玩具水印在合成音频上嵌入/检测。
2. **中等。** 安装 `audioseal`，在 TTS 输出中嵌入 16 位有效载荷，重新解码。用噪声损坏音频并测量比特恢复准确率。
3. **困难。** 在 ASVspoof 2019 LA 上微调 RawNet2 或 AASIST。测量 EER。在保留的 F5-TTS 生成片段集上测试——看看分布外检测如何退化。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| ASVspoof | 基准测试 | 两年一次挑战；2024 = ASVspoof 5。 |
| CM（对策） | 检测器 | 分类器：真实语音 vs 合成 / 转换。 |
| SASV | 说话人验证 + CM | 集成生物识别 + 欺骗检测。 |
| AudioSeal | Meta 水印 | 本地化、16 位有效载荷、比 WavMark 快 485 倍。 |
| 比特恢复准确率 | 水印存活率 | 攻击后恢复的有效载荷比特比例。 |
| C2PA | 来源清单 | 关于创建 / 作者身份的密码学元数据。 |
| AASIST | 检测器族 | 基于图注意力的最先进反欺骗。 |

## 扩展阅读

- [Todisco et al. (2024). ASVspoof 5](https://dl.acm.org/doi/10.1016/j.csl.2025.101825) — 当前基准。
- [Defossez et al. (2024). AudioSeal](https://arxiv.org/abs/2401.17264) — 水印默认选择。
- [Chen et al. (2025). WaveVerify](https://arxiv.org/abs/2507.21150) — 用于时间攻击的 MoE 检测器。
- [Jung et al. (2022). AASIST](https://arxiv.org/abs/2110.01200) — 最先进检测骨干。
- [AudioMarkBench (2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/5d9b7775296a641a1913ab6b4425d5e8-Paper-Datasets_and_Benchmarks_Track.pdf) — 鲁棒性评估。
- [C2PA specification](https://c2pa.org/specifications/specifications/) — 来源清单格式。
