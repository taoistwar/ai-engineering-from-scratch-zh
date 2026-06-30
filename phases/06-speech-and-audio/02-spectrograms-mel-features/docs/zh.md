# 频谱图、Mel 尺度与音频特征

> 神经网络不能很好地消费原始波形。它们消费频谱图。它们消费 Mel 频谱图更好。2026 年的每个 ASR、TTS 和音频分类器都取决于这个单一的预处理选择。

**类型：** 构建
**语言：** Python
**先修要求：** 第六阶段 · 01（音频基础）
**预计时间：** 约45分钟

## 问题

取一个 10 秒 16 kHz 的片段。那是 160,000 个浮点数，都在 `[-1, 1]` 中，几乎与标签"狗叫"或"词cat"完全不相关。原始波形具有信息，但以模型无法轻易提取的形式存在。相隔 100 毫秒说的两个相同音素有完全不同的原始样本。

频谱图修复了这个问题。它在人类感知忽略的地方（微秒抖动）折叠时间细节，并在感知关注的地方（哪些频率是活跃的，在约 10–25 ms 的时间窗口上）保留结构。

Mel 频谱图更进一步。人类以对数方式感知音高：100 Hz 与 200 Hz 听起来和 1000 Hz 与 2000 Hz "距离相同"。Mel 尺度通过变形频率轴来匹配这一点。Mel 尺度频谱图是 2010 到 2026 年语音 ML 中唯一最重要的特征。

## 概念

![Waveform to STFT to mel spectrogram to MFCC ladder](../assets/mel-features.svg)

**STFT（短时傅里叶变换）。** 将波形切成重叠的帧（典型：25 ms 窗口、10 ms 跳跃 = 在 16 kHz 下 400 样本 / 160 样本）。将每帧乘以窗口函数（Hann 是默认；Hamming 有略微不同的权衡）。对每帧进行 FFT。将幅度谱堆叠为 `(n_frames, n_freq_bins)` 形状的矩阵。那就是你的频谱图。

**对数幅度。** 原始幅度跨越 5-6 个数量级。取 `log(|X| + 1e-6)` 或 `20 * log10(|X|)` 来压缩动态范围。每个生产流程使用对数幅度，而不是原始幅度。

**Mel 尺度。** 频率 `f` (Hz) 映射到 mel `m` 通过 `m = 2595 * log10(1 + f / 700)`。映射在 1 kHz 以下大致线性，以上大致对数。80 个 mel bin 覆盖 0–8 kHz 是标准的 ASR 输入。

**Mel 滤波器组。** 一组在 Mel 尺度上等间隔的三角滤波器。每个滤波器是相邻 FFT bin 的加权和。将 STFT 幅度乘以滤波器组矩阵在一个矩阵乘法中得到 Mel 频谱图。

**对数 Mel 频谱图。** `log(mel_spec + 1e-10)`。Whisper 的输入。Parakeet 的输入。SeamlessM4T 的输入。2026 年通用的音频前端。

**MFCC。** 取对数 Mel 频谱图，应用 DCT（类型 II），保留前 13 个系数。对特征进行去相关并进一步压缩。直到 2015 年左右占据主导地位，当时 CNN/Transformer 在原始对数 Mel 上赶了上来。仍在说话人识别（x-vectors、ECAPA）中使用。

**分辨率权衡。** 更大的 FFT = 更好的频率分辨率但更差的时间分辨率。25 ms / 10 ms 是音频 ML 默认值；50 ms / 12.5 ms 用于音乐；5 ms / 2 ms 用于瞬态检测（鼓点、爆破音）。

```figure
spectrogram-window
```

## 构建它

### 步骤 1：分帧波形

```python
def frame(signal, frame_len, hop):
    n = 1 + (len(signal) - frame_len) // hop
    return [signal[i * hop : i * hop + frame_len] for i in range(n)]
```

带 `frame_len=400, hop=160` 的 10 秒 16 kHz 片段产生 998 帧。

### 步骤 2：Hann 窗口

```python
import math

def hann(N):
    return [0.5 * (1 - math.cos(2 * math.pi * n / (N - 1))) for n in range(N)]
```

在 FFT 之前逐元素相乘。消除由非零端点截断引起的频谱泄漏。

### 步骤 3：STFT 幅度

```python
def stft_magnitude(signal, frame_len=400, hop=160):
    win = hann(frame_len)
    frames = frame(signal, frame_len, hop)
    return [magnitudes(dft([w * s for w, s in zip(win, f)])) for f in frames]
```

生产使用 `torch.stft` 或 `librosa.stft`（FFT 支持、矢量化）。这里的循环是教学性质的；它在 `code/main.py` 中的短片段上运行。

### 步骤 4：Mel 滤波器组

```python
def hz_to_mel(f):
    return 2595.0 * math.log10(1.0 + f / 700.0)

def mel_to_hz(m):
    return 700.0 * (10 ** (m / 2595.0) - 1)

def mel_filterbank(n_mels, n_fft, sr, fmin=0, fmax=None):
    fmax = fmax or sr / 2
    mels = [hz_to_mel(fmin) + (hz_to_mel(fmax) - hz_to_mel(fmin)) * i / (n_mels + 1)
            for i in range(n_mels + 2)]
    hzs = [mel_to_hz(m) for m in mels]
    bins = [int(h * n_fft / sr) for h in hzs]
    fb = [[0.0] * (n_fft // 2 + 1) for _ in range(n_mels)]
    for m in range(n_mels):
        for k in range(bins[m], bins[m + 1]):
            fb[m][k] = (k - bins[m]) / max(1, bins[m + 1] - bins[m])
        for k in range(bins[m + 1], bins[m + 2]):
            fb[m][k] = (bins[m + 2] - k) / max(1, bins[m + 2] - bins[m + 1])
    return fb
```

80 mel 覆盖 0–8 kHz，`n_fft=400` 给出一个 `(80, 201)` 矩阵。将 `(n_frames, 201)` STFT 幅度乘以转置得到 `(n_frames, 80)` Mel 频谱图。

### 步骤 5：对数 Mel

```python
def log_mel(mel_spec, eps=1e-10):
    return [[math.log(max(v, eps)) for v in frame] for frame in mel_spec]
```

常见替代：`librosa.power_to_db`（参考归一化 dB）、`10 * log10(power + eps)`。Whisper 使用更复杂的剪切 + 归一化例程（参见 Whisper 的 `log_mel_spectrogram`）。

### 步骤 6：MFCC

```python
def dct_ii(x, n_coeffs):
    N = len(x)
    return [
        sum(x[n] * math.cos(math.pi * k * (2 * n + 1) / (2 * N)) for n in range(N))
        for k in range(n_coeffs)
    ]
```

对每个对数 Mel 帧应用 DCT，保留前 13 个系数。那就是你的 MFCC 矩阵。第一个系数通常被丢弃（它编码总体能量）。

## 使用它

2026 年技术栈：

| 任务 | 特征 |
|------|----------|
| ASR（Whisper、Parakeet、SeamlessM4T） | 80 对数 Mel、10 ms 跳跃、25 ms 窗口 |
| TTS 声学模型（VITS、F5-TTS、Kokoro） | 80 Mel、5–12 ms 跳跃用于精细时间控制 |
| 音频分类（AST、PANNs、BEATs） | 128 对数 Mel、10 ms 跳跃 |
| 说话人嵌入（ECAPA-TDNN、WavLM） | 80 对数 Mel 或原始波形 SSL |
| 音乐（MusicGen、Stable Audio 2） | EnCodec 离散 token（非 Mel） |
| 关键词识别 | 40 MFCC 用于小型设备 |

经验法则：**如果你不是在处理音乐，从 80 对数 Mel 开始。** 任何偏离的举证责任都在你。

## 2026 年仍未被捕获就发布的陷阱

- **Mel 数量不匹配。** 用 80 Mel 训练，用 128 Mel 推理。静默失败。在两端记录特征形状。
- **上游采样率不匹配。** 在 22.05 kHz 计算的 Mel 看起来与 16 kHz 不同。在特征化之前修复 SR。
- **dB vs 对数。** Whisper 期望对数 Mel，而不是 dB-Mel。某些 HF 流程自动检测；你的自定义代码不会。
- **归一化漂移。** 训练期间每句话归一化，推理期间全局归一化。使 WER 翻倍的生产 bug。
- **来自填充的泄漏。** 在片段末尾零填充在尾随帧中产生平坦频谱。对称填充或复制。

## 交付它

保存为 `outputs/skill-feature-extractor.md`。该技能为给定的模型目标选择特征类型、Mel 数量、帧/跳跃和归一化。

## 练习

1. **简单。** 运行 `code/main.py`。它合成一个啁啾信号（频率从 200 → 4000 Hz 扫描）并打印每帧的 argmax Mel bin。绘制（可选）并确认它与扫描匹配。
2. **中等。** 以 `n_mels` ∈ `{40, 80, 128}` 和 `frame_len` ∈ `{200, 400, 800}` 重新运行。测量时间轴上的锐峰带宽。哪种组合最优化地解析啁啾？
3. **困难。** 实现 `power_to_db` 并使用小型 CNN 分类器在 AudioMNIST 上比较 ASR 准确率：(a) 原始对数 Mel，(b) dB-Mel `ref=max`，(c) MFCC-13 + delta + delta-delta。报告 top-1 准确率。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| 帧 | 一个切片 | 输入一个 FFT 的 25 ms 波形块。 |
| 跳跃 | 步幅 | 连续帧之间的样本；10 ms 是 ASR 默认值。 |
| 窗口 | Hann/Hamming 的东西 | 将帧边缘逐渐缩减到零的逐点乘法。 |
| STFT | 频谱图生成器 | 分帧 + 窗口 FFT；产生时间 × 频率矩阵。 |
| Mel | 变形频率 | 对数感知尺度；`m = 2595·log10(1 + f/700)`。 |
| 滤波器组 | 矩阵 | 将 STFT 投影到 Mel bin 的三角滤波器。 |
| 对数 Mel | Whisper 的输入 | `log(mel_spec + eps)`；2026 年标准化。 |
| MFCC | 老派特征 | 对数 Mel 的 DCT；13 个系数，去相关。 |

## 扩展阅读

- [Davis, Mermelstein (1980). Comparison of parametric representations for monosyllabic word recognition](https://ieeexplore.ieee.org/document/1163420) — MFCC 论文。
- [Stevens, Volkmann, Newman (1937). A Scale for the Measurement of the Psychological Magnitude Pitch](https://pubs.aip.org/asa/jasa/article-abstract/8/3/185/735757/) — 原始的 Mel 尺度。
- [OpenAI — Whisper source, log_mel_spectrogram](https://github.com/openai/whisper/blob/main/whisper/audio.py) — 阅读参考实现。
- [librosa feature extraction docs](https://librosa.org/doc/main/feature.html) — `mfcc`、`melspectrogram` 和 hop/window 的参考。
- [NVIDIA NeMo — audio preprocessing](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/asr_all.html#featurizers) — Parakeet + Canary 模型的生产规模流程。
