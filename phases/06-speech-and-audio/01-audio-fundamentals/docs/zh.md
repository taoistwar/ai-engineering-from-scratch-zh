# 音频基础 — 波形、采样、傅里叶变换

> 波形是原始信号。频谱图是表示形式。Mel 特征是 ML 友好的形式。每个现代 ASR 和 TTS 流程都走这个阶梯，而第一级是理解采样和傅里叶。

**类型：** 学习
**语言：** Python
**先修要求：** 第一阶段 · 06（向量与矩阵），第一阶段 · 14（概率分布）
**预计时间：** 约45分钟

## 问题

麦克风产生压力-时间信号。你的神经网络消费张量。在两者之间是一系列惯例，当被违反时会产生静默的 bug：模型训练正常但词错率翻倍，或 TTS 发出嘶声，或声音克隆系统记住了麦克风而不是说话者。

语音系统中的每个 bug 都可追溯到三个问题之一：

1. 数据以什么采样率录制，模型期望什么？
2. 信号是否混叠？
3. 你是在原始样本上操作还是在频率表示上操作？

搞对这些，第六阶段的其余部分就是可处理的。搞错了，即使是 Whisper-Large-v4 也会产生垃圾。

## 概念

![Waveform, sampling, DFT, and frequency bins visualized](../assets/audio-fundamentals.svg)

**波形。** 一维 `[-1.0, 1.0]` 浮点数组。按样本编号索引。要转换为秒，除以采样率：`t = n / sr`。16 kHz 下的 10 秒片段是一个 160,000 浮点数组。

**采样率（sr）。** 每秒有多少样本。2026 年常见采样率：

| 采样率 | 用途 |
|------|-----|
| 8 kHz | 电话通信、遗留 VOIP。4 kHz 的奈奎斯特频率杀死辅音。避免用于 ASR。 |
| 16 kHz | ASR 标准。Whisper、Parakeet、SeamlessM4T v2 都消费 16 kHz。 |
| 22.05 kHz | 旧模型的 TTS vocoder 训练。 |
| 24 kHz | 现代 TTS（Kokoro、F5-TTS、xTTS v2）。 |
| 44.1 kHz | CD 音频、音乐。 |
| 48 kHz | 电影、专业音频、高保真 TTS（VALL-E 2、NaturalSpeech 3）。 |

**奈奎斯特-香农。** 采样率 `sr` 可以无歧义地表示高达 `sr/2` 的频率。`sr/2` 边界是*奈奎斯特频率*。高于奈奎斯特的能量被*混叠*——折叠回较低频率——并破坏信号。在降采样前始终低通滤波。

**位深度。** 16 位 PCM（有符号 int16，范围 ±32,767）是通用交换格式。24 位用于音乐，32 位浮点用于内部 DSP。像 `soundfile` 这样的库读取 int16 但暴露 `[-1, 1]` 中的 float32 数组。

**傅里叶变换。** 任何有限信号是不同频率的正弦波之和。离散傅里叶变换（DFT）为 `N` 个样本计算 `N` 个复系数——每个频率 bin 一个。`bin k` 映射到频率 `k · sr / N` Hz。幅度是该频率的振幅，角度是相位。

**FFT。** 快速傅里叶变换：当 `N` 为 2 的幂时用于 DFT 的 `O(N log N)` 算法。每个音频库在底层使用 FFT。16 kHz 下的 1024 样本 FFT 在 15.6 Hz 分辨率下给出 512 个可用频率 bin，跨度 0–8 kHz。

**分帧 + 窗口。** 我们不对整个片段进行 FFT。我们将其切成重叠的*帧*（通常 25 ms，10 ms 跳跃），将每帧乘以窗口函数（Hann、Hamming）以消除边缘不连续性，然后对每帧进行 FFT。这就是短时傅里叶变换（STFT）。第 02 课从此处继续。

```figure
mel-scale
```

## 构建它

### 步骤 1：读取片段并绘制波形

`code/main.py`仅使用标准库的 `wave` 模块以保持演示无依赖。对于生产，你将使用 `soundfile` 或 `torchaudio.load`（都返回 `(waveform, sr)` 元组）：

```python
import soundfile as sf
waveform, sr = sf.read("clip.wav", dtype="float32")  # shape (T,), sr=int
```

### 步骤 2：从第一原理合成正弦波

```python
import math

def sine(freq_hz, sr, seconds, amp=0.5):
    n = int(sr * seconds)
    return [amp * math.sin(2 * math.pi * freq_hz * i / sr) for i in range(n)]
```

16 kHz 下 1 秒 440 Hz 正弦波（音乐会 A）是 16,000 个浮点数。使用 16 位 PCM 编码用 `wave.open(..., "wb")` 写入。

### 步骤 3：手工计算 DFT

```python
def dft(x):
    N = len(x)
    out = []
    for k in range(N):
        re = sum(x[n] * math.cos(-2 * math.pi * k * n / N) for n in range(N))
        im = sum(x[n] * math.sin(-2 * math.pi * k * n / N) for n in range(N))
        out.append((re, im))
    return out
```

`O(N²)`——对 `N=256` 可以确认正确性，对真实音频无用。真实代码调用 `numpy.fft.rfft` 或 `torch.fft.rfft`。

### 步骤 4：找到主导频率

幅度峰值索引 `k_star` 映射到频率 `k_star * sr / N`。在 440 Hz 正弦波上运行此过程应返回一个峰值在 bin `440 * N / sr`。

### 步骤 5：演示混叠

在 10 kHz（奈奎斯特 = 5 kHz）采样一个 7 kHz 正弦波。7 kHz 音高于奈奎斯特并折叠到 `10 − 7 = 3 kHz`。FFT 峰值出现在 3 kHz。这是经典的混叠演示，也是每个 DAC/ADC 都配有峭壁式低通滤波器的原因。

## 使用它

你在 2026 年实际部署的技术栈：

| 任务 | 库 | 为什么 |
|------|---------|-----|
| 读/写 WAV/FLAC/OGG | `soundfile`（libsndfile 包装器） | 最快、稳定、返回 float32。 |
| 重采样 | `torchaudio.transforms.Resample` 或 `librosa.resample` | 内置正确的抗混叠。 |
| STFT / Mel | `torchaudio` 或 `librosa` | GPU 友好；PyTorch 生态系统。 |
| 实时流式 | `sounddevice` 或 `pyaudio` | 跨平台 PortAudio 绑定。 |
| 检查文件 | `ffprobe` 或 `soxi` | CLI、快速、报告 sr/channels/codec。 |

决策规则：**在匹配任何其他东西之前匹配采样率**。Whisper 期望 16 kHz 单声道 float32。传递给它 44.1 kHz 立体声，你将得到看起来像模型 bug 的垃圾。

## 交付它

保存为 `outputs/skill-audio-loader.md`。该技能帮助你检查音频输入是否匹配下游模型的期望，并在不匹配时正确重采样。

## 练习

1. **简单。** 在 16 kHz 下合成 220 Hz + 440 Hz + 880 Hz 混合的 1 秒信号。运行 DFT。确认在预期 bin 处有三个峰值。
2. **中等。** 在 48 kHz 下录制 3 秒你的声音的 WAV。使用 `torchaudio.transforms.Resample`（带抗混叠）降采样到 16 kHz，然后使用朴素抽取（每隔三个样本取一个）降采样到 16 kHz。对两者进行 FFT。混叠出现在哪里？
3. **困难。** 仅使用 `math` 和步骤 3 的 DFT 从零构建 STFT。帧大小 400、跳跃 160、Hann 窗口。用 `matplotlib.pyplot.imshow` 绘制幅度。这是第 02 课的频谱图。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| 采样率 | 每秒多少样本 | ADC 测量信号的频率（Hz）。 |
| 奈奎斯特 | 你能表示的最大频率 | `sr/2`；高于它的能量混叠回来。 |
| 位深度 | 每个样本的分辨率 | `int16` = 65,536 个级别；`float32` = `[-1, 1]` 中的 24 位精度。 |
| DFT | 序列的傅里叶变换 | `N` 个样本 → `N` 个复频率系数。 |
| FFT | 快速 DFT | 需要 `N` = 2 的幂的 `O(N log N)` 算法。 |
| Bin | 频率列 | `k · sr / N` Hz；分辨率 = `sr / N`。 |
| STFT | 底层的频谱图 | 随时间进行的分帧 + 窗口 FFT。 |
| 混叠 | 奇怪的频率幽灵 | 高于奈奎斯特的能量镜像反射到更低的 bin。 |

## 扩展阅读

- [Shannon (1949). Communication in the Presence of Noise](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) — 采样定理背后的论文。
- [Smith — The Scientist and Engineer's Guide to Digital Signal Processing](https://www.dspguide.com/ch8.htm) — 免费、规范的 DSP 教科书。
- [librosa docs — audio primer](https://librosa.org/doc/latest/tutorial.html) — 带代码的实用教程。
- [Heinrich Kuttruff — Room Acoustics (6th ed.)](https://www.routledge.com/Room-Acoustics/Kuttruff/p/book/9781482260434) — 真实世界音频为什么不是干净正弦波的参考。
- [Steve Eddins — FFT Interpretation notebook](https://blogs.mathworks.com/steve/2020/03/30/fft-spectrum-and-spectral-densities/) — 10 分钟内理清频率 bin 直觉。
