# 神经音频编解码器 — EnCodec、SNAC、Mimi、DAC 与语义-声学分离

> 2026 年音频生成几乎全是 token。EnCodec、SNAC、Mimi 和 DAC 将连续波形转换为 transformer 可以预测的离散序列。语义-vs-声学 token 分离——第一个编解码本作为语义，其余作为声学——是自音频 Transformer 以来最重要的架构转变。

**类型：** 学习
**语言：** Python
**先修要求：** 第六阶段 · 02（频谱图），第十阶段 · 11（量化），第五阶段 · 19（子词 Tokenization）
**预计时间：** 约60分钟

## 问题

语言模型在离散 token 上工作。音频是连续的。如果你想要一个用于语音 / 音乐的 LLM 风格模型——MusicGen、Moshi、Sesame CSM、VibeVoice、Orpheus——你首先需要一个**神经音频编解码器**：一个将音频离散化为小词汇量 token 的学习编码器，以及一个重建波形的匹配解码器。

两个族已经出现：

1. **以重建优先的编解码器**——EnCodec、DAC。优化感知音频质量。Token 是"声学的"——它们捕获一切，包括说话人身份、音色、背景噪声。
2. **以语义优先的编解码器**——Mimi（Kyutai）、SpeechTokenizer。强制第一个编解码本编码语言 / 语音内容（通常通过从 WavLM 蒸馏）。随后的编解码本是声学细节。

2024-2026 年的洞察：**纯重建编解码器在你尝试从文本生成时给你模糊的语音。** 编解码 token 上的 LLM 必须在同一个编解码本中同时学习语言结构和声学结构，这无法扩展。将它们分离——语义编解码本 0、声学编解码本 1-N——是 Moshi 和 Sesame CSM 能够工作的原因。

## 概念

![Four codec landscape: EnCodec, DAC, SNAC (multi-scale), Mimi (semantic+acoustic)](../assets/codec-comparison.svg)

### 核心技巧：残差向量量化（RVQ）

不是一个大编解码本（需要数百万个码才能达到良好质量），所有现代音频编解码器都使用 **RVQ**：小型编解码本的级联。第一个编解码本量化编码器输出；第二个量化残差；以此类推。每个编解码本是 1024 个码。8 个编解码本 = 有效词汇量为 1024^8 = 10^24。

在推理时，解码器将每帧所有选定的码求和以重建。

### 2026 年重要的四个编解码器

**EnCodec（Meta，2022）。** 基线。波形上的编码器-解码器，RVQ 瓶颈。24 kHz、最多 32 个编解码本、默认 4 编解码本 @ 1.5 kbps。使用 `1D conv + transformer + 1D conv` 架构。被 MusicGen 使用。

**DAC（Descript，2023）。** 带 L2 归一化编解码本、周期性激活函数、改进的损失函数的 RVQ。任何开放编解码器中最高的重建保真度——使用 12 个编解码本时有时与原始语音无法区分。44.1 kHz 全频带。

**SNAC（Hubert Siuzdak，2024）。** 多尺度 RVQ——粗编解码本以比精细编解码本更低的帧率运行。有效地对音频进行分层建模：约 12 Hz 的粗"草图"加上 50 Hz 的细节。被 Orpheus-3B 使用，因为分层结构很好地映射到基于 LM 的生成。

**Mimi（Kyutai，2024）。** 2026 年改变游戏规则者。12.5 Hz 帧率（极低）、8 个编解码本 @ 4.4 kbps。编解码本 0 是**从 WavLM 蒸馏的**——训练用于预测 WavLM 的语音内容特征。编解码本 1-7 是声学残差。这种分离驱动 Moshi（第 15 课）和 Sesame CSM。

### 帧率对语言建模很重要

更低的帧率 = 更短的序列 = 更快的 LM。

| 编解码器 | 帧率 | 1 s = N 帧 | 适合 |
|-------|-----------|----------------|---------|
| EnCodec-24k | 75 Hz | 75 | 音乐、通用音频 |
| DAC-44.1k | 86 Hz | 86 | 高保真音乐 |
| SNAC-24k（粗） | ~12 Hz | 12 | AR-LM 高效 |
| Mimi | 12.5 Hz | 12.5 | 流式语音 |

在 12.5 Hz 下，一个 10 秒的话语只有 125 个编解码帧——一个 transformer 可以轻松预测它们。

### 语义 vs 声学 token

```
frame_t → [semantic_token_t, acoustic_token_0_t, acoustic_token_1_t, ..., acoustic_token_6_t]
```

- **语义 token（Mimi 中的编解码本 0）。** 编码说了什么——音素、词、内容。通过辅助预测损失从 WavLM 蒸馏。
- **声学 token（编解码本 1-7）。** 编码音色、说话人身份、韵律、背景噪声、细节。

一个 AR LM 首先预测语义 token（以文本为条件），然后预测声学 token（以语义 + 说话人参考为条件）。这种因子分解是现代 TTS 能够零样本克隆声音的原因：语义模型处理内容；声学模型处理音色。

### 2026 年重建质量（每秒比特，更低比特率更好）

| 编解码器 | 比特率 | PESQ | ViSQOL |
|-------|---------|------|--------|
| Opus-20kbps | 20 kbps | 4.0 | 4.3 |
| EnCodec-6kbps | 6 kbps | 3.2 | 3.8 |
| DAC-6kbps | 6 kbps | 3.5 | 4.0 |
| SNAC-3kbps | 3 kbps | 3.3 | 3.8 |
| Mimi-4.4kbps | 4.4 kbps | 3.1 | 3.7 |

传统编解码器如 Opus 在每比特感知质量上仍然胜出。神经编解码器在**离散 token**（Opus 不产生）和**生成模型质量**（LM 可以用那些 token 做什么）上胜出。

## 构建它

### 步骤 1：使用 EnCodec 编码

```python
from encodec import EncodecModel
import torch

model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)  # kbps

wav = torch.randn(1, 1, 24000)
with torch.no_grad():
    encoded = model.encode(wav)
codes, scale = encoded[0]
# codes: (1, n_codebooks, n_frames), dtype=int64
```

在 6 kbps 下 `n_codebooks=8`。每个码是 0-1023（10 位）。

### 步骤 2：解码并测量重建

```python
with torch.no_grad():
    wav_recon = model.decode([(codes, scale)])

from torchaudio.functional import compute_deltas
import torch.nn.functional as F

mse = F.mse_loss(wav_recon[:, :, :wav.shape[-1]], wav).item()
```

### 步骤 3：语义-声学分离（Mimi 风格）

```python
from moshi.models import loaders
mimi = loaders.get_mimi()

with torch.no_grad():
    codes = mimi.encode(wav)  # shape (1, 8, frames@12.5Hz)

semantic = codes[:, 0]
acoustic = codes[:, 1:]
```

语义编解码本 0 与 WavLM 对齐。你可以训练一个文本到语义 transformer——比直接到音频的词汇量小得多。然后一个单独的声学到波形解码器以说话人参考为条件。

### 步骤 4：为什么编解码 token 上的 AR LM 有效

对于 Mimi 的 12.5 Hz × 8 编解码本下的 10 秒语音片段：

```
N_tokens = 10 * 12.5 * 8 = 1000 tokens
```

1000 个 token 对 transformer 来说是微不足道的上下文。一个 256M 参数的 transformer 可以在现代 GPU 上以毫秒为单位生成 10 秒的语音。

## 使用它

问题 → 编解码器映射：

| 任务 | 编解码器 |
|------|-------|
| 通用音乐生成 | EnCodec-24k |
| 最高保真重建 | DAC-44.1k |
| 语音上的 AR LM（TTS） | SNAC 或 Mimi |
| 流式全双工语音 | Mimi（12.5 Hz） |
| 带文本的音效库 | EnCodec + T5 条件 |
| 细粒度音频编辑 | DAC + 修复 |

经验法则：**如果你在构建生成模型，从 Mimi 或 SNAC 开始。如果你在构建压缩流程，使用 Opus。**

## 陷阱

- **太多编解码本。** 添加编解码本线性增加保真度，但也线性增加 LM 序列长度。停在 8-12。
- **帧率不匹配。** 在 12.5 Hz Mimi 上训练 LM 然后在 50 Hz EnCodec 上微调静默失败。
- **假设所有编解码本相同。** 在 Mimi 中，编解码本 0 携带内容；丢失它破坏可懂度。丢失编解码本 7 几乎不可察觉。
- **仅使用重建质量作为指标。** 一个编解码器可以有很棒的重建，但如果语义结构差，对于基于 LM 的生成是无用的。

## 交付它

保存为 `outputs/skill-codec-picker.md`。为给定的生成或压缩任务选择一个编解码器。

## 练习

1. **简单。** 运行 `code/main.py`。它实现了一个玩具标量 + 残差量化器，并测量随着添加编解码本的重建误差。
2. **中等。** 安装 `encodec` 并在保留语音片段上比较 1、4、8、32 个编解码本。绘制 PESQ 或 MSE vs 比特率。
3. **困难。** 加载 Mimi。编码一个片段。用随机整数替换编解码本 0；解码。然后类似地替换编解码本 7。比较两种损坏——编解码本 0 损坏应破坏可懂度；编解码本 7 损坏应几乎没有变化。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| RVQ | 残差量化 | 小型编解码本级联；每个量化前一个的残差。 |
| 帧率 | 编解码器速度 | 每秒多少 token 帧。更低 = 更快的 LM。 |
| 语义编解码本 | 编解码本 0（Mimi） | 从 SSL 特征蒸馏的编解码本；编码内容。 |
| 声学编解码本 | 其他一切 | 音色、韵律、噪声、细节。 |
| PESQ / ViSQOL | 感知质量 | 与 MOS 相关的客观指标。 |
| EnCodec | Meta 编解码器 | RVQ 基线；被 MusicGen 使用。 |
| Mimi | Kyutai 编解码器 | 12.5 Hz 帧率；语义-声学分离；驱动 Moshi。 |

## 扩展阅读

- [Défossez et al. (2023). EnCodec](https://arxiv.org/abs/2210.13438) — RVQ 基线。
- [Kumar et al. (2023). Descript Audio Codec (DAC)](https://arxiv.org/abs/2306.06546) — 最高保真开源。
- [Siuzdak (2024). SNAC](https://arxiv.org/abs/2410.14411) — 多尺度 RVQ。
- [Kyutai (2024). Mimi codec](https://kyutai.org/codec-explainer) — 语义-声学分离、WavLM 蒸馏。
- [Borsos et al. (2023). AudioLM](https://arxiv.org/abs/2209.03143) — 两阶段语义/声学范式。
- [Zeghidour et al. (2021). SoundStream](https://arxiv.org/abs/2107.03312) — 原始可流式 RVQ 编解码器。
