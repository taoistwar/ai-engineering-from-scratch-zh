# 音乐生成 — MusicGen、Stable Audio、Suno 与许可地震

> 2026 年音乐生成：Suno v5 和 Udio v4 主导商业；MusicGen、Stable Audio Open 和 ACE-Step 引领开源。技术问题基本解决。法律问题（Warner Music 5 亿美元和解、UMG 和解）在 2025-2026 年重塑了该领域。

**类型：** 构建
**语言：** Python
**先修要求：** 第六阶段 · 02（频谱图），第四阶段 · 10（扩散模型）
**预计时间：** 约75分钟

## 问题

文本 → 一段 30 秒到 4 分钟的音乐片段，带歌词、人声和结构。三个子问题：

1. **器乐生成。** 文本如"lo-fi hip-hop drums with warm keys" → 音频。MusicGen、Stable Audio、AudioLDM。
2. **歌曲生成（带人声 + 歌词）。** "Country song about rainy Texas nights" → 完整歌曲。Suno、Udio、YuE、ACE-Step。
3. **条件/可控。** 扩展现有片段、重新生成桥段、切换风格、分轨分离或修复。Udio 的修复 + 分轨分离是 2026 年的对标功能。

## 概念

![Music generation: token-LM vs diffusion, the 2026 model map](../assets/music-generation.svg)

### 在神经编解码 token 上的 Token LM

Meta 的 **MusicGen**（2023, MIT）及许多衍生物：以文本/旋律嵌入为条件，自回归预测 EnCodec token（32 kHz、4 编解码本），用 EnCodec 解码。300M - 3.3B 参数。强基线；超过 30 秒挣扎。

**ACE-Step**（开源，4B XL 版 2026 年 4 月发布）扩展了用于全曲歌词条件生成。开源社区最接近 Suno 的东西。

### 在 Mel 或潜在空间上的扩散

**Stable Audio（2023）** 和 **Stable Audio Open（2024）**：在压缩音频上的潜在扩散。擅长循环、声音设计、氛围纹理。在结构化完整歌曲上不太好。

**AudioLDM / AudioLDM2**：通过 T2I 风格的潜在扩散进行文本转音频，泛化到音乐、音效、语音。

### 混合（生产）—— Suno、Udio、Lyria

封闭权重。可能是 AR 编解码 LM + 基于扩散的 vocoder，带有专门的语音 / 鼓 / 旋律头。Suno v5（2026）是 ELO 1293 质量领袖。Udio v4 添加了修复 + 分轨分离（贝斯、鼓、人声单独下载）。

### 评估

- **FAD（Fréchet 音频距离）。** 使用 VGGish 或 PANNs 特征的生成与真实音频分布之间的嵌入级距离。越低越好。MusicGen small：MusicCaps 上 4.5 FAD；最先进约 3.0。
- **音乐性（主观）。** 人类偏好。Suno v5 以 ELO 1293 领先。
- **文本-音频对齐。** 提示和输出之间的 CLAP 分数。
- **音乐性伪像。** 脱拍过渡、人声短语漂移、30 秒后结构丢失。

## 2026 年模型地图

| 模型 | 参数 | 长度 | 人声 | 许可 |
|-------|--------|--------|--------|---------|
| MusicGen-large | 3.3B | 30 s | 否 | MIT |
| Stable Audio Open | 1.2B | 47 s | 否 | Stability 非商业 |
| ACE-Step XL（2026 年 4 月）| 4B | > 2 min | 是 | Apache-2.0 |
| YuE | 7B | > 2 min | 是，多语言 | Apache-2.0 |
| Suno v5（封闭） | ? | 4 min | 是，ELO 1293 | 商业 |
| Udio v4（封闭） | ? | 4 min | 是 + 分轨 | 商业 |
| Google Lyria 3（封闭） | ? | 实时 | 是 | 商业 |
| MiniMax Music 2.5 | ? | 4 min | 是 | 商业 API |

## 法律景观（2025-2026）

- **Warner Music vs Suno 和解。** 5 亿美元。WMG 现在对 Suno 上的 AI 形象、音乐权利和用户生成曲目有监督权。类似的 UMG 在 Udio 上的和解。
- **EU AI 法案** + **加利福尼亚 SB 942**：AI 生成的音乐必须被披露。
- **Riffusion / MusicGen** 在 MIT 下没有合规负担，但也没有商业人声。

安全发布的模式：

1. 仅生成器乐（MusicGen、Stable Audio Open、MIT/CC0 输出）。
2. 使用带逐次生成许可的商业 API（Suno、Udio、ElevenLabs Music）。
3. 在自有或授权目录上训练（大多数企业最终选择此路径）。
4. 用水印 + 元数据标记生成物。

## 构建它

### 步骤 1：使用 MusicGen 生成

```python
from audiocraft.models import MusicGen
import torchaudio

model = MusicGen.get_pretrained("facebook/musicgen-small")
model.set_generation_params(duration=10)
wav = model.generate(["upbeat synthwave with driving drums, 128 BPM"])
torchaudio.save("out.wav", wav[0].cpu(), 32000)
```

三种大小：`small`（300M，快速）、`medium`（1.5B）、`large`（3.3B）。Small 足以用于"想法是否落地"。

### 步骤 2：旋律条件化

```python
melody, sr = torchaudio.load("humming.wav")
wav = model.generate_with_chroma(
    ["jazz piano cover"],
    melody.squeeze(),
    sr,
)
```

MusicGen-melody 取一个色度图，在交换音色的同时保留曲调。用于"将这个旋律变成弦乐四重奏"。

### 步骤 3：FAD 评估

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()

fad.get_fad_score("generated_folder/", "reference_folder/")
```

计算 VGGish 嵌入距离。用于流派级别的回归测试；不替代人类听者。

### 步骤 4：添加到 LLM-音乐工作流程

结合第 7-8 课的思想：

```python
prompt = "Write a 30-second jazz loop. Describe the drums, bass, and piano voicing."
description = llm.complete(prompt)
music = musicgen.generate([description], duration=30)
```

## 使用它

| 目标 | 技术栈 |
|------|-------|
| 器乐声音设计 | Stable Audio Open |
| 游戏 / 自适应音乐 | Google Lyria RealTime（封闭） |
| 带人声的完整歌曲（商业） | Suno v5 或 Udio v4 带显式许可 |
| 带人声的完整歌曲（开源） | ACE-Step XL 或 YuE |
| 短广告歌曲 | 以哼哼参考为条件的 MusicGen melody-conditioned |
| 音乐视频背景 | MusicGen + Stable Video Diffusion |

## 2026 年仍未被捕获就发布的陷阱

- **版权洗钱提示。** "Song in the style of Taylor Swift"——商业 Suno/Udio 现在过滤这些，开源模型不会。添加你自己的过滤列表。
- **30 秒后的重复 / 漂移。** AR 模型循环。交叉渐变多个生成，或使用 ACE-Step 获得结构连贯性。
- **速度漂移。** 模型游离 BPM。在提示中使用 BPM 标签并用 librosa 的 `beat_track` 进行后过滤。
- **人声清晰度。** Suno 出色；开源模型在词上通常含糊。如果歌词重要，使用商业 API 或微调。
- **单声道输出。** 开源模型生成单声道或假立体声。用适当的立体声重建升级（ezst、Cartesia 的立体声扩散）。

## 交付它

保存为 `outputs/skill-music-designer.md`。为音乐生成部署选择模型、许可策略、长度 / 结构计划和披露元数据。

## 练习

1. **简单。** 运行 `code/main.py`。它生成一个"生成式"和弦进行 + 鼓模式作为 ASCII 符号——一个音乐生成卡通。如果需要，通过任何 MIDI 渲染器播放。
2. **中等。** 安装 `audiocraft`，用 MusicGen-small 跨 4 种流派提示生成 10 秒片段，测量相对于参考流派集的 FAD。
3. **困难。** 使用 ACE-Step（或 MusicGen-melody），使用不同音色提示生成相同曲调的三种变体。计算与提示的 CLAP 相似度以验证对齐。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| FAD | 音频 FID | 真实与生成嵌入分布之间的 Fréchet 距离。 |
| 色度图 | 作为音高的旋律 | 每帧 12 维向量；旋律条件化的输入。 |
| 分轨 | 器乐音轨 | 分离的贝斯 / 鼓 / 人声 / 旋律为 WAV。 |
| 修复 | 重新生成一段 | 掩蔽时间窗口；模型仅重新生成该部分。 |
| CLAP | 文本-音频 CLIP | 对比式音频-文本嵌入；评估文本-音频对齐。 |
| EnCodec | 音乐编解码器 | Meta 的神经编解码器，被 MusicGen 使用；32 kHz、4 编解码本。 |

## 扩展阅读

- [Copet et al. (2023). MusicGen](https://arxiv.org/abs/2306.05284) — 开源自回归基准。
- [Evans et al. (2024). Stable Audio Open](https://arxiv.org/abs/2407.14358) — 声音设计默认选择。
- [ACE-Step](https://github.com/ace-step/ACE-Step) — 开源 4B 全曲生成器，2026 年 4 月。
- [Suno v5 platform docs](https://suno.com) — 商业质量领袖。
- [AudioLDM2](https://arxiv.org/abs/2308.05734) — 音乐 + 音效的潜在扩散。
- [WMG-Suno settlement coverage](https://www.musicbusinessworldwide.com/suno-warner-music-settlement/) — 2025 年 11 月先例。
