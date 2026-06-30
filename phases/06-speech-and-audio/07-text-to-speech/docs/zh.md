# 文本转语音（TTS） — 从 Tacotron 到 F5 和 Kokoro

> ASR 将语音反转成文本；TTS 将文本反转成语音。2026 年技术栈分三部分：文本 → token、token → Mel、Mel → 波形。每部分有一个适合笔记本的默认模型。

**类型：** 构建
**语言：** Python
**先修要求：** 第六阶段 · 02（频谱图与 Mel），第五阶段 · 09（Seq2Seq），第七阶段 · 05（完整 Transformer）
**预计时间：** 约75分钟

## 问题

你有一个字符串："Please remind me to water the plants at 6 pm."你需要一个 3 秒的音频片段，听起来自然，有正确的韵律（停顿、重音），用正确的元音发音"plants"，并在 CPU 上以低于 300 毫秒的速度运行以用于实时语音助手。你还需要替换声音、处理代码切换的输入（"remind me at 6 pm, daijoubu?"），并且不要在名字上出丑。

现代 TTS 流程看起来像这样：

1. **文本前端。** 归一化文本（日期、数字、电子邮件）、转换为音素或子词 token、预测韵律特征。
2. **声学模型。** 文本 → Mel 频谱图。Tacotron 2（2017）、FastSpeech 2（2020）、VITS（2021）、F5-TTS（2024）、Kokoro（2024）。
3. **Vocoder。** Mel → 波形。WaveNet（2016）、WaveRNN、HiFi-GAN（2020）、BigVGAN（2022）、2024+ 神经编解码 vocoder。

在 2026 年，声学 + vocoder 的分割随着端到端扩散和流匹配模型变得模糊。但三部分的心智模型对于调试仍然成立。

## 概念

![Tacotron, FastSpeech, VITS, F5/Kokoro side-by-side](../assets/tts.svg)

**Tacotron 2（2017）。** Seq2seq：字符嵌入 → BiLSTM 编码器 → 位置敏感注意力 → 自回归 LSTM 解码器发出 Mel 帧。慢（AR），长文本上不稳定。仍被引用为基线。

**FastSpeech 2（2020）。** 非自回归。持续时间预测器输出每个音素获得多少 Mel 帧。单次传递，比 Tacotron 快 10 倍。丢失一些自然度（单调对齐）但在到处发布。

**VITS（2021）。** 联合训练编码器 + 基于流的持续时间 + HiFi-GAN vocoder 端到端，使用变分推理。高质量，单一模型。2022–2024 主导的开源 TTS。变体：YourTTS（多说话人零样本）、XTTS v2（2024、Coqui）。

**F5-TTS（2024）。** 基于流匹配的扩散 transformer。自然韵律、使用 5 秒参考音频的零样本声音克隆。2026 年开源 TTS 排行榜的顶部。335M 参数。

**Kokoro（2024）。** 小（82M）、CPU 可运行、最佳实时英语 TTS。封闭词汇仅英语、apache-2.0。

**OpenAI TTS-1-HD、ElevenLabs v2.5、Google Chirp-3。** 商业最先进。ElevenLabs v2.5 情感标签（"[whispered]"、"[laughing]"）和角色声音在 2026 年主导有声读物制作。

### Vocoder 演进

| 时代 | Vocoder | 延迟 | 质量 |
|-----|---------|---------|---------|
| 2016 | WaveNet | 仅离线 | 发布时最先进 |
| 2018 | WaveRNN | 约实时 | 良好 |
| 2020 | HiFi-GAN | 100× 实时 | 接近人类 |
| 2022 | BigVGAN | 50× 实时 | 跨说话人/语言泛化 |
| 2024 | SNAC、DAC（神经编解码器） | 与 AR 模型集成 | 离散 token、比特高效 |

到 2026 年，大多数"TTS"模型从文本到波形是端到端的；Mel 频谱图是内部表示。

### 评估

- **MOS（平均意见分）。** 1–5 分制，众包。仍然是黄金标准；痛苦地慢。
- **CMOS（比较 MOS）。** A-vs-B 偏好。每个标注有更紧密的置信区间。
- **UTMOS、DNSMOS。** 无参考神经 MOS 预测器。用于排行榜。
- **通过 ASR 的 CER（字符错误率）。** 通过 Whisper 运行 TTS 输出，计算对输入文本的 CER。可懂度的代理。
- **SECS（说话人嵌入余弦相似度）。** 声音克隆质量。

2026 年 LibriTTS test-clean 上的数字：

| 模型 | UTMOS | CER（通过 Whisper） | 大小 |
|-------|-------|-------------------|------|
| 地面真实 | 4.08 | 1.2% | — |
| F5-TTS | 3.95 | 2.1% | 335M |
| XTTS v2 | 3.81 | 3.5% | 470M |
| VITS | 3.62 | 3.1% | 25M |
| Kokoro v0.19 | 3.87 | 1.8% | 82M |
| Parler-TTS Large | 3.76 | 2.8% | 2.3B |

## 构建它

### 步骤 1：将输入音素化

```python
from phonemizer import phonemize
ph = phonemize("Hello world", language="en-us", backend="espeak")
# 'həloʊ wɜːld'
```

音素是通用桥梁。避免将原始文本喂给低于 VITS 级别质量的任何东西。

### 步骤 2：运行 Kokoro（2026 CPU 默认）

```python
from kokoro import KPipeline
tts = KPipeline(lang_code="a")  # "a" = 美式英语
audio, sr = tts("Please remind me to water the plants at 6 pm.", voice="af_bella")
# audio: float32 tensor, sr=24000
```

离线运行，单文件，82M 参数。

### 步骤 3：使用声音克隆运行 F5-TTS

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="my_voice_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please remind me to water the plants.",
)
```

传递一个 5 秒参考片段 + 其转录；F5 克隆韵律和音色。

### 步骤 4：从零实现 HiFi-GAN vocoder

太大不适合放在教程脚本中，但形态是：

```python
class HiFiGAN(nn.Module):
    def __init__(self, mel_channels=80, upsample_rates=[8, 8, 2, 2]):
        super().__init__()
        # 4 个上采样块，总共 256 倍从 Mel 采样率到音频采样率
        ...
    def forward(self, mel):
        return self.blocks(mel)  # -> waveform
```

训练：对抗性（短窗口上的判别器）+ Mel 频谱图重建损失 + 特征匹配损失。商品化——使用来自 `hifi-gan` 仓库或 nvidia-NeMo 的预训练检查点。

### 步骤 5：完整流程（伪代码）

```python
text = "Please remind me at 6 pm."
phones = phonemize(text)
mel = acoustic_model(phones, speaker=alice)      # [T, 80]
wav = vocoder(mel)                                # [T * 256]
soundfile.write("out.wav", wav, 24000)
```

## 使用它

2026 年技术栈：

| 场景 | 选择 |
|-----------|------|
| 实时英语语音助手 | Kokoro（CPU）或 XTTS v2（GPU） |
| 从 5 秒参考进行声音克隆 | F5-TTS |
| 商业角色声音 | ElevenLabs v2.5 |
| 有声读物旁白 | ElevenLabs v2.5 或 XTTS v2 + 微调 |
| 低资源语言 | 在 5–20 小时目标语言数据上训练 VITS |
| 表现力 / 情感标签 | ElevenLabs v2.5 或 StyleTTS 2 微调 |

截至 2026 年开源领导者：**质量用 F5-TTS，效率用 Kokoro**。除非你是历史学家，不要选择 Tacotron。

## 陷阱

- **没有文本归一化器。** "Dr. Smith"读作"Doctor"还是"Drive"？"2026"是"twenty twenty six "还是"two zero two six"？在音素化器之前归一化。
- **OOV 专有名词。** "Ghumare" → "ghyu-mair"？为未知 token 部署后备字素到音素模型。
- **削波。** Vocoder 输出很少削波，但推理时的 Mel 缩放不匹配可能超过 ±1.0。始终 `np.clip(wav, -1, 1)`。
- **采样率不匹配。** Kokoro 输出 24 kHz；你的下游流程期望 16 kHz → 重采样否则混叠。

## 交付它

保存为 `outputs/skill-tts-designer.md`。为给定的声音、延迟和语言目标设计 TTS 流程。

## 练习

1. **简单。** 运行 `code/main.py`。从玩具词汇构建音素字典，估计每个音素的持续时间，并打印一个假的"Mel"计划。
2. **中等。** 安装 Kokoro，以声音 `af_bella` 和 `am_adam` 合成相同句子。比较音频时长和主观质量。
3. **困难。** 录制你自己的 5 秒参考片段。使用 F5-TTS 克隆它。报告参考和克隆输出之间的 SECS。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| 音素 | 声音单元 | 抽象声音类别；英语有 39 个（ARPABet）。 |
| 持续时间预测器 | 每个音素持续多久 | 非自回归模型输出；每个音素的整数帧。 |
| Vocoder | Mel → 波形 | 将 Mel 谱映射到原始样本的神经网络。 |
| HiFi-GAN | 标准 vocoder | 基于 GAN；2020–2024 主导。 |
| MOS | 主观质量 | 来自人类评分者的 1–5 平均意见分。 |
| SECS | 声音克隆指标 | 目标和输出说话人嵌入之间的余弦相似度。 |
| F5-TTS | 2024 开源最先进 | 流匹配扩散；零样本克隆。 |
| Kokoro | CPU 英语领袖 | 82M 参数模型、Apache 2.0。 |

## 扩展阅读

- [Shen et al. (2017). Tacotron 2](https://arxiv.org/abs/1712.05884) — seq2seq 基线。
- [Kim, Kong, Son (2021). VITS](https://arxiv.org/abs/2106.06103) — 基于流的端到端。
- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — 当前开源最先进。
- [Kong, Kim, Bae (2020). HiFi-GAN](https://arxiv.org/abs/2010.05646) — 2026 年仍在发布的 vocoder。
- [Kokoro-82M on HuggingFace](https://huggingface.co/hexgrad/Kokoro-82M) — 2024 CPU 友好的英语 TTS。
