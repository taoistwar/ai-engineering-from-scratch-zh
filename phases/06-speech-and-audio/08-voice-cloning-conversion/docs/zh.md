# 声音克隆与语音转换

> 声音克隆用别人的声音朗读你的文本。语音转换保留你所说的内容，将你的声音重写成别人的声音。两者都依赖于相同的分解：将说话人身份与内容分离。

**类型：** 构建
**语言：** Python
**先修要求：** 第六阶段 · 06（说话人识别），第六阶段 · 07（TTS）
**预计时间：** 约75分钟

## 问题

在 2026 年，一个 5 秒的音频片段足以在消费级 GPU 上产生任何人的高质量克隆。ElevenLabs、F5-TTS、OpenVoice v2、VoiceBox 都提供零样本或少样本克隆。该技术是祝福（无障碍 TTS、配音、辅助声音）也是武器（诈骗电话、政治深度伪造、知识产权盗窃）。

两个密切相关的任务：

- **声音克隆（TTS 侧）：** 文本 + 5 秒参考声音 → 该声音的音频。
- **语音转换（语音侧）：** 源音频（人物 A 说 X）+ 人物 B 的参考声音 → 人物 B 说 X 的音频。

两者将波形分解为（内容、说话人、韵律），并将来自一个源的内容与另一个的说话人重新组合。

你现在在 2026 年发布时的关键约束：**水印和同意门禁在欧盟（AI 法案，2026 年 8 月强制执行）和加利福尼亚（AB 2905，2025 年生效）是法律要求的**。你的流程必须发出听不见的水印并拒绝非经同意的克隆。

## 概念

![Voice cloning vs conversion: factorize, swap speaker, recombine](../assets/voice-cloning.svg)

**零样本克隆。** 将一个 5 秒片段传递给已在数千名说话人上训练的模型。说话人编码器将片段映射为说话人嵌入；TTS 解码器以该嵌入加文本为条件。

使用者：F5-TTS（2024）、YourTTS（2022）、XTTS v2（2024）、OpenVoice v2（2024）。

**少样本微调。** 录制 5-30 分钟目标声音。对基础模型进行 LoRA 微调一小时。质量从"还行"跃升到"无法区分"。Coqui 和 ElevenLabs 都支持此模式；社区将其用于 F5-TTS。

**语音转换（VC）。** 两个族：

- **识别-合成。** 运行类似 ASR 的模型提取内容表示（例如，软音素后验概率、PPG），然后用目标说话人嵌入重新合成。对语言和口音鲁棒。被 KNN-VC（2023）、Diff-HierVC（2023）使用。
- **解耦。** 训练一个自动编码器，在瓶颈处将内容、说话人和韵律在潜在空间中分离。在推理时交换说话人嵌入。质量更低但更快。被 AutoVC（2019）、VITS-VC 变体使用。

**基于神经编解码的克隆（2024+）。** VALL-E、VALL-E 2、NaturalSpeech 3、VoiceBox——将音频视为来自 SoundStream / EnCodec 的离散 token，在编解码 token 上训练大型自回归或流匹配模型。短提示上质量与 ElevenLabs 相当。

### 伦理部分，不是附加件

**水印。** PerTh（Perth）和 SilentCipher（2024）在音频中不可感知地嵌入约 16-32 位 ID。能在重新编码、流式和常见编辑中存活。生产就绪的开源。

**同意门禁。** 必须将每个克隆输出与可验证的同意记录配对。"我，Rohit，于 2026-04-22，授权此声音用于 X 目的。"存储在防篡改日志中。

**检测。** AASIST、RawNet2 和 Wav2Vec2-AASIST 作为检测器发布。ASVspoof 2025 挑战赛发布了对 ElevenLabs、VALL-E 2 和 Bark 输出的最先进检测器的 EER 为 0.8–2.3%。

### 数字（2026）

| 模型 | 零样本？ | SECS（目标相似度） | WER（可懂度） | 参数 |
|-------|-----------|--------------------|--------------|--------|
| F5-TTS | 是 | 0.72 | 2.1% | 335M |
| XTTS v2 | 是 | 0.65 | 3.5% | 470M |
| OpenVoice v2 | 是 | 0.70 | 2.8% | 220M |
| VALL-E 2 | 是 | 0.77 | 2.4% | 370M |
| VoiceBox | 是 | 0.78 | 2.1% | 330M |

SECS > 0.70 对大多数听者来说通常与目标无法区分。

## 构建它

### 步骤 1：使用识别-合成进行分解（仅限 main.py 中的代码演示）

```python
def clone_pipeline(ref_audio, text, target_embedder, tts_model):
    speaker_emb = target_embedder.encode(ref_audio)
    mel = tts_model(text, speaker=speaker_emb)
    return vocoder(mel)
```

概念上简单；实现量在 `tts_model` 和说话人编码器中。

### 步骤 2：使用 F5-TTS 进行零样本克隆

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="rohit_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please add milk and bread to my list.",
)
```

参考转录必须与音频精确匹配；不匹配会破坏对齐。

### 步骤 3：使用 KNN-VC 进行语音转换

```python
import torch
from knnvc import KNNVC  # 2023 模型, https://github.com/bshall/knn-vc
vc = KNNVC.load("wavlm-base-plus")
out_wav = vc.convert(source="my_voice.wav", target_pool=["alice_1.wav", "alice_2.wav"])
```

KNN-VC 运行 WavLM 为源和目标池提取每帧嵌入，然后将每个源帧替换为池中最近的邻居。非参数化，分钟级目标语音即可工作。

### 步骤 4：嵌入水印

```python
from silentcipher import SilentCipher
sc = SilentCipher(model="2024-06-01")
payload = b"consent_id:abc123;ts:1745353200"
watermarked = sc.embed(wav, sr=24000, message=payload)
detected = sc.detect(watermarked, sr=24000)   # 返回有效载荷字节
```

约 32 位有效载荷，在 MP3 重新编码和轻度噪声后可检测。

### 步骤 5：同意门禁

```python
def cloned_inference(text, ref_audio, consent_record):
    assert verify_signature(consent_record), "需要签名同意"
    assert consent_record["speaker_id"] == hash_speaker(ref_audio)
    wav = tts.infer(ref_file=ref_audio, gen_text=text)
    wav = watermark(wav, payload=consent_record["id"])
    return wav
```

## 使用它

2026 年技术栈：

| 场景 | 选择 |
|-----------|------|
| 5 秒零样本克隆、开源 | F5-TTS 或 OpenVoice v2 |
| 商业生产克隆 | ElevenLabs Instant Voice Clone v2.5 |
| 语音转换（重写） | KNN-VC 或 Diff-HierVC |
| 多说话人微调 | StyleTTS 2 + 说话人适配器 |
| 跨语言克隆 | XTTS v2 或 VALL-E X |
| 深度伪造检测 | Wav2Vec2-AASIST |

## 陷阱

- **不匹配的参考转录。** F5-TTS 等需要参考文本与参考音频精确匹配，包括标点。
- **混响参考。** 回声杀死克隆。干声录制、近距离麦克风。
- **情感不匹配。** 训练参考"欢快"会对一切产生欢快的克隆。将参考情感与目标用途匹配。
- **语言泄漏。** 克隆英语说话人然后要求模型说法语，通常仍然带着口音；使用跨语言模型（XTTS、VALL-E X）。
- **无水印。** 从 2026 年 8 月起在欧盟法律上不可发布。

## 交付它

保存为 `outputs/skill-voice-cloner.md`。设计带同意门禁 + 水印 + 质量目标的克隆或转换流程。

## 练习

1. **简单。** 运行 `code/main.py`。通过计算交换前后两个"说话人"之间的余弦来演示说话人嵌入交换。
2. **中等。** 使用 OpenVoice v2 克隆你自己的声音。测量参考与克隆之间的 SECS。通过 Whisper 测量 CER。
3. **困难。** 将 SilentCipher 水印应用于 20 个克隆，通过 128 kbps MP3 编码+解码运行它们，检测有效载荷。报告比特准确率。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| 零样本克隆 | 5 秒就够了 | 预训练模型 + 说话人嵌入；无需训练。 |
| PPG | 音素后验概率图 | 用作语言无关内容表示的每帧 ASR 后验概率。 |
| KNN-VC | 最近邻转换 | 将每个源帧替换为最近的目标池帧。 |
| 神经编解码 TTS | VALL-E 风格 | 在 EnCodec/SoundStream token 上的自回归模型。 |
| 水印 | 听不见的签名 | 嵌入音频中的比特，能在重新编码后存活。 |
| SECS | 克隆保真度 | 目标和克隆说话人嵌入之间的余弦。 |
| AASIST | 深度伪造检测器 | 反欺骗模型；检测合成语音。 |

## 扩展阅读

- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — 开源最先进零样本克隆。
- [Baevski et al. / Microsoft (2023). VALL-E](https://arxiv.org/abs/2301.02111) 和 [VALL-E 2 (2024)](https://arxiv.org/abs/2406.05370) — 神经编解码 TTS。
- [Qian et al. (2019). AutoVC](https://arxiv.org/abs/1905.05879) — 基于解耦的语音转换。
- [Baas, Waubert de Puiseau, Kamper (2023). KNN-VC](https://arxiv.org/abs/2305.18975) — 基于检索的 VC。
- [SilentCipher (2024) — Audio Watermarking](https://github.com/sony/silentcipher) — 生产就绪的 32 位音频水印。
- [ASVspoof 2025 results](https://www.asvspoof.org/) — 检测器 vs 合成器军备竞赛，2026 年更新。
