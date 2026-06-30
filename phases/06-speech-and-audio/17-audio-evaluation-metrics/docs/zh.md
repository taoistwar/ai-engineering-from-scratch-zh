# 音频评估 — WER、MOS、UTMOS、MMAU、FAD 与开放排行榜

> 你无法发布你不能测量的东西。本课列出 2026 年每个音频任务的指标：ASR（WER、CER、RTFx）、TTS（MOS、UTMOS、SECS、WER-on-ASR-round-trip）、音频-语言（MMAU、LongAudioBench）、音乐（FAD、CLAP）和说话人（EER）。加上你进行比较的排行榜。

**类型：** 学习
**语言：** Python
**先修要求：** 第六阶段 · 04、06、07、09、10；第二阶段 · 09（模型评估）
**预计时间：** 约60分钟

## 问题

每个音频任务都有多个指标，每个测量不同的轴。使用错误的指标就是你如何发布一个在你仪表板上看起来很好但在生产上很糟糕的模型。2026 年规范列表：

| 任务 | 主要指标 | 次要指标 |
|------|---------|-----------|
| ASR | WER | CER · RTFx · 首个 token 延迟 |
| TTS | MOS / UTMOS | SECS · WER-on-ASR-round-trip · CER · TTFA |
| 声音克隆 | SECS（ECAPA 余弦） | MOS · CER |
| 说话人验证 | EER | minDCF · 在操作点上的 FAR / FRR |
| 说话人分割 | DER | JER · 说话人混淆 |
| 音频分类 | top-1 · mAP | 宏 F1 · 每类召回率 |
| 音乐生成 | FAD | CLAP · 听审 MOS |
| 音频语言模型 | MMAU-Pro | LongAudioBench · AudioCaps FENSE |
| 流式 S2S | 延迟 P50/P95 | WER · MOS |

## 概念

![Audio evaluation matrix — metrics vs tasks vs 2026 leaderboards](../assets/eval-landscape.svg)

### ASR 指标

**WER（词错率）。** `(S + D + I) / N`。评分前小写、去除标点、归一化数字。使用 `jiwer` 或 OpenAI 的 `whisper_normalizer`。< 5% = 人类水平朗读语音。

**CER（字符错误率）。** 相同公式，字符级。用于词分割模糊的声调语言（普通话、粤语）。

**RTFx（逆实时因子）。** 每墙钟秒处理的音频秒数。越高越好。Parakeet-TDT 达到 3380×。Whisper-large-v3 约 30×。

**首个 token 延迟。** 从音频输入到首个转录 token 的墙钟时间。对流式至关重要。Deepgram Nova-3：约 150 毫秒。

### TTS 指标

**MOS（平均意见分）。** 1-5 人类评分。黄金标准但慢。每个样本收集 20+ 听者，每个模型 100+ 样本。

**UTMOS（2022-2026）。** 学到的 MOS 预测器。在标准基准上与人类 MOS 相关性约 0.9。F5-TTS：UTMOS 3.95；地面真实：4.08。

**SECS（说话人编码器余弦相似度）。** 用于声音克隆。参考和克隆输出之间的 ECAPA 嵌入余弦。> 0.75 = 可识别的克隆。

**WER-on-ASR-round-trip。** 在 TTS 输出上运行 Whisper，计算相对于输入文本的 WER。捕获可懂度退化。2026 最先进：< 2% CER。

**TTFA（首个音频时间）。** 墙钟延迟。Kokoro-82M：约 100 毫秒；F5-TTS：约 1 秒。

### 声音克隆特定指标

**SECS + MOS + CER** 作为三重奏。高 SECS 但低 MOS 的克隆意味着音色正确但不自然；反之意味着自然声音但错误的说话人。

### 说话人验证

**EER（等错误率）。** 误接受率等于误拒绝率的阈值。ECAPA 在 VoxCeleb1-O 上：0.87%。

**minDCF（最小检测成本）。** 在选定操作点（通常 FAR=0.01）的加权成本。比 EER 更生产相关。

### 说话人分割

**DER（分割错误率）。** `(FA + Miss + Confusion) / total_speaker_time`。遗漏语音 + 误报语音 + 说话人混淆，每一项作为总说话时间的比例。AMI 会议：DER 约 10-20% 是现实的。pyannote 3.1 + Precision-2 商业：在良好录制的音频上 <10% DER。

**JER（Jaccard 错误率）。** DER 的替代方案，对短片段偏置鲁棒。

### 音频分类

多标签：所有类上的 **mAP（平均精度均值）**。AudioSet：BEATs-iter3 为 0.548 mAP。

互斥多类：**top-1、top-5 准确率**。Speech Commands v2：99.0% top-1（Audio-MAE）。

不均衡：**宏 F1** + **每类召回率**。按类别报告——聚合准确率隐藏哪些类别失败。

### 音乐生成

**FAD（Fréchet 音频距离）。** 真实与生成音频的 VGGish 嵌入分布之间的距离。MusicGen-small 在 MusicCaps 上：4.5。MusicLM：4.0。更低更好。

**CLAP 分数。** 使用 CLAP 嵌入的文本-音频对齐分数。> 0.3 = 合理对齐。

**听审 MOS。** 对于消费级音乐仍然是最终结论。Suno v5 在 TTS Arena 上 ELO 1293（来自配对人类偏好）。

### 音频-语言基准

**MMAU（大规模多音频理解）。** 10k 音频-QA 对。

**MMAU-Pro。** 1800 个困难项目，四个类别：语音 / 声音 / 音乐 / 多音频。4 选 1 随机水平 25%。Gemini 2.5 Pro 总体约 60%；所有模型多音频约 22%。

**LongAudioBench。** 带语义查询的多分钟片段。Audio Flamingo Next 击败 Gemini 2.5 Pro。

**AudioCaps / Clotho。** 字幕基准。SPICE、CIDEr、FENSE 指标。

### 流式语音到语音

**延迟 P50 / P95 / P99。** 从用户语音结束到首个可听响应的墙钟时间。Moshi：200 毫秒；GPT-4o Realtime：300 毫秒。

**输出上的 WER / MOS。**

**插话响应性。** 从用户打断到助手静音的时间。目标 < 150 毫秒。

### 2026 年排行榜

| 排行榜 | 赛道 | URL |
|------------|--------|-----|
| Open ASR Leaderboard（HF） | 英语 + 多语言 + 长格式 | `huggingface.co/spaces/hf-audio/open_asr_leaderboard` |
| TTS Arena（HF） | 英语 TTS | `huggingface.co/spaces/TTS-AGI/TTS-Arena` |
| Artificial Analysis Speech | TTS + STT，来自配对投票的 ELO | `artificialanalysis.ai/speech` |
| MMAU-Pro | LALM 推理 | `mmaubenchmark.github.io` |
| SpeakerBench / VoxSRC | 说话人识别 | `voxsrc.github.io` |
| MMAU music subset | 音乐 LALM | （在 MMAU 内） |
| HEAR benchmark | 自监督音频 | `hearbenchmark.com` |

## 构建它

### 步骤 1：带归一化的 WER

```python
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, Strip

transform = Compose([ToLowerCase(), RemovePunctuation(), Strip()])
score = wer(
    truth="Please turn on the lights.",
    hypothesis="please turn on the light",
    truth_transform=transform,
    hypothesis_transform=transform,
)
# ~0.17
```

### 步骤 2：TTS 往返 WER

```python
def ttr_wer(tts_model, asr_model, texts):
    errors = []
    for txt in texts:
        audio = tts_model.synthesize(txt)
        recog = asr_model.transcribe(audio)
        errors.append(wer(truth=txt, hypothesis=recog))
    return sum(errors) / len(errors)
```

### 步骤 3：声音克隆的 SECS

```python
from speechbrain.inference.speaker import EncoderClassifier
sv = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb")

emb_ref = sv.encode_batch(load_wav("reference.wav"))
emb_clone = sv.encode_batch(load_wav("cloned.wav"))
secs = torch.nn.functional.cosine_similarity(emb_ref, emb_clone, dim=-1).item()
```

### 步骤 4：音乐生成的 FAD

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()
score = fad.get_fad_score("generated_folder/", "reference_folder/")
```

### 步骤 5：说话人验证的 EER（与第 6 课相同代码）

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        frr = sum(1 for s in same_scores if s < t) / len(same_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

## 使用它

为每次部署配对在每个模型更新上运行的固定评估工具。三条基本规则：

1. **评分前归一化。** 小写、标点去除、数字展开。报告归一化规则。
2. **报告分布，而不是平均值。** 延迟用 P50/P95/P99。分类用每类召回率。MMAU 用每类别。
3. **运行一个规范公开基准。** 即使你的生产数据不同，在 Open ASR / TTS Arena / MMAU 上报告可以让审查者进行苹果对苹果的比较。

## 陷阱

- **UTMOS 外推。** 在 VCTK 风格干净语音上训练；对嘈杂 / 克隆 / 情感音频评分很差。
- **MOS 面板偏置。** 20 个 Amazon Mechanical Turk 工人 ≠ 20 个目标用户。如果风险高，为领域面板付费。
- **FAD 依赖于参考集。** 跨模型与相同参考分布比较。
- **聚合 WER。** 总体 5% WER 可以隐藏口音语音上的 30% WER。按人口统计学切片报告。
- **公开基准饱和。** 大多数前沿模型在标准基准上接近上限。构建反映你流量的内部保留集。

## 交付它

保存为 `outputs/skill-audio-evaluator.md`。为任何音频模型发布选择指标、基准和报告格式。

## 练习

1. **简单。** 运行 `code/main.py`。在玩具输入上计算 WER / CER / EER / SECS / 类似 FAD / 类似 MMAU。
2. **中等。** 构建 TTS 往返 WER 工具。通过 Whisper 运行你的 Kokoro 或 F5-TTS 输出。在 50 个提示上计算 WER。标记 WER > 10% 的提示。
3. **困难。** 在 MMAU-Pro 语音 + 多音频子集（各 50 项）上为你的第 10 课 LALM 选择评分。报告每类别准确率并与发布数字比较。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| WER | ASR 分数 | 归一化后词级的 `(S+D+I)/N`。 |
| CER | 字符 WER | 用于声调语言或字符级系统。 |
| MOS | 人类意见 | 1-5 评分；20+ 听者 × 100 样本。 |
| UTMOS | ML MOS 预测器 | 学到的模型；与人类 MOS 相关性约 0.9。 |
| SECS | 语音克隆相似度 | 参考和克隆之间的 ECAPA 余弦。 |
| EER | 说话人验证分数 | FAR = FRR 的阈值。 |
| DER | 说话人分割分数 | (FA + Miss + Confusion) / total。 |
| FAD | 音乐生成质量 | 在 VGGish 嵌入上的 Fréchet 距离。 |
| RTFx | 吞吐量 | 每墙钟秒的音频秒数。 |

## 扩展阅读

- [jiwer](https://github.com/jitsi/jiwer) — 带归一化工具的 WER/CER 库。
- [UTMOS (Saeki et al. 2022)](https://arxiv.org/abs/2204.02152) — 学到的 MOS 预测器。
- [Fréchet Audio Distance (Kilgour et al. 2019)](https://arxiv.org/abs/1812.08466) — 音乐生成标准。
- [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — 2026 实时排名。
- [TTS Arena](https://huggingface.co/spaces/TTS-AGI/TTS-Arena) — 人工投票 TTS 排行榜。
- [MMAU-Pro benchmark](https://mmaubenchmark.github.io/) — LALM 推理排行榜。
- [HEAR benchmark](https://hearbenchmark.com/) — 音频 SSL 基准。
