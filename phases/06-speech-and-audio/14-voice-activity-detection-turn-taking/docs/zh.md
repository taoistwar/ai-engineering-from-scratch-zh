# 语音活动检测与轮次交替 — Silero、Cobra 与同花顺技巧

> 每个语音代理都取决于两个决定：用户现在在说话吗，他们说完了吗？VAD 回答第一个。轮次检测（VAD + 静音挂起 + 语义终点模型）回答第二个。搞错任何一个，你的助手要么打断用户要么永远不闭嘴。

**类型：** 构建
**语言：** Python
**先修要求：** 第六阶段 · 11（实时音频），第六阶段 · 12（语音助手）
**预计时间：** 约45分钟

## 问题

语音代理在每 20 毫秒块上做出的三个不同决定：

1. **这个帧是语音吗？** — VAD。二值、每帧。
2. **用户已经开始新话语了吗？** — 起始检测。
3. **用户已经说完了吗？** — 终点检测（轮次结束）。

朴素的答案（能量阈值）在任何噪声上都会失败——交通、键盘、人群嘈杂声。2026 年的答案：Silero VAD（开源、深度学习）+ 轮次检测模型（语义终点检测）+ VAD 校准的静音挂起。

## 概念

![VAD cascade: energy → Silero → turn-detector → flush trick](../assets/vad-turn-taking.svg)

### 三级 VAD 级联

**第 1 级：能量门。** 最便宜。阈值 RMS 在 -40 dBFS。过滤明显静音但在任何高于阈值的噪声上触发。

**第 2 级：Silero VAD**（2020-2026，MIT）。1M 参数。在 6000+ 语言上训练。在单个 CPU 线程上以每 30 毫秒块约 1 毫秒运行。5% FPR 下 87.7% TPR。开源默认选择。

**第 3 级：语义轮次检测器。** LiveKit 的轮次检测模型（2024-2026）或你自己的小型分类器。区分"句中停顿"和"说完"。使用语言上下文（语调 + 最近的词），而不仅仅是静音。

### 关键参数及其默认值

- **阈值。** Silero 输出概率；在 > 0.5（默认）或 > 0.3（敏感）时分类为语音。更低阈值 = 更少的首词剪切、更多的误报。
- **最小语音持续时间。** 拒绝短于 250 毫秒的语音——通常是咳嗽或椅子噪声。
- **静音挂起（终点检测）。** VAD 回到 0 后，等待 500-800 毫秒再声明轮次结束。太短 → 打断用户。太长 → 感觉迟钝。
- **预录缓冲区。** 在 VAD 触发前保留 300-500 毫秒音频。防止"hey"被剪切。

### 同花顺技巧（Kyutai 2025）

流式 STT 模型有前瞻延迟（Kyutai STT-1B 为 500 毫秒，STT-2.6B 为 2.5 秒）。通常你会在语音结束后等待那么长时间才能获得转录。同花顺技巧：当 VAD 触发语音结束时，**向 STT 发送一个刷新信号**，强制立即输出。STT 以约 4 倍实时处理，所以 500 毫秒缓冲在约 125 毫秒内完成。

端到端：125 毫秒 VAD + flush STT = 对话级延迟。

### 2026 年 VAD 比较

| VAD | TPR @ 5% FPR | 延迟 | 许可 |
|-----|--------------|---------|---------|
| WebRTC VAD（Google，2013） | 50.0% | 30 ms | BSD |
| Silero VAD（2020-2026） | 87.7% | ~1 ms | MIT |
| Cobra VAD（Picovoice） | 98.9% | ~1 ms | 商业 |
| pyannote segmentation | 95% | ~10 ms | MIT-ish |

Silero 是正确的默认选择。Cobra 是合规 / 准确率升级。仅能量 VAD 在 2026 年生产中没有位置。

## 构建它

### 步骤 1：能量门

```python
def energy_vad(chunk, threshold_dbfs=-40.0):
    rms = (sum(x * x for x in chunk) / len(chunk)) ** 0.5
    dbfs = 20.0 * math.log10(max(rms, 1e-10))
    return dbfs > threshold_dbfs
```

### 步骤 2：Python 中的 Silero VAD

```python
from silero_vad import load_silero_vad, get_speech_timestamps

vad = load_silero_vad()
audio = torch.tensor(waveform_16k, dtype=torch.float32)
segments = get_speech_timestamps(
    audio, vad, sampling_rate=16000,
    threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=500,
    speech_pad_ms=300,
)
for s in segments:
    print(f"{s['start']/16000:.2f}s - {s['end']/16000:.2f}s")
```

### 步骤 3：轮次结束状态机

```python
class TurnDetector:
    def __init__(self, silence_hangover_ms=500, min_speech_ms=250):
        self.state = "idle"
        self.speech_ms = 0
        self.silence_ms = 0
        self.silence_hangover_ms = silence_hangover_ms
        self.min_speech_ms = min_speech_ms

    def update(self, is_speech, chunk_ms=20):
        if is_speech:
            self.speech_ms += chunk_ms
            self.silence_ms = 0
            if self.state == "idle" and self.speech_ms >= self.min_speech_ms:
                self.state = "speaking"
                return "START"
        else:
            self.silence_ms += chunk_ms
            if self.state == "speaking" and self.silence_ms >= self.silence_hangover_ms:
                self.state = "idle"
                self.speech_ms = 0
                return "END"
        return None
```

### 步骤 4：同花顺技巧骨架

```python
def flush_on_end(stt_client, audio_buffer):
    stt_client.send_audio(audio_buffer)
    stt_client.send_flush()
    return stt_client.recv_transcript(timeout_ms=150)
```

STT（Kyutai、Deepgram、AssemblyAI）必须支持刷新才能工作。Whisper streaming 不支持——它是基于块的，总是等待块。

## 使用它

| 场景 | VAD 选择 |
|-----------|-----------|
| 开源、快速、通用 | Silero VAD |
| 商业呼叫中心 | Cobra VAD |
| 设备端（手机） | Silero VAD ONNX |
| 研究 / 说话人分割 | pyannote segmentation |
| 零依赖后备 | WebRTC VAD（遗留） |
| 需要轮次结束质量 | Silero + 分层的 LiveKit turn-detector |

经验法则：永远不要发布仅能量 VAD，除非你真的没有其他选择。

## 陷阱

- **固定阈值。** 在安静中工作，在嘈杂中失败。要么在设备上校准，要么切换到 Silero。
- **静音挂起太短。** 智能体在句中打断。500-800 毫秒是对话语音的最佳平衡点。
- **挂起太长。** 感觉迟钝。与目标用户进行 A/B 测试。
- **无预录缓冲区。** 丢失用户音频的前 200-300 毫秒。始终保持滚动的预录。
- **忽略语义终点检测。** "Hmm, let me think..."包含长停顿。用户讨厌在思考过程中被切断。使用 LiveKit 的 turn-detector 或类似工具。

## 交付它

保存为 `outputs/skill-vad-tuner.md`。为工作负载选择 VAD 模型、阈值、挂起、预录和轮次检测策略。

## 练习

1. **简单。** 运行 `code/main.py`。它模拟语音 + 静音 + 语音 + 咳嗽序列并测试三个 VAD 级别。
2. **中等。** 安装 `silero-vad`，处理 5 分钟录音，调优阈值以最小化首词剪切和误报。报告精确率/召回率。
3. **困难。** 构建一个小型轮次检测器：Silero VAD + 在最后 10 个词嵌入上的 3 层 MLP（使用 sentence-transformers）。在人工标注的轮次结束数据集上训练。超过仅 Silero 10% F1。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| VAD | 语音检测器 | 每帧二值：这是语音吗？ |
| 轮次检测 | 终点检测 | VAD + 静音挂起 + 语义终点。 |
| 静音挂起 | 语音后等待 | 声明轮次结束前等待的时间；500-800 毫秒。 |
| 预录 | 语音前缓冲区 | VAD 触发前保留 300-500 毫秒音频。 |
| 同花顺技巧 | Kyutai hack | VAD → flush-STT → 125 毫秒而不是 500 毫秒延迟。 |
| 语义终点 | "他们是打算停止吗？" | 看单词而不只是静音的 ML 分类器。 |
| TPR @ FPR 5% | ROC 点 | 标准 VAD 基准；Silero 87.7%、WebRTC 50%。 |

## 扩展阅读

- [Silero VAD](https://github.com/snakers4/silero-vad) — 参考开源 VAD。
- [Picovoice Cobra VAD](https://picovoice.ai/products/cobra/) — 商业准确率领袖。
- [Kyutai — Unmute + flush trick](https://kyutai.org/stt) — 低于 200 毫秒的工程技巧。
- [LiveKit — turn detection](https://docs.livekit.io/agents/logic/turns/) — 生产中的语义终点检测。
- [WebRTC VAD](https://webrtc.googlesource.com/src/) — 遗留基线。
- [pyannote segmentation](https://github.com/pyannote/pyannote-audio) — 说话人分割级的分割。
