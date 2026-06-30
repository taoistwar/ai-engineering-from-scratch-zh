# 实时音频处理

> 批量流程处理一个文件。实时流程在下一个 20 毫秒到达之前处理当前的 20 毫秒。每个对话式 AI、广播工作室和电话机器人都在这个延迟预算上存亡。

**类型：** 构建
**语言：** Python
**先修要求：** 第六阶段 · 02（频谱图），第六阶段 · 04（ASR），第六阶段 · 07（TTS）
**预计时间：** 约75分钟

## 问题

你想要一个有生命感的语音助手。人类对话轮次交替延迟约 230 毫秒（静默到响应）。超过 500 毫秒感觉机器人化；超过 1500 毫秒感觉坏了。2026 年完整**听 → 理解 → 回应 → 说**循环的预算是：

| 阶段 | 预算 |
|-------|--------|
| 麦克风 → 缓冲区 | 20 ms |
| VAD | 10 ms |
| ASR（流式） | 150 ms |
| LLM（首个 token） | 100 ms |
| TTS（首个块） | 100 ms |
| 渲染 → 扬声器 | 20 ms |
| **总计** | **~400 ms** |

Moshi（Kyutai, 2024）记录到 200 毫秒全双工。GPT-4o-realtime（2024）记录约 320 毫秒。2022 年的级联发布在 2500 毫秒。10 倍改进来自三个技术：(1) 各处流式，(2) 带部分结果的异步流水线，(3) 可打断的生成。

## 概念

![Streaming audio pipeline with ring buffer, VAD gate, interruption](../assets/real-time.svg)

**帧 / 块 / 窗口。** 实时音频以固定大小块流动。常见选择：20 毫秒（16 kHz 下 320 样本）。下游的一切必须跟上这个节奏。

**环形缓冲区。** 固定大小的循环缓冲区。生产者线程写入新帧，消费者线程读取。防止热路径中的分配。大小 ≈ 最大延迟 × 采样率；一个 2 秒 16 kHz 环形 = 32,000 样本。

**VAD（语音活动检测）。** 当没有人在说话时关闭下游工作。Silero VAD 4.0（2024）在 CPU 上每 30 毫秒帧运行 <1 毫秒。`webrtcvad` 是较旧的替代方案。

**流式 ASR。** 在音频到达时发出部分转录的模型。流式模式下的 Parakeet-CTC-0.6B（NeMo, 2024）在 320 毫秒延迟下做到 2–5% WER。Whisper-Streaming（Macháček et al., 2023）分块 Whisper 以在约 2 秒延迟下实现接近流式。

**打断。** 当用户在助手说话时说话，你必须 (a) 检测插话，(b) 停止 TTS，(c) 丢弃剩余的 LLM 输出。全部在 100 毫秒内，否则用户感知到聋哑助手。

**WebRTC Opus 传输。** 20 毫秒帧、48 kHz、自适应比特率 8–128 kbps。浏览器和移动端的标准。LiveKit、Daily.co、Pion 是 2026 年构建语音应用的技术栈。

**抖动缓冲区。** 网络数据包到达乱序/迟到。抖动缓冲区重新排序和平滑；太小 → 可听间隙，太大 → 延迟。通常 60–80 毫秒。

### 常见陷阱

- **线程争用。** Python 的 GIL + 重量级模型可能饿死音频线程。使用 C 回调音频库（sounddevice、PortAudio）并保持 Python 离开热路径。
- **采样率转换延迟。** 流程内重采样增加 5–20 毫秒。要么预先重采样，要么使用零延迟重采样器（PolyPhase、`soxr_hq`）。
- **TTS 预热。** 即使像 Kokoro 这样的快速 TTS 在首次请求上也有 100–200 毫秒的预热。缓存模型并在第一次真实轮次之前用一次虚拟运行预热。
- **回声消除。** 没有 AEC，TTS 输出重新进入麦克风并在机器人自己的声音上触发 ASR。WebRTC AEC3 是开源的默认选择。

```figure
nyquist-aliasing
```

## 构建它

### 步骤 1：环形缓冲区

```python
import collections

class RingBuffer:
    def __init__(self, capacity):
        self.buf = collections.deque(maxlen=capacity)
    def write(self, frame):
        self.buf.extend(frame)
    def read(self, n):
        return [self.buf.popleft() for _ in range(min(n, len(self.buf)))]
    def level(self):
        return len(self.buf)
```

容量决定最大缓冲延迟。16 kHz 下 32,000 样本 = 2 s。

### 步骤 2：VAD 门控

```python
def simple_energy_vad(frame, threshold=0.01):
    return sum(x * x for x in frame) / len(frame) > threshold ** 2
```

在生产中用 Silero VAD 替换：

```python
import torch
vad, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
is_speech = vad(torch.tensor(frame), 16000).item() > 0.5
```

### 步骤 3：流式 ASR

```python
# Parakeet-CTC-0.6B 通过 NeMo 进行流式
from nemo.collections.asr.models import EncDecCTCModelBPE
asr = EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-0.6b")
# chunk_ms=320 ms, look_ahead_ms=80 ms
for chunk in audio_stream():
    partial_text = asr.transcribe_streaming(chunk)
    print(partial_text, end="\r")
```

### 步骤 4：打断处理器

```python
class Dialog:
    def __init__(self):
        self.tts_task = None

    def on_user_speech(self, frame):
        if self.tts_task and not self.tts_task.done():
            self.tts_task.cancel()   # 插话
        # 然后喂入流式 ASR

    def on_final_user_utterance(self, text):
        self.tts_task = asyncio.create_task(self.reply(text))

    async def reply(self, text):
        async for tts_chunk in llm_then_tts(text):
            speaker.write(tts_chunk)
```

依赖于异步 I/O 和可取消的 TTS 流式。WebRTC peerconnection.stop() 在音频轨道上是规范方式。

## 使用它

2026 年技术栈：

| 层 | 选择 |
|-------|------|
| 传输 | LiveKit（WebRTC）或 Pion（Go） |
| VAD | Silero VAD 4.0 |
| 流式 ASR | Parakeet-CTC-0.6B 或 Whisper-Streaming |
| LLM 首个 token | Groq、Cerebras、vLLM-streaming |
| 流式 TTS | Kokoro 或 ElevenLabs Turbo v2.5 |
| 回声消除 | WebRTC AEC3 |
| 端到端原生 | OpenAI Realtime API 或 Moshi |

## 陷阱

- **缓冲 500 毫秒以求安全。** 缓冲区*就是*你的延迟底线。缩小它。
- **不固定线程。** 音频回调在低于 UI 线程的优先级上 = 负载下出现毛刺。
- **TTS 块太小。** 低于 200 毫秒的块使 vocoder 伪影可听。320 毫秒块是最佳平衡点。
- **无抖动缓冲区。** 真实网络有抖动；没有平滑你会得到爆音。
- **单次错误处理。** 音频流程必须防崩溃。一个异常就杀掉会话。

## 交付它

保存为 `outputs/skill-realtime-designer.md`。设计带每阶段具体延迟预算的实时音频流程。

## 练习

1. **简单。** 运行 `code/main.py`。模拟环形缓冲区 + 能量 VAD；为假 10 秒流打印阶段延迟。
2. **中等。** 使用 `sounddevice` 构建一个直通循环，以 20 毫秒帧处理你的麦克风并在每帧打印 VAD 状态。
3. **困难。** 用 `aiortc` 构建全双工回声测试：浏览器 → WebRTC → Python → WebRTC → 浏览器。用 1 kHz 脉冲测量镜面到镜面延迟。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| 环形缓冲区 | 循环队列 | 用于音频帧的固定大小、无锁（或 SPSC 锁）FIFO。 |
| VAD | 静音关 | 标记语音与非语音的模型或启发式。 |
| 流式 ASR | 实时 STT | 在音频到达时发出部分文本；有界前瞻。 |
| 抖动缓冲区 | 网络平滑器 | 重新排序乱序数据包的队列；典型 60–80 毫秒。 |
| AEC | 回声消除 | 减去扬声器到麦克风的反馈路径。 |
| 插话 | 用户打断 | 系统在 TTS 中间检测到用户语音；必须取消播放。 |
| 全双工 | 同时双向 | 用户和机器人可以同时说话；Moshi 是全双工。 |

## 扩展阅读

- [Macháček et al. (2023). Whisper-Streaming](https://arxiv.org/abs/2307.14743) — 分块近流式 Whisper。
- [Kyutai (2024). Moshi](https://kyutai.org/Moshi.pdf) — 全双工 200 毫秒延迟。
- [LiveKit Agents framework (2024)](https://docs.livekit.io/agents/) — 生产音频智能体编排。
- [Silero VAD repo](https://github.com/snakers4/silero-vad) — 低于 1 毫秒 VAD、Apache 2.0。
- [WebRTC AEC3 paper](https://webrtc.googlesource.com/src/+/main/modules/audio_processing/aec3/) — 开源下的回声消除。
