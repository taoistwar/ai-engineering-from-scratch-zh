# 构建语音助手流程 — 第六阶段结课项目

> 第 01-11 课的一切，缝合在一起。构建一个听、推理和回话的语音助手。在 2026 年，这是一个已解决的工程问题，不是研究问题——但集成细节决定了它是否能发布。

**类型：** 构建
**语言：** Python
**先修要求：** 第六阶段 · 04、05、06、07、11；第十一阶段 · 09（函数调用）；第十四阶段 · 01（智能体循环）
**预计时间：** 约120分钟

## 问题

构建一个端到端的助手：

1. 捕获麦克风输入（16 kHz 单声道）。
2. 检测用户语音的开始/结束。
3. 流式转录。
4. 将转录传递给可以调用工具（计时器、天气、日历）的 LLM。
5. 将 LLM 文本流式传递给 TTS。
6. 将音频播放回用户。
7. 如果用户在响应中间打断则停止。

延迟目标：在笔记本电脑 CPU 上用户完成话语后 800 毫秒内第一个 TTS 音频字节。质量目标：没有遗漏词、没有在静音上幻觉字幕、没有声音克隆泄漏、没有提示注入成功。

## 概念

![Voice assistant pipeline: mic → VAD → STT → LLM+tools → TTS → speaker](../assets/voice-assistant.svg)

### 七个组件

1. **音频捕获。** 麦克风 → 16 kHz 单声道 → 20 毫秒块。Python 中通常是 `sounddevice`，生产中为原生 AudioUnit/ALSA/WASAPI。
2. **VAD（第 11 课）。** Silero VAD @ 阈值 0.5，最小语音 250 毫秒，静音挂起 500 毫秒。发送"开始"和"结束"信号。
3. **流式 STT（第 4-5 课）。** Whisper-streaming、Parakeet-TDT 或 Deepgram Nova-3（API）。部分 + 最终转录。
4. **带工具调用的 LLM。** GPT-4o / Claude 3.5 / Gemini 2.5 Flash。工具的 JSON schema。流式 token。
5. **流式 TTS（第 7 课）。** Kokoro-82M（最快开源）或 Cartesia Sonic（商业）。在 20 个 LLM token 后启动 TTS。
6. **播放。** 扬声器输出；opus 编码用于低带宽网络。
7. **打断处理器。** 如果在 TTS 播放期间 VAD 触发，停止播放，取消 LLM，重启 STT。

### 你将遇到的三个失败模式

1. **首词剪切。** VAD 启动晚了半拍。用户的"hey"丢失。将开始阈值设为 0.3 而不是 0.5。
2. **响应中打断混淆。** 用户打断后 LLM 继续生成；助手在用户说话时还在说。连线 VAD → 取消 LLM。
3. **静音幻觉。** Whisper 在静音预热帧上输出"Thanks for watching"。始终 VAD 把关。

### 2026 生产参考技术栈

| 技术栈 | 延迟 | 许可 | 备注 |
|-------|---------|---------|-------|
| LiveKit + Deepgram + GPT-4o + Cartesia | 350-500 ms | 商业 API | 2026 年行业默认 |
| Pipecat + Whisper-streaming + GPT-4o + Kokoro | 500-800 ms | 大部分开源 | 适合 DIY |
| Moshi（全双工） | 200-300 ms | CC-BY 4.0 | 单一模型；不同的架构，第 15 课 |
| Vapi / Retell（托管） | 300-500 ms | 商业 | 最快启动；有限定制 |
| Whisper.cpp + llama.cpp + Kokoro-ONNX | 离线 | 开源 | 隐私 / 边缘 |

## 构建它

### 步骤 1：带分块的麦克风捕获（伪代码）

```python
import sounddevice as sd

def mic_stream(chunk_ms=20, sr=16000):
    q = queue.Queue()
    def cb(indata, frames, time, status):
        q.put(indata.copy().flatten())
    with sd.InputStream(channels=1, samplerate=sr, blocksize=int(sr * chunk_ms/1000), callback=cb):
        while True:
            yield q.get()
```

### 步骤 2：VAD 把关轮次捕获

```python
def capture_turn(stream, vad, pre_roll_ms=300, silence_ms=500):
    buf, pre, triggered = [], collections.deque(maxlen=pre_roll_ms // 20), False
    silent = 0
    for chunk in stream:
        pre.append(chunk)
        if vad(chunk):
            if not triggered:
                buf = list(pre)
                triggered = True
            buf.append(chunk)
            silent = 0
        elif triggered:
            silent += 20
            buf.append(chunk)
            if silent >= silence_ms:
                return b"".join(buf)
```

### 步骤 3：流式 STT → LLM → TTS

```python
async def turn(audio_bytes):
    transcript = await stt.transcribe(audio_bytes)
    async for token in llm.stream(transcript):
        async for audio in tts.stream(token):
            await speaker.play(audio)
```

### 步骤 4：LLM 循环内的工具调用

```python
tools = [
    {"name": "get_weather", "parameters": {"location": "string"}},
    {"name": "set_timer", "parameters": {"seconds": "int"}},
]

async for chunk in llm.stream(user_text, tools=tools):
    if chunk.type == "tool_call":
        result = dispatch(chunk.name, chunk.args)
        continue_streaming(result)
    if chunk.type == "text":
        await tts.stream(chunk.text)
```

### 步骤 5：打断处理

```python
tts_task = asyncio.create_task(tts_loop())
while True:
    chunk = await mic.get()
    if vad(chunk):
        tts_task.cancel()
        await speaker.stop()
        await new_turn()
        break
```

## 使用它

参见 `code/main.py` 获取可运行的模拟，该模拟用存根模型连接所有七个组件，所以即使没有硬件也能看到流程形态。对于真实实现，用以下内容替换存根：

- `silero-vad` (`pip install silero-vad`)
- `deepgram-sdk` 或 `openai-whisper`
- `openai` (`gpt-4o`) 或 `anthropic`
- `kokoro` 或 `cartesia`
- `sounddevice` 用于 I/O

## 陷阱

- **永远记录 PII。** 完整轮次音频在大多数司法管辖区是 PII。30 天保留、静态加密。
- **无插话。** 用户会打断。你的助手必须停止说话。
- **阻塞性 TTS。** 同步 TTS 阻塞事件循环。使用异步或单独线程。
- **无工具调用错误处理。** 工具会失败。LLM 必须获取错误 + 重试一次，然后优雅降级。
- **过度热心的幻觉过滤。** 过度过滤，助手重复"I can't help with that。"过滤不足，它说任何话。在保留集上校准。
- **无唤醒词选项。** 始终监听是隐私负担。添加唤醒词门（Porcupine 或 openWakeWord）。

## 交付它

保存为 `outputs/skill-voice-assistant-architect.md`。给定预算 + 规模 + 语言 + 合规约束，生成完整的技术栈规格。

## 练习

1. **简单。** 运行 `code/main.py`。它端到端模拟一个完整轮次，使用存根模块并打印每阶段延迟。
2. **中等。** 在预录制的 `.wav` 上用真实的 Whisper 模型替换 STT 存根。测量 WER 和端到端延迟。
3. **困难。** 添加工具调用：实现 `get_weather`（任意 API）和 `set_timer`。通过工具路由 LLM，并验证当用户说"set a 5 minute timer"时正确的函数触发且语音回复确认它。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| 轮次 | 用户 + 助手往返 | 一次 VAD 限定的用户语音 + 一次 LLM-TTS 响应。 |
| 插话 | 打断 | 用户在助手说话时说话；助手停止。 |
| 唤醒词 | "Hey assistant" | 短关键词检测器；Porcupine、Snowboy、openWakeWord。 |
| 终点检测 | 轮次结束 | VAD + 最小静音决定用户已完成。 |
| 预录 | 语音前缓冲区 | 在 VAD 触发前保留 200-400 毫秒音频以避免首词剪切。 |
| 工具调用 | 函数调用 | LLM 发出 JSON；运行时调度；结果在循环内反馈。 |

## 扩展阅读

- [LiveKit — voice agent quickstart](https://docs.livekit.io/agents/) — 生产级参考。
- [Pipecat — voice agent examples](https://github.com/pipecat-ai/pipecat) — 适合 DIY 的框架。
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — 托管语音原生路径。
- [Kyutai Moshi](https://github.com/kyutai-labs/moshi) — 全双工参考（第 15 课）。
- [Porcupine wake-word](https://picovoice.ai/products/porcupine/) — 唤醒词门槛。
- [Anthropic — tool use guide](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) — LLM 函数调用。
