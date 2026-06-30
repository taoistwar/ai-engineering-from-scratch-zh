# 实践项目 03 — 实时语音助手（ASR 到 LLM 到 TTS）

> 一个感觉对劲的语音智能体需要端到端延迟低于 800ms，知道何时你已停止讲话，处理打断，并且能够调用工具而不会卡顿。Retell、Vapi、LiveKit Agents 和 Pipecat 在 2026 年都达到了这个标准。它们用相同的形态做到：流式 ASR、轮次检测器、流式 LLM 和流式 TTS，全部通过 WebRTC 连接，每个环节都有严格的延迟预算。构建一个，测量 WER、MOS 和虚假截断率，并在丢包情况下运行。

**类型:** 实践项目
**语言:** Python（智能体 + 管道），TypeScript（Web 客户端）
**前置条件:** 阶段 6（语音和音频），阶段 7（transformers），阶段 11（LLM 工程），阶段 13（工具），阶段 14（智能体），阶段 17（基础设施）
**涉及的阶段:** P6 · P7 · P11 · P13 · P14 · P17
**时间:** 30 小时

## 问题

语音是 2025-2026 年发展最快的 AI 用户体验类别。技术天花板每个季度都在降低。OpenAI Realtime API、Gemini 2.5 Live、Cartesia Sonic-2、ElevenLabs Flash v3、LiveKit Agents 1.0 和 Pipecat 0.0.70 都使首次音频输出低于 800ms 成为可能。标准不仅仅是延迟，而是交互感受：不会截断用户的话、不会被用户截断、在句子中间被打断后能恢复、在对话中途调用工具而不会使音频卡顿、在抖动的移动网络中也能存活。

你无法通过拼接三个 REST 调用来实现。架构是端到端的管道化流式传输。构建它，故障模式就变得可见：一个针对电话音频调优的 VAD 在背景电视上误触发、一个轮次检测器等待永远不会出现的标点符号、一个在发出前缓冲 400ms 的 TTS。本实践项目是在负载下逐个修复这些问题，并发布延迟和质量报告。

## 概念

管道有五个流式阶段：**音频输入**（来自浏览器或 PSTN 的 WebRTC）、**ASR**（来自 Deepgram Nova-3 或 faster-whisper 的流式部分转录）、**轮次检测**（VAD 加上一个读取部分转录以获取完成线索的小型轮次检测器模型）、**LLM**（一旦判断轮次完成就流式输出 token）、**TTS**（在第一个 LLM token 后约 200ms 内流式输出音频）。

三个横切关注点。**打断**：当用户在智能体说话时开始说话，TTS 取消，ASR 立即接起。**工具使用**：对话中的函数调用（天气、日历）必须在旁路上运行，不会使音频卡顿；如果延迟超过 300ms，智能体会预先生成一个确认 token（"等一下……"）。**背压**：在丢包情况下，部分转录被暂存，VAD 提高语音门控阈值，智能体避免在未确认的消息上发言。

测量标准是定量的。在 15 dB SNR 的 Hamming VAD 基准上 WER 低于 8%。在 100 次测量通话上，首次音频输出 p50 低于 800ms。虚假截断率低于 3%。TTS 的 MOS 高于 4.2。在单个 g5.xlarge 上 50 个并发呼叫。这些数字就是可交付成果。

## 架构

```
browser / Twilio PSTN
        |
        v
   WebRTC / SIP edge
        |
        v
  LiveKit Agents 1.0  (or Pipecat 0.0.70)
        |
   +----+--------------+--------------+-----------------+
   |                   |              |                 |
   v                   v              v                 v
  ASR              VAD v5         turn-detector     side-channel
(Deepgram         (Silero)          (LiveKit)        tools
 Nova-3 /         speech-gate    completion score    (weather,
 Whisper-v3)      per 20ms        on partials        calendar)
   |                   |              |
   +--------+----------+--------------+
            v
        LLM (streaming)
     GPT-4o-realtime / Gemini 2.5 Flash /
     cascaded Claude Haiku 4.5
            |
            v
        TTS streaming
     Cartesia Sonic-2 / ElevenLabs Flash v3
            |
            v
     audio back to caller
            |
            v
   OpenTelemetry voice traces -> Langfuse
```

## 技术栈

- 传输: LiveKit Agents 1.0（WebRTC）加 Twilio PSTN 网关；Pipecat 0.0.70 作为替代框架
- ASR: Deepgram Nova-3（流式，首次部分转录 < 300ms）或自托管的 faster-whisper Whisper-v3-turbo
- VAD: Silero VAD v5 加 LiveKit 轮次检测器（读取部分转录的小型 transformer）
- LLM: OpenAI GPT-4o-realtime 实现紧密集成，Gemini 2.5 Flash Live，或级联的 Claude Haiku 4.5（流式完成，单独的音频路径）
- TTS: Cartesia Sonic-2（最低首字节延迟），ElevenLabs Flash v3，或用于自托管的开源 Orpheus
- 工具: FastMCP 旁路用于天气/日历/预订；如果工具耗时超过 300ms，智能体会预先发出填充语
- 可观测性: OpenTelemetry 语音 span，Langfuse 语音跟踪带音频回放
- 部署: 单个 g5.xlarge（24GB VRAM）用于自托管 Whisper + Orpheus；托管 API 实现最低延迟

## 构建它

1. **WebRTC 会话。** 搭建一个 LiveKit 房间和一个流式传输麦克风音频的 Web 客户端。在服务器上，连接一个加入房间的智能体 worker。

2. **ASR 流式传输。** 将 20ms 的 PCM 帧送入 Deepgram Nova-3（或 GPU 上的 faster-whisper）。订阅部分和最终转录。记录每次部分的延迟。

3. **VAD 和轮次检测器。** 在帧流上运行 Silero VAD v5。在语音结束事件上，对最新的部分转录触发 LiveKit 轮次检测器。仅当 VAD 指示静音 500ms 且轮次检测器的完成分数 > 0.6 时才承诺"轮次完成"。

4. **LLM 流。** 在轮次完成时，使用正在进行的对话和最终转录启动 LLM 调用。流式输出 token。在第一个 token 时，将控制权交给 TTS。

5. **TTS 流。** Cartesia Sonic-2 将音频块流式传输回来。第一个块必须在第一个 LLM token 后 200ms 内离开服务器。将块发送到 LiveKit 房间；客户端通过 WebRTC 抖动缓冲区播放。

6. **打断。** 当 VAD 在 TTS 播放中检测到新的用户语音时，立即取消 TTS 流，丢弃剩余的 LLM 输出，重新启动 ASR。发布 `tts_canceled` span。

7. **工具旁路。** 将天气和日历注册为函数调用工具。当被调用时，并发地触发调用；如果在 300ms 内没有解决，让 LLM 发出"等一下，让我看看"作为填充语；一旦工具返回就恢复。

8. **评估框架。** 录制 100 次通话。计算 WER（对照留存转录）、虚假截断率（TTS 在用户句子中间被取消）、首次音频输出 p50、TTS MOS（人工或 NISQA），以及抖动丢包测试（丢弃 3% 的数据包）。

9. **负载测试。** 在单个 g5.xlarge 上用合成呼叫者驱动 50 个并发呼叫。测量持续的首次音频输出 p95。

## 使用它

```
caller: "what is the weather in tokyo tomorrow"
[asr  ] partial @280ms: "what is the"
[asr  ] partial @540ms: "what is the weather"
[turn ] completion score 0.82 at @820ms; commit
[llm  ] first token @960ms
[tool ] weather.tokyo tomorrow -> 68/52 partly cloudy @1140ms
[tts  ] first audio-out @1040ms: "Tokyo tomorrow will be partly cloudy..."
turn latency: 1040ms user-stop -> audio-out
```

## 交付它

`outputs/skill-voice-agent.md` 是可交付成果。给定一个领域（客户支持、调度或售货亭），它搭建一个 LiveKit 智能体，ASR/VAD/LLM/TTS 管道针对测量标准调优。评分标准：

| 权重 | 标准 | 如何衡量 |
|:-:|---|---|
| 25 | 端到端延迟 | 在 100 次录制通话中，首次音频输出 p50 低于 800ms |
| 20 | 轮次转换质量 | 在 Hamming VAD 基准上，虚假截断率低于 3% |
| 20 | 工具使用正确性 | 对话中的工具调用在不卡顿音频的情况下返回正确数据 |
| 20 | 丢包情况下的可靠性 | 注入 3% 数据包丢弃时的 WER 和轮次转换稳定性 |
| 15 | 评估框架完整性 | 使用公开配置的可复现测量 |
| **100** | | |

## 练习

1. 在 g5.xlarge 上将 Deepgram Nova-3 替换为 faster-whisper v3 turbo。测量延迟和 WER 差距。识别 CPU 与 GPU 决策的关键点。

2. 添加打断仲裁策略：当用户在工具调用期间打断时，智能体会做什么？比较三种策略（硬取消、完成工具然后停止、排队下一轮）。

3. 运行对抗性轮次检测器测试：让用户在句子中间长停顿。调整 VAD 静音阈值和轮次检测器分数阈值，以达到最低的虚假截断率，同时不超过 900ms。

4. 将同一智能体通过 Twilio 部署在 PSTN 上。比较 PSTN 首次音频输出与 WebRTC 的差异。解释抖动缓冲区和编解码器的差异。

5. 为非英语语言（日语、西班牙语）添加语音活动检测。测量 Silero VAD v5 的虚假触发率与语言特定微调的对比。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| Turn detection | "发音结束" | 分类器，在 VAD 静音和部分转录的基础上决定用户是否说完话 |
| Barge-in | "打断处理" | 当 VAD 检测到新的用户语音时，在播放中取消 TTS |
| First-audio-out | "延迟" | 从用户停止说话到第一个音频包离开服务器的时间 |
| VAD | "语音门控" | 将音频帧分类为语音 vs 静音的模型；Silero VAD v5 是 2026 年的默认模型 |
| Jitter buffer | "音频平滑" | 客户端缓冲区，短暂保持数据包以吸收网络方差 |
| Filler | "确认 token" | 智能体发出的短句，当工具响应慢时避免沉默 |
| MOS | "平均意见分" | 感知语音质量评分；NISQA 是自动化替代方案 |

## 扩展阅读

- [LiveKit Agents 1.0](https://github.com/livekit/agents) — 参考 WebRTC 智能体框架
- [Pipecat](https://github.com/pipecat-ai/pipecat) — 替代的 Python 优先流式智能体框架
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — 集成语音模型的参考
- [Deepgram Nova-3 文档](https://developers.deepgram.com/docs) — 流式 ASR 参考
- [Silero VAD v5](https://github.com/snakers4/silero-vad) — VAD 参考模型
- [Cartesia Sonic-2](https://docs.cartesia.ai) — 低延迟 TTS 参考
- [Retell AI 架构](https://docs.retellai.com) — 生产级语音智能体架构
- [Vapi.ai 生产栈](https://docs.vapi.ai) — 替代的生产参考
