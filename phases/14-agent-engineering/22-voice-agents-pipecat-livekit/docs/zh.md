# 语音 Agent：Pipecat 与 LiveKit

> 语音 agent 是 2026 年的一流生产类别。Pipecat 给你一个 Python 基于帧的管道（VAD → STT → LLM → TTS → 传输）。LiveKit Agents 通过 WebRTC 将 AI 模型桥接到用户。生产延迟目标在高端技术栈上达到 450–600ms 端到端。

**类型：** 学习
**语言：** Python（标准库）
**前置条件：** 第 14 阶段 · 01（Agent 循环），第 14 阶段 · 12（工作流模式）
**时间：** ~60 分钟

## 学习目标

- 描述 Pipecat 基于帧的管道：DOWNSTREAM（源→汇）和 UPSTREAM（控制）。
- 列举规范的语音管道阶段以及 Pipecat 支持的传输方式。
- 解释 LiveKit Agents 的两个语音 agent 类（MultimodalAgent、VoicePipelineAgent）以及各自适合的场景。
- 总结 2026 年生产延迟预期以及它们如何驱动架构选择。

## 问题

语音 agent 不是文本循环附加上 TTS。延迟预算很苛刻（~600ms），部分音频是默认的，轮次检测本身是一个模型，传输方式从电话 SIP 到 WebRTC 不等。要么你构建一个基于帧的管道（Pipecat），要么你依赖一个平台（LiveKit）。

## 概念

### Pipecat（pipecat-ai/pipecat）

- Python 基于帧的管道框架。
- `Frame` → `FrameProcessor` 链。
- 两个流向：
  - **DOWNSTREAM** — 源 → 汇（音频输入，TTS 输出）。
  - **UPSTREAM** — 反馈和控制（取消、指标、打断）。
- `PipelineTask` 通过事件管理生命周期（`on_pipeline_started`、`on_pipeline_finished`、`on_idle_timeout`）以及用于指标/追踪/RTVI 的观察者。

典型管道：

```
VAD (Silero) → STT → LLM (上下文交替用户/助手) → TTS → 传输
```

传输方式：Daily、LiveKit、SmallWebRTCTransport、FastAPI WebSocket、WhatsApp。

Pipecat Flows 添加了结构化对话（状态机）。Pipecat Cloud 是托管运行时。

### LiveKit Agents（livekit/agents）

- 通过 WebRTC 将 AI 模型桥接到用户。
- 关键概念：`Agent`、`AgentSession`、`entrypoint`、`AgentServer`。
- 两个语音 agent 类：
  - **MultimodalAgent** — 通过 OpenAI Realtime 或等效方式的直接音频。
  - **VoicePipelineAgent** — STT → LLM → TTS 级联；提供文本级控制。
- 通过 transformer 模型进行语义轮次检测。
- 原生 MCP 集成。
- 通过 SIP 的电话接入。
- 通过 LiveKit Inference 提供 50+ 模型无需 API 密钥；通过插件提供 200+ 更多。

### 商业平台

Vapi（在优化的高端技术栈上约 450–600ms）和 Retell（在 180 个测试呼叫中端到端约 600ms）建立在它们之上。当你想要一个托管的语音技术栈而不需要 WebRTC 团队时，选择平台。

### 这种模式可能出错的地方

- **没有打断处理。** 用户打断；agent 继续说话。需要在 Pipecat 中使用 UPSTREAM 取消帧，在 LiveKit 中使用等效方式。
- **STT 置信度被忽略。** 低置信度转录被当作真理输入给 LLM。根据置信度门控或请求确认。
- **TTS 句中截断。** 当管道在语句中途取消时，TTS 需要知道或切断音频。
- **延迟预算被忽略。** 每个组件增加 50–200ms。在发布前对你的链求和。

### 典型的 2026 年延迟

- VAD：20–60ms
- STT 部分：100–250ms
- LLM 首个 token：150–400ms
- TTS 首个音频：100–200ms
- 传输 RTT：30–80ms

端到端 450–600ms 是高端水平。800–1200ms 是常见的。任何 > 1500ms 都会感觉坏了。

## 构建它

`code/main.py` 是一个基于帧的玩具管道，包含：

- `Frame` 类型（audio、transcript、text、tts_audio、control）。
- `Processor` 接口，带有 `process(frame)`。
- 一个五阶段管道（VAD → STT → LLM → TTS → 传输）作为脚本化处理器。
- 一个 UPSTREAM 取消帧以演示打断。

运行它：

```
python3 code/main.py
```

跟踪显示正常流程和一次打断取消，在语句中途停止 TTS。

## 使用它

- **Pipecat** 用于完全控制——自定义处理器，Python 优先，可插拔提供者。
- **LiveKit Agents** 用于 WebRTC 优先部署和电话。
- **Vapi / Retell** 用于托管语音 agent，无需 WebRTC 团队。
- **OpenAI Realtime / Gemini Live** 用于直接音频输入/音频输出（MultimodalAgent）。

## 交付它

`outputs/skill-voice-pipeline.md` 构建一个 Pipecat 形态的语音管道，带有 VAD + STT + LLM + TTS + 传输加上打断处理。

## 练习

1. 向你的玩具管道添加一个指标观察者：计算每秒每个阶段的帧数。延迟在哪里积累？
2. 实现基于置信度的 STT 门控：低于阈值，请求"你能重复一遍吗？"
3. 添加语义轮次检测：简单规则——如果转录以"?"结尾，轮次结束。
4. 阅读 Pipecat 的传输文档。将标准库传输替换为 SmallWebRTCTransport 配置（桩）。
5. 在相同查询上测量 OpenAI Realtime vs STT+LLM+TTS 级联。文本级控制带来了什么延迟代价？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| Frame | "事件" | 管道中的类型化数据单元（audio、transcript、text、control） |
| Processor | "管道阶段" | 带有 process(frame) 的处理器 |
| DOWNSTREAM | "前向流" | 源到汇：音频输入，语音输出 |
| UPSTREAM | "反馈流" | 控制：取消、指标、打断 |
| VAD | "语音活动检测" | 检测用户何时在说话 |
| 语义轮次检测（Semantic turn detection） | "智能轮次结束" | 基于模型的判断用户是否说完 |
| MultimodalAgent | "直接音频 agent" | 音频输入，音频输出；中间没有文本 |
| VoicePipelineAgent | "级联 agent" | STT + LLM + TTS；文本级控制 |

## 进一步阅读

- [Pipecat 文档](https://docs.pipecat.ai/getting-started/introduction) — 基于帧的管道、处理器、传输
- [LiveKit Agents 文档](https://docs.livekit.io/agents/) — WebRTC + 语音原语
- [Vapi](https://vapi.ai/) — 托管语音平台
- [Retell AI](https://www.retellai.com/) — 托管语音，延迟基准测试
