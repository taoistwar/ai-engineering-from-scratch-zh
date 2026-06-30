# 流式语音到语音 — Moshi、Hibiki 与全双工对话

> 2024-2026 年重新定义了语音 AI。Moshi 发布了一个以 200 毫秒延迟同时听和说的单一模型。Hibiki 逐块进行语音到语音翻译。两者都放弃了 ASR → LLM → TTS 流程，转而采用基于 Mimi 编解码 token 的统一全双工架构。这是新的参考设计。

**类型：** 学习
**语言：** Python
**先修要求：** 第六阶段 · 13（神经音频编解码器），第六阶段 · 11（实时音频），第七阶段 · 05（完整 Transformer）
**预计时间：** 约75分钟

## 问题

每个从第 11 + 12 课构建的语音代理都有大约 300-500 毫秒的基本延迟底限：VAD 触发、STT 处理、LLM 推理、TTS 生成。每个阶段都有自己的最小延迟。你可以调优和并行化，但流程形态有上限。

Moshi（Kyutai，2024-2026）问了一个不同的问题：如果没有流程呢？如果一个模型直接将音频作为输入并直接发出音频作为输出，持续地，以文本作为中间"内心独白"而不是必需的阶段呢？

答案是**全双工语音到语音**。理论延迟 160 毫秒（80 毫秒 Mimi 帧 + 80 毫秒声学延迟）。实用延迟在单个 L4 GPU 上为 200 毫秒。这是一流流程语音代理所达到的一半。

## 概念

![Moshi architecture: two parallel Mimi streams + inner-monologue text](../assets/moshi-hibiki.svg)

### Moshi 架构

**输入。** 两个 Mimi 编解码流，均为 12.5 Hz × 8 编解码本：

- 流 1：用户音频（Mimi 编码，持续到达）
- 流 2：Moshi 自己的音频（由 Moshi 生成）

**Transformer。** 一个 7B 参数的时序 Transformer 处理两个流和一个文本"内心独白"流。在每个 80 毫秒步骤中，它：

1. 消费最新的用户 Mimi token（8 编解码本）。
2. 消费最近的 Moshi Mimi token（8 编解码本，如生成时）。
3. 生成下一个 Moshi 文本 token（内心独白）。
4. 生成下一个 Moshi Mimi token（通过一个小型深度 Transformer 的 8 编解码本）。

所有三个流——用户音频、Moshi 音频、Moshi 文本——并行运行。Moshi 可以在说话时听到用户；可以在用户打断时自我打断；可以发出"mhm"后声道而不破坏其主话语。

**深度 transformer。** 在一个帧内，8 个编解码本不是并行预测的——它们有编解码本间的依赖关系。一个小型 2 层"深度 transformer"在 80 毫秒内顺序预测它们。这是 AR 编解码 LM 的标准因子分解（也被 VALL-E、VibeVoice 使用）。

### 为什么内心独白文本有帮助

没有显式文本的情况下，模型必须在声学流中隐式建模语言。Moshi 的洞察：强制它同时发出文本 token 和音频。文本流实质上是 Moshi 正在说内容的转录。这提高了语义连贯性，使其更容易替换语言模型头，并免费给你转录。

### Hibiki：流式语音到语音翻译

相同的架构，在翻译对上训练。源语言音频输入，目标语言音频输出，持续地。Hibiki-Zero（2026 年 2 月）消除了对词级对齐训练数据的需求——使用句子级数据 + GRPO 强化学习进行延迟优化。

初始支持四个语言对；可用约 1000 小时适应新语言。

### 更广泛的 Kyutai 技术栈（2026）

- **Moshi** — 全双工对话（法语为主、英语良好支持）
- **Hibiki / Hibiki-Zero** — 同声语音翻译
- **Kyutai STT** — 流式 ASR（500 毫秒或 2.5 秒前瞻）
- **Kyutai Pocket TTS** — 100M 参数 TTS 在 CPU 上运行（2026 年 1 月）
- **Unmute** — 在公共服务器上组合这些的完整流程

在 L40S GPU 上的吞吐量：3 倍实时下 64 个并发会话。

### Sesame CSM — 表亲

Sesame CSM（2025）使用类似的想法——带 Mimi 编解码头的 Llama-3 骨干。但 CSM 是单向的（取上下文 + 文本、产生语音）而不是全双工。它是市场上最好的"声音存在感"TTS；与 Moshi 的全双工能力不完全相同。

### 2026 年性能数字

| 模型 | 延迟 | 用例 | 许可 |
|-------|---------|----------|---------|
| Moshi | 200 ms（L4） | 全双工英语 / 法语对话 | CC-BY 4.0 |
| Hibiki | 12.5 Hz 帧率 | 法语 ↔ 英语流式翻译 | CC-BY 4.0 |
| Hibiki-Zero | 相同 | 5 个语言对、无对齐数据 | CC-BY 4.0 |
| Sesame CSM-1B | 200 ms TTFA | 以上下文为条件的 TTS | Apache-2.0 |
| GPT-4o Realtime | ~300 ms | 封闭、OpenAI API | 商业 |
| Gemini 2.5 Live | ~350 ms | 封闭、Google API | 商业 |

## 构建它

### 步骤 1：接口

Moshi 暴露一个 WebSocket 服务器，接收 80 毫秒块 Mimi 编码音频并返回 80 毫秒块 Mimi 编码音频。双向。持续。

```python
import asyncio
import websockets
from moshi.client_utils import encode_audio_mimi, decode_audio_mimi

async def moshi_chat():
    async with websockets.connect("ws://localhost:8998/api/chat") as ws:
        mic_task = asyncio.create_task(stream_mic_to(ws))
        spk_task = asyncio.create_task(stream_from_to_speaker(ws))
        await asyncio.gather(mic_task, spk_task)
```

### 步骤 2：全双工循环

```python
async def stream_mic_to(ws):
    async for chunk_80ms in mic_stream_at_12_5_hz():
        mimi_tokens = encode_audio_mimi(chunk_80ms)
        await ws.send(serialize(mimi_tokens))

async def stream_from_to_speaker(ws):
    async for msg in ws:
        mimi_tokens, text_token = deserialize(msg)
        audio = decode_audio_mimi(mimi_tokens)
        await play(audio)
```

两个方向同时运行。Python asyncio 或 Rust futures 是标准传输。

### 步骤 3：训练目标（概念）

对于每 80 毫秒帧 `t`：

- 输入：`user_mimi[0..t]`、`moshi_mimi[0..t-1]`、`moshi_text[0..t-1]`
- 预测：`moshi_text[t]`，然后 `moshi_mimi[t, codebook_0..7]`

文本在音频之前预测（内心独白）；音频在深度 transformer 内按编解码本顺序预测。

### 步骤 4：Moshi 胜出的地方和不胜出的地方

Moshi 胜出：

- 在廉价硬件上低于 250 毫秒端到端。
- 自然的后声道和打断。
- 无流程粘合代码。

Moshi 不胜出：

- 工具调用（未为此训练；需要一个单独的 LLM 路径）。
- 长推理（Moshi 是一个约 8B 的对话模型，不是 Claude/GPT-4）。
- 小众话题上的事实准确性。
- 大多数生产企业用例（2026 年仍然使用流程）。

## 使用它

| 场景 | 选择 |
|-----------|------|
| 最低延迟语音伴侣 | Moshi |
| 实时翻译通话 | Hibiki |
| 语音演示 / 研究 | Moshi、CSM |
| 带工具的企业智能体 | 流程（第 12 课），不是 Moshi |
| 上下文中的自定义声音 TTS | Sesame CSM |
| 语音到语音、任意语言 | GPT-4o Realtime 或 Gemini 2.5 Live（商业） |

## 陷阱

- **有限的工具调用。** Moshi 是对话模型，不是智能体框架。结合流程用于工具。
- **特定声音条件。** Moshi 使用单个训练角色；克隆是独立的训练运行。
- **语言覆盖。** 法语 + 英语出色；其他语言有限。Hibiki-Zero 有帮助，但你仍然需要训练数据。
- **资源成本。** 一个完整 Moshi 会话占用一个 GPU 槽位；不是一个廉价的共享租户部署模式。

## 交付它

保存为 `outputs/skill-duplex-pipeline.md`。为语音代理工作负载选择流程 vs 全双工架构，并附理由。

## 练习

1. **简单。** 运行 `code/main.py`。它符号性地模拟双流 + 内心独白架构。
2. **中等。** 从 HuggingFace 拉取 Moshi，运行服务器，测试一次对话。测量从用户语音结束到 Moshi 响应开始的墙钟延迟。
3. **困难。** 取你的第 12 课流程智能体，在 20 个匹配的测试话语上比较 P50 延迟 vs Moshi。撰写何时流程在架构上无论如何仍然胜出。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| 全双工 | 同时听和说 | 同一模型上同时活跃的两个音频流。 |
| 内心独白 | 模型的文本流 | Moshi 在其音频输出旁发出文本 token。 |
| 深度 transformer | 编解码本间预测器 | 在一个 80 毫秒帧内预测 8 个编解码本的小型 transformer。 |
| Mimi | Kyutai 的编解码器 | 12.5 Hz × 8 编解码本；语义+声学；驱动 Moshi。 |
| 流式 S2S | 音频 → 音频实时 | 逐块翻译/对话，无流程阶段。 |
| 后声道 | "Mhm"反应 | Moshi 可以在不破坏其轮次的情况下发出小确认。 |

## 扩展阅读

- [Défossez et al. (2024). Moshi — speech-text foundation model](https://arxiv.org/html/2410.00037v2) — 论文。
- [Kyutai Labs (2026). Hibiki-Zero](https://arxiv.org/abs/2602.12345) — 无对齐数据的流式翻译。
- [Sesame (2025). Crossing the uncanny valley of voice](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice) — CSM 规格。
- [Kyutai — Moshi repo](https://github.com/kyutai-labs/moshi) — 安装 + 服务器。
- [OpenAI — Realtime API](https://platform.openai.com/docs/guides/realtime) — 封闭商业对等体。
- [Kyutai — Delayed Streams Modeling](https://github.com/kyutai-labs/delayed-streams-modeling) — 底层 STT/TTS 框架。
