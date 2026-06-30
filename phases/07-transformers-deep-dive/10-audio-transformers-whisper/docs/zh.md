# 音频 Transformer — Whisper 架构

> 音频是频率随时间变化的图像。Whisper 是一个处理 mel 频谱图并回话的 ViT。

**类型：** 学习
**语言：** Python
**前置条件：** 第七阶段 · 05（完整 Transformer），第七阶段 · 08（编码器-解码器），第七阶段 · 09（ViT）
**时间：** 约 45 分钟

## 问题

在 Whisper（OpenAI，Radford 等人 2022）之前，最先进的自动语音识别（ASR）意味着 wav2vec 2.0 和 HuBERT——自监督特征提取器加上微调头。高质量，昂贵的数据流水线，领域脆弱。多语言语音识别需要每个语言家族单独的模型。

Whisper 下了三个赌注：

1. **在所有数据上训练。** 从互联网上抓取的 680,000 小时弱标注音频，涵盖 97 种语言。没有干净的学术语料库。没有音素标签。
2. **多任务单一模型。** 一个解码器在转录、翻译、语音活动检测、语言识别和时间戳上通过任务 token 联合训练。
3. **标准编码器-解码器 transformer。** 编码器消费 log-mel 频谱图。解码器自回归地产生文本 token。没有声码器，没有 CTC，没有 HMM。

结果：Whisper large-v3 在口音、噪声和零干净标注数据的语言上都很鲁棒。它是每个开源语音助手和大多数 2026 年商业语音助手的默认语音前端。

## 概念

![Whisper 流水线：音频 → mel → 编码器 → 解码器 → 文本](../assets/whisper.svg)

### 步骤 1 — 重采样 + 窗口

16 kHz 的音频。裁剪/填充到 30 秒。计算 log-mel 频谱图：80 个 mel 频带，10 ms 步幅 → ~3,000 帧 × 80 个特征。这是 Whisper 看到的"输入图像"。

### 步骤 2 — 卷积前端

两个内核为 3、步幅为 2 的 Conv1D 层将 3,000 帧减少到 1,500 帧。将序列长度减半而不添加大量参数。

### 步骤 3 — 编码器

在 1,500 个时间步上的 24 层（对于 large）transformer 编码器。正弦位置编码，自注意力，GELU FFN。产生 1,500 × 1,280 的隐藏状态。

### 步骤 4 — 解码器

24 层 transformer 解码器。它从 BPE 词汇表自回归地产生 token，该词汇表是 GPT-2 的超集，外加一些音频特定的特殊 token。

### 步骤 5 — 任务 token

解码器提示以控制 token 开始，告诉模型该做什么：

```
<|startoftranscript|>  <|en|>  <|transcribe|>  <|0.00|>
```

或

```
<|startoftranscript|>  <|fr|>  <|translate|>   <|0.00|>
```

模型是在这个约定上训练的。你通过前缀控制任务。2026 年相当于指令微调，但应用于语音。

### 步骤 6 — 输出

具有 log-prob 阈值的束搜索（宽度 5）。当 `<|notimestamps|>` token 不存在时，每 0.02 秒音频预测时间戳。

### Whisper 尺寸

| 模型 | 参数 | 层数 | d_model | 头数 | VRAM (fp16) |
|-------|--------|--------|---------|-------|-------------|
| Tiny | 39M | 4 | 384 | 6 | ~1 GB |
| Base | 74M | 6 | 512 | 8 | ~1 GB |
| Small | 244M | 12 | 768 | 12 | ~2 GB |
| Medium | 769M | 24 | 1024 | 16 | ~5 GB |
| Large | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3 | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3-turbo | 809M | 32 | 1280 | 20 | ~6 GB（4 层解码器） |

Large-v3-turbo（2024）将解码器从 32 层削减到 4 层。解码速度提升 8 倍，WER 回归 <1 点。解码速度的解锁正是 Whisper-turbo 成为 2026 年实时语音代理默认方案的原因。

### Whisper 不能做什么

- 不进行说话人分离（谁在说话）。配合 pyannote 实现。
- 不自带实时流式——30 秒窗口是固定的。现代包装器（`faster-whisper`、`WhisperX`）通过 VAD + 重叠附加流式能力。
- 没有外部分块的情况下不处理超过 30 秒的长形式上下文。在实践中效果良好，因为人类语音在转录时很少需要长距离上下文。

### 2026 年格局

| 任务 | 模型 | 备注 |
|------|-------|-------|
| 英语 ASR | Whisper-turbo、Moonshine | Moonshine 在边缘设备上快 4 倍 |
| 多语言 ASR | Whisper-large-v3 | 97 种语言 |
| 流式 ASR | faster-whisper + VAD | 可实现 150 ms 延迟目标 |
| TTS | Piper、XTTS-v2、Kokoro | 编码器-解码器模式，但形状类似 Whisper |
| 音频 + 语言 | AudioLM、SeamlessM4T | 文本 token + 音频 token 在一个 transformer 中 |

## 动手构建

参见 `code/main.py`。我们不训练 Whisper——我们构建 log-mel 频谱图流水线 + 任务 token 提示格式化器。这些是你在生产中实际接触的部分。

### 步骤 1：合成音频

生成 1 秒的 440 Hz 正弦波，以 16 kHz 采样。16,000 个样本。

### 步骤 2：log-mel 频谱图（简化版）

完整的 mel 频谱图需要 FFT。我们做一个简化的分帧 + 每帧能量版本，展示流水线而不需要 `librosa`：

```python
def frame_signal(x, frame_size=400, hop=160):
    frames = []
    for start in range(0, len(x) - frame_size + 1, hop):
        frames.append(x[start:start + frame_size])
    return frames
```

帧 = 25 ms，跳跃 = 10 ms。匹配 Whisper 的窗口。每帧能量在教学上代替 mel 频带。

### 步骤 3：填充到 30 秒

Whisper 总是处理 30 秒的块。将频谱图填充（或裁剪）到 3,000 帧。

### 步骤 4：构建提示 token

```python
def whisper_prompt(lang="en", task="transcribe", timestamps=True):
    tokens = ["<|startoftranscript|>", f"<|{lang}|>", f"<|{task}|>"]
    if not timestamps:
        tokens.append("<|notimestamps|>")
    return tokens
```

这就是整个任务控制界面。一个 4 token 的前缀。

## 使用它

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("meeting.wav", language="en", task="transcribe")
print(result["text"])
print(result["segments"][0]["start"], result["segments"][0]["end"])
```

更快，兼容 OpenAI：

```python
from faster_whisper import WhisperModel
model = WhisperModel("large-v3-turbo", compute_type="int8_float16")
segments, info = model.transcribe("meeting.wav", vad_filter=True)
for s in segments:
    print(f"{s.start:.2f} - {s.end:.2f}: {s.text}")
```

**2026 年何时选择 Whisper：**

- 用一个模型进行多语言 ASR。
- 对嘈杂、多样音频的鲁棒转录。
- 研究/原型 ASR——最快的起点。

**何时选择其他方案：**

- 边缘设备上的超低延迟流式——Moonshine 在匹配质量上击败 Whisper。
- 需要 <200 ms 的实时对话式 AI——专用的流式 ASR。
- 说话人分离——Whisper 不做这个；附加 pyannote。

## 交付成果

参见 `outputs/skill-asr-configurator.md`。该技能为新的语音应用选择 ASR 模型、解码参数和预处理流水线。

## 练习

1. **简单。** 运行 `code/main.py`。确认以 16 kHz 采样、10 ms 跳跃的 1 秒信号的帧数约为 ~100 帧。对于 30 秒：~3,000 帧。
2. **中等。** 使用 `numpy.fft` 构建完整的 log-mel 频谱图。验证 80 个 mel 频带在数值误差内匹配 `librosa.feature.melspectrogram(n_mels=80)`。
3. **困难。** 实现流式推理：将音频分块为 10 秒窗口，2 秒重叠，在每个块上运行 Whisper，合并转录。在 5 分钟播客样本上测量词错误率 vs 单次传递。

## 关键术语

| 术语 | 人们怎么说 | 它实际上的含义 |
|------|-----------------|-----------------------|
| Mel 频谱图 | "音频图像" | 2D 表示：一个轴上是频率频带，另一个轴上是时间帧；每个单元格的能量经过对数缩放。 |
| Log-mel | "Whisper 看到的东西" | 经过对数的 mel 频谱图；近似人类对响度的感知。 |
| 帧 | "一个时间切片" | 25 ms 样本窗口；以 10 ms 步幅重叠。 |
| 任务 token | "语音的提示前缀" | 解码器提示中如 `<|transcribe|>` / `<|translate|>` 的特殊 token。 |
| 语音活动检测 (VAD) | "找到语音" | 在 ASR 之前移除静音的门控；大幅削减成本。 |
| CTC | "联结时间分类" | 用于无对齐训练的经典 ASR 损失；Whisper 不使用它。 |
| Whisper-turbo | "小解码器，完整编码器" | large-v3 编码器 + 4 层解码器；解码速度提升 8 倍。 |
| Faster-whisper | "生产包装器" | CTranslate2 重新实现；int8 量化；比 OpenAI 参考实现快 4 倍。 |

## 延伸阅读

- [Radford 等人 (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Whisper 论文。
- [OpenAI Whisper 仓库](https://github.com/openai/whisper) — 参考代码 + 模型权重。阅读 `whisper/model.py` 以约 400 行自上而下查看 Conv1D 前端 + 编码器 + 解码器。
- [OpenAI Whisper — `whisper/decoding.py`](https://github.com/openai/whisper/blob/main/whisper/decoding.py) — 步骤 5–6 中描述的束搜索 + 任务 token 逻辑在此处；500 行，完全可读。
- [Baevski 等人 (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) — 前身；在某些设置中仍然是 SOTA 特征。
- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) — 生产包装器，比参考实现快 4 倍。
- [Jia 等人 (2024). Moonshine: Speech Recognition for Live Transcription and Voice Commands](https://arxiv.org/abs/2410.15608) — 2024 年边缘友好的 ASR，形状类似 Whisper 但更小。
- [HuggingFace 博客 — "Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers"](https://huggingface.co/blog/fine-tune-whisper) — 规范微调配方，包括 mel 频谱图预处理器和 token-时间戳处理。
- [HuggingFace `modeling_whisper.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py) — 完整实现（编码器、解码器、交叉注意力、生成），镜像本课的架构图。
