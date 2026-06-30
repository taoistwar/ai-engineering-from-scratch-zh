# 语音识别（ASR） — CTC、RNN-T、注意力

> 语音识别是在每个时间步进行音频分类，由一个了解英语和静音的序列模型粘合在一起。CTC、RNN-T 和注意力是实现这一点的三种方式。选择一个并理解为什么。

**类型：** 构建
**语言：** Python
**先修要求：** 第六阶段 · 02（频谱图与 Mel），第五阶段 · 08（文本上的 CNN 和 RNN），第五阶段 · 10（注意力）
**预计时间：** 约45分钟

## 问题

你有一个 10 秒 16 kHz 片段。你想要一个字符串："turn on the kitchen lights"。挑战是结构性的：音频帧与字符不是一对一对齐的。词"okay"可能花费 200 毫秒或 1200 毫秒。静音将话语标点。某些音素比其他音素长。输出 token 的数量事先未知。

三种公式解决了这个问题：

1. **CTC（联结时序分类）。** 发出每帧 token 概率，包括一个特殊的*空白*。在解码时折叠重复和空白。非自回归、快速。被 wav2vec 2.0、MMS 使用。
2. **RNN-T（循环神经网络转换器）。** 联合网络预测给定编码器帧和先前 token 的下一个 token。可流式。被 Google 的设备端 ASR、NVIDIA Parakeet 使用。
3. **注意力编码器-解码器。** 编码器将音频压缩为隐藏状态，解码器进行交叉注意力以自回归生成 token。被 Whisper、SeamlessM4T 使用。

在 2026 年，LibriSpeech test-clean 上的最先进 WER 是 1.4%（Parakeet-TDT-1.1B，NVIDIA）和 1.58%（Whisper-Large-v3-turbo）。差异很小；部署差异巨大。

## 概念

![Three ASR formulations: CTC, RNN-T, attention-encoder-decoder](../assets/asr-formulations.svg)

**CTC 直觉。** 让编码器在 `V+1` 个 token（V 字符 + 空白）上输出 `T` 个帧级分布。对于长度 `U < T` 的目标字符串 `y`，任何折叠为 `y` 的帧对齐都算数。CTC 损失在所有此类对齐上求和。推理：每帧 argmax、折叠重复、移除空白。

优点：非自回归、可流式、零前瞻。缺点：*条件独立假设*——每个帧预测彼此独立，因此没有内部语言模型。通过束搜索或浅融合的外部 LM 来修复。

**RNN-T 直觉。** 添加一个嵌入 token 历史的*预测器*网络和一个将预测器状态与编码器帧结合为 `V+1` 上的联合分布的*连接器*（`+1` 是空/不发出）。显式地建模 CTC 忽略的条件依赖性。可流式，因为每步仅以过去的帧和过去的 token 为条件。

优点：可流式 + 内部 LM。缺点：训练更复杂且消耗内存（3D 损失格子）；RNN-T 损失核心自成一个库类别。

**注意力编码器-解码器。** 编码器（6-32 个 transformer 层）在对数 Mel 帧上。解码器（6-32 个 transformer 层）对编码器输出进行交叉注意力来自回归生成 token。无对齐约束——注意力可以查看音频中的任何位置。除非你限制注意力，否则不可流式（分块 Whisper-Streaming，2024）。

优点：离线 ASR 的最高质量，使用标准 seq2seq 工具容易训练。缺点：自回归延迟与输出长度成比例；没有工程化无法流式。

### WER：一个数字

**词错率** = `(S + D + I) / N`，其中 S=替换、D=删除、I=插入、N=参考词数。在词级别匹配 Levenshtein 编辑距离。越低越好。WER 超过 20% 通常不可用；低于 5% 是朗读语音的人类水平。2026 年标准基准上的数字：

| 模型 | LibriSpeech test-clean | LibriSpeech test-other | 大小 |
|-------|------------------------|------------------------|------|
| Parakeet-TDT-1.1B | 1.40% | 2.78% | 1.1B 参数 |
| Whisper-Large-v3-turbo | 1.58% | 3.03% | 809M |
| Canary-1B Flash | 1.48% | 2.87% | 1B |
| Seamless M4T v2 | 1.7% | 3.5% | 2.3B |

所有这些都是基于编码器-解码器或 RNN-T 的。纯 CTC 系统（wav2vec 2.0）在 test-clean 上约为 1.8–2.1%。

## 构建它

### 步骤 1：贪心 CTC 解码

```python
def ctc_greedy(frame_logits, blank=0, vocab=None):
    # frame_logits: 每帧概率向量的列表
    preds = [max(range(len(p)), key=lambda i: p[i]) for p in frame_logits]
    out = []
    prev = -1
    for p in preds:
        if p != prev and p != blank:
            out.append(p)
        prev = p
    return "".join(vocab[i] for i in out) if vocab else out
```

两条规则：折叠连续重复、丢弃空白。示例：`a a _ _ a b b _ c` → `a a b c`。

### 步骤 2：束搜索 CTC

```python
def ctc_beam(frame_logits, beam=8, blank=0):
    import math
    beams = [([], 0.0)]  # (tokens, log_prob)
    for p in frame_logits:
        log_p = [math.log(max(pi, 1e-10)) for pi in p]
        candidates = []
        for seq, lp in beams:
            for t, lpt in enumerate(log_p):
                new = seq[:] if t == blank else (seq + [t] if not seq or seq[-1] != t else seq)
                candidates.append((new, lp + lpt))
        candidates.sort(key=lambda x: -x[1])
        beams = candidates[:beam]
    return beams[0][0]
```

生产使用带 LM 融合的前缀树束搜索；这是概念骨架。

### 步骤 3：WER

```python
def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        dp[i][0] = i
    for j in range(len(h) + 1):
        dp[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[len(r)][len(h)] / max(1, len(r))
```

### 步骤 4：针对 Whisper 的推理

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("clip.wav")
print(result["text"])
```

2026 年最强通用 ASR 的一行代码。在 24 GB GPU 上以约 20 倍实时运行。

### 步骤 5：使用 Parakeet 或 wav2vec 2.0 进行流式处理

```python
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="nvidia/parakeet-tdt-1.1b")
for chunk in streaming_audio():
    print(asr(chunk, return_timestamps=True))
```

流式 ASR 需要分块编码器注意力和担带状态；使用支持它的库（Parakeet 用 NeMo，带 `chunk_length_s` 的 `transformers` 流程）。

## 使用它

2026 年技术栈：

| 场景 | 选择 |
|-----------|------|
| 英语、离线、最高质量 | Whisper-large-v3-turbo |
| 多语言、鲁棒 | SeamlessM4T v2 |
| 流式、低延迟 | Parakeet-TDT-1.1B 或 Riva |
| 边缘、移动、<500 毫秒延迟 | Whisper-Tiny 量化或 Moonshine（2024） |
| 长格式 | Whisper 带基于 VAD 的分块（WhisperX） |
| 领域特定（医学、法律） | 微调 wav2vec 2.0 + 领域 LM 融合 |

## 2026 年仍未被捕获就发布的陷阱

- **无 VAD。** 在静音上运行 Whisper 会产生幻觉（"Thanks for watching!"）。始终用 VAD 把关。
- **字符 vs 词 vs 子词 WER。** 报告归一化后的词级 WER（小写、标点去除）。
- **语言 ID 漂移。** Whisper 的自动 LID 将嘈杂片段误路由到日语或威尔士语；当你知道时强制 `language="en"`。
- **无分块的长片段。** Whisper 有 30 秒窗口。对于任何更长的使用 `chunk_length_s=30, stride=5`。

## 交付它

保存为 `outputs/skill-asr-picker.md`。为给定的部署目标选择模型、解码策略、分块和 LM 融合。

## 练习

1. **简单。** 运行 `code/main.py`。它贪心解码一个手工制作的 CTC 输出并计算相对于参考的 WER。
2. **中等。** 正确实现步骤 2 中的前缀树束搜索（考虑空白合并规则）。在 10 样本合成数据集上与贪心比较。
3. **困难。** 在 [LibriSpeech test-clean](https://www.openslr.org/12) 上使用 `whisper-large-v3-turbo`。在前 100 个话语上计算 WER。与已发布数字比较。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| CTC | 空白 token 损失 | 在所有帧到 token 对齐上的边缘分布；非自回归。 |
| RNN-T | 流式损失 | CTC + 下一个 token 预测器；处理词序。 |
| 注意力编码-解码 | Whisper 风格 | 编码器 + 交叉注意力解码器；最佳离线质量。 |
| WER | 你报告的数字 | 词级的 `(S+D+I)/N`。 |
| 空白 | 空 | CTC 中表示"此帧不发出"的特殊 token。 |
| LM 融合 | 外部语言模型 | 在束搜索期间添加加权的 LM 对数概率。 |
| VAD | 静音关 | 语音活动检测器；修剪非语音。 |

## 扩展阅读

- [Graves et al. (2006). Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf) — CTC 论文。
- [Graves (2012). Sequence Transduction with RNNs](https://arxiv.org/abs/1211.3711) — RNN-T 论文。
- [Radford et al. / OpenAI (2022). Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — 2022 年规范论文；2024 年 v3-turbo 扩展。
- [NVIDIA NeMo — Parakeet-TDT card](https://huggingface.co/nvidia/parakeet-tdt-1.1b) — 2026 年开放 ASR 排行榜领先者。
- [Hugging Face — Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — 跨 25+ 模型的实时基准。
