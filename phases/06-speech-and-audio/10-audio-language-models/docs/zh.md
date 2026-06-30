# 音频-语言模型 — Qwen2.5-Omni、Audio Flamingo、GPT-4o Audio

> 2026 年音频语言模型对语音 + 环境声音 + 音乐进行推理。Qwen2.5-Omni-7B 在 MMAU-Pro 上与 GPT-4o Audio 匹配。Audio Flamingo Next 在 LongAudioBench 上击败 Gemini 2.5 Pro。开放和封闭之间的差距基本关闭——除了多音频任务，所有人都接近随机水平。

**类型：** 学习
**语言：** Python
**先修要求：** 第六阶段 · 04（ASR），第十二阶段 · 03（视觉-语言模型），第七阶段 · 10（音频 Transformer）
**预计时间：** 约45分钟

## 问题

你有 5 秒的音频：狗叫，有人喊"stop!"，然后静音。有用的问题跨越多轴：

- **转录。** "说了什么？"——ASR 领域。
- **语义推理。** "这个人处于危险中吗？"——需要对吠声 + 喊叫 + 静音的联合理解。
- **音乐推理。** "哪些乐器演奏旋律？"
- **长音频检索。** "在这 90 分钟的讲座中，讲师在哪里解释了梯度下降？"

用单一提示回答所有这些问题的一个模型就是一个**音频-语言模型**（LALM / ALM）。与纯 ASR 分开：LALM 产生自由形式的自然语言答案，而不仅仅是转录。

## 概念

![Audio-language model: audio encoder + projector + LLM decoder](../assets/alm-architecture.svg)

### 三组件模板

每个 2026 LALM 具有相同的骨架：

1. **音频编码器。** Whisper 编码器 · BEATs · CLAP · WavLM · 或每个模型的自定义编码器。
2. **投影器。** 将音频编码器特征桥接到 LLM 的 token 嵌入空间的线性或 MLP。
3. **LLM。** 基于 Llama / Qwen / Gemma 的解码器。取交错的文本 + 音频 token；生成文本。

训练：

- **阶段 1。** 冻结编码器 + LLM；仅在 ASR / 字幕数据上训练投影器。
- **阶段 2。** 在遵循指令的音频任务（QA、推理、音乐理解）上进行全 / LoRA 微调。
- **阶段 3（可选）。** 语音输入 / 语音输出添加一个语音解码器。Qwen2.5-Omni 和 AF3-Chat 做到了。

### 2026 年模型地图

| 模型 | 骨干 | 音频编码器 | 输出模态 | 访问 |
|-------|----------|---------------|-----------------|--------|
| Qwen2.5-Omni-7B | Qwen2.5-7B | 自定义 + Whisper | 文本 + 语音 | Apache-2.0 |
| Qwen3-Omni | Qwen3 | 自定义 | 文本 + 语音 | Apache-2.0 |
| Audio Flamingo 3 | Qwen2 | AF-CLAP | 文本 | NVIDIA 非商业 |
| Audio Flamingo Next | Qwen2 | AF-CLAP v2 | 文本 | NVIDIA 非商业 |
| SALMONN | Vicuna | Whisper + BEATs | 文本 | Apache-2.0 |
| LTU / LTU-AS | Llama | CAV-MAE | 文本 | Apache-2.0 |
| GAMA | Llama | AST + Q-Former | 文本 | Apache-2.0 |
| Gemini 2.5 Flash/Pro（封闭） | Gemini | 专有 | 文本 + 语音 | API |
| GPT-4o Audio（封闭） | GPT-4o | 专有 | 文本 + 语音 | API |

### 基准现实检查（2026）

**MMAU-Pro。** 1800 个 QA 对，涵盖语音 / 声音 / 音乐 / 混合。包括多音频子集。

| 模型 | 总体 | 语音 | 声音 | 音乐 | 多音频 |
|-------|---------|--------|-------|-------|-------------|
| Gemini 2.5 Pro | ~60% | 73.4% | 51.9% | 64.9% | ~22% |
| Gemini 2.5 Flash | ~57% | 73.4% | 50.5% | 64.9% | 21.2% |
| GPT-4o Audio | 52.5% | — | — | — | 26.5% |
| Qwen2.5-Omni-7B | 52.2% | 57.4% | 47.6% | 61.5% | ~20% |
| Audio Flamingo 3 | ~54% | — | — | — | — |
| Audio Flamingo Next | LongAudioBench 上最先进 | — | — | — | — |

**多音频一栏对所有人都是谴责性的。** 4 选项多选的随机水平 = 25%；大多数模型得分在此附近。LALM 在比较两个片段上仍然挣扎。

### 2026 年 LALM 有用之处

- **呼叫中心录音的合规审计。** "代理是否提到了所需的披露？"
- **无障碍。** 向聋人用户描述声音事件（不仅仅是转录）。
- **内容审核。** 检测暴力语言 + 威胁语气 + 背景上下文。
- **播客 / 会议章节化。** 语义摘要，而不仅仅是说话人轮次。
- **音乐目录分析。** "找到所有带有 B 段调性变化的曲目。"

### 它们（还）不有用之处

- 低于和弦级的细粒度音乐理论。
- 长对话的说话人属性推理（超过 10 分钟退化）。
- 多音频比较（22-26% 略高于随机）。
- 实时流式推理（大多数是离线批量推理）。

## 构建它

### 步骤 1：查询 Qwen2.5-Omni

```python
from transformers import AutoModelForCausalLM, AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype="auto")

audio, sr = load_wav("clip.wav", sr=16000)
messages = [{
    "role": "user",
    "content": [
        {"type": "audio", "audio": audio},
        {"type": "text", "text": "What sounds do you hear, and what's happening?"},
    ],
}]
inputs = processor.apply_chat_template(messages, tokenize=True, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(output[0], skip_special_tokens=True))
```

### 步骤 2：投影器模式

```python
import torch.nn as nn

class AudioProjector(nn.Module):
    def __init__(self, audio_dim=1280, llm_dim=4096):
        super().__init__()
        self.down = nn.Linear(audio_dim, llm_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(llm_dim, llm_dim)

    def forward(self, audio_features):
        return self.up(self.act(self.down(audio_features)))
```

就是这样。投影器通常是 1-3 个线性层。在 ASR 对上训练它（音频 → 转录）是阶段 1 的预训练任务。

### 步骤 3：基准测试 MMAU / LongAudioBench

```python
from datasets import load_dataset
mmau = load_dataset("MMAU/MMAU-Pro")

correct = 0
for item in mmau["test"]:
    answer = call_model(item["audio"], item["question"], item["choices"])
    if answer == item["correct_choice"]:
        correct += 1
print(f"Accuracy: {correct / len(mmau['test']):.3f}")
```

分别报告每类别（语音 / 声音 / 音乐 / 多音频）。聚合数字隐藏模型失败的位置。

## 使用它

| 任务 | 2026 年选择 |
|------|-----------|
| 自由形式音频 QA（开源） | Qwen2.5-Omni-7B |
| 长音频最佳开源 | Audio Flamingo Next |
| 最佳封闭 | Gemini 2.5 Pro |
| 语音输入 / 语音输出智能体 | Qwen2.5-Omni 或 GPT-4o Audio |
| 音乐推理 | Audio Flamingo 3 或 2（音乐专用 AF-CLAP） |
| 呼叫中心审计 | 通过 API 的 Gemini 2.5 Pro，带你的政策文档上的 RAG |

## 陷阱

- **对多音频的过度信任。** 如果你的任务需要"哪个片段有 X"，随机水平性能是真实的。
- **长音频退化。** 超过 10 分钟，大多数模型的说话人属性中断。先说话人分割（第 6 课），然后摘要。
- **静音上的幻觉。** 使用 Whisper 编码器的 LALM 继承了相同的 Whisper 式问题。VAD 把关。
- **基准 cherry-picking。** 供应商博文突出最佳情况类别。自己运行 MMAU-Pro 多音频子集。

## 交付它

保存为 `outputs/skill-alm-picker.md`。为给定的音频理解任务选择 LALM + 基准子集 + 输出模态（文本 vs 语音）。

## 练习

1. **简单。** 运行 `code/main.py` 以查看玩具投影器模式 + (audio-embedding, text-tokens) → output tokens 的虚假 LALM 路由。
2. **中等。** 在 100 个 MMAU-Pro 语音项目上评分 Qwen2.5-Omni-7B。与论文报告的数字比较。
3. **困难。** 构建最小音频字幕基线：BEATs 编码器 + 2 层投影器 + 冻结的 Llama-3.2-1B。仅在 AudioCaps 上微调投影器。在 Clotho-AQA 上与 SALMONN 比较。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| LALM | 音频 ChatGPT | 音频编码器 + 投影器 + LLM 解码器。 |
| 投影器 | 适配器 | 将音频特征映射到 LLM 嵌入空间的小型 MLP。 |
| MMAU | 基准测试 | 跨语音、声音、音乐的 10k 音频-QA 对。 |
| MMAU-Pro | 更难的 MMAU | 1800 个多音频 / 推理重度问题。 |
| LongAudioBench | 长格式评估 | 带语义查询的多分钟片段。 |
| 语音输入 / 语音输出 | 语音原生 | 模型摄入语音并发出语音，无需文本绕路。 |

## 扩展阅读

- [Chu et al. (2024). Qwen2-Audio](https://arxiv.org/abs/2407.10759) — 参考架构。
- [Alibaba (2025). Qwen2.5-Omni](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) — 语音输入-语音输出。
- [NVIDIA (2025). Audio Flamingo 3](https://arxiv.org/abs/2507.08128) — 开源长音频领袖。
- [NVIDIA (2026). Audio Flamingo Next](https://arxiv.org/abs/2604.10905) — LongAudioBench 最先进。
- [Tang et al. (2023). SALMONN](https://arxiv.org/abs/2310.13289) — 双编码器先驱。
- [MMAU-Pro leaderboard](https://mmaubenchmark.github.io/) — 实时 2026 排名。
