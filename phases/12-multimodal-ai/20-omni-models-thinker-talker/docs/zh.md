# Omni 模型：Qwen2.5-Omni 与 Thinker-Talker 分离

> GPT-4o 2024 年 5 月的产品演示之所以具有颠覆性，不是因为它底层的模型，而是因为产品的形态——一个语音界面，你可以说话，模型能看到摄像头看到的内容，并在 250ms 内回话。开源生态系统在 2024 年和 2025 年剩下的时间里竞相追赶这个产品形态。Qwen2.5-Omni（2025 年 3 月）是参考开源设计：一个 Thinker（大型文本生成 Transformer）加上一个 Talker（并行语音生成 Transformer），通过流式语音 token 连接。Mini-Omni 简化了它，Moshi 匹配了它的延迟，GLM-4-Voice 将其扩展到中文。本课阅读 Thinker-Talker 架构，以及使流式实时对话成为可能的延迟预算。

**Type:** Build
**Languages:** Python (stdlib, streaming pipeline latency simulator + VAD loop)
**Prerequisites:** Phase 12 · 19 (audio-LLMs), Phase 12 · 16 (any-to-any)
**Time:** ~180 minutes

## 学习目标

- 将推理流水线拆分为 Thinker（文本推理）和 Talker（语音合成），并解释为什么并行流式是有效的。
- 按组件依次计算对话交互的首个音频字节时间（TTFAB）预算。
- 描述 Thinker 内部跨视觉、音频和文本的 TMRoPE 时间对齐位置编码。
- 列举三种实时对话模式：半双工、轮流对话、全双工。

## 问题

实时语音助手必须快速做很多事情：

1. 听用户说话。实时语音分词，语音活动检测（VAD）以知道用户何时说完。
2. 可选地看。以 2-4 FPS 的摄像头输入，与音频一起流式传入 Thinker。
3. 思考。以对话历史为条件组成响应。
4. 说话。合成音频 token，解码为波形，流式传输到用户的扬声器。

每一步都会增加延迟。对话体验要求总往返时间 < 500ms——低于此阈值，用户就不再注意到延迟。GPT-4o 声称约 250ms。Moshi 约 160ms。Qwen2.5-Omni 约 350-500ms。

每个组件都需要流式。不能"先批量处理一切然后解码"。

## 概念

### Thinker 与 Talker

Qwen2.5-Omni 的分解：

- Thinker：一个 7B-80B 的文本生成 Transformer。消费交错的文本 + 图像 + 音频 token。输出表示要说什么的文本 token。
- Talker：一个更小的语音生成 Transformer（200M-1B）。消费 Thinker 的文本输出 token 加上最近的语音上下文 token。输出离散语音 token（residual-VQ 索引）。
- 语音解码器：一个流式波形解码器（SNAC、MoVQGAN 家族），实时将语音 token 转换为音频样本。

这种分离很重要。Thinker 必须大才能有好的推理能力。Talker 可以小，因为它的工作是局部的——将文本转换为语音 token。更大的 Talker 并不会更具表现力；只会更慢。

两者并行运行：

1. Thinker 发出文本 token t_i。
2. Talker 消费 t_i（通过流式）并发出语音 token s_i, s_{i+1}, ..., s_{i+k}。
3. 语音解码器在语音 token 到达时消费它们并发出音频样本。
4. 当 Thinker 到达文本 token t_{i+3} 时，Talker 已经为 t_0..t_{i+2} 流式传输了音频。

### TMRoPE——时间对齐的多模态位置

Thinker 需要整合以 4 FPS 到达的图像帧、以 50 帧/秒到达的音频帧以及对话历史中的文本。一个朴素的序列顺序（所有图像，然后所有音频，然后文本）会丢失时间对齐。

TMRoPE 为每个 token 分配绝对时间戳。t=2.3s 的视觉 token。t=2.32s 的音频 token。t=2.35s 的用户"停止"文本 token。RoPE 按时间戳旋转注意力；模型将它们视为时间上并发的。

这是让"他在说你好时挥手"能够工作的基础设施——模型在同一概念时刻看到视频帧和音频。

### 流式语音合成

语音 token 必须流式。Mini-Omni（Xie & Wu，2024）引入了"语言模型可以边思考边听和说"：Thinker 输出 token 和 Talker 输出 token 在同一序列中交错排列。Talker 在 Thinker 提交下一个文本 token 后就立刻启动。没有批处理边界。

Moshi（Défossez 等人，2024 年 10 月）是最快的开源实现。单 A100 上 160ms TTFAB。架构：一个单一的 7B Transformer，在交替位置发出文本和语音 token，具有将思考流与说话流分开的"内心独白"。这实际上是将 Thinker + Talker 融合到一个经过仔细训练的模型中。

### VAD 与轮流对话

语音活动检测在输入侧运行。两种模式：

- 半双工：用户说话，模型听。模型说话，用户听。通过 VAD 静音检测（~200ms）清晰交接。
- 全双工：两者可以同时说话。模型可以发出反馈音（"嗯哼"）或打断。困难得多。Moshi 支持这个。

Qwen2.5-Omni 默认支持半双工，通过静音阈值轮流对话。全双工需要应用层处理。

### Qwen3-Omni（2025 年 11 月）

继任者。Qwen3-80B Thinker，更大的 Talker，改进的 TMRoPE-v2。延迟接近 GPT-4o 的 250ms。开源权重。在 OmniBench 上的基准与 Gemini 2.0 Live 有竞争力。

### 生产延迟预算

对于典型的流式交互：

- 麦克风 → 音频 token：40-80ms。
- 预填充（提示 + 历史）：7B 约 100-200ms，70B 则远多于此。
- 首个 Thinker 文本 token：40ms。
- Talker 处理第一个文本 token：20ms。
- 首个语音 token 提交：40ms。
- Residual-VQ 解码：30ms。
- 语音波形解码：50-80ms。

总 TTFAB：7B 下 320-510ms，70B 下 600-900ms。前沿质量通常意味着 70B+；因此存在前沿延迟差距。

### Token 速率数学

在 16kHz 语音和 50 Hz 基础语音 token 下，每秒输出需要 50 个语音 token。Talker 必须发射 ≥50 tok/s 才能跟上。在 H100 上 LLM 典型吞吐量为 30-80 tok/s 的情况下，小型（200-300M）Talker 足够快；7B Talker 会落后。

这就是为什么存在小型专用 Talker 模型，而不是"直接用主模型"。

## 使用

`code/main.py`：

- 使用模拟 token 发射速率模拟 Thinker-Talker 流水线。
- 为可配置的模型大小和麦克风采样率计算 TTFAB。
- 演示使用 VAD 静音阈值的半双工轮流对话。

## 产出

本课产出 `outputs/skill-omni-streaming-budget.md`。给定实时语音产品的目标 TTFAB 和功能集（视觉输入、双语、全双工），选择 Qwen2.5-Omni、Qwen3-Omni、Moshi 或 Mini-Omni，并确定 Thinker/Talker 大小。

## 练习

1. 你的目标 TTFAB 是 300ms。在 7B Thinker 和 300M Talker 上，写出每个组件的延迟。

2. Qwen2.5-Omni 使用 TMRoPE。描述模型在用户 t=1s 开始说话且摄像头在 t=1.2s 捕捉到手势的提示下看到了什么。

3. 全双工支持要求模型在听的同时发出音频。提出一种教导此能力的训练数据格式。

4. 阅读 Moshi 论文第 4 节。描述"内心独白"分离以及它为什么避免了 Thinker-Talker 拆分的需要。

5. 计算吞吐预算：在 16kHz 语音和 50 基础层 token/秒下，Talker 必须以多快的速度发射 token 才能跟上？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Thinker | "推理大脑" | 产生"说什么"的大型文本生成 Transformer |
| Talker | "语音生成嘴" | 从 Thinker 的文本产生离散语音 token 的小型 Transformer |
| TTFAB | "延迟预算" | 首个音频字节时间：从用户语音结束到首个音频样本输出 |
| TMRoPE | "时间对齐 RoPE" | 跨视觉、音频、文本使用绝对时间戳的位置编码 |
| 半双工 | "轮流对话" | 用户和模型交替；VAD 静音检测用户完成 |
| 全双工 | "同时" | 模型可以同时说和听；支持反馈音 |
| 内心独白 | "Moshi 分离" | 单模型设计，思考流和说话流交错排列 |

## 拓展阅读

- [Xu et al. — Qwen2.5-Omni (arXiv:2503.20215)](https://arxiv.org/abs/2503.20215)
- [Qwen Team — Qwen3-Omni (arXiv:2509.17765)](https://arxiv.org/html/2509.17765v1)
- [Xie & Wu — Mini-Omni (arXiv:2408.16725)](https://arxiv.org/abs/2408.16725)
- [Défossez et al. — Moshi (arXiv:2410.00037)](https://arxiv.org/abs/2410.00037)
- [Zeng et al. — GLM-4-Voice (arXiv:2412.02612)](https://arxiv.org/abs/2412.02612)
