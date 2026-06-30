# 音频-语言模型：从 Whisper 到 Audio Flamingo 3 的演进弧线

> Whisper（Radford 等人，2022 年 12 月）解决了语音识别——68 万小时弱监督多语言语音、一个简单的编码器-解码器 Transformer、一个让此后每个 ASR 发布都引用它的基准。但识别不是推理。问"这段录音中有哪些乐器"或"说话人表达了什么情绪"或"第三分钟发生了什么"需要的是音频理解，而非转录。Qwen-Audio、SALMONN、LTU 和 NVIDIA 的 Audio Flamingo 3（AF3，2025 年 7 月）逐步构建了那个技术栈：保留 Whisper 类编码器，加上 Q-former，在音频-文本指令数据上训练，添加思维链推理。本课走完这条弧线。

**Type:** Build
**Languages:** Python (stdlib, log-Mel spectrogram + audio Q-former skeleton)
**Prerequisites:** Phase 6 (Speech and Audio), Phase 12 · 03 (Q-Former)
**Time:** ~180 minutes

## 学习目标

- 从波形计算 log-Mel 频谱图：加窗、FFT、滤波器组、对数变换。
- 比较编码器选项：Whisper 编码器、BEATs、AF-Whisper 混合。每种何时胜出。
- 构建音频 Q-former：N 个可学习查询对频谱图 patch 进行交叉注意力。
- 解释级联（先 Whisper 转录再 LLM）vs 端到端音频-LLM 训练：为什么端到端对推理的扩展性更好。

## 问题

语音识别已被 Whisper 解决。音频的 OCR 已成为商品。但"商品"在转录处结束。如果模型不能对听到的内容进行推理——时间、说话人、情绪、音乐结构、环境声音——仅靠转录无法驱动产品功能。

三条显而易见的路线：

1. 级联：Whisper 转录，LLM 对转录文本进行推理。适用于纯语音场景。对音乐、环境音频、多说话人重叠、情绪则失败。

2. 端到端音频-LLM：音频编码器直接将音频 token 馈送入 LLM，跳过转录。保留声学信息（情绪、说话人、环境）。需要新的训练数据。

3. 混合：音频编码器 + 文本解码器，既能转录又能推理。Qwen-Audio 和 Audio Flamingo 选择了这条路线。

## 概念

### Log-Mel 频谱图：输入特征

每个音频编码器都从相同的特征开始：log-Mel 频谱图。

1. 重采样至 16 kHz。
2. 使用 25ms 窗口、10ms 跳跃进行短时傅里叶变换。
3. 取 FFT 结果的幅度。
4. 应用 Mel 滤波器组（通常 80 个滤波器，对数间隔 0-8000 Hz）将频率扭曲为感知频率。
5. 对数压缩（log(1 + x)）以处理动态范围。

结果：一个形状为 (T, 80) 的二维数组，其中 T 是时间帧数。对于 30 秒的片段在 100 Hz 帧率下：(3000, 80)。

### Whisper 的编码器

Whisper 的编码器是一个 12 层 ViT 风格 Transformer，将 log-Mel 频谱图作为时间帧序列处理。输出：每个时间帧一个隐藏状态向量。

对于 ASR，Whisper 的解码器是一个交叉注意力 Transformer，以编码器输出为条件生成文本 token。标准编码器-解码器。

对于 ALM（音频-LLM），你希望编码器输出作为不同 LLM 的输入。模式：Whisper 编码器冻结，Q-former 可训练，LLM 冻结或微调。

### BEATs 和音频特定编码器

Whisper 在语音主导的数据上训练。对音乐和环境音频较弱。

BEATs（Chen 等人，2022）是在 AudioSet 上训练的自监督 Transformer。在相同参数数量下比 Whisper 更好地捕获音乐和环境声音。

AF-Whisper（Audio Flamingo 3 的混合）：将 Whisper + BEATs 特征拼接作为音频输入。Whisper 携带语言信号，BEATs 携带声学信号。

### 音频 Q-former

与 BLIP-2 的视觉 Q-former 模式相同。固定数量的可学习查询（通常 32 或 64 个）对音频编码器的输出帧进行交叉注意力。查询成为 LLM 消费的音频 token。

训练对齐阶段：仅 Q-former，在音频-文本对（AudioCaps、Clotho）上使用对比 + 标题生成损失。指令阶段：端到端，解冻 LLM，在指令数据上训练。

### 弧线——SALMONN、Qwen-Audio、AF3

SALMONN（Tang 等人，2023）：Whisper + BEATs + Q-former + LLaMA。第一个具有严肃推理能力的开源音频-LLM。在 MMAU 上的基准约为 0.55 的综合分。

Qwen-Audio（Chu 等人，2023）：类似架构，在更丰富的数据集上训练，针对多轮对话调优。MMAU 约 0.60。

LTU——Listen, Think, Understand（Gong 等人，2023）：显式推理数据，专注于音频片段的思维链。规模更小但更专注。

Audio Flamingo 3（Goel 等人，2025 年 7 月）：当前开源 SOTA。8B LLM 骨干网络（Qwen2 7B）、Whisper-large 编码器拼接 BEATs、64 查询 Q-former、在 1M+ 音频-文本指令对上训练。MMAU 0.72，在某些子任务上匹敌专有前沿模型。

AF3 还引入了音频的按需思维链：模型可以在最终答案之前可选地发出思考 token（"让我先识别乐器：……"）。开启思考时，复杂推理任务的准确率提升 3-5 分。

### 级联 vs 端到端

级联流水线：

1. Whisper 将音频转录为文本。
2. LLM 对文本进行推理。

对"摘要这个播客"完美有效。对以下情况失败：
- "这首歌的情绪是什么？"——情绪在声音中，不在词语中。
- "谁在说话，Alice 还是 Bob？"——需要说话人识别。
- "爆炸发生在第几秒？"——时间定位在文本中丢失。
- "这是真实音频还是生成的音频？"——深度伪造检测需要声学特征。

端到端保留声学信号。Qwen-Audio 和 AF3 原生处理音乐、环境和情绪。

### 2026 年生产配方

对于一个新的音频理解产品：

- 如果仅为转录目标、无音乐、无情绪推断：使用级联。
- 如果有音乐、情绪、多说话人或复杂音频推理：使用 AF3 / Qwen-Audio 家族。

级联更便宜、更简单。端到端能力更强。

### MMAU——音频推理基准

MMAU（大规模多模态音频理解）是 2024-2025 年的音频推理基准：

- 10,000 个音频-文本 QA 对，涵盖语音、音乐、环境声音。
- 涵盖分类、时间推理、因果推理、开放式 QA。
- 测试级联流水线系统性地遗漏的内容。

开源 SOTA（AF3）在 0.72；专有前沿约 0.78（Gemini 2.5 Pro、Claude Opus 4.7）。差距小于 VideoMME 的开源 vs 闭源差，表明音频-LLM 正在成熟。

## 使用

`code/main.py`：

- 用标准库实现 log-Mel 频谱图计算：加窗、朴素 DFT、Mel 滤波器组。
- 音频 Q-former 骨架：给定编码器输出帧，计算 Q、K、V、注意力，并发出 N 个 token。
- 在玩具任务上级联 vs 端到端比较。

## 产出

本课产出 `outputs/skill-audio-llm-pipeline-picker.md`。给定一个音频任务（转录、音乐标签、情绪推断、多说话人日记化、环境分类），在级联、端到端 AF3 或混合方式之间做出选择。

## 练习

1. 计算一段 30 秒片段在 16kHz、25ms 窗口、10ms 跳跃、80 Mel 区间下的 log-Mel 频谱图维度。在 48kHz 下有何变化？

2. 为什么 Whisper 在音乐上表现不佳？BEATs 捕获了哪些 Whisper 无法捕获的音频特征？

3. 音频 Q-former 使用 64 个查询 vs 32 个：在什么任务复杂度下 64 是值得的？32 个能节省什么计算？

4. 阅读 AF3 第 4 节关于按需思考。提出三个思维链最有帮助的音频任务。

5. 使用 AF3 的输出实现最小化的日记化管道。如何标记说话人变化？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Log-Mel 频谱图 | "Mel 特征" | Mel 滤波器组之后的对数幅度值的二维（时间，频率）数组 |
| 音频 Q-former | "音频 Perceiver" | 从音频编码器输出到馈入 LLM 的固定长度查询的交叉注意力瓶颈 |
| 级联 | "先 ASR 再 LLM" | Whisper 转录、文本 LLM 推理的流水线；会丢失声学信息 |
| 端到端 | "音频-LLM" | 音频特征通过 Q-former 直接进入 LLM；保留声学信号 |
| BEATs | "AudioSet 音频编码器" | 在 AudioSet 上训练的自监督 Transformer；在音乐 + 环境声音上表现强 |
| MMAU | "音频推理基准" | 涵盖语音、音乐、环境的 10k QA 对；2024 评估标准 |
| 按需思考 | "音频 CoT" | 模型可在最终答案前可选地发出推理 token，准确率提升 3-5 分 |

## 拓展阅读

- [Radford et al. — Whisper (arXiv:2212.04356)](https://arxiv.org/abs/2212.04356)
- [Chu et al. — Qwen-Audio (arXiv:2311.07919)](https://arxiv.org/abs/2311.07919)
- [Goel et al. — Audio Flamingo 3 (arXiv:2507.08128)](https://arxiv.org/abs/2507.08128)
- [Tang et al. — SALMONN (arXiv:2310.13289)](https://arxiv.org/abs/2310.13289)
- [Gong et al. — LTU (arXiv:2305.10790)](https://arxiv.org/abs/2305.10790)
