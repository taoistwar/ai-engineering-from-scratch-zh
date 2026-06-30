# 音频生成

> 音频是 16-48 kHz 的 1-D 信号。五秒的剪辑是 80-240k 个样本。没有任何 transformer 直接关注那个序列。2026 年每个生产音频模型的解决方案是相同的：一个神经编解码器（Encodec、SoundStream、DAC）以 50-75 Hz 将音频压缩为离散 token，然后一个 transformer 或扩散模型生成 token。

**类型：** 构建
**语言：** Python
**前置条件：** 第六阶段 · 02（音频特征），第六阶段 · 04（ASR），第八阶段 · 06（DDPM）
**时间：** 约 45 分钟

## 问题

三种音频生成任务：

1. **文本到语音。** 给定文本，产生语音。干净语音是窄带且有强语音结构——由 token 上的 transformer 很好地解决。VALL-E（微软）、NaturalSpeech 3、ElevenLabs、OpenAI TTS。
2. **音乐生成。** 给定提示（文本、旋律、和弦进行、流派），产生音乐。分布广阔得多。MusicGen（Meta）、Stable Audio 2.5、Suno v4、Udio、Riffusion。
3. **音频效果 / 声音设计。** 给定提示，产生环境声或 Foley。AudioGen、AudioLDM 2、Stable Audio Open。

所有三种在相同基质上运行：神经音频编解码器 + token-AR 或扩散生成器。

## 概念

![音频生成：编解码器 token + transformer 或扩散](../assets/audio-generation.svg)

### 神经音频编解码器

Encodec（Meta，2022）、SoundStream（Google，2021）、Descript Audio Codec（DAC，2023）。卷积编码器将波形压缩为每时间步向量；残差向量量化（RVQ）将每个向量转换为 K 个码本索引的级联。解码器反转它。24 kHz 音频以 2 kbps 使用 8 个 RVQ 码本，75 Hz = 600 tokens/sec。

```
waveform (16000 samples/sec)
    └─ encoder conv ─┐
                     ├─ RVQ layer 1 → indices at 75 Hz
                     ├─ RVQ layer 2 → indices at 75 Hz
                     ├─ ...
                     └─ RVQ layer 8
```

### 两种生成范式在其上

**Token 自回归。** 将 RVQ token 展平为序列，运行仅解码器 transformer。MusicGen 使用"延迟并行"以每流偏移并行发出 K 个码本流。VALL-E 从文本提示 + 3 秒语音样本生成语音 token。

**潜在扩散。** 将编解码器 token 打包为连续潜在向量或用类别扩散建模。Stable Audio 2.5 在连续音频潜在向量上使用流匹配。AudioLDM 2 使用文本到 mel 到音频扩散。

2024-2026 趋势：流匹配在音乐上获胜（更快的推理，更干净的样本），而 token-AR 仍然主导语音，因为它是自然因果的且流式传输良好。

## 生产格局

| 系统 | 任务 | 骨干 | 延迟 |
|--------|------|----------|---------|
| ElevenLabs V3 | TTS | Token-AR + neural vocoder | ~300ms 首个 token |
| OpenAI GPT-4o audio | 全双工语音 | 端到端多模态 AR | ~200ms |
| NaturalSpeech 3 | TTS | 潜在流匹配 | 非流式 |
| Stable Audio 2.5 | 音乐 / SFX | DiT + 音频潜在向量上的流匹配 | 1 分钟剪辑约 10s |
| Suno v4 | 完整歌曲 | 未公开；怀疑是 token-AR | 每首歌约 30s |
| Udio v1.5 | 完整歌曲 | 未公开 | 每首歌约 30s |
| MusicGen 3.3B | 音乐 | Encodec 32kHz 上的 Token-AR | 实时 |
| AudioCraft 2 | 音乐 + SFX | 流匹配 | 5s 剪辑约 5s |
| Riffusion v2 | 音乐 | 频谱图扩散 | 约 10s |

## 动手构建

`code/main.py` 模拟核心思想：在从两种不同"风格"生成的合成"音频 token"序列上训练微型下一个 token transformer（风格 A 为交替低和高 token，风格 B 为单调斜坡）。以风格为条件并采样。

### 步骤 1：合成音频 token

```python
def make_tokens(style, length, vocab_size, rng):
    if style == 0:  # "类语音"：交替
        return [i % vocab_size for i in range(length)]
    # "类音乐"：斜坡
    return [(i * 3) % vocab_size for i in range(length)]
```

### 步骤 2：训练微型 token 预测器

以风格为条件的二元组风格预测器。重点在于模式：编解码器 token → 交叉熵训练 → 自回归采样。

### 步骤 3：条件采样

给定风格 token 和起始 token，从预测分布中采样下一个 token。继续 20-40 个 token。

## 缺陷陷阱

- **编解码器质量限制输出质量。** 如果编解码器不能忠实地表示声音，再多的生成器质量也无帮助。DAC 是当前开源最佳。
- **RVQ 误差累积。** 每个 RVQ 层建模前一层的残差。第 1 层的误差传播。在更高层以温度 0 采样有帮助。
- **音乐结构。** 在 75 Hz 下 30 秒的 token 是 20k+ token。对 transformer 来说很难。MusicGen 使用滑动窗口 + 提示续写；Stable Audio 使用更短剪辑 + 交叉淡入淡出。
- **边界伪影。** 生成剪辑之间的交叉淡入淡出需要仔细 overlap-add。
- **干净数据需求。** 音乐生成器需要数千小时许可音乐。Suno / Udio RIAA 诉讼（2024）将此问题浮出水面。
- **声音克隆伦理。** 3 秒样本加文本提示足以让 VALL-E / XTTS / ElevenLabs 克隆声音。每个生产模型都需要滥用检测 + 退出名单。

## 使用它

| 任务 | 2026 年技术栈 |
|------|------------|
| 商业 TTS | ElevenLabs、OpenAI TTS 或 Azure Neural |
| 声音克隆（已确认同意） | XTTS v2（开源）或 ElevenLabs Pro |
| 背景音乐，快速 | Stable Audio 2.5 API、Suno 或 Udio |
| 带歌词的音乐 | Suno v4 或 Udio v1.5 |
| 音效 / Foley | AudioCraft 2、ElevenLabs SFX 或 Stable Audio Open |
| 实时语音代理 | GPT-4o realtime 或 Gemini Live |
| 开源权重音乐研究 | MusicGen 3.3B、Stable Audio Open 1.0、AudioLDM 2 |
| 配音 / 翻译 | HeyGen、ElevenLabs Dubbing |

## 交付成果

保存 `outputs/skill-audio-brief.md`。技能接收音频需求（任务、时长、风格、声音、许可）并输出：模型 + 托管、提示格式（流派标签、风格描述符、结构标记）、编解码器 + 生成器 + 声码器链、种子协议和评估计划（MOS / CLAP 分数 / 对 TTS 的 CER / 用户 A/B）。

## 练习

1. **简单。** 运行 `code/main.py` 并显式设置风格。验证生成的序列匹配风格模式。
2. **中等。** 添加延迟并行解码：模拟 2 个必须保持偏移 1 步的 token 流。训练联合预测器。
3. **困难。** 使用 HuggingFace transformers 在本地运行 MusicGen-small。用三种不同提示生成 10 秒剪辑；A/B 风格遵循度。

## 关键术语

| 术语 | 人们怎么说 | 它实际上的含义 |
|------|-----------------|-----------------------|
| 编解码器 | "神经压缩" | 音频的编码器 / 解码器；典型输出是 50-75 Hz token。 |
| RVQ | "残差 VQ" | K 个量化器的级联；每个建模前一层的残差。 |
| Token | "一个编解码器符号" | 码本中的离散索引；通常 1024 或 2048。 |
| 延迟并行 | "偏移码本" | 以交错偏移发出 K 个 token 流以减少序列长度。 |
| 流匹配 | "2024 年音频的胜利" | 扩散的更直路径替代；更快采样。 |
| 声音提示 | "3 秒样本" | 引导克隆声音的说话人嵌入或 token 前缀。 |
| Mel 频谱图 | "视觉表示" | 对数幅度感知频谱图；被许多 TTS 系统使用。 |
| 声码器 | "Mel 到波形" | 将 mel 频谱图转换回音频的神经组件。 |

## 生产笔记：音频是一个流式问题

音频是用户期望*在生成时*到达、而非一次性全部到达的唯一输出模态。在生产术语中，这意味着 TPOT 很重要（每个输出 token 的时间），因为用户的收听速度是目标吞吐量——不是他们的阅读速度。对于以约 75 tokens/秒进行分词的 16kHz 音频（Encodec），服务器必须每用户生成 ≥75 tokens/秒以保持播放平滑。

两个架构后果：

- **流匹配音频模型不能简单地流式传输。** Stable Audio 2.5 和 AudioCraft 2 在一次传递中渲染固定剪辑长度。要流式传输，你分块剪辑并重叠边界——想想滑动窗口扩散——与编解码器 AR 模型相比增加了 100-300ms 的延迟开销。

如果产品是"实时语音聊天"或"实时音乐续写"，选择编解码器 AR 路径。如果是"在提交时渲染 30 秒剪辑"，流匹配在质量和总延迟上获胜。

## 延伸阅读

- [Défossez 等人 (2022). Encodec: High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — 编解码器标准。
- [Zeghidour 等人 (2021). SoundStream](https://arxiv.org/abs/2107.03312) — 第一个被广泛使用的神经音频编解码器。
- [Kumar 等人 (2023). High-Fidelity Audio Compression with Improved RVQGAN (DAC)](https://arxiv.org/abs/2306.06546) — DAC。
- [Wang 等人 (2023). Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers (VALL-E)](https://arxiv.org/abs/2301.02111) — VALL-E。
- [Copet 等人 (2023). Simple and Controllable Music Generation (MusicGen)](https://arxiv.org/abs/2306.05284) — MusicGen。
- [Liu 等人 (2023). AudioLDM 2: Learning Holistic Audio Generation with Self-supervised Pretraining](https://arxiv.org/abs/2308.05734) — AudioLDM 2。
- [Stability AI (2024). Stable Audio 2.5](https://stability.ai/news/introducing-stable-audio-2-5) — 2025 年使用流匹配的文本到音乐生成。
