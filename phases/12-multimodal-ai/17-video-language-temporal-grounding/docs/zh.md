# 视频-语言模型：时间 Token 与定位

> 视频不是照片的堆叠。一个 5 秒的片段具有因果顺序、动作动词和图像模型无法表示的事件计时。Video-LLaMA（Zhang 等人，2023 年 6 月）发布了第一个具有视听定位能力的开源视频-LLM。VideoChat 和 Video-LLaVA 扩展了该模式。到 2025 年，Qwen2.5-VL 的 TMRoPE 缩小了与前沿专有模型的差距。每个系统以不同方式解决时间 token 问题——每片段 Q-former、每帧拼接池化、每 token TMRoPE。本课阅读这些模式，构建一个均匀 vs 动态帧采样器，并在时间定位任务上进行评估。

**Type:** Build
**Languages:** Python (stdlib, frame sampler + temporal-grounding evaluator)
**Prerequisites:** Phase 12 · 08 (LLaVA-OneVision)
**Time:** ~180 minutes

## 学习目标

- 解释为什么时间位置编码会独立于视觉编码器改变视频 VLM 的性能。
- 在每秒 token 数 vs 定位准确率方面比较均匀、动态 FPS 和事件驱动的帧采样。
- 描述每片段 Q-former（Video-LLaMA）vs 每帧池化（Video-LLaVA）vs 每 token M-RoPE（Qwen2.5-VL）的设计。
- 列举四个视频基准：VideoMME、TempCompass、EgoSchema、Video-MMMU。

## 问题

一段 1 分钟 30 FPS 的视频有 1800 帧。每帧 196 个视觉 token（ViT-B @ 224），总计 352k 个 token——比任何 2024 时代的 LLM 上下文都大。

存在三种缩减策略：

1. 对帧进行子采样（根据内容 1-8 FPS）。
2. 对每帧的 patch token 进行激进的池化（3x3 或 4x4 双线性池化）。
3. 通过 Q-former 压缩，该 Q-former 接受 16 帧片段并输出 64 个 token。

每种权衡不同。子采样会损失时间细节。池化会损失空间细节。Q-former 两者都损失一点但节省 token。

时间位置编码是另一个轴：模型如何知道第 5 帧在第 6 帧之前？选项包括简单的 1D 时间 RoPE（Video-LLaMA）、可学习的时间嵌入（Video-LLaVA）和 TMRoPE（Qwen2.5-VL，完整的 3D）。

## 概念

### Video-LLaMA：每片段 Q-former + 音频分支

Video-LLaMA（2023）是第一个开源的视频-LLM。架构：

- 以 2 FPS 采样的 16 帧片段（即 8 秒）。
- 每帧 ViT 特征 → 对所有 16 帧进行交叉注意力的 Video Q-former → 32 个可学习查询 → LLM。
- 并行音频分支：波形 → ImageBind 音频编码器 → Audio Q-former → 32 个查询 → LLM。

优势：音视频联合推理。弱点：片段长度固定，无法进行任意时间定位。

### VideoChat 与 Video-LLaVA

VideoChat 保留了 Video-LLaMA 的思想但放弃了音频并简化。Video-LLaVA（Lin 等人，2023）在图像和视频帧上训练了单个视觉编码器（"投影前对齐"），给出统一表示。两者都是冻结 CLIP 编码器 + MLP + LLM。

两者都不能处理长视频。两者都是 8-16 帧系统。

### Qwen2.5-VL 与 TMRoPE

Qwen2.5-VL 引入了 TMRoPE——时间-模态旋转位置嵌入（Temporal-Modality Rotary Position Embedding）。每个 patch token 携带一个（t, h, w）位置，其中 t 是实际时间戳（而非帧索引）。

与简单时间嵌入的关键区别：

- 绝对时间，而非索引。模型看到的是"在 4.2 秒"而不是"在第 15 帧"。
- 每 token 旋转，而非每片段。每个视觉 token 按其时间戳独立旋转。
- 与动态 FPS 兼容。如果你在这里以 2 FPS 采样，在那里以 4 FPS 采样，TMRoPE 原生处理不均匀的间隔。

TMRoPE 使"猫在什么秒数跳跃？"这样的查询成为可能。模型可以输出"在 4.2 秒"。Video-LLaMA 只能说"在片段的早期"。

### 帧采样策略

均匀：在持续时间内均匀采样 N 帧。简单，但会丢失运动峰值。

动态 FPS：根据运动强度自适应采样。光流或帧差分选择高运动段进行更密集的采样。Qwen2.5-VL 在此上训练。

事件驱动：运行轻量级检测器，在动作发生的地方采样更多。VideoAgent 使用。

关键帧 + 上下文：在镜头边界 + 一些相邻帧处采样。用于电影内容。

### 每帧池化

在 1 FPS 和每帧 576 个 token 下，5 分钟的片段是 172,800 个 token。Qwen2.5-VL-72B 的 128k 上下文可行但昂贵。

3x3 双线性池化减少了每帧 64 个 token → 5 分钟 19,200 个 token。对大多数任务是最佳点。

更激进的池化（6x6 → 每帧 16 个 token）用于空间细节不太重要的代理工作流程。

### 四个视频基准

- VideoMME：全面的视频理解，短 + 中 + 长。
- TempCompass：细粒度时间推理，"之前"/"之后"类问题。
- EgoSchema：长时第一人称视频。
- Video-MMMU：多模态多学科视频问题。

一个完整的视频-VLM 评估应涵盖全部四个。它们分别强调不同的轴——TempCompass 完全关于排序，EgoSchema 关于 3+ 分钟推理，VideoMME 跨越不同时长。

### 定位输出格式

时间定位的输出格式：

- 自由文本："猫大约在 4 秒标记时跳跃。"容易解析但不精确。
- 结构化 JSON：`{"event": "jump", "start": 4.1, "end": 4.3}`。Qwen2.5-VL 训练此格式。
- 基于 Token：特殊的 `<time>4.1</time>` token 与答案交错排列。Qwen2.5-VL 的内部格式。

基于 Token 的格式对下游使用最准确。Qwen2.5-VL 的 JSON 输出格式可直接解析。

### 2026 年最佳实践

2026 年视频 VLM 的最佳实践：

- 编码器：带有 M-RoPE 或 TMRoPE 的 SigLIP 2（Qwen2.5-VL）。
- 帧采样：动态 FPS（根据运动 1-4），带最大帧上限。
- 每帧池化：3x3 双线性。
- 输出：带时间和事件字段的结构化 JSON。
- 基准：通用用 VideoMME + TempCompass；长时用 EgoSchema。

## 使用

`code/main.py` 包含：

- 均匀和动态 FPS 帧采样器。
- 一个玩具时间定位评估器：给定在时间 T 的"真实"事件和模型输出，以容差评估准确率。
- Video-LLaMA（16 帧，Q-former）、Video-LLaVA（8 帧，MLP）、Qwen2.5-VL（动态 FPS + TMRoPE）之间的对比。

## 产出

本课产出 `outputs/skill-video-vlm-frame-planner.md`。给定一个视频任务（监控、动作识别、时间定位、摘要），它选择帧采样器、池化因子、输出格式和预期准确率层级。

## 练习

1. 对于一个 3 分钟的烹饪演示，选择均匀 vs 动态 FPS。用 token 计数论证。

2. TMRoPE 具体添加了什么简单时间嵌入表无法做到的事？

3. 编写一个 VLM 可以学习发出的时间定位 JSON schema。包括错误情况。

4. 阅读 Video-LLaVA 关于"投影前对齐"的第 3 节。为什么这比训练独立的图像和视频编码器更好？

5. 给定 VideoMME 排行榜，截至 2026 年，顶级开源模型和顶级专有模型之间的差距是多少？该差距有多大程度可归因于时间编码 vs 基础 LLM 规模？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 时间定位 | "时间本地化答案" | VLM 为事件发生输出特定的时间戳范围 |
| TMRoPE | "时间-多模态 RoPE" | 具有绝对时间戳的 3D 旋转位置，Qwen2.5-VL 使用 |
| 动态 FPS | "运动感知采样" | 在高运动段采样更多帧，在静态段采样更少 |
| 帧池化 | "每帧空间压缩" | 在 LLM 前用双线性插值减少每帧的 patch 数 |
| Video Q-former | "片段压缩器" | 将 N 帧映射到 K 个可学习查询的交叉注意力瓶颈 |
| VideoMME | "视频基准" | 全面的短/中/长视频基准，2500+ 样本 |

## 拓展阅读

- [Zhang et al. — Video-LLaMA (arXiv:2306.02858)](https://arxiv.org/abs/2306.02858)
- [Li et al. — VideoChat (arXiv:2305.06355)](https://arxiv.org/abs/2305.06355)
- [Lin et al. — Video-LLaVA (arXiv:2311.10122)](https://arxiv.org/abs/2311.10122)
- [Qwen Team — Qwen2.5-VL (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
- [Lin et al. — VILA-1.5 (arXiv:2312.07533)](https://arxiv.org/abs/2312.07533)
