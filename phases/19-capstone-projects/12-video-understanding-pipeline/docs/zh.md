# 实践项目 12 — 视频理解管道（场景、问答、搜索）

> Twelve Labs 产品化了 Marengo + Pegasus。VideoDB 推出了视频 CRUD API。AI2 的 Molmo 2 发布了开放的 VLM 检查点。Gemini 长上下文原生处理数小时的视频。TimeLens-100K 定义了规模化的时间定位。2026 年的管道已经定型：场景分割、每个场景的标题 + 嵌入、转录对齐、多向量索引，以及带有 (start, end) 时间戳和帧预览的查询回答。实践项目是摄入 100 小时，命中公共基准，并测量在计数和动作问题上的幻觉。

**类型:** 实践项目
**语言:** Python（管道），TypeScript（UI）
**前置条件:** 阶段 4（CV），阶段 6（语音），阶段 7（transformers），阶段 11（LLM 工程），阶段 12（多模态），阶段 17（基础设施）
**涉及的阶段:** P4 · P6 · P7 · P11 · P12 · P17
**时间:** 30 小时

## 问题

长视频问答是 2026 年规模下最消耗带宽的多模态问题。Gemini 2.5 Pro 可以原生读取 2 小时的视频，但将 100 小时的视频摄入到可查询的语料库中仍然需要场景级索引。生产形态结合了场景分割（TransNetV2 或 PySceneDetect）、使用 VLM 的每个场景标题（Gemini 2.5、Qwen3-VL-Max 或 Molmo 2）、转录对齐（Whisper-v3-turbo 带字级时间戳），以及存储标题、帧嵌入和转录并排的多向量索引。查询管道以 (start, end) 时间戳加帧预览进行回答。

基准是公开的（ActivityNet-QA、NeXT-GQA）加上你自己的 100 个查询自定义集。计数和动作类型问题上的幻觉是已知的困难故障类；实践项目明确测量它。

## 概念

三条管道在摄入时并行运行。**场景分割**将视频切割为场景。**VLM 标题**为每个场景生成标题和关键帧嵌入。**ASR 对齐**生成字级时间戳。三条流按 (scene_id, time range) 连接。每个场景在多向量索引（Qdrant）中获得三种向量类型：标题嵌入、关键帧嵌入、转录嵌入。

在查询时，自然语言问题对三种向量都进行触发；结果使用 RRF 合并；一个时间定位适配器（TimeLens 风格）在顶部场景内细化 (start, end) 窗口。VLM 合成器（Gemini 2.5 Pro 或 Qwen3-VL-Max）接受查询 + 顶部场景 + 裁剪的帧，并使用引用的时间戳和帧预览进行回答。

幻觉测量很重要。计数（"有多少人进入房间？"）和动作类型（"厨师是在倒之前搅拌吗？"）问题是众所周知的不可靠。报告准确度要将其与描述性问题分开。

## 架构

```
video file / URL
      |
      v
PySceneDetect / TransNetV2  (scene segmentation)
      |
      +--- per-scene keyframe --- VLM caption + frame embedding
      |                            (Gemini 2.5 Pro / Qwen3-VL-Max / Molmo 2)
      |
      +--- audio channel --- Whisper-v3-turbo ASR + word timestamps
      |
      v
multi-vector Qdrant: {caption_emb, keyframe_emb, transcript_emb}
      |
query:
  dense queries against all three -> RRF merge -> top-k scenes
      |
      v
TimeLens / VideoITG temporal grounding (refine start/end within scene)
      |
      v
VLM synth: query + top scenes + frame previews
      |
      v
answer + (start, end) timestamps + frame thumbs + citations
```

## 技术栈

- 场景分割: TransNetV2（最先进 2024-26）或 PySceneDetect
- ASR: Whisper-v3-turbo via faster-whisper 带字级时间戳
- VLM 标题器 + 回答器: Gemini 2.5 Pro 或 Qwen3-VL-Max 或 Molmo 2
- 时间定位: TimeLens-100K 训练的适配器或 VideoITG
- 索引: Qdrant 带多向量支持（标题 / 帧 / 转录）
- UI: Next.js 15 带 HTML5 视频播放器和场景缩略图
- 评估: ActivityNet-QA、NeXT-GQA、自定义 100 个问题手动标注集
- 幻觉基准: 计数和动作类型子集带手动标签

## 构建它

1. **摄入遍历器。** 接受 YouTube URL 或本地 MP4。如需降级到 720p。持久化 `{video_id, file_path}`。

2. **场景分割。** 运行 TransNetV2 或 PySceneDetect 生成 `[{scene_id, start_ms, end_ms, keyframe_path}]`。目标 100 小时：约 6k-8k 个场景。

3. **ASR 传递。** 在音频上运行 Whisper-v3-turbo；导出字级时间戳；拆分为每个场景的转录切片。

4. **VLM 标题。** 每个场景，使用关键帧和简短标题模板调用 Gemini 2.5 Pro（或 Qwen3-VL-Max）。生成标题 + 帧嵌入。

5. **多向量索引。** Qdrant 集合包含三个命名向量。载荷：`{video_id, scene_id, start_ms, end_ms, keyframe_url}`。

6. **查询。** 自然语言问题触发三个稠密查询；使用倒数排名融合合并；top-k=5 个场景。

7. **时间定位。** 在顶部场景上运行 TimeLens 风格适配器以细化场景内的 (start, end) 窗口。

8. **VLM 合成。** 使用查询 + top-3 场景剪辑（作为图像或短视频） + 转录调用 Gemini 2.5 Pro。要求 `(video_id, start_ms, end_ms)` 引文。

9. **评估。** 运行 ActivityNet-QA 和 NeXT-GQA。构建 100 个查询自定义集。报告总体准确度 + 每类细分（计数、动作、描述）。

## 使用它

```
$ video-qa ask --url=https://youtube.com/watch?v=X "how many cars pass the intersection in the first minute?"
[scene]    23 scenes detected
[asr]      transcript complete, 4m12s
[index]    69 vectors written (23 scenes x 3)
[query]    top scene: scene 3 [01:32-01:54], confidence 0.84
[ground]   refined window: [00:12-00:58]
[synth]    gemini 2.5 pro, 1.4s
answer:    5 cars pass the intersection between 00:12 and 00:58.
citations: [scene 3: 00:12-00:58]
          [frame preview at 00:14, 00:27, 00:44, 00:51, 00:57]
```

## 交付它

`outputs/skill-video-qa.md` 是可交付成果。给定一个 YouTube URL 或上传的视频，管道索引场景并带有时间戳引文回答问题。

| 权重 | 标准 | 如何衡量 |
|:-:|---|---|
| 25 | 时间定位 IoU | 留存定位集上的交并比 |
| 20 | QA 准确度 | NeXT-GQA 和自定义 100 个问题 |
| 20 | 摄入吞吐量 | 每花费一美元的视频小时数 |
| 20 | UI 和引文 UX | 时间戳链接、缩略图条、跳转到帧 |
| 15 | 幻觉率 | 单独计算计数和动作类型准确度 |
| **100** | | |

## 练习

1. 将 Gemini 2.5 Pro 替换为 Qwen3-VL-Max 用于标题传递。在人工评分的 50 个场景样本上报告标题质量差值。

2. 将每个场景的帧嵌入减少到单个汇聚向量，而不是多向量。测量检索回归。

3. 构建"严格计数"模式：合成器提取每个计数的实例并附带时间戳，用户点击验证。测量用户验证是否减少幻觉。

4. 对摄入成本进行基准测试：在三个 VLM 选择下的每花费一美元的视频小时数。选出最优点。

5. 添加说话人分离转录：在音频上运行 pyannote 说话人分离，并嵌入每个说话人的转录。演示"Alice 说了什么关于 X？"的查询。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| Scene segmentation | "镜头检测" | 在镜头边界处将视频切割为场景 |
| Multi-vector index | "标题 + 帧 + 转录" | 每种表示具有命名向量的 Qdrant 集合 |
| Temporal grounding | "它到底什么时候发生的" | 细化查询答案的 (start, end) 窗口 |
| Frame embedding | "视觉表示" | 关键帧的向量嵌入；用于场景视觉相似性 |
| RRF fusion | "倒数排名融合" | 跨多个排序列表的合并策略；经典的混合检索技巧 |
| Counting hallucination | "数错" | VLM 在"有多少个 X"类问题上的已知故障模式 |
| ActivityNet-QA | "视频 QA 基准" | 长视频 QA 准确度基准 |

## 扩展阅读

- [AI2 Molmo 2](https://allenai.org/blog/molmo2) — 开放的 VLM 检查点
- [TimeLens (CVPR 2026)](https://github.com/TencentARC/TimeLens) — 规模化的时间定位
- [Gemini Video 长上下文](https://deepmind.google/technologies/gemini) — 托管参考
- [VideoDB](https://videodb.io) — 视频 CRUD API 参考
- [Twelve Labs Marengo + Pegasus](https://www.twelvelabs.io) — 商业参考
- [TransNetV2](https://github.com/soCzech/TransNetV2) — 场景分割模型
- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) — 经典开放替代方案
- [ActivityNet-QA](https://arxiv.org/abs/1906.02467) — 参考评估基准
