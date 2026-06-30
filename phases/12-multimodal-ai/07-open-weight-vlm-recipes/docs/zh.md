# 开源 VLM 配方：真正重要的因素

> 2024-2026 年的开源 VLM 文献是一片消融表森林。Apple 的 MM1 测试了 13 种图像编码器、连接器和数据混合的组合。Allen AI 的 Molmo 证明了详细人工标注标题优于 GPT-4V 蒸馏。Cambrian-1 运行了 20+ 种编码器对比。Idefics2 形式化了五轴设计空间。Prismatic VLMs 在受控基准上比较了 27 种训练配方的。从所有这些噪音中，一小部分结果在论文之间保持一致：图像编码器比连接器架构更重要，数据混合比两者都重要，而详细人工标注标题优于蒸馏的合成数据。本课阅读这些表格，这样你就不必自己读了。

**Type:** Learn + lab
**Languages:** Python (stdlib, ablation table parser + recipe picker)
**Prerequisites:** Phase 12 · 05 (LLaVA baseline)
**Time:** ~180 minutes

## 学习目标

- 列举五轴 VLM 设计空间：图像编码器、连接器、LLM、数据混合、分辨率策略。
- 阅读 MM1 / Idefics2 / Cambrian-1 消融表，预测哪个旋钮会推动特定基准。
- 在给定计算预算和任务混合的情况下，为新的 VLM 选择配方（编码器、连接器、数据、分辨率）。
- 解释为什么在相同 token 数下，详细人工标题优于 GPT-4V 蒸馏。

## 问题

存在数百个开源 VLM。"好"和"最先进"之间的大部分差距不在于架构。而在于数据、分辨率策略和编码器选择。当你的模型表现不佳时，知道首先转动哪个旋钮，可以让你避免一次耗费 500 万 GPU 小时的错误。

2023 年浪潮（LLaVA-1.5、InstructBLIP、MiniGPT-4）使用标题对预训练 + LLaVA-Instruct-150k。不错的基线。MMMU 得分最多在 35% 左右。

2024 年浪潮（MM1、Idefics2、Molmo、Cambrian-1、Prismatic VLMs）运行了详尽的消融。结果令人惊讶且实用。

## 概念

### 五轴设计空间

Idefics2（Laurençon 等人，2024）命名了这些轴：

1. 图像编码器。CLIP ViT-L/14、SigLIP SO400m/14、DINOv2 ViT-g/14、InternViT-6B。编码器在 patch 大小、分辨率和预训练目标上有所不同。
2. 连接器。MLP（2-4 层）、Q-Former（32 查询 + 交叉注意力）、Perceiver 重采样器（64 查询）、C-Abstractor（卷积 + 双线性池化）。
3. 语言模型。Llama-3 8B / 70B、Mistral 7B、Phi-3、Gemma-2、Qwen2.5。LLM 大小是主导的参数成本。
4. 训练数据。标题对（CC3M、LAION）、交错数据（OBELICS、MMC4）、指令数据（LLaVA-Instruct、ShareGPT4V、PixMo、Cauldron）。
5. 分辨率策略。固定 224/336/448、AnyRes、原生动态。训练期间递增或恒定。

每个生产级 VLM 在每个轴上做出选择。MMMU 分数的大部分方差由轴 1、4 和 5 解释——而不是你选择了哪种连接器。

### 轴 1：编码器 > 连接器

MM1 第 3.2 节表明：从 CLIP ViT-L/14 切换到 SigLIP SO400m/14 增加了 3+ 分 MMMU。将连接器从 MLP 切换到 Perceiver 重采样器增加不到 1 分。Idefics2 复制了这一结论：SigLIP > CLIP，在相同 token 数下 Q-Former ≈ MLP ≈ Perceiver。

Cambrian-1 的 "Cambrian Vision Encoders Match-Up"（Tong 等人，2024）在视觉中心基准（CV-Bench）上运行了 20+ 种编码器。排行榜顶端是 DINOv2 和 SigLIP 的混合；CLIP 处于中等水平；ImageBind 和 ViT-MAE 较低。CLIP ViT-L 到 DINOv2 ViT-g/14 的差距在 CV-Bench 上约为 5-7 分。

2026 年开源 VLM 的默认编码器是用于语义 + 密集特征的 SigLIP 2 SO400m/14，有时会与 DINOv2 ViT-g/14 特征拼接（Cambrian 的 "Spatial Vision Aggregator" 就是这样做的）。

### 轴 2：连接器设计不相上下

MM1、Idefics2、Prismatic 和 MM-Interleaved 都得出了相同的结论：在固定的视觉 token 数下，连接器架构几乎不重要。在相同的 token 预算下，对均值池化 patch 使用 2 层 MLP 的表现，与 32 查询 Q-Former 相差在 1 分以内。

真正重要的是 token 数。更多的视觉 token = 更多的 LLM 计算 = 更好的性能，直到一定程度后收益递减。每张图像 64 个 token 对 OCR 来说太少了。576-1024 个 token 是大多数开源 VLM 的最佳点。2048+ 仅对文档和图表有帮助。

Q-Former 与 MLP 是成本问题，不是质量问题：无论图像分辨率如何，Q-Former 将 token 限制在 32-64 个；MLP 发出所有 patch token。对于高分辨率输入，Q-Former 节省 LLM 上下文；对于低分辨率输入，差异是噪音。

### 轴 3：LLM 大小设定了上限

将 LLM 从 7B 翻倍到 13B 在每篇 VLM 论文中都能可靠地增加 2-4 分的 MMMU。在 70B 时，大多数基准达到饱和。VLM 的多模态推理上限是 LLM 的文本推理上限——视觉编码器只能为它提供信息，不能代替它推理。

这就是为什么 Qwen2.5-VL-72B 和 Claude Opus 4.7 能在 MMMU-Pro 和 ScreenSpot-Pro 上碾压：语言大脑非常巨大。一个 7B 的 VLM 无法通过巧妙的连接器设计替代 70B 的 VLM。

### 轴 4：数据——详细人工标题优于蒸馏

Molmo + PixMo（Deitke 等人，2024）是每个人都应该阅读的 2024 年成果。Allen AI 让人类标注者以 1-3 分钟的密集语音转文本方式描述图像，产生了 712K 张密集标注的图像。训练数据中没有任何 GPT-4V 蒸馏。

Molmo-72B 在 11 个基准中的 11 个上击败了 Llama-3.2-90B-Vision。差距不在于架构——而在于标题质量。详细的人工标题每张图像包含 5-10 倍于短网页标题的信息，并在 GPT-4V 蒸馏会产生幻觉的地方保持事实基础。

ShareGPT4V（Chen 等人，2023）和 Cauldron（Idefics2）遵循相同的策略，使用混合人工 + GPT-4V 标题。趋势很明确：对于 2026 年前沿，标题密度 > 标题数量 > 蒸馏便利性。

### 轴 5：分辨率及其策略

Idefics2 的消融：384 → 448 增加 1-2 分。448 → 980 使用图像分割（AnyRes）在 OCR 基准上再增加 3-5 分。平坦分辨率训练在中等准确率处达到平台期；分辨率递增（从 224 开始，到 448 或原生结束）训练更快且最终得分更高。

Cambrian-1 运行了分辨率 vs token 的权衡：在固定计算量下，你可以有较低分辨率下的更多 token，或较高分辨率下的更少 token。较高分辨率在 OCR 中胜出；较低分辨率更多 token 在通用场景理解中胜出。

2026 年生产配方：阶段 1 以固定 384 进行训练，阶段 2 使用动态分辨率最高到 1280，适用于 OCR 密集型任务。

### Prismatic 受控对比

Prismatic VLMs（Karamcheti 等人，2024）是控制了所有轴的论文。相同的 13B LLM，相同的指令数据，相同的评估——一次只变化一个轴。结果：

- 每张图像的视觉 token 数解释了约 60% 的方差。
- 编码器选择解释了约 20%。
- 连接器架构解释了约 5%。
- 其他所有（数据混合、调度器、学习率）解释了剩余的约 15%。

这是一个粗略的分解，但它是文献中对"我应该首先消融什么"最清晰的回答。

### 2026 年选择器

鉴于以上证据，2026 年新项目的默认开源 VLM 配方：

- 编码器：原生分辨率下的 SigLIP 2 SO400m/14 配合 NaFlex，如果需要分割/定位则与 DINOv2 ViT-g/14 拼接以获取密集特征。
- 连接器：在 patch token 上的 2 层 MLP。除非 token 受限，否则跳过 Q-Former。
- LLM：Qwen2.5 / Llama-3.1 / Gemma 2，7B 用于节省成本，70B 用于追求质量，根据目标延迟选择。
- 数据：PixMo + ShareGPT4V + Cauldron，并补充任务特定的指令数据。
- 分辨率：动态（长边 min 256，max 1280 像素）。
- 策略：阶段 1 对齐（仅投影器），阶段 2 全微调，阶段 3 任务特定微调。

每一个默认值都可以追溯到本课末尾引用的论文中的可测量消融。

## 使用

`code/main.py` 是一个消融表解析器和配方选择器。它编码了 MM1 和 Idefics2 的消融表（精简版），并允许你查询：

- "给定预算 X 和任务 Y，哪个配方会胜出？"
- "如果我在 7B Llama 上将 SigLIP 换成 CLIP，预期的 MMMU 变化是多少？"
- "对于 80% 置信度答案，我应该首先消融哪个轴？"

输出是一个带预期基准变化的排名配方列表，以及一个"首先消融"的建议。

## 产出

本课产出 `outputs/skill-vlm-recipe-picker.md`。给定目标任务混合、计算预算和延迟目标，它发出完整配方（编码器、连接器、LLM、数据混合、分辨率策略），并引用证明每种选择的消融。防止工程师在每次启动新 VLM 项目时重新发明 Idefics2 消融表。

## 练习

1. 阅读 MM1 第 3.2 节。对于预算 50M 图像的固定 2B LLM，哪种编码器会胜出？答案在 13B LLM 下是否会反转？为什么？

2. Cambrian-1 发现 DINOv2 + SigLIP 拼接在视觉中心基准上优于任何单独一个，但在 MMMU 上没有增加信号。预测哪些基准会受益，哪些保持不变。

3. 你的目标是在 2B LLM 上开发移动 UI 代理。选择编码器、连接器、分辨率和数据混合。用具体的消融表证明每个选择。

4. Molmo 发布了 4B 和 72B 模型。4B 版本与封闭的 7B VLM 具有竞争力；72B 版本在 11/11 基准上击败了 Llama-3.2-90B-Vision。这告诉你关于 LLM 大小平台期假说的什么信息？

5. 设计一个消融表，在 7B VLM 上隔离数据混合质量与编码器质量。至少需要多少次训练运行？提出四个轴的设置。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 消融 | "转动一个旋钮" | 多次训练运行，仅在一个设计空间轴上不同，其他一切保持不变 |
| 连接器 | "桥梁" / "投影器" | 将视觉编码器输出映射到 LLM token 空间的可训练模块（MLP、Q-Former、Perceiver） |
| 详细人工标题 | "密集标题" | 由人工编写的多句话描述（通常 80-300 个 token），比网页 alt 文本更丰富 |
| 蒸馏 | "GPT-4V 标题" | 由更强的专有 VLM 生成的训练数据；方便但容易继承幻觉 |
| AnyRes / 动态分辨率 | "高分辨率路径" | 通过平铺或 M-RoPE 将大于编码器本地分辨率的图像输入模型的策略 |
| 分辨率递增 | "课程" | 从低分辨率开始并逐步增加的训练时间表，加速对齐学习 |
| 视觉中心基准 | "CV-Bench / BLINK" | 强调细粒度视觉感知而非语言密集型推理的评估 |
| PixMo | "Molmo 的数据" | Allen AI 的 712K 密集标注图像数据集；人类语音转录为密集标题 |

## 拓展阅读

- [McKinzie et al. — MM1 (arXiv:2403.09611)](https://arxiv.org/abs/2403.09611)
- [Laurençon et al. — Idefics2 / What matters building VLMs (arXiv:2405.02246)](https://arxiv.org/abs/2405.02246)
- [Deitke et al. — Molmo and PixMo (arXiv:2409.17146)](https://arxiv.org/abs/2409.17146)
- [Tong et al. — Cambrian-1 (arXiv:2406.16860)](https://arxiv.org/abs/2406.16860)
- [Karamcheti et al. — Prismatic VLMs (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865)
