# LLaVA 与视觉指令微调

> LLaVA（2023 年 4 月）是地球上被复制最多的多模态架构。它用 2 层 MLP 替换了 BLIP-2 的 Q-Former，用朴素的 token 拼接替换了 Flamingo 的门控交叉注意力，并在 GPT-4 从纯文本标题生成的 158k 视觉指令对话轮次上进行训练。2023 年至 2026 年间构建 VLM 的每一位从业者都构建了某种 LLaVA 变体。LLaVA-1.5 添加了 AnyRes。LLaVA-NeXT 提升了分辨率。LLaVA-OneVision 将单图像、多图像和视频统一到了一套配方中。本课阅读该配方，实现投影器，并解释为什么"更简单的赢了"。

**Type:** Build
**Languages:** Python (stdlib, projector + instruction-template builder)
**Prerequisites:** Phase 12 · 02 (CLIP), Phase 11 (LLM Engineering — instruction tuning)
**Time:** ~180 minutes

## 学习目标

- 构建一个将 ViT patch 嵌入（dim 1024）映射到 LLM 嵌入维度（dim 4096）的 2 层 MLP 投影器。
- 走过 LLaVA 两阶段配方：(1) 在 558k 标题对上投影器对齐，(2) 在 158k GPT-4 生成的对话轮次上视觉指令微调。
- 构建一个带有图像 token 占位符、系统提示和用户/助手对话轮次的 LLaVA 格式提示。
- 解释为什么社区从 Q-Former 转向了 MLP，尽管 Q-Former 在 token 预算上占优。

## 问题

BLIP-2 的 Q-Former（第 12.03 课）将一张图像压缩到 32 个 token。干净、高效，对基准测试友好。但它有两个问题。

首先，Q-Former 是可训练的，但其损失并非最终任务。阶段 1 训练 ITC+ITM+ITG。阶段 2 训练 LM 损失。查询学习了某种中间表示，然后 LLM 必须解码它。信息在瓶颈中丢失。

其次，Q-Former 占用 188M 参数，在 LLaVA 的 2023 年规模下，你必须为目标 LLM 协同设计它。更换 LLM，重新训练 Q-Former。更换视觉编码器，重新训练。每种组合都是一个独立的研发项目。

LLaVA 的答案简单到令人尴尬：取 ViT 的 576 个 patch token，将每个通过一个 2 层 MLP（`1024 → 4096 → 4096`），然后将全部 576 个转储到 LLM 的输入序列中。没有瓶颈。没有在奇怪目标上的阶段 1 预训练。只需在直接的 LM 损失上训练 MLP。

数据从哪里来？LLaVA 的第二个洞见：使用 GPT-4（纯文本）来生成指令数据。将图像的 COCO 标题和边界框数据输入 GPT-4，要求它生成对话、描述和复杂推理问题。158k 条指令-响应对话轮次，免费获得。无需人工标注。

结果：一个在 8 块 A100 上运行一天就训练完成的 VLM，在 MMMU 上击败了 Flamingo，并发布了一个社区可以扩展的开源检查点。到 2023 年底，它已经催生了 50+ 个分支。

## 概念

### 架构

LLaVA-1.5 的 13B 版本：
- 视觉编码器：CLIP ViT-L/14 @ 336（阶段 1 冻结，阶段 2 可选解冻）。
- 投影器：带 GELU 激活的 2 层 MLP，`1024 → 4096 → 4096`。
- LLM：Vicuna-13B（后为 Llama-3.1-8B）。

在图像 + 文本提示上的前向传播：

```
img -> ViT -> 576 个 dim 1024 的 patch
patches -> MLP -> 576 个 dim 4096 的 token
prompt: system + "<image>" 占位符 + 用户问题
用 576 个投影后的 token 替换 <image> token
将完整序列馈入 LLM
解码响应
```

图像占据 LLM 上下文的 576 个 token。在 2048 上下文中，剩下 1472 个 token 给文本。在 32k 上下文中，这只是一个舍入误差。

### 阶段 1：投影器对齐

冻结 ViT。冻结 LLM。仅训练 2 层 MLP。数据集：558k 图像-标题对（LAION-CC-SBU）。损失：基于投影图像 token 条件的标题语言建模。

在 batch 128 下单个 epoch 数小时内完成。投影器学会将 ViT 空间映射到 LLM 空间。没有任务特定的监督。

### 阶段 2：视觉指令微调

解冻投影器（仍然可训练）。解冻 LLM（通常完全解冻，有时使用 LoRA）。在 158k 视觉指令对话轮次上训练。

指令数据是诀窍所在。Liu 等人通过以下方式生成：
1. 取一张 COCO 图像。
2. 提取文本描述（5 条人工标题 + 边界框列表）。
3. 使用三种提示模板发送给 GPT-4：
   - 对话："生成用户和助手之间关于这张图像的来回对话。"
   - 详细描述："给出图像丰富、详细的描述。"
   - 复杂推理："提出一个需要推理图像的问题，然后回答。"
4. 将 GPT-4 的输出解析为（指令，响应）对。

这些都不直接触及图像——只有文本描述。GPT-4 虚构了合理的图像内容。有些噪音，但它成功了：158k 轮次足以解锁对话能力。

### 社区为什么复制这个

- 无需调整阶段 1 特定的损失。全程使用 LM 损失。
- 投影器在数小时而非数天完成训练。
- LLM 可以更换（LLaVA-Llama2、LLaVA-Mistral、LLaVA-Llama3），只需重新训练投影器。
- 视觉指令数据管道使用 GPT-4，为新领域重新生成成本低廉。

### LLaVA-1.5 与 LLaVA-NeXT

LLaVA-1.5（2023 年 10 月）新增：
- 将学术任务数据（VQA、OKVQA、RefCOCO）混合到指令微调中。
- 更好的系统提示。
- 2048 → 32k 上下文。

LLaVA-NeXT（2024 年 1 月）新增：
- AnyRes：将高分辨率图像分割成 2x2 或 1x3 的 336x336 裁剪网格，外加一个全局低分辨率缩略图。每个裁剪产生 576 个 token；总计每张图像约 2880 个视觉 token。OCR 和图表任务大幅跃升。
- 更好的指令数据混合，使用 ShareGPT4V（高质量 GPT-4V 标题）。
- 更强的基础 LLM（Mistral-7B、Yi-34B）。

### LLaVA-OneVision

第 12.08 课将深入介绍 OneVision。简而言之：相同的投影器，但使用涵盖单图像、多图像和视频的课程进行训练，在单个模型中共享视觉 token 预算。

### 与 Q-Former 的比较

| | Q-Former (BLIP-2) | MLP (LLaVA) |
|---|---|---|
| 每张图像视觉 token | 32 | 576（基础）或 2880（AnyRes） |
| 可训练参数 | 188M + LM | 40M + LM |
| 阶段 1 损失 | ITC+ITM+ITG | 仅 LM |
| LLM 即插即用 | 需要重新训练 | 通过最少重新训练即可切换 |
| 多图像 | 笨拙 | 自然（拼接） |
| 视频 | 笨拙 | 自然（逐帧拼接） |
| Token 预算 | 小 | 大 |

MLP 在简单性和 token 灵活性上胜出。Q-Former 在 token 预算上胜出。到 2023 年底，token 预算不再是约束性瓶颈（LLM 上下文增长到 32k-128k+），简单性主导了战场。

### 提示格式

```
A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image> Describe this image in detail. ASSISTANT: The image shows ...
```

`<image>` 是一个占位符 token。在分词之前，它被替换为 576 个视觉 token（或 AnyRes 下的 2880 个）。分词器看到一个比训练时略长的序列，但 LLM 处理这种新输入，因为阶段 1 教会了它。

### 参数经济学

LLaVA-1.5-7B 分解：
- CLIP ViT-L/14 @ 336：303M（阶段 1 冻结，阶段 2 经常解冻）。
- 投影器（2x 线性）：~22M 可训练。
- Llama-7B：7B。
- 总计：7.3B 参数。阶段 2 期间可训练：全部 7B + 22M 投影器。

阶段 2 的训练成本：8x A100 约 20 小时。这是关键数字——一天，一个节点，可复现。这就是 LLaVA 传播开来的原因。

## 使用

`code/main.py` 实现了：

1. 纯 Python 的 2 层 MLP 投影器（玩具规模，dim 16 → 32 → 32）。
2. 提示构建流水线：系统提示 + 用 N 个投影 token 替换的 `<image>` + 用户对话轮次 + 助手生成占位符。
3. 一个可视化工具，展示 576 个 token 的视觉块在 LLM 上下文中占多大比例（2k / 32k / 128k 上下文消耗的百分比）。

## 产出

本课产出 `outputs/skill-llava-vibes-eval.md`。给定一个 LLaVA 系列的检查点，它运行一个 10 提示的 vibe 评估套件（3 个标题生成、3 个 VQA、2 个推理、2 个拒绝）并报告一份人类可读的记分卡。不是基准测试；是一个烟雾测试，确认投影器和 LLM 连接良好。

## 练习

1. 计算 2 层 MLP 投影器在 `1024 → 4096 → 4096` 下的可训练参数计数。带有 GELU 和偏置，它占 LLaVA-13B 的百分比是多少？

2. 为一个"拒绝"场景构建一个 LLaVA 提示——图像包含私人个体。编写期望的助手响应。为什么 LLaVA 应该对此进行零样本拒绝，以及需要什么训练数据来强化这种拒绝？

3. 阅读 LLaVA-NeXT 博客的 AnyRes 部分。计算一张 1344x672 图像在 AnyRes 下的视觉 token 数。与 336x336 下的基础 576 个 token 进行比较。

4. LLaVA 阶段 1 投影器使用 LM 损失对标题进行训练。如果跳过阶段 1 直接进入阶段 2（视觉指令微调）会发生什么？引用 Prismatic VLMs 消融研究（arXiv:2402.07865）来回答。

5. LLaVA-Instruct-150k 使用 GPT-4 和 COCO 标题生成指令。对于一个新领域（医学 X 光、卫星图像），描述生成领域指令的四步数据管道。每一步可能出现什么问题？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 投影器 | "MLP 桥梁" | 2 层 MLP，带 GELU，将 ViT 维度映射到 LLM 维度 |
| 图像 token | "<image> 占位符" | 提示标记，在推理前被 N 个投影视觉 token 替换 |
| 视觉指令微调 | "LLaVA 阶段 2" | 在 GPT-4 生成的（图像，指令，响应）三元组上的训练 |
| 阶段 1 对齐 | "投影器预训练" | 冻结 ViT 和 LLM，在标题上使用 LM 损失训练投影器 |
| AnyRes | "多裁剪平铺" | 将高分辨率图像分割成平铺网格，并拼接每个平铺的视觉 token |
| LLaVA-Instruct | "GPT-4 生成" | 从 COCO 标题 + GPT-4 合成的 158k 条指令-响应对 |
| 视觉编码器冻结 | "骨干网络锁定" | CLIP 权重在阶段 1 不更新，有时在阶段 2 也不更新 |
| ShareGPT4V | "更好的标题" | GPT-4V 生成的 1M 密集标题，用于更高质量的对齐 |
| VQA | "视觉问答" | 回答关于图像的开放式问题的任务 |
| Prismatic VLMs | "设计空间论文" | Karamcheti 2024 消融研究，系统性地测试投影器和数据选择 |

## 拓展阅读

- [Liu et al. — Visual Instruction Tuning (arXiv:2304.08485)](https://arxiv.org/abs/2304.08485) —— LLaVA 论文。
- [Liu et al. — Improved Baselines with Visual Instruction Tuning (arXiv:2310.03744)](https://arxiv.org/abs/2310.03744) —— LLaVA-1.5。
- [Chen et al. — ShareGPT4V (arXiv:2311.12793)](https://arxiv.org/abs/2311.12793) —— 密集标题数据集。
- [Karamcheti et al. — Prismatic VLMs (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865) —— 设计空间消融。
- [Li et al. — LLaVA-OneVision (arXiv:2408.03326)](https://arxiv.org/abs/2408.03326) —— 统一的单图像、多图像、视频。
