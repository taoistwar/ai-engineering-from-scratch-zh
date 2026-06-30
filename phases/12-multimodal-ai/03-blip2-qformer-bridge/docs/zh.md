# 从 CLIP 到 BLIP-2 —— Q-Former 作为模态桥梁

> CLIP 对齐了图像和文本，但无法生成标题、回答问题或进行对话。BLIP-2（Salesforce，2023）用一个可训练的小型桥梁解决了这个问题：32 个可学习查询向量通过交叉注意力关注冻结的 ViT 特征，然后直接插入到冻结的 LLM 输入流中。188M 参数的桥梁将一个 11B 的 LLM 连接到一个 ViT-g/14 上。直到 2026 年，每一款基于适配器的 VLM——MiniGPT-4、InstructBLIP、LLaVA 的衍生模型——都是它的后代。本课阅读 Q-Former 的架构，解释其两阶段训练，并构建一个玩具版本将视觉 token 馈入冻结的文本解码器。

**Type:** Build
**Languages:** Python (stdlib, cross-attention + learnable-query demo)
**Prerequisites:** Phase 12 · 02 (CLIP), Phase 7 (Transformers)
**Time:** ~180 minutes

## 学习目标

- 解释为什么在冻结的视觉编码器和冻结的 LLM 之间使用可训练的瓶颈层，在成本和稳定性上优于端到端微调。
- 实现一个交叉注意力块，其中一组固定的可学习查询关注外部图像特征。
- 走过 BLIP-2 的两阶段预训练：表示学习（ITC + ITM + ITG）然后生成学习（冻结解码器的 LM 损失）。
- 将 Q-Former 与 LLaVA 中使用的更简单的 MLP 投影器进行比较，并论证每种选择何时占优。

## 问题

你有一个冻结的 ViT，每张图像产生 256 个 dim 1408 的 patch token。你有一个冻结的 7B LLM，期望 dim 4096 的 token 嵌入。最直观的桥梁——从 1408 到 4096 的线性层——能工作，但将全部 256 个 patch token 输入 LLM 上下文每张图像要消耗 256 个额外 token。在 32 张图像的 batch 下，仅视觉模态就消耗了 8192 个 token。

BLIP-2 的问题是：能否将 256 个 token 的图像表示压缩到少得多的 token（比如 32 个），同时保留足够信息让 LLM 能生成标题、回答问题和推理图像内容？以及能否在不触及冻结骨干网络的情况下训练此桥梁，使训练成本仅为桥梁的参数？

答案是：Q-Former。32 个可学习的"查询"向量对 ViT 的 patch token 进行交叉注意力，产生一个 32 个 token 的视觉摘要，供 LLM 消费。总计 188M 参数。在与 LLM 接触之前，先用对比、匹配和生成目标进行训练。

## 概念

### 可学习查询

Q-Former 的核心技巧：不让 LLM 的文本 token 直接关注图像 patch，而是引入一组新的 32 个可学习查询向量 `Q`，并让*它们*关注图像 patch。查询是模型的参数——它们在训练中被学习，相同的 32 个查询用于每一张图像。

交叉注意力后，每个查询持有一个图像压缩摘要——"描述主要物体"、"描述背景"、"计数物体"等。查询并不会按语义标签明确特化；它们学习任何能使下游损失下降的编码方式。

### 架构

Q-Former 是一个小型 Transformer（12 层，~100M 参数），有两条路径：

1. 查询路径：32 个查询向量流经自注意力（在它们之间），然后对冻结的 ViT 的 patch token 进行交叉注意力，最后是 FFN。
2. 文本路径：一个类似 BERT 的文本编码器与查询路径共享自注意力和 FFN 权重。文本路径禁用交叉注意力。

训练时两条路径同时运行。查询和文本通过共享的自注意力进行交互，这意味着查询可以根据任务所需的文本进行条件化（ITM、ITG）。在 VLM 移交的推理阶段，只有查询流经，产生 32 个视觉 token。

### 两阶段训练

BLIP-2 分两个阶段进行预训练：

阶段 1：表示学习（不涉及 LLM）。三个损失：
- ITC（图像-文本对比）：在池化后的查询 token 与文本 CLS token 之间的 CLIP 风格对比。
- ITM（图像-文本匹配）：二分类器——这组图文对是否匹配？使用硬负样本挖掘。
- ITG（图像条件文本生成）：在文本上使用因果 LM 头，以查询为条件。强制查询编码可生成文本的内容。

仅训练 Q-Former。ViT 冻结。不涉及 LLM。

阶段 2：生成学习。附加一个冻结的 LLM（OPT-2.7B 或 Flan-T5-XL 等）。通过一个小型线性层将 32 个查询输出投影到 LLM 的嵌入维度。将它们前置到文本提示之前。仅在拼接后的提示 + 图像 + 标题序列上训练线性投影和 Q-Former，损失为 LM 损失。

阶段 2 之后，Q-Former + 投影就是完整的视觉适配器。在推理时：图像 → ViT → Q-Former → 线性投影 → 前置到文本 → 冻结的 LLM 输出。

### 参数经济学

BLIP-2 使用 ViT-g/14（1.1B，冻结）+ OPT-6.7B（6.7B，冻结）+ Q-Former（188M，训练）= 总计 8B，188M 接受训练。Q-Former 单独仅占整个栈参数的 ~2.4%。训练成本反映了这一点：几天时间在几块 A100 上 vs 端到端需要数周。

质量：BLIP-2 在零样本 VQA 上匹配或击败 Flamingo-80B，同时体积小 50 倍。桥梁确实有效。

### InstructBLIP 与指令感知的 Q-Former

InstructBLIP（2023）扩展了 Q-Former，增加了一个额外输入：指令文本本身。在交叉注意力时，查询现在可以同时访问图像 patch 和指令。查询可以按每条指令进行特化（"数有多少辆车"、"描述氛围"），而不是学习一个固定的单一摘要。在保留任务上获得了基准收益。

### MiniGPT-4 与仅投影器的方法

MiniGPT-4 保留了 Q-Former，但只训练输出线性投影层，其他全部冻结。便宜，但代价是质量——查询是 BLIP-2 的，不是你的。适合快速迭代，但不是最佳架构。

### 为什么 LLaVA 选择了更简单的方式

LLaVA（2023，第 12.05 课）用简单的 2 层 MLP 替换了 Q-Former，将每个 ViT patch token 投影到 LLM 空间——24x24 网格每张图像 576 个 token，全部馈入 LLM。压缩更差，但让 LLM 直接关注原始 patch。这在当时是有争议的；到 2023 年底它成了主流，因为视觉指令数据（LLaVA-Instruct-150k）证明了 MLP 可以通过训练保留足够的信号。权衡：LLaVA 的上下文填满得更快，但它自然地扩展到多图像和视频。

到 2026 年，领域出现了分化：Q-Former 在 token 预算紧张时存活（长视频、多图像）；MLP 投影器在每 token 原始质量优先时占主导。

### 门控交叉注意力：Flamingo，祖先

Flamingo（第 12.04 课）早于 BLIP-2，使用了相同的交叉注意力思想，但在每个冻结的 LLM 层上都使用，而不是作为单一桥梁。BLIP-2 证明了你可以在只输入层上进行压缩，并且仍然有效。Gemini 和 Idefics 结合了两者：交错输入 token 加上可选的上下文少样本门控交叉注意力。

### 2026 年衍生模型

- Q-Former：BLIP-2、InstructBLIP、MiniGPT-4，以及大多数因 token 预算原因使用的视频语言模型。
- Perceiver 重采样器：Flamingo 的变体（第 12.04 课）；Idefics 系列、Eagle、OmniMAE。
- MLP 投影器：LLaVA、LLaVA-NeXT、LLaVA-OneVision、Cambrian-1。
- 注意力池化：VILA、PaliGemma。

四种方案都是可行的。决定性问题在于你受 token 预算约束还是受每 token 质量约束。

## 使用

`code/main.py` 构建了一个标准库 Q-Former 风格的交叉注意力：

1. 模拟 256 个图像 patch token（dim 128）。
2. 实例化 32 个可学习查询（dim 128）。
3. 运行缩放点积交叉注意力（Q 来自查询，K/V 来自 patch）。
4. 通过一个线性层投影到 LLM 维度（512）。
5. 输出 32 个 LLM 就绪的视觉 token。

所有数学运算使用纯 Python（向量上的嵌套循环）。玩具级但形状正确。打印注意力权重矩阵，让你看到每个查询从哪些 patch 提取信息。

## 产出

本课产出 `outputs/skill-modality-bridge-picker.md`。给定一个目标 VLM 配置（视觉编码器 token 数、LLM 上下文预算、部署约束、质量目标），它推荐 Q-Former vs MLP vs Perceiver 重采样器，并附有简短理由和每种桥梁的参数计数估算。

## 练习

1. 在 PyTorch 中实现交叉注意力块。验证使用 32 个查询和 256 个键/值时，注意力权重矩阵为 32×256，且 softmax 后每行之和为 1。

2. 在 BLIP-2 阶段 1 中，Q-Former 同时运行三个损失：ITC、ITM、ITG。用伪代码为每个编写前向签名。哪一个需要文本编码器路径处于活动状态？

3. 比较参数计数：Q-Former（12 层，768 隐藏）vs 2 层 MLP 投影器（1408 → 4096，两层）。在什么 LLM 规模下，188M 的 Q-Former 成本在训练效率上是值得的？

4. 阅读 BLIP-2 论文（arXiv:2301.12597）第 3.2 节关于 Q-Former 如何初始化。解释为什么从 BERT-base（而非随机）初始化能加速收敛。

5. 对于一段 10 分钟的视频，以 1 FPS 采样到 60 帧，计算每帧 token 成本：使用 Q-Former → 32 tokens/帧 vs 使用 MLP 投影器 → 576 tokens/帧。哪种方案能放入 128k token 的 LLM 上下文窗口？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Q-Former | "查询 Transformer" | 小型 Transformer，具有 32 个可学习查询向量，对冻结的 ViT 特征进行交叉注意力 |
| 可学习查询 | "视觉的软提示" | 一组固定参数，作为交叉注意力的查询端；按模型学习，对所有输入共享 |
| 交叉注意力 | "Q 来自这边，K/V 来自那边" | 查询、键和值来自不同源的注意力机制；查询如何从 ViT patch 提取信息 |
| ITC | "图像-文本对比" | 应用于 Q-Former 池化查询 vs 文本 CLS 的 CLIP 风格损失 |
| ITM | "图像-文本匹配" | 在硬负样本挖掘配对上的二分类器；强制查询辨别细粒度不匹配 |
| ITG | "图像条件文本生成" | 因果 LM 损失，以查询为条件生成文本；强制查询编码可被文本解码的内容 |
| 两阶段预训练 | "表示然后生成" | 阶段 1 单独训练 Q-Former（ITC/ITM/ITG）；阶段 2 附加冻结 LLM，仅训练投影 + Q-Former |
| 冻结骨干网络 | "不微调" | 视觉编码器和 LLM 的权重固定不变；只有桥梁接受训练 |
| 投影头 | "线性投影到 LLM 维度" | 将 Q-Former 输出映射到 LLM 嵌入维度的最终线性层 |
| Perceiver 重采样器 | "Flamingo 版本" | 类似的可学习查询交叉注意力，Flamingo 在每一层使用，而非作为单一桥梁 |

## 拓展阅读

- [Li et al. — BLIP-2 (arXiv:2301.12597)](https://arxiv.org/abs/2301.12597) —— 核心论文。
- [Li et al. — BLIP (arXiv:2201.12086)](https://arxiv.org/abs/2201.12086) —— 前身，具有 ITC/ITM/ITG 三者。
- [Li et al. — ALBEF (arXiv:2107.07651)](https://arxiv.org/abs/2107.07651) —— "先对齐再融合"——阶段 1 训练的概念祖先。
- [Dai et al. — InstructBLIP (arXiv:2305.06500)](https://arxiv.org/abs/2305.06500) —— 指令感知的 Q-Former。
- [Zhu et al. — MiniGPT-4 (arXiv:2304.10592)](https://arxiv.org/abs/2304.10592) —— 仅投影器的方法。
- [Jaegle et al. — Perceiver IO (arXiv:2107.14795)](https://arxiv.org/abs/2107.14795) —— 可学习查询交叉注意力的通用架构。
