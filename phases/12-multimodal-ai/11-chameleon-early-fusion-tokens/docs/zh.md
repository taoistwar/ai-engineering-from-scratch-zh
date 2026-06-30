# Chameleon 与早期融合的纯 Token 多模态模型

> 我们到目前为止看到的每个 VLM 都将图像和文本分开处理。视觉 token 来自视觉编码器，流入投影器，然后在 LLM 内部与文本交汇。视觉和文本的词表从不重叠。Chameleon（Meta，2024 年 5 月）问：如果它们重叠会怎样？训练一个将图像转化为共享词表中离散 token 序列的 VQ-VAE。现在每个多模态文档都是一个序列——文本 token 和图像 token 交错排列，一个单一的自回归损失。副作用：模型可以生成混合模态的输出——在单次推理调用中交替输出文本和图像 token。本课阅读早期融合的论点，并从头到尾构建一个玩具版本。

**Type:** Build
**Languages:** Python (stdlib, VQ-VAE tokenizer + interleaved decoder)
**Prerequisites:** Phase 12 · 05, Phase 8 (Generative AI)
**Time:** ~180 minutes

## 学习目标

- 解释为什么共享词表 + 单一损失改变了模型的能力。
- 描述 VQ-VAE 如何将图像转换为与 Transformer 下一 token 目标兼容的离散序列。
- 列举 Chameleon 的训练稳定性技巧：QK-Norm、dropout 位置、LayerNorm 排序。
- 比较 Chameleon 与 BLIP-2 的 Q-Former 方法，描述每种方法什么时候是正确的选择。

## 问题

基于适配器的 VLM（LLaVA、BLIP-2、Qwen-VL）将文本和图像视为两种不同的东西。文本 token 经过 `embed(text_token)`；图像经过 `visual_encoder(image) → projector → ... pseudo_tokens`。模型有两条输入路径，在中途汇合。

三个后果：

1. LLM 只能消费图像，不能输出图像。输出仅为文本。
2. 混合模态文档（交替的段落和图像，如文章中的那样）很尴尬——你要么在模型外部解析多模态输入，要么链式生成。
3. 分布不匹配。视觉 token 和文本 token 存在于隐藏空间的不同区域，产生了微妙的对齐问题。

Chameleon 拒绝了这个前提：图像只是来自共享词表的离散 token 序列。在交错文档上训练模型，一个损失，一个自回归解码器，你就免费解锁了混合模态生成。

## 概念

### VQ-VAE 作为图像分词器

该分词器是一个向量量化的变分自编码器。架构：

- 编码器：CNN + ViT，将图像映射到空间特征图，比如 dim 256 的 32x32 特征。
- 码本：K 个向量的可学习词表（Chameleon 使用 8192），也是 dim 256。
- 量化：对每个空间特征，通过 L2 距离查找最近的码本条。用整数索引替换连续特征。
- 解码器：CNN，将量化特征还原为像素。

训练：VAE 重建损失 + commitment 损失 + 码本损失。码本索引形成了图像的离散字母表。

对于 Chameleon：一张图像变为 32*32 = 1024 个 token，来自一个 8192 的词表。与文本 token 连接（来自 LLM 的 BPE 词表，假设 32000）。最终词表：40192。Transformer 看到一个序列，一个损失。

### 共享词表

Chameleon 的词表组合了文本 token、图像 token 和模态分隔符。每个 token 有单一 ID。输入嵌入层将每个 ID 映射到 D 维隐藏向量。输出投影将隐藏状态映射回词表 logits。Softmax 选择下一个 token，无论是什么模态。

分隔符很重要：`<image>` 和 `</image>` 标签括起图像 token 序列。在生成时，如果模型发出 `<image>`，下游软件就知道接下来的 1024 个 token 是 VQ 索引，要送入解码器进行像素渲染。

### 混合模态生成

推理是在共享词表中进行下一 token 预测。示例提示："Draw a cat and describe it." Chameleon 输出：

```
<image> 4821 1029 2891 ...（1024 个图像 token）</image>
The cat is orange, sitting on a windowsill...
```

模型自主选择顺序——它可以生成图像然后文本，文本然后图像，或交错排列。相同的解码器，相同的损失。

与适配器式 VLM 的对比，生成仅限于文本。Chameleon 重新开启了模型输出模态的问题。

### 训练稳定性——QK-Norm、Dropout、LayerNorm 排序

早期融合训练在大规模下不稳定。Chameleon 论文记录了三个技巧：

- QK-Norm。在注意力内部，对查询和键投影应用 LayerNorm，在点积之前进行。防止深层处的 logit 幅度爆炸。多个 2024 年后的大模型使用。
- Dropout 位置。在每个残差加法之后进行 Dropout，而不仅仅在注意力和 MLP 之后。当来自图像 token 的梯度可能主导时需要更多的正则化。
- LayerNorm 排序。残差分支上的 Pre-LN（标准），加上最后一个块的跳跃连接上的额外 LN。稳定最后一层的梯度流。

没有这些技巧，34B 参数的 Chameleon 训练在多个检查点处发散。有了它们，训练收敛。训练配方与架构本身同样是贡献。

### 分词器的重建上限

VQ-VAE 是有损的。在 8192 个码本条和每张 512x512 图像 1024 个 token 的情况下，重建 PSNR 上限约在 26-28 dB。这对可识别的图像生成来说足够了，但明显比连续空间扩散（Stable Diffusion 3 达到 32+ dB）差。

分词器是瓶颈。更好的分词器（MAGVIT-v2、IBQ、SBER-MoVQGAN）提升了上限。Emu3（第 12.12 课）仅通过更好的分词器就达到了 SDXL 质量的生成。

### Chameleon vs BLIP-2 / LLaVA

Chameleon（早期融合，共享词表）：
- 一个损失，一个解码器。
- 生成混合模态输出。
- 分词器是质量上限。
- 昂贵：推理路径上每张生成的图像需要 VQ-VAE 解码器。

BLIP-2 / LLaVA（后期融合，独立塔）：
- 仅视觉输入，文本输出。
- 复用了预训练的 LLM。
- 理解能力上没有分词器瓶颈。
- 便宜：单次前向传播。

按任务选择。如果你需要图像生成，选 Chameleon 家族。如果你只需要理解能力，适配器式 VLM 更简单，复用了更多的预训练计算。

### Fuyu 与 AnyGPT

Fuyu（Adept，2023）是一种相关的方法：完全跳过单独的视觉编码器，将原始图像 patch 通过 LLM 的输入投影输入，就像它们是 token 一样，无需分词器。比 Chameleon 简单，但失去了共享词表的输出生成能力。

AnyGPT（Zhan 等人，2024）将 Chameleon 扩展到四种模态：文本、图像、语音、音乐。每种使用相同的 VQ-VAE 技巧，共享 Transformer。任意模态到任意模态的生成。在第 12.16 课中有更多介绍。

## 使用

`code/main.py` 构建了一个玩具端到端早期融合模型：

- 一个微型 VQ-VAE 风格量化器，将 8x8 patch 映射到码本索引（K=16）。
- 一个共享词表：（文本 id 0..31）+（图像 id 32..47）+（分隔符 48、49）。
- 一个玩具自回归解码器（二元组表），在合成标题 + 图像 token 序列上进行训练。
- 采样循环，根据提示交替发出文本 + 图像 token。

代码有意将 Transformer 保持为微型（二元组），这样你可以从头到尾追踪信号流。

## 产出

本课产出 `outputs/skill-tokenizer-vs-adapter-picker.md`。给定产品规格（仅理解 vs 理解 + 生成、所需图像质量、成本预算），它在 Chameleon 家族（早期融合）和 LLaVA 家族（后期融合）之间做出选择，并用定量的经验法则解释原因。

## 练习

1. Chameleon 使用 K=8192 个码本条，每张 512x512 图像 1024 个 token。估算与 24 位 RGB 图像相比的压缩比。它是有损的吗？有多大的损失？

2. 一张 4K 图像（3840x2160）在相同的 VQ-VAE 密度下产生多少个图像 token？一个 Chameleon 风格的模型能否在单次推理调用中生成 4K 图像？什么最先崩溃——上下文、分词器质量还是 KV 缓存？

3. 用纯 Python 实现 QK-Norm。给定一个 64 维查询和键，展示 LayerNorm 前后的点积。为什么幅度控制在深层很重要？

4. 阅读 Chameleon 第 2.3 节关于训练稳定性。描述论文在 34B 时没有 QK-Norm 情况下观察到的确切故障模式。"范数爆炸"的签名是什么？

5. 扩展玩具解码器，在给定纯文本提示的情况下发出混合模态响应。给定训练数据分布为 60% 文本优先 / 40% 图像优先，测量模型选择图像优先 vs 文本优先的频率。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 早期融合 | "统一 token" | 将图像从第一步开始转换为共享 Transformer 词表的离散 token |
| VQ-VAE | "图像分词器" | CNN + ViT + 码本，将图像映射为 Transformer 可以预测的整数索引 |
| 共享词表 | "一个字典" | 涵盖文本 + 图像 + 模态分隔符的单一 token ID 空间 |
| QK-Norm | "注意力稳定器" | 在点积之前对查询和键应用 LayerNorm，防止范数爆炸 |
| 混合模态生成 | "文本 + 图像输出" | 在一次传递中自主产生交错文本和图像 token 的推理 |
| 码本大小 | "K 个条目" | VQ-VAE 可以量化的离散向量数量；在压缩和保真度之间权衡 |
| 分词器上限 | "重建极限" | 解码 VQ token 可达的最佳 PSNR；限制了模型的图像质量 |

## 拓展阅读

- [Chameleon Team — Chameleon: Mixed-Modal Early-Fusion Foundation Models (arXiv:2405.09818)](https://arxiv.org/abs/2405.09818)
- [Aghajanyan et al. — CM3 (arXiv:2201.07520)](https://arxiv.org/abs/2201.07520)
- [Yu et al. — CM3Leon (arXiv:2309.02591)](https://arxiv.org/abs/2309.02591)
- [Zhan et al. — AnyGPT (arXiv:2402.12226)](https://arxiv.org/abs/2402.12226)
- [Adept — Fuyu-8B blog (adept.ai)](https://www.adept.ai/blog/fuyu-8b)
