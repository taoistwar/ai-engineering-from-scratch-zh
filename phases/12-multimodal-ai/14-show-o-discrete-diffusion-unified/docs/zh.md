# Show-o 与离散扩散统一模型

> Transfusion 混合了连续和离散表示。Show-o（Xie 等人，2024 年 8 月）走了另一条路：文本 token 使用因果下一 token 预测，图像 token 使用 MaskGIT 精神的掩码离散扩散。两者位于一个带有混合注意力掩码的 Transformer 内。结果是在一个骨干网络、每种模态一个分词器、一个损失公式（下一 token 预测扩展到掩码预测）上统一了 VQA、文本到图像、修复和混合模态生成。本课走一遍 Show-o 的设计——为什么掩码离散扩散是一个并行的、少步的图像生成器——并与 Transfusion 和 Emu3 进行对比。

**Type:** Learn
**Languages:** Python (stdlib, masked-discrete-diffusion sampler)
**Prerequisites:** Phase 12 · 13 (Transfusion)
**Time:** ~120 minutes

## 学习目标

- 解释掩码离散扩散：均匀掩码 token 然后让 Transformer 恢复它们的调度。
- 在速度和质量上比较并行图像解码（Show-o、MaskGIT）与自回归图像解码（Chameleon、Emu3）。
- 列举 Show-o 在一个检查点中处理的三种任务：T2I、VQA、图像修复。
- 选择一个掩码调度（余弦、线性、截断），并推理其对样本质量的影响。

## 问题

Transfusion 的双损失训练可行但动力学更复杂——连续扩散损失与离散 NTP 损失在数值规模上不同。平衡损失权重是一项超参数搜索。架构有效但复杂。

Show-o 的答案：保持两种模态离散（如 Chameleon），但通过掩码离散扩散而非顺序地并行生成图像。训练目标变成单一的掩码 token 预测，自然地推广了下一 token 预测。

## 概念

### 掩码离散扩散（MaskGIT）

原始 Chang 等人（2022）的 MaskGIT 技巧很优雅。从完全掩码的图像开始（每个 token 都是特殊的 `<MASK>` id）。每一步，并行预测所有被掩码的 token，然后保留置信度最高的 top-K 预测并重新掩码其余的。经过大约 8-16 次迭代，所有 token 都被填充。每步取消掩码的 token 数量调度是经过调优的——余弦调度效果良好。

训练很简单：从 [0, 1] 中均匀采样掩码比率，将其应用于图像的 VQ token，训练 Transformer 恢复被掩码的 token。这正是 BERT 对文本所做的，扩展到图像生成。

### Show-o：一个 Transformer，混合掩码

Show-o 将 MaskGIT 放入一个因果语言模型 Transformer 中。注意力掩码如下：

- 文本 token：因果（标准 LLM）。
- 图像 token：图像块内完全双向（因此在预测期间被掩码的 token 可以看到其他所有图像 token）。
- 文本到图像：文本关注先前的图像，图像关注先前的文本。

训练在以下内容之间交替：
1. 文本序列上的标准 NTP。
2. T2I 样本：文本 → 图像，带被掩码的图像 token，掩码 token 预测损失。
3. VQA 样本：图像 → 文本，带被掩码的文本 token（实际上就是 NTP）。

统一损失是 `<MASK>` token 上的交叉熵，既涵盖文本 NTP（只有最后一个 token 被"掩码"），也涵盖图像掩码扩散（随机子集被掩码）。

### 并行采样

Show-o 在大约 16 步中生成图像，而自回归每 token 需要约 1000 步，或扩散需要约 20 步。每一步，并行预测所有被掩码的 token；提交置信度最高的 top-K；重复。

对比：
- Chameleon / Emu3（基于 token 的自回归）：N_tokens 次前向传播，通常每张图像 1024-4096 次。
- Transfusion（连续扩散）：约 20 步，每步一次完整的 Transformer 传播。
- Show-o（掩码离散扩散）：约 16 步，每步一次完整的 Transformer 传播。

Show-o 在相似规模模型下比 Chameleon 更快，大致匹配 Transfusion 的步骤数但每步成本更低（离散词表 logits vs 连续 MSE 损失）。

### 一个检查点中的任务

Show-o 在推理时支持四种任务，通过提示格式选择：

- 文本生成：标准自回归文本输出。
- VQA：图像输入，文本输出。
- T2I：文本输入，通过掩码离散扩散输出图像。
- 修复：包含一些被掩码 token 的图像，填充缺失部分。

修复能力是掩码预测训练免费提供的。掩码 VQ token 网格的一个区域，输入其余部分加文本提示，预测被掩码的 token。

### 掩码调度

每步取消掩码的 token 数量调度决定了质量。Show-o 推荐余弦：

```
mask_ratio(t) = cos(pi * t / (2 * T))   # t = 0..T
```

在步骤 0，所有 token 被掩码（比率 1.0）。在步骤 T，没有 token 被掩码。余弦将质量集中在预测最有信息的中段比率。线性调度也可行但更快达到平台期。

### Show-o2

Show-o2（2025 年后续版，arXiv 2506.15564）扩展了 Show-o：更大的 LLM 基础、更好的分词器、改进的掩码调度。相同的架构模式。

### Show-o 的位置

在 2026 年的分类法中：

- 离散 token + NTP：Chameleon、Emu3。简单但推理慢。
- 离散 token + 掩码扩散：Show-o、MaskGIT、LlamaGen、Muse。并行采样，仍然受分词器损失影响。
- 连续 + 扩散：Transfusion、MMDiT、DiT。最高质量，训练更复杂。
- VLM 中的连续 + 流匹配：JanusFlow、InternVL-U。最新的。

按任务选择：当你想在一个开源模型中同时拥有 T2I + 修复 + VQA 且速度合理时选 Show-o；当质量最重要且你负担得起双损失管道时选 Transfusion。

## 使用

`code/main.py` 模拟 Show-o 采样：

- 一个 16 个 VQ token 的玩具网格。
- 一个模拟 "Transformer"，根据提示和当前未掩码的 token 预测 logits。
- 使用余弦调度进行 8 步的并行掩码采样。
- 打印中间状态（掩码模式演化）和最终 token。

运行它，观察掩码逐步消解。

## 产出

本课产出 `outputs/skill-unified-gen-model-picker.md`。给定一个需要在开源权重约束下同时具备理解（VQA、标题生成）和生成（T2I、修复）能力的产品，在 Show-o 家族、Transfusion/MMDiT 家族和 Emu3 / Chameleon 家族之间做出选择，并给出具体权衡。

## 练习

1. 掩码离散扩散约需 16 步采样。为什么不是 1 步？如果在第 0 步就全部取消掩码会出什么问题？

2. 修复是掩码扩散的免费能力。提出一个产品用例（真实或假设），其中 Show-o 的修复能力优于专门的模型。

3. 余弦调度 vs 线性调度：追踪 T=8 时每步取消掩码的 token 数量。哪个更均衡？

4. 一张 512x512 的 Show-o 图像是 1024 个 token。在词表 K=16384 下，模型输出 1024 * log2(16384) = 14,336 位（约 1.75 KiB）的数据。Stable Diffusion 输出 512*512*24 位 = 6,291,456 位（约 768 KiB）的原始像素。压缩比是多少，它换来了什么质量？

5. 阅读 LlamaGen（arXiv:2406.06525）。LlamaGen 的类别条件自回归图像模型与 Show-o 的掩码方法有何不同？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 掩码离散扩散 | "MaskGIT 风格" | 训练以预测被掩码的 token；在推理时，迭代地取消掩码最置信的预测 |
| 余弦调度 | "取消掩码调度" | 掩码比率在推理步骤上的衰减；将置信度增长集中在中段 |
| 并行解码 | "一次性所有 token" | 每步在一次前向传播中预测被掩码 token 的完整序列，然后提交 top-K |
| 混合注意力 | "因果 + 双向" | 对文本 token 是因果的，对图像块内是双向的掩码 |
| 修复 | "填充生成" | 以某些 token 被掩码的图像为条件，预测缺失的部分；从训练目标中免费获得 |
| 提交率 | "每步 Top-K" | 每次迭代声明"完成"的 token 数量；控制推理与质量的权衡 |

## 拓展阅读

- [Xie et al. — Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)
- [Show-o2 (arXiv:2506.15564)](https://arxiv.org/abs/2506.15564)
- [Chang et al. — MaskGIT (arXiv:2202.04200)](https://arxiv.org/abs/2202.04200)
- [Sun et al. — LlamaGen (arXiv:2406.06525)](https://arxiv.org/abs/2406.06525)
- [Chang et al. — Muse (arXiv:2301.00704)](https://arxiv.org/abs/2301.00704)
