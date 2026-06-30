# 评估 — FID、CLIP 分数、人类偏好

> 每个生成模型排行榜都引用 FID、CLIP 分数和来自人类偏好竞技场的胜率。每个数字都有一个坚定研究者可以博弈的失败模式。如果你不了解失败模式，就无法区分真正的改进和博弈运行。

**类型：** 构建
**语言：** Python
**前置条件：** 第八阶段 · 01（分类）、第二阶段 · 04（评估指标）
**时间：** 约 45 分钟

## 问题

生成模型根据*样本质量*和*条件遵循度*来评判。两者都没有闭式度量。你的模型必须渲染 10,000 张图像；某些东西必须为它们分配数字；你必须跨模型家族、跨分辨率、跨架构信任这些数字。三个指标在 2014-2026 年的考验中幸存下来：

- **FID（Fréchet Inception Distance）。** 在 Inception 网络的特征空间中，两个分布——真实和生成——之间的距离。越低越好。
- **CLIP 分数。** 生成图像的 CLIP 图像嵌入与提示的 CLIP 文本嵌入之间的余弦相似度。越高越好。衡量提示遵循度。
- **人类偏好。** 在相同提示下对两个模型进行一对一比较，让人类（或 GPT-4 类模型）选择更好的，聚合为 Elo 分数。

你还会看到：IS（Inception 分数，大部分已退役）、KID、CMMD、ImageReward、PickScore、HPSv2、MJHQ-30k。每个都纠正了前一个的某个失败。

## 概念

![FID、CLIP 和偏好：三个轴，不同的失败模式](../assets/evaluation.svg)

### FID — 样本质量

Heusel 等人（2017）。步骤：

1. 为 N 个真实图像和 N 个生成图像提取 Inception-v3 特征（2048-D）。
2. 对每个池拟合高斯：计算均值 `μ_r, μ_g` 和协方差 `Σ_r, Σ_g`。
3. FID = `||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2 · (Σ_r · Σ_g)^0.5)`。

解释：特征空间中两个多变量高斯之间的 Fréchet 距离。越低 = 分布越相似。

失败模式：
- **小 N 上有偏。** FID 是特征分布上的均方误差——小 N 低估协方差，给出错误地低的 FID。始终使用 N ≥ 10,000。
- **依赖 Inception。** Inception-v3 是在 ImageNet 上训练的。远离 ImageNet 的领域（人脸、艺术、文本图像）产生无意义的 FID。使用领域特定的特征提取器。
- **博弈。** 过拟合 Inception 先验在不提高视觉质量的情况下获得低 FID。用 CMMD 击败它（见下文）。

### CLIP 分数 — 提示遵循度

Radford 等人（2021）。对于生成的图像 + 提示：

```
clip_score = cos_sim( CLIP_image(x_gen), CLIP_text(prompt) )
```

在 30k 个生成图像上取平均 → 在模型之间可比较的标量。

失败模式：
- **CLIP 自身的盲点。** CLIP 的组合推理弱（"a red cube on a blue sphere"经常失败）。模型可以在 CLIP 分数上排名高但没有真正遵循复杂提示。
- **短提示偏差。** 短提示在自然分布中有更多 CLIP 图像匹配。长提示机械地拥有更低的 CLIP 分数。
- **提示博弈。** 在提示中包含"high quality, 4k, masterpiece"会膨胀 CLIP 分数而不提高图像-文本绑定。

CMMD（Jayasumana 等人，2024）修复了其中一些：使用 CLIP 特征而非 Inception，最大均值差异而非 Fréchet。在检测微小的质量差异上更好。

### 人类偏好 — 真实基准

选一个提示池。用模型 A 和模型 B 生成。向人类（或强 LLM 评判者）展示配对。将胜场聚合为 Elo 或 Bradley-Terry 分数。基准：

- **PartiPrompts（Google）**：1,600 个多样化提示，12 个类别。
- **HPSv2**：107k 人类标注，广泛用作自动代理。
- **ImageReward**：137k 提示-图像偏好对，MIT 许可。
- **PickScore**：在 Pick-a-Pic 2.6M 偏好上训练。
- **Chatbot-Arena 风格图像竞技场**：https://imagearena.ai/ 及其他。

失败模式：
- **评判者方差。** 非专家与专家有不同的偏好。两者都使用。
- **提示分布。** 精心挑选的提示有利于某个家族。始终记录。
- **LLM 评判者奖励黑客。** GPT-4 评判者被漂亮但错误的输出欺骗。与人类三角验证。

## 一起使用

生产评估报告应包括：

1. 对保留的真实分布在 10-30k 样本上的 FID（样本质量）。
2. 在相同样本上对其提示的 CLIP 分数 / CMMD（遵循度）。
3. 在盲测竞技场中与先前模型的胜率（总体偏好）。
4. 失败模式分析：50 个随机采样输出，标记已知问题（手部解剖、文本渲染、一致物体数量）。

任何单一指标都不可信。三个相互印证的指标 + 定性评审才是可信的声明。

## 动手构建

`code/main.py` 在合成"特征向量"上实现 FID、类 CLIP 分数和 Elo 聚合（我们使用 4-D 向量代替 Inception 特征）。你看到：

- 在小的 N 和大的 N 上的 FID 计算——偏差。
- "CLIP 分数"作为特征池之间的余弦相似度。
- 来自合成偏好流的 Elo 更新规则。

### 步骤 1：四行代码的 FID

```python
def fid(real_features, gen_features):
    mu_r, cov_r = mean_and_cov(real_features)
    mu_g, cov_g = mean_and_cov(gen_features)
    mean_diff = sum((a - b) ** 2 for a, b in zip(mu_r, mu_g))
    trace_term = trace(cov_r) + trace(cov_g) - 2 * sqrt_cov_product(cov_r, cov_g)
    return mean_diff + trace_term
```

### 步骤 2：CLIP 风格的余弦相似度

```python
def clip_like(image_feat, text_feat):
    dot = sum(a * b for a, b in zip(image_feat, text_feat))
    norm = math.sqrt(dot_self(image_feat) * dot_self(text_feat))
    return dot / max(norm, 1e-8)
```

### 步骤 3：Elo 聚合

```python
def elo_update(r_a, r_b, winner, k=32):
    expected_a = 1 / (1 + 10 ** ((r_b - r_a) / 400))
    actual_a = 1.0 if winner == "a" else 0.0
    r_a_new = r_a + k * (actual_a - expected_a)
    r_b_new = r_b - k * (actual_a - expected_a)
    return r_a_new, r_b_new
```

## 缺陷陷阱

- **N=1000 的 FID。** 在 N<10k 下启发式不可靠。报告低 N FID 的论文在博弈。
- **跨分辨率比较 FID。** Inception 的 299×299 调整大小改变了特征分布。仅在匹配分辨率下比较。
- **报告一个种子。** 至少运行 3 个种子。报告标准差。
- **通过负向提示膨胀 CLIP 分数。** 某些流水线通过过拟合提示来提升 CLIP。检查视觉饱和度。
- **提示重叠导致的 Elo 偏差。** 如果两个模型在训练中见过基准提示，Elo 无意义。使用保留提示集。
- **人类评估付费众包偏差。** Prolific、MTurk 标注者偏向年轻人/技术爱好者。混合招募艺术/设计专家。

## 使用它

2026 年生产评估协议：

| 支柱 | 最低 | 推荐 |
|--------|---------|-------------|
| 样本质量 | 对保留真实分布 10k 样本的 FID | + 5k 样本的 CMMD + 每类别子集的 FID |
| 提示遵循度 | 30k 样本上的 CLIP 分数 | + HPSv2 + ImageReward + VQA 风格问答 |
| 偏好 | 200 对盲测 vs 基线 | + 2000 对人类 + LLM 评判者 + Chatbot Arena |
| 失败分析 | 50 个手工标记 | 500 个手工标记 + 自动化安全分类器 |

一份报告中的全部四个支柱 = 声明。任何一个单独 = 营销。

## 交付成果

保存 `outputs/skill-eval-report.md`。技能接收新模型检查点 + 基线并输出完整评估计划：样本大小、指标、失败模式探测、签署标准。

## 练习

1. **简单。** 运行 `code/main.py`。比较相同合成分布上 N=100 vs N=1000 的 FID。报告偏差大小。
2. **中等。** 从合成 CLIP 风格特征实现 CMMD（公式见 Jayasumana 等人，2024）。比较对质量差异的敏感度 vs FID。
3. **困难。** 复制 HPSv2 设置：从 Pick-a-Pic 子集取 1000 个图像-提示对，在偏好上微调一个小型基于 CLIP 的评分器，并测量其与保留集的一致性。

## 关键术语

| 术语 | 人们怎么说 | 它实际上的含义 |
|------|-----------------|-----------------------|
| FID | "Fréchet Inception Distance" | 对真实 vs 生成 Inception 特征的高斯拟合的 Fréchet 距离。 |
| CLIP 分数 | "文本-图像相似度" | CLIP 图像和文本嵌入之间的余弦相似度。 |
| CMMD | "FID 的替代" | CLIP 特征 MMD；偏差更小，无高斯假设。 |
| IS | "Inception 分数" | Exp KL(p(y|x) || p(y))；在现代模型上相关性差，已退役。 |
| HPSv2 / ImageReward / PickScore | "学到的偏好代理" | 在人类偏好上训练的小模型；用作自动评判者。 |
| Elo | "国际象棋评分" | 成对胜利的 Bradley-Terry 聚合。 |
| PartiPrompts | "基准提示集" | 1,600 个 Google 策划的提示，跨越 12 个类别。 |
| FD-DINO | "自监督替代" | 使用 DINOv2 特征的 FD；对非 ImageNet 领域更好。 |

## 生产笔记：评估也是一个推理工作负载

在 10k 样本上运行 FID 意味着生成 10k 张图像。对于单张 L4 上 1024² 的 50 步 SDXL 基础模型，那是约 11 小时的单请求推理。评估预算是真实的，框架正是离线推理场景（最大化吞吐量，忽略 TTFT）：

- **积极批处理，忘记延迟。** 离线评估 = 在适合内存的最大尺寸下静态批处理。在 80GB H100 上 `pipe(...).images` 带 `num_images_per_prompt=8` 的挂钟速度快 4-6 倍于单请求。
- **缓存真实特征。** 对真实参考集进行 Inception（FID）或 CLIP（CLIP-score、CMMD）特征提取仅*一次*，存储为 `.npz`。不要每次评估重新计算。

对于 CI / 回归门：每次 PR 在 500 样本子集上运行 FID + CLIP 分数（约 30 分钟）；夜间运行完整 10k FID + HPSv2 + Elo。

## 延伸阅读

- [Heusel 等人 (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID)](https://arxiv.org/abs/1706.08500) — FID 论文。
- [Jayasumana 等人 (2024). Rethinking FID: Towards a Better Evaluation Metric for Image Generation (CMMD)](https://arxiv.org/abs/2401.09603) — CMMD。
- [Radford 等人 (2021). Learning Transferable Visual Models from Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) — CLIP。
- [Wu 等人 (2023). HPSv2: A Comprehensive Human Preference Score](https://arxiv.org/abs/2306.09341) — HPSv2。
- [Xu 等人 (2023). ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation](https://arxiv.org/abs/2304.05977) — ImageReward。
- [Yu 等人 (2023). Scaling Autoregressive Models for Content-Rich Text-to-Image Generation (Parti + PartiPrompts)](https://arxiv.org/abs/2206.10789) — PartiPrompts。
- [Stein 等人 (2023). Exposing flaws of generative model evaluation metrics](https://arxiv.org/abs/2306.04675) — 失败模式调查。
