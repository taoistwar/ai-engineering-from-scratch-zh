# 长上下文评估 — NIAH、RULER、LongBench、MRCR

> Gemini 3 Pro 声称 10M token 的上下文。在 1M token 时，8 针 MRCR 下降到 26.3%。广告 ≠ 可用。长上下文评估告诉你你部署的模型的实际容量。

**类型：** 学习
**语言：** Python
**先修要求：** 第五阶段 · 13（问答），第五阶段 · 23（分块策略）
**预计时间：** 约60分钟

## 问题

你有一份 200 页的合同。模型声称 1M token 的上下文。你粘贴合同并问："终止条款是什么？"模型回答了——但从封面页回答，因为终止条款位于 120k token 深处，超出了模型实际关注的位置。

这是 2026 年的上下文容量差距。规格表上说 1M 或 10M。现实说其中的 60-70% 是可用的，且"可用"取决于任务。

- **检索（大海捞针单针）：** 在前沿模型上接近完美，直到广告的最大值。
- **多跳 / 聚合：** 在大多数模型上超过约 128k 时急剧下降。
- **分散事实的推理：** 第一个失败的任务。

长上下文评估测量这些轴。本课指出基准测试、每个实际测量什么，以及如何为你的领域构建自定义针测试。

## 概念

![NIAH baseline, RULER multi-task, LongBench holistic](../assets/long-context-eval.svg)

**大海捞针（NIAH, 2023）。** 在长上下文的受控深度放置一个事实（"the magic word is pineapple"）。要求模型检索它。扫描深度 × 长度。原始的长上下文基准测试。前沿模型现在饱和了这一点；它是一个必要但不充分的基线。

**RULER（Nvidia, 2024）。** 跨四个类别的 13 种任务类型：检索（单 / 多键 / 多值）、多跳追踪（变量跟踪）、聚合（常见词频率）、QA。可配置的上下文长度（4k 到 128k+）。揭示了那些饱和 NIAH 但在多跳上失败的模型。在 2024 年发布中，声称 32k+ 上下文的 17 个模型中只有一半在 32k 上保持质量。

**LongBench v2（2024）。** 503 道多选题，8k-2M 词上下文，六个任务类别：单文档 QA、多文档 QA、长上下文学习、长对话、代码仓库、长结构化数据。真实世界长上下文行为的生产基准测试。

**MRCR（多轮共指消解）。** 大规模多轮共指。8 针、24 针、100 针变体。揭示了在注意力退化之前模型能同时处理多少事实。

**NoLiMa。** "非词汇针。"针和查询共享零字面重叠；检索需要一步语义推理。比 NIAH 更难。

**HELMET。** 拼接许多文档，从任意一个中提问。测试选择性注意力。

**BABILong。** 将 bAbI 推理链嵌入无关干草堆中。测试干草堆中的推理，而不仅仅是检索。

### 实际报告什么

- **广告的上下文窗口。** 规格表数字。
- **有效检索长度。** 在某个阈值（例如 90%）下的 NIAH 通过率。
- **有效推理长度。** 在相同阈值下的多跳或聚合通过率。
- **退化曲线。** 准确率与上下文长度，按任务类型绘制。

你的规格表的两个数字：检索有效和推理有效。通常推理有效是广告窗口的 25-50%。

## 构建它

### 步骤 1：为你的领域构建自定义 NIAH

参见`code/main.py`。骨架：

```python
def build_haystack(filler_text, needle, depth_ratio, total_tokens):
    if not (0.0 <= depth_ratio <= 1.0):
        raise ValueError(f"depth_ratio must be in [0, 1], got {depth_ratio}")
    if total_tokens <= 0:
        raise ValueError(f"total_tokens must be positive, got {total_tokens}")

    filler_tokens = tokenize(filler_text)
    needle_tokens = tokenize(needle)
    if not filler_tokens:
        raise ValueError("filler_text produced no tokens")

    # 重复干草直到足够长以填充干草堆主体。
    body_len = max(total_tokens - len(needle_tokens), 0)
    while len(filler_tokens) < body_len:
        filler_tokens = filler_tokens + filler_tokens
    filler_tokens = filler_tokens[:body_len]

    insert_at = min(int(body_len * depth_ratio), body_len)
    haystack = filler_tokens[:insert_at] + needle_tokens + filler_tokens[insert_at:]
    return " ".join(haystack)


def score_niah(model, haystack, question, expected):
    answer = model.complete(f"Context: {haystack}\nQ: {question}\nA:", max_tokens=50)
    return 1 if expected.lower() in answer.lower() else 0
```

扫描`depth_ratio` ∈ {0, 0.25, 0.5, 0.75, 1.0} × `total_tokens` ∈ {1k, 4k, 16k, 64k}。绘制热力图。那是你目标模型的 NIAH 卡片。

### 步骤 2：多针变体

```python
def build_multi_needle(filler, needles, total_tokens):
    depths = [0.1, 0.4, 0.7]
    chunks = [filler[:int(total_tokens * 0.1)]]
    for depth, needle in zip(depths, needles):
        chunks.append(needle)
        next_chunk = filler[int(total_tokens * depth): int(total_tokens * (depth + 0.3))]
        chunks.append(next_chunk)
    return " ".join(chunks)
```

诸如"What are the three magic words?"的问题需要检索全部三个。单针成功不能预测多针成功。

### 步骤 3：多跳变量追踪（RULER 风格）

```python
haystack = """X1 = 42. ... (filler) ... X2 = X1 + 10. ... (filler) ... X3 = X2 * 2."""
question = "What is X3?"
```

答案需要链接三个赋值。128k 下的前沿模型在此通常下降到 50-70% 准确率。

### 步骤 4：在你的技术栈上的 LongBench v2

```python
from datasets import load_dataset
longbench = load_dataset("THUDM/LongBench-v2")

def eval_model_on_longbench(model, subset="single-doc-qa"):
    tasks = [x for x in longbench["test"] if x["task"] == subset]
    correct = 0
    for x in tasks:
        answer = model.complete(x["context"] + "\n\nQ: " + x["question"], max_tokens=20)
        if normalize(answer) == normalize(x["answer"]):
            correct += 1
    return correct / len(tasks)
```

报告每类别准确率。聚合分数隐藏大的任务级差异。

## 陷阱

- **仅 NIAH 评估。** 在 1M token 通过 NIAH 对多跳毫无意义。始终运行 RULER 或自定义多跳测试。
- **均匀深度采样。** 许多实现只测试 depth=0.5。测试 depth=0, 0.25, 0.5, 0.75, 1.0——"迷失在中间"效应是真实存在的。
- **与干草堆的词汇重叠。** 如果针与干草堆共享关键词，检索变得平凡。使用 NoLiMa 式非重叠针。
- **忽略延迟。** 1M token 的提示需要 30-120 秒进行预填充。在准确率的同时测量首次 token 时间。
- **供应商自报数字。** OpenAI、Google、Anthropic 都发布自己的分数。始终在你的用例上独立重新运行。

## 使用它

2026 年技术栈：

| 场景 | 基准测试 |
|-----------|-----------|
| 快速健全检查 | 自定义 NIAH 在 3 深度 × 3 长度 |
| 用于生产的模型选择 | RULER（13 任务）在你的目标长度 |
| 真实世界 QA 质量 | LongBench v2 single-doc-QA 子集 |
| 多跳推理 | BABILong 或自定义变量追踪 |
| 会话 / 对话 | MRCR 8 针在你的目标长度 |
| 模型升级回归 | 固定的内部 NIAH + RULER 工具，在每新模型上运行 |

生产经验法则：在你有预期长度的 NIAH + 1 个推理任务之前，永远不要信任上下文窗口。

## 交付它

保存为 `outputs/skill-long-context-eval.md`：

```markdown
---
name: long-context-eval
description: 为给定的模型和用例设计长上下文评估组合。
version: 1.0.0
phase: 5
lesson: 28
tags: [nlp, long-context, evaluation]
---

给定目标模型、目标上下文长度和用例，输出：

1. 测试。NIAH 深度 × 长度网格；RULER 多跳；自定义领域任务。
2. 采样。每个长度处深度 0, 0.25, 0.5, 0.75, 1.0。
3. 指标。检索通过率；推理通过率；首次 token 时间；每查询成本。
4. 截止。有效检索长度（90% 通过）和有效推理长度（70% 通过）。报告两者。
5. 回归。固定工具，在每次模型升级时重新运行，呈现差异。

拒绝仅从模型卡片信任上下文窗口。拒绝任何多跳工作负载的仅 NIAH 评估。拒绝供应商自报的长上下文分数作为独立证据。
```

## 练习

1. **简单。** 构建带 3 深度（0.25, 0.5, 0.75）× 3 长度（1k, 4k, 16k）的 NIAH。在任意模型上运行。绘制通过率为 3×3 热力图。
2. **中等。** 添加 3 针变体。在每个长度测量全部 3 针的检索。与相同长度的单针通过率比较。
3. **困难。** 构建嵌入在 64k 干草堆中的变量追踪任务（X1 → X2 → X3，3 跳）。在 3 个前沿模型上测量准确率。报告每个模型的有效推理长度。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| NIAH | 大海捞针 | 在填料中放置一个事实，要求模型检索它。 |
| RULER | NIAH 的升级版 | 跨检索 / 多跳 / 聚合 / QA 的 13 种任务类型。 |
| 有效上下文 | 真实容量 | 准确率仍保持在阈值以上的长度。 |
| 迷失在中间 | 深度偏置 | 模型对长输入中间的内容关注不足。 |
| 多针 | 同时多个事实 | 多个植入；测试注意力并行处理，而不仅仅是检索。 |
| MRCR | 多轮共指 | 8、24 或 100 针共指；暴露注意力饱和。 |
| NoLiMa | 非词汇针 | 针和查询共享零字面 token；需要推理。 |

## 扩展阅读

- [Kamradt (2023). Needle in a Haystack analysis](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) — 原始的 NIAH 仓库。
- [Hsieh et al. (2024). RULER: What's the Real Context Size of Your Long-Context LMs?](https://arxiv.org/abs/2404.06654) — 多任务基准。
- [Bai et al. (2024). LongBench v2](https://arxiv.org/abs/2412.15204) — 真实世界长上下文评估。
- [Modarressi et al. (2024). NoLiMa: Non-lexical needles](https://arxiv.org/abs/2404.06666) — 更难的针。
- [Kuratov et al. (2024). BABILong](https://arxiv.org/abs/2406.10149) — 干草堆中的推理。
- [Liu et al. (2024). Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) — 深度偏置论文。
