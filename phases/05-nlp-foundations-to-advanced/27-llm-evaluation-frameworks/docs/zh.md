# LLM 评估 — RAGAS、DeepEval、G-Eval

> 精确匹配和 F1 遗漏语义等价。人工审核不能扩展。LLM 作为评判者是生产答案——需足够校准才能信任该数字。

**类型：** 构建
**语言：** Python
**先修要求：** 第五阶段 · 13（问答），第五阶段 · 14（信息检索）
**预计时间：** 约75分钟

## 问题

你的 RAG 系统回答："June 29th, 2007."
黄金参考是："June 29, 2007."
精确匹配得 0。F1 得约 75%。人类会给出 100%。

现在乘以 10,000 个测试用例。再乘以对检索器、分块、提示或模型的每次更改。你需要一个理解含义的评估器，能在大规模上廉价运行，不会对退化撒谎，并呈现正确的失败模式。

2026 年有三个框架解决了这个问题。

- **RAGAS。** 检索增强生成评估。四个 RAG 指标（忠实度、答案相关性、上下文精确率、上下文召回率），NLI + LLM 评判者后端。研究支持、轻量级。
- **DeepEval。** LLM 的 Pytest。G-Eval、任务完成、幻觉、偏置指标。CI/CD 原生。
- **G-Eval。** 一种方法（和一个 DeepEval 指标）：带思维链、自定义标准、0-1 分数的 LLM 作为评判者。

三者都依赖于 LLM 作为评判者。本课为该方法及其周围的信任层建立直觉。

## 概念

![Four evaluation dimensions, LLM-as-judge architecture](../assets/llm-evaluation.svg)

**LLM 作为评判者。** 用一个 LLM 替代静态指标，该 LLM 根据评分标准对输出评分。给定`(query, context, answer)`，提示一个评判 LLM："Score 0-1 on faithfulness。"返回分数。

为什么它有效：LLM 以极小的一部分成本近似人类判断。GPT-4o-mini 每评分案例约 $0.003，使 1000 样本回归评估运行成本低于 $5。

为什么它静默失败：

1. **评判者偏置。** 评判者偏爱更长的答案、来自自己模型族的答案、与提示风格匹配的答案。
2. **JSON 解析失败。** 错误的 JSON → NaN 分数 → 静默从聚合中排除。RAGAS 用户知道这种痛苦。用 try/except + 显式失败模式设门槛。
3. **模型版本间漂移。** 升级评判者会改变每个指标。冻结评判者模型 + 版本。

**RAG 四项。**

| 指标 | 问题 | 后端 |
|--------|----------|---------|
| 忠实度 | 答案中的每个声明是否来自检索到的上下文？ | 基于 NLI 的蕴涵 |
| 答案相关性 | 答案是否回应了问题？ | 从答案生成假设性问题；与真实问题比较 |
| 上下文精确率 | 在检索到的分块中，哪些比例是相关的？ | LLM 评判者 |
| 上下文召回率 | 检索是否返回了所需的一切？ | 针对黄金答案的 LLM 评判者 |

**G-Eval。** 定义自定义标准："答案是否引用了正确的来源？"框架自动扩展为思维链评估步骤，然后评分 0-1。适用于 RAGAS 不覆盖的领域特定质量维度。

**校准。** 在对人工标签有相关性之前，永远不要信任原始评判者分数。运行 100 个人工标注样本。绘制评判者对人工。计算 Spearman rho。如果 rho < 0.7，你的评判者评分标准需要改进。

## 构建它

### 步骤 1：用 NLI 进行忠实度评估（RAGAS 风格）

```python
from typing import Callable
from transformers import pipeline

nli = pipeline("text-classification",
               model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
               top_k=None)

# `llm` 是任何可调用对象: prompt str -> generated str。
# 例如：llm = lambda p: client.messages.create(model="claude-haiku-4-5", ...).content[0].text
LLM = Callable[[str], str]


def atomic_claims(answer: str, llm: LLM) -> list[str]:
    prompt = f"""Break this answer into simple factual claims (one per line):
{answer}
"""
    return llm(prompt).splitlines()


def faithfulness(answer: str, context: str, llm: LLM) -> float:
    claims = atomic_claims(answer, llm)
    if not claims:
        return 0.0
    supported = 0
    for claim in claims:
        result = nli({"text": context, "text_pair": claim})[0]
        entail = next((s for s in result if s["label"] == "entailment"), None)
        if entail and entail["score"] > 0.5:
            supported += 1
    return supported / len(claims)
```

将答案分解为原子声明。针对检索到的上下文对每个声明进行 NLI 检查。忠实度 = 被支持的比例。

### 步骤 2：答案相关性

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# encoder: 任何实现 .encode(texts, normalize_embeddings=True) -> ndarray 的模型
# 例如，encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")

def answer_relevance(question: str, answer: str, encoder, llm: LLM, n: int = 3) -> float:
    prompt = f"Write {n} questions this answer could be the answer to:\n{answer}"
    generated = [line for line in llm(prompt).splitlines() if line.strip()][:n]
    if not generated:
        return 0.0
    q_emb = np.asarray(encoder.encode([question], normalize_embeddings=True)[0])
    g_embs = np.asarray(encoder.encode(generated, normalize_embeddings=True))
    sims = [float(q_emb @ g_emb) for g_emb in g_embs]
    return sum(sims) / len(sims)
```

如果答案暗示的问题与所问的问题不同，相关性下降。

### 步骤 3：G-Eval 自定义指标

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

metric = GEval(
    name="Correctness",
    criteria="The answer should be factually accurate and match the expected output.",
    evaluation_steps=[
        "Read the expected output.",
        "Read the actual output.",
        "List factual claims in the actual output.",
        "For each claim, mark supported or unsupported by the expected output.",
        "Return score = fraction supported.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
)

test = LLMTestCase(input="When was the first iPhone released?",
                   actual_output="June 29th, 2007.",
                   expected_output="June 29, 2007.")
metric.measure(test)
print(metric.score, metric.reason)
```

评估步骤就是评分标准。显式步骤比隐式的"score 0-1"提示更稳定。

### 步骤 4：CI 门控

```python
import deepeval
from deepeval.metrics import FaithfulnessMetric, ContextualRelevancyMetric


def test_rag_system():
    cases = load_regression_cases()
    faith = FaithfulnessMetric(threshold=0.85)
    rel = ContextualRelevancyMetric(threshold=0.7)
    for case in cases:
        faith.measure(case)
        assert faith.score >= 0.85, f"faithfulness regression on {case.id}"
        rel.measure(case)
        assert rel.score >= 0.7, f"relevancy regression on {case.id}"
```

作为 pytest 文件发布。在每个 PR 上运行。对退化阻止合并。

### 步骤 5：从零构建玩具评估器

参见`code/main.py`。仅标准库的忠实度近似（答案声明与上下文的重叠）和相关性（答案 token 与问题 token 的重叠）。不用于生产。展示形态。

## 陷阱

- **没有校准。** 与人工标签相关性为 0.3 的评判者是噪声。在发布前要求校准运行。
- **自我评估。** 使用相同的 LLM 生成和评判会膨胀 10-20% 的分数。对评判者使用不同的模型族。
- **成对评判中的位置偏置。** 评判者偏好呈现的第一个选项。始终随机化顺序并进行两次。
- **原始聚合隐藏失败。** 平均分数 0.85 经常隐藏 5% 的灾难性失败。始终检查底分位数。
- **黄金数据集腐化。** 随时间漂移的无版本评估集破坏纵向比较。用每次更改标记数据集。
- **LLM 成本。** 在大规模下，评判者调用主导成本。使用满足校准阈值的最便宜模型。GPT-4o-mini、Claude Haiku、Mistral-small。

## 使用它

2026 年技术栈：

| 用例 | 框架 |
|---------|-----------|
| RAG 质量监控 | RAGAS（4 个指标）|
| CI/CD 回归门控 | DeepEval + pytest |
| 自定义领域标准 | DeepEval 内的 G-Eval |
| 在线实时流量监控 | RAGAS 带无参考模式 |
| 人工回环抽查 | LangSmith 或带注释 UI 的 Phoenix |
| 红队 / 安全评估 | Promptfoo + DeepEval |

典型技术栈：RAGAS 用于监控，DeepEval 用于 CI，G-Eval 用于新维度。运行全部三者；它们有用地不一致。

## 交付它

保存为 `outputs/skill-eval-architect.md`：

```markdown
---
name: eval-architect
description: 设计带校准评判者和 CI 门控的 LLM 评估计划。
version: 1.0.0
phase: 5
lesson: 27
tags: [nlp, evaluation, rag]
---

给定一个用例（RAG / 智能体 / 生成任务），输出：

1. 指标。忠实度 / 相关性 / 上下文精确率 / 上下文召回率 + 任何带标准的自定义 G-Eval 指标。
2. 评判者模型。命名的模型 + 版本，成本与准确率权衡的理由。
3. 校准。人工标注集大小，目标 Spearman rho vs 人工 > 0.7。
4. 数据集版本管理。标记策略、更改日志、分层。
5. CI 门控。每个指标的阈值、回归窗口逻辑、底分位数警报。

拒绝依赖未在 ≥50 个人工标注样本上测试过的评判者。拒绝自我评估（同一模型生成 + 评判）。拒绝没有底 10% 呈现的仅聚合报告。标记任何评判者升级但没有并行基线评估的流程。
```

## 练习

1. **简单。** 在 10 个有已知幻觉的 RAG 样本上使用 RAGAS。验证忠实度指标捕获了每一个。
2. **中等。** 为 50 个 QA 答案人工标注 0-1 的正确性。用 G-Eval 评分。测量评判者与人工之间的 Spearman rho。
3. **困难。** 用 DeepEval 构建 pytest CI 门控。故意退化检索器。验证门控失败。通过对最低 10% 的阈值检查添加底分位数警报。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| LLM 作为评判者 | 用 LLM 评分 | 提示评判模型根据评分标准对输出评分 0-1。 |
| RAGAS | RAG 指标库 | 开源评估框架，带 4 个无参考 RAG 指标。 |
| 忠实度 | 答案是否接地？ | 答案声明被检索上下文蕴含的比例。 |
| 上下文精确率 | 检索到的分块相关吗？ | top-K 分块中实际有用的比例。 |
| 上下文召回率 | 检索找到了所有内容吗？ | 黄金答案声明被检索分块支持的比例。 |
| G-Eval | 自定义 LLM 评判者 | 评分标准 + 思维链评估步骤 + 0-1 分数。 |
| 校准 | 信任但验证 | 评判者分数与人工分数之间的 Spearman 相关性。 |

## 扩展阅读

- [Es et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217) — RAGAS 论文。
- [Liu et al. (2023). G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment](https://arxiv.org/abs/2303.16634) — G-Eval 论文。
- [DeepEval docs](https://deepeval.com/docs/metrics-introduction) — 开放生产技术栈。
- [Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) — 偏置、校准、限制。
- [MLflow GenAI Scorer](https://mlflow.org/blog/third-party-scorers) — 集成 RAGAS、DeepEval、Phoenix 的统一框架。
