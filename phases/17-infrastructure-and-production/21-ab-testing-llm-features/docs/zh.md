# LLM 功能的 A/B 测试 — GrowthBook、Statsig 与凭感觉问题

> 传统 A/B 测试不是为非确定性 LLM 构建的。关键区别：评估回答"模型能做到这个工作吗？"A/B 测试回答"用户在乎吗？"两者都需要；凭感觉发布已经过时。2026 年需要测试的内容：提示工程（措辞）、模型选择（GPT-4 vs GPT-3.5 vs OSS；准确率 vs 成本 vs 延迟）、生参数（temperature、top-p）。真实案例：一个聊天机器人奖励模型变体带来了 +70% 会话长度和 +30% 留存率；Nextdoor AI 主题行实验在奖励函数优化后带来了 +1% CTR；Khan Academy Khanmigo 在延迟与数学准确率轴上迭代。平台分裂：**Statsig**（2025 年 9 月被 OpenAI 以 $11 亿收购）— 序列测试、CUPED、一体化。**GrowthBook** — 开源、仓库原生、贝叶斯 + 频率学派 + 序列引擎、CUPED、SRM 检查、Benjamini-Hochberg + Bonferroni 校正。你基于仓库 SQL 偏好以及"被 OpenAI 收购"是否对你的组织重要来选择。

**Type:** Learn
**Languages:** Python (stdlib, toy sequential test simulator)
**Prerequisites:** Phase 17 · 13 (Observability), Phase 17 · 20 (Progressive Deployment)
**Time:** ~60 minutes

## 学习目标

- 区分评估（"模型能做到这个工作吗"）和 A/B 测试（"用户在乎吗"）。
- 列举三个可测试轴（提示、模型、参数）并为每个选择指标。
- 解释 CUPED、序列测试和 Benjamini-Hochberg 多重比较校正。
- 基于仓库 SQL 姿态和企业收购立场选择 Statsig 或 GrowthBook。

## 问题

你手动调整了一个系统提示。感觉更好了。你发布了它。转化率随噪声变化。你责怪指标。或者你发布了一个新模型而转化率没有变化 — 是模型退化了还是变化太小无法检测？你不知道，因为你没有 A/B 就发布了。

评估回答模型能否在标记集上完成任务。它们不回答用户是否偏好输出。只有受控在线实验能回答这个问题，并且只有当实验有足够的统计功效、控制非确定性并校正多重比较时才有效。

## 概念

### 评估 vs A/B 测试

**评估** — 离线、标记集、裁判（评分标准或 LLM 作为裁判或人工）。回答："在这个固定分布上输出是否正确/有帮助/安全？"

**A/B 测试** — 在线、真实用户、随机化。回答："新变体是否移动了重要的用户级指标？"

两者都需要。评估在暴露前捕获退化；A/B 在之后确认产品影响。

### 测试什么

1. **提示工程** — 措辞、系统提示结构、示例。指标：任务成功率、用户留存率、每请求成本。
2. **模型选择** — GPT-4 vs GPT-3.5-Turbo vs Llama-OSS。指标：准确率（任务）+ 每请求成本 + 延迟 P99。多目标。
3. **生参数** — temperature、top-p、max_tokens。指标：任务特定（输出多样性 vs 确定性）。

### CUPED — 方差降低

使用实验前数据的受控实验（Controlled-experiments Using Pre-Experiment Data）。在比较后周期之前回退前周期方差。典型方差降低：30-70%。有效样本量免费提升。

实现：Statsig 和 GrowthBook 都实现。

### 序列测试

经典 A/B 假设固定样本量。序列测试（"可窥视并决定"）在反复查看下控制假阳性率。始终有效的序列程序（mSPRT、Howard 的置信序列）让你在明显赢家上提前停止。

### 多重比较校正

在 95% 置信度下运行 20 次 A/B 测试会产生一个偶然的假阳性。Bonferroni 校正收紧每次测试的 α；Benjamini-Hochberg 控制错误发现率。GrowthBook 实现两者。

### SRM — 样本比率不匹配

分配哈希将用户随机分配给变体。如果 50/50 分配产生了 47/53，有地方坏了 — SRM 检查标记它。两个平台都实现。

### Statsig vs GrowthBook

**Statsig**：
- 2025 年 9 月被 OpenAI 以 $11 亿收购。托管、SaaS。
- 序列测试、CUPED、保留人群。
- 一体化：功能标志 + 实验 + 可观测性。
- 最适合：团队已经想要捆绑产品，不关心 OpenAI 所有权。

**GrowthBook**：
- 开源（MIT）；仓库原生（直接从 Snowflake/BigQuery/Redshift 读取）。
- 多引擎：贝叶斯、频率学派、序列。
- CUPED、SRM、Bonferroni、BH 校正。
- 自托管或托管云端。
- 最适合：仓库 SQL 环境，数据团队控制指标层，想要开源。

### 非确定性使统计功效复杂化

相同提示产生不同输出。传统功效计算假设 IID 观测。对于 LLM 非确定性，有效样本量低于名义值。将所需样本量乘以约 1.3-1.5 倍作为安全余量。

### 真实案例结果

- 聊天机器人奖励模型变体：+70% 会话长度，+30% 留存率。
- Nextdoor 主题行：奖励函数优化后 +1% CTR。
- Khan Academy Khanmigo：迭代延迟 vs 数学准确率权衡。

### 反模式：凭感觉发布

每个高级工程师都能说出一项因为"感觉更好"而发布但没有 A/B 的功能。其中大多数退化产品指标，团队数月未注意到。A/B 是强制函数。

### 你应该记住的数字

- Statsig 被 OpenAI 收购：$11 亿，2025 年 9 月。
- GrowthBook：开源 MIT；贝叶斯 + 频率学派 + 序列。
- CUPED 方差降低：30-70%。
- LLM 非确定性 → +30-50% 样本量缓冲。

## 使用它

`code/main.py` 模拟一个带固定和序列边界的序列 A/B 测试。展示序列如何让你提前停止。

## 交付它

本课产出 `outputs/skill-ab-plan.md`。给定功能变化、工作负载、基线，选择平台、门控、样本量。

## 练习

1. 运行 `code/main.py`。对预期 5% 提升、基线 3% 转化率，80% 统计功效需要什么样本量？
2. 为受医疗监管的本地客户选择 Statsig 或 GrowthBook。
3. 设计一个测试 GPT-4 vs GPT-3.5 在每解决工单成本上的 A/B 实验。主要指标是什么？护栏指标？次要指标？
4. 你的金丝雀通过但 A/B 显示 -1.2% 转化率。你发布吗？写出升级标准。
5. 将 CUPED 应用于方差为后周期 60% 的前周期。计算有效样本量提升。

## 核心术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Eval（评估） | "离线测试" | 模型能力的标记集评估 |
| A/B test | "实验" | 用户上的实时随机化比较 |
| CUPED | "方差降低" | 前周期回归以降低方差 |
| Sequential test（序列测试） | "可窥视测试" | 允许提前停止的始终有效程序 |
| Multiple comparison | "家庭错误" | 运行许多测试膨胀假阳性 |
| Bonferroni | "严格校正" | 将 α 除以测试次数 |
| Benjamini-Hochberg | "BH FDR" | 错误发现率控制，较不保守 |
| SRM | "坏分流" | 样本比率不匹配；分配错误 |
| Statsig | "OpenAI 所有" | 商业一体化，2025 年被收购 |
| GrowthBook | "开源那个" | MIT 仓库原生平台 |
| mSPRT | "序列概率比测试" | 经典序列程序 |

## 进一步阅读

- [GrowthBook — 如何 A/B 测试 AI](https://blog.growthbook.io/how-to-a-b-test-ai-a-practical-guide/)
- [Statsig — 超越提示：数据驱动的 LLM 优化](https://www.statsig.com/blog/llm-optimization-online-experimentation)
- [Statsig vs GrowthBook 比较](https://www.statsig.com/perspectives/ab-testing-feature-flags-comparison-tools)
- [Deng et al. — CUPED](https://www.exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf)
- [Howard — 置信序列](https://arxiv.org/abs/1810.08240)
