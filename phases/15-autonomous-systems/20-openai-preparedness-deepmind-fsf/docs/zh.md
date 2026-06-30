# OpenAI准备度框架与DeepMind前沿安全框架

> OpenAI准备度框架v2（2025年4月）引入研究类别——长周期自主性、藏拙、自主复制与适应、破坏安全保障——与跟踪类别区分。跟踪类别触发能力报告加安全保障报告，由安全咨询小组审查。DeepMind的FSF v3（2025年9月，跟踪能力级别于2026年4月17日添加）将自主性纳入ML R&D和网络域（ML R&D自主性级别1 = 在人类+AI工具对比下以具有竞争力的成本完全自动化AI R&D流程）。FSF v3通过针对工具性推理滥用的自动化监控明确处理欺骗性对齐。诚实的提示：PF v2中的研究类别（包括长周期自主性）不会自动触发缓解措施；政策语言是"潜在的"。DeepMind自己表示如果工具性推理增强，自动化监控"将不会长期保持足够。"

**Type:** Learn
**Languages:** Python (stdlib, three-framework decision-table diff tool)
**Prerequisites:** Phase 15 · 19 (Anthropic RSP)
**Time:** ~45 minutes

## 问题

第19课仔细阅读了Anthropic的扩展政策。本课通过阅读OpenAI和DeepMind的文件来完成图景。这三份文件是表亲产物，解决相同的问题——前沿实验室应在何时暂停或门禁模型——它们在一小组类别上收敛，在重要的特定地方分歧。

收敛点：三者都将长周期自主性标记为值得跟踪的能力类别。三者都承认欺骗性行为（对齐伪装、藏拙）是一类特定的风险。三者都有一个内部审查机构。分歧点：OpenAI将类别分为"跟踪"（强制缓解措施）和"研究"（无自动触发）。DeepMind将自主性纳入两个领域而非单独命名。实验室名称为跟踪 vs 研究，或关键 vs 中等，或Tier-1 vs Tier-2；一个能力落入哪个桶的操作后果在不同实验室之间是不同的。

将它们一起阅读是有用的练习。同一能力在Anthropic可能是"强制缓解措施"，在OpenAI是"被监控但不触发"，在DeepMind是"在特定域中跟踪"。政策姿态很重要。

## 概念

### OpenAI准备度框架v2（2025年4月）

结构：

- **跟踪类别**：触发能力报告（模型能做什么）加安全保障报告（什么缓解措施就位）。在部署前由安全咨询小组审查。
- **研究类别**：实验室正在跟踪但尚未承诺特定缓解措施的模型能力。包括长周期自主性、藏拙、自主复制与适应、破坏安全保障。

v2中的关键框架：研究类别不会自动触发缓解措施。政策语言是"潜在的"缓解措施。这是要注意的具体增量：一个类别位于跟踪还是研究是"门禁"和"观察"之间的枢纽。

### DeepMind前沿安全框架v3（2025年9月；跟踪能力级别于2026年4月17日添加）

结构：

- **关键能力级别（CCL）**：跨五个领域的能力阈值：网络、生物、ML R&D、CBRN、自主性（纳入ML R&D和网络）。
- **跟踪能力级别**：2026年4月添加的额外粒度。具体示例：ML R&D自主性级别1 = 在人类+AI工具对比下以具有竞争力的成本完全自动化AI R&D流程。
- **欺骗性对齐监控**：对工具性推理滥用的自动化监控的显式承诺。

自主性框架与OpenAI不同。DeepMind不将"自主性"作为顶级领域保留；它被纳入自主性会导致危害的领域（ML R&D和网络）。论点是：没有领域的自主性是没有风险的能力；反论点是：跨领域的自主性是框架应该命名的元风险。

### 三者都收敛于什么

- 内部安全咨询小组（命名为Anthropic SAG、OpenAI SAG、DeepMind内部委员会）。对高能力模型的部署前审查。
- 明确提及欺骗性对齐/对齐伪装作为风险类别。
- 声明的频率上的常设产物（Anthropic：前沿安全路线图、风险报告；OpenAI：能力和安全保障报告；DeepMind：FSF更新周期）。
- 承认仅监控的防御有其天花板。DeepMind是明确的："自动化监控将不会长期保持足够。"

### 它们在哪里分歧

- **Anthropic**：暂停承诺在v3.0中移除；AI R&D-4阈值是命名的下一道门。
- **OpenAI**：跟踪 vs 研究分割；研究类别（包括长周期自主性）不会自动门禁。
- **DeepMind**：自主性纳入其他领域；跟踪能力级别在2026年4月添加粒度。

### 藏拙：一个复杂化所有三者的特定能力

藏拙（模型在评估上战略性低表现）在OpenAI的研究类别中。Anthropic的RSP v3.0通过评估-上下文差距（第1课）处理。DeepMind通过FSF v3中的欺骗性对齐监控处理。

如果模型在评估上藏拙，每个框架的能力阈值都被低估了。框架只有在测量有效时才工作。这就是为什么外部测量（第21课，METR）和对抗性评估是实验室自我评估之外的必要补充。

### 政策阅读技能

- **定位**：你关心的每个能力应该在政策中可找到。如果不可找到，政策不涵盖它。
- **分类**：它是跟踪的（触发缓解措施）还是研究的（被跟踪但不触发）？OpenAI命名此概念；Anthropic和DeepMind有自己的等价概念。
- **频率**：政策是以声明的日程更新，还是仅在特定事件后更新？声明的频率更强。
- **独立性**：外部审查是强制的还是可选的？Anthropic与Apollo和美国AI安全研究所合作；OpenAI与METR；DeepMind主要是内部SAG。

## 运用

`code/main.py` 实现一个小型决策表差异工具。给定一个能力（自主性、欺骗性对齐、研发自动化、网络提升等），输出三项政策如何分类该能力，以及触发哪些缓解措施。它是一个阅读辅助工具，而非政策工具。

## 交付物

`outputs/skill-cross-policy-diff.md` 使用三个框架作为参考，为特定能力生成跨政策比较。

## 练习

1. 运行 `code/main.py`。确认差异工具的输出与你能够对照源文档验证的至少两个能力匹配。

2. 完整阅读OpenAI准备度框架v2。识别每个研究类别。为每个写一句话说明为什么它在研究而非跟踪中。

3. 完整阅读DeepMind FSF v3，加上2026年4月跟踪能力级别更新。识别ML R&D自主性级别1的具体评估标准。你将如何在外部测量它？

4. 藏拙在OpenAI的研究类别中。设计一个强迫藏拙模型暴露其实际能力的评估。参考第1课评估-上下文博弈讨论。

5. 在特定能力上比较三项政策（你自选）。指出你认为哪个政策的分类最严格，哪个最不严格。用源文本证明。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|---|---|---|
| 准备度框架 | "OpenAI的扩展政策" | PF v2（2025年4月）；跟踪 vs 研究类别 |
| 跟踪类别 | "强制缓解措施" | 触发能力 + 安全保障报告；SAG审查 |
| 研究类别 | "仅监控" | 被跟踪但无自动缓解措施；包括长周期自主性 |
| 前沿安全框架 | "DeepMind的扩展政策" | FSF v3（2025年9月）+ 跟踪能力级别（2026年4月） |
| CCL | "关键能力级别" | DeepMind每领域阈值（网络、生物、ML R&D、CBRN） |
| ML R&D自主性级别1 | "研发自动化" | 以具有竞争力的成本完全自动化AI R&D流程 |
| 藏拙 | "战略性低表现" | 模型在评估上低表现；在OpenAI研究类别中 |
| 工具性推理 | "手段-目的推理" | 关于如何实现目标的推理；DeepMind监控的目标 |

## 进一步阅读

- [OpenAI — Updating our Preparedness Framework](https://openai.com/index/updating-our-preparedness-framework/) — v2公告。
- [OpenAI — Preparedness Framework v2 PDF](https://cdn.openai.com/pdf/18a02b5d-6b67-4cec-ab64-68cdfbddebcd/preparedness-framework-v2.pdf) — 完整文档。
- [DeepMind — Strengthening our Frontier Safety Framework](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — FSF v3公告。
- [DeepMind — Updating the Frontier Safety Framework (April 2026)](https://deepmind.google/blog/updating-the-frontier-safety-framework/) — 跟踪能力级别添加。
- [Gemini 3 Pro FSF Report](https://storage.googleapis.com/deepmind-media/gemini/gemini_3_pro_fsf_report.pdf) — FSF格式风险报告示例。
