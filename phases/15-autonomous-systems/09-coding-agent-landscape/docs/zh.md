# 自主编程智能体格局（2026年）

> SWE-bench Verified在不到三年内从4%上升到80.9%。同一个Claude Sonnet 4.5在SWE-agent v1上得分为43.2%，在Cline自主模式上得分为59.8%——模型周围的脚手架现在与模型本身一样重要。OpenHands（前身为OpenDevin）是最活跃的MIT许可平台，其CodeAct循环在沙盒中直接执行Python动作，而不是JSON工具调用。头条数字掩盖了一个方法论问题：500个SWE-bench Verified任务中有161个只需要1-2行变更，而SWE-bench Pro（10行以上任务）对同样的前沿模型得分仅为23-59%。

**Type:** Learn
**Languages:** Python (stdlib, CodeAct vs JSON tool-call comparison)
**Prerequisites:** Phase 14 · 07 (Tool use), Phase 15 · 01 (Long-horizon agents)
**Time:** ~45 minutes

## 问题

"哪个编程智能体最好"是错误的问题。正确的问题是：在与我工作匹配的任务分布上，在我将在生产中运行的脚手架下，我得到的端到端可靠性是多少？

从2022年到2026年，该领域认识到脚手架——检索层、规划器、沙盒、编辑-验证循环、反馈格式——是承重的。Claude Sonnet 4.5在SWE-agent v1上在SWE-bench Verified上得分为43.2%；同一个模型在Cline的自主脚手架内得分为59.8%。16.6个绝对百分点的差异，相同的权重。基础模型是一个组件；循环是产品。

伴随的问题是基准饱和隐藏了退步。SWE-bench Verified接近饱和，简单任务尾巴（500个任务中161个需要≤2行）拉高了最高分。现实世界的质量在如SWE-bench Pro（10行以上变更）这样的分布上测量得更好，同样的领先者仍处于23-59%。

## 概念

### SWE-bench，一段话概括

SWE-bench（Jimenez等人）取真实的GitHub Issue及其真实补丁，要求智能体生成一个使测试套件通过的补丁。SWE-bench Verified（OpenAI，2024年）是一个人工策展的500任务子集，移除了模糊和破损的任务。SWE-bench Pro是更难的继任者——需要10行以上变更的任务，当前前沿智能体处于23-59%。

### 2022 → 2026曲线实际展示的内容

- **2022年**：研究模型在原始SWE-bench上约4%。
- **2024年**：GPT-4 + Devin风格脚手架约14%；SWE-agent约12%。
- **2025年**：Claude 3.5/3.7 Sonnet在Aider和SWE-agent内推进到40-55%区间。
- **2026年**：Claude Sonnet 4.5和前沿竞争者在SWE-bench Verified上达到70-80%以上。Epoch AI的排行榜实时跟踪这一数据。

斜率来自三个复利增长的来源：更好的基础模型、更好的脚手架（CodeAct、反思、验证器循环）以及更好的基准（Verified移除了噪声）。

### CodeAct vs JSON工具调用

OpenHands（All-Hands-AI，arXiv:2407.16741，前身为OpenDevin）做出了一个特定的架构赌注：模型不再发出由宿主机解码和执行的JSON工具调用，而是发出Python代码，由Jupyter风格的内核在沙盒中运行。智能体可以在一次动作中遍历文件、链式调用工具并捕获自己的异常。

权衡：

- **JSON工具调用**：每个动作一个回合；易于审计；组合性有限；默认安全，因为每次调用都通过显式验证器。
- **CodeAct**：一个动作可以是一个完整的程序；可组合性强；需要强化的沙盒（OpenHands使用Docker隔离）；失败模式包括沙盒运行时允许的任何操作。

两种架构都在生产中使用。CodeAct在开源平台（OpenHands、smolagents）中占主导地位。JSON工具调用在托管服务（Anthropic Managed Agents、OpenAI Assistants）中仍然占主导地位，提供者控制执行器。

### 2026年格局中的脚手架

| 脚手架 | 许可证 | 执行模型 | 显著特性 |
|---|---|---|---|
| OpenHands (OpenDevin) | MIT | Docker中的CodeAct | 最活跃的开源平台；事件流可重放 |
| SWE-agent | MIT | 智能体-计算机接口（ACI） | 第一个端到端SWE-bench脚手架 |
| Aider | Apache-2 | 本地仓库中通过diff编辑 | 极简脚手架，回归稳定性强 |
| Cline | Apache-2 | 带工具策略的VS Code智能体 | 在Sonnet 4.5上得分最高的开源脚手架 |
| Devin (Cognition) | 专有 | 托管VM + 规划器 | 第一个"AI软件工程师"产品类别 |
| Claude Code | 专有 | 权限模式 + 例程 | 第10课详细讲解智能体循环 |

### 为什么脚手架占主导地位

一次编程运行是一个长周期轨迹（第1课）。可靠性在步骤间复利增长。脚手架在三个地方赢得分数：

1. **检索**：找到正确的文件来阅读是无声的瓶颈。SWE-agent的ACI、OpenHands的文件索引和Aider的仓库地图都在攻克这一点。
2. **验证器循环**：运行测试、阅读堆栈跟踪并重试，在SWE-bench上是10分以上的增量。
3. **故障隔离**：在错误时回滚的沙盒防止复利增长的损坏。使用和不使用验证器循环的同一模型看起来像是两个不同的产品。

### 基准饱和和真实分布

OpenHands的作者和Epoch AI都指出，SWE-bench Verified有一个简单的尾巴：500个任务中161个只需要1-2行变更。高分数部分由这个尾巴驱动。SWE-bench Pro限制为10行以上的变更，即使对前沿系统也返回23-59%的分数。你的生产分布几乎肯定更接近Pro而不是Verified。

选择智能体的含义：在你自己的Bug积压中运行一个类似Pro的子集。重要的分数是在代表你交付内容的代表性任务上的分数。

## 运用

`code/main.py` 在固定的迷你任务分布上比较两个玩具智能体脚手架：

1. 每回合执行一个动作的 **JSON工具调用** 脚手架。
2. 每个动作可以发出一个小型Python代码片段的 **CodeAct** 脚手架。

两者都使用一个桩"模型"（确定性规则），这样比较可以将脚手架与模型质量隔离开来。输出显示CodeAct脚手架在更少的回合内解决更多任务，代价是每次动作的爆炸半径更大。

## 交付物

`outputs/skill-scaffold-audit.md` 帮助你在采用之前审计一个提议的编程智能体脚手架：检索质量、验证器存在、沙盒隔离以及基准到分布的匹配度。

## 练习

1. 运行 `code/main.py`。在相同的任务集上每种脚手架需要多少回合？每种脚手架的每次动作爆炸半径是多少？

2. 阅读OpenHands论文（arXiv:2407.16741）。论文认为CodeAct在复杂任务上击败JSON工具调用。确定论文承认的一种失败模式，写一句话说明该模式何时会在生产中占主导地位。

3. 从你的Bug积压中选择一个需要在两个文件间进行10行以上变更的任务。估计前沿模型在(a) JSON工具调用和(b) CodeAct下的端到端成功概率。证明差距。

4. SWE-bench Verified有161个单文件、1-2行任务。构建一个排除它们的分数。排行榜会如何洗牌？

5. 阅读"Introducing SWE-bench Verified"（OpenAI）。解释用于移除模糊任务的具体方法论，并列举策展会遗漏的一个类别。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|---|---|---|
| SWE-bench | "编程基准" | 带有真实补丁和测试套件的真实GitHub Issue |
| SWE-bench Verified | "清理后的子集" | 500个人工策展任务，存在简单任务尾巴 |
| SWE-bench Pro | "更难的子集" | 10行以上变更；前沿水平处于23-59% |
| CodeAct | "代码即动作" | 智能体发出Python；Jupyter风格内核在沙盒中执行 |
| JSON工具调用 | "函数调用" | 每个动作是在执行前验证的结构化JSON负载 |
| 脚手架 | "智能体框架" | 基础模型周围的检索 + 规划器 + 执行器 + 验证器循环 |
| ACI（智能体-计算机接口） | "SWE-agent的格式" | 为LLM人机工程设计的命令集，而非人类shell |
| 验证器循环 | "测试并重试" | 运行测试，阅读输出，修改补丁；最大的非模型可靠性增益 |

## 进一步阅读

- [Jimenez et al. — SWE-bench](https://www.swebench.com/) — 原始基准和方法论。
- [OpenAI — Introducing SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — 策展子集是如何构建的。
- [Wang et al. — OpenHands: An Open Platform for AI Software Developers](https://arxiv.org/abs/2407.16741) — CodeAct架构和事件流设计。
- [Epoch AI — SWE-bench leaderboard](https://epoch.ai/benchmarks) — 实时追踪的分数。
- [Anthropic — Measuring agent autonomy](https://www.anthropic.com/research/measuring-agent-autonomy) — 长周期编程智能体可靠性框架。
