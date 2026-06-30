# ReWOO 与 Plan-and-Execute：解耦式规划

> ReAct 将思考与行动交织在一个流中。ReWOO 将它们分离：先做一个大规划，然后执行。比 ReAct 少用 5 倍 token，在 HotpotQA 上准确率提升 4%，并且你可以将规划器蒸馏到一个 7B 模型中。Plan-and-Execute 将其泛化；Plan-and-Act 将其扩展到网页导航。

**类型:** Build
**语言:** Python（标准库）
**前置要求:** Phase 14 · 01（Agent Loop）
**时间:** ~60 分钟

## 学习目标

- 解释为什么 ReWOO 的 Planner / Worker / Solver 分离能比 ReAct 的交织式循环节省 token 并提高鲁棒性。
- 实现一个规划 DAG、一个依赖顺序执行器，以及一个组合 Worker 输出的求解器——全部使用标准库。
- 使用 2026 年 Anthropic 的"五种工作流模式"框架，判断一个任务应该采用先规划后执行还是交织式 ReAct。
- 识别何时需要 Plan-and-Act 的合成规划数据来处理长周期的网页或移动端任务。

## 问题

ReAct 的交织式思考-行动-观察循环简单而灵活，但每个工具调用都必须携带完整的先前上下文——包括此前的每一次思考。token 使用量随深度呈二次增长。更糟的是：当工具在循环中途失败时，模型必须从错误观察中重新推导整个计划。

ReWOO（Xu et al., arXiv:2305.18323, 2023 年 5 月）注意到了这一点并下了一个赌注：预先规划全部内容，并行获取证据，最后组合答案。一次 LLM 调用用于规划，N 次工具调用获取证据（可以并行），一次 LLM 调用用于求解。代价是灵活性降低（计划是静态的），换来了更好的 token 效率和更清晰的失败模式。

## 概念

### 三个角色

```
Planner:  user_question -> [plan_dag]
Workers:  [plan_dag]     -> [evidence]        （工具调用，可能并行）
Solver:   user_question, plan_dag, evidence -> final_answer
```

Planner 生成一个 DAG。每个节点命名一个工具、它的参数，以及它依赖哪些更早的节点（引用如 `#E1`、`#E2`）。Workers 按拓扑顺序执行节点。Solver 将所有内容缝合在一起。

### 为什么少用 5 倍 token

ReAct 的提示长度随步骤数线性增长。在第 10 步，提示包含思考 1 加行动 1 加观察 1 加思考 2 加行动 2 加观察 2，以此类推。每个中间步骤还冗余包含原始提示。

ReWOO 支付一次规划器提示（大），N 次小型 Worker 提示（每次只有工具调用，无链条），以及一次求解器提示。论文在 HotpotQA 上测量到约 5 倍的 token 减少，同时绝对准确率提升 4 分。

### 为什么更鲁棒

如果在 ReAct 中 Worker 3 失败，循环必须在流的中途推理出错误。在 ReWOO 中，Worker 3 返回一个错误字符串；求解器在原始计划的上下文中看到它，并可以优雅降级。失败定位是按节点的，而非按步骤的。

### 规划器蒸馏

论文的第二个成果：因为规划器不看到观察结果，你可以用 175B 教师模型的规划器输出微调一个 7B 模型。小模型处理规划；大模型在推理时不需要。这现在是标准做法——许多 2026 年的生产级 agent 使用小型规划器和大型执行器，或反之。

### Plan-and-Execute（LangChain, 2023）

LangChain 团队 2023 年 8 月的文章将 ReWOO 泛化为一个模式名称：Plan-and-Execute。前置规划器产出一个步骤列表，执行器运行每个步骤，一个可选的重规划器可以在观察到结果后进行修订。这比 ReWOO 更接近 ReAct（重规划器将观察带回规划中），但保留了 token 节省。

### Plan-and-Act（Erdogan et al., arXiv:2503.09572, ICML 2025）

Plan-and-Act 将该模式扩展到长周期的网页和移动端 agent。关键贡献是合成规划数据：一个标注轨迹生成器产生训练数据，其中计划是显式的。用于微调规划模型，使其在类似 WebArena 的任务上持续工作超过 30-50 步——而单个 ReAct 轨迹在此会失去连贯性。

### 何时选择哪种

| 模式 | 适用场景 |
|---------|------|
| ReAct | 短任务、未知环境、需要反应式异常处理 |
| ReWOO | 结构化任务、已知工具、token 敏感、可并行化的证据 |
| Plan-and-Execute | 类似 ReWOO 但在部分执行后可以重规划 |
| Plan-and-Act | 长周期（>30 步）、网页/移动端/计算机使用 |
| Tree of Thoughts | 搜索值得付出成本时（第 04 课） |

Anthropic 2024 年 12 月的指导：从最简单的开始。如果任务是一个工具调用加一个摘要，不要构建 ReWOO。如果任务是一个 40 步的研究任务，不要只用 ReAct。

## Build It

`code/main.py` 实现一个玩具 ReWOO：

- `Planner` — 一个脚本化策略，从提示中发出一个规划 DAG。
- `Worker` — 通过注册表调度每个节点的工具调用。
- `Solver` — 脚本化组合，读取证据并生成最终答案。
- 依赖解析 — 如 `#E1` 的引用被替换为先前 Worker 的输出。

该演示使用两步计划回答"法国首都的人口是多少，四舍五入到百万？"：（1）查找首都，（2）查找人口，然后求解。

运行它：

```
python3 code/main.py
```

追踪首先显示完整计划，然后是 Worker 结果，最后是求解器组合。将 token 计数（我们打印粗略的字符计数）与 ReAct 风格的交织式运行比较——在这种结构化任务上 ReWOO 胜出。

## Use It

LangGraph 将 Plan-and-Execute 作为配方提供（`create_react_agent` 用于 ReAct，自定义图用于 plan-execute）。CrewAI 的 Flows 直接编码该模式：你预先定义任务，Flow DAG 执行它们。Plan-and-Act 的合成数据方法仍主要在研究中；运行时模式（显式规划 DAG）通过 LangGraph 和 CrewAI Flows 在生产中交付。

## Ship It

`outputs/skill-rewoo-planner.md` 根据用户请求和工具目录生成一个 ReWOO 规划 DAG。它将计划验证（无环、每个引用都已解析、每个工具都存在），然后交给执行器。

## 练习

1. 并行化独立规划节点的 Worker 执行。在一个有 2 个并行组的 6 节点 DAG 上，它能带来什么好处？
2. 添加一个重规划器节点，当任何 Worker 返回错误时触发。将 ReWOO 变成 Plan-and-Execute 的最小改动是什么？
3. 用一个轻量模型（7B 级别）替换 `Planner`，并在前沿模型上保留 `Solver`。比较端到端质量——分离在哪里会失败？
4. 阅读 ReWOO 论文第 4 节关于规划器蒸馏的部分。从概念上复现 175B -> 7B 的结果：你需要什么训练数据，如何评估计划质量？
5. 将玩具移植到 Plan-and-Act 的轨迹形状：计划是一个序列，而不是 DAG。哪些权衡会改变？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| ReWOO | "无需观察的推理" | 先规划，然后并行获取证据，然后求解——规划提示中没有观察 |
| Plan-and-Execute | "LangChain 的 plan-execute 模式" | ReWOO 加上执行后的可选重规划器节点 |
| Plan-and-Act | "规模化 plan-execute" | 显式规划器/执行器分离，配合同义规划训练数据用于长周期任务 |
| 证据引用 | "#E1, #E2, ..." | 规划节点占位符，在分发时替换为先前的 Worker 输出 |
| 规划器蒸馏 | "小规划器，大执行器" | 用大型教师模型的规划器轨迹微调小型模型 |
| Token 效率 | "更少的往返" | 论文中在 HotpotQA 上比 ReAct 少用 5 倍 token |
| DAG 执行器 | "拓扑分发器" | 按依赖顺序运行规划节点；每层并行 |

## 进一步阅读

- [Xu et al., ReWOO: Decoupling Reasoning from Observations (arXiv:2305.18323)](https://arxiv.org/abs/2305.18323) — 经典论文
- [Erdogan et al., Plan-and-Act (arXiv:2503.09572)](https://arxiv.org/abs/2503.09572) — 采用合成计划的规模化 planner-executor
- [LangGraph Plan-and-Execute 教程](https://docs.langchain.com/oss/python/langgraph/overview) — 框架配方
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — 选择最简单可行的模式
