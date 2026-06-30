# 异步与 Hogwild! 推理

> 推测解码（第 10 阶段 · 第 15 课）在一个序列内并行化 token。多智能体框架跨整个序列并行化，但强制显式协调（投票、子任务拆分）。Hogwild! 推理（Rodionov 等人，arXiv:2504.06261）做了不同的事：运行 N 个相同 LLM 的实例并行对抗一个共享的键值缓存。每个 worker 立即看到每个其他 worker 生成的 token。现代推理模型——QwQ、DeepSeek-R1——可以通过该共享缓存自协调，而无需任何微调。该方法是实验性的，但它开辟了一个全新的推理并行轴，与推测解码正交。本课在 stdlib Python 中实现一个双 worker Hogwild! 模拟器，并解释为什么共享缓存协作从现有模型的推理能力中涌现。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 10 · 12 (inference optimization), Phase 10 · 15 (speculative decoding)
**Time:** ~60 minutes

## 学习目标

- 描述三种常见的并行 LLM 拓扑（投票、子任务、Hogwild!）并指出每个针对什么问题。
- 阐述核心 Hogwild! 设置：多个 worker，一个共享 KV 缓存，通过自提示涌现的协调。
- 计算 Hogwild! 的挂钟加速比作为 worker 数量 `N`、任务级并行度 `p` 和协调开销 `c` 的函数。
- 实现一个玩具问题上的双 worker Hogwild! 模拟器并观察涌现的任务分工。

## 问题

现代 LLM 通过产生长推理链来解决难题——5000 token 的逐步逻辑是常见的，在深度数学问题上数万 token 也会发生。在一个 70B 模型上以 35 tokens/sec 解码，50k token 是 24 分钟。交互式的模型不是。

推测解码（第 10 阶段 · 第 15 课）通过在一个序列内并行化给你 3-5x 加速。除此之外，自回归解码的顺序依赖是硬上限。每个新 token 依赖于之前的每个 token。

显而易见的问题：我们能跨序列并行化吗？在同一问题上运行同一模型的多个副本，让它们协作，让它们分工？

之前的工作：投票集成（运行 N 个模型，选择多数答案）、思维树（分支推理路径然后重新组合）和多智能体框架（分配每个智能体一个子任务，使用协调器）。这些在特定任务领域都有帮助。它们也都引入了显式协调机制——投票规则、分支和剪枝逻辑、智能体到智能体的消息传递协议。

Hogwild! 推理采取了不同的方法。N 个 worker 共享一个 KV 缓存。每个 worker 立即看到每个其他 worker 生成的 token，就好像它们是自己的上下文。worker——没有任何训练或微调——想办法分工。现代推理模型（QwQ、DeepSeek-R1、Claude 家族推理模式）可以读取共享缓存，然后说出诸如"我看到 worker 2 已经处理了基本情况，所以我来处理归纳步骤"之类的话。

加速是与工作负载相关的，并且截至 2026 年 4 月是实验性的。但这个想法值得了解，因为它开辟了一个新的推理并行轴。

## 概念

### 设置

初始化 N 个 worker 进程，全部运行相同的 LLM。不采用每个 worker 自己独立的 KV 缓存，而是维护一个共享缓存。当 worker `i` 生成 token `t_j` 时，该 token 被写入共享缓存的下一个位置。当 worker `k` 执行下一步时，它读取缓存的当前状态（包括所有 N 个 worker 到目前为止生成的一切）。

在步骤时，worker 竞争写入 token。没有每个 worker 的位置索引——缓存是一个单一的增长序列。顺序由写入到达时间决定。

### 为什么协调会涌现

Worker 共享一个提示。通常是类似"你是 N 个在此问题上协作的实例之一。每个实例读取共享内存，可以看到其他实例写了什么。避免多余工作。"的内容。提示加上共享缓存就足够了。推理模型读取缓存，注意到问题的哪些部分已经被尝试，然后（经常但不总是）转向未探索的部分。

Hogwild! 论文（Rodionov 等人，2025）报告了如下观察：

- Worker 制定计划并通过缓存与其他 worker 沟通这些计划。
- Worker 注意到其他 worker 推理中的错误并指出它们。
- Worker 在计划失败时适应并提出替代方案。
- 当被提示检查冗余时，worker 检测到它并转向。

这些都不需要微调。涌现行为来自模型已经拥有的推理能力。

### 命名

论文的名称戏仿了 Hogwild! SGD（Recht 等人，2011），一个异步更新优化器。类比：SGD 的异步 worker 都写入共享参数向量；Hogwild! 推理的 worker 都写入共享 KV 缓存。两者都依赖经验收敛而非同步保证。

### RoPE 使这变得可行

旋转位置嵌入（RoPE，Su 等人 2021）通过 Q 和 K 向量中的旋转编码位置信息。因为位置是旋转而不是硬编码偏移量，一个 token 的位置可以偏移而无需重新计算 KV 缓存条目。当 worker `i` 在位置 `p` 写入共享缓存时，读取该位置的其他 worker 可以直接使用缓存条目——不需要重新旋转。

在学习式位置或绝对位置模型中，Hogwild! 需要在每次并发写入时使缓存无效。RoPE 让缓存保持稳定。

### 挂钟时间数学

令 `T_serial` 为一个 worker 单独解决问题的时间。令 `p` 为任务级可并行化比例。令 `c` 为每步协调开销（读取扩展的缓存，决定写什么）。

单 worker 时间：`T_serial`。
N-worker Hogwild! 时间，如果协调是免费的：`T_serial * ((1 - p) + p / N)`。经典 Amdahl。
有协调开销：`T_serial * ((1 - p) + p / N) + c * steps_per_worker`。

对于一个 worker 要有产出，`c` 相对于每步解码时间必须很小。在产生 5k+ token 的推理模型上，worker 可以承受数百 token 的协调开销仍然盈利。在短对话任务上，协调占主导，Hogwild! 比串行更差。

### 具体例子

推理问题：10k token 的思维链。假设问题有 `p = 0.7` 可并行化内容（不同的证明策略、不同情况分析）和每个 worker `c = 200` token 的协调开销。使用 `N = 4` worker：

- 串行时间：10000 解码步。
- Hogwild! 时间：10000 * (0.3 + 0.7 / 4) + 200 * 4 = 10000 * 0.475 + 800 = 5550 解码步。
- 加速：10000 / 5550 = 1.8x。

那是适度的。但在更长的推理问题（50k token）上，协调开销摊销，加速推向 2.5-3x。Hogwild! 是在一种让你自然写多线程代码的语言中的线程级并行推理等价物。

### 何时使用 Hogwild!

- 长推理问题（数千 token），其中任务可以跨独立子目标并行化。
- 已被训练为逐步思考的推理模型。非推理模型自协调不佳。
- 具有足够 VRAM 容纳共享缓存加 N 个 worker 进程的单节点部署。缓存是共享的，但每个 worker 有自己的激活内存。

### 何时不用

- 短交互对话。协调开销占主导。
- 不可并行化的任务（单一线性证明、单一编译）。N=1 是最大值。
- 非推理模型。没有协调涌现。
- 多节点部署。共享缓存需要非常快速的跨 worker 同步。节点内可以；跨节点是延迟灾难。

### 实验状态

截至 2026 年 4 月，Hogwild! 是一种具有开源 PyTorch 实现的研究方法。生产采用尚未发生。三个障碍：

1. 跨并发进程的共享 KV 缓存管理是非平凡的工程。
2. 涌现协调是任务依赖的；基准测试仍在构建中。
3. 与推测解码已经提供的加速相比，加速是适度的，两者可以结合，但结合的工程是另一层。

值得了解。值得实验。还不值得在产品上下注。

```figure
continuous-batching
```

## 构建它

`code/main.py` 实现了一个玩具 Hogwild! 模拟器：

- 两个 worker 进程，每个是一个确定性的"LLM"，以已知概率产生几种 token 类别之一（工作 token、观察 token、协调 token）。
- 一个共享缓存（只是一个 token 列表），两个 worker 都读写。
- 一个简单的协调逻辑：当一个 worker 看到另一个已经在某类别中产生了足够多的工作 token，它选择不同类别。

模拟器运行固定的步骤预算并报告：

- 产生的总工作 token。
- 总挂钟时间（worker 步骤数）。
- 相对于单 worker 的有效加速。
- 哪个 worker 写了哪个 token 的追踪。

### 步骤 1：共享缓存

两个 worker 都追加的列表。真实实现中使用简单锁（Python `threading.Lock`）；我们用一个计数器模拟。

### 步骤 2：worker 循环

每个 worker，每一步：

- 读取当前共享缓存。
- 基于已经有什么内容决定写哪个类别的 token。
- 写一个 token。

### 步骤 3：协调启发式

如果类别 X 在缓存中已有 K 个 token，而 worker 的目标类别是 X，worker 切换到类别 Y。这是推理模型"注意这已被覆盖，改做其他事情"行为的玩具替代。

### 步骤 4：测量加速

以 N=1 worker 和 N=2 worker 运行模拟器，相同总步骤预算。统计产生的工作 token。N=2 应产生大约 1.5-1.8 倍更多的工作 token，因为协调驱动的任务分工。

### 步骤 5：压力协调

降低协调启发式的敏感度。再次运行。观察没有良好协调，N=2 冗余地产出相同 token，加速下降到 1 以下。这与论文的观察一致：该技巧仅在 worker 具有自协调推理能力时才有效。

## 使用它

截至 2026 年 4 月，Hogwild! 在生产中的集成是研究级别的。来自 Yandex/HSE/IST 的参考实现是基于 PyTorch 的，目标是在 DeepSeek-R1 和 QwQ 模型上的单节点多进程设置。

务实的采纳路径：

1. 对推理任务工作负载进行剖析。测量探索性的 token 比例（多个策略、案例分析、搜索）vs 线性。
2. 如果探索主导，运行一个双 worker Hogwild! 实验。测量挂钟时间提升。
3. 如果提升低于 1.3x，你处于协调主导的状态。回退到单 worker。
4. 如果提升超过 1.5x，推到 N=4 并重新测量。收益递减通常在大约 N=4-8 时击中。

与推测解码结合：每个 Hogwild! worker 可以独立使用推测解码。两个加速相乘（大致），将 3x 推测解码和 1.8x Hogwild! 带到相对于天真的单 worker 解码的有效 5.4x。

## 产出

本课产出 `outputs/skill-parallel-inference-router.md`。给定一个推理工作负载概况（token 预算、任务并行度概况、模型家族、部署目标），它在投票、思维树、多智能体、Hogwild! 和推测解码策略之间进行路由。

## 练习

1. 以默认设置运行 `code/main.py`。确认在相同挂钟时间内，N=2 Hogwild! 配置比 N=1 基线产生更多的工作 token。

2. 降低协调启发式的强度（设置 `coordination_weight=0.1`）。重新运行。展示加速崩塌。解释原因：当 worker 无法协调时，它们重复工作。

3. 为一个 50k token 推理任务计算期望 Hogwild! 加速，`p=0.8, c=500` 和 N=4 worker。对 1k token 对话任务，`p=0.3, c=200` 和 N=4 做同样的计算。为什么一个是赢另个是输？

4. 阅读 Hogwild! 论文的第 4 节（初步评估）。识别作者报告的两种失败模式。描述更好的协调提示如何可能缓解每种模式。

5. 在玩具中将 Hogwild! 与推测解码结合：每个 worker 内部使用 2-token 推测解码。报告乘法加速比。当两个 worker 都想扩展同一共享缓存前缀时，会出现什么记账问题？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| Hogwild! | "并行 worker，共享缓存" | 同一 LLM 的 N 个实例并发运行，具有一个共享 KV 缓存；通过自提示涌现协调 |
| 共享 KV 缓存 | "协调媒介" | 所有 worker 读写的单个增长 KV 缓冲区；使 token 在 worker 间即时可见 |
| 涌现协调 | "无需训练" | 具有推理能力的 LLM 可以读取共享缓存并分工，无需任何微调或显式协议 |
| 协调开销（c）| "花在定向上 token" | 每个 worker 读取扩展缓存并决定做什么的成本；必须相对于总解码时间保持很小 |
| 可并行化比例（p）| "什么可以并行运行" | 任务级并行度：不是固有顺序的总工作的比例 |
| RoPE 使 Hogwild! 成为可能 | "旋转位置是位移不变的" | 因为位置是旋转，写入共享缓存不需要重新计算之前的 token |
| 投票集成 | "运行 N 个，选择多数" | 最简单的并行推理拓扑；对分类有用，对长程推理则不然 |
| 思维树 | "分支和剪枝" | 推理策略，探索多个分支并剪枝；显式协调逻辑 |
| 多智能体框架 | "分配子任务" | 每个智能体获得一个角色；协调器指挥；沉重的协议开销 |

## 延伸阅读

- [Rodionov et al. — Hogwild! Inference: Parallel LLM Generation via Concurrent Attention (arXiv:2504.06261)](https://arxiv.org/abs/2504.06261) —— Hogwild! 论文，QwQ 和 DeepSeek-R1 上的初步评估
- [Recht, Re, Wright, Niu — Hogwild!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent (arXiv:1106.5730, NeurIPS 2011)](https://arxiv.org/abs/1106.5730) —— 原始 Hogwild!，命名来源
- [Su et al. — RoFormer: Enhanced Transformer with Rotary Position Embedding (arXiv:2104.09864)](https://arxiv.org/abs/2104.09864) —— RoPE，使共享缓存推理成为可能的属性
- [Yao et al. — Tree of Thoughts: Deliberate Problem Solving with Large Language Models (arXiv:2305.10601)](https://arxiv.org/abs/2305.10601) —— Hogwild! 与其正交的思维树推理策略
- [Leviathan et al. — Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192)](https://arxiv.org/abs/2211.17192) —— 推测解码，Hogwild! 与之组合的序列内并行
- [Hogwild! reference PyTorch implementation](https://github.com/eqimp/hogwild_llm) —— 论文实验的唯一权威来源
