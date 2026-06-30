# DualPipe 并行

> DeepSeek-V3 在 2,048 张 H800 GPU 上训练，MoE 专家分散在节点间。跨节点专家 all-to-all 通信每 1 GPU 小时计算消耗 1 GPU 小时通信。GPU 有一半时间处于空闲。DualPipe（DeepSeek，2024 年 12 月）是一个双向流水线，将前向和后向计算与它们触发的 all-to-all 通信重叠在一起。气泡消失，吞吐量攀升，而保持两个模型参数副本（赋予名称的"dual"部分）在 Expert Parallelism 已经将专家分散到各个 ranks 的情况下是廉价的。本课是一个 Learn 类型的详解，讲解 DualPipe 实际做了什么，以及为什么 Sea AI Lab 的 DualPipeV 精炼版在略微更大气泡的代价下放弃了 2x 参数复制的成本。

**Type:** Learn
**Languages:** Python (stdlib, schedule simulator)
**Prerequisites:** Phase 10 · 05 (distributed training, FSDP, DeepSpeed), Phase 10 · 14 (open-model architectures and MoE)
**Time:** ~60 minutes

## 学习目标

- 命名一个 DualPipe 前向-后向块的四个组成部分以及为什么每个部分获得自己的重叠窗口。
- 解释大规模下的流水线气泡问题，以及"无气泡"在实践中与在营销中的含义。
- 手动追踪一个 8 个 PP ranks 和 16 个微批次的 DualPipe 时间表，并确认正向和反向流填补了彼此的空闲时间槽。
- 阐述 DualPipeV（Sea AI Lab，2025）所做的权衡：在当 Expert Parallelism 不活跃时略大气泡的代价下放弃了 2x 参数复制。

## 问题

在 2k H800 GPU 上训练一个 671B MoE 模型会遇到三个复合瓶颈：

1. **内存压力。** 每个 GPU 持有模型的一个切片。128 个头 61 层序列 8k 的激活内存是巨大的。
2. **流水线气泡。** 传统流水线并行（GPipe、1F1B）在等待其阶段的输入或梯度时让 GPU 空闲。8 个阶段下，即使使用 1F1B 调度，大约 12% 的 GPU 时间可能成为气泡。
3. **跨节点 all-to-all。** 带有专家并行的 MoE 将专家分散在节点之间。每次前向传递触发一个 all-to-all 将 token 分发到它们的专家，以及另一个来合并。在 2k GPU 上，这很容易变成 1:1 的计算对通信比率。

其中每一个都有独立的解决方案：梯度检查点用于内存，Zero Bubble（Sea AI Lab，2023）用于流水线气泡，专家并行通信 kernel 用于 all-to-all。DualPipe 所做的就是让它们协同工作。调度在单个前向-后向块内重叠计算和通信，从流水线两端同时注入微批次，并使用所得调度将 all-to-all 隐藏在计算窗口内。

报告结果：几乎消除了流水线气泡，DeepSeek-V3 的 14.8T token 训练运行中 GPU 利用率超过 95%。

## 概念

### 流水线并行复习

将一个 N 层模型分布在 P 个设备上。设备 `i` 持有层 `i * N/P .. (i+1) * N/P - 1`。一个微批次前向流过设备 0 到 P-1，然后从 P-1 到 0 反向流动。每个设备只有在之前的设备发送其输出后才能开始其前向阶段，只有在后一设备发送上传梯度后才能开始后向阶段。

GPipe（Huang 等人，2019）一次调度一个微批次，浪费了大部分 GPU 时间。1F1B（Narayanan 等人，2021）为多个微批次交错前向和后向传递。Zero Bubble（Qi 等人，2023）将后向传递分成两个部分——backward-for-input (B) 和 backward-for-weights (W)——并调度它们以填补气泡。在 Zero Bubble 之后，流水线几乎是紧凑的。

DualPipe 是其上的下一步。它在此基础上添加了两个思想：

### 思想 1：块分解

每个前向块被分成四个组件：

- **注意力。** Q/K/V 投影、注意力、输出投影。
- **All-to-all 分发。** 将 token 发送到其专家的跨节点通信。
- **MLP。** MoE 专家计算。
- **All-to-all 合并。** 将专家输出带回的跨节点通信。

后向块添加每个的梯度版本。DualPipe 对它们进行调度，使 all-to-all 分发与下一个块的注意力计算并行发生，all-to-all 合并与随后的 MLP 计算并行发生。

### 思想 2：双向调度

大多数流水线调度从阶段 0 注入微批次并流向阶段 P-1。DualPipe 从两端注入微批次。阶段 0 看到从那里发起的正向微批次；阶段 P-1 同样看到从那里发起的正向微批次。两个流在中间相遇。

为此，设备 `i` 必须同时持有早期流水线层 `i` 和后期流水线层 `P - 1 - i`。这就是 DualPipe 的"dual"部分：每个设备保持其需要服务的两个模型层的副本（每个方向一个）。在 DeepSeek-V3 的规模上，这是 2x 的参数复制成本。它是可负担的，因为 Expert Parallelism 已经将 MoE 专家分散得如此稀疏，以至于复制两次非专家层微不足道。

关键的是，一个方向上的前向流和另一个方向上的后向流在单方向调度会有气泡的地方精确重叠。气泡消失了。

### 手动追踪的调度

考虑 P = 4 ranks，8 个微批次，分 4 个前向 / 4 个反向。时间从左到右；行是设备 ranks。

```
           Time →
rank 0:  F1 F2 F3 F4  F5R F6R F7R F8R  B1 B2 B3 B4  ...
rank 1:     F1 F2 F3  F4/F5R F6R F7R   B1 B2 ...
rank 2:        F1 F2  F3/F5R F4/F6R    B1 ...
rank 3:           F1  F2/F5R F3/F6R    ...
```

解读 "F4/F5R" 标记：rank 1 在同一时间槽中运行微批次 4 的前向（在流水线中从左到右）和微批次 5 的前向（从右到左）。这就是"双向"的操作含义。

在 rank 2 交叉流重叠更早，在 rank 0 和 P-1 重叠最晚。在调度的稳定中期阶段，每个 rank 运行 X 方向的前向与 Y 方向的后向重叠在一起。计算是繁忙的。前向传递的 all-to-all 分发隐藏在后向计算内。all-to-all 合并隐藏在前向计算内。气泡被挤出了。

### 气泡计数

标准 1F1B 流水线气泡（每个 rank 浪费的时间）：

```
bubble_1F1B = (P - 1) * forward_chunk_time
```

Zero Bubble 精炼版将其降低但未降到零。DualPipe，在稳定阶段，如果微批次计数可被 2 倍的流水线深度整除，气泡为零。在稳定阶段之外（预热和冷却），有一些气泡但它不随微批次数量增长——这是论文强调的一个关键属性。

在营销术语中："无气泡"。在技术术语中：气泡不随微批次计数增长。Sea AI Lab 的后续分析（DualPipeV / Cut-in-half）表明，完全零气泡仅当 Expert Parallelism 不是瓶颈时才成立；在有 EP 驱动的 all-to-all 时，总会有一些调度妥协。

### DualPipeV——精炼版

Sea AI Lab（2025）观察到，当 EP 通信重叠不是重点时，2x 参数复制是浪费的。他们的 DualPipeV 调度将双向注入折叠为一个"V 形"调度，在单个参数副本上运行。气泡比 DualPipe 略大，但内存节省是显著的。DeepSeek 在其开源的 DualPipe 实现中采用了 DualPipeV 作为一个 EP-off 模式。

权衡：

| 特性 | DualPipe | DualPipeV | 1F1B | Zero Bubble |
|---------|---------|-----------|------|------------|
| 每个设备的参数副本 | 2 | 1 | 1 | 1 |
| 气泡 vs 微批次 | 常量 | 小幅增长 | 增长 | 增长 |
| 计算-通信重叠 | 完全 | 部分 | 最小 | 部分 |
| 何时使用 | EP 重的 MoE | 密集或 EP 轻 | 基线 | 任何流水线 |

### 它对一个 14.8T token 运行意味着什么

DeepSeek-V3 的预训练在 2,048 张 H800 GPU 上消耗了 14.8T token，大约 2.8M GPU 小时。使用天真的 1F1B，他们将损失其中的 12-15% 于流水线气泡——340-420K GPU 小时，足以训练一个完整的 70B 模型。DualPipe 收回了其中大部分。在没有内部日志的情况下直接量化其贡献是困难的，但论文中的声明是训练期间平均 GPU 利用率超过 95%。

对于较小的运行（1k GPU 以下），DualPipe 是过于庞大的——相对于总成本，流水线气泡较小，密集模型训练很少达到 all-to-all 瓶颈。对于数千 GPU 规模的前沿 MoE 训练，它实际上是必需的。

### 它位于栈中的位置

- 与 **FSDP**（第 10 阶段 · 第 05 课）互补。FSDP 将模型参数分片到 ranks；DualPipe 将计算调度到 ranks。它们结合在一起。
- 与 **ZeRO-3** 梯度分片兼容。双副本复制的记账需要与 ZeRO 的分片梯度协作。
- 需要针对特定集群拓扑调优的**自定义 all-to-all kernel**。DeepSeek 的开源 kernel 是参考实现。

```figure
expert-capacity
```

## 使用它

`code/main.py` 是一个流水线调度模拟器。它接受 `(P, n_micro_batches, schedule)` 并打印 1F1B、Zero Bubble、DualPipe 和 DualPipeV 各自的稳定阶段利用率。它是一个教学工具——数字匹配论文中的定性声明，它们并非对生产测量加速比的声明。

模拟器的价值：以不同的 P 和微批次计数运行它，观察气泡比例如何对 1F1B 增长而对 DualPipe 不增长。

真实训练运行中的集成考虑：

- 选择一个能被微批次计数整除的流水线并行深度。
- 确保你的专家并行网格支持双向 all-to-all。DeepSeek 的 kernel 是参考。
- 预期第一次需要花费一周调试调度本身。记账很繁琐。
- 监控每个 rank 的 GPU 利用率，而不仅仅是聚合值。DualPipe 的收益来自收紧落后者。

## 产出

本课产出 `outputs/skill-dualpipe-planner.md`。给定一个训练集群规格（GPU 数量、拓扑、互连、模型形状），它推荐一个流水线并行策略、使用的调度算法以及在目标规模上的期望气泡比例。

## 练习

1. 在 `(P=8, micro_batches=16, schedule=dualpipe)` 和 `(P=8, micro_batches=16, schedule=1f1b)` 上运行 `code/main.py`。计算 GPU 利用率差异并将其表示为每百万训练 token 回收的 GPU 小时。

2. 手动草绘 `(P=4, micro_batches=8, schedule=dualpipe)` 的调度表。用微批次 ID 和方向标记每个时间槽。识别气泡消失的第一个时间槽。

3. 阅读 DeepSeek-V3 技术报告（arXiv:2412.19437）的图 5。识别 DualPipe 前向块内 all-to-all 分发的重叠窗口。解释计算调度如何隐藏它。

4. 为一个 70B 密集模型（P=8 流水线阶段）和一个 671B MoE 模型（P=16 流水线阶段）计算 DualPipe 的 2x 参数开销。展示为什么 MoE 情况的开销比例更小（大多数参数是专家，分散在一个大的 EP 组中）。

5. 比较 DualPipe 与 Chimera（一个 2021 年的竞争性双向调度器）。使用论文第 3.4 节作为参考，识别 DualPipe 添加的而 Chimera 没有的两个特定属性。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| 流水线气泡 | "每个 rank 的空闲时间" | 因为流水线阶段等待其输入或梯度而浪费的 GPU 周期 |
| 1F1B | "默认流水线调度" | 一个前向 / 一个后向交错调度；DualPipe 击败的基线 |
| Zero Bubble | "Sea AI Lab 2023" | 将后向拆分为 B（输入梯度）和 W（权重梯度）；几乎完全紧凑化流水线 |
| DualPipe | "DeepSeek-V3 调度" | 双向流水线 + 计算-通信重叠；气泡不随微批次计数增长 |
| DualPipeV | "Cut-in-half" | V 形精炼版，在略大气泡的代价下放弃 2x 参数复制 |
| 块 | "流水线工作单元" | 一个微批次通过一个流水线阶段的前向或后向传递 |
| All-to-all 分发 | "发送 token 到专家" | 将 token 路由到其分配的 MoE 专家的跨节点通信 |
| All-to-all 合并 | "将专家输出带回" | MLP 之后收集专家输出的跨节点通信 |
| Expert Parallelism (EP) | "专家跨 GPU" | 将 MoE 专家分片到 ranks，使不同的 GPU 持有不同的专家 |
| Pipeline Parallelism (PP) | "层跨 GPU" | 将模型层分片到 ranks；DualPipe 调度的维度 |
| 气泡比例 | "浪费的 GPU 时间" | (bubble_time / total_time)；DualPipe 驱动到零的比例 |

## 延伸阅读

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437), Section 3.3.2 and Figure 5](https://arxiv.org/abs/2412.19437) —— 主要 DualPipe 参考
- [DeepSeek — DualPipe GitHub repository](https://github.com/deepseek-ai/DualPipe) —— 开源参考实现，包括 DualPipeV (Cut-in-half) 模式
- [Qi et al. — Zero Bubble Pipeline Parallelism (arXiv:2401.10241, Sea AI Lab 2023)](https://arxiv.org/abs/2401.10241) —— Zero Bubble 前身
- [Sea AI Lab — DualPipe could be better without the Dual](https://sail.sea.com/blog/articles/63) —— DualPipeV 分析，为 DeepSeek 的 EP-off 模式提供了信息
- [Narayanan et al. — PipeDream / 1F1B (arXiv:1806.03377, 2018-2021)](https://arxiv.org/abs/1806.03377) —— DualPipe 比较的 1F1B 调度
- [Huang et al. — GPipe (arXiv:1811.06965, 2018)](https://arxiv.org/abs/1811.06965) —— 原始流水线并行论文和气泡问题
