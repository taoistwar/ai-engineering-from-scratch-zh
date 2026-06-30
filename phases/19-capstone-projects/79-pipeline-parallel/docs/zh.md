# 流水线并行和气泡分析

> 张量并行拆分跨 rank 的矩阵乘法。流水线并行拆分跨 rank 的模型，每个 rank 一个阶段。微批量流经流水线。开始和结束时的空时间是气泡；最小化它才是全部技艺。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 19 Track C lessons 42-49
**Time:** ~90 min

## Learning Objectives

- 将一个顺序模型拆分成 N 个阶段，并模拟一个跨 N 个 rank 的前向流水线。
- 使用 GPipe 调度（仅前向填充，然后反向）调度 M 个微批量流经流水线，并计算气泡比例。
- 将气泡与 Megatron-LM 和 PipeDream 中使用的交错 1F1B 调度进行比较。
- 辩护阶段分配：每个阶段相等计算量比每个阶段相等参数数量更重要。

## The Problem

一个 70B 参数的模型在 fp16 中仅参数就需要 140 GB。没有消费级 GPU 能容纳它。ZeRO-3 跨 rank 分片参数，但仍然需要每个 rank 对每个前向步骤 allgather 完整层，每层支付 log(N) 跳。流水线并行采取不同的路线：将模型切成 N 个阶段，每个 rank 上放一个阶段。层 1 的前向在 rank 0 上完成，并将激活张量交给 rank 1；rank 1 运行层 2 并交给 rank 2；依此类推。反向以相反方向流动。内存线性下降，因为每个 rank 只持有一个阶段；计算是顺序的，这是气泡问题。

气泡是流水线开始时的空闲时间（等待第一个微批量到达最后一个阶段）和结束时的空闲时间（等待最后一个微批量反向排空回来）。有 M 个微批量和 N 个阶段，每阶段气泡比例是 (N-1)/(M+N-1)。M=8，N=4 时是 27%。M=64，N=4 时是 4.5%。当每步有很多微批量时气泡收缩，这意味着每个微批量的批量较小，这是驱动微批量设计的约束。

## The Concept

```mermaid
flowchart LR
  R0[rank 0: stage 0 / layer 0] --> R1[rank 1: stage 1 / layer 1]
  R1 --> R2[rank 2: stage 2 / layer 2]
  R2 --> R3[rank 3: stage 3 / loss]
  R3 -.backward.-> R2
  R2 -.backward.-> R1
  R1 -.backward.-> R0
```

### GPipe 调度

在所有 M 个微批量开始任何反向之前填满流水线前向；然后以相反方向排空反向。每个微批量的激活必须保持到其反向，因此内存随 M 线性增长。前向花费 M+N-1 个周期，反向再花费 M+N-1 个周期。每阶段有用工作是 2M 个周期；每阶段气泡是 2(N-1) 个周期。当每个前向和反向花费一个时间单位时，气泡比例是 (N-1)/(M+N-1)。选择 M 远大于 N 可以隐藏气泡。

### 1F1B 调度

交错：一旦微批量的前向到达最后一个阶段，就启动其反向并让它流回来。调度在每个阶段交替一个前向和一个反向。气泡仍然是 N-1，但激活内存受限于流水线深度，而非微批量数量。生产流水线使用 1F1B（Megatron、PipeDream）。本课首先实现 GPipe，因为它更简单，1F1B 作为练习。

### 为什么每个阶段相等计算量很重要

如果阶段 0 花费 50 毫秒而阶段 1 花费 100 毫秒，每个周期都被阶段 1 卡住。其他阶段每周期空转 50 毫秒等待阶段 1 释放。相等参数数量是错误的轴：Transformer 的计算由每层注意力加 MLP 主导，嵌入层有很多参数但计算很少。阶段分配应均衡每阶段 FLOPs，而非每阶段权重。

### 微批量 vs 批次

流水线运行 M 个大小为 B 的微批量。有效批量大小是 M*B。流水线步骤结束时的梯度是组合的 M*B 示例上的梯度。气泡比例取决于 M；优化器看到 M*B。调整 M 意味着在气泡（高 M 时更低）和每微批量内存（高 M 时 GPipe 激活内存更高）之间权衡。

## Build It

`code/main.py` 实现了：

- `PipelineStage`：一个小的 `nn.Module`，持有一个阶段的参数并暴露 `forward(activation)`。
- `Pipeline(stages, num_microbatches)`：在模拟阶段上使用模拟每阶段挂墙时钟编排 GPipe 调度。
- `bubble_fraction(num_stages, num_microbatches)`：闭式 (N-1)/(M+N-1)。
- 一个 4 阶段演示，打印每微批量跟踪和测量的气泡比例。

运行方式：

```bash
python3 code/main.py
```

输出：一个阶段-微批量甘特图以及气泡百分比对比闭式预测。

## 生产实践

三种实践强化流水线并行到可交付水平。

**激活检查点与流水线配对。** 在 GPipe 上有 M 个飞行中的微批量，激活内存是 M 倍一个微批量。激活检查点在反向时间重新计算前向，用计算换内存；这种组合使得流水线对长序列变得可行。

**阶段平衡是测量的，而非假定的。** 生产团队运行一个分析过程，测量目标硬件上的实际每层计算（FLOPs 和挂墙时钟），然后按该测量结果分区。Megatron-LM 的 `--num-layers-per-stage` 标志接受列表以允许当阶段具有不同每层成本时的不均匀层计数。

**发送-接收调度必须避免死锁。** 每个阶段都先发送再接收的流水线会在线路上死锁。标准修复是交错：偶数 rank 阶段先发送然后接收，奇数 rank 阶段先接收然后发送。本课显式调度 rank，使模式可见。

## Use It

生产模式：

- **Megatron-LM。** 大规模流水线并行的参考。使用 1F1B 并支持张量 + 流水线 + 数据并行组合。
- **DeepSpeed Pipeline。** 与 ZeRO 集成；ZeRO-1 + 流水线是最大开放模型的常见组合。
- **PyTorch Pipe。** PyTorch 原生流水线包装器，构建在 `torch.distributed.pipeline.sync.Pipe` 上。

## Ship It

第 80 课将每阶段参数分片存储在分片检查点中。第 81 课在端到端演示上组合 DDP + ZeRO + 流水线（精神上；演示为运行时将流水线保持为模拟）。

## Exercises

1. 实现 1F1B 并验证气泡比例匹配 GPipe 但激活内存有界。
2. 在更深的模型上分析真实每阶段时间，并按测量的挂墙时钟重新平衡阶段。
3. 在流水线微批量上添加梯度累积，并检查梯度是否等于等效完整批次前向的梯度。
4. 将流水线与激活检查点配对，并测量内存下降 vs 计算成本。
5. 将流水线与 DDP 组合（每个流水线 rank 在数据并行组上复制），并推理 2D 调度。

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Pipeline | "沿深度的模型并行" | 每个 rank 一个阶段，激活从一个阶段流向另一个 |
| Bubble | "流水线空闲时间" | 开始和结束时的 (N-1) 步，其中一些阶段没有工作 |
| Microbatch | "批次的切片" | 一个前向/反向单元；气泡随 M 增长而收缩 |
| GPipe | "填充然后排空" | 所有 M 个前向在任何反向之前；高激活内存 |
| 1F1B | "交错调度" | 每个阶段一个前向一个反向；有界激活内存 |

## Further Reading

- [Huang et al, GPipe: Efficient Training of Giant Neural Networks](https://arxiv.org/abs/1811.06965)
- [Narayanan et al, PipeDream: Generalized Pipeline Parallelism for DNN Training](https://arxiv.org/abs/1806.03377)
- [Megatron-LM pipeline parallel docs](https://github.com/NVIDIA/Megatron-LM)
- Phase 19 Lesson 76 - 调度使用的发送/接收原语
- Phase 19 Lesson 78 - ZeRO 与流水线正交且经常组合
