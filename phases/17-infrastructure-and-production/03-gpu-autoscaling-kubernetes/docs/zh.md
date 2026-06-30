# Kubernetes 上的 GPU 自动扩展 — Karpenter、KAI Scheduler、Gang 调度

> 三个层次，不是一个。Karpenter 动态供应节点（不到一分钟，比 Cluster Autoscaler 快 40%）。KAI Scheduler 处理 gang 调度、拓扑感知和分层队列 — 它防止那个 8 缺 1 的部分分配陷阱，即七个节点等待而因为缺一个 GPU 而空烧资源。应用层自动扩展器（NVIDIA Dynamo Planner、llm-d Workload Variant Autoscaler）根据推理特定信号 — 队列深度、KV 缓存利用率 — 而非 CPU/DCGM 占空比来扩展。经典的 HPA 陷阱是 `DCGM_FI_DEV_GPU_UTIL` 是一个占空比指标：100% 可能是 10 个请求，也可能是 100 个。vLLM 预分配 KV 缓存内存，因此内存永远不会触发缩容。本课教你组合三个层次，并避开默认的 Karpenter `WhenEmptyOrUnderutilized` 策略，该策略会在推理中途终止正在运行的 GPU 作业。

**Type:** Learn
**Languages:** Python (stdlib, toy queue-depth autoscaler simulator)
**Prerequisites:** Phase 17 · 02 (Inference Platform Economics), Phase 17 · 04 (vLLM Serving Internals)
**Time:** ~75 minutes

## 学习目标

- 绘制三个自动扩展层次（节点供应、gang 调度、应用层），并命名每个层次使用的工具。
- 解释为什么 `DCGM_FI_DEV_GPU_UTIL` 是 vLLM 的错误 HPA 信号，并说出两个替代方案（队列深度、KV 缓存利用率）。
- 描述 gang 调度以及 KAI Scheduler 防止的部分分配失败模式（8 个 GPU 中 7 个空闲）。
- 说出会终止正在运行的 GPU 作业的 Karpenter 合并策略（`WhenEmptyOrUnderutilized`），并陈述 2026 年的安全替代方案。

## 问题

你的团队在 Kubernetes 上交付一个 LLM 推理服务。你使用 `DCGM_FI_DEV_GPU_UTIL` 作为信号设置了 HPA。该服务在工作时间内固定在 100% 利用率。HPA 从不扩展 — 它认为你已经满了。你手动添加一个副本；TTFT 下降。HPA 依然不扩展。这个信号在欺骗你。

另外，你对节点使用 Cluster Autoscaler。一个 100 万 token 的提示在凌晨 2 点到达；集群花了 3 分钟供应一个节点，而请求已经超时。

再另外，你部署一个需要 8 个 GPU 分布在 2 个节点上的 70B 模型。集群有 7 个空闲 GPU，还有 1 个分散在 3 个节点上。Cluster Autoscaler 为缺失的那 1 个 GPU 供应节点。七个节点空等 4 分钟烧钱，而 Kubernetes 还在启动最后的 GPU。

三个层次，三种不同的失败模式。2026 年的 GPU 感知自动扩展不是"开启 HPA"。它需要组合节点供应、gang 调度和应用信号自动扩展。

## 概念

### 第一层 — 节点供应 (Karpenter)

Karpenter 监控待调度 pod，约 45-60 秒内供应节点（Cluster Autoscaler 对 GPU 节点通常需要 90-120 秒）。它根据 `NodePool` 约束动态选择实例类型 — 如果你的 pod 需要 8 个 H100 而集群没有匹配的节点，Karpenter 直接供应一个，而不是扩展现有组。

**合并陷阱**：Karpenter 的默认 `consolidationPolicy: WhenEmptyOrUnderutilized` 对 GPU 池是危险的。它会在运行中终止一个 GPU 节点，将 pod 迁移到一个更便宜的合适大小的实例。对推理工作负载来说，这意味着驱逐运行中的请求并在新节点上重新加载一个 70B 模型。损失是数分钟的容量加请求失败。

GPU 池的安全设置：

```yaml
disruption:
  consolidationPolicy: WhenEmpty
  consolidateAfter: 1h
```

让 Karpenter 在一小时后合并真正空闲的节点，但绝不驱逐正在运行的作业。

### 第二层 — gang 调度 (KAI Scheduler)

KAI Scheduler（项目 "Karp" 后改名）处理默认 kube-scheduler 不处理的事情：

**Gang 调度** — 全有或全无调度。一个需要 8 个 GPU 的分布式推理 pod，要么 8 个一起启动，要么都不启动。没有这个，你就会陷入部分分配陷阱：8 个 pod 中 7 个启动，无限等待，烧钱。

**拓扑感知** — 知道哪些 GPU 共享 NVLink，哪些在同一机架上，哪些之间有 InfiniBand。相应放置 pod。一个 DeepSeek-V3 67B 张量并行工作负载必须保持在一个 NVLink 域内；KAI Scheduler 遵守这一点。

**分层队列** — 多个团队竞争同一 GPU 池，具有优先级和配额。团队 A 的生产级推理只在优先级规则允许时才被团队 B 的训练作业抢占。

KAI 作为辅助调度器部署在 kube-scheduler 旁边；你给工作负载加注解来使用它。Ray 和 vLLM 生产栈都有集成。

### 第三层 — 应用层信号

**HPA 陷阱**：`DCGM_FI_DEV_GPU_UTIL` 是一个占空比指标 — 它测量在每个采样间隔 GPU 是否在做工作。100% 利用率可能意味着 10 个并发请求，也可能是 100 个；GPU 在任何一种情况下都是忙碌的。基于占空比的扩展是盲目扩展。

更糟的是，vLLM 和类似引擎预分配 KV 缓存内存（最大到 `--gpu-memory-utilization`）。即使只有一个请求，内存使用也保持在约 90%。基于内存的 HPA 从不缩容。

**2026 年替代信号**：

- 队列深度（等待预填充的请求数）。
- KV 缓存利用率（活跃序列已分配块的占比）。
- 每副本 P99 TTFT（你的 SLA 信号）。
- 有效吞吐量（每秒满足所有 SLO 的请求数）。

NVIDIA Dynamo Planner 和 llm-d Workload Variant Autoscaler 消费这些信号并扩展副本。它们完全取代了 LLM 推理的 HPA。

### 何时使用什么

| 扩展决策 | 工具 |
|---------|------|
| 添加/移除节点 | Karpenter |
| 调度多 GPU 作业 | KAI Scheduler |
| 添加/移除副本 | Dynamo Planner / llm-d WVA（或自定义基于队列深度的 HPA） |
| 选择 GPU 类型 | Karpenter NodePool |
| 抢占低优先级 | KAI Scheduler 队列 |

### 解耦预填充/解码使一切复杂化

如果你运行解耦预填充/解码（Phase 17 · 17），你有两类 pod 具有不同的扩展触发器：预填充 pod 按队列深度扩展，解码 pod 按 KV 缓存压力扩展。llm-d 将这些暴露为独立的带有各自角色 HPA 的 `Services`。不要试图把单一 HPA 放在两者前面。

### 冷启动在这里也很重要

冷启动缓解（Phase 17 · 10）是节点供应时间变为用户可见的地方。Karpenter 的 45-60 秒预热加上 20GB 模型加载加上引擎初始化意味着一个从零开始的请求需要 2-5 分钟。对 SLO 关键路径保持一个热池（`min_workers=1`），或在应用层使用 Modal 风格的检查点。

### 你应该记住的数据

- Karpenter 节点供应：~45-60 秒 vs Cluster Autoscaler ~90-120 秒（GPU 节点）。
- KAI Scheduler 防止部分分配浪费 — 8 缺 1 陷阱。
- `DCGM_FI_DEV_GPU_UTIL` 作为 HPA 信号：有问题；使用队列深度或 KV 利用率。
- Karpenter `WhenEmptyOrUnderutilized`：终止正在运行的 GPU 作业。推理使用 `WhenEmpty + consolidateAfter: 1h`。

```figure
autoscaling
```

## 使用它

`code/main.py` 在突发 GPU 工作负载上模拟三层自动扩展器。比较朴素 HPA（占空比）、队列深度 HPA 和 KAI-gang 调度扩展。报告未满足请求数、空闲 GPU 分钟数和综合得分。

## 交付它

本课产出 `outputs/skill-gpu-autoscaler-plan.md`。给定集群拓扑、工作负载形状和 SLO，设计一个三层自动扩展计划。

## 练习

1. 运行 `code/main.py`。在突发工作负载下，朴素占空比 HPA 丢弃了多少队列深度 HPA 能捕获的请求？差距从何而来？
2. 为在 H100 SXM5 上运行 Llama 3.3 70B FP8 的集群设计一个 Karpenter NodePool。指定 `capacity-type`、`disruption.consolidationPolicy`、`consolidateAfter`，以及一个将非 GPU 工作负载排除在这些节点之外的污点。
3. 你的团队报告部署卡在 Pending 因为"GPU 可用但 pod 无法调度"。诊断 — 这是 Karpenter、kube-scheduler 还是 KAI Scheduler 的问题？哪些指标可以确认？
4. 为解耦预填充 pod 选择一个扩展信号，为解码 pod 选择一个不同的信号。论证两者。
5. 计算 `WhenEmptyOrUnderutilized` 合并陷阱在 7x24 生产服务上的成本，该服务平均每天有 60 次请求丢失事件，P99 TTFT > 10 秒。

## 核心术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Karpenter | "节点供应器" | Kubernetes 节点自动扩展器；亚分钟供应 |
| Cluster Autoscaler | "旧的扩展器" | Kubernetes 节点自动扩展器前代；更慢，基于组 |
| KAI Scheduler | "GPU 调度器" | 用于 gang + 拓扑 + 队列的辅助调度器 |
| Gang 调度 | "全有或全无" | 原子调度 N 个 pod 或全部延迟 |
| 拓扑感知 | "机架感知" | 基于 NVLink/IB/机架位置放置 pod |
| `DCGM_FI_DEV_GPU_UTIL` | "GPU 利用率" | 占空比指标；对 LLM 不是有效的扩展信号 |
| 队列深度 | "等待中的请求" | 对于预填充受限扩展的正确 HPA 信号 |
| KV 缓存利用率 | "内存压力" | 对于解码受限扩展的正确 HPA 信号 |
| 合并 | "Karpenter 合并" | 终止节点迁移到更便宜的实例类型 |
| `WhenEmpty + 1h` | "安全合并" | 不驱逐正在运行的 GPU 作业的策略 |

## 进一步阅读

- [KAI Scheduler GitHub](https://github.com/kai-scheduler/KAI-Scheduler) — 设计文档和配置示例。
- [Karpenter 中断控制](https://karpenter.sh/docs/concepts/disruption/) — 合并策略语义和 GPU 安全默认值。
- [NVIDIA — Kubernetes 上的解耦 LLM 推理](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/) — Dynamo Planner 扩展信号。
- [Ray 文档 — 用于 RayClusters 的 KAI Scheduler](https://docs.ray.io/en/latest/cluster/kubernetes/k8s-ecosystem/kai-scheduler.html) — Ray 集成模式。
- [AWS EKS 计算和自动扩展最佳实践](https://docs.aws.amazon.com/eks/latest/best-practices/aiml-compute.html) — 托管 Kubernetes 特定指导。
- [llm-d GitHub](https://github.com/llm-d/llm-d) — Workload Variant Autoscaler 设计。
