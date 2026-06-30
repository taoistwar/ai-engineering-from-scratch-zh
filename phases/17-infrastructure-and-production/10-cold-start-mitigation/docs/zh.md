# 无服务器 LLM 的冷启动缓解

> 一个 20 GB 模型镜像从冷到服务需要 5-10 分钟（7B）到 20+ 分钟（70B）。在真正的无服务器世界中，这不是预热 — 这是一次停机。缓解措施在五个层面运作：预种节点镜像（AWS 的 Bottlerocket，双卷架构）、模型流式传输（NVIDIA Run:ai Model Streamer，vLLM 原生）、GPU 内存快照（Modal 检查点，重启速度提升高达 10 倍）、热池（`min_workers=1`）、分层加载（ServerlessLLM 的 NVMe→DRAM→HBM 管线，延迟降低 10-200 倍），以及将输入 token（KB）而非 KV 缓存（GB）传输的实时迁移。Modal 发布 2-4 秒冷启动作为下限；Baseten 默认 5-10 秒，预热下亚秒级。本课教你测量、预算和叠加这五层。

**Type:** Learn
**Languages:** Python (stdlib, toy cold-start path simulator)
**Prerequisites:** Phase 17 · 02 (Inference Platform Economics), Phase 17 · 03 (GPU Autoscaling)
**Time:** ~60 minutes

## 学习目标

- 列举冷启动缓解的五个层次，并说出每一层的一个工具或模式。
- 计算 70B 模型的总冷启动时间为（节点供应）+（权重下载）+（权重加载到 HBM）+（引擎初始化）之和。
- 解释为什么实时迁移传输输入 token（KB）而非 KV 缓存（GB）以及代价是什么（重计算）。
- 说出热池权衡（为空闲 GPU 付费或接受冷启动尾部）以及 `min_workers > 0` 成为强制时的 SLA 阈值。

## 问题

你的无服务器 LLM 端点在夜间缩容到零。早上 8 点流量激增。第一个请求等待，同时：

1. Karpenter 供应一个 GPU 节点：45-60 秒。
2. 容器拉取带权重的 30 GB 镜像：120-300 秒。
3. 引擎将权重加载到 HBM：45-120 秒，取决于模型大小和存储速度。
4. vLLM 或 TRT-LLM 初始化 CUDA 图、KV 缓存池、分词器：10-30 秒。

总计：220-510 秒（大约 3-8 分钟）才有一个 token 返回。你的 SLA 是 2 秒。你部署一个热池（`min_workers=1`）问题似乎消失了 — 但你现在为 24x7 的一台空闲 GPU 付费。如果你的服务有 5 个产品各带一个热副本，那就是 5 × 24 × 30 = 3,600 GPU-小时/月，无论是否有单个用户调用。

冷启动缓解是如何在保持无服务器经济性的同时逼近常驻延迟。

## 概念

### 第一层 — 预种节点镜像（Bottlerocket）

在 AWS 上，Bottlerocket 的双卷架构将操作系统与数据分离。在预拉取容器镜像的情况下快照数据卷；在 `EC2NodeClass` 中引用快照 ID。新节点在权重已位于本地 NVMe 的情况下启动 — 步骤 2 和步骤 3 的一部分消失。与 Karpenter 原生配合。典型节省：大型模型每次冷启动 2-4 分钟。

GCP 上的等效：带预烘焙容器层的自定义 VM 镜像。Azure 上：相同模式的托管磁盘快照。

### 第二层 — 模型流式传输（Run:ai Model Streamer）

不是在回答第一个请求之前加载完整文件，而是逐层将权重流式传输到 GPU 内存，并在第一个 Transformer 块驻留时立即开始处理。NVIDIA Run:ai Model Streamer 在 2026 年 vLLM 中原生提供。与 S3、GCS 和本地 NVMe 配合。通过在计算设置中重叠 I/O，将大型模型的权重加载时间大致减半。

### 第三层 — GPU 内存快照（Modal）

Modal 在首次加载后对 GPU 状态（权重、CUDA 图、KV 缓存区域）进行检查点。后续重启直接将快照反序列化到 HBM — 比重初始化快 10 倍。这是最接近"2 秒内启动热 GPU"的东西。权衡：快照是每 GPU 拓扑的，所以如果 Karpenter 将你迁移到不同 SKU，你需要重新检查点。

### 第四层 — 热池（min_workers=1）

最简单的缓解：保持一个副本始终就绪。成本是一台 GPU 的小时费率 24x7。算起来对小模型很残酷（你为 $0.85-$1.50/小时付费以避免 30 秒冷启动），对大模型则友好（支付 $4/小时以避免 5 分钟冷启动）。热池成为强制的 SLA 阈值：通常是 70B+ 模型的 TTFT P99 < 60 秒。

### 第五层 — 分层加载（ServerlessLLM）

ServerlessLLM 将存储视为分层结构：NVMe（快但大）、DRAM（中等但分层）、HBM（微小但即时）。权重预加载到 DRAM；按需加载到 HBM。论文报告冷加载相比天真的磁盘到 HBM 有 10-200 倍延迟降低。生产采用尚早，但与 vLLM 的集成已存在。

### 第六层 — 实时迁移（额外模式）

当节点不可用（竞价实例驱逐、节点排空），传统模式是冷启动另一个副本并排空请求队列。实时迁移将输入 token（千字节）传输到已加载模型的目标，并在目标上重计算 KV 缓存。重计算比通过网络传输 GB 级 KV 缓存便宜。适用于解耦部署。

### 热池数学

对于 P99 TTFT SLA 为 2 秒的服务，问题不是"热池是/否"，而是"多少个热副本，以及哪些路径获得它们"。

- 高价值交互路径（实时聊天、语音代理）：`min_workers=1-2`。
- 后台批处理路径（夜间分类）：接受缩容到零，5-10 分钟冷启动可容忍。
- 高级层：每租户 `min_workers`，带专用容量。

### 优化前先测量

新节点上 70B 模型的冷启动解剖（示意）：

| 阶段 | 时间 | 缓解 |
|-------|------|-----------|
| 节点供应 | 50s | Bottlerocket + 预种镜像、热池 |
| 镜像拉取 | 180s | 预种数据卷（消除） |
| 权重到 HBM | 75s | 模型流式传输（减半）；GPU 快照（消除） |
| 引擎初始化 | 20s | 持久化 CUDA 图缓存 |
| 首次前向 | 3s | 最小固有延迟 |
| **冷启动总计** | **328s** | |
| **缓解后总计** | **~15s** | 22 倍降低 |

### 你应该记住的数字

- Modal 冷启动：2-4 秒（使用 GPU 快照）。
- Baseten 默认冷启动：5-10 秒；预热下亚秒级。
- 原始 70B 冷启动：3-8 分钟。
- Run:ai Model Streamer：约 2 倍权重加载加速。
- ServerlessLLM 分层加载：10-200 倍延迟降低（论文数字）。

## 使用它

`code/main.py` 建模带和不带每种缓解的冷启动路径。报告总冷启动时间、热池成本以及热池自付盈亏的盈亏平衡请求率。

## 交付它

本课产出 `outputs/skill-cold-start-planner.md`。给定 SLA、模型大小和流量形状，选择要叠加的缓解措施。

## 练习

1. 运行 `code/main.py`。计算热副本比通过额外请求丢失支付冷启动税更便宜的盈亏平衡请求率。
2. 你部署一个 P99 TTFT SLA 为 3 秒的 13B 模型。选择实现它的最小缓解栈（最少层数）。
3. Bottlerocket 预种消除了镜像拉取，但权重仍然从快照加载到 HBM。如果快照支持的 NVMe 以 7 GB/s 读取，计算 70B 模型的墙上时钟时间。
4. 你的无服务器供应商提供 GPU 快照（Modal），你的团队拒绝，因为"快照泄露 PII"。论证双方 — 现实风险是什么，缓解措施是什么（短暂快照、加密、命名空间隔离）？
5. 设计分层热池策略：多少热副本给付费用户、试用用户和批处理工作负载？展示数学。

## 核心术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 冷启动 | "大暂停" | 从请求到新副本上第一个 token 的时间 |
| 热池 | "常驻最小值" | `min_workers >= 1` 以保持至少一个副本就绪 |
| 预种镜像 | "烘焙 AMI" | 容器权重已预驻留的节点镜像 |
| Bottlerocket | "AWS 节点操作系统" | 带双卷快照支持的 AWS 容器优化操作系统 |
| 模型流式传输 | "流式加载" | 将权重 I/O 与计算设置重叠 |
| GPU 快照 | "检查点到 HBM" | 序列化加载后 GPU 状态；重启时反序列化 |
| 分层加载 | "NVMe + DRAM + HBM" | 存储层级的分层；按需加载 |
| 实时迁移 | "移动 token" | 传输输入（KB），在目标上重计算 KV |
| `min_workers` | "热副本" | 无服务器最小保持活跃计数 |
| 缩容到零 | "完全无服务器" | 空闲时无成本；接受完整冷启动税 |

## 进一步阅读

- [Modal — 冷启动性能](https://modal.com/docs/guide/cold-start) — Modal 发布的基准和检查点架构。
- [AWS Bottlerocket](https://github.com/bottlerocket-os/bottlerocket) — 预种数据卷快照模式。
- [NVIDIA Run:ai Model Streamer](https://github.com/run-ai/runai-model-streamer) — 将权重加载与计算设置重叠。
- [Baseten — 冷启动缓解](https://www.baseten.co/blog/cold-start-mitigation/) — 预热手册。
- [ServerlessLLM 论文 (USENIX OSDI'24)](https://www.usenix.org/conference/osdi24/presentation/fu) — 分层加载设计。
- [NVIDIA — Kubernetes 上的解耦 LLM 推理](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/) — 解耦部署的实时迁移。
