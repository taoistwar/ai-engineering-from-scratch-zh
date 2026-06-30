# 解耦预填充/解码 — NVIDIA Dynamo 和 llm-d

> 预填充是计算受限的；解码是内存受限的。在同一 GPU 上运行两者就浪费了一种资源。解耦将它们分配到独立的池中，并通过 NIXL（RDMA/InfiniBand 或 TCP 降级）在它们之间传输 KV 缓存。NVIDIA Dynamo（GTC 2025 宣布，1.0 GA）位于 vLLM/SGLang/TRT-LLM 之上 — 其 Planner Profiler + SLA Planner 自动调整预填充:解码比率以满足 SLO。NVIDIA 发布了在此范围内的吞吐量增益 — developer.nvidia.com（2025-06）展示了 DeepSeek-R1 MoE 在 GB200 NVL72 + Dynamo 上中等延迟区域的约 6 倍改进，Dynamo 产品页面（developer.nvidia.com，未注明日期）宣传 GB300 NVL72 + Dynamo 上最高 50 倍 MoE 吞吐量 vs Hopper。"30 倍"的数字是跨完整 Blackwell + Dynamo + DeepSeek-R1 栈的社区综合数据；我们没有找到单一原始来源明确陈述 30 倍，因此将其视为方向性声明。llm-d（Red Hat + AWS）是 Kubernetes 原生的：预填充/解码/路由器作为独立 Services 并带有各自角色的 HPA。llm-d 0.5 添加了分层 KV 卸载、缓存感知 LoRA 路由、UCCL 网络、缩容到零。经济学：从多个客户披露综合得出，在恒定 SLA 下从并置推理切换到带 Dynamo 的解耦推理，$2M 级别的推理支出可节省 30–40%（即 $600-800K/年）；具体 $2M→$600-800K 的数字是内部综合，不是单一发表的案例研究 — 将其用作数量级锚定，而非参考引用。短提示（<512 token，短输出）不值得传输成本。

**Type:** Learn
**Languages:** Python (stdlib, toy disaggregated-vs-colocated simulator)
**Prerequisites:** Phase 17 · 04 (vLLM Serving Internals), Phase 17 · 08 (Inference Metrics)
**Time:** ~75 minutes

## 学习目标

- 解释为什么预填充和解码有不同的最优 GPU 分配，并量化并置时的浪费。
- 绘制解耦架构：预填充池、解码池、通过 NIXL 的 KV 传输、路由器。
- 说出解耦不划算的条件（短提示、短输出）。
- 区分 NVIDIA Dynamo（栈上层）和 llm-d（Kubernetes 原生），并将每个匹配到运维上下文。

## 问题

你在 8 个 H100 上运行 Llama 3.3 70B。在混合工作负载（长提示 + 短输出）下，GPU 在解码期间空闲，因为大部分计算花费在预填充上。在不同工作负载（短提示 + 长输出）下，相反的情况发生。并置的预填充 + 解码意味着你两者都过度配置。

预算影响：20-40% 的 GPU 时间浪费在错误的资源上。你在购买 H100 计算来运行内存受限的解码，或购买 H100 HBM 带宽来运行计算受限的预填充。两者都是昂贵的浪费。

解耦将预填充和解码分配到独立的池中，每个池针对各自的瓶颈进行大小调整。KV 缓存通过高带宽互连从预填充池传输到解码池。

## 概念

### 为什么瓶颈不同

**预填充** — 在一次前向传播中对整个输入提示运行 transformer。矩阵乘法占主导；计算受限。H100 FP8 提供约 2000 TFLOPS 的有用吞吐量。批处理效率好 — 一次前向处理许多 token。

**解码** — 每次生成一个 token，每次迭代读取完整权重。内存带宽受限。HBM3 提供约 3 TB/s。批处理效率仅在高并发时好 — 权重读取在批次间分摊。

并置它们：你购买对两者都优化的 GPU。H100 两者都擅长但无论怎样成本相同。规模化时，你希望预填充池在 H100/计算密集型上；解码池在 H200/内存密集型上，或使用激进的量化。

### 架构

```
            ┌──────────────┐
  Request → │    Router    │ ───────────────────────┐
            └──────┬───────┘                        │
                   │                                │
                   ▼ (prompt only)                  │
            ┌──────────────┐    KV cache    ┌───────▼──────┐
            │ Prefill pool │ ─── NIXL ────► │ Decode pool  │
            │  (compute)   │                │  (memory)    │
            └──────────────┘                └──────┬───────┘
                                                   │ tokens
                                                   ▼
                                                 Client
```

NIXL 是 NVIDIA 的节点间传输。可用时使用 RDMA/InfiniBand，否则降到 TCP。传输延迟是真实的 — 对于 70B FP8 上 4K-token 提示的 KV 缓存通常 20-80 毫秒。这就是为什么短提示不值得解耦：传输税超过节省。

### Dynamo vs llm-d

**NVIDIA Dynamo**（GTC 2025 宣布，1.0 GA）：
- 作为编排器位于 vLLM、SGLang、TRT-LLM 之上。
- Planner Profiler 测量工作负载，SLA Planner 自动配置预填充:解码比率。
- Rust 核心，Python 可扩展性。
- 吞吐量增益：NVIDIA 在中等延迟区域报告 DeepSeek-R1 MoE 在 GB200 NVL72 + Dynamo 上 6 倍 (developer.nvidia.com, 2025-06)；社区关于完整 Blackwell + Dynamo + DeepSeek-R1 栈上"最高 30 倍"的报告缺乏单一原始来源，应视为方向性。
- GB300 NVL72 + Dynamo：Dynamo 产品页面上最高 50 倍 MoE 吞吐量 vs Hopper (developer.nvidia.com，未注明日期)。

**llm-d**（Red Hat + AWS，Kubernetes 原生）：
- 预填充/解码/路由器作为独立的 Kubernetes Services。
- 各自角色 HPA，使用队列深度（预填充）/ KV 利用率（解码）信号。
- `topologyConstraint packDomain: rack` 将预填充+解码团组打包在同一机架上以实现高带宽 KV 传输。
- llm-d 0.5 (2026)：分层 KV 卸载、缓存感知 LoRA 路由、UCCL 网络、缩容到零。

如果你想要托管栈上层编排器，使用 Dynamo。如果你想要 Kubernetes 原生原语并致力于 CNCF 生态，使用 llm-d。

### 经济学

内部综合（非单一发表的案例研究 — 数量级锚定）：

- $2M/年并置推理支出。
- 切换到带 Dynamo 的解耦。
- 相同请求量，相同 P99 延迟 SLA。
- 报告的节省：$600K–$800K/年（30–40% 降低）。
- 无新硬件。

我们从多个客户披露综合此数字，而非单一可引用的案例研究；最接近的发表数据点是 Baseten 在 Dynamo KV 路由上快 2 倍的 TTFT / 高 61% 的吞吐量 (baseten.co, 2025-10)，以及 VAST + CoreWeave 在 40–60% KV 命中率下每美元 token 提高 60–130% 的预测 (vastdata.com, 2025-12)。节省来自对每个池进行合适大小调整；预填充密集型工作负载（带 8K+ 前缀的 RAG）比平衡型获益更多。

### 何时不解耦

- 提示 < 512 token 且输出 < 200 token：传输税超过增益。
- 小集群（< 4 GPU）：池多样性不足。
- 团队无法运维带各自角色扩展的两个 GPU 池：Dynamo 有帮助但不是轻易的。
- 无 RDMA 结构：TCP 传输税更重。

### 路由器与 Phase 17 · 11 集成

解耦路由器是 KV 缓存感知的（Phase 17 · 11）。请求落在持有其前缀的解码池 — 如果没有匹配，则流向预填充 → 解码。命中率和解耦复合 — 缓存感知路由器决定是否甚至需要新的预填充。

### Blackwell 上的 MoE 才是真正数字所在

GB300 NVL72 + Dynamo 展示相比 Hopper 基线 50 倍 MoE 吞吐量。MoE 专家路由在预填充上是计算密集型的，但在解码上是内存密集型的（专家缓存），因此解耦是双重收益。2026 年前沿模型推理以 MoE 主导（DeepSeek-V3、未来的 GPT-5 变体）。

### 你应该记住的数字

基准数字漂移 — NVIDIA 和推理栈每季度发布更新结果。引用前重新检查。

- DeepSeek-R1 在 GB200 NVL72 + Dynamo：中等延迟区域相比基线的约 6 倍吞吐量 (developer.nvidia.com, 2025-06)；社区关于完整 Blackwell + Dynamo 栈上"最高 30 倍"的说法是方向性综合，缺乏单一原始来源。
- GB300 NVL72 + Dynamo：最高 50 倍 MoE 吞吐量 vs Hopper (developer.nvidia.com，未注明日期)。
- 节省锚定（内部综合，非单一案例研究）：在恒定 SLA 下，每年 $2M 支出的 $600-800K/年。
- 解耦阈值：提示 >512 token + 输出 >200 token。
- 通过 NIXL 的 KV 传输：70B FP8 上 4K 提示的 KV 20-80 毫秒。

## 使用它

`code/main.py` 模拟并置 vs 解耦推理。报告吞吐量、每请求成本以及提示长度交叉点。

## 交付它

本课产出 `outputs/skill-disaggregation-decider.md`。给定工作负载和集群，决定是否解耦。

## 练习

1. 运行 `code/main.py`。在什么提示长度下解耦击败并置？
2. 为 P99 前缀长度 8K、输出 300 的 RAG 服务设计预填充池和解码池。
3. Dynamo vs llm-d：选择一个无 Python 运行时偏好的纯 Kubernetes 环境。
4. 计算 KV 传输成本：70B FP8 上 4K 预填充 = 约 500 MB KV。在 RDMA 100 GB/s 下，传输 = 5 毫秒。在 TCP 10 GB/s = 50 毫秒。哪个对你的 SLA 重要？
5. MoE 专家路由改变了 KV 访问模式。对于每个 token 激活不同专家的 MoE，解耦表现如何？

## 核心术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 解耦推理 | "分离预填充/解码" | 每个阶段独立的 GPU 池 |
| NIXL | "NVIDIA 传输" | Dynamo 的节点间 KV 传输 (RDMA/TCP) |
| NVIDIA Dynamo | "编排器" | vLLM/SGLang/TRT-LLM 的栈上层协调器 |
| llm-d | "Kubernetes 原生" | Red Hat + AWS K8s 解耦栈 |
| Planner Profiler | "Dynamo 自动配置" | 测量工作负载，配置池比率 |
| SLA Planner | "Dynamo 策略" | 自动调整预填充:解码比率以满足 SLO |
| `packDomain: rack` | "llm-d 拓扑" | 打包预填充+解码在同一机架以实现快速 KV |
| UCCL | "统一集体通信" | llm-d 0.5 的缩容到零网络层 |
| MoE 专家路由 | "每 token 专家" | DeepSeek-V3 模式；解耦有帮助 |

## 进一步阅读

- [NVIDIA — 引入 Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/)
- [NVIDIA — Kubernetes 上的解耦 LLM 推理](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/)
- [TensorRT-LLM 解耦推理博客](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog5_Disaggregated_Serving_in_TensorRT-LLM.html)
- [llm-d GitHub](https://github.com/llm-d/llm-d)
- [llm-d 0.5 发布说明](https://github.com/llm-d/llm-d/releases)
