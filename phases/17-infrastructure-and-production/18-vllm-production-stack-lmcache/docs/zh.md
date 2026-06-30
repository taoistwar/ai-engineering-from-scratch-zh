# vLLM 生产栈与 LMCache KV 卸载

> vLLM 的生产栈是参考 Kubernetes 部署 — 路由器、引擎和可观测性连接在一起。LMCache 是 KV 卸载层，从 GPU 内存中提取 KV 缓存并在查询和引擎之间重用它（CPU DRAM、然后是磁盘/Ceph）。vLLM 0.11.0 的 KV 卸载连接器（2026 年 1 月）通过连接器 API（v0.9.0+）使其异步和可插拔。卸载延迟不是用户感知的。LMCache 即使在没有共享前缀时也有价值 — 当 GPU 耗尽 KV 槽时，被抢占的请求可以从 CPU 恢复，而非重计算预填充。在跨 4 个 a3-highgpu-4g 的 16x H100（80GB HBM）上发布的基准测试：当 KV 缓存超出 HBM 时，原生 CPU 卸载和 LMCache 均显著提高吞吐量；在低 KV 占用下，所有配置与基线匹配，具有较小的开销。

**Type:** Learn
**Languages:** Python (stdlib, toy KV-spill simulator)
**Prerequisites:** Phase 17 · 04 (vLLM Serving Internals), Phase 17 · 06 (SGLang/RadixAttention)
**Time:** ~60 minutes

## 学习目标

- 绘制 vLLM 生产栈层次：路由器、引擎、KV 卸载、可观测性。
- 解释 KV 卸载连接器 API（v0.9.0+）以及 0.11.0 的异步路径如何隐藏卸载延迟。
- 量化何时 LMCache CPU-DRAM 有帮助（KV > HBM）vs 增加开销（KV 足够小可放入 HBM）。
- 给定部署约束，在原生 vLLM CPU 卸载和 LMCache 连接器之间选择。

## 问题

你的 vLLM 推理显示 GPU 在 100% HBM 下，每当并发爬升时就有抢占事件。请求被驱逐、重新排队，你在一分钟内重新预填充相同的 2K-token 提示四次。GPU 计算花在冗余预填充上；有效吞吐量远低于原始吞吐量。

添加更多 GPU 是线性增加成本。添加更多 HBM 不可能。但 CPU DRAM 便宜 — 一个插槽有 512 GB+，延迟数量级比 HBM 差但适合"暂时热"的 KV 缓存。

LMCache 将 KV 缓存提取到 CPU DRAM，使被抢占的请求快速恢复，并使引擎间重复前缀共享缓存而无需每个引擎重新预填充。

## 概念

### vLLM 生产栈

`github.com/vllm-project/production-stack` 是参考 Kubernetes 部署：

- **路由器** — 缓存感知（Phase 17 · 11）。消费 KV 事件。
- **引擎** — vLLM 工作节点。每个 GPU 或每个 TP/PP 组一个。
- **KV 缓存卸载** — LMCache 部署或原生连接器。
- **可观测性** — Prometheus 抓取、Grafana 仪表板、OTel 追踪。
- **控制平面** — 服务发现、配置、滚动更新。

以 Helm chart + operator 形式发布。

### KV 卸载连接器 API（v0.9.0+）

vLLM 0.9.0 引入了用于可插拔 KV 缓存后端的连接器 API。你的引擎将块卸载到连接器；连接器存储它们（RAM、磁盘、对象存储、LMCache）。请求需要一个块，连接器将其加载回来。

vLLM 0.11.0（2026 年 1 月）添加了异步卸载路径 — 卸载可以在后台发生，因此引擎在常见情况下不会因此阻塞。端到端延迟和吞吐量仍取决于工作负载形状、KV 缓存命中率和系统压力；vLLM 自己的备注指出自定义内核卸载在低命中率下可能降低吞吐量，并且异步调度存在与推测解码的已知交互问题。

### 原生 CPU 卸载 vs LMCache

**原生 vLLM CPU 卸载**：引擎本地。将 KV 块存储在主机 RAM 中。实现快速，零网络跳转。不跨引擎。

**LMCache 连接器**：集群规模。将块存储在共享 LMCache 服务器（CPU DRAM + Ceph/S3 层）。块可由任何引擎访问。发布了 16x H100 基准测试。

当单个引擎有 HBM 压力时选择原生。当多个引擎共享前缀（带通用系统提示的 RAG、带共享模板的多租户）时选择 LMCache。

### 基准测试行为

跨 4 个 a3-highgpu-4g 的 16x H100（80 GB HBM）测试：

- 低 KV 占用（短提示、低并发）：所有配置与基线匹配，LMCache 增加约 3-5% 开销。
- 中等占用：LMCache 开始在引擎间前缀重用上有帮助。
- KV 超出 HBM：原生 CPU 卸载和 LMCache 均显著提高吞吐量；LMCache 由于跨引擎共享获得更大增益。

### 何时 LMCache 是决定性的

- 多租户推理，系统提示在租户间共享。
- RAG，文档块在查询间重复。
- 相同基础上微调变体（LoRA），基础模型 KV 重用削减冗余工作。
- 抢占密集型工作负载：从 CPU 恢复比重新预填充便宜。

### 何时不启用

- 小的 HBM 压力 — 你支付开销而无收益。
- 短上下文（<1K token）— 传输时间 > 重新预填充。
- 单租户单提示工作负载 — 无可捕获的重用。

### 与解耦推理的集成

Phase 17 · 17 解耦推理 + LMCache 复合：从预填充池到解码池的 KV 传输若未使用则落入 LMCache；后续查询从 LMCache 拉取。Phase 17 · 11 的缓存感知路由器可以路由到本地或 LMCache 共享缓存匹配的引擎。

### 你应该记住的数字

- vLLM 0.9.0：连接器 API 发布。
- vLLM 0.11.0（2026 年 1 月）：异步卸载路径；端到端延迟影响取决于工作负载、KV 命中率和系统压力（非绝对保证）。
- 16x H100 基准测试：LMCache 在 KV 占用超出 HBM 时有帮助。
- 小 HBM 压力：无收益的 3-5% 开销。

```figure
zero-sharding
```

## 使用它

`code/main.py` 模拟带和不带 LMCache 的抢占密集型工作负载。报告避免的重新预填充次数、吞吐量增益和盈亏平衡 HBM 利用率。

## 交付它

本课产出 `outputs/skill-vllm-stack-decider.md`。给定工作负载形状和 vLLM 部署，决定原生 vs LMCache vs 都不用。

## 练习

1. 运行 `code/main.py`。在什么 HBM 利用率下 LMCache 开始有回报？
2. 一个租户在 200 查询/小时中共享 6K-token 系统提示。计算每租户的预期 LMCache 节省。
3. LMCache 服务器是单点故障。设计 HA 策略（副本、降级到原生）。
4. LMCache 存储到旋转磁盘上的 Ceph。对于 70B FP8 上的 4K-token KV（500 MB），读取时间 vs 重新预填充是多少？
5. 论证 vLLM 0.11.0 异步路径是否"免费" — 开销隐藏在哪里？

## 核心术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Production-stack | "参考部署" | vLLM 的 Kubernetes Helm chart + operator |
| Connector API | "KV 后端接口" | vLLM 0.9.0+ 可插拔 KV 存储接口 |
| 原生 CPU 卸载 | "引擎本地溢出" | 在同一引擎的主机 RAM 中存储 KV |
| LMCache | "集群 KV 缓存" | CPU DRAM + 磁盘上的跨引擎 KV 缓存服务器 |
| 0.11.0 异步 | "非阻塞卸载" | 卸载隐藏在引擎流之后 |
| 抢占 | "驱逐以腾出空间" | HBM 满时的 KV 缓存洗牌 |
| 前缀重用 | "相同系统提示" | 多个查询共享开头；缓存命中 |
| Ceph 层 | "磁盘层" | 缓存层级中 DRAM 以下的持久化存储 |

## 进一步阅读

- [vLLM 博客 — KV 卸载连接器（2026 年 1 月）](https://blog.vllm.ai/2026/01/08/kv-offloading-connector.html)
- [vLLM 生产栈 GitHub](https://github.com/vllm-project/production-stack) — Helm chart + operator。
- [企业规模 LLM 推理的 LMCache (arXiv:2510.09665)](https://arxiv.org/html/2510.09665v2)
- [LMCache GitHub](https://github.com/LMCache/LMCache) — 连接器实现。
- [vLLM 0.11.0 发布说明](https://github.com/vllm-project/vllm/releases) — 异步路径详情。
