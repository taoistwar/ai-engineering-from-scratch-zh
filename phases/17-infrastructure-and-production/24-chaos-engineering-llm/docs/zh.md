# LLM 生产的混沌工程

> LLM 的混沌工程在 2026 年是其自身的学科。在生产中运行实验之前的先决条件：定义的 SLI/SLO、追踪+指标+日志可观测性、自动化回滚、运行手册、值班。架构有四个平面：控制平面（实验调度器）、目标平面（服务、基础设施、数据存储）、安全平面（护栏 + 中止 + 流量过滤器）、可观测性平面（指标 + 追踪 + 日志）、反馈回路（注入 SLO 调整）。护栏是强制性的：如果每日错误预算消耗 > 预期的 2 倍，燃速警报暂停实验；抑制窗口 + 追踪 ID 关联去重警报噪声。节奏：每周小规模金丝雀 + SLO 审查；每月游戏日 + 事后分析；每季度跨团队弹性审计 + 依赖映射。LLM 特定实验：内存过载、网络失败、供应商中断、格式错误的提示、KV 缓存驱逐风暴。工具：Harness Chaos Engineering（LLM 衍生建议、爆炸半径缩小、MCP 工具集成）；LitmusChaos（CNCF）；Chaos Mesh（CNCF Kubernetes 原生）。

**Type:** Learn
**Languages:** Python (stdlib, toy chaos experiment runner)
**Prerequisites:** Phase 17 · 23 (SRE for AI), Phase 17 · 13 (Observability)
**Time:** ~60 minutes

## 学习目标

- 说出混沌工程的五个先决条件（SLI/SLO、可观测性、回滚、运行手册、值班）并解释跳过任何一个为什么会破坏实践。
- 绘制四个平面（控制、目标、安全、可观测性）和反馈回路进入 SLO。
- 列举五种 LLM 特定实验（内存过载、网络失败、供应商中断、格式错误提示、KV 驱逐风暴）。
- 给定栈选择工具 — Harness、LitmusChaos、Chaos Mesh。

## 问题

传统栈中的混沌测试已经成熟。LLM 栈添加了新的失败模式。一个带有特制字符的 4K-token 提示使分词器卡住 12 秒。上游供应商 429；你的网关重试；你的服务在重试放大并发下 OOM。在突发负载下 KV 缓存驱逐风暴导致重预填充级联，使计算饱和。

这些在单元测试中都不会出现。混沌工程是你在用户发现之前发现它们的方式。

## 概念

### 先决条件

没有以下条件不在生产中运行混沌：

1. **SLI/SLO** — 定义的服务级别指标和目标。
2. **可观测性** — 追踪、指标、日志，连接到仪表板。
3. **自动化回滚** — Phase 17 · 20 策略标志回滚。
4. **运行手册** — 结构化，Phase 17 · 23。
5. **值班** — 有人响应。

缺少任何一个意味着混沌变成真实事件。

### 四个平面 + 反馈

**控制平面** — 实验调度器（Litmus 工作流、Chaos Mesh 调度、Harness UI）。

**目标平面** — 服务、pod、节点、负载均衡器、数据存储。

**安全平面** — 终止开关、抑制窗口、爆炸半径限制、错误预算门控。

**可观测性平面** — 正常指标 + 追踪 ID 关联以区分混沌诱导和自然失败。

**反馈回路** — 发现反馈到 SLO 调整、运行手册更新、代码修复。

### 护栏是强制性的

- **燃速警报**：如果每日错误预算消耗超过预期的 2 倍，暂停实验。
- **抑制窗口**：在实验期间静音爆炸半径内的非实验警报。
- **追踪 ID 关联**：所有实验诱导的错误携带标签，使值班人员可以去重。

### 五种 LLM 特定实验

1. **内存过载** — 通过以高并发发送长上下文请求强制 KV 缓存抢占风暴。观察：服务是优雅降级还是崩溃？

2. **网络失败** — 切断推理网关与供应商之间的连接。观察：降级是否在 SLA 内启动？（Phase 17 · 19）

3. **供应商中断模拟** — OpenAI 100% 429。观察：路由是否故障转移到 Anthropic？（Phase 17 · 16, 19）

4. **格式错误提示** — 注入使分词器停滞的有效载荷（例如，深度嵌套 unicode、超大 UTF-8 码点）。观察：单个请求是否锁住一个工作线程？

5. **KV 驱逐风暴** — 通过使 vLLM 块预算饱和强制驱逐。观察：LMCache 恢复还是服务降级？

### 节奏

- **每周** — 在预发中进行小规模金丝雀实验，可能 5% 生产。
- **每月** — 在特定场景上安排游戏日；跨团队参加；事后分析。
- **每季度** — 跨团队弹性审计；依赖图更新。

### 工具

- **Harness Chaos Engineering** — 商业；AI 衍生实验建议；爆炸半径缩小；MCP 工具集成。
- **LitmusChaos** — CNCF 毕业；基于 Kubernetes 工作流。
- **Chaos Mesh** — CNCF 沙箱；Kubernetes 原生 CRD 风格。
- **Gremlin** — 商业；广泛支持。
- **AWS FIS** / **Azure Chaos Studio** — 托管云产品。

### 从小开始

第一个实验：在稳定流量下杀死一个解码副本。观察重新路由和恢复。如果这个工作并且看起来安全，升级到网络混沌。

第一个 LLM 特定实验：注入一个供应商 429 持续 5 分钟。观察降级。大多数团队发现他们的降级没有完全测试过。

### 你应该记住的数字

- 四个平面：控制、目标、安全、可观测性。
- 燃速暂停：2 倍预期每日预算消耗。
- 节奏：每周金丝雀、每月游戏日、每季度审计。
- 五种 LLM 实验：内存、网络、供应商、格式错误提示、KV 风暴。

## 使用它

`code/main.py` 模拟带安全平面门控的三个混沌实验。报告哪些实验会触发燃速中止。

## 交付它

本课产出 `outputs/skill-chaos-plan.md`。给定栈和成熟度，选择前三个实验和工具。

## 练习

1. 运行 `code/main.py`。哪个实验触发燃速门控，为什么？
2. 为基于 vLLM 的 RAG 服务设计前五个混沌实验。包括成功标准。
3. 你的燃速警报暂停了一个实验。你如何确定根因 — 混沌还是自然？
4. 论证混沌应在生产中运行还是仅在预发中。什么时候生产是正确答案？
5. 说出三种通用网络混沌无法复现的 LLM 特定失败模式。

## 核心术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| SLI / SLO | "服务目标" | 指标 + 目标；必需的先决条件 |
| Blast radius | "范围" | 受实验影响的服务/用户集合 |
| Burn-rate alert | "预算门控" | 当错误预算消耗率 > 2 倍预期时触发 |
| Game day | "每月演练" | 安排的跨团队混沌演练 |
| LitmusChaos | "CNCF 工作流" | 毕业的 CNCF Kubernetes 混沌工具 |
| Chaos Mesh | "CNCF CRD" | CNCF 沙箱 Kubernetes 原生混沌 |
| Harness CE | "商业 AI 辅助" | 带 AI 建议的 Harness 混沌 |
| Malformed prompt | "分词器炸弹" | 使分词停滞的输入 |
| KV eviction storm | "抢占级联" | 大规模驱逐触发重预填充 |

## 进一步阅读

- [DevSecOps School — 混沌工程 2026 指南](https://devsecopsschool.com/blog/chaos-engineering/)
- [Ankush Sharma — LLM 可观测性（书籍）](https://www.amazon.com/Observability-Large-Language-Models-Engineering-ebook/dp/B0DJSR65TR)
- [LitmusChaos (CNCF)](https://litmuschaos.io/)
- [Chaos Mesh (CNCF)](https://chaos-mesh.org/)
- [Harness Chaos Engineering](https://www.harness.io/products/chaos-engineering)
- [AWS FIS](https://aws.amazon.com/fis/)
