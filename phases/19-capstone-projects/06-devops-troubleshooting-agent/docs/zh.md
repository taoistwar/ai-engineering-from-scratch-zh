# 实践项目 06 — Kubernetes DevOps 故障排除智能体

> AWS 的 DevOps Agent 正式发布，Resolve AI 发布了其 K8s 剧本，NeuBird 演示了语义监控，Metoro 将 AI SRE 与每服务 SLO 绑定。生产形态已经确定：告警 webhook 触发，智能体读取遥测数据，遍历 K8s 对象图，对根因假设进行排序，并发布带有审批按钮的 Slack 简报。默认只读。每个修复操作都需要人工审批。这个实践项目就是那个智能体，在 20 个合成事件上评估并与 AWS 的 Agent 在三个共享案例上进行比较。

**类型:** 实践项目
**语言:** Python（智能体），TypeScript（Slack 集成）
**前置条件:** 阶段 11（LLM 工程），阶段 13（工具与 MCP），阶段 14（智能体），阶段 15（自主），阶段 17（基础设施），阶段 18（安全）
**涉及的阶段:** P11 · P13 · P14 · P15 · P17 · P18
**时间:** 30 小时

## 问题

2025-2026 年的 SRE 叙事变成了："AI 智能体分诊事件，人工审批修复。" AWS DevOps Agent、Resolve AI、NeuBird、Metoro、PagerDuty AIOps 都在生产中推出了这种形态。智能体读取 Prometheus 指标、Loki 日志、Tempo 追踪、kube-state-metrics 和 K8s 对象的知识图谱。它在五分钟内生成带有遥测引文的排名根因假设。它从不未经 Slack 中明确人工审批就执行破坏性命令。

大多数困难在工作范围界定和安全性上，而非推理。智能体需要一个默认只读的 RBAC 表面、一个加固的 MCP 工具服务器，以及关于每个考虑过 vs 执行过的命令的审计日志。它需要知道何时超出其深度并升级上报。而且它需要运行得足够便宜，使得 OOM-kill 级联不会产生 5000 美元的智能体账单。

## 概念

智能体在知识图谱上运行。节点包括 K8s 对象（Pod、Deployment、Service、Node、HPA、PVC）加上遥测源（Prometheus 序列、Loki 流、Tempo 追踪）。边编码所有权（Pod -> ReplicaSet -> Deployment）、调度（Pod -> Node）和观察（Pod -> Prometheus 序列）。图谱通过 kube-state-metrics 同步保持新鲜，并在每次告警时重新采样。

当告警触发时，智能体从受影响的对象出发进行根因分析。它遍历边，拉取相关的遥测切片（最近 15 分钟），并草拟一份假设。假设按证据排序：有多少遥测引文支持它、它们有多近、它们有多具体。前 3 个假设发送到 Slack，附带图谱路径可视化和修复操作的审批按钮。

修复是有门控的。允许的默认操作是只读的。破坏性操作（缩容、回滚、删除 Pod）需要 Slack 审批；ArgoCD 回滚钩子需要一个智能体从不持有的认证 token。审计日志记录了智能体*考虑过*的每个命令——不仅仅是执行过的——以便审查过程捕获未遂。

## 架构

```
PagerDuty / Alertmanager webhook
           |
           v
     FastAPI receiver
           |
           v
   LangGraph root-cause agent
           |
           +---- read-only MCP tools ----+
           |                             |
           v                             v
   K8s knowledge graph              telemetry slices
     (Neo4j / kuzu)              Prometheus, Loki, Tempo
   ownership + scheduling          last 15m, scoped
           |
           v
   hypothesis ranking (evidence weight)
           |
           v
   Slack brief + approval buttons
           |
           v (approved)
   ArgoCD rollback hook / PagerDuty escalate
           |
           v
   audit log: considered vs executed, every command
```

## 技术栈

- 可观测性源: Prometheus、Loki、Tempo、kube-state-metrics
- 知识图谱: Neo4j（托管）或 kuzu（嵌入式），包含 K8s 对象 + 遥测边
- 智能体: LangGraph 带每个工具的允许列表，默认只读
- 工具传输: FastMCP over StreamableHTTP；单独的服务器用于审批门背后的破坏性工具
- 模型: Claude Sonnet 4.7 用于根因推理，Gemini 2.5 Flash 用于日志摘要
- 修复: ArgoCD 回滚 webhook，PagerDuty 升级，Slack 审批卡片
- 审计: 追加式结构化日志（considered、executed、approved、outcome）
- 部署: K8s 部署，具有自己的狭小 RBAC 角色；独立的命名空间

## 构建它

1. **图谱摄入。** 每 30 秒将 kube-state-metrics 同步到 Neo4j/kuzu 中。节点：Pod、Deployment、Node、Service、PVC、HPA。边：OWNED_BY、SCHEDULED_ON、EXPOSES、MOUNTS、SCALES。遥测覆盖边：OBSERVED_BY（Pod 被 Prometheus 序列观察）。

2. **告警接收器。** FastAPI 端点，接受 PagerDuty 或 Alertmanager webhooks。提取受影响的对象和 SLO 违反。

3. **只读工具表面。** 通过 FastMCP 包装 kubectl、Prometheus 查询、Loki logql、Tempo traceql。每个工具都有狭窄的 RBAC 动词（"get"、"list"、"describe"）。默认服务器中没有"delete"、"exec"、"scale"。

4. **根因智能体。** LangGraph 包含三个节点：`sample` 拉取最近 15 分钟的遥测切片，`walk` 查询图中邻近的对象，`hypothesize` 草拟带有遥测引文的排名根因候选。

5. **证据评分。** 每个假设都有一个分数 = 新近度 * 特异性 * 图路径长度倒数 * 引文计数。返回前 3 个。

6. **Slack 简报。** 发布一个附件，包含假设、图路径可视化（服务端渲染的子图图像）以及最多一个修复操作的审批按钮。

7. **修复门控。** 破坏性工具（缩容、回滚、删除）位于第二个需要审批 token 的 MCP 服务器上。只有在 Slack 卡片被人工审批后，智能体才能调用它们。

8. **审计日志。** 追加式 JSONL：对于每个候选命令，记录它是否被考虑过、是否被执行过、谁审批了它。每天发送到 S3。

9. **合成事件套件。** 构建 20 个场景：OOMKill 级联、DNS 抖动、HPA 震荡、PVC 填满、噪声邻居、故障 sidecar、错误的 ConfigMap 部署、证书轮换、镜像拉取退避等。根据根因准确度和假设时间对智能体评分。

## 使用它

```
webhook: alert.pagerduty.com -> checkout-api SLO breach, error rate 14%
[graph]   affected: Deployment checkout-api (3 Pods, Node ip-10-2-3-4)
[walk]    neighbors: ReplicaSet checkout-api-abc, Service checkout-api,
           recent rollout 14m ago
[sample]  prometheus error_rate 14%, up-trend; loki 500s on /api/v2/pay
[hypo]    #1 bad rollout: latest image checkout-api:v2.41 fails /healthz
          citations: deploy.yaml (rev 42), prometheus errorRate, loki 500 stack
[slack]   [ROLL BACK to v2.40]  [ESCALATE]  [IGNORE]
          (approval required; agent does not roll back unilaterally)
```

## 交付它

`outputs/skill-devops-agent.md` 是可交付成果。给定一个 K8s 集群和告警源，智能体生成排名根因假设和一个 Slack 门控的修复流程。

| 权重 | 标准 | 如何衡量 |
|:-:|---|---|
| 25 | 场景套件上的 RCA 准确度 | 在 20 个合成事件上 ≥80% 正确的根因 |
| 20 | 安全性 | 审计日志中破坏性操作守卫从未在无 Slack 审批的情况下触发 |
| 20 | 假设时间 | 从告警到 Slack 简报的 p50 低于 5 分钟 |
| 20 | 可解释性 | 每个假设都有图路径和遥测引文 |
| 15 | 集成完整性 | PagerDuty、Slack、ArgoCD、Prometheus 端到端工作 |
| **100** | | |

## 练习

1. 在与 AWS DevOps Agent 演示的三个相同事件上运行你的智能体。发布并列对比。报告智能体在何处出现分歧。

2. 添加一个"未遂"审计，标记智能体*考虑过*的、如果在无审批情况下执行将是破坏性的任何命令。测量一周内的未遂率。

3. 将假设模型从 Claude Sonnet 4.7 替换为自托管的 Llama 3.3 70B。测量 RCA 准确度差值和每个事件的美元成本。

4. 构建一个因果过滤器：区分相关的遥测峰值和真正的根因。在 20 个场景标签上训练一个小型分类器。

5. 添加回滚演练：ArgoCD 回滚针对具有相同清单的暂存集群。在 Slack 审批按钮之前，在实时集群中验证回滚计划。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| K8s knowledge graph | "集群图谱" | 节点 = K8s 对象 + 遥测序列；边 = 所有权、调度、观察 |
| Read-only-by-default | "有范围的 RBAC" | 智能体的服务账户只有 get/list/describe 动词；破坏性动词存在于审批背后的独立服务器中 |
| Audit log | "考虑过 vs 执行过" | 每个候选命令的追加记录，是否运行了，谁审批了 |
| Hypothesis ranking | "证据分数" | 新近度 × 特异性 × 图路径长度倒数 × 引文计数 |
| Slack approval card | "HITL 门" | 带有修复按钮的交互式 Slack 消息；智能体在人工点击前不能继续 |
| Telemetry citation | "证据指针" | 支持声明的 Prometheus 查询、Loki 选择器或 Tempo 追踪 URL |
| MTTR | "解析时间" | 从告警触发到 SLO 恢复的墙钟时间 |

## 扩展阅读

- [AWS DevOps Agent GA](https://aws.amazon.com/blogs/aws/aws-devops-agent-helps-you-accelerate-incident-response-and-improve-system-reliability-preview/) — 2026 年的规范参考
- [Resolve AI K8s 故障排除](https://resolve.ai/blog/kubernetes-troubleshooting-in-resolve-ai) — 竞争对手参考
- [NeuBird 语义监控](https://www.neubird.ai) — 语义图谱方法
- [Metoro AI SRE](https://metoro.io) — SLO 优先的生产框架
- [kube-state-metrics](https://github.com/kubernetes/kube-state-metrics) — 集群状态源
- [LangGraph](https://langchain-ai.github.io/langgraph/) — 参考智能体编排器
- [FastMCP](https://github.com/jlowin/fastmcp) — Python MCP 服务器框架
- [ArgoCD 回滚](https://argo-cd.readthedocs.io/en/stable/user-guide/commands/argocd_app_rollback/) — 门控修复目标
