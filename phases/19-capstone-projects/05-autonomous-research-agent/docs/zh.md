# 实践项目 05 — 自主研究智能体（AI-Scientist 级别）

> Sakana 的 AI-Scientist-v2 发表了完整论文。Agent Laboratory 运行了实验。Allen AI 分享了追踪记录。2026 年的形态是对实验进行"计划-执行-验证"的树搜索，带有成本预算、沙箱化的代码执行、带有视觉反馈的 LaTeX 编写器，以及自动化的 NeurIPS 风格评审委员会。实践项目是构建一个，在每篇论文 30 美元以内端到端运行，并在 Sakana 记录的沙箱逃逸红队测试中存活。

**类型:** 实践项目
**语言:** Python（智能体 + 沙箱），LaTeX（输出）
**前置条件:** 阶段 2（ML），阶段 3（深度学习），阶段 7（transformers），阶段 10（从头构建 LLM），阶段 14（智能体），阶段 15（自主），阶段 16（多智能体），阶段 18（安全）
**涉及的阶段:** P0 · P2 · P3 · P7 · P10 · P14 · P15 · P16 · P18
**时间:** 40 小时

## 问题

自主研究智能体在 2026 年跨越了一个临界点。Sakana AI 的 AI-Scientist-v2 在 Nature 上发表，其生成的论文通过了研讨会同行评审。ShinkaEvolve（ICLR 2026）将这一思路扩展到进化假设。AMD 的 Agent Laboratory 推出了可复现的追踪记录。智能体并非魔法——它们是运行在候选实验树上的"计划-执行-验证"循环，具有成本上限、基于种子的沙箱和自动评审。工艺在循环、预算和安全故事中。

你通过在一个狭窄领域（例如，在 100M 参数的 transformer 上的注意力稀疏性消融实验）上针对一个种子想法实现一个来学习这个循环。价值不在于首次运行就发现新东西。价值在于基础设施：树搜索、实验沙箱、编写-评审循环、红队报告。Sakana 团队记录了沙箱逃逸失败；你的智能体必须通过相同的红队测试。

## 概念

智能体是最佳优先树搜索。节点是实验规格：(假设, 配置, 代码, 预期结果)。扩展步骤提出带有小的修改的子节点（交换优化器、改变批次大小、消融一个组件）。每个子节点在具有硬性资源上限的全新沙箱中运行。结果反馈到一个评分函数中，该函数按 (新颖性 × 质量 × 剩余预算) 对节点进行排序。树在预算耗尽之前不断增长，然后最佳分支被写出。

编写器是多模态的。它生成 LaTeX 草稿，编译它，渲染图表，并将渲染后的 PDF 反馈给 Claude Opus 4.7 的视觉模式进行关于布局、图表可读性和声明-证据对齐的评判。一个由五个 LLM 评委组成的评审委员会给出 NeurIPS 风格的分数（新颖性、严谨性、清晰度、可复现性、影响力）；如果平均分降到了阈值以下，论文带着评判回到编写器。

安全是承重的。每个实验在 E2B 或 Daytona 沙箱中运行，没有网络出口，有界墙钟时间，并固定资源限制。智能体的代码生成步骤经过一个策略层，该层阻止能够逃逸沙箱的系统调用。红队报告复现 Sakana 记录的攻击面（fork 炸弹、文件系统逃逸、LLM 编写的网络调用）。

## 架构

```
seed idea + domain
      |
      v
  literature search (Semantic Scholar + OpenAlex + FAISS cache)
      |
      v
  LangGraph plan-execute-verify tree
      |
      v
  +--- expand node ----+      per-node sandbox
  |                    |      (E2B / Daytona)
  v                    v      resource caps
  child_1           child_k   no network egress
  |                    |      deterministic seeds
  v                    v
  run experiment       run experiment
  |                    |
  v                    v
  score nodes by (novelty, quality, budget)
      |
      v
  best branch -> LaTeX writer
      |
      v
  compile + vision critique (Opus 4.7 vision)
      |
      v
  reviewer ensemble (5 LLM judges, NeurIPS rubric)
      |
      v
  paper.pdf + review.md + trace.json
```

## 技术栈

- 编排: LangGraph 带检查点和人工批准门
- 树搜索: 自定义最佳优先遍历实验节点（来自 Sakana v2 的 AB-MCTS 风格）
- 沙箱: E2B 每个实验，Docker-in-Docker 备用；通过 cgroups 实现资源上限
- 文献: Semantic Scholar Graph API + OpenAlex + 本地的摘要 FAISS 缓存
- 编写器: LaTeX 模板 + Claude Opus 4.7（视觉模式）用于图表评判和布局
- 评审委员会: 5 个评委的集合（Opus 4.7、GPT-5.4、Gemini 3 Pro、DeepSeek R1、Qwen3-Max），加权聚合
- 实验框架: PyTorch 2.5 用于物理实验，W&B 用于日志记录
- 可观测性: Langfuse 用于智能体追踪，每篇论文 30 美元硬预算

## 构建它

1. **种子和领域范围界定。** 采用一个种子想法（例如，"研究 sub-1B transformer 注意力图中稀疏性模式"）。定义搜索空间：模型、数据集、计算预算。

2. **文献通行。** 查询 Semantic Scholar + OpenAlex 获取 50 篇最常被引用的相关论文；在本地缓存摘要；生成 1 页的领域摘要。

3. **树脚手架。** 用种子假设初始化根节点。实现 `expand(node) -> children`，使用小的编辑建议（每个子节点一个配置更改）。将 `score(node)` 实现为加权的新颖性 × 质量 × 预算项。

4. **沙箱包装。** 每个实验运行 `docker run --network=none --memory=8g --cpus=2 --pids-limit=256 --read-only`（或等效的 E2B 策略）。种子写入沙箱；输出以只读方式挂载回去。

5. **计划-执行-验证循环。** `plan` 提出子节点。`execute` 运行沙箱，捕获日志和指标。`verify` 对指标运行单元检查（损失下降了吗？消融隔离了效果吗？）。失败的节点在树上存储失败原因。

6. **编写器。** 预算用完后，选择最佳分支。用 matplotlib 渲染图表。通过 Claude Opus 4.7 生成 LaTeX 草稿，将分支追踪放入上下文中。编译。将编译后的 PDF 反馈给 Opus 4.7 视觉模式进行评判。迭代。

7. **评审委员会。** 五个评委按 NeurIPS 风格标准对草稿评分（新颖性、严谨性、清晰度、可复现性、影响力）。如果均值 < 4.0/5，返回编写器并附带评判。3 次重写后硬性停止。

8. **红队。** 构建或集成一组针对沙箱的对抗性任务：fork 炸弹、网络外传尝试、文件系统逃逸、LLM 编写的 shell 元字符。确认所有都被阻止。写下发现。

9. **可复现性。** 每篇论文都附带其树搜索追踪 JSON、种子、W&B 运行链接、沙箱配置和一个端到端复现它的 README。

## 使用它

```
$ ai-scientist run --seed "attention sparsity in sub-1B transformers" --budget 30
[lit]    50 papers, digest in 12s
[tree]   expanded 8 nodes, budget 12/30
[exec]   node #3 sparsity=top-8, loss=2.83 (best so far)
[exec]   node #6 sparsity=top-4, loss=3.12 (worse)
[exec]   ...
[tree]   chose branch rooted at node #3 (novelty 0.62, quality 0.81)
[write]  LaTeX draft v1 complete
[vision] critique: figure 2 legend too small, claim-evidence ok
[write]  draft v2 after 3 edits
[review] mean 4.2/5 (novelty 3.9, rigor 4.3, clarity 4.1, repro 4.5, impact 4.2)
[done]   paper.pdf + review.md + trace.json     $28.40 spent
```

## 交付它

`outputs/skill-ai-scientist.md` 是可交付成果。给定一个种子想法 + 一个领域 + 30 美元预算，它运行完整管道并产出一篇可评审的论文和一个可复现包。

| 权重 | 标准 | 如何衡量 |
|:-:|---|---|
| 25 | 论文质量 | 对照已发表的研讨会论文进行盲评标准审查 |
| 20 | 实验严谨性 | 基线、种子、消融；每条声明都有结果表中的单元格支持 |
| 20 | 成本和计算纪律 | 每篇论文 30 美元上限得到执行，Langfuse 追踪 |
| 20 | 安全性 | 沙箱红队通过；网络策略和杀开关已验证 |
| 15 | 可复现性 | 使用相同种子的一键重跑能够复现论文 |
| **100** | | |

## 练习

1. 在同一领域内针对三个不同的种子想法运行管道。比较树搜索的哪些部分重叠。识别重复的浪费计算。

2. 在估计成本超过 5 美元的节点的实验执行之前添加一个人机交互门。衡量总成本下降了多少。

3. 将评审委员会替换为单个评委。在一组已知是坏论文的留存集上测量误接受率。

4. 引入网络外传红队测试：智能体编写试图 `curl` 外部地址的代码。确认 `--network=none` 策略阻止了它。记录这一尝试。

5. 将你的树搜索与平坦随机基线（相同预算，无扩展策略）进行比较。报告新颖性 × 质量的增益。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| Tree search | "AB-MCTS 风格扩展" | 在具有新颖性×质量×预算分数的实验节点上进行最佳优先探索 |
| Sandbox | "实验隔离" | 没有网络、有界 CPU/内存、固定种子、只读输入的容器 |
| Vision critique | "渲染后阅读" | 将论文编译为 PDF，将 PDF 反馈给 VLM 进行布局和声明-证据的评判 |
| Reviewer ensemble | "自动同行评审" | 多个 LLM 评委以 NeurIPS 标准评审论文；加权聚合作为管道门控 |
| Novelty score | "这是新东西吗？" | 惩罚与 50 篇论文文献缓存的接近度的启发式方法 |
| Cost ceiling | "美元预算" | 每篇论文总花费的硬性上限；Langfuse 计数器 + 运行前估算 |
| Red team | "沙箱逃逸审计" | 如果策略错误就会逃逸沙箱的对抗性任务 |

## 扩展阅读

- [Sakana AI-Scientist-v2 仓库](https://github.com/SakanaAI/AI-Scientist-v2) — 参考生产级研究智能体
- [Sakana AI-Scientist-v1 论文 (arXiv:2408.06292)](https://arxiv.org/abs/2408.06292) — 原始方法
- [ShinkaEvolve (Sakana ICLR 2026)](https://sakana.ai) — 进化扩展
- [Agent Laboratory (AMD)](https://github.com/SamuelSchmidgall/AgentLaboratory) — 多角色研究实验室框架
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/) — 参考编排层
- [Semantic Scholar Graph API](https://api.semanticscholar.org/) — 文献搜索
- [E2B 沙箱](https://e2b.dev) — 参考实验隔离
- [NeurIPS 评审指南](https://neurips.cc/Conferences/2026/Reviewer-Guidelines) — 评审委员会编码的评分标准
