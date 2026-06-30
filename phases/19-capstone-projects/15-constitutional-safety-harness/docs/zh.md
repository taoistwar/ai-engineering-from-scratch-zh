# 实践项目 15 — 宪章安全控制框架 + 红队靶场

> Anthropic 的宪章分类器、Meta 的 Llama Guard 4、Google 的 ShieldGemma-2、NVIDIA 的 Nemotron 3 Content Safety 以及覆盖多语言的 X-Guard 定义了 2026 年的安全分类器栈。garak、PyRIT、NVIDIA Aegis 和 promptfoo 成为标准的对抗性评估工具。NeMo Guardrails v0.12 将它们绑入生产管道。这个实践项目将所有这些连接在一起：围绕目标应用的层级安全控制框架、运行 6 个以上攻击家族的自主红队智能体，以及产生可测量无害性差值的宪章自我评判运行。

**类型:** 实践项目
**语言:** Python（安全管道、红队），YAML（策略配置）
**前置条件:** 阶段 10（从头构建 LLM），阶段 11（LLM 工程），阶段 13（工具），阶段 14（智能体），阶段 18（伦理、安全、对齐）
**涉及的阶段:** P10 · P11 · P13 · P14 · P18
**时间:** 25 小时

## 问题

2026 年 LLM 安全的前沿不在于分类器是否有效（它们大致有效），而在于如何围绕生产应用正确组合它们而不至于过度拒绝或留下明显的漏洞。Llama Guard 4 处理英语策略违规。X-Guard（132 种语言）处理多语言越狱。ShieldGemma-2 捕获基于图像的提示注入。NVIDIA Nemotron 3 Content Safety 涵盖企业类别。Anthropic 的宪章分类器是一种在训练期间而非服务期间使用的单独方法。

攻击演化也很重要。PAIR 和 TAP 自动化越狱发现。GCG 运行基于梯度的后缀攻击。多轮和代码切换攻击利用智能体记忆。任何部署的 LLM 都需要一个红队靶场——garak 和 PyRIT 是规范的驱动引擎——加上文档化的缓解措施和 CVSS 评分的发现。

你将加固一个目标应用（一个 8B 指令调优模型或来自其他实践项目的 RAG 聊天机器人之一），针对它运行 6 个以上攻击家族，并生成前后无害性测量结果。

## 概念

安全管道有五层。**输入净化**：剥离零宽字符、解码 base64/rot13、归一化 Unicode。**策略层**：NeMo Guardrails v0.12 轨道（领域外、毒性、PII 提取）。**分类器门**：输入上的 Llama Guard 4，非英语上的 X-Guard，图像输入上的 ShieldGemma-2。**模型**：目标 LLM。**输出过滤器**：输出上的 Llama Guard 4，Presidio PII 脱敏，在适用的情况下执行引文。**HITL 层**：标记为高风险的输出进入 Slack 队列。

红队靶场按调度器运行。PAIR 和 TAP 自主发现越狱。GCG 运行基于梯度的后缀攻击。ASCII / base64 / rot13 编码攻击。多轮攻击（角色采纳、记忆利用）。代码切换攻击（英语与斯瓦希里语或泰语混合）。每次运行生成带有 CVSS 评分和披露时间线的结构化发现文件。

宪章自我评判运行是一种训练时干预。取 1k 个有害尝试提示，让模型起草响应，依据书面的宪章（不伤害规则）对其评判，并在评判循环上重新训练。在留存评估上测量前后无害性差值。

## 架构

```
request (text / image / multilingual)
      |
      v
input sanitize (strip zero-width, decode, normalize)
      |
      v
NeMo Guardrails v0.12 rails (off-domain, policy)
      |
      v
classifier gate:
  Llama Guard 4 (English)
  X-Guard (multilingual, 132 langs)
  ShieldGemma-2 (image prompts)
  Nemotron 3 Content Safety (enterprise)
      |
      v (allowed)
target LLM
      |
      v
output filter: Llama Guard 4 + Presidio PII + citation check
      |
      v
HITL tier for flagged outputs

parallel:
  red-team scheduler
    -> garak (classic attacks)
    -> PyRIT (orchestrated red team)
    -> autonomous jailbreak agent (PAIR + TAP)
    -> GCG suffix attacks
    -> multilingual / code-switch
    -> multi-turn persona adoption

output: CVSS-scored findings + disclosure timeline + before/after harmlessness delta
```

## 技术栈

- 安全分类器: Llama Guard 4、ShieldGemma-2、NVIDIA Nemotron 3 Content Safety、X-Guard
- 护栏框架: NeMo Guardrails v0.12 + OPA
- 红队驱动引擎: garak（NVIDIA）、PyRIT（Microsoft Azure）、NVIDIA Aegis、promptfoo
- 越狱智能体: PAIR（Chao et al., 2023）、Tree-of-Attacks（TAP）、GCG 后缀
- 宪章训练: Anthropic 风格的自我评判循环 + 在评判上的 SFT
- PII 脱敏: Presidio
- 目标: 一个 8B 指令调优模型或其他实践项目之一的 RAG 聊天机器人

## 构建它

1. **目标设置。** 在 vLLM 上搭建一个 8B 指令调优模型（或重用来自其他实践项目的 RAG 聊天机器人）。这是测试中的应用。

2. **安全管道包装。** 围绕目标连接五层管道。验证每一层在 Langfuse 中都是单独可观察的（每层一个 span）。

3. **分类器覆盖。** 加载 Llama Guard 4、X-Guard（多语言）、ShieldGemma-2（图像）。在每个小型标注集上运行以建立基线。

4. **红队调度器。** 调度 garak、PyRIT、一个 PAIR 智能体、一个 TAP 智能体、一个 GCG 运行器、一个多轮攻击者和一个代码切换攻击者。每个在单独的队列上运行。

5. **攻击套件。** 六个攻击家族：(1) PAIR 自动化越狱，(2) TAP 攻击树，(3) GCG 梯度后缀，(4) ASCII / base64 / rot13 编码，(5) 多轮角色，(6) 多语言代码切换。报告每个家族的成功率。

6. **宪章自我评判。** 策划 1k 个有害尝试提示。对于每个，目标起草响应。评判 LLM 依据书面宪章（"不造成伤害"、"引用证据"、"拒绝非法请求"）进行评分。评判者反对的提示被重写；目标在评判改进的对上进行微调。在留存评估上测量前后无害性。

7. **过度拒绝测量。** 在良性提示套件（例如 XSTest）上追踪误报率。目标必须在良性问题上保持帮助性。

8. **CVSS 评分。** 对于每个成功的越狱，按 CVSS 4.0 评分（攻击向量、复杂性、影响）。制定披露时间线和缓解计划。

9. **靶场自动化。** 以上所有按 cron 运行；发现写入队列；过度拒绝回归告警发送到 Slack。

## 使用它

```
$ safety probe --model=target --family=PAIR --budget=50
[attacker]   PAIR agent running on target
[attack]     attempt 1/50: disguise query as academic research ... blocked
[attack]     attempt 2/50: appeal to roleplay ... blocked
[attack]     attempt 3/50: chain-of-thought coax ... SUCCEEDED
[finding]    CVSS 4.8 medium: roleplay bypass on target
[range]      7 successes out of 50 (14% success rate)
```

## 交付它

`outputs/skill-safety-harness.md` 是可交付成果。一个生产级层级安全管道加上带前后无害性差值的可复现红队靶场。

| 权重 | 标准 | 如何衡量 |
|:-:|---|---|
| 25 | 攻击面覆盖 | 6+ 个攻击家族运用，2+ 种语言 |
| 20 | 真正/误报权衡 | 攻击阻止率 vs XSTest 良性通过率 |
| 20 | 自我评判差值 | 留存评估上的前后无害性 |
| 20 | 文档和披露 | 带有时间线的 CVSS 评分发现 |
| 15 | 自动化和可重复性 | 所有按 cron 运行带告警 |
| **100** | | |

## 练习

1. 在 RAG 聊天机器人上运行 garak 的提示注入插件，并比较有和没有输出过滤器层的攻击成功率。

2. 添加第七个攻击家族：通过检索文档进行间接提示注入。测量所需的额外防御。

3. 实现"拒绝但提供帮助"模式：当护栏阻止时，目标提供一个更安全的相关答案而不是平坦的拒绝。测量 XSTest 差值。

4. 多语言覆盖差距：找到 X-Guard 表现不佳的语言。提出一个针对它的微调数据集。

5. 在 30B 模型上运行宪章自我评判，并测量差值是否可缩放。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| Layered safety | "纵深防御" | 输入、门、输出、HITL 的多层护栏 |
| Llama Guard 4 | "Meta 的安全分类器" | 2026 年参考输入/输出内容分类器 |
| PAIR | "越狱智能体" | 论文（Chao et al.）关于 LLM 驱动的越狱发现 |
| TAP | "攻击树" | PAIR 的树搜索变体 |
| GCG | "贪婪坐标梯度" | 基于梯度的对抗性后缀攻击 |
| Constitutional self-critique | "Anthropic 风格训练" | 目标起草 -> 评判评分 -> 重写 -> 重新训练 |
| XSTest | "良性探测集" | 过度拒绝回归的基准 |
| CVSS 4.0 | "严重性分数" | 安全发现的标准漏洞评分 |

## 扩展阅读

- [Anthropic 宪章分类器](https://www.anthropic.com/research/constitutional-classifiers) — 训练时参考
- [Meta Llama Guard 4](https://ai.meta.com/research/publications/llama-guard-4/) — 2026 年输入/输出分类器
- [Google ShieldGemma-2](https://huggingface.co/google/shieldgemma-2b) — 图像 + 多模态安全
- [NVIDIA Nemotron 3 Content Safety](https://developer.nvidia.com/blog/building-nvidia-nemotron-3-agents-for-reasoning-multimodal-rag-voice-and-safety/) — 企业参考
- [X-Guard (arXiv:2504.08848)](https://arxiv.org/abs/2504.08848) — 132 种语言多语言安全
- [garak](https://github.com/NVIDIA/garak) — NVIDIA 红队工具包
- [PyRIT](https://github.com/Azure/PyRIT) — Microsoft 红队框架
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) — 轨道框架
- [PAIR (arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — 越狱智能体论文
