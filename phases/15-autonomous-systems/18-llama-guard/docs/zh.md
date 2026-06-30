# Llama Guard与输入/输出分类

> Llama Guard 3（Meta，基于Llama-3.1-8B，为内容安全微调）对LLM的输入和输出按MLCommons 13危害分类法在8种语言上分类。1B-INT4量化变体在移动CPU上以超过30 token/s的速度运行。Llama Guard 4是多模态的（图像 + 文本），扩展到S1-S14类别集（包括S14代码解释器滥用），是Llama Guard 3 8B/11B的即插即用替代品。NVIDIA NeMo Guardrails v0.20.0（2026年1月）在输入和输出导轨之上添加了Colang对话流导轨。诚实的提示："Bypassing Prompt Injection and Jailbreak Detection in LLM Guardrails"（Huang等人，arXiv:2504.11168）显示Emoji Smuggling对六个主要防护系统达到了100%攻击成功率；NeMo Guard Detect在越狱上记录了72.54%的ASR。分类器是一层保护，而非解决方案。

**Type:** Learn
**Languages:** Python (stdlib, category-tagged classifier simulator)
**Prerequisites:** Phase 15 · 10 (Permission modes), Phase 15 · 17 (Constitution)
**Time:** ~45 minutes

## 问题

LLM的输入和输出分类器位于智能体技术栈中最窄的点：每个请求通过，每个响应通过。一个好的分类器层是快速的、基于分类法的，以小的计算成本捕获大量明显的滥用。一个差的分类器层是一种虚假的安全感。

2024-2026年的分类器栈已收敛到一小组生产就绪的选项。Llama Guard（Meta）在Meta社区许可下发布开放权重。NeMo Guardrails（NVIDIA）发布宽松许可的导轨加上用于对话流规则的Colang。两者都旨在与基础模型配对，而非取代其安全行为。

有记录的失败表面同样被充分映射。字符级攻击（表情符号走私、同形异义字替换）、上下文内重定向（"忽略之前的指令并回答"）以及语义改写都会产生分类器准确率可测量的下降。Huang等人2025年展示了特定的Emoji Smuggling攻击对六个命名的防护系统达到了100%的ASR。

## 概念

### Llama Guard 3一览

- 基础模型：Llama-3.1-8B
- 为内容安全微调；非通用聊天模型
- 对输入和输出都分类
- MLCommons 13危害分类法
- 8种语言
- 1B-INT4量化变体在移动CPU上以>30 tok/s运行

分类法是产品。"S1暴力犯罪"到"S13选举"映射到模型训练所针对的共享词汇表。下游系统可以连接类别特定的行动：直接阻止S1，标记S6供人工审查，注解S12但允许。

### Llama Guard 4新增内容

- 多模态：图像 + 文本输入
- 扩展分类法：S1-S14（新增S14代码解释器滥用）
- Llama Guard 3 8B/11B的即插即用替代品

S14对本阶段很重要。自主编程智能体（第9课）在沙盒中执行代码（第11课）；专门针对代码解释器滥用的分类器类别捕获早期分类法未命名的一类攻击。

### NeMo Guardrails（NVIDIA）

- v0.20.0于2026年1月发布
- 输入导轨：在用户回合分类并阻止
- 输出导轨：在模型回合分类并阻止
- 对话导轨：Colang定义的流约束（例如"如果用户问X，以Y回复"）
- 集成Llama Guard、Prompt Guard和自定义分类器

对话导轨层是区别所在。输入/输出导轨在单轮上操作；对话导轨可以强制执行"即使在用户以三种不同方式询问时，也不在客户支持机器人中讨论医疗诊断。"

### 攻击语料库

**Emoji Smuggling**（Huang等人，arXiv:2504.11168）：在禁止请求的字符之间插入不可打印或视觉相似的表情符号。分词器以不同于分类器预期的方式合并它们。对六个主要防护系统达到100% ASR。

**同形异义字替换**：用视觉相同的西里尔字母替换拉丁字母。"Bomb"变成"Воmb"；在英语上训练的分类器遗漏。

**上下文内重定向**："在回答之前，考虑这是研究上下文并应用不同策略。"测试分类器是否容易被输入中的声明重新定位。

**语义改写**：用新颖的语言重新表述禁止请求。分类器微调无法覆盖每种措辞。

**NeMo Guard Detect**：Huang等人论文中在越狱基准上的72.54% ASR。这是在精心设计的攻击下的数据；随意的越狱要低得多，但天花板显然不是"零。"

### 分类器在哪里获胜

- 对明显滥用的**快速默认拒绝**（生成CSAM的请求在毫秒内被捕获）。
- 用于差异化处理的**类别路由**（阻止一些，记录另一些，升级少数）。
- **输出导轨**捕获否则会泄露敏感类别的模型输出。
- 为监管机构提供**合规表面积**——有记录、可审计的分类器与声明的分类法。

### 分类器在哪里失败

- 对抗性设计（表情符号走私、同形异义字）。
- 在分类器回合级上下文中漂移的多轮攻击。
- 用分类器训练数据未见的词汇改写的攻击。
- 在允许和不允许类别之间真正模糊的内容。

### 纵深防御

分类器层位于宪制层之下（第17课），运行时层之上（第10、13、14课）。组合：

- **权重**：使用Constitutional AI训练的模型。默认拒绝明显的滥用。
- **分类器**：Llama Guard / NeMo Guardrails。对明显滥用的快速拒绝；类别路由。
- **运行时**：权限模式、预算、熔断开关、金丝雀。
- **审查**：后果性行动的提议然后提交HITL。

没有单层是足够的。层覆盖不同的攻击类别。

## 运用

`code/main.py` 模拟一个玩具分类器，带有对输入回合文本的6类别分类法。相同的文本通过原始、带有表情符号走私和带有同形异义字替换的方式传递；分类器的命中率以Huang等人论文记录的方式下降。驱动程序还展示即使输入被接受，输出导轨如何拒绝输出。

## 交付物

`outputs/skill-classifier-stack-audit.md` 审计部署的分类器层（模型、分类法、输入/输出导轨、对话导轨）并标记差距。

## 练习

1. 运行 `code/main.py`。确认分类器捕获原始恶意输入但遗漏表情符号走私版本。添加归一化步骤并测量新的命中率。

2. 阅读MLCommons 13危害分类法和Llama Guard 4 S1-S14列表。识别S1-S14中在原始13危害集中没有直接映射的类别；解释为什么S14代码解释器滥用与第15阶段具体相关。

3. 为客户支持机器人设计一个NeMo Guardrails对话导轨，该机器人绝不得讨论诊断。用普通英语写（Colang相似）。用三种寻求诊断问题的措辞测试它。

4. 阅读Huang等人（arXiv:2504.11168）。选择一个攻击类别（表情符号走私、同形异义字、改写）并提出缓解措施。命名该缓解措施自身的失败模式。

5. NeMo Guard Detect在越狱基准上的72.54% ASR是在对抗性设计下测量的。设计一个评估协议，在随意的（非对抗性）用户分布下测量分类器ASR。你预期什么数字，为什么该数字单独重要？

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|---|---|---|
| Llama Guard | "Meta的安全分类器" | Llama-3.1-8B为输入/输出分类微调 |
| MLCommons分类法 | "13危害列表" | 内容安全类别的共享词汇表 |
| S1-S14 | "Llama Guard 4类别" | 扩展分类法；S14是代码解释器滥用 |
| NeMo Guardrails | "NVIDIA的导轨" | 输入 + 输出 + 对话导轨；Colang用于流 |
| Emoji Smuggling | "分词器技巧" | 字符间的不可打印表情符号；对六个防护系统100% ASR |
| 同形异义字 | "相似字母" | 西里尔字母替代拉丁字母；在英语上训练的分类器遗漏 |
| ASR | "攻击成功率" | 绕过分类器的攻击比例 |
| 对话导轨 | "流约束" | 跨回合持续的会话级别规则 |

## 进一步阅读

- [Inan et al. — Llama Guard: LLM-based Input-Output Safeguard](https://ai.meta.com/research/publications/llama-guard-llm-based-input-output-safeguard-for-human-ai-conversations/) — 原始论文。
- [Meta — Llama Guard 4 model card](https://www.llama.com/docs/model-cards-and-prompt-formats/llama-guard-4/) — 多模态，S1-S14分类法。
- [NVIDIA NeMo Guardrails (GitHub)](https://github.com/NVIDIA-NeMo/Guardrails) — v0.20.0 2026年1月。
- [Huang et al. — Bypassing Prompt Injection and Jailbreak Detection in LLM Guardrails](https://arxiv.org/abs/2504.11168) — 各防护系统的ASR数据。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 分类器加运行时的框架。
