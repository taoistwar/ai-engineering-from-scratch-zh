# 聊天机器人 — 从基于规则到神经再到 LLM 智能体

> ELIZA 用模式匹配回复。DialogFlow 映射意图。GPT 从权重中回答。Claude 运行工具并验证。每个时代解决了前一个时代最严重的失败。

**类型：** 学习
**语言：** Python
**先修要求：** 第五阶段 · 13（问答），第五阶段 · 14（信息检索）
**预计时间：** 约75分钟

## 问题

用户说"I want to change my flight。"系统必须弄清楚他们想要什么、缺少什么信息、如何获取它、以及如何完成操作。然后用户说"wait, what if I cancel instead?"系统必须记住上下文、切换任务并保留状态。

对话对于 ML 系统来说是困难的。输入是开放式的。输出必须在多轮中连贯。系统可能需要在世界上执行操作（修改航班、扣款）。每一步错误都对用户可见。

聊天机器人架构经历了四个范式的循环，每个范式的引入都是因为前一个失败得太明显。本课按顺序讲解它们。2026 年的生产场景是最后两者的混合。

## 概念

![Chatbot evolution: rule-based → retrieval → neural → agent](../assets/chatbot.svg)

**基于规则的（ELIZA、AIML、DialogFlow）。** 手工编写的模式匹配用户输入并产生响应。意图分类器路由到预定义的流程。槽位填充状态机收集所需信息。在其设计的狭窄范围内表现出色。在其外部立即失败。仍然在不容忍幻觉的安全关键领域（银行认证、航空预订）发布。

**基于检索的。** 一个 FAQ 风格的系统。编码每对（话语、响应）。在运行时，编码用户的消息并检索最接近的存储响应。想想 Zendesk 经典的"相似文章"功能。对释义的处理比规则更好。不生成，所以没有幻觉。

**神经的（seq2seq）。** 在对话日志上训练的编码器-解码器。从零生成响应。流畅但倾向于通用输出（"I don't know"）和事实漂移。永远不能可靠地切题。Google、Facebook 和 Microsoft 在 2016-2019 年都拥有令人失望的聊天机器人的原因。

**LLM 智能体。** 一个包裹在循环中的语言模型，该循环规划、调用工具并验证结果。不是一个带有长提示的聊天机器人。一个智能体循环：规划 → 调用工具 → 观察结果 → 决定下一步。基于检索优先的接地（RAG）防止其产生幻觉。工具调用使其能实际做事情。这是 2026 年的架构。

这四个范式不是顺序替换关系。2026 年的生产聊天机器人通过所有四个范式路由：破坏性操作用基于规则的、FAQ 用基于检索的、自然措辞用神经生成、模糊的开放式查询用 LLM 智能体。

## 构建它

### 步骤 1：基于规则的模式匹配

```python
import re


class RulePattern:
    def __init__(self, pattern, response_template):
        self.regex = re.compile(pattern, re.IGNORECASE)
        self.template = response_template


PATTERNS = [
    RulePattern(r"my name is (\w+)", "Nice to meet you, {0}."),
    RulePattern(r"i (need|want) (.+)", "Why do you {0} {1}?"),
    RulePattern(r"i feel (.+)", "Why do you feel {0}?"),
    RulePattern(r"(.*)", "Tell me more about that."),
]


def rule_based_respond(user_input):
    for pattern in PATTERNS:
        m = pattern.regex.match(user_input.strip())
        if m:
            return pattern.template.format(*m.groups())
    return "I don't understand."
```

20 行的 ELIZA。反射技巧（"I feel sad" → "Why do you feel sad"）是 Weizenbaum 1966 年的经典心理治疗师演示。仍然具有启发性。

### 步骤 2：基于检索的（FAQ）

此示例代码需要`pip install sentence-transformers`（其依赖 torch）。本课的可运行`code/main.py`使用标准库的 Jaccard 相似度代替，因此课程无需外部依赖即可运行。

```python
from sentence_transformers import SentenceTransformer
import numpy as np


FAQ = [
    ("how do i reset my password", "Go to Settings > Security > Reset Password."),
    ("how do i cancel my order", "Go to Orders, find the order, click Cancel."),
    ("what is your return policy", "30-day returns on unused items, original packaging."),
]


encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
faq_questions = [q for q, _ in FAQ]
faq_embeddings = encoder.encode(faq_questions, normalize_embeddings=True)


def faq_respond(user_input, threshold=0.5):
    q_emb = encoder.encode([user_input], normalize_embeddings=True)[0]
    sims = faq_embeddings @ q_emb
    best = int(np.argmax(sims))
    if sims[best] < threshold:
        return None
    return FAQ[best][1]
```

基于阈值的拒绝是关键设计选择。如果最佳匹配不够接近，返回`None`并让系统升级。

### 步骤 3：神经生成（基线）

使用一个小的指令调优的编码器-解码器（FLAN-T5）或微调的对话模型。在 2026 年单独使用它无法用于生产（矛盾、离题漂移、事实错误），但在混合系统中用于自然措辞。DialoGPT 式仅解码器模型需要显式的轮次分隔符和 EOS 处理才能产生连贯的回复；FLAN-T5 的 text2text 流程对教学示例开箱即用。

```python
from transformers import pipeline

chatbot = pipeline("text2text-generation", model="google/flan-t5-small")

response = chatbot("Respond politely to: Hi there!", max_new_tokens=40)
print(response[0]["generated_text"])
```

### 步骤 4：LLM 智能体循环

2026 年生产形态：

```python
def agent_loop(user_message, tools, llm, max_steps=5):
    history = [{"role": "user", "content": user_message}]
    for _ in range(max_steps):
        response = llm(history, tools=tools)
        tool_call = response.get("tool_call")
        if tool_call:
            tool_name = tool_call.get("name")
            args = tool_call.get("arguments")
            if not isinstance(tool_name, str) or tool_name not in tools:
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": str(tool_name), "content": f"error: unknown tool {tool_name!r}"})
                continue
            if not isinstance(args, dict):
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": tool_name, "content": f"error: arguments must be a dict, got {type(args).__name__}"})
                continue
            fn = tools[tool_name]
            result = fn(**args)
            history.append({"role": "assistant", "tool_call": tool_call})
            history.append({"role": "tool", "name": tool_name, "content": result})
        else:
            return response["content"]
    return "I could not complete the task in the step budget."
```

需要指出的三件事。工具是 LLM 可以调用的可调用函数。当 LLM 返回最终答案而不是工具调用时，循环终止。步数预算防止在模糊任务上的无限循环。

真实生产添加：检索优先的接地（在每次 LLM 调用前注入相关文档）、护栏（拒绝在没有确认的情况下进行破坏性操作）、可观测性（记录每个步骤）和评估（自动检查智能体行为保持在规范范围内）。

### 步骤 5：混合路由

```python
def hybrid_chat(user_input):
    if is_destructive_action(user_input):
        return structured_flow(user_input)

    faq_answer = faq_respond(user_input, threshold=0.6)
    if faq_answer:
        return faq_answer

    return agent_loop(user_input, tools, llm)


def is_destructive_action(text):
    danger_words = ["delete", "cancel", "charge", "refund", "transfer"]
    return any(w in text.lower() for w in danger_words)
```

模式：任何破坏性事物用确定性规则，预设 FAQ 用检索，其他一切用 LLM 智能体。这就是 2026 年客户支持系统发布的内容。

## 使用它

2026 年技术栈：

| 用例 | 架构 |
|---------|---------------|
| 预订、支付、认证 | 基于规则的状态机 + 槽位填充 |
| 客户支持 FAQ | 在策划答案上的检索 |
| 开放式帮助聊天 | 带 RAG + 工具调用的 LLM 智能体 |
| 内部工具 / IDE 助手 | 带工具调用的 LLM 智能体（搜索、读取、写入） |
| 伴侣 / 角色聊天机器人 | 调优的 LLM 带角色系统提示、知识检索 |

在生产中始终使用混合路由。没有单一架构能很好地处理每个请求。路由层本身通常是一个小的意图分类器。

## 仍未被捕获就发布的失败模式

- **自信编造。** LLM 智能体声称完成了一个它没有做的操作。缓解：验证结果，记录工具调用，永远不要让 LLM 在没有成功工具返回的情况下声称做了某事。
- **提示注入。** 用户插入覆盖系统提示的文本。在 2025 年 OWASP 面向 LLM 应用的 Top 10 中排名 LLM01。两种类型：直接注入（粘贴到聊天中）和间接注入（隐藏在智能体读取的文档、电子邮件或工具输出中）。

  攻击率因场景而异。在通用工具使用和编码基准测试中，前沿模型的测量成功率约 0.5-8.5%。特定的高风险设置（对 AI 编码智能体的自适应攻击、脆弱的编排）已达到约 84%。生产 CVE 包括 EchoLeak（CVE-2025-32711，CVSS 9.3）——一个由攻击者控制的电子邮件触发的 Microsoft 365 Copilot 中的零点击数据泄露漏洞。

  缓解措施：在整个循环中将用户输入视为不可信的；在工具调用前净化；将工具输出与主提示隔离；使用规划-验证-执行（PVE）模式，智能体先规划，然后根据该规划验证每个操作再执行（这阻止工具结果注入新的未规划操作）；要求用户确认破坏性操作；对工具范围应用最小权限。

  在运行时，没有任何量的提示工程可以完全消除这种风险。需要外部运行时防御层（LLM Guard、允许列表验证、语义异常检测）。
- **范围蠕变。** 智能体因工具调用返回了切向相关信息而偏离任务。缓解：收窄工具契约；保持系统提示聚焦；为离题率添加评估。
- **无限循环。** 智能体不断调用相同的工具。缓解：步数预算，工具调用去重，LLM 评判"我们是否在取得进展。"
- **上下文窗口耗尽。** 长对话将最早的轮次推出上下文。缓解：总结旧轮次，按相似度检索相关过去轮次，或使用长上下文模型。

## 交付它

保存为 `outputs/skill-chatbot-architect.md`：

```markdown
---
name: chatbot-architect
description: 为给定的用例设计聊天机器人技术栈。
version: 1.0.0
phase: 5
lesson: 17
tags: [nlp, agents, chatbot]
---

给定一个产品上下文（用户需求、合规约束、可用工具、数据量），输出：

1. 架构。基于规则、基于检索、神经、LLM 智能体，或混合（指定哪些路径走向哪里）。
2. 如果适用，选择 LLM。命名模型族（Claude、GPT-4、Llama-3.1、Mixtral）。匹配工具使用质量和成本。
3. 接地策略。RAG 源、检索方法（见第 14 课）、工具契约。
4. 评估计划。任务成功率、工具调用正确性、离题率、在保留对话上的幻觉率。

拒绝为任何破坏性操作（支付、账户删除、数据修改）推荐没有结构化确认流的纯 LLM 智能体。如果智能体对任何内容具有写入权限，拒绝跳过提示注入审计。
```

## 练习

1. **简单。** 用 10 个模式实现上述基于规则的响应，用于咖啡店订购机器人。测试边缘情况：双重订单、修改、取消、不清楚的意图。
2. **中等。** 构建一个混合 FAQ + LLM 后备方案。SaaS 产品 50 个预设 FAQ 条目，LLM 后备方案带文档站点的检索。在 100 个真实支持问题上测量拒绝率和准确率。
3. **困难。** 用三个工具（搜索、读取用户数据、发送邮件）实现上述智能体循环。在 50 个测试场景上运行评估，包括提示注入尝试。报告离题率、失败任务率和任何注入成功。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| 意图 | 用户想要什么 | 类别标签（book_flight、reset_password）。路由到一个处理器。 |
| 槽位 | 一条信息 | 机器人需要的参数（日期、目的地）。槽位填充是询问序列。 |
| RAG | 检索加生成 | 检索相关文档，然后为 LLM 响应接地。 |
| 工具调用 | 函数调用 | LLM 发出一个带有名称 + 参数的结构化调用。运行时执行，返回结果。 |
| 智能体循环 | 规划、执行、验证 | 控制器运行 LLM 调用与工具调用交替，直到任务完成。 |
| 提示注入 | 用户攻击提示 | 试图覆盖系统提示的恶意输入。 |

## 扩展阅读

- [Weizenbaum (1966). ELIZA — A Computer Program For the Study of Natural Language Communication](https://web.stanford.edu/class/cs124/p36-weizenabaum.pdf) — 原始的基于规则聊天机器人论文。
- [Thoppilan et al. (2022). LaMDA: Language Models for Dialog Applications](https://arxiv.org/abs/2201.08239) — Google 晚期的神经聊天机器人论文，就在 LLM 智能体接管之前。
- [Yao et al. (2022). ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — 命名智能体循环模式的论文。
- [Anthropic's guide on building effective agents](https://www.anthropic.com/research/building-effective-agents) — 2024 年生产指南，在 2026 年仍然有效。
- [Greshake et al. (2023). Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection](https://arxiv.org/abs/2302.12173) — 提示注入论文。
- [OWASP Top 10 for LLM Applications 2025 — LLM01 Prompt Injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) — 使提示注入成为首要安全关切的排名。
- [AWS — Securing Amazon Bedrock Agents against Indirect Prompt Injections](https://aws.amazon.com/blogs/machine-learning/securing-amazon-bedrock-agents-a-guide-to-safeguarding-against-indirect-prompt-injections/) — 实用的编排层防御，包括规划-验证-执行和用户确认流。
- [EchoLeak (CVE-2025-32711)](https://www.vectra.ai/topics/prompt-injection) — 来自间接提示注入的规范零点击数据泄露 CVE。为什么具有写入权限的智能体需要运行时防御的参考案例。
