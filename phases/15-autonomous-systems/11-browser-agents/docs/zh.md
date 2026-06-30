# 浏览器智能体与长周期网络任务

> ChatGPT智能体（2025年7月）将Operator和深度研究合并为一个浏览器/终端智能体，并在BrowseComp上达到68.9%的SOTA。OpenAI于2025年8月31日关闭了Operator——产品层的整合。Anthropic收购Vercept将Claude Sonnet在OSWorld上从不到15%提升到72.5%。WebArena-Verified（ServiceNow，ICLR 2026）修复了原始WebArena中11.3个百分点的假阴性率，并发布了258任务的Hard子集。数字是真实的。攻击面也是：OpenAI的准备度负责人公开表示，对浏览器智能体的间接提示注入"不是一个可以完全修补的漏洞"。2025-2026年有记录的攻击：Tainted Memories（Atlas CSRF）、HashJack（Cato Networks）以及Perplexity Comet中的一键劫持。

**Type:** Learn
**Languages:** Python (stdlib, indirect prompt-injection attack surface model)
**Prerequisites:** Phase 15 · 10 (Permission modes), Phase 15 · 01 (Long-horizon agents)
**Time:** ~45 minutes

## 问题

浏览器智能体是一个长周期智能体，它读取不受信任的内容并采取有后果的行为。智能体访问的每个页面都是用户未编写的输入。每个页面上的每个表单都是一个潜在的命令通道。2025-2026年的攻击语料库证明这不是假设性的：Tainted Memories允许攻击者通过构造的页面将恶意指令绑定到智能体的内存中；HashJack将命令隐藏在智能体访问的URL片段中；Perplexity Comet劫持在一次点击中就发生。

防御图景令人不安。OpenAI的准备度负责人说出了沉默的部分：间接提示注入"不是一个可以完全修补的漏洞"。这是因为攻击存在于智能体的读取与行动边界中，而这在架构上是模糊的——模型读取的每个token在原则上都可以被读取为一条指令。

本课命名攻击面，命名基准格局（BrowseComp、OSWorld、WebArena-Verified），并建模一个极简的间接提示注入场景，以便你能够在第14课和第18课中对真实防御进行推理。

## 概念

### 2026年格局，每个系统一段话

**ChatGPT智能体（OpenAI）。** 于2025年7月发布。将Operator（浏览）和深度研究（多小时间歇研究）统一。于2025年8月31日关闭了独立Operator。在BrowseComp上SOTA达68.9%；在OSWorld和WebArena-Verified上有强劲数据。

**Claude Sonnet + Vercept（Anthropic）。** Anthropic收购Vercept专注于计算机使用能力。将Claude Sonnet在OSWorld上从<15%提升到72.5%。Claude Computer Use作为工具API发布。

**Gemini 3 Pro with Browser Use（DeepMind）。** Browser Use集成发布了计算机使用控制；FSF v3（2026年4月，第20课）特别跟踪ML研发领域中的自主性。

**WebArena-Verified（ServiceNow，ICLR 2026）。** 修复了一个有据可查的问题：原始WebArena约有11.3%的假阴性率（标记为失败的任务实际上已解决）。Verified发布版使用人工策展的成功标准重新评分，并添加了一个258任务的Hard子集（ICLR 2026论文，openreview.net/forum?id=94tlGxmqkN）。

### BrowseComp vs OSWorld vs WebArena

| 基准 | 测量内容 | 时间周期 |
|---|---|---|
| BrowseComp | 在时间压力下在开放网络中找到特定事实 | 分钟级 |
| OSWorld | 智能体操作完整桌面（鼠标、键盘、Shell） | 数十分钟级 |
| WebArena-Verified | 在模拟站点中完成事务性网络任务 | 分钟级 |
| Hard子集 | 具有多页面状态转换的WebArena-Verified任务 | 数十分钟级 |

不同的轴。高BrowseComp分数表明智能体能找到事实；它不表明智能体能够预订航班。OSWorld分数更接近"它能否在我的桌面上工作"。WebArena-Verified更接近"它能完成一个流程吗"。任何生产决策都需要匹配任务分布的基准。

### 攻击面，具名罗列

1. **间接提示注入。** 不受信任的页面内容包含指令。智能体读取它们。智能体执行它们。公开示例：2024年Kai Greshake等人，2025年Tainted Memories论文，2026年HashJack（Cato Networks）。
2. **URL片段/查询注入。** 被爬取URL的 `#fragment` 或查询字符串包含命令。从不视觉渲染；仍在智能体的上下文中。
3. **内存绑定攻击。** 页面指示智能体写入持久内存（第12课涵盖持久状态）。下一次会话中，内存触发有效负载，没有可见的触发器。
4. **针对认证会话的CSRF形攻击。** Tainted Memories类：智能体已登录某处；攻击者页面发出状态变更请求，智能体使用用户的Cookie执行。
5. **一键劫持。** 一个视觉上无害的按钮携带智能体遵循的有效负载。Comet类。
6. **智能体宿主机表面的Content-Security-Policy漏洞。** 渲染和工具层本身可以成为攻击向量；浏览器中的浏览器智能体技术栈十分广泛。

### 为什么"无法完全修补"

攻击与智能体的能力同构。智能体必须读取不受信任的内容才能完成工作。智能体读取的任何内容都可能包含指令。智能体遵循的任何指令都可能与用户的实际请求不对齐。防御措施（信任边界、分类器、工具允许列表、对后果性行为的HITL）提高了攻击成本并减小了爆炸半径。它们不闭合此类问题。

这与Lob定理（第8课）的推理模式相同：智能体无法证明下一个token是安全的；它只能建立一个系统，使不安全的token更可检测。

### 实际交付的防御姿态

- **读取/写入边界。** 读取从不具有后果性。写入（提交表单、发布内容、调用具有副作用的工具）如果初始内容来自信任边界之外，则需要新的人类批准。
- **每个任务的工具允许列表。** 智能体可以浏览；它不能发起电汇，除非该工具为该任务显式启用。第13课涵盖预算。
- **会话隔离。** 浏览器智能体会话仅以限定范围的凭据运行。没有生产认证，没有个人邮箱。每次HTTP请求的日志保留用于审计。
- **内容净化器。** 在将获取的HTML拼接到模型上下文之前，剥离已知的不良模式。（减少简单攻击；不能阻止复杂的有效负载。）
- **对后果性行为的HITL。** 提议然后提交模式（第15课）。
- **内存上的金丝雀令牌。** 如果内存条目触发，用户可以看到它（第14课）。

## 运用

`code/main.py` 对三个合成页面对一个微型浏览器智能体运行进行建模。一个页面是良性的，一个在可见文本中有直接提示注入块，一个有URL片段注入（不可见但在智能体上下文中）。脚本展示(a)一个天真智能体会做什么，(b)读取/写入边界捕获什么，(c)净化器捕获什么，(d)两者都无法捕获什么。

## 交付物

`outputs/skill-browser-agent-trust-boundary.md` 范围界定一个提议的浏览器智能体部署：它触及哪些信任区域，它被授权写入什么，以及在第一次运行之前必须就位的防御措施。

## 练习

1. 运行 `code/main.py`。识别哪种攻击被净化器捕获但读取/写入边界没有，以及哪种攻击仅被读取/写入边界捕获。

2. 扩展净化器以检测一类HashJack风格的URL片段注入。在具有合法片段的良性URL上测量误报率。

3. 选择一个你了解的真实浏览器智能体工作流（例如"预订航班"）。列出每次读取和每次写入。标记哪些写入需要HITL以及为什么。

4. 阅读WebArena-Verified ICLR 2026论文。识别原始WebArena评分不可靠的任务类别，解释Verified子集如何解决该问题。

5. 为浏览器智能体设置设计一个内存金丝雀。你将存储什么，在哪里，什么触发警报？

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|---|---|---|
| 间接提示注入 | "不良页面文本" | 智能体读取的页面中不受信任的内容包含智能体执行的指令 |
| Tainted Memories | "内存攻击" | 智能体将攻击者提供的指令写入持久内存；下次会话触发 |
| HashJack | "URL片段攻击" | 隐藏在URL片段/查询字符串中的有效负载在智能体上下文中，但不被视觉渲染 |
| 一键劫持 | "不良按钮" | 可见的交互元素携带后续有效负载，智能体执行之 |
| BrowseComp | "网络搜索基准" | 在开放网络上找到特定事实；分钟级时间周期 |
| OSWorld | "桌面基准" | 完整操作系统控制；多步GUI任务 |
| WebArena-Verified | "修复后的网络任务基准" | ServiceNow重新评分的WebArena，带有Hard子集 |
| 读取/写入边界 | "副作用门控" | 读取从不具有后果性；如果内容不在信任范围内，写入需要新批准 |

## 进一步阅读

- [OpenAI — Introducing ChatGPT agent](https://openai.com/index/introducing-chatgpt-agent/) — Operator和深度研究的合并；BrowseComp SOTA。
- [OpenAI — Computer-Using Agent](https://openai.com/index/computer-using-agent/) — Operator谱系及成为ChatGPT智能体的架构。
- [Zhou et al. — WebArena](https://webarena.dev/) — 原始基准。
- [WebArena-Verified (OpenReview)](https://openreview.net/forum?id=94tlGxmqkN) — ICLR 2026修复子集论文。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 包含计算机使用智能体的攻击面讨论。
