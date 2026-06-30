# 群组聊天与发言者选择

> AutoGen GroupChat和AG2 GroupChat在N个智能体之间共享一个对话；一个选择器函数（LLM、轮询或自定义）选择谁下一个发言。这是涌现式多智能体对话的原型——智能体不知道自己在静态图中的角色，它们只是对共享池作出反应。AutoGen v0.2的GroupChat语义在AG2分支中被保留；AutoGen v0.4将其重写为事件驱动的actor模型。微软于2026年2月将AutoGen进入维护模式，并将其与Semantic Kernel合并入Microsoft Agent Framework（RC 2026年2月）。GroupChat原语在AG2和Microsoft Agent Framework中都存活了——学一次，到处使用。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 04 (Primitive Model)
**Time:** ~60 minutes

## 问题

当工作流已知时，静态图（LangGraph）很棒。真实对话不是静态的：有时编码者问审阅者，有时研究者，有时写作者。硬编码每种可能的移交产生边爆炸。你想要*智能体对共享池作出反应*，由某个函数决定谁下一个说话。

这正是AutoGen GroupChat所做的。

## 概念

### 形状

```
              ┌─── 共享池 ────┐
              │  m1  m2  m3...  │
              └─────────┬──────────┘
                        │（每个人都读所有）
      ┌───────┬─────────┼─────────┬───────┐
      ▼       ▼         ▼         ▼       ▼
    智能体A  智能体B  智能体C  智能体D  选择器
                                           │
                                           ▼
                                  "下一个发言者 = C"
```

每个智能体看到每条消息。每个回合一选择器函数被调用来选择谁下一个发言。

### 三种选择器风格

**轮询。** 固定循环。确定性。线性扩展于N但忽略上下文——即使主题是法务审查，编码者也得到发言权。

**LLM选择。** 一次LLM调用阅读最近的池并返回最佳下一个发言者。上下文感知但慢：每回合增加一次LLM调用。AutoGen的默认值。

**自定义。** 一个带有你想要的任何逻辑的Python函数。典型：LLM选择加回退规则（例如"总是在编码者后将发言权给验证者"）。

### ConversableAgent API

```
agent = ConversableAgent(
    name="coder",
    system_message="You write Python.",
    llm_config={...},
)
chat = GroupChat(agents=[coder, reviewer, tester], messages=[])
manager = GroupChatManager(groupchat=chat, llm_config={...})
```

`GroupChatManager` 持有选择器。当智能体完成回合时，管理者调用选择器，返回下一个智能体。循环继续直到终止条件。

### 终止

三种常见模式：

- **最大轮数。** 总回合的硬上限。
- **"TERMINATE"令牌。** 智能体可以发出哨兵消息；管理者在出现时停止。
- **目标达成检查。** 每个回合运行轻量级验证器，完成时停止聊天。

### AutoGen → AG2分裂与Microsoft Agent Framework合并

2025年初，Microsoft开始了围绕事件驱动actor模型的AutoGen（v0.4）重大重写。社区将AutoGen v0.2的GroupChat语义分支为AG2，保留了早期采用者已集成的API。

2026年2月，Microsoft宣布AutoGen将进入维护模式，事件驱动actor模型合并入**Microsoft Agent Framework**（RC 2026年2月，现已与Semantic Kernel合并）。GroupChat概念在两个轨道中都存活；实现细节不同。AG2是v0.2兼容代码的首选上游。

### 何时GroupChat适合

- **涌现对话。** 你不想预连所有可能的下一个发言者。
- **角色混合任务。** 编码者问研究者，研究者问档案员，档案员再问编码者。流不是DAG。
- **探索性问题解决。** 像"头脑风暴会议"，而非"装配线"。

### 何时它失败

- **严格确定性。** LLM选择器可能不一致。相同提示，不同运行，不同下一个发言者。
- **谄媚级联。** 智能体遵从说话最自信的人。显式反提示。
- **上下文膨胀。** 每个智能体阅读每条消息；10个回合后上下文巨大。使用投影（第15课）限定视图范围。
- **热门发言者。** 一个智能体主导对话，因为选择器偏好其专业。引入发言者平衡作为选择器功能。

### 群聊 vs 监督者

相同原语，不同默认值：

- 监督者：一个智能体规划，其他执行。选择器是"问规划者做什么。"
- 群聊：所有智能体是对等的；选择器是共享池上的函数。

两者都使用第04课的四个原语。群聊默认为LLM选择编排和全池共享状态。

## 构建它

`code/main.py` 在标准库中从零实现GroupChat。三个智能体（编码者、审阅者、管理者），轮询和LLM选择变体，以及基于 `TERMINATE` 令牌的终止。

演示打印对话记录加上两种变体的选择器决策追踪。

运行：

```
python3 code/main.py
```

## 运用

`outputs/skill-groupchat-selector.md` 为给定任务配置GroupChat选择器——轮询 vs LLM选择 vs 自定义，以及使用什么选择器输入（最近消息、智能体专业、回合计数）。

## 交付物

检查清单：

- **最大轮数上限。** 始终。典型任务10-20轮。
- **发言者平衡指标。** 追踪每个智能体的回合；当不平衡超过阈值时告警。
- **终止令牌。** `TERMINATE` 或专用验证智能体。
- **投影或限定内存。** 约10条消息后，考虑给每个智能体仅限定视图以防止上下文膨胀。
- **选择器日志。** 对LLM选择变体，同时记录选择器的输入和选择。否则调试不可能。

## 练习

1. 运行 `code/main.py`。在轮询 vs LLM选择下比较对话。每种下哪个智能体主导？
2. 在选择器中添加"每个智能体最多发言"规则。它如何影响记录？
3. 实现目标达成终止：当审阅者返回"approved"时停止。它在轮次上限之前触发的频率如何？
4. 阅读AutoGen GroupChat稳定文档（https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html）。识别 `GroupChatManager` 使用的默认选择器。
5. 阅读AG2仓库（https://github.com/ag2ai/ag2）并比较其v0.2 GroupChat与v0.4事件驱动版本。v0.4添加了什么具体属性（吞吐量、故障容忍、组合性）？

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|----------------|------------------------|
| GroupChat | "一个聊天室中的智能体" | 共享消息池 + 选择器函数。AutoGen / AG2原语。 |
| 发言者选择 | "谁下一个说话" | 选择下一个智能体的函数。轮询、LLM选择或自定义。 |
| GroupChatManager | "会议主持者" | AutoGen组件，拥有选择器并循环回合。 |
| ConversableAgent | "基础智能体" | AutoGen基类；一个可以发送和接收消息的智能体。 |
| 终止令牌 | "'停止'词" | 结束聊天的哨兵字符串（通常是 `TERMINATE`）。 |
| 热门发言者 | "一个智能体主导" | 选择器继续选择同一智能体的失败模式。 |
| 上下文膨胀 | "池无限增长" | 每个智能体阅读每条先前消息；上下文随回合增长。 |
| 投影 | "限定视图" | 角色特定的共享池视图以防止上下文膨胀。 |

## 进一步阅读

- [AutoGen group chat docs](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html) — 参考实现
- [AG2 repo](https://github.com/ag2ai/ag2) — 社区AutoGen v0.2延续
- [Microsoft Agent Framework docs](https://microsoft.github.io/agent-framework/) — 合并的后继者，RC 2026年2月
- [AutoGen v0.4 release notes](https://microsoft.github.io/autogen/stable/) — 事件驱动act或模型重写细节
