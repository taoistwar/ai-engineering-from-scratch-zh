# 实践项目 17 — 个人 AI 导师（自适应、多模态、带记忆）

> Khanmigo（Khan Academy）、Duolingo Max、Google LearnLM / Gemini for Education、Quizlet Q-Chat 和 Synthesis Tutor 在 2026 年都规模化了自适应多模态辅导。共同形态是苏格拉底式策略（永远不要直接给出答案）、每次交互后更新的学习者模型（贝叶斯知识追踪风格）、语音 + 文本 + 拍照数学输入、课程图谱检索、间隔重复调度以及用于适龄内容的硬性安全过滤器。实践项目是交付一个针对特定学科的导师（K-12 代数或 Python 入门），与 10 名学习者进行为期两周的效能研究，并通过内容安全审计。

**类型:** 实践项目
**语言:** Python（后端，学习者模型），TypeScript（Web 应用），SQL（课程图谱 via Postgres + Neo4j）
**前置条件:** 阶段 5（NLP），阶段 6（语音），阶段 11（LLM 工程），阶段 12（多模态），阶段 14（智能体），阶段 17（基础设施），阶段 18（安全）
**涉及的阶段:** P5 · P6 · P11 · P12 · P14 · P17 · P18
**时间:** 30 小时

## 问题

自适应辅导曾经是教育科技的研究利基。到 2026 年，它是一个消费者产品。Khanmigo 已部署在美国大多数学区。Duolingo Max 达到数千万 MAU。Google 的 LearnLM / Gemini for Education 在 Google Classroom 中提供辅导。Quizlet Q-Chat 与抽认卡并排存在。Synthesis Tutor 凭借面向好奇孩子的导师实现了病毒式传播。共同要素：多模态输入（打字、说话、拍照方程式）、苏格拉底式教学法（先问，后解释）、每次交互后更新的学习者模型以及严格的适龄安全。

你将为一个特定群体构建一个这样的导师。测量标准是一个真正的效能研究：与 10 名学习者进行为期两周的前测和后测。语音循环必须感觉自然（实践项目 03 子栈）。记忆必须尊重隐私。安全过滤器必须通过对 K-12 的 COPPA 意识红队测试。

## 概念

四个组件。**导师策略**是一个苏格拉底式循环：当学习者要求答案时，策略提出引导性问题；当他们回答正确时，它转移到下一个概念；当他们卡住时，它提供脚手架式的提示。**学习者模型**是贝叶斯知识追踪（或一个简单变体），在每个交互后更新每个课程节点的掌握概率。**课程图谱**是一个包含前提边概念的 Neo4j；策略遍历图谱以选取下一个概念。**记忆**是一个事件 + 语义存储（agentmemory 风格），保存过去的互动、错误和偏好。

用户体验是多模态的。文本输入用于打字的答案。语音输入通过 LiveKit + Whisper（重用实践项目 03）。拍照数学问题通过 dots.ocr 或 PaliGemma 2 输入。语音输出通过 Cartesia Sonic-2。安全使用 Llama Guard 4 加上适龄过滤器（阻止成人内容、暴力、自残）和 COPPA 意识记忆保留策略。

效能研究是可交付成果。10 名学习者，前测和后测，两周。报告学习增益差值和置信区间。与非自适应基线（相同内容线性交付，没有导师策略）进行比较。

## 架构

```
learner device
  |
  +-- text         -> web app
  +-- voice        -> LiveKit Agents (ASR + TTS)
  +-- photo math   -> dots.ocr / PaliGemma 2
       |
       v
  tutor policy (LangGraph)
       - Socratic decision head
       - next-concept chooser (curriculum graph walk)
       - hint scaffolder
       - mastery update
       |
       v
  learner model (BKT / item-response theory)
       - per-concept mastery probability
       - spaced-repetition scheduler (SM-2 or FSRS)
       |
       v
  memory (agentmemory-style)
       - episodic: every interaction
       - semantic: learned mistakes, preferences
       - retention policy: COPPA / GDPR aware
       |
       v
  curriculum graph (Neo4j)
       - prerequisite edges
       - OER content attached
       |
       v
  safety:
    Llama Guard 4 + age-appropriate filter
    memory access guarded by learner ID scope
```

## 技术栈

- 学科选择: K-12 代数或 Python 入门（选一个深入）
- 导师策略: LangGraph over Claude Sonnet 4.7（带提示缓存）
- 学习者模型: 贝叶斯知识追踪（经典）或用于间隔的 FSRS
- 课程图谱: Neo4j，包含概念 + 前提边 + OER 内容
- 记忆: agentmemory 风格的持久向量 + 事件 + 语义存储
- 语音: LiveKit Agents 1.0 + Cartesia Sonic-2（重用实践项目 03 子栈）
- 拍照数学: dots.ocr 或 PaliGemma 2 用于方程识别
- 安全: Llama Guard 4 + 自定义适龄过滤器
- 评估: Bloom 级别问题生成、前/后测框架、效能研究工具

## 构建它

1. **课程图谱。** 构建一个包含 50-150 个概念节点的 Neo4j（例如，从"数轴"到"二次公式"的 K-12 代数），带有前提边。将 OER 内容附加到每个节点（Open Textbook、OpenStax）。

2. **学习者模型。** 使用先验初始化贝叶斯知识追踪：猜测、滑过、学习率。在每次交互后更新每个概念的掌握概率。为每个学习者持久化。

3. **导师策略。** LangGraph 包含节点：`read_signal`（学习者的答案是正确的 / 部分的 / 卡住的？）、`select_concept`（遍历课程图谱选取最高优先级的概念）、`scaffold`（苏格拉底式提示）、`update_mastery`。

4. **记忆。** 每次交互写入事件存储。错误和偏好提升到语义记忆。COPPA 意识保留策略：1 年后自动删除，家长可访问。

5. **语音路径。** 连接到导师策略的 LiveKit Agents 工作进程。ASR via Whisper-v3-turbo。TTS via Cartesia Sonic-2。支持打断（重用实践项目 03 机制）。

6. **拍照数学路径。** 上传或捕获图像；运行 dots.ocr 或 PaliGemma 2 识别方程；作为结构化输入提供给导师。

7. **安全。** 每个模型输出通过 Llama Guard 4 + 适龄过滤器（阻止自残、成人内容、暴力）。记忆访问按学习者 ID 范围；家长访问面用于删除。

8. **效能研究。** 10 名学习者，前测（标准化 30 题基线），为期两周的导师互动（每周 3 次会话），后测。与在相同内容上的非自适应基线群体（10 名学习者）进行比较。

9. **每周进度报告。** 每个学习者自动生成一份 PDF 摘要，包含探索的主题、掌握轨迹和推荐的下一步。

## 使用它

```
learner: "I don't understand why 3x + 6 = 12 means x = 2"
[signal]   stuck
[concept]  'isolating variables' (prerequisite: addition-subtraction-equality)
[scaffold] "what number would you subtract from both sides to start?"
learner: "6"
[signal]   correct
[mastery]  addition-subtraction-equality: 0.62 -> 0.77
[concept]  continue 'isolating variables'
[scaffold] "great. now what is 3x / 3 equal to?"
```

## 交付它

`outputs/skill-ai-tutor.md` 是可交付成果。一个特定学科的自适应导师，具有多模态输入、学习者模型、记忆、安全和测量到的效能。

| 权重 | 标准 | 如何衡量 |
|:-:|---|---|
| 25 | 学习增益差值 | 在为期两周的 10 名学习者研究中的前/后测差值 |
| 20 | 苏格拉底式保真度 | 转录样本上的评分标准分数 |
| 20 | 多模态用户体验 | 语音 + 拍照 + 文本端到端一致性 |
| 20 | 安全 + 隐私姿态 | Llama Guard 4 通过率 + COPPA 意识保留 |
| 15 | 课程广度和图谱质量 | 概念覆盖 + 前提图一致性 |
| **100** | | |

## 练习

1. 在有和没有自适应学习者模型（随机概念顺序）的情况下运行效能研究。报告差值。期望自适应胜出，但幅度是有趣的数字。

2. 添加多模态探测：通过文本、语音和照片传递相同的概念问题。测量学习者是否更快地收敛于他们偏好的模态。

3. 构建家长仪表盘：练习的主题、掌握轨迹、即将到来的概念、安全事件（任何护栏触发）。COPPA 对齐。

4. 添加语言切换模式：导师接受西班牙语输入并用西班牙语教学。测量 X-Guard 覆盖。

5. 对记忆隐私进行压力测试：验证即使通过语音片段重新摄入攻击，学习者 A 不能看到学习者 B 的数据。记录尝试的访问并报警。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| Socratic policy | "问，不要直接给" | 导师提出引导性问题而不是给出答案 |
| Bayesian knowledge tracing | "BKT" | 经典学习者模型方程，用于每个概念的掌握概率 |
| FSRS | "自由间隔重复调度器" | 2024 年间隔重复调度器，优于 SM-2 |
| Curriculum graph | "概念 DAG" | Neo4j 包含前提边概念 |
| Episodic memory | "每次交互日志" | 存储每次交互以备后续检索 |
| Semantic memory | "学习到的模式存储" | 从事件中提升的压缩错误和偏好 |
| COPPA | "儿童隐私法" | 美国法律，限制收集 13 岁以下儿童的数据 |

## 扩展阅读

- [Khanmigo（Khan Academy）](https://www.khanmigo.ai) — 参考消费者 K-12 导师
- [Duolingo Max](https://blog.duolingo.com/duolingo-max/) — 参考语言学习导师
- [Google LearnLM / Gemini for Education](https://blog.google/technology/google-deepmind/learnlm) — 托管参考模型
- [Quizlet Q-Chat](https://quizlet.com) — 替代参考
- [Synthesis Tutor](https://www.synthesis.com) — 创业公司参考
- [FSRS 算法](https://github.com/open-spaced-repetition/fsrs4anki) — 间隔重复调度器
- [贝叶斯知识追踪](https://en.wikipedia.org/wiki/Bayesian_knowledge_tracing) — 学习者模型经典
- [LiveKit Agents](https://github.com/livekit/agents) — 语音栈
