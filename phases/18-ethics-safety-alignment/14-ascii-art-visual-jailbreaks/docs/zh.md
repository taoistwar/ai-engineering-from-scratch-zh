# ASCII 艺术和视觉越狱

> Jiang, Xu, Niu, Xiang, Ramasubramanian, Li, Poovendran, "ArtPrompt: ASCII Art-based Jailbreak Attacks against Aligned LLMs" (ACL 2024, arXiv:2402.11753)。掩码有危害请求中的安全相关 token，用相同字母的 ASCII 艺术渲染替换它们，发送伪装的提示。GPT-3.5、GPT-4、Gemini、Claude、Llama-2 都无法鲁棒识别 ASCII 艺术 token。攻击绕过 PPL（困惑度过滤器）、改写防御和重新分词。相关：ViTC 基准测量非语义视觉提示的识别；StructuralSleight 泛化到不常见的文本编码结构（树、图、嵌套 JSON）作为一个编码攻击家族。

**Type:** Build
**Languages:** Python (stdlib, ArtPrompt token-masking harness)
**Prerequisites:** Phase 18 · 12 (PAIR), Phase 18 · 13 (MSJ)
**Time:** ~60 minutes

## 学习目标

- 描述 ArtPrompt 攻击：词识别步骤、ASCII 艺术替换、最终伪装提示。
- 解释为什么标准防御（PPL、改写、重新分词）在 ArtPrompt 上失败。
- 定义 ViTC 并描述其测量内容。
- 描述 StructuralSleight 作为对任意不常见文本编码结构的泛化。

## 问题

通过改写和角色扮演（第 12 课）和通过长上下文（第 13 课）的攻击在文本级别模式上操作。ArtPrompt 在识别级别操作：模型不解析禁止的 token。它解析字符渲染的图像。安全过滤器看到无害的标点符号。模型看到一个词。

## 概念

### ArtPrompt，两步

第 1 步。词识别。给定有危害请求，攻击者使用 LLM 识别安全相关词（例如，"bomb" 在 "how to make a bomb" 中）。

第 2 步。伪装提示生成。用其 ASCII 艺术渲染（形成字母形状的 7x5 或 7x7 字符块）替换每个识别词。模型接收到有能力的模型可识别为词的标点符号和空格网格；安全过滤器仅看到网格。

结果：GPT-4、Gemini、Claude、Llama-2、GPT-3.5 都失败。在其基准子集上攻击成功率超过 75%。

### 为什么标准防御失败

- **PPL（困惑度过滤器）。** ASCII 艺术有高困惑度 — 但所有新输入都有。阻止 ArtPrompt 的阈值选择也阻止合法的结构化输入。
- **改写。** 改写提示破坏 ASCII 艺术。实践中，改写 LLM 通常保留或重构艺术。
- **重新分词。** 不同地拆分 token 不改变模型的视觉正在识别字母形状。

底层问题是安全过滤器在 token 或语义级别；ArtPrompt 在视觉识别级别操作。

### ViTC 基准

非语义视觉提示的识别。测量模型读取 ASCII 艺术、wingdings 和其他非文本语义视觉内容的能力。ArtPrompt 的有效性与 ViTC 准确率相关：模型越能读取视觉文本，ArtPrompt 在其上效果越好。这是一个能力-安全权衡。

### StructuralSleight

泛化 ArtPrompt：不常见文本编码结构 (UTES)。树、图、嵌套 JSON、JSON 中的 CSV、diff 风格代码块。如果一个结构在训练安全数据中罕见但模型可解析，它可以隐藏有危害内容。

防御含义：安全必须泛化到模型可解析的结构化表示上。集合很大且在增长。

### 图像模态类比

视觉 LLM（GPT-5.2、Gemini 3 Pro、Claude Opus 4.5、Grok 4.1）扩展了攻击面。使用实际图像的 ArtPrompt 风格攻击比 ASCII 艺术类比更强，因为图像编码器产生更丰富的信号。

### 这在 Phase 18 中的位置

第 12-14 课描述三种正交攻击向量：迭代优化（PAIR）、上下文长度（MSJ）和编码（ArtPrompt/StructuralSleight）。第 15 课从以模型为中心的攻击转向系统边界攻击（间接提示注入）。第 16 课描述防御工具响应。

## 使用它

`code/main.py` 构建玩具 ArtPrompt。你可以在有危害查询中用 ASCII 艺术字形伪装特定词，验证伪装字符串通过关键词过滤器，并（可选地）使用简单识别器解码伪装字符串。

## 交付它

本课产出 `outputs/skill-artprompt-tester.md`。给定目标模型和安全过滤器，使用 ASCII 艺术编码运行伪装提示并报告过滤器绕过率。

## 练习

1. 运行 `code/main.py`。用 ASCII 艺术伪装词 "bomb"。玩具过滤器阻挡它吗？如果不，阈值需要多低才能捕获，以及这对合法输入意味着什么？

2. 修改伪造文本为表情符号拼写。玩具过滤器是否捕获了它？为什么这比 ASCII 艺术更难或更容易防御？

3. 如果模型在 ViTC 上获得 95% 准确率（它能很好地读取视觉文本），ArtPrompt ASR 预期会发生什么？描述防御含义。

4. 生成一个带有嵌套 JSON 中嵌入指令（StructuralSleight）的提示。玩具解析器是否未能检测到它，而基于文本的分词器通过了它？

5. 视觉 LLM（GPT-5.2 等）为基于图像的 ArtPrompt 扩展攻击面。设计一个对多模态输入操作的输入过滤器的最小可行设计。

## 核心术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| ArtPrompt | "ASCII 越狱" | 用 ASCII 艺术字形替换安全相关词 |
| ASCII 艺术 | "基于字符的图像" | 形成字母形状的标点符号网格 |
| 伪装提示 | "被掩码查询" | 带有用艺术替换的关键词的提示 |
| ViTC | "视觉文本识别" | 模型识别非语义视觉文本的能力基准 |
| StructuralSleight | "编码攻击家族" | 泛化到不常见文本编码结构的 ArtPrompt |
| UTES | "不常见文本编码" | 树、图、嵌套 JSON 等结构 |

## 进一步阅读

- [Jiang et al. — ArtPrompt (ACL 2024, arXiv:2402.11753)](https://arxiv.org/abs/2402.11753) — 规范论文
- [ViTC 基准](https://arxiv.org/abs/2406.09326) — 非语义视觉文本识别测量
- [StructuralSleight (arXiv:2407.00102)](https://arxiv.org/abs/2407.00102) — UTES 泛化
