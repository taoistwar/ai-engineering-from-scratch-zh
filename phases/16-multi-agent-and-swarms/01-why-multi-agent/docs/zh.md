# 为什么需要多智能体？

> 一个智能体会撞墙。聪明的做法不是更大的智能体——而是更多的智能体。

**Type:** Learn
**Languages:** TypeScript
**Prerequisites:** Phase 14 (Agent Engineering)
**Time:** ~60 minutes

## 学习目标

- 识别单智能体天花板（上下文溢出、混合专业知识、顺序瓶颈），并解释何时拆分为多个智能体是正确的做法
- 比较编排模式（流水线、并行扇出、监督者、层次化），为给定的任务结构选择合适的模式
- 设计具有清晰角色边界、共享状态和通信契约的多智能体系统
- 分析多智能体复杂性的权衡（延迟、成本、调试难度）与单智能体简单性的对比

## 问题

你在第14阶段构建了一个单智能体。它能工作。它可以读取文件、运行命令、调用API并对结果进行推理。然后你将它指向一个真实的代码库：200个文件、三种语言、依赖基础设施的测试，以及在编写代码之前研究外部API的需求。

智能体喘不过气来。不是因为LLM愚蠢，而是因为任务超出了单个智能体循环能够处理的范围。上下文窗口被文件内容填满。智能体忘记了它在40次工具调用之前读取的内容。它试图同时成为研究者、编码者和审阅者，三者都做得不好。

这就是单智能体天花板。每当任务需要以下条件时你就会遇到它：

- **超出单个窗口容纳范围的更多上下文**——读取50个文件超过200k token
- **不同阶段需要不同的专业知识**——研究需要的提示与代码生成不同
- **可以并行进行的工作**——为什么不并行读取三个文件，而要顺序读取？

## 概念

### 单智能体天花板

单智能体是一个循环、一个上下文窗口、一个系统提示。想象一下：

```
┌─────────────────────────────────────────┐
│            单智能体                        │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │         上下文窗口                   │  │
│  │                                   │  │
│  │  研究笔记                          │  │
│  │  + 代码文件                        │  │
│  │  + 测试输出                        │  │
│  │  + 审查反馈                        │  │
│  │  + API文档                         │  │
│  │  + ...                            │  │
│  │                                   │  │
│  │  ██████████████████████ 已满 ███   │  │
│  └───────────────────────────────────┘  │
│                                         │
│  一个系统提示试图覆盖                     │
│  研究 + 编码 + 审查 + 测试                │
│                                         │
│  结果: 什么都平庸                         │
└─────────────────────────────────────────┘
```

三个方面出现问题：

1. **上下文饱和**——工具结果不断累积。到第30回合，智能体已经消耗了150k token的文件内容、命令输出和先前的推理。第5回合的关键细节丢失了。

2. **角色混淆**——一个写着"你是研究者、编码者、审阅者和测试者"的系统提示产生了一个半研究、半编码且永远无法完成审查的智能体。

3. **顺序瓶颈**——智能体读取文件A，然后文件B，然后文件C。三次串行LLM调用。三次串行工具执行。没有并行性。

### 多智能体解决方案

拆分工作。每个智能体只做一项工作，拥有一个上下文窗口和一个为该工作调优的系统提示：

```
┌──────────────────────────────────────────────────────────┐
│                    编排者                                  │
│                                                          │
│  "构建一个用于用户管理的REST API"                            │
│                                                          │
│         ┌──────────┬──────────┬──────────┐               │
│         │          │          │          │               │
│         ▼          ▼          ▼          ▼               │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│   │  研究者   │ │  编码者   │ │  审阅者   │ │  测试者   │  │
│   │          │ │          │ │          │ │          │  │
│   │ 阅读     │ │ 基于     │ │ 检查     │ │ 运行     │  │
│   │ 文档，   │ │ 研究和   │ │ 代码     │ │ 测试，   │  │
│   │ 寻找     │ │ 规范     │ │ 质量，   │ │ 报告     │  │
│   │ 模式     │ │ 编写代码 │ │ 发现Bug  │ │ 结果     │  │
│   └─────┬────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │
│         │           │            │             │         │
│         └───────────┴────────────┴─────────────┘         │
│                          │                               │
│                     合并结果                              │
└──────────────────────────────────────────────────────────┘
```

每个智能体都有：
- 一个聚焦的系统提示（"你是一个代码审阅者。你唯一的工作是发现Bug。"）
- 自己的上下文窗口（不被其他智能体的工作污染）
- 清晰的输入/输出契约（接收研究笔记，输出代码）

### 这样做的真实系统

**Claude Code子智能体**——当Claude Code使用 `Task` 生成子智能体时，它创建了一个具有限定任务的子智能体。父智能体保持其上下文清洁。子智能体做聚焦的工作并返回摘要。

**Devin**——运行规划器智能体、编码者智能体和浏览器智能体。规划器将工作分解为步骤。编码者编写代码。浏览器研究文档。每个都有独立的上下文。

**多智能体编程团队（SWE-bench）**——SWE-bench上表现最好的系统使用一个阅读代码库的研究者、一个设计修复方案的规划者和一个实现修复方案的编码者。单智能体系统得分更低。

**ChatGPT深度研究**——并行生成多个搜索智能体，每个探索不同的角度，然后综合结果。

### 光谱

多智能体不是二元的。它是一个光谱：

```
简单 ──────────────────────────────────────────── 复杂

 单智能体      子智能体      流水线         团队           群体

 ┌───┐       ┌───┐        ┌───┐───┐    ┌───┐───┐    ┌─┐┌─┐┌─┐
 │ A │       │ A │        │ A │ B │    │ A │ B │    │ ││ ││ │
 └───┘       └─┬─┘        └───┘─┬─┘    └─┬─┘─┬─┘    └┬┘└┬┘└┬┘
               │                │        │   │       ┌┴──┴──┴┐
             ┌─┴─┐          ┌───┘───┐    │   │       │共享   │
             │ a │          │ C │ D │  ┌─┴───┴─┐    │状态   │
             └───┘          └───┘───┘  │ 消息   │    └───────┘
                                       │ 总线   │
 1个循环      父 +           逐个阶段    │       │    N个对等体，
 1个上下文    子任务                     └───────┘    涌现性
                                       显式角色      行为
```

**单智能体**——一个循环，一个提示。适合简单任务。

**子智能体**——父智能体为聚焦的子任务生成子智能体。父智能体维护计划。子智能体报告回来。这是Claude Code所做的。

**流水线**——智能体按顺序运行。智能体A的输出成为智能体B的输入。适合阶段性工作流：研究 -> 编码 -> 审查 -> 测试。

**团队**——智能体在共享消息总线上并行运行。每个都有一个角色。编排者进行协调。适合不同的技能需要同时使用的情况。

**群体**——许多相同或近乎相同的智能体，共享状态。没有固定的编排者。智能体从队列中拾取工作。适合高吞吐量并行任务。

### 四种多智能体模式

#### 模式1：流水线

```
输入 ──▶ 智能体A ──▶ 智能体B ──▶ 智能体C ──▶ 输出
          (研究)     (编码)     (审查)
```

每个智能体转换数据并向前传递。推理简单。一个阶段的失败会阻塞其余部分。

#### 模式2：扇出/扇入

```
                ┌──▶ 智能体A ──┐
                │              │
输入 ──▶ 拆分  ├──▶ 智能体B ──├──▶ 合并 ──▶ 输出
                │              │
                └──▶ 智能体C ──┘
```

将工作拆分到并行的智能体，然后合并结果。适合可分解为独立子任务的任务。

#### 模式3：编排者-工作者

```
                    ┌──────────┐
                    │  编排者   │
                    └──┬───┬───┘
                  任务 │   │ 任务
                 ┌─────┘   └─────┐
                 ▼               ▼
           ┌──────────┐   ┌──────────┐
           │ 工作者A   │   │ 工作者B   │
           └──────────┘   └──────────┘
```

一个智能编排者决定做什么，委派给工作者，并综合结果。编排者本身就是一个具有生成工作者工具的智能体。

#### 模式4：对等群体

```
         ┌───┐ ◄──── 消息 ────▶ ┌───┐
         │ A │                  │ B │
         └─┬─┘                  └─┬─┘
           │                      │
      消息 │     ┌──────────────┐  │ 消息
           └────▶│    共享      │◄─┘
                 │    状态      │
           ┌────▶│   / 队列     │◄──┐
           │     └──────────────┘    │
      消息 │                          │ 消息
         ┌─┴─┐                      ┌─┴─┐
         │ C │ ◄──── 消息 ────▶    │ D │
         └───┘                      └───┘
```

没有中央编排者。智能体点对点通信。决策从交互中涌现。更难调试，但可扩展到许多智能体。

### 何时不应使用多智能体

多智能体增加了复杂性。智能体之间的每条消息都是一个潜在的故障点。调试从"阅读一次对话"变成"追踪五个智能体间的消息"。

**保持单智能体当：**
- 任务适配单个上下文窗口（约100k token以下的工作数据）
- 你不需要不同阶段使用不同的系统提示
- 顺序执行足够快
- 任务足够简单，拆分它增加的开销超过价值

**复杂性成本：**
- 每个智能体边界是一个有损压缩步骤：智能体A的完整上下文被概括为一条给智能体B的消息
- 协调逻辑（谁做什么、何时、以什么顺序）本身就是一个Bug来源
- 延迟增加：N个智能体意味着至少N次串行LLM调用，如果需要来回通信则更多
- 成本倍增：每个智能体独立消耗token

经验法则：如果任务少于20次工具调用且适配100k token，保持单智能体。

```figure
swarm-messages
```

## 构建它

### 步骤1：过载的单智能体

这里是一个试图做所有事的单智能体。它有一个庞大的系统提示和一个容纳研究、代码和审查的上下文窗口：

```typescript
type AgentResult = {
  content: string;
  tokensUsed: number;
  toolCalls: number;
};

async function singleAgentApproach(task: string): Promise<AgentResult> {
  const systemPrompt = `You are a full-stack developer. You must:
1. Research the requirements
2. Write the code
3. Review the code for bugs
4. Write tests
Do ALL of these in a single conversation.`;

  const contextWindow: string[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const research = await fakeLLMCall(systemPrompt, `Research: ${task}`);
  contextWindow.push(research.output);
  totalTokens += research.tokens;
  totalToolCalls += research.calls;

  const code = await fakeLLMCall(
    systemPrompt,
    `Given this research:\n${contextWindow.join("\n")}\n\nNow write code for: ${task}`
  );
  contextWindow.push(code.output);
  totalTokens += code.tokens;
  totalToolCalls += code.calls;

  const review = await fakeLLMCall(
    systemPrompt,
    `Given all previous context:\n${contextWindow.join("\n")}\n\nReview the code.`
  );
  contextWindow.push(review.output);
  totalTokens += review.tokens;
  totalToolCalls += review.calls;

  return {
    content: contextWindow.join("\n---\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

这种方法的问题：
- 上下文窗口随着每个阶段增长。到审查步骤时，它包含研究笔记 + 代码 + 先前的推理。
- 系统提示是通用的。不能为每个阶段调优。
- 没有任何东西是并行运行的。

### 步骤2：专家智能体

现在拆分它。每个智能体只做一项工作：

```typescript
type SpecialistAgent = {
  name: string;
  systemPrompt: string;
  run: (input: string) => Promise<AgentResult>;
};

function createSpecialist(name: string, systemPrompt: string): SpecialistAgent {
  return {
    name,
    systemPrompt,
    run: async (input: string) => {
      const result = await fakeLLMCall(systemPrompt, input);
      return {
        content: result.output,
        tokensUsed: result.tokens,
        toolCalls: result.calls,
      };
    },
  };
}

const researcher = createSpecialist(
  "researcher",
  "You are a technical researcher. Read documentation, find patterns, and summarize findings. Output only the facts needed for implementation."
);

const coder = createSpecialist(
  "coder",
  "You are a senior TypeScript developer. Given requirements and research notes, write clean, tested code. Nothing else."
);

const reviewer = createSpecialist(
  "reviewer",
  "You are a code reviewer. Find bugs, security issues, and logic errors. Be specific. Cite line numbers."
);
```

每个专家都有一个聚焦的提示。每个都获得一个干净的上下文窗口，只包含它需要的输入。

### 步骤3：通过消息协调

用显式的消息传递将专家们连接起来：

```typescript
type AgentMessage = {
  from: string;
  to: string;
  content: string;
  timestamp: number;
};

async function multiAgentApproach(task: string): Promise<AgentResult> {
  const messages: AgentMessage[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const researchResult = await researcher.run(task);
  messages.push({
    from: "researcher",
    to: "coder",
    content: researchResult.content,
    timestamp: Date.now(),
  });
  totalTokens += researchResult.tokensUsed;
  totalToolCalls += researchResult.toolCalls;

  const coderInput = messages
    .filter((m) => m.to === "coder")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const codeResult = await coder.run(coderInput);
  messages.push({
    from: "coder",
    to: "reviewer",
    content: codeResult.content,
    timestamp: Date.now(),
  });
  totalTokens += codeResult.tokensUsed;
  totalToolCalls += codeResult.toolCalls;

  const reviewerInput = messages
    .filter((m) => m.to === "reviewer")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const reviewResult = await reviewer.run(reviewerInput);
  messages.push({
    from: "reviewer",
    to: "orchestrator",
    content: reviewResult.content,
    timestamp: Date.now(),
  });
  totalTokens += reviewResult.tokensUsed;
  totalToolCalls += reviewResult.toolCalls;

  return {
    content: messages.map((m) => `[${m.from} -> ${m.to}]: ${m.content}`).join("\n\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

每个智能体只接收发给它的消息。没有上下文污染。研究者的50k token文档阅读从不进入审阅者的上下文。

### 步骤4：比较

```typescript
async function compare() {
  const task = "Build a rate limiter middleware for an Express.js API";

  console.log("=== Single Agent ===");
  const single = await singleAgentApproach(task);
  console.log(`Tokens: ${single.tokensUsed}`);
  console.log(`Tool calls: ${single.toolCalls}`);

  console.log("\n=== Multi-Agent ===");
  const multi = await multiAgentApproach(task);
  console.log(`Tokens: ${multi.tokensUsed}`);
  console.log(`Tool calls: ${multi.toolCalls}`);
}
```

多智能体版本使用更多的总token（三个智能体，三次独立的LLM调用），但每个智能体的上下文保持清洁。每个阶段的质量提高了，因为系统提示是专门化的。

## 运用

本课产生一个可复用的提示，用于决定何时转向多智能体。参见 `outputs/prompt-multi-agent-decision.md`。

## 练习

1. 添加第四个专家：一个"测试者"智能体，接收编码者的代码和审阅者的审查反馈，然后编写测试
2. 修改流水线，使审阅者可以将反馈发送回编码者进行修订循环（最多2轮）
3. 将顺序流水线转换为扇出：并行运行研究者和一个"需求分析器"智能体，然后在传递给编码者之前合并它们的输出

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|----------------|----------------------|
| 群体 | "AI智能体的蜂巢思维" | 一组具有共享状态且无固定领导者的对等智能体。行为从局部交互中涌现。 |
| 编排者 | "老板智能体" | 一个其工具包括生成和管理其他智能体的智能体。它规划和委派，但可能不执行实际工作。 |
| 协调者 | "交通警察" | 一个非智能体组件（通常只是代码，而非LLM），根据规则在智能体之间路由消息。 |
| 共识 | "智能体达成一致" | 多个智能体在继续之前必须达成协议的协议。当冲突输出需要解决时使用。 |
| 涌现行为 | "智能体自己弄明白了" | 从智能体交互中产生但未被显式编程的系统级模式。可能有用也可能有害。 |
| 扇出/扇入 | "智能体的Map-Reduce" | 将任务拆分到并行智能体（扇出），然后组合它们的结果（扇入）。 |
| 消息传递 | "智能体彼此交谈" | 智能体之间的通信机制：从一个智能体发送到另一个智能体的结构化数据，替代共享上下文窗口。 |

## 进一步阅读

- [The Landscape of Emerging AI Agent Architectures](https://arxiv.org/abs/2409.02977) - 多智能体模式综述
- [AutoGen: Enabling Next-Gen LLM Applications](https://arxiv.org/abs/2308.08155) - 微软的多智能体对话框架
- [Claude Code subagents documentation](https://docs.anthropic.com/en/docs/claude-code) - Claude Code如何通过Task委派
- [CrewAI documentation](https://docs.crewai.com/) - 基于角色的多智能体框架
