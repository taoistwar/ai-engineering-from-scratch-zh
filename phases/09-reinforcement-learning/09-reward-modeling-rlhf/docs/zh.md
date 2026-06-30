# 奖励建模与 RLHF

> 人类无法为"好的助手响应"编写奖励函数，但他们可以比较两个响应并选择更好的。在那些比较上拟合奖励模型，然后用 RL 优化语言模型。Christiano 2017。InstructGPT 2022。将 GPT-3 变成 ChatGPT 的配方。在 2026 年它大部分正被 DPO 取代——但心理模型保留。

**类型：** 构建
**语言：** Python
**前置条件：** 第五阶段 · 05（情感分析），第九阶段 · 08（PPO）
**时间：** 约 45 分钟

## 问题

你在下一个 token 预测目标上训练了语言模型。它写出语法正确的英语。它也撒谎、漫无边际并拒绝拒绝。你不能用更多的预训练来修复此——网页文本是问题，不是解药。

你想要一个*标量奖励*，说"对于指令 X，响应 A 比响应 B 好。"用手写该奖励函数是不可能的。"有帮助性"不是 token 上的闭式表达式。但人类可以比较两个输出并标记偏好。这在大规模上收集很便宜。

RLHF（Christiano 等人 2017；Ouyang 等人 2022）将偏好转换为奖励模型，然后通过 PPO 优化 LM。三步：SFT → RM → PPO。它是发布 ChatGPT、Claude、Gemini 和 2023-2025 年每个其他对齐 LLM 的配方。

在 2026 年，PPO 步大部分被 DPO（第十阶段 · 08）取代，因为它更便宜且对对齐调整几乎一样好。但*奖励模型*部分仍然支撑着每个 Best-of-N 采样器、每个来自可验证奖励的 RL 流水线和每个使用过程奖励模型的推理模型。理解 RLHF 你就理解了整个对齐技术栈。

## 概念

![三阶段 RLHF：SFT、在配对偏好上的 RM 训练、带有 KL 惩罚的 PPO](../assets/rlhf.svg)

**阶段 1：监督微调（SFT）。** 从预训练基座模型开始。在目标行为的人类书写示例上微调（遵循指令的响应、有帮助的回复等）。结果：模型 `π_SFT` 被*偏向良好行为*但仍具有无界动作空间。

**阶段 2：奖励模型训练。**

- 收集对提示 `x` 的成对响应 `(y_+, y_-)`，由人类标注为"y_+ 优于 y_-"。
- 训练奖励模型 `R_φ(x, y)` 为 `y_+` 分配更高分数。
- 损失：**Bradley-Terry 配对 logistic**：

  `L(φ) = -E[ log σ(R_φ(x, y_+) - R_φ(x, y_-)) ]`

  σ 是 sigmoid。奖励差异隐含偏好的对数几率。BT 自 1952 年（Bradley-Terry）起就成为标准，是现代 RLHF 中的主导选择。

- `R_φ` 通常从 SFT 模型初始化，顶部带有一个标量头。相同的 transformer 骨干；单线性层输出奖励。

**阶段 3：针对 RM 的 PPO，带 KL 惩罚。**

- 从 `π_SFT` 初始化可训练策略 `π_θ`。保持冻结*参考* `π_ref = π_SFT`。
- 在响应 `y` 末尾的奖励：

  `r_total(x, y) = R_φ(x, y) - β · KL(π_θ(·|x) || π_ref(·|x))`

  KL 惩罚防止 `π_θ` 从 `π_SFT` 任意漂移——它是*正则化器*，不是硬信任域。`β` 通常 `0.01`-`0.05`。
- 用此奖励运行 PPO（第 08 课）。优势在 token 级轨迹上计算，但 RM 仅对完整响应评分。

**为什么 KL？** 没有它，PPO 会愉快地找到奖励黑客策略——RM 只在分布内补全上训练。分布外响应可能比任何人类写的都得分更高。KL 保持 `π_θ` 靠近 RM 被训练的流形。它是 RLHF 中最重要的旋钮。

**2026 年状态：**

- **DPO**（Rafailov 2023）：闭式代数将阶段 2+3 折叠为对偏好数据的单一监督损失。无 RM，无 PPO。在计算量的一小部分上达到对齐基准相同质量。在第十阶段 · 08 中涵盖。
- **GRPO**（DeepSeek 2024–2025）：PPO 以群相对基线代替评论家，奖励来自*验证器*（代码运行 / 数学答案匹配）而非人类训练的 RM。对推理模型占优势。在第九阶段 · 12 中涵盖。
- **过程奖励模型（PRM）：** 评分部分解（每个推理步骤），在 RLHF 和 GRPO 变体中用于推理。
- **宪法 AI / RLAIF：** 使用对齐的 LLM 来生成偏好而非人类。扩展偏好预算。

## 动手构建

本课使用表示为字符串的微型合成"提示"和"响应"。RM 是 token 包表示上的线性评分器。没有真正的 LLM——流水线的*形状*重要，而非规模。参见 `code/main.py`。

### 步骤 1：合成偏好数据

```python
PROMPTS = ["help me", "answer me", "explain this"]
GOOD_WORDS = {"clear", "specific", "kind", "thorough"}
BAD_WORDS = {"vague", "rude", "wrong", "short"}

def make_pair(rng):
    x = rng.choice(PROMPTS)
    y_good = rng.choice(list(GOOD_WORDS)) + " " + rng.choice(list(GOOD_WORDS))
    y_bad = rng.choice(list(BAD_WORDS)) + " " + rng.choice(list(BAD_WORDS))
    return (x, y_good, y_bad)
```

在真实 RLHF 中，这被人类标注者替换。形状——`(prompt, preferred_response, rejected_response)`——相同。

### 步骤 2：Bradley-Terry 奖励模型

线性分数：`R(x, y) = w · bag(y)`。训练最小化 BT 配对对数损失：

```python
def rm_train_step(w, x, y_pos, y_neg, lr):
    r_pos = dot(w, bag(y_pos))
    r_neg = dot(w, bag(y_neg))
    p = sigmoid(r_pos - r_neg)
    for tok, cnt in bag(y_pos).items():
        w[tok] += lr * (1 - p) * cnt
    for tok, cnt in bag(y_neg).items():
        w[tok] -= lr * (1 - p) * cnt
```

几百次更新后，`w` 为好词 token 分配正权重，为坏词分配负权重。

### 步骤 3：RM 之上的类 PPO 策略

我们的玩具策略从词汇表中产生单个 token。我们在 RM 下对该 token 评分，计算 `log π_θ(token | prompt)`，添加 KL 到参考的惩罚，并应用裁剪的 PPO 代理。

```python
def rlhf_step(theta, ref, w, prompt, rng, eps=0.2, beta=0.1, lr=0.05):
    logits_theta = policy_logits(theta, prompt)
    probs = softmax(logits_theta)
    token = sample(probs, rng)
    logits_ref = policy_logits(ref, prompt)
    probs_ref = softmax(logits_ref)
    reward = dot(w, bag([token])) - beta * kl(probs, probs_ref)
    # ppo 风格在 theta 上的更新，将 reward 视为回报
    ...
```

### 步骤 4：监控 KL

每次更新跟踪均值 `KL(π_θ || π_ref)`。如果它攀升超过 `~5-10`，策略已从 `π_SFT` 漂移远——降低 `β` 在上升或奖励黑客在开始。这是真实 RLHF 中的顶级诊断。

### 步骤 5：使用 TRL 的生产配方

一旦你理解了玩具流水线，这里是真实库用户编写的相同循环。Hugging Face 的 [TRL](https://huggingface.co/docs/trl) 是参考实现——`RewardTrainer` 用于阶段 2，`PPOTrainer`（内置到参考的 KL）用于阶段 3。

```python
# Stage 2: 来自配对偏好的奖励模型
from trl import RewardTrainer, RewardConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
rm = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", num_labels=1
)

# dataset rows: {"prompt", "chosen", "rejected"} — Bradley-Terry 格式
trainer = RewardTrainer(
    model=rm,
    tokenizer=tok,
    train_dataset=preference_data,
    args=RewardConfig(output_dir="./rm", num_train_epochs=1, learning_rate=1e-5),
)
trainer.train()
```

```python
# Stage 3: 针对 RM 的 PPO，带对 SFT 参考的 KL 惩罚
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

policy = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")
ref    = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")  # 冻结

ppo = PPOTrainer(
    config=PPOConfig(learning_rate=1.41e-5, batch_size=64, init_kl_coef=0.05,
                     target_kl=6.0, adap_kl_ctrl=True),
    model=policy, ref_model=ref, tokenizer=tok,
)

for batch in dataloader:
    responses = ppo.generate(batch["query_ids"], max_new_tokens=128)
    rewards   = rm(torch.cat([batch["query_ids"], responses], dim=-1)).logits[:, 0]
    stats     = ppo.step(batch["query_ids"], responses, rewards)
    # stats includes: mean_kl, clip_frac, value_loss — 三个 PPO 诊断
```

库为你做了三件事。`adap_kl_ctrl=True` 实现自适应 β 调度：如果观察到的 KL 超过 `target_kl`，β 加倍；如果低于一半，减半。参考模型按约定冻结——你绝对不能意外与 `policy` 共享参数。价值头生活在与策略相同的骨干上（`AutoModelForCausalLMWithValueHead` 附加一个标量 MLP 头），这就是为什么 TRL 分别报告 `policy/kl` 和 `value/loss`。

## 缺陷陷阱

- **过度优化 / 奖励黑客。** RM 不完美；`π_θ` 找到得分高但坏的对抗性补全。症状：奖励无限攀升而人类评估分数停滞或下降。修复：提早停止，提高 `β`，拓宽 RM 训练数据。
- **长度黑客。** 在有帮助的响应上训练的 RM 经常隐式奖励长度。策略学会填充响应。修正：长度归一化奖励，或带有长度感知 RM 的 RLAIF。
- **RM 太小。** RM 需要至少与策略一样大。小型 RM 无法忠实地评分策略的输出。
- **KL 调优。** β 太低 → 漂移和奖励黑客。β 太高 → 策略几乎不改变。标准技巧是以每步固定 KL 为目标的*自适应* β。
- **偏好数据噪声。** 约 30% 的人类标签是嘈杂或不明确的。通过在一致性过滤数据上训练 RM 校准，或在 BT 上使用温度。
- **离策略问题。** PPO 数据在第一个 epoch 后略离策略。像第 08 课监控裁剪比例。

## 使用它

2026 年的 RLHF 是分层的：

| 层 | 目标 | 方法 |
|-------|--------|--------|
| 遵循指令、有帮助性、无害性 | 对齐 | DPO（第十阶段 · 08）优于 RLHF-PPO。 |
| 推理正确性（数学、代码） | 能力 | 带验证器奖励的 GRPO（第九阶段 · 12）。 |
| 长视野多步任务 | 代理 | 在步骤上带有过程奖励模型的 PPO / GRPO。 |
| 安全 / 拒绝行为 | 安全 | 带单独安全 RM 的 RLHF-PPO，或宪法 AI。 |
| 推理时的 Best-of-N | 快速对齐 | 在解码时使用 RM；不需要策略训练。 |
| 奖励蒸馏 | 推理计算 | 在冻结 LM 上训练小型"奖励头"。 |

RLHF 在 2022-2024 年是*那个*方法。在 2026 年，生产对齐流水线是 DPO 优先，PPO 仅用于 RM 密集型或安全关键步骤。

## 交付成果

保存为 `outputs/skill-rlhf-architect.md`：

```markdown
---
name: rlhf-architect
description: 为语言模型设计 RLHF / DPO / GRPO 对齐流水线，包括 RM、KL 和数据策略。
version: 1.0.0
phase: 9
lesson: 9
tags: [rl, rlhf, alignment, llm]
---

给定基座 LM、目标行为（对齐 / 推理 / 拒绝 / 代理）和偏好或验证器预算，输出：

1. 阶段。SFT？RM？DPO？GRPO？附证明。
2. 偏好或验证器来源。人类、AI 反馈、基于规则、单元测试通过或奖励蒸馏。
3. KL 策略。固定 β、自适应 β 或 DPO（隐式 KL）。
4. 诊断。均值 KL、奖励稳定性、过度优化守卫（保留人类评估）。
5. 安全门。红队集、拒绝率、与有帮助性 RM 分开的安全 RM。

拒绝发布无 KL 监控的 RLHF-PPO。拒绝使用比目标策略小的 RM。拒绝仅长度的奖励。标记任何未保留盲测人类评估集为缺少过度优化保护的流水线。
```

## 练习

1. **简单。** 在 `code/main.py` 中的 500 合成偏好对上训练 Bradley-Terry 奖励模型。在保留的 100 对上测量配对准确率。应超过 90%。
2. **中等。** 以 `β ∈ {0.0, 0.1, 1.0}` 运行玩具 PPO-RLHF 循环。对每个，绘制 RM 分数 vs 随更新对参考的 KL。哪个运行奖励黑客？
3. **困难。** 在相同偏好数据上实现 DPO（闭式偏好似然损失）并与使用的计算量和达到的最终 RM 分数比较 RLHF-PPO 流水线。

## 关键术语

| 术语 | 人们怎么说 | 它实际上的含义 |
|------|-----------------|-----------------------|
| RLHF | "对齐 RL" | 三阶段 SFT + RM + PPO 流水线（Christiano 2017、Ouyang 2022）。 |
| 奖励模型（RM） | "评分网络" | 通过 Bradley-Terry 拟合到配对偏好的学习标量函数。 |
| Bradley-Terry | "配对 logistic 损失" | `P(y_+ ≻ y_-) = σ(R(y_+) - R(y_-))`；标准 RM 目标。 |
| KL 惩罚 | "保持在参考附近" | 奖励中的 `β · KL(π_θ || π_ref)`；抗奖励黑客的正则化器。 |
| 奖励黑客 | "Goodhart 定律" | 策略利用 RM 缺陷；症状：奖励上升，人类评估持平。 |
| RLAIF | "AI 标注的偏好" | RLHF 其中标签来自另一个 LM 而非人类。 |
| PRM | "过程奖励模型" | 评分部分推理步骤；在推理流水线中使用。 |
| 宪法 AI | "Anthropic 的方法" | 由显式规则引导的 AI 生成偏好。 |

## 延伸阅读

- [Christiano 等人 (2017). Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) — 开创 RLHF 的论文。
- [Ouyang 等人 (2022). InstructGPT — Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — ChatGPT 背后的配方。
- [Stiennon 等人 (2020). Learning to summarize with human feedback](https://arxiv.org/abs/2009.01325) — 更早的用于摘要的 RLHF。
- [Rafailov 等人 (2023). Direct Preference Optimization](https://arxiv.org/abs/2305.18290) — DPO；2026 年后 RLHF 的默认方案。
- [Bai 等人 (2022). Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — RLAIF 和自我批判循环。
- [Anthropic RLHF paper (Bai 等人 2022). Training a Helpful and Harmless Assistant](https://arxiv.org/abs/2204.05862) — HH 论文。
- [Hugging Face TRL 库](https://huggingface.co/docs/trl) — 生产 `RewardTrainer` 和 `PPOTrainer`。阅读训练器源代码了解自适应 KL 和价值头细节。
- [Hugging Face — Illustrating Reinforcement Learning from Human Feedback](https://huggingface.co/blog/rlhf) 作者 Lambert、Castricato、von Werra、Havrilla — 带有图表的三阶段流水线规范导览。
- [von Werra 等人 (2020). TRL: Transformer Reinforcement Learning](https://github.com/huggingface/trl) — 该库；`examples/` 有用于 Llama、Mistral 和 Qwen 的端到端 RLHF 脚本。
- [Sutton 和 Barto (2018). 第 17.4 章 — Designing Reward Signals](http://incompleteideas.net/book/RLbook2020.pdf) — 奖励假设视角；思考奖励黑客的必要前提。
