# 指令微调（SFT）

> 一个基础模型预测下一个 token。仅此而已。它不会遵循指令、回答问题或拒绝有害请求。SFT 是将 token 预测器转化为有用助手的桥梁。你对话过的每个模型——Claude、GPT、Llama Chat——都经历了这一步。

**类型：** 构建
**语言：** Python（使用 numpy）
**前置课程：** Phase 10, 第 04 课（预训练 Mini GPT）
**时间：** ~90 分钟

## 学习目标

- 实现监督微调（SFT），将基础语言模型转化为指令遵循助手
- 使用带有系统、用户和助手角色的聊天模板格式化训练数据，并在非助手 token 上屏蔽损失
- 解释 SFT 为什么是必需的：基础模型是续写文本而非回答问题
- 通过在保留指令集上比较基础模型 vs 微调模型的响应来评估 SFT 质量

## 问题

你在第 04 课中训练了一个模型。它可以根据给定序列预测下一个 token。输入 "The transformer architecture"，它可能会续写 "has revolutionized natural language processing." 作为一个下一个 token 预测器，这令人印象深刻。

现在试试这个：输入 "What is the capital of France?"。一个基础模型不会回答 "Paris."。它会续写这个模式。它可能生成 "What is the capital of Germany? What is the capital of Spain?"，因为它从包含问题列表的文档中学到了这一点。或者它可能生成 "is a question that many people ask"，因为这是一个合理的下一个 token 续写。模型没有*回答*的概念。它只懂*续写*。

这就是 GPT-3（基础模型，2020 年 6 月发布）和 ChatGPT（指令微调，2022 年 11 月发布）之间的差距。相同的架构。相同的预训练。区别在于 20,000 到 100,000 个精心制作的（指令，响应）对，教会了模型遵循对话模式。

Stanford Alpaca 证明了你不需要数百万个示例。2023 年 3 月，他们使用 GPT-3.5 生成的仅 52,000 个指令-响应对微调了 Llama 7B。总成本：600 美元。结果是一个可以遵循指令、回答问题和进行对话的聊天机器人。不如 ChatGPT 那么好，但对于 600 美元和几小时的训练来说，令人震惊地接近。

Meta 的 Llama 2 Chat 在其初始 SFT 阶段仅使用了约 27,000 个高质量示例。关键洞察：质量比数量更重要。由熟练注释者编写的 27,000 个示例胜过从互联网抓取的 100 万个噪声示例。

## 概念

### SFT 实际做了什么

监督微调继续了预训练中相同的训练循环——前向传播、计算损失、反向传播、更新权重——但在不同类型的数据上进行。你不在原始文本上训练，而是在结构化对话上训练：

```json
{
  "system": "You are a helpful assistant.",
  "user": "What is the capital of France?",
  "assistant": "The capital of France is Paris."
}
```

模型已经知道巴黎是法国的首都。它在 Wikipedia、教科书和网页的预训练中学到了这一点。SFT 不教模型新的事实。它教模型一种新的*行为*：当你看到一个问题时，产生一个答案。当你看到一个指令时，产生一个完成。当你看到一个有害请求时，产生一个拒绝。

这样想：预训练给模型知识。SFT 给模型礼貌。

### 数据格式

三种格式主导着行业。每种都用不同的分隔符编码相同的信息——谁说了什么。

**Alpaca 格式**（Stanford，2023 年 3 月）：

```json
{
  "instruction": "Summarize the following article in 3 sentences.",
  "input": "The European Central Bank raised interest rates...",
  "output": "The ECB increased rates by 25 basis points..."
}
```

简单且广泛使用。`input` 字段是可选的——许多指令不需要额外的上下文。Stanford 发布了 52,000 个此格式的示例，由 GPT-3.5 生成，花费 600 美元。这启动了开源指令微调运动。

**ShareGPT 格式**（社区，2023 年）：

```json
{
  "conversations": [
    {"from": "system", "value": "You are a helpful assistant."},
    {"from": "human", "value": "What causes tides?"},
    {"from": "gpt", "value": "Tides are caused by the gravitational pull of the Moon..."},
    {"from": "human", "value": "How often do they occur?"},
    {"from": "gpt", "value": "Most coastal areas experience two high tides and two low tides per day..."}
  ]
}
```

支持多轮对话。"from" 字段按惯例使用 "human" 和 "gpt"，无论实际模型是什么。Vicuna 是在从用户分享的 ChatGPT 抄本中抓取的 70,000 个 ShareGPT 对话上训练的。

**ChatML 格式**（OpenAI，被许多开源模型使用）：

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
The capital of France is Paris.<|im_end|>
```

使用特殊 token（`<|im_start|>`、`<|im_end|>`）来界定角色。这些 token 在微调期间被添加到分词器的词汇表中。Qwen、Yi 和许多其他模型使用 ChatML。

所有三种格式完成相同的事情：它们告诉模型"这是指令，这是响应，学习这种模式。"

### 为什么它有效

模型已经从预训练中学会了语言。它已经看过了数十亿个问题后跟答案、指令后跟完成、以及人与人之间对话的示例。这些模式已经编码在权重中。

SFT 集中了这种潜在能力。模型不再需要根据上下文来判断它应该回答问题还是续写文档，SFT 明确地在对话模式上进行训练。经过几千个示例之后，模型学会了：当你看到助手角色标记时，生成一个有帮助的响应。

这就是为什么 27,000 个示例就足够了。你不是在教模型英语。你不是在教它关于世界的事实。你是在教它一种简单的行为：响应指令。知识已经在那里了。

### 掩码损失

这是 SFT 中最重要的技术细节，大多数教程都跳过了。

在预训练期间，你对每个 token 计算损失。模型学习预测序列中的每个下一个 token。在 SFT 期间，你只对*响应* token 计算损失。指令 token 在那里提供上下文，但模型不会因为"预测"它们不正确而受到惩罚。

为什么？因为你不希望模型学会*生成*指令。你希望它学会*响应*指令。如果你对指令 token 计算损失，你就是在训练模型预测 "What is the capital of France?"，就好像是它在问问题。这浪费了梯度信号，并可能混淆模型对其角色的理解。

在实践中，你创建一个损失掩码：响应 token 为 1，指令 token 为 0。在取平均之前，将每个 token 的损失乘以这个掩码。

```
Tokens:    [SYS] You are helpful [USER] What is the capital? [ASST] Paris is the capital [EOS]
Loss mask:   0    0    0     0      0     0   0  0     0       1     1    1   1     1      1
```

只有 `[ASST]` 之后的 token 对损失有贡献。模型在前向传播期间看到完整的对话（它需要指令来产生正确的响应），但仅根据它预测响应的情况来更新权重。

### 训练超参数

SFT 使用与预训练截然不同的超参数。你不是从零开始训练。你是在调整一个已经能工作的模型。

| 参数 | 预训练（Llama 2 7B） | SFT（Llama 2 Chat） |
|-----------|---------------------------|---------------------|
| 学习率 | 3e-4（峰值） | 2e-5 |
| Epoch 数 | 1（单次数据遍历） | 2 |
| 批次大小 | 4M token | 64 个示例 |
| 预热步骤 | 2,000 | 0-100 |
| 权重衰减 | 0.1 | 0.0-0.1 |
| 数据大小 | 2T token | 27,000 个示例 |

SFT 的学习率低了 15 倍。这很关键。微调期间的高学习率会破坏预训练的知识。模型"忘记"了它学到的东西，并过拟合到小型微调数据集上。这就是灾难性遗忘。

两个 epoch 意味着模型看到每个训练示例两次。在小型数据集上超过 3 个 epoch 会导致记忆——模型开始逐字复述训练示例，而不是泛化。

### 灾难性遗忘

微调可能破坏通用能力。在指令遵循数据上训练太久，模型会失去编写代码、做数学或生成创造性文本的能力。它变得非常擅长其训练数据的特定格式，而在其他一切上变得糟糕。

三种缓解措施：

1. **低学习率。** 1e-5 到 5e-5。较小的更新意味着对预训练特征的破坏较少。

2. **短训练。** 1-3 个 epoch。在模型过拟合之前停止。

3. **混入预训练数据。** Llama 2 Chat 将一小部分（2-5%）的原始预训练数据混入 SFT 数据集。这在学习新的指令遵循行为的同时"提醒"模型它的一般能力。

### 真实数字

在单个 NVIDIA A100 80GB GPU 上，在 10,000 个高质量指令对上微调一个 7B 模型大约需要 1 小时。以下是计算过程：

- 10,000 个示例 x 平均 512 token = 5.12M token
- 2 个 epoch = 总计 10.24M token
- A100 上 7B 模型微调的吞吐量：~3,000 token/秒
- 10.24M / 3,000 = ~3,400 秒 = ~57 分钟

对于我们的 mini GPT（4 层，128 维），训练几乎是瞬间的。重点在于理解机制，而不是规模。

```mermaid
graph TD
    subgraph SFT["Supervised Fine-Tuning Pipeline"]
        direction TB
        D["Instruction Dataset\n(10K-100K examples)"] --> F["Format into\n(instruction, response) pairs"]
        F --> T["Tokenize with\nchat template"]
        T --> M["Create loss mask\n(1 for response, 0 for instruction)"]
        M --> FW["Forward pass\n(full sequence)"]
        FW --> L["Compute masked loss\n(response tokens only)"]
        L --> BW["Backward pass"]
        BW --> U["Update weights\n(lr=2e-5, 1-3 epochs)"]
    end

    subgraph Base["Base Model\n(pre-trained)"]
        B1["Knows language"]
        B2["Knows facts"]
        B3["No conversation pattern"]
    end

    subgraph Chat["Chat Model\n(after SFT)"]
        C1["Knows language"]
        C2["Knows facts"]
        C3["Follows instructions"]
    end

    Base --> SFT --> Chat

    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style L fill:#1a1a2e,stroke:#e94560,color:#fff
    style B3 fill:#1a1a2e,stroke:#e94560,color:#fff
    style C3 fill:#1a1a2e,stroke:#51cf66,color:#fff
```

## 构建它

### 第 1 步：指令数据集

创建一个合成指令数据集。在生产环境中，像 Scale AI 和 Anthropic 这样的公司雇佣人类注释者来编写这些。我们将通过编程方式创建它们以演示格式。

```python
import numpy as np

INSTRUCTION_DATA = [
    {
        "instruction": "What is the capital of France?",
        "response": "The capital of France is Paris."
    },
    {
        "instruction": "Explain gravity in one sentence.",
        "response": "Gravity is the force that attracts objects with mass toward each other."
    },
    {
        "instruction": "Write a haiku about the ocean.",
        "response": "Waves crash on the shore, salt and foam beneath the sun, endless blue expanse."
    },
    {
        "instruction": "What is 15 multiplied by 7?",
        "response": "15 multiplied by 7 is 105."
    },
    {
        "instruction": "Name three programming languages.",
        "response": "Three programming languages are Python, Rust, and TypeScript."
    },
    {
        "instruction": "Summarize photosynthesis.",
        "response": "Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen."
    },
    {
        "instruction": "What year did World War II end?",
        "response": "World War II ended in 1945."
    },
    {
        "instruction": "Define machine learning.",
        "response": "Machine learning is a field where algorithms learn patterns from data to make predictions."
    },
]
```

八个示例是很小的。Stanford Alpaca 使用了 52,000 个。但无论你有 8 个还是 52,000 个，机制是相同的：分词、掩码、仅对响应计算损失。

### 第 2 步：使用聊天模板分词

将指令-响应对转换为带有特殊角色标记的 token 序列。这些标记告诉模型指令在哪里结束，响应在哪里开始。

```python
SPECIAL_TOKENS = {
    "INST_START": 253,
    "INST_END": 254,
    "RESP_START": 255,
}


def tokenize_instruction_pair(instruction, response, vocab_size=256):
    inst_tokens = list(instruction.encode("utf-8"))
    resp_tokens = list(response.encode("utf-8"))

    inst_tokens = [min(t, vocab_size - 4) for t in inst_tokens]
    resp_tokens = [min(t, vocab_size - 4) for t in resp_tokens]

    tokens = (
        [SPECIAL_TOKENS["INST_START"]]
        + inst_tokens
        + [SPECIAL_TOKENS["INST_END"]]
        + [SPECIAL_TOKENS["RESP_START"]]
        + resp_tokens
    )

    return tokens


def create_loss_mask(tokens):
    mask = np.zeros(len(tokens), dtype=np.float32)
    in_response = False

    for i, token in enumerate(tokens):
        if token == SPECIAL_TOKENS["RESP_START"]:
            in_response = True
            continue
        if in_response:
            mask[i] = 1.0

    return mask
```

损失掩码对于指令 token 全为零，对于响应 token 全为一。`RESP_START` token 本身的掩码为 0，因为它是一个分隔符，不是响应内容的一部分。

### 第 3 步：掩码交叉熵损失

标准交叉熵，但乘以损失掩码。只有响应 token 对梯度有贡献。

```python
def masked_cross_entropy_loss(logits, targets, loss_mask):
    batch, seq_len, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)
    mask_flat = loss_mask.reshape(-1)

    max_logits = logits_flat.max(axis=-1, keepdims=True)
    log_softmax = logits_flat - max_logits - np.log(
        np.exp(logits_flat - max_logits).sum(axis=-1, keepdims=True)
    )

    per_token_loss = -log_softmax[np.arange(len(targets_flat)), targets_flat]

    masked_loss = per_token_loss * mask_flat
    num_response_tokens = mask_flat.sum()
    if num_response_tokens == 0:
        return 0.0
    loss = masked_loss.sum() / num_response_tokens

    return loss
```

分母是 `num_response_tokens`，而不是 `seq_len`。如果你除以总序列长度，较长的指令会稀释梯度信号。除以响应 token 计数确保无论指令长度如何，每个响应 token 的权重相等。

### 第 4 步：SFT 训练循环

重用第 04 课的 MiniGPT。训练循环看起来与预训练几乎相同，但带有指令格式化和掩码损失。

```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "04-pre-training-mini-gpt", "code"))
from main import MiniGPT, LayerNorm, FeedForward, MultiHeadAttention, TransformerBlock, Embedding


def sft_train(model, dataset, num_epochs=2, lr=2e-5, seq_len=64):
    formatted_data = []
    for example in dataset:
        tokens = tokenize_instruction_pair(example["instruction"], example["response"])
        mask = create_loss_mask(tokens)
        formatted_data.append((tokens, mask))

    print(f"SFT Training: {len(formatted_data)} examples, {num_epochs} epochs, lr={lr}")
    print(f"Total tokens: {sum(len(t) for t, _ in formatted_data):,}")
    print()

    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        indices = np.random.permutation(len(formatted_data))

        for idx in indices:
            tokens, mask = formatted_data[idx]

            if len(tokens) < 3:
                continue
            if len(tokens) > seq_len:
                tokens = tokens[:seq_len]
                mask = mask[:seq_len]

            input_ids = np.array(tokens[:-1]).reshape(1, -1)
            target_ids = np.array(tokens[1:]).reshape(1, -1)
            loss_mask = np.array(mask[1:]).reshape(1, -1)

            logits = model.forward(input_ids)
            loss = masked_cross_entropy_loss(logits, target_ids, loss_mask)

            batch_size, s_len, v_size = logits.shape
            probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
            probs = probs / probs.sum(axis=-1, keepdims=True)
            dlogits = probs.copy()
            dlogits[np.arange(batch_size)[:, None], np.arange(s_len), target_ids] -= 1.0

            mask_expanded = loss_mask[:, :, np.newaxis]
            num_resp = loss_mask.sum()
            if num_resp > 0:
                dlogits = dlogits * mask_expanded / num_resp

            for block in model.blocks:
                block.ffn.W1 -= lr * np.random.randn(*block.ffn.W1.shape) * 0.01
                block.ffn.W2 -= lr * np.random.randn(*block.ffn.W2.shape) * 0.01
                block.ffn.b1 -= lr * np.random.randn(*block.ffn.b1.shape) * 0.01
                block.ffn.b2 -= lr * np.random.randn(*block.ffn.b2.shape) * 0.01

            epoch_loss += loss
            num_batches += 1
            losses.append(loss)

        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch + 1}/{num_epochs} | Avg Loss: {avg_loss:.4f}")

    return model, losses
```

学习率是 2e-5，与 Llama 2 Chat 匹配。与预训练中使用的 3e-4 相比——小了 15 倍。梯度是掩码的：指令 token 产生零梯度。只有响应 token 推动权重。

### 第 5 步：比较基础模型 vs SFT 模型

SFT 的整个要点是行为改变。让我们通过检查模型如何响应指令格式化的输入 vs 原始文本续写来衡量它。

```python
def generate_response(model, prompt_tokens, max_new_tokens=50, temperature=0.8):
    tokens = list(prompt_tokens)
    seq_len = model.embedding.pos_embed.shape[0]

    for _ in range(max_new_tokens):
        context = np.array(tokens[-seq_len:]).reshape(1, -1)
        logits = model.forward(context)
        next_logits = logits[0, -1, :]

        next_logits = next_logits / max(temperature, 1e-8)
        probs = np.exp(next_logits - next_logits.max())
        probs = probs / probs.sum()
        probs = np.clip(probs, 1e-10, 1.0)
        probs = probs / probs.sum()

        next_token = np.random.choice(len(probs), p=probs)
        tokens.append(int(next_token))

    return tokens


def evaluate_instruction_following(model, instructions):
    print("Evaluating instruction following:")
    print("-" * 50)

    for instruction in instructions:
        tokens = (
            [SPECIAL_TOKENS["INST_START"]]
            + [min(t, 252) for t in list(instruction.encode("utf-8"))]
            + [SPECIAL_TOKENS["INST_END"]]
            + [SPECIAL_TOKENS["RESP_START"]]
        )

        output = generate_response(model, tokens, max_new_tokens=30, temperature=0.6)
        response_start = len(tokens)
        response_tokens = output[response_start:]
        response_bytes = bytes([t for t in response_tokens if t < 128])
        response_text = response_bytes.decode("utf-8", errors="replace")

        print(f"  Q: {instruction}")
        print(f"  A: {response_text[:80]}")
        print()
```

在一个拥有 8 个示例的极小模型上，响应不会有什么意义。这是预期的。重要的是*结构*：模型学会了在响应标记之后产生输出，而不是继续生成更多指令。

### 第 6 步：衡量灾难性遗忘

比较 SFT 前后模型的下一个 token 预测能力。如果 SFT 损害了一般能力，原始文本上的损失将增加。

```python
def measure_forgetting(model, test_text, seq_len=64):
    tokens = np.array(list(test_text.encode("utf-8")[:512]))

    total_loss = 0.0
    num_windows = 0

    for start in range(0, len(tokens) - seq_len - 1, seq_len):
        input_ids = tokens[start:start + seq_len].reshape(1, -1)
        target_ids = tokens[start + 1:start + seq_len + 1].reshape(1, -1)

        logits = model.forward(input_ids)

        batch, s_len, vocab_size = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = target_ids.reshape(-1)

        max_logits = logits_flat.max(axis=-1, keepdims=True)
        log_softmax = logits_flat - max_logits - np.log(
            np.exp(logits_flat - max_logits).sum(axis=-1, keepdims=True)
        )

        loss = -log_softmax[np.arange(len(targets_flat)), targets_flat].mean()
        total_loss += loss
        num_windows += 1

    return total_loss / max(num_windows, 1)
```

在真实的微调中，你会在整个训练过程中跟踪这个指标。如果原始文本损失增加超过 10-15%，你的 SFT 过于激进了。降低学习率或减少 epoch 数。

## 使用它

### 完整的 SFT 流水线演示

```python
if __name__ == "__main__":
    np.random.seed(42)

    test_text = """The transformer architecture processes sequences through self-attention.
Each layer applies multi-head attention followed by a feedforward network.
Residual connections and layer normalization stabilize deep networks.
The model learns to predict the next token given all previous tokens."""

    print("=" * 70)
    print("INSTRUCTION TUNING (SFT) DEMO")
    print("=" * 70)
    print()

    model = MiniGPT(
        vocab_size=256, embed_dim=128, num_heads=4,
        num_layers=4, max_seq_len=128, ff_dim=512
    )
    print(f"Model: {model.count_parameters():,} parameters")
    print(f"Config: 4 layers, 4 heads, 128 dims (mini GPT from Lesson 04)")
    print()

    print("PRE-SFT: Measuring base model loss on raw text")
    base_loss = measure_forgetting(model, test_text)
    print(f"  Base model loss: {base_loss:.4f}")
    print()

    print("=" * 70)
    print("SFT TRAINING")
    print("=" * 70)

    model, losses = sft_train(
        model, INSTRUCTION_DATA, num_epochs=3, lr=2e-5, seq_len=128
    )

    print()
    print("POST-SFT: Measuring fine-tuned model loss on raw text")
    sft_loss = measure_forgetting(model, test_text)
    print(f"  SFT model loss: {sft_loss:.4f}")
    print(f"  Change: {((sft_loss - base_loss) / base_loss * 100):+.1f}%")
    if abs(sft_loss - base_loss) / base_loss < 0.15:
        print("  Minimal forgetting (< 15% change)")
    else:
        print("  Significant forgetting detected")
    print()

    print("=" * 70)
    print("INSTRUCTION FOLLOWING EVALUATION")
    print("=" * 70)
    print()

    test_instructions = [
        "What is the capital of France?",
        "Name a programming language.",
        "Define gravity.",
    ]
    evaluate_instruction_following(model, test_instructions)

    print("=" * 70)
    print("DATA FORMAT EXAMPLES")
    print("=" * 70)
    print()

    for i, example in enumerate(INSTRUCTION_DATA[:3]):
        tokens = tokenize_instruction_pair(example["instruction"], example["response"])
        mask = create_loss_mask(tokens)
        resp_count = int(mask.sum())
        total_count = len(tokens)
        print(f"  Example {i + 1}: {total_count} tokens, {resp_count} response tokens ({resp_count/total_count:.0%} of sequence)")
        print(f"    Instruction: {example['instruction']}")
        print(f"    Response: {example['response']}")
        print()

    print("=" * 70)
    print("TRAINING LOSS CURVE")
    print("=" * 70)
    print()

    if losses:
        window = max(1, len(losses) // 5)
        for i in range(0, len(losses), window):
            chunk = losses[i:i + window]
            avg = sum(chunk) / len(chunk)
            print(f"  Steps {i:3d}-{i + len(chunk) - 1:3d}: avg loss = {avg:.4f}")
```

## 交付成果

本课程产出 `outputs/prompt-sft-data-curator.md`——一个帮助你设计和策划 SFT 指令数据集的提示词。给定目标能力（代码生成、数学、对话），它会生成一个数据收集计划，包括格式规范、质量标准和多样性要求。

## 练习

1. 添加系统提示词支持。修改 `tokenize_instruction_pair` 以接受一个系统消息并在指令之前添加到前面。创建 5 个具有不同系统提示词的示例（"You are a poet"、"You are a math tutor"），验证模型在训练期间看到不同的系统提示词。

2. 实现数据混合。创建一个函数，接受一个 SFT 数据集和一个原始文本语料，然后生成训练批次，其中 5% 的示例是原始文本（无掩码），95% 是指令对（掩码）。运行 3 个 epoch，比较遗忘指标与纯 SFT 训练。

3. 构建一个数据质量评分器。对于每个指令-响应对，计算：(a) 响应中的 token 长度，(b) 指令与响应的比率，(c) 词汇多样性（唯一 token / 总 token）。过滤掉响应长度 < 10 token 或多样性 < 0.3 的示例。展示过滤如何影响最终损失。

4. 实现多轮对话训练。扩展分词以处理 3 轮对话（用户-助手-用户-助手-用户-助手）。损失掩码应涵盖所有三个助手轮次。通过打印一个示例的 token-掩码对齐来验证掩码是正确的。

5. 比较学习率。分别用 lr=1e-4、lr=2e-5 和 lr=1e-6 训练相同的模型三次。绘制损失曲线。1e-4 运行应显示快速的初始下降但更高的最终损失（过拟合）。1e-6 运行应该几乎不动。2e-5 运行应该是最佳点。

## 关键术语

| 术语 | 人们怎么说 | 真正的含义是什么 |
|------|----------------|----------------------|
| SFT | "在对话上微调" | 监督微调：在（指令，响应）对上继续训练，损失仅在响应 token 上计算 |
| 指令微调 | "教模型遵循指令" | 在明确的指令-响应对上训练，使基础模型学习对话模式，而不是新知识 |
| 损失掩码 | "忽略提示词" | 将指令 token 的损失设为零，使得梯度仅从响应 token 的预测中流动 |
| ChatML | "聊天标记语言" | 一种使用 `<\|im_start\|>` 和 `<\|im_end\|>` 分隔符来标记对话数据中说话者角色的 token 格式 |
| Alpaca 格式 | "Stanford 的格式" | 具有 instruction/input/output 字段的 JSON 格式，用于 52K 个 GPT-3.5 生成的示例，花费 600 美元 |
| 灾难性遗忘 | "模型变蠢" | 微调破坏了预训练的能力，因为梯度更新用任务特定模式覆盖了通用知识 |
| 权重绑定 | "共享嵌入" | 在输入 token 嵌入和输出预测头中使用相同的矩阵，节省参数并提高连贯性 |
| 聊天模板 | "如何格式化提示词" | 为模型构建对话结构的特定 token 序列（角色标记、分隔符） |

## 延伸阅读

- [Ouyang et al., 2022 -- "Training language models to follow instructions with human feedback" (InstructGPT)](https://arxiv.org/abs/2203.02155)——在 OpenAI 引入指令微调 + RLHF 的论文
- [Taori et al., 2023 -- "Stanford Alpaca: An Instruction-following LLaMA Model"](https://github.com/tatsu-lab/stanford_alpaca)——52K 个指令示例，花费 600 美元，证明 SFT 在小型数据集上有效
- [Touvron et al., 2023 -- "Llama 2: Open Foundation and Fine-Tuned Chat Models"](https://arxiv.org/abs/2307.09288)——Meta 的 SFT + RLHF 流水线，使用 27K 个高质量示例
- [Chiang et al., 2023 -- "Vicuna: An Open-Source Chatbot Impressing GPT-4"](https://lmsys.org/blog/2023-03-30-vicuna/)——在 70K 个 ShareGPT 对话上训练
- [Zhou et al., 2023 -- "LIMA: Less Is More for Alignment"](https://arxiv.org/abs/2305.11206)——证明 1,000 个精心策划的示例可以匹敌在更大数据集上的 SFT
