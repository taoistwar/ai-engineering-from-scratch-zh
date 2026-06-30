# 子词 Tokenization — BPE、WordPiece、Unigram、SentencePiece

> 词级 tokenizer 在未见过的词上出错。字符级 tokenizer 使序列长度爆炸。子词 tokenizer 取得平衡。每个现代 LLM 都发布在一个子词 tokenizer 上。

**类型：** 学习
**语言：** Python
**先修要求：** 第五阶段 · 01（文本处理），第五阶段 · 04（GloVe / FastText / 子词）
**预计时间：** 约60分钟

## 问题

你的词汇表有 50,000 个词。用户输入"untokenizable"。你的 tokenizer 返回`[UNK]`。模型现在对该词没有任何信号。更糟糕的是：你的语料库中第 90 百分位的文档有 40 个罕见词，这意味着每文档丢失 40 位信息。

子词 tokenization 解决了这个问题。常见词保持为单个 token。罕见词分解为有意义的片段：`untokenizable` → `un`、`token`、`izable`。训练数据覆盖一切，因为任何字符串最终都是一系列字节。

2026 年每个前沿 LLM 都发布在三种算法之一（BPE、Unigram、WordPiece）上，封装在三个库之一（tiktoken、SentencePiece、HF Tokenizers）中。不选择一个你就无法发布语言模型。

## 概念

![BPE vs Unigram vs WordPiece, character-by-character](../assets/subword-tokenization.svg)

**BPE（字节对编码）。** 从字符级词汇表开始。统计每个相邻对。将最频繁的对合并为一个新的 token。重复直到达到目标词汇表大小。主导算法：GPT-2/3/4、Llama、Gemma、Qwen2、Mistral。

**字节级 BPE。** 相同的算法但在原始字节（256 个基础 token）上而不是 Unicode 字符上。保证零`[UNK]` token——任何字节序列都可以编码。GPT-2 使用 50,257 个 token（256 字节 + 50,000 次合并 + 1 个特殊 token）。

**Unigram。** 从一个巨大的词汇表开始。为每个 token 分配一个单元概率。迭代剪枝那些移除后最小化增加语料库对数似然的 token。在推理时是概率性的：可以采样 tokenization（通过子词正则化用于数据增强）。被 T5、mBART、ALBERT、XLNet、Gemma 使用。

**WordPiece。** 合并最大化训练语料库似然而不是原始频率的对。被 BERT、DistilBERT、ELECTRA 使用。

**SentencePiece vs tiktoken。** SentencePiece 是在原始 Unicode 文本上直接*训练*词汇表（BPE 或 Unigram）的库，将空格编码为`▁`。tiktoken 是 OpenAI 针对预构建词汇表的快速*编码器*；它不进行训练。

经验法则：

- **训练新词汇表：** SentencePiece（多语言，无需预分词）或 HF Tokenizers。
- **针对 GPT 词汇表进行快速推理：** tiktoken（cl100k_base、o200k_base）。
- **两者：** HF Tokenizers——一个库，训练 + 服务。

```figure
bpe-merge
```

## 构建它

### 步骤 1：从零实现 BPE

参见`code/main.py`。循环：

```python
def train_bpe(corpus, num_merges):
    vocab = {tuple(word) + ("</w>",): count for word, count in corpus.items()}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for symbols, freq in vocab.items():
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]
        merges.append(best)
        vocab = apply_merge(vocab, best)
    return merges
```

算法编码的三个事实。`</w>`标记词尾，使"low"（后缀）和"lower"（前缀）保持不同。频率加权使高频对早期胜出。合并列表是有序的——推理时按训练顺序应用合并。

### 步骤 2：用学习到的合并进行编码

```python
def encode_bpe(word, merges):
    symbols = list(word) + ["</w>"]
    for a, b in merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols = symbols[:i] + [a + b] + symbols[i + 2:]
            else:
                i += 1
    return symbols
```

朴素的 O(n·|merges|)。生产实现（tiktoken、HF Tokenizers）使用带优先队列的合并排名查找，以近线性时间运行。

### 步骤 3：实践中的 SentencePiece

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="my_tokenizer",
    vocab_size=8000,
    model_type="bpe",          # 或 "unigram"
    character_coverage=0.9995, # CJK 较低（例如英语 0.9995，日语 0.995）
    normalization_rule_name="nmt_nfkc",
)

sp = spm.SentencePieceProcessor(model_file="my_tokenizer.model")
print(sp.encode("untokenizable", out_type=str))
# ['▁un', 'token', 'izable']
```

注意：无需预分词，空格编码为`▁`，`character_coverage`控制罕见字符被保留还是被映射为`<unk>`的激进程度。

### 步骤 4：tiktoken 用于 OpenAI 兼容词汇表

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
print(enc.encode("untokenizable"))        # [127340, 101028]
print(len(enc.encode("Hello, world!")))   # 4
```

仅编码。快速（Rust 后端）。与 GPT-4/5 tokenization 精确匹配用于字节计数、成本估算、上下文窗口预算。

## 2026 年仍未被捕获就发布的陷阱

- **Tokenizer 漂移。** 在词汇表 A 上训练，部署时使用词汇表 B。Token ID 不同；模型输出垃圾。在 CI 中检查`tokenizer.json`哈希。
- **空格歧义。** BPE "hello"与" hello"产生不同的 token。始终显式指定`add_special_tokens`和`add_prefix_space`。
- **多语言训练不足。** 英语主导的语料库产生的词汇表将非拉丁文字分割为 5-10 倍多的 token。相同提示在 GPT-3.5 上的日语/阿拉伯语成本高 5-10 倍。o200k_base 部分修复了这一点。
- **表情符号分割。** 单个表情符号可能需要 5 个 token。在预算上下文时检查表情符号处理。

## 使用它

2026 年技术栈：

| 场景 | 选择 |
|-----------|------|
| 从零训练单语言模型 | HF Tokenizers（BPE） |
| 训练多语言模型 | SentencePiece（Unigram，`character_coverage=0.9995`） |
| 服务 OpenAI 兼容 API | tiktoken（GPT-4+ 用`o200k_base`） |
| 领域特定词汇表（代码、数学、蛋白质） | 在领域语料库上训练自定义 BPE，与基础词汇表合并 |
| 边缘推理、小模型 | Unigram（较小的词汇表效果更好） |

词汇表大小是一个尺度决策，不是一个常量。粗略启发：<1B 参数用 32k，1-10B 用 50-100k，多语言/前沿用 200k+。

## 交付它

保存为 `outputs/skill-bpe-vs-wordpiece.md`：

```markdown
---
name: tokenizer-picker
description: 为给定语料库和部署目标选择 tokenizer 算法、词汇表大小、库。
version: 1.0.0
phase: 5
lesson: 19
tags: [nlp, tokenization]
---

给定一个语料库（大小、语言、领域）和部署目标（从零训练 / 微调 / API 兼容推理），输出：

1. 算法。BPE、Unigram 或 WordPiece。一句理由。
2. 库。SentencePiece、HF Tokenizers 或 tiktoken。理由。
3. 词汇表大小。四舍五入到最近的 1k。与模型大小和语言覆盖相关联的理由。
4. 覆盖设置。`character_coverage`、`byte_fallback`、特殊 token 列表。
5. 验证计划。保留集上的每词平均 token 数、OOV 率、压缩比、往返解码等价性。

拒绝在包含罕见文字内容的语料库上训练 character-coverage <0.995 的 tokenizer。拒绝发布没有 CI 中冻结的`tokenizer.json`哈希检查的词汇表。标记任何低于 16k 词汇表的单语言 tokenizer 为可能不足规格。
```

## 练习

1. **简单。** 在`code/main.py`的小型语料库上训练 500 次合并的 BPE。编码三个保留词。有多少产生恰好 1 个 token 对比 >1 个 token？
2. **中等。** 在 100 个英语 Wikipedia 句子上比较`cl100k_base`、`o200k_base`和以 vocab=32k 训练的 SentencePiece BPE 之间的 token 数。报告每种方案的压缩比。
3. **困难。** 用 BPE、Unigram 和 WordPiece 训练相同的语料库。测量每种方案用于小型情感分类器时的下游准确率。该选择是否使 F1 移动超过 1 分？

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| BPE | 字节对编码 | 贪心合并最频繁字符对直到达到目标词汇表大小。 |
| 字节级 BPE | 永远无未知 token | 在原始 256 字节上的 BPE；GPT-2 / Llama 使用此方案。 |
| Unigram | 概率性 tokenizer | 使用对数似然从大型候选集中剪枝；被 T5、Gemma 使用。 |
| SentencePiece | 处理空格的那个 | 在原始文本上训练 BPE/Unigram 的库；空格编码为`▁`。 |
| tiktoken | 快速的那个 | OpenAI 的 Rust 支持 BPE 编码器，用于预构建词汇表。无训练。 |
| 合并列表 | 魔法数字 | 有序列表`(a, b) → ab`合并；推理时按顺序应用。 |
| 字符覆盖率 | 多罕见才算太罕见？ | tokenizer 必须覆盖的训练语料库中字符的比例；通常约 0.9995。 |

## 扩展阅读

- [Sennrich, Haddow, Birch (2015). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — BPE 论文。
- [Kudo (2018). Subword Regularization with Unigram Language Model](https://arxiv.org/abs/1804.10959) — Unigram 论文。
- [Kudo, Richardson (2018). SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226) — 库的论文。
- [Hugging Face — Summary of the tokenizers](https://huggingface.co/docs/transformers/tokenizer_summary) — 简洁参考。
- [OpenAI tiktoken repo](https://github.com/openai/tiktoken) — 教程 + 编码列表。
