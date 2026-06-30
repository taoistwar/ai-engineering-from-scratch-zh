# 从零开始构建分词器

> 第 01 课给了你一个玩具。本课给你一件武器。

**类型：** 构建
**语言：** Python
**前置课程：** Phase 10, 第 01 课（分词器：BPE、WordPiece、SentencePiece）
**时间：** ~90 分钟

## 学习目标

- 构建一个生产级的 BPE 分词器，处理 Unicode、空白字符标准化和特殊 token
- 实现字节级回退机制，使分词器可以编码任何输入（包括 emoji、CJK 和代码），不会出现未知 token
- 添加预分词正则模式，在应用 BPE 合并之前按词边界分割文本
- 在语料上训练自定义分词器，并在多语言文本上评估其压缩比与 tiktoken 的差异

## 问题

你在第 01 课中构建的 BPE 分词器可以处理英语文本。现在向它输入日语。或者 emoji。或者混合了制表符和空格的 Python 代码。

它会崩溃。

不是因为 BPE 错了——而是因为实现不完整。一个生产级分词器能处理任何编码中的原始字节，在分割之前标准化 Unicode，管理永不合并的特殊 token，将预分词与子词分割串联执行，并且速度足够快，不会成为处理 15 万亿 token 的训练流程的瓶颈。

GPT-2 的分词器有 50,257 个 token。Llama 3 有 128,256 个。GPT-4 大约有 100,000 个。这些不是玩具数字。这些词汇表背后的合并表是在数百 GB 的文本上训练的，而周围的机制——标准化、预分词、特殊 token 注入、聊天模板格式化——正是区分一个能处理 "hello world" 的分词器和一个能处理整个互联网的分词器的关键。

你将构建这些机制。

## 概念

### 完整流水线

一个生产级分词器不是一个算法，而是由五个阶段组成的流水线，每个阶段解决不同的问题。

```mermaid
graph LR
    A[Raw Text] --> B[Normalize]
    B --> C[Pre-Tokenize]
    C --> D[BPE Merge]
    D --> E[Special Tokens]
    E --> F[Token IDs]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#e94560,color:#fff
    style C fill:#1a1a2e,stroke:#e94560,color:#fff
    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
```

每个阶段都有特定的任务：

| 阶段 | 它做什么 | 为什么重要 |
|-------|-------------|----------------|
| 标准化 | NFKC Unicode 标准化，可选小写，可选去除重音 | "fi" 连字（U+FB01）变成 "fi"（两个字符）。没有这一步，同一个词会得到不同的 token。 |
| 预分词 | 在 BPE 之前将文本分割成块 | 防止 BPE 跨词边界合并。"the cat" 永远不应该产生 "e c" 这样的 token。 |
| BPE 合并 | 将学习到的合并规则应用于字节序列 | 核心压缩。将原始字节转换为子词 token。 |
| 特殊 Token | 注入 [BOS]、[EOS]、[PAD]、聊天模板标记 | 这些 token 有固定的 ID。它们绝不参与 BPE 合并。模型需要它们来理解结构。 |
| ID 映射 | 将 token 字符串转换为整数 ID | 模型看到的是整数，而不是字符串。 |

### 字节级 BPE

第 01 课的分词器在 UTF-8 字节上操作。那是对的。但我们跳过了重要的东西：当这些字节不是有效的 UTF-8 时会发生什么？

字节级 BPE 通过将每一个可能的字节值（0-255）都视为有效 token 来解决这个问题。你的基础词汇表恰好有 256 个条目。任何文件——文本、二进制、损坏的——都可以被分词，而不会产生未知 token。

GPT-2 增加了一个技巧：将每个字节映射到一个可打印的 Unicode 字符，使词汇表保持人类可读。字节 0x20（空格）在他们的映射中变成了字符 "G"。这纯粹是表面上的改动。算法不在乎。

真正的威力：字节级 BPE 处理地球上的每一种语言。中文字符每个是 3 个 UTF-8 字节。日语可以是 3-4 个字节。阿拉伯语、天城文、emoji——都只是字节序列。BPE 算法在这些字节序列中发现模式的方式，与它在英语 ASCII 字节中发现模式的方式完全相同。

### 预分词

在 BPE 接触你的文本之前，你需要将其分割成块。这防止了合并算法创建跨越词边界的 token。

GPT-2 使用正则表达式模式来分割文本：

```
'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

这个模式在缩写词（"don't" 变成 "don" + "'t"）、带有可选前导空格的单词、数字、标点符号和空白字符上进行分割。前导空格被保留并附加到单词上——所以 "the cat" 变成 [" the", " cat"]，而不是 ["the", " ", "cat"]。

Llama 使用 SentencePiece，它完全跳过正则表达式。它将原始字节流视为一个长序列，让 BPE 算法自行找出边界。这更简单，但给了 BPE 更多创建跨词 token 的自由。

这个选择很重要。GPT-2 的正则表达式阻止分词器学习到一个词末尾的 "the" 和下一个词开头的 "the" 应该合并。SentencePiece 允许这样做，有时能产生更高效的压缩，但 token 的可解释性更差。

### 特殊 Token

每个生产级分词器都为结构标记预留了 token ID：

| Token | 用途 | 使用者 |
|-------|---------|---------|
| `[BOS]` / `<s>` | 序列开始 | Llama 3、GPT |
| `[EOS]` / `</s>` | 序列结束 | 所有模型 |
| `[PAD]` | 批次对齐的填充 | BERT、T5 |
| `[UNK]` | 未知 token（字节级 BPE 消除了它） | BERT、WordPiece |
| `<\|im_start\|>` | 聊天消息边界开始 | ChatGPT、Qwen |
| `<\|im_end\|>` | 聊天消息边界结束 | ChatGPT、Qwen |
| `<\|user\|>` | 用户轮次标记 | Llama 3 |
| `<\|assistant\|>` | 助手轮次标记 | Llama 3 |

特殊 token 永远不会被 BPE 分割。它们在合并算法运行之前被精确匹配，替换为它们的固定 ID，周围的文本正常分词。

### 聊天模板

这是大多数人感到困惑的地方，也是大多数实现会出错的地方。

当你向聊天模型发送消息时，API 接受一个消息列表：

```
[
  {"role": "system", "content": "You are helpful."},
  {"role": "user", "content": "Hello"},
  {"role": "assistant", "content": "Hi there!"}
]
```

模型看到的不是 JSON。它看到的是一个扁平的 token 序列。聊天模板使用特殊 token 将消息转换为那个扁平序列。每个模型处理方式不同：

```
Llama 3:
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hi there!<|eot_id|>

ChatGPT:
<|im_start|>system
You are helpful.<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi there!<|im_end|>
```

模板搞错了，模型就会产生垃圾输出。它是用唯一一种确切格式训练的。任何偏差——一个缺失的换行、一个交换的 token、一个多余的空格——都会将输入置于训练分布之外。

### 速度

Python 对于生产级分词来说太慢了。

tiktoken（OpenAI）是用 Rust 编写的，带有 Python 绑定。HuggingFace tokenizers 也是 Rust。SentencePiece 是 C++。这些实现了比纯 Python 高 10-100 倍的加速。

做个比较：以每秒 100 万个 token 的速度（快速的 Python）对 Llama 3 预训练的 15 万亿 token 进行分词需要 174 天。以每秒 1 亿个 token 的速度（Rust），只需要 1.7 天。

你正在用 Python 构建以理解算法。在生产中，你会使用编译实现，只接触 Python 包装器。

```figure
weight-tying
```

## 构建它

### 第 1 步：字节级编码

基础。将任何字符串转换为字节序列，将每个字节映射为可打印字符以便显示，并反向操作。

```python
def bytes_to_tokens(text):
    return list(text.encode("utf-8"))

def tokens_to_text(token_bytes):
    return bytes(token_bytes).decode("utf-8", errors="replace")
```

在多语言文本上测试，查看字节计数：

```python
texts = [
    ("English", "hello"),
    ("Chinese", "你好"),
    ("Emoji", "🔥"),
    ("Mixed", "hello你好🔥"),
]

for label, text in texts:
    b = bytes_to_tokens(text)
    print(f"{label}: {len(text)} chars -> {len(b)} bytes -> {b}")
```

"hello" 是 5 个字节。"你好" 是 6 个字节（每个字符 3 字节）。火焰 emoji 是 4 个字节。字节级分词器不关心这是什么语言。字节就是字节。

### 第 2 步：带正则表达式的预分词器

使用 GPT-2 的正则表达式模式将文本分割成块。每个块由 BPE 独立分词。

```python
import re

try:
    import regex
    GPT2_PATTERN = regex.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
except ImportError:
    GPT2_PATTERN = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""
    )

def pre_tokenize(text):
    return [match.group() for match in GPT2_PATTERN.finditer(text)]
```

`regex` 模块支持 Unicode 属性转义（`\p{L}` 表示字母，`\p{N}` 表示数字）。标准库的 `re` 模块不支持，所以我们回退到 ASCII 字符类。对于生产级多语言分词器，安装 `regex`。

试试看：

```python
print(pre_tokenize("Hello, world! Don't stop."))
# [' Hello', ',', ' world', '!', " Don", "'t", ' stop', '.']
```

前导空格保持附加在单词上。缩写词在撇号处分割。标点符号成为自己的块。BPE 永远不会跨这些边界合并 token。

### 第 3 步：在字节序列上的 BPE

第 01 课中的核心算法，但现在在预分词的块上独立操作。

```python
from collections import Counter

def get_byte_pairs(chunks):
    pairs = Counter()
    for chunk in chunks:
        byte_seq = list(chunk.encode("utf-8"))
        for i in range(len(byte_seq) - 1):
            pairs[(byte_seq[i], byte_seq[i + 1])] += 1
    return pairs

def apply_merge(byte_seq, pair, new_id):
    merged = []
    i = 0
    while i < len(byte_seq):
        if i < len(byte_seq) - 1 and byte_seq[i] == pair[0] and byte_seq[i + 1] == pair[1]:
            merged.append(new_id)
            i += 2
        else:
            merged.append(byte_seq[i])
            i += 1
    return merged
```

### 第 4 步：特殊 Token 处理

特殊 token 需要精确匹配和固定 ID。它们完全绕过 BPE。

```python
class SpecialTokenHandler:
    def __init__(self):
        self.special_tokens = {}
        self.pattern = None

    def add_token(self, token_str, token_id):
        self.special_tokens[token_str] = token_id
        escaped = [re.escape(t) for t in sorted(self.special_tokens.keys(), key=len, reverse=True)]
        self.pattern = re.compile("|".join(escaped))

    def split_with_specials(self, text):
        if not self.pattern:
            return [(text, False)]
        parts = []
        last_end = 0
        for match in self.pattern.finditer(text):
            if match.start() > last_end:
                parts.append((text[last_end:match.start()], False))
            parts.append((match.group(), True))
            last_end = match.end()
        if last_end < len(text):
            parts.append((text[last_end:], False))
        return parts
```

### 第 5 步：完整的分词器类

把一切串联起来：标准化、在特殊 token 处分割、预分词、BPE 合并、映射到 ID。

```python
import unicodedata

class ProductionTokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special_handler = SpecialTokenHandler()
        self.next_id = 256

    def normalize(self, text):
        return unicodedata.normalize("NFKC", text)

    def train(self, text, num_merges):
        text = self.normalize(text)
        chunks = pre_tokenize(text)
        chunk_bytes = [list(chunk.encode("utf-8")) for chunk in chunks]

        for i in range(num_merges):
            pairs = Counter()
            for seq in chunk_bytes:
                for j in range(len(seq) - 1):
                    pairs[(seq[j], seq[j + 1])] += 1
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            new_id = self.next_id
            self.next_id += 1
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            chunk_bytes = [apply_merge(seq, best, new_id) for seq in chunk_bytes]

    def add_special_token(self, token_str):
        token_id = self.next_id
        self.next_id += 1
        self.special_handler.add_token(token_str, token_id)
        self.vocab[token_id] = token_str.encode("utf-8")
        return token_id

    def encode(self, text):
        text = self.normalize(text)
        parts = self.special_handler.split_with_specials(text)
        all_ids = []
        for part_text, is_special in parts:
            if is_special:
                all_ids.append(self.special_handler.special_tokens[part_text])
            else:
                for chunk in pre_tokenize(part_text):
                    byte_seq = list(chunk.encode("utf-8"))
                    for pair, new_id in self.merges.items():
                        byte_seq = apply_merge(byte_seq, pair, new_id)
                    all_ids.extend(byte_seq)
        return all_ids

    def decode(self, ids):
        byte_parts = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_parts.append(self.vocab[token_id])
        return b"".join(byte_parts).decode("utf-8", errors="replace")

    def vocab_size(self):
        return len(self.vocab)
```

### 第 6 步：多语言测试

真正的测试。向它输入英语、中文、emoji 和代码。

```python
corpus = (
    "The quick brown fox jumps over the lazy dog. "
    "The quick brown fox runs through the forest. "
    "Machine learning models process natural language. "
    "Deep learning transforms how we build software. "
    "def train(model, data): return model.fit(data) "
    "def predict(model, x): return model(x) "
)

tok = ProductionTokenizer()
tok.train(corpus, num_merges=50)

bos = tok.add_special_token("<|begin|>")
eos = tok.add_special_token("<|end|>")

test_texts = [
    "The quick brown fox.",
    "你好世界",
    "Hello 🌍 World",
    "def foo(x): return x + 1",
    f"<|begin|>Hello<|end|>",
]

for text in test_texts:
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    print(f"Input:   {text}")
    print(f"Tokens:  {len(ids)} ids")
    print(f"Decoded: {decoded}")
    print()
```

中文字符每个产生 3 个字节。emoji 产生 4 个字节。这些都不会让分词器崩溃。都不会产生未知 token。这就是字节级 BPE 的威力。

## 使用它

### 比较真正的分词器

加载 Llama 3、GPT-4 和 Mistral 的实际分词器。看看每个分词器如何处理相同的多语言段落。

```python
import tiktoken

gpt4_enc = tiktoken.get_encoding("cl100k_base")

test_paragraph = "Machine learning is powerful. 机器学习很强大。 L'apprentissage automatique est puissant. 🤖💪"

tokens = gpt4_enc.encode(test_paragraph)
pieces = [gpt4_enc.decode([t]) for t in tokens]
print(f"GPT-4 ({len(tokens)} tokens): {pieces}")
```

```python
from transformers import AutoTokenizer

llama_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

for name, tok in [("Llama 3", llama_tok), ("Mistral", mistral_tok)]:
    tokens = tok.encode(test_paragraph)
    pieces = tok.convert_ids_to_tokens(tokens)
    print(f"{name} ({len(tokens)} tokens): {pieces[:20]}...")
```

你会看到相同的文本有不同的 token 计数。拥有 128K 词汇表的 Llama 3 在合并常见模式上更激进。拥有 100K 的 GPT-4 处于中间。拥有 32K 的 Mistral 产生更多的 token，但嵌入层更小。

权衡总是相同的：更大的词汇表意味着更短的序列，但更多的参数。

## 交付成果

本课程产出一个用于构建和调试生产级分词器的提示词。参见 `outputs/prompt-tokenizer-builder.md`。

## 练习

1. **简单：** 添加一个 `get_token_bytes(id)` 方法，显示任何 token ID 的原始字节。用它来检查你最常用的合并 token 实际代表什么。
2. **中等：** 实现 Llama 风格的预分词器，按空格和数字分割但保留前导空格。在相同语料上比较其词汇表与 GPT-2 正则表达式的方法。
3. **困难：** 添加一个聊天模板方法，接受 `{"role": ..., "content": ...}` 消息列表，并为 Llama 3 聊天格式生成正确的 token 序列。对照 HuggingFace 实现测试它。

## 关键术语

| 术语 | 人们怎么说 | 真正的含义是什么 |
|------|----------------|----------------------|
| 字节级 BPE | "在字节上工作的分词器" | 具有 256 个字节值基础词汇表的 BPE——处理任何输入都不会有未知 token |
| 预分词 | "BPE 之前的分割" | 基于正则或规则的分割，防止 BPE 跨词边界合并 |
| NFKC 标准化 | "Unicode 清理" | 规范分解后进行兼容性组合——"fi" 连字变成 "fi"，全角 "A" 变成 "A" |
| 聊天模板 | "消息如何变成 token" | 将角色/内容消息列表转换为扁平 token 序列的确切格式——特定于模型，必须匹配训练格式 |
| 特殊 Token | "控制 token" | 绕过 BPE 的保留 token ID——[BOS]、[EOS]、[PAD]、聊天标记——在合并前精确匹配 |
| 繁殖率 | "每个词的 token 数" | 输出 token 与输入单词的比率——英语在 GPT-4 中是 1.3，韩语是 2-3，越高意味着浪费的上下文越多 |
| tiktoken | "OpenAI 分词器" | Rust 编写的 BPE 实现，带 Python 绑定——比纯 Python 快 10-100 倍 |
| 合并表 | "词汇表" | 训练期间学习到的字节对合并的有序列表——这就是分词器的学习知识 |

## 延伸阅读

- [OpenAI tiktoken 源码](https://github.com/openai/tiktoken)——GPT-3.5/4 使用的 Rust BPE 实现
- [HuggingFace tokenizers](https://github.com/huggingface/tokenizers)——支持 BPE、WordPiece、Unigram 的 Rust 分词器库
- [Llama 3 论文（Meta, 2024）](https://arxiv.org/abs/2407.21783)——关于 128K 词汇表和分词器训练的细节
- [SentencePiece（Kudo & Richardson, 2018）](https://arxiv.org/abs/1808.06226)——与语言无关的分词方案
- [GPT-2 分词器源码](https://github.com/openai/gpt-2/blob/master/src/encoder.py)——原始的字节到 Unicode 映射
