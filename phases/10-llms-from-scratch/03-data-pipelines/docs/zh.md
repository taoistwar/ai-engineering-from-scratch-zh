# 用于预训练的数据流水线

> 模型是一面镜子。它反映出你输入的任何数据。输入垃圾，它就会完美流畅地反映出垃圾。

**类型：** 构建
**语言：** Python
**前置课程：** Phase 10, 第 01-02 课（分词器、构建分词器）
**时间：** ~90 分钟

## 学习目标

- 构建一个流式数据流水线，对 TB 级别的文本进行分词、分块、随机化和批处理，无需将所有数据加载到内存中
- 实现真实预训练流水线中使用的数据质量过滤器（去重、语言检测、内容过滤）
- 创建具有适当注意力掩码和文档边界处理的固定长度训练序列
- 对流水线吞吐量进行分析，确保数据加载器能跟上 GPU 训练的速度

## 问题

你有了一个分词器。现在你需要数据。

不是一个数据集。不是一个 CSV 文件。而是 TB 级别的文本——经过清洗、去重、质量过滤、分词为固定长度序列，并以随机化批次提供，速度足够快，使得你的 8 GPU 集群永远不会等待下一个批次。

大多数人认为训练 LLM 是关于模型架构。其实不是。Llama 3 使用了 15.6 万亿个 token。GPT-3 使用了 3000 亿个。DeepSeek-V2 使用了 8.1 万亿个。这三者的架构大致相同：堆叠的 Transformer 块，包含注意力层和前馈层。输出质量的差异绝大部分来自数据。

来自 DeepMind 的 Chinchilla 论文对此做出了精确的阐述。对于给定的计算预算，存在模型参数与训练 token 的最优比率。Chinchilla 表明，2022 年的大多数模型都严重训练不足——对于它们看到的数据量，它们的参数太多了。一个按 Chinchilla 最优训练的 70B 参数模型（使用了 1.4 万亿 token），超越了在 3000 亿 token 上训练的 280B 模型（Gopher）。

你的数据流水线决定了你的模型是学习语言还是学习噪声。

## 概念

### 数据从哪里来

每个大型语言模型都是在多种来源的混合数据上训练的。大多数实验室对确切的组成保密，但我们知道足够多来理解这些类别。

| 来源 | 大小 | 质量 | 使用者 |
|--------|------|---------|---------|
| Common Crawl | ~250 TB 原始 | 低（需要大量过滤） | GPT-3、Llama、大多数开放模型 |
| Wikipedia | ~20 GB | 高 | 每个主要的 LLM |
| GitHub 代码 | ~1 TB+ | 中（大量重复、死代码） | StarCoder、CodeLlama、DeepSeek-Coder |
| 书籍（BookCorpus、Pile） | ~100 GB | 高 | GPT-2、GPT-3、早期模型 |
| 学术论文（arXiv、S2ORC） | ~100 GB | 对于 STEM 高 | Llama、Galactica |
| StackOverflow、Reddit | ~100 GB | 中 | Llama、Falcon |
| 精选网络数据（C4、RefinedWeb） | ~5 TB | 中-高（已预过滤） | T5、Falcon |

Llama 3 公开了它的数据混合：大约 50% 网络数据、25% 代码、13% 书籍和学术论文、8% 数学数据以及 4% 多语言网络数据。总量是从超过 5 TB 的原始文本来源中提取的 15.6 万亿 token。

比率和总量同样重要。网络数据太多，模型就会变成 Reddit 鹦鹉。代码太少，它就无法编程。数学太少，它在推理上就会失败。正确的混合配比是训练 LLM 最困难的部分之一，没有公式——需要实验和评估。

### 数据清洗

原始网络数据非常肮脏。一个典型的 Common Crawl 转储包含：

- HTML 标签和 JavaScript
- 样板化的页眉、页脚、导航菜单
- 重复页面（完全重复和近似重复）
- 机器生成的垃圾信息
- 个人身份信息（PII）
- 低质量文本（关键词列表、SEO 垃圾信息）
- 编码为文本的非文本内容

清洗这些不是可选项。这决定了一个模型是生成连贯的段落，还是输出混杂着产品列表的 HTML 标签。

```mermaid
graph TD
    A[Raw Text] --> B[HTML Strip]
    B --> C[Language Detection]
    C --> D[Quality Filter]
    D --> E[Deduplication]
    E --> F[PII Removal]
    F --> G[Clean Text]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#e94560,color:#fff
    style C fill:#1a1a2e,stroke:#e94560,color:#fff
    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
    style G fill:#1a1a2e,stroke:#e94560,color:#fff
```

每个步骤消除一类噪声：

**HTML 剥离：** 去除所有标记。只保留可见的文本内容。像 `trafilatura` 或 `readability` 这样的库可以在丢弃导航、广告和样板内容的同时提取文章内容。

**语言检测：** 使用 fastText 的语言识别模型（lid.176.bin）对每个文档进行分类。过滤到你需要的目标语言。对于被分类为英语但置信度低于 0.8 的文档，可能不是干净的英语。

**质量过滤：** 这里变得有意思了。RefinedWeb（Falcon 背后的数据集）使用基于困惑度的过滤器：在 Wikipedia 上训练一个小型语言模型，然后对每个文档进行评分。高困惑度意味着文档不像 Wikipedia——很可能是垃圾信息、关键词列表或机器生成的内容。困惑度超过阈值的文档会被移除。

**去重：** 最具影响力的单步清洗操作。Common Crawl 包含大量的重复页面——法律免责声明、cookie 通知、服务条款。在重复内容上训练浪费计算资源，并且可能导致模型逐字记忆和复述特定的段落。

**PII 移除：** 姓名、电子邮件地址、电话号码、社会安全号码。基于正则表达式的结构化 PII 检测，以及用于上下文中人名的 NER 模型。

### 使用 MinHash 去重

完全去重很简单：对每个文档哈希，移除重复。但近似重复才是真正的问题。同一篇新闻文章的两个副本，周围带有略微不同的广告，就是近似重复。内容 95% 相同，但逐字节不同。

MinHash + 局部敏感哈希（LSH）有效地解决了这个问题。

```mermaid
graph LR
    A[Document] --> B[Shingling]
    B --> C[MinHash Signature]
    C --> D[LSH Buckets]
    D --> E[Candidate Pairs]
    E --> F[Jaccard Similarity]
    F --> G[Deduplicated Set]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#e94560,color:#fff
    style C fill:#1a1a2e,stroke:#e94560,color:#fff
    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
    style G fill:#1a1a2e,stroke:#e94560,color:#fff
```

思路：

1. **Shingling：** 将每个文档转换为 n-gram 的集合（例如，5-gram 的单词或字符）。"the quick brown fox" 以 3-单词 shingles 变成 {"the quick brown", "quick brown fox"}。

2. **MinHash：** 对于每个文档的 shingle 集合，计算 k 个哈希值。每个哈希值是所有 shingle 在不同哈希函数下的最小哈希值。这创建了一个固定大小的"签名"，近似地表示任意两个文档之间的 Jaccard 相似度。

3. **LSH：** 根据文档 MinHash 签名中的带（bands）将文档分组到桶中。在同一个桶中的文档是候选的近似重复。这避免了比对每一对——你只比对候选对。

4. **验证：** 对于每个候选对，计算精确的 Jaccard 相似度。如果相似度超过阈值（通常为 0.8），移除一个副本。

Llama 团队报告说，通过去重移除了大约 38% 的网络数据。这不是一个小数字。超过三分之一的 Common Crawl 是重复或近似重复的内容。

### 序列打包

你的模型期望固定长度的输入序列。你的文档是可变长度的。有些 50 个 token。有些 50,000 个 token。

朴素方法：将每个文档填充到最大序列长度。这在填充 token 上浪费了大量计算，对学习没有任何贡献。

更好的方法：将多个文档打包到一个序列中，用序列结束 token 分隔。一个 2048 token 的序列可能包含三个短文档，用 [EOS] token 连接起来。

```mermaid
graph TD
    subgraph Naive Packing
        A1["Doc A (200 tokens)"] --> P1["[PAD] x 1848"]
        A2["Doc B (500 tokens)"] --> P2["[PAD] x 1548"]
        A3["Doc C (100 tokens)"] --> P3["[PAD] x 1948"]
    end

    subgraph Efficient Packing
        B1["Doc A (200) | Doc B (500) | Doc C (100) | Doc D (400) | Doc E (848)"]
    end

    style A1 fill:#1a1a2e,stroke:#e94560,color:#fff
    style A2 fill:#1a1a2e,stroke:#e94560,color:#fff
    style A3 fill:#1a1a2e,stroke:#e94560,color:#fff
    style P1 fill:#333,stroke:#666,color:#999
    style P2 fill:#333,stroke:#666,color:#999
    style P3 fill:#333,stroke:#666,color:#999
    style B1 fill:#1a1a2e,stroke:#16c784,color:#fff
```

注意力掩码必须正确设置。在同一打包序列中，文档 A 的 token 不应关注文档 B 的 token。这需要一个分块对角注意力掩码。

长文档在序列边界处被截断或分割成块。分割点很重要：在句子中间分割会迫使模型看到不完整的思维。一些流水线尽可能将分割对齐到段落或句子边界。

### Chinchilla 缩放定律

对于固定的计算预算 C（以 FLOPs 计），最优模型大小 N 和数据集大小 D 遵循：

```
N_opt ~ C^0.5
D_opt ~ C^0.5
```

在实践中，这意味着你应该大致相等地缩放模型大小和数据集大小。参数多 10 倍的模型大约需要多 10 倍的训练 token 才能达到相同的损失。

| 模型 | 参数 | 训练 Token | 符合 Chinchilla 最优？ |
|-------|-----------|----------------|-------------------|
| GPT-3 | 175B | 300B | 否（训练不足 3-4 倍） |
| Chinchilla | 70B | 1.4T | 是（按设计） |
| Llama 2 | 70B | 2T | 过度训练（有意为之） |
| Llama 3 | 70B | 15T | 严重过度训练 |

Llama 3 有意违反了 Chinchilla 定律。Meta 发现，在远超计算最优比例的数据上进行过度训练，能为推理产生更好的模型。额外的训练成本是一次性的，但更小的模型永远更便宜地提供服务。这有时被称为"推理最优"的缩放方法，自 2024 年以来已成为行业标准。

## 构建它

### 第 1 步：文本清洗

剥离 HTML，标准化空白字符，移除非文本内容。我们将使用公共领域文本（Project Gutenberg）作为我们的小型语料。

```python
import re

def clean_text(text):
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\x20-\x7E\n]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()

def quality_filter(text, min_words=50, max_ratio_caps=0.3, max_ratio_special=0.1):
    words = text.split()
    if len(words) < min_words:
        return False
    caps_ratio = sum(1 for w in words if w.isupper()) / len(words)
    if caps_ratio > max_ratio_caps:
        return False
    special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
    if special_chars / max(len(text), 1) > max_ratio_special:
        return False
    return True
```

质量过滤器能捕获 SEO 垃圾信息（全大写）、机器生成的噪声（高特殊字符比例）和短页面（太短）。仅这三项检查就能从网络爬取中移除惊人的大量垃圾。

### 第 2 步：MinHash 去重

从零开始实现 MinHash。不需要外部库——只需 `hashlib`。

```python
import hashlib
from collections import defaultdict

def get_shingles(text, k=5):
    words = text.lower().split()
    if len(words) < k:
        return set()
    return {" ".join(words[i:i+k]) for i in range(len(words) - k + 1)}

def minhash_signature(shingles, num_hashes=128):
    signature = []
    for i in range(num_hashes):
        min_hash = float("inf")
        for shingle in shingles:
            h = int(hashlib.sha256(f"{i}:{shingle}".encode()).hexdigest(), 16)
            min_hash = min(min_hash, h)
        signature.append(min_hash)
    return signature

def lsh_buckets(signature, bands=16):
    rows_per_band = len(signature) // bands
    buckets = []
    for b in range(bands):
        start = b * rows_per_band
        band_data = tuple(signature[start:start + rows_per_band])
        bucket_hash = hashlib.md5(str(band_data).encode()).hexdigest()
        buckets.append((b, bucket_hash))
    return buckets

def deduplicate(documents, threshold=0.8, num_hashes=128, bands=16):
    signatures = []
    shingle_sets = []
    for doc in documents:
        shingles = get_shingles(doc)
        shingle_sets.append(shingles)
        signatures.append(minhash_signature(shingles, num_hashes))

    bucket_map = defaultdict(list)
    for doc_idx, sig in enumerate(signatures):
        for band_id, bucket_hash in lsh_buckets(sig, bands):
            bucket_map[(band_id, bucket_hash)].append(doc_idx)

    duplicate_pairs = set()
    for bucket_docs in bucket_map.values():
        if len(bucket_docs) < 2:
            continue
        for i in range(len(bucket_docs)):
            for j in range(i + 1, len(bucket_docs)):
                duplicate_pairs.add((bucket_docs[i], bucket_docs[j]))

    removed = set()
    for i, j in duplicate_pairs:
        if i in removed or j in removed:
            continue
        s1, s2 = shingle_sets[i], shingle_sets[j]
        if not s1 or not s2:
            continue
        jaccard = len(s1 & s2) / len(s1 | s2)
        if jaccard >= threshold:
            removed.add(j)

    return [doc for idx, doc in enumerate(documents) if idx not in removed], len(removed)
```

`num_hashes=128` 和 `bands=16` 参数控制着精确率与召回率的权衡。更多的哈希给出更准确的相似度估计。更多的带增加召回率（捕获更多重复），但代价是更多误报。这些值对典型的网络文本效果良好。

### 第 3 步：分词和打包序列

拿取清洗过的、去重后的文本，进行分词，并打包成固定长度的训练序列。

```python
def tokenize_corpus(documents, tokenizer):
    all_tokens = []
    for doc in documents:
        tokens = tokenizer.encode(doc)
        all_tokens.extend(tokens)
        all_tokens.append(tokenizer.eos_id)
    return all_tokens

def pack_sequences(token_ids, seq_length, pad_id=0):
    sequences = []
    attention_masks = []
    for i in range(0, len(token_ids), seq_length):
        seq = token_ids[i:i + seq_length]
        mask = [1] * len(seq)
        if len(seq) < seq_length:
            pad_count = seq_length - len(seq)
            seq = seq + [pad_id] * pad_count
            mask = mask + [0] * pad_count
        sequences.append(seq)
        attention_masks.append(mask)
    return sequences, attention_masks
```

### 第 4 步：用于训练的 DataLoader

生成随机化的打包序列批次。这是训练循环消费的东西。

```python
import random

class PreTrainingDataLoader:
    def __init__(self, sequences, attention_masks, batch_size, shuffle=True):
        self.sequences = sequences
        self.attention_masks = attention_masks
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return (len(self.sequences) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = list(range(len(self.sequences)))
        if self.shuffle:
            random.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch_idx = indices[start:start + self.batch_size]
            batch_seqs = [self.sequences[i] for i in batch_idx]
            batch_masks = [self.attention_masks[i] for i in batch_idx]
            yield batch_seqs, batch_masks
```

### 第 5 步：数据集统计

计算重要的数字：总 token 数、唯一 token 数、压缩比、文档长度分布。

```python
from collections import Counter

def compute_statistics(documents, token_ids, sequences, tokenizer_vocab_size):
    total_chars = sum(len(d) for d in documents)
    total_tokens = len(token_ids)
    unique_tokens = len(set(token_ids))
    compression_ratio = total_chars / total_tokens

    doc_lengths = [len(d.split()) for d in documents]
    avg_doc_length = sum(doc_lengths) / max(len(doc_lengths), 1)
    max_doc_length = max(doc_lengths) if doc_lengths else 0
    min_doc_length = min(doc_lengths) if doc_lengths else 0

    token_counts = Counter(token_ids)
    top_tokens = token_counts.most_common(10)

    non_pad_tokens = sum(sum(1 for t in seq if t != 0) for seq in sequences)
    total_positions = sum(len(seq) for seq in sequences)
    utilization = non_pad_tokens / max(total_positions, 1)

    stats = {
        "total_documents": len(documents),
        "total_characters": total_chars,
        "total_tokens": total_tokens,
        "unique_tokens": unique_tokens,
        "vocab_utilization": unique_tokens / tokenizer_vocab_size,
        "compression_ratio": compression_ratio,
        "avg_doc_length_words": avg_doc_length,
        "max_doc_length_words": max_doc_length,
        "min_doc_length_words": min_doc_length,
        "num_sequences": len(sequences),
        "sequence_utilization": utilization,
        "top_10_tokens": top_tokens,
    }
    return stats
```

压缩比告诉你分词器在这个语料上的效率。英语文本通常压缩到大约每个 token 3-4 个字符。如果你看到每个 token 1.5 个字符，你的分词器分割得过于激进。如果你看到 8+，它已经学会了非常领域特定的合并。

序列利用率告诉你打包序列中有多少是真实数据 vs 填充。低于 90% 意味着你的打包不够高效——你在填充 token 上浪费了计算。

## 使用它

### 与 HuggingFace Datasets 比较

通过 HuggingFace 的 datasets 库加载相同的语料，比较流水线速度。

```python
from datasets import load_dataset
from transformers import AutoTokenizer

ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

import time

start = time.time()
tokenized = ds.map(
    lambda x: tokenizer(x["text"], truncation=True, max_length=2048),
    batched=True,
    num_proc=4,
)
hf_time = time.time() - start
total_tokens = sum(len(t) for t in tokenized["input_ids"])
print(f"HuggingFace: {total_tokens:,} tokens in {hf_time:.2f}s ({total_tokens/hf_time:,.0f} tokens/sec)")
```

HuggingFace 流水线底层使用 Rust 分词器，并在 4 个核心上并行处理。你的纯 Python 流水线将慢 10-50 倍。这个差距就是生产团队使用编译分词器的原因。算法是相同的。实现语言是区别。

## 交付成果

本课程产出一个用于验证和调试 LLM 训练流水线中数据质量的提示词。参见 `outputs/prompt-data-quality-checker.md`。

## 练习

1. **简单：** 使用简单的启发式方法（字符集分析）向清洗流水线添加语言检测。只过滤到英语文档，并测量有多少文档被移除。
2. **中等：** 在使用 MinHash 近去重的同时，使用 SHA-256 哈希实现完全去重。在一个网络爬取的语料上，比较每种方法捕获的重复数量。
3. **困难：** 构建一个基于困惑度的质量过滤器。在 Wikipedia 文本上训练一个小型双元语言模型，按困惑度对每个文档评分，并移除底部 20%。比较在过滤和未过滤数据上训练后模型输出的质量。

## 关键术语

| 术语 | 人们怎么说 | 真正的含义是什么 |
|------|----------------|----------------------|
| Common Crawl | "互联网" | 一个每月爬取网络的非营利组织——约 250TB 原始数据，是大多数 LLM 训练数据的起点 |
| MinHash | "某种哈希技巧" | 一种使用固定大小签名来估计集合间 Jaccard 相似度的技术——实现大规模近似重复检测 |
| LSH | "局部敏感哈希" | "一种将相似项分组到同一个桶中的方法——将从 O(n^2) 的成对比较减少到近乎线性" |
| 序列打包 | "连接文档" | 将多个文档放入具有适当注意力掩码的固定长度序列——消除填充浪费 |
| Chinchilla 缩放 | "在更多数据上训练" | 对于固定的计算预算，最优性能需要大致相等地缩放模型大小和训练 token 数 |
| 繁殖率 | "每个词的 token 数" | 每个单词的平均 token 数——英语在 GPT-4 中是 1.3，非拉丁文字更高 |
| 数据混合 | "选择训练数据" | 代码 vs 文本 vs 数学 vs 多语言数据的比例——没有公式，需要实验 |
| 困惑度过滤器 | "质量评分" | 使用小型语言模型对文档评分——高困惑度意味着文本不像干净的参考数据 |
| 去重 | "移除副本" | 消除完全重复和近似重复的文档——通常移除 30-40% 的原始网络数据 |
| 注意力掩码 | "要关注哪些 token" | 一个二进制掩码，防止打包序列中跨文档边界的注意力 |

## 延伸阅读

- [Hoffmann et al., 2022 -- Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/abs/2203.15556)——改变了我们对数据规模看法的论文
- [Penedo et al., 2023 -- The RefinedWeb Dataset for Falcon LLM](https://arxiv.org/abs/2306.01116)——如何将 Common Crawl 过滤到高质量
- [Touvron et al., 2023 -- Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)——Llama 2 的数据流水线细节
- [Lee et al., 2022 -- Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499)——为什么去重比你想象的更重要
- [Broder, 1997 -- On the Resemblance and Containment of Documents](https://ieeexplore.ieee.org/document/666900)——原始的 MinHash 论文
- [Meta, 2024 -- Llama 3 Technical Report](https://arxiv.org/abs/2407.21783)——15.6T token、数据混合比率、过滤流水线
