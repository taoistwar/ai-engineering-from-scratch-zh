# KV 缓存、Flash Attention 和推理优化

> 训练是并行且受限于 FLOPs 的。推理是串行且受限于内存的。不同的瓶颈，不同的技巧。

**类型：** 构建
**语言：** Python
**前置条件：** 第七阶段 · 02（自注意力），第七阶段 · 05（完整 Transformer），第七阶段 · 07（GPT）
**时间：** 约 75 分钟

## 问题

一个朴素的自回归解码器做 `O(N²)` 工作来生成 `N` 个 token：每一步它都在整个前缀上重新计算注意力。对于 4K token 的响应，那是 1600 万次注意力操作，其中大部分是冗余的。前缀 token 的每个隐藏状态一旦计算就是确定性的——你只需要用新 token 的 query 对照之前所有缓存的 key 和 value 运行即可。

除此之外，注意力本身移动了大量数据。标准注意力实例化一个 N×N 分数矩阵、N×d softmax 输出、N×d 最终输出——对 HBM 的读写太多了。对于 N≥2K，注意力在受限于 FLOPs 之前先受限于内存。经典注意力内核对现代 GPU 的利用率低 4–10 倍。

两项优化，都来自 Dao 等人，将前沿推理从"慢"推到"快"：

1. **KV 缓存。** 存储每个前缀 token 的 K 和 V 向量。每个新 token 的注意力是一次 query 对照缓存的 keys。推理从 `O(N²)` 减少到每个生成步骤的 `O(N)`。
2. **Flash Attention。** 将注意力计算分块，使完整的 N×N 矩阵永远不会到达 HBM。所有 softmax + 矩阵乘法都在 SRAM 中发生。A100 上挂钟加速 2–4 倍；H100 上用 FP8 达到 5–10 倍。

到 2026 年，两者都是普遍的。每个生产推理技术栈（vLLM、TensorRT-LLM、SGLang、llama.cpp）都假设它们。每个前沿模型都以 Flash Attention 启用。

## 概念

![KV 缓存增长和 Flash Attention 分块](../assets/kv-cache-flash-attn.svg)

### KV 缓存数学

每个解码器层，每个 token，每个头：

```
bytes_per_token_per_layer = 2 * d_head * dtype_size
                            ^
                            K 和 V
```

对于具有 32 层、32 头、d_head=128、fp16 的 7B 模型：

```
每个 token 每层 = 2 * 128 * 2 = 512 字节
每个 token（32 层）= 16 KB
每 32K 上下文 = 512 MB
```

对于 Llama 3 70B（80 层，d_head=128，GQA 具有 8 个 KV 头）：

```
每个 token 每层 = 2 * 8 * 128 * 2 = 4096 字节（4 KB）
每 32K 上下文 = 10.4 GB
```

那 10 GB 就是为什么 Llama 3 70B 在 128K 上下文时，仅 KV 缓存在批量大小为 1 时就需要 40 GB A100 的大部分。

**GQA 是 KV 缓存的胜利。** 具有 64 头的 MHA 将是 32 GB。MLA 压缩得更多。

拖动维度并观察缓存大小的变化。将序列长度或批量推高，看看它如何快速超过单个 GPU：

```figure
kv-cache-sizer
```

### Flash Attention — 分块技巧

标准注意力：

```
S = Q @ K^T          （HBM 读取，N×N，HBM 写入）
P = softmax(S)       （HBM 读取，HBM 写入）
O = P @ V            （HBM 读取，HBM 写入）
```

三次 HBM 往返。在 H100 上，HBM 带宽是 3 TB/s；SRAM 是 30 TB/s。每次 HBM 往返相比将所有内容保留在芯片上是 10 倍的减速。

Flash Attention：

```
对于每个 Q 块（块大小 ~128 × 128）：
    将 Q_tile 加载到 SRAM
    对于每个 K、V 块：
        将 K_tile、V_tile 加载到 SRAM
        计算 S_tile = Q_tile @ K_tile^T    （SRAM）
        运行 softmax 聚合                    （SRAM）
        累积到 O_tile                       （SRAM）
    将 O_tile 写入 HBM
```

每个块一次 HBM 往返。总内存占用从 `O(N²)` 下降到 `O(N)`。反向传播从前向传播重新计算一些值而不是存储它们——另一个内存收益。

**数值技巧。** 运行 softmax 在块之间维护 `(max, sum)` 以使最终归一化是精确的。不是近似——Flash Attention 计算与标准注意力位对位相同的输出（模 fp16 非结合性）。

**版本演进：**

| 版本 | 年份 | 关键变化 | 参考硬件上的加速 |
|---------|------|-----------|-------------------------------|
| Flash 1 | 2022 | 分块 SRAM 内核 | A100 上 2× |
| Flash 2 | 2023 | 更好的并行性，因果优先排序 | A100 上 3× |
| Flash 3 | 2024 | Hopper 异步，FP8 | H100 上 1.5–2×（~740 TFLOPs FP16） |
| Flash 4 | 2026 | Blackwell 5 阶段流水线，软件 exp2 | 推理优先（初始仅前向传播） |

Flash 4 在发布时仅支持前向传播。训练仍然使用 Flash 3。Flash 4 的 GQA 和变长支持待定（2026 年中）。

### 推测解码 — 另一个延迟胜利

廉价模型提出 N 个 token。大模型并行验证所有 N 个。如果验证接受 k 个 token，你为 k 代支付了 1 次大模型前向传播。在代码和散文上典型的 k=3–5。

2026 年默认方案：
- **EAGLE 2 / Medusa。** 共享验证器隐藏状态的集成草稿头。在无质量损失下加速 2–3 倍。
- **使用草稿模型的推测解码。** 在消费级硬件上加速 2–4 倍。
- **前瞻解码。** Jacobi 迭代；不需要草稿模型。小众但免费。

### 连续批处理

经典批处理推理：等待最慢的序列完成，然后开始新的批次。当短响应提前完成时浪费 GPU。

连续批处理（首次在 Orca 中发布，现在在 vLLM、TensorRT-LLM、SGLang 中）：一旦旧请求完成就将新请求交换到批次中。对于典型聊天工作负载，吞吐量增益 5–10 倍。

### PagedAttention — 作为虚拟内存的 KV 缓存

vLLM 的头条功能。KV 缓存在 16 token 的块中分配；页表将逻辑位置映射到物理块。让你可以跨并行样本共享 KV（束搜索、并行采样），为提示缓存热交换前缀，并整理内存碎片。相比朴素的连续分配，吞吐量提高 4 倍。

```figure
flash-attention-memory
```

## 动手构建

参见 `code/main.py`。我们实现：

1. 一个朴素的 `O(N²)` 增量解码器。
2. 一个 `O(N)` KV 缓存解码器。
3. 一个模拟 Flash Attention 运行最大值算法的分块 softmax。

### 步骤 1：KV 缓存

```python
class KVCache:
    def __init__(self, n_layers, n_heads, d_head):
        self.K = [[[] for _ in range(n_heads)] for _ in range(n_layers)]
        self.V = [[[] for _ in range(n_heads)] for _ in range(n_layers)]

    def append(self, layer, head, k, v):
        self.K[layer][head].append(k)
        self.V[layer][head].append(v)

    def read(self, layer, head):
        return self.K[layer][head], self.V[layer][head]
```

简单：保持每个 token 的 K、V 向量在每层、每头的列表中增长。

### 步骤 2：分块 softmax

```python
def tiled_softmax_dot(q, K, V, tile=4):
    """Flash-attention 风格的 softmax(qK^T)V，使用运行 max/sum。"""
    m = float("-inf")
    s = 0.0
    out = [0.0] * len(V[0])
    for start in range(0, len(K), tile):
        k_block = K[start:start + tile]
        v_block = V[start:start + tile]
        scores = [sum(qi * ki for qi, ki in zip(q, k)) for k in k_block]
        new_m = max(m, *scores)
        exp_old = math.exp(m - new_m) if m != float("-inf") else 0.0
        exp_new = [math.exp(sc - new_m) for sc in scores]
        s = s * exp_old + sum(exp_new)
        for j in range(len(out)):
            out[j] = out[j] * exp_old + sum(e * v[j] for e, v in zip(exp_new, v_block))
        m = new_m
    return [o / s for o in out]
```

一次输出与 `softmax(qK) V` 位对位相同，但任何时候工作集是一个 `tile × d_head` 块，不是完整的 `N × d_head`。

### 步骤 3：在 100 token 生成上比较朴素 vs 缓存解码

计算注意力操作。朴素：`O(N²)` = 5050。缓存：`O(N)` = 100。代码打印两者。

## 使用它

```python
# HuggingFace transformers 在仅解码器 generate() 上自动启用 KV 缓存。
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    attn_implementation="flash_attention_2",  # 如果是 Hopper 使用 FA3
    torch_dtype="bfloat16",
)
# generate() 自动使用 KV 缓存
```

vLLM 生产：

```bash
pip install vllm
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8
```

跨请求的前缀缓存是 2026 年的一大胜利——相同的系统提示、few-shot 示例或长上下文文档在调用之间重用 KV。对于具有重复工具提示的代理工作负载，前缀缓存通常实现 5 倍吞吐量增益。

## 交付成果

参见 `outputs/skill-inference-optimizer.md`。该技能为新的推理部署选择注意力实现、KV 缓存策略、量化和推测解码。

## 练习

1. **简单。** 运行 `code/main.py`。确认朴素和缓存解码器产生相同的输出；注意操作计数的差异。
2. **中等。** 实现前缀缓存：给定一个提示 P 和几个补全，在 P 上运行一次前向传播以填充 KV 缓存，然后按补全分支。测量与为每个补全重新编码 P 相比的加速。
3. **困难。** 实现一个玩具 PagedAttention：在固定 16 token 块中使用空闲列表的 KV 缓存。当序列完成时，将其块返回到池中。模拟 1,000 个不同长度的聊天补全。比较内存碎片与连续分配。

## 关键术语

| 术语 | 人们怎么说 | 它实际上的含义 |
|------|-----------------|-----------------------|
| KV 缓存 | "使解码变快的技巧" | 存储每个前缀 token 的 K 和 V；新查询对照它们进行注意力而不是重新计算。 |
| HBM | "GPU 主内存" | 高带宽内存；H100 上 80 GB，B200 上 192 GB。~3 TB/s 带宽。 |
| SRAM | "片上内存" | 每个 SM 的快速内存，H100 上每个 SM ~256 KB。~30 TB/s 带宽。 |
| Flash Attention | "分块注意力内核" | 在不实例化 N×N 矩阵于 HBM 的情况下计算注意力。 |
| 连续批处理 | "无等待批处理" | 在不排空批次的情况下将完成的序列换出，新序列换入。 |
| PagedAttention | "vLLM 的头条功能" | KV 缓存在固定块中分配，带有页表；消除碎片。 |
| 前缀缓存 | "重用长提示" | 跨请求缓存共享前缀的 KV；为代理大幅削减成本。 |
| 推测解码 | "草稿 + 验证" | 廉价草稿模型提出 token；大模型在一次传递中验证 k 个。 |

## 延伸阅读

- [Dao 等人 (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Flash 1。
- [Dao (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) — Flash 2。
- [Shah 等人 (2024). FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) — Flash 3。
- [FlashAttention-4 发布说明（Dao-AILab，2026）](https://github.com/Dao-AILab/flash-attention) — Blackwell 5 阶段流水线和软件 exp2 技巧；阅读仓库 README 以了解本课提到的仅前向传播启动注意事项。
- [Kwon 等人 (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — vLLM 论文。
- [Leviathan 等人 (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — 推测解码。
- [Li 等人 (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — EAGLE-1/2 论文，针对本课引用的集成草稿方法。
- [Cai 等人 (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — 与 EAGLE 并列引用的 Medusa 方法。
- [vLLM 文档 — PagedAttention](https://docs.vllm.ai/en/latest/design/kernel/paged_attention.html) — 关于 16 token 块和页表设计的规范深入解析。
