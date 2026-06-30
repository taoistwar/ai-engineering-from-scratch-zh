# vLLM 推理内部机制：PagedAttention、连续批处理、分块预填充

> vLLM 在 2026 年的统治地位建立在三个复合默认值上，而不是一个技巧。PagedAttention 始终开启。连续批处理在解码迭代之间将新请求注入活跃批次。分块预填充将长提示切片，使解码 token 永远不会饥饿。全部开启三项优化，一台 H100 SXM5 上的 Llama 3.3 70B FP8 在 128 并发下可压出 2,200-2,400 tok/s — 比 vLLM 自身默认值高约 25%，比朴素的 PyTorch 循环快 3-4 倍。本课以你能画图理解的层次阅读调度器和注意力内核，并以 `code/main.py` 中的一个玩具连续批处理器结束，该批处理器以 vLLM 的方式调度预填充和解码。

**Type:** Learn
**Languages:** Python (stdlib, toy continuous batching scheduler)
**Prerequisites:** Phase 17 · 01 (Model Serving), Phase 11 (LLM Engineering)
**Time:** ~75 minutes

## 学习目标

- 将 PagedAttention 解释为 KV 缓存分配器：块、块表，以及为什么在生产负载下碎片化保持在 4% 以下。
- 在迭代层面画图解释连续批处理：已完成的序列如何离开批次，新序列如何在不排空的情况下加入。
- 用一句话描述分块预填充，并说出它保护了哪个延迟指标（提示：是 TTFT 尾部，而非平均吞吐量）。
- 说出 2026 年 vLLM v0.18.0 中那个同时启用所有优化时会咬到团队的坑。

## 问题

朴素的 PyTorch 推理循环每次处理一个请求：分词、预填充、解码直到 EOS、返回。一个用户时这可以工作。一百个用户时，这是一个充满耐心排队的人。显而易见的修复 — 静态批处理 — 将每个请求填充到窗口中提示最长的那一个，将每个解码填充到预期输出最长的那一个，并使整个批次卡在最慢的序列上。你为从未用到的填充付费，而快请求等慢请求。

vLLM 同时解决三个问题。PagedAttention 阻止 KV 缓存碎片化吞噬 60-80% 的 GPU 内存，这是经典连续分配的做法。连续批处理让请求在每个解码迭代之间加入和离开批次，因此批次始终充满真正的工作。分块预填充将 32k token 的提示分解为约 512 token 的切片，与解码交错进行，使长提示不会冻结 GPU 上的每个解码 token。

2026 年的生产默认是三项全部开启。你需要理解每项的作用，因为失败模式全在调度器上，而不是模型上。

## 概念

### PagedAttention 作为虚拟内存系统

KV 缓存是每条序列 `num_layers × 2 × num_heads × head_dim × seq_len × bytes_per_element`。对于 Llama 3.3 70B 在 8192 token 下，BF16 中约每序列 1.25 GB。如果你为每个请求预保留 8192 个槽但平均请求只使用 1500 个 token，你浪费了约 82% 你预留的 HBM。经典批处理承受这种浪费。

PagedAttention 借用操作系统的虚拟内存思想。KV 缓存不是按序列连续的。它分配为固定大小的块（默认 16 个 token）。每条序列有一个块表，将其逻辑 token 位置映射到物理块 ID。当序列增长超出其已分配的块时，添加一个额外块。当序列完成时，其块返回到池中。

碎片化从 60-80%（经典）降到 4% 以下（PagedAttention）。PagedAttention 并非通过一个标志启用 — 它是 vLLM 唯一的分配器。可调参数是 `--gpu-memory-utilization`（默认 0.9），告诉 vLLM 在加载权重和激活值后为 KV 块预留多少 HBM。

### 迭代层面的连续批处理

旧的"动态批处理"等待一个窗口（比如 10 毫秒）来填满一个批次，然后运行预填充 + 解码 + 解码 + 解码直到每个序列完成。快序列提前离开，在 GPU 完成慢序列时闲置。

连续批处理在每次解码步骤之间操作。将正在运行序列的集合称为 `RUNNING` 列表。每次迭代中：

1. `RUNNING` 中任何刚刚命中 EOS 或 max_tokens 的序列被移除。
2. 调度器查看等待队列。如果有空闲 KV 块，允许新序列进入（预填充或恢复）。
3. 前向传播在 `RUNNING` 中任何现有内容上运行，每条序列发出一个 token。

批次大小从不填充到固定数字。处于不同输出位置的序列共享一个融合的前向传播。在 2026 年的 vLLM 中，这称为 `V1 调度器`。关键不变性：调度器每次解码迭代运行一次，而不是每次请求运行一次。

### 分块预填充保护 TTFT 尾部

预填充是计算密集型的。Llama 3.3 70B 上一个 32k token 的提示在单个 H100 上需要约 800 毫秒的纯预填充。当预填充运行时，批次中所有其他序列的解码 token 都在等待。在推理循环中，一个长提示的首 token 延迟 (TTFT) 成为数十个其他用户的 token 间延迟 (ITL) 波动。

分块预填充将预填充拆分为固定大小的块（默认 512 个 token），并将每个块作为一个单元调度。在块之间，调度器可以将解码序列向前推进一个 token。你用一个小的绝对预填充延迟损失（每块几毫秒）换取低得多的解码时抖动。在混合负载下，P99 ITL 从发布的基准测试中的约 50 毫秒降至约 15 毫秒。

### 三个默认值互相作用

所有三个特性都假设其他的存在。PagedAttention 给调度器提供了细粒度的 KV 资源用于权衡。连续批处理需要这种细粒度资源，这样允许新序列进入时才不会强制全局重排。分块预填充是调度器在同一个 `RUNNING` 列表上做出的一个决策 — 它只是多了一个调度器策略，而不是一个独立系统。

你不需要知道每个标志。你需要知道调度器优化的是什么：在 KV 块预算约束下的有效吞吐量，受分块预填充切片的影响。

### 2026 年 v0.18.0 的坑

在 vLLM v0.18.0 中，你不能将 `--enable-chunked-prefill` 与草稿模型推测解码（`--speculative-model`）组合使用。文档记录的例外是 V1 调度器中的 N-gram GPU 推测解码。没有阅读发布说明就打开所有标志的团队会在启动时遇到运行时错误，而不是软性能下降。如果你的推测收益值得在分块预填充之上开启，重新考虑这个选择 — 2026 年的正确答案通常是不带分块预填充的 EAGLE-3，而非不编译的草稿模型加分块预填充。

### 你应该记住的数据

- Llama 3.3 70B FP8，H100 SXM5，128 并发，三项全开：2,200-2,400 tok/s。
- 同模型，默认 vLLM（无分块预填充）：~1,800 tok/s。
- 同模型，朴素 PyTorch 前向循环：~600 tok/s。
- PagedAttention 下生产负载的 KV 碎片化浪费：<4%。
- 混合负载下 P99 ITL：分块预填充下约 15 毫秒，无分块预填充约 50 毫秒。

### 调度器长什么样

```
while True:
    finished = [s for s in RUNNING if s.is_done()]
    for s in finished: release_blocks(s); RUNNING.remove(s)

    while WAITING and have_free_blocks_for(WAITING[0]):
        s = WAITING.pop(0)
        allocate_initial_blocks(s)
        RUNNING.append(s)

    # 在一次批处理中调度预填充块 + 解码
    batch = []
    for s in RUNNING:
        if s.in_prefill:
            batch.append(next_prefill_chunk(s))   # e.g. 512 tokens
        else:
            batch.append(decode_one_token(s))     # 1 token

    run_forward(batch)                            # 一次融合的 GPU 调用
```

`code/main.py` 正是标准库 Python 中的这个循环，带有假的 token 计数和假的前向延迟。运行它可以看到分块预填充如何在长预填充期间保持解码序列活跃。

```figure
tensor-parallel
```

## 使用它

`code/main.py` 模拟一个 vLLM 风格的调度器，具有可切换的特性。运行它以查看：

- `NAIVE` 模式：每次一个请求，无批处理。
- `STATIC` 模式：填充和等待，经典批处理。
- `CONTINUOUS` 模式：迭代层面的准入和释放。
- `CONTINUOUS + CHUNKED` 模式：预填充切片与解码交错。

输出显示总吞吐量（每虚拟秒 token 数）、TTFT 均值和 P99 ITL。`CONTINUOUS + CHUNKED` 行应在混合流量上占主导地位。

## 交付它

本课产出 `outputs/skill-vllm-scheduler-reader.md`。给定推理配置（批次大小、KV 内存利用率、分块预填充大小、推测配置），它生成一个调度器诊断，指出三个默认值中哪个是瓶颈以及需要调优什么。

## 练习

1. 运行 `code/main.py`。在混合短长请求的工作负载上比较 `STATIC` 和 `CONTINUOUS`。吞吐量差距从何而来 — 预填充效率、解码效率还是尾部延迟？
2. 修改玩具调度器添加 `--max-num-batched-tokens`。对运行 Llama 3.3 70B FP8 的 H100，正确值是多少？（提示：它是 KV 块大小和空闲块数量的函数，而非原始 HBM。）
3. 重新阅读 vLLM v0.18.0 发布说明。哪些标志组合是互斥的？列出它们。
4. 计算一个有 1,000 个请求、均值 1,500 输出 token、标准差 600 token 的追踪的 KV 缓存碎片化浪费，分别在 (a) 按序列 8192 最大值连续分配和 (b) PagedAttention 16 token 块下。
5. 用一段话解释为什么分块预填充有助于 P99 ITL 而非单独提升吞吐量。实践中吞吐量提升从何而来？

## 核心术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| PagedAttention | "KV 技巧" | 固定大小块的 KV 缓存分配器；碎片化 <4% |
| 块表 | "页表" | 每条序列从逻辑 token 位置到物理 KV 块的映射 |
| 连续批处理 | "动态批处理，但对" | 每次解码迭代做出的准入/释放决策 |
| 分块预填充 | "预填充拆分" | 将长预填充分解为 512 token 切片，与解码交错 |
| TTFT | "首 token 时间" | 预填充 + 队列 + 网络；以长提示时的预填充为主导 |
| ITL | "token 间延迟" | 连续解码 token 之间的时间；以批次大小为主导 |
| Goodput（有效吞吐） | "满足 SLO 的吞吐量" | 每个请求仍满足 TTFT 和 ITL 目标的 tok/sec |
| V1 调度器 | "新调度器" | vLLM 的 2026 年调度器；N-gram 推测解码是与分块预填充兼容的路径 |
| `--gpu-memory-utilization` | "内存旋钮" | 在权重和激活值之后为 KV 块预留的 HBM 比例 |

## 进一步阅读

- [vLLM 文档 — 推测解码](https://docs.vllm.ai/en/latest/features/spec_decode/) — 关于分块预填充和推测解码兼容性的官方来源。
- [vLLM 发布说明 (NVIDIA)](https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/index.html) — 2026 年发布节奏和特定版本行为。
- [vLLM 博客 — PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html) — 仍然定义了如何思考分配器的原始文章。
- [PagedAttention 论文 (arXiv:2309.06180)](https://arxiv.org/abs/2309.06180) — 碎片化分析和调度器设计。
- [Aleksa Gordic — Inside vLLM](https://www.aleksagordic.com/blog/vllm) — 带火焰图的详细 V1 调度器走读。
