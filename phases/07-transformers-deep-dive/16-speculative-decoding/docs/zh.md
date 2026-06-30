# 推测解码 — 草稿、验证、重复

> 自回归解码是串行的。每个 token 等待前一个。推测解码打破了链条：一个廉价的模型草拟 N 个 token，昂贵的模型在一次前向传播中验证所有 N 个。当草稿正确时，你为 N 代支付了一次大型前向传播。

**类型：** 构建
**语言：** Python
**前置条件：** 第七阶段 · 07（GPT 因果 LM），第七阶段 · 12（KV 缓存和 Flash Attention）
**时间：** 约 60 分钟

## 问题

一个 70B LLM 在 H100 上采样一个 token 大约需要 30 ms。一个 3B 草稿模型需要约 3 ms。如果我们让 3B 草拟提前 5 个 token，然后运行 70B *一次*来验证所有 5 个，总时间是 `5×3 + 30 = 45 ms`，对于最多 5 个被接受的 token——对比直接生成的 `5×30 = 150 ms`。这就是推测解码的全部卖点：用一点额外的 GPU 内存（草稿模型）换取 2–4 倍更低的解码延迟。

这个技巧必须保留分布。推测采样由 Leviathan 等人（2023）和 Chen 等人同时引入，保证输出序列与大模型自己产生的**分布完全相同**。没有质量权衡。只是更快。

四类草稿-验证器配对在 2026 年推理中占主导：

1. **标准推测（Leviathan 2023）。** 单独的草稿模型（例如 Llama 3 1B）+ 验证器（例如 Llama 3 70B）。
2. **Medusa（Cai 2024）。** 验证器上的多个解码头并行预测位置 `t+1..t+k`。没有单独的草稿模型。
3. **EAGLE 家族（Li 2024、2025）。** 轻量级草稿重用验证器的隐藏状态；比标准方案更接近的接受率；通常 3–4×。
4. **前瞻解码（Fu 2024）。** Jacobi 迭代；完全不需要草稿模型。自我推测。小众但无依赖。

每个 2026 年的生产推理技术栈默认都提供推测解码。vLLM、TensorRT-LLM、SGLang 和 llama.cpp 都至少支持标准版 + EAGLE-2。

## 概念

### 核心算法

给定一个验证器 `M_q` 和一个更便宜的草稿 `M_p`：

1. 设 `x_1..x_k` 是已经解码的前缀。
2. **草拟**：使用 `M_p` 自回归地提出 `d_{k+1}, d_{k+2}, ..., d_{k+N}`，带有草稿概率 `p_1..p_N`。
3. **并行验证**：在 `x_1..x_k, d_{k+1}, ..., d_{k+N}` 上运行 `M_q` 一次，得到位置 `k+1..k+N+1` 的验证器概率 `q_1..q_{N+1}`。
4. **从左到右接受/拒绝每个草稿 token**：对于每个 `i`，以概率 `min(1, q_i(d_i) / p_i(d_i))` 接受。
5. 在位置 `j` 首次拒绝时：从归一化的"残差"分布 `(q_j - p_j)_+` 中采样 `t_j`。`j` 之后的所有草稿被丢弃。
6. 接受所有 `N` 个时：从 `q_{N+1}` 采样一个额外的 token `t_{N+1}`（免费奖励 token）。

残差分布技巧是保持输出分布与 `M_q` 从头采样的完全相同这一数学洞见。

### 什么决定加速

设 `α` = 每个草稿 token 的预期接受率。设 `c` = 草稿与验证器成本比率。每一步：

- 朴素生成每个 token 调用 1 次大模型。
- 推测每 `(1 - α^{N+1}) / (1 - α) ≈ 1/(1-α)` 个 token 调用 1 次大模型，当 `α` 高时。

在 `α = 0.75` 和 `N = 5` 时的典型经验法则：大模型调用少 3×。草稿成本是 5× 廉价。总挂钟时间下降约 2.5×。

**α 取决于：**

- 草稿在多大程度上近似验证器。同一家族 / 相同训练数据显著提升 α。
- 解码策略。贪心草稿对照贪心验证器：高 α。温度采样：更难匹配；接受率下降。
- 任务类型。代码和结构化输出接受更多（可预测）；自由形式创意写作接受更少。

### Medusa — 无草稿模型的草拟

Medusa 用在验证器上的额外输出头替换草稿模型。在位置 `t`：

```
shared trunk → hidden h_t
    ├── head_0: 预测位置 t+1 的 token  （标准 LM head）
    ├── head_1: 预测位置 t+2 的 token
    ├── head_2: 预测位置 t+3 的 token
    ├── head_3: 预测位置 t+4 的 token
```

每个头输出自己的 logits。推理时你从每个头采样以获得候选序列，然后使用树注意力方案通过一次前向传播进行验证，该方案同时考虑所有候选续写。

优点：没有第二个模型。缺点：增加可训练参数；需要监督微调阶段（~1B token）；接受率比使用良好草稿的标准推测稍低。

### EAGLE — 通过重用隐藏状态实现更好的草拟

EAGLE-1/2/3（Li 等人，2024–2025）使草稿模型成为一个微小的 transformer（通常 1 层），它摄取验证器的最后层隐藏状态。因为草稿看到了验证器的特征表示，其预测与验证器的输出分布强相关。接受率从约 0.6（标准）上升到 0.85+。

EAGLE-3（2025）在候选续写上添加了树搜索。vLLM 和 SGLang 将 EAGLE-2/3 作为 Llama 3/4 和 Qwen 3 的默认推测路径发布。

### KV 缓存舞蹈

验证在一次前向传播中将 `N` 个草稿 token 输入验证器。这将验证器的 KV 缓存扩展了 `N` 个条目。如果一些草稿被拒绝，你必须将缓存回滚到接受的前缀长度。

生产实现（vLLM 的 `--speculative-model`、TensorRT-LLM 的 LookaheadDecoder）使用暂存 KV 缓冲区处理此问题。先写入，在接受时提交。概念上不难，但实现上有细节。

## 动手构建

参见 `code/main.py`。我们实现核心推测采样算法（拒绝步骤 + 残差分布）：

- 一个"大模型"，它是在手工编码分布上的确定性 softmax（这样我们可以解析地验证接受数学）。
- 一个"草稿模型"，它是大模型的扰动。
- 一个产生与直接采样相同边际分布的接受/拒绝循环。

### 步骤 1：拒绝步骤

```python
def accept_or_reject(q_prob, p_prob, draft_token, u):
    ratio = q_prob / p_prob if p_prob > 0 else float("inf")
    return u < min(1.0, ratio)
```

`u` 是均匀随机数。`q_prob` 是验证器对草拟 token 的概率。`p_prob` 是草稿模型的概率。Leviathan 定理是，这个 Bernoulli 决定，随后在拒绝时从残差分布采样，精确地保留了验证器的分布。

### 步骤 2：残差分布

```python
def residual_dist(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    return [r / s for r in raw]
```

从 `q` 中逐元素减去 `p`，将负值钳位为零，重新归一化。在任何拒绝时从此分布采样。

### 步骤 3：一次推测步骤

```python
def spec_step(prefix, q_model, p_model, N, rng):
    drafts = []
    p_probs = []
    ctx = list(prefix)
    for _ in range(N):
        p_dist = p_model(ctx)
        d = sample(p_dist, rng)
        drafts.append(d)
        p_probs.append(p_dist[d])
        ctx.append(d)

    q_dists = [q_model(prefix + drafts[:i]) for i in range(N + 1)]

    for i, d in enumerate(drafts):
        u = rng.random()
        q_prob = q_dists[i][d]
        p_prob = p_probs[i]
        if u < min(1.0, q_prob / p_prob if p_prob > 0 else float("inf")):
            prefix = prefix + [d]
        else:
            res = residual_dist(q_dists[i], p_model(prefix))
            prefix = prefix + [sample(res, rng)]
            return prefix
    prefix = prefix + [sample(q_dists[N], rng)]
    return prefix
```

五个接受 → 一个奖励 → 在一次验证器传递中产生六个 token。

### 步骤 4：测量接受率

在不同草稿质量水平上运行 10,000 次推测步骤。绘制接受率 vs 草稿和验证器分布之间的 KL 散度。你应该看到一个干净单调的关系。

### 步骤 5：验证分布等价性

经验上：推测循环产生的 token 直方图应该与从验证器直接采样产生的直方图匹配。这是实践中的 Leviathan 定理。卡方检验在采样误差内确认。

## 使用它

生产：

```bash
# vLLM with EAGLE
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model /models/llama-3.1-eagle-70b \
    --speculative-draft-tensor-parallel-size 1 \
    --num-speculative-tokens 5

# vLLM with vanilla draft model
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model meta-llama/Llama-3.2-1B-Instruct \
    --num-speculative-tokens 5
```

截至 2026 年中，TensorRT-LLM 拥有最快的 Medusa 路径。`faster-whisper` 为 Whisper-large 包裹了推测解码，使用一个小型草稿。

**选择草稿：**

| 策略 | 何时选择 | 加速 |
|----------|--------------|---------|
| 标准草稿（1B/3B Llama 家族） | 快速原型，无训练 | 1.8–2.3× |
| Medusa 头 | 你可以微调验证器 | 2–3× |
| EAGLE-2 / 3 | 生产，最大速度 | 3–4× |
| 前瞻 | 无草稿，无训练，无额外参数 | 1.3–1.6× |

**何时不进行推测解码：**

- 1–5 个 token 的单一序列生成。开销占主导。
- 极端创意 / 高温采样（α 下降）。
- 内存受限部署（草稿模型增加 VRAM）。

## 交付成果

参见 `outputs/skill-spec-decode-picker.md`。该技能为新的推理工作负载选择推测解码策略（标准 / Medusa / EAGLE / 前瞻）和调优参数（N、草稿温度）。

## 练习

1. **简单。** 运行 `code/main.py`。确认推测 token 分布在 50,000 个 token 上与验证器的直接采样分布匹配，卡方 p > 0.05。
2. **中等。** 将加速（每次大模型前向传播的 token 数）绘制为 `N` 的函数，针对 `α = 0.5, 0.7, 0.85`。确定每个 α 的最佳 `N`。（提示：每次验证调用的预期 token 数 = `(1 - α^{N+1}) / (1 - α)`。）
3. **困难。** 实现一个微型 Medusa：取第 14 课的毕业项目 GPT，添加 3 个额外的 LM 头，预测位置 t+2, t+3, t+4。在 tinyshakespeare 上使用联合多头损失进行训练。与通过截断相同模型制作的标准草稿比较接受率。
4. **困难。** 实现回滚：从 10 token 前缀 KV 缓存开始，输入 5 个草稿 token，模拟在位置 3 的拒绝。验证在下次迭代时你的缓存读取正确匹配"前缀 + 前 2 个接受的草稿"。

## 关键术语

| 术语 | 人们怎么说 | 它实际上的含义 |
|------|-----------------|-----------------------|
| 草稿模型 | "廉价的那个" | 一个提出候选 token 的较小模型；通常比验证器便宜 10–50×。 |
| 验证器 | "大的那个" | 我们保留其分布的目标模型；每次推测步骤运行一次。 |
| 接受率（α） | "草稿正确的频率" | 验证器接受草稿的每个 token 概率。通常 0.7–0.9。 |
| 残差分布 | "拒绝后备方案" | `(q - p)_+` 归一化；在拒绝时从此采样保留了验证器的分布。 |
| 奖励 token | "免费的那个" | 当所有 N 个草稿被接受时，从验证器的下一步分布再采样一个。 |
| Medusa | "无草稿推测" | 验证器上的多个 LM 头并行预测位置 t+1..t+k。 |
| EAGLE | "隐藏状态草稿" | 以验证器的最后层隐藏状态为条件的微型 transformer 草稿。 |
| 前瞻解码 | "Jacobi 迭代" | 使用不动点迭代的自我推测；无草稿模型。 |
| 树注意力 | "一次验证多个候选" | 同时考虑几个草稿续写的分支验证。 |
| KV 回滚 | "撤销被拒绝的草稿" | 暂存 KV 缓冲区；在接受时提交，在拒绝时丢弃。 |

## 延伸阅读

- [Leviathan、Kalman、Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — 核心算法和等价定理。
- [Chen 等人 (2023). Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) — 同时引入；干净的 Bernoulli 拒绝证明。
- [Cai 等人 (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — Medusa 论文；树注意力验证。
- [Li 等人 (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — EAGLE-1；以隐藏状态为条件的草稿。
- [Li 等人 (2024). EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858) — EAGLE-2；动态树深度。
- [Li 等人 (2025). EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840) — EAGLE-3。
- [Fu 等人 (2024). Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://arxiv.org/abs/2402.02057) — 前瞻，无草稿方法。
- [vLLM 文档 — 推测解码](https://docs.vllm.ai/en/latest/features/spec_decode.html) — 连接了所有四种策略的规范生产参考。
- [SafeAILab / EAGLE 参考实现](https://github.com/SafeAILab/EAGLE) — EAGLE-1/2/3 的参考代码。
