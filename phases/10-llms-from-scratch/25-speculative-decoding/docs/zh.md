# 推测解码与 EAGLE

> 一个前沿 LLM 生成一个 token 需要对数十亿参数进行完整的前向传递。该前向传递被大量过量配置：大多数时候一个小得多的模型可以正确猜测接下来的 3-5 个 token，大模型只需要*验证*这个猜测。当猜测正确时，你以一个的价格获得 5 个 token。推测解码（Leviathan 等人 2023）使其精确，而 EAGLE-3（2025）将接受率推至每次验证约 4.5 个 token——在匹配的输出分布下实现 4-5x 加速。

**Type:** Build
**Languages:** Python (with numpy)
**Prerequisites:** Phase 10 Lesson 12 (Inference Optimization), Phase 10 Lesson 04 (Pre-training Mini-GPT)
**Time:** ~75 minutes

## 问题

在 H100 上 70B 级别模型的解码吞吐量通常是每秒 40-80 个 token。每个 token 需要一个从前读取所有模型权重的完整前向传递。你不能不改变模型输出而缩小模型。你不能在不超出内存的情况下增加批次大小。你被困住了——除非你能让模型每个前向传递输出超过一个 token。

自回归生成看起来本质上是串行的：`x_{t+1} = sample(p(· | x_{1:t}))`。但是有一个并发机会。如果你有一个廉价的预测器说"接下来的 4 个 token 大概是 [a, b, c, d]"，你可以在大模型的**单次前向传递**中验证所有 5 个位置，并接受最长匹配前缀。

Leviathan、Kalai、Matias（2023，"Fast Inference from Transformers via Speculative Decoding"）通过一个巧妙的接受/拒绝规则使其精确，该规则保留了目标模型的采样分布。相同的输出分布，2-4× 更快。

## 概念

### 双模型设置

- **目标模型** `M_p`：大、慢、高质量模型，你实际想要从中采样。分布：`p(x)`。
- **草稿模型** `M_q`：小、快、低质量模型。分布：`q(x)`。5-30× 更小。

每步：

1. 草稿模型自回归地提出 `K` 个 token：`x_1, x_2, ..., x_K ~ q`。
2. 目标模型在所有 `K+1` 位置上并行运行一次前向传递，为每个被提议的 token 产生 `p(x_k)`。
3. 通过下面修改的拒绝采样规则从左到右接受/拒绝每个 token。接受最长匹配前缀。
4. 如果任何 token 被拒绝，从修正分布中采样替换并停止。否则从 `p(· | x_1...x_K)` 采样一个奖励 token。

如果草稿完美匹配目标，每个目标前向得到 K+1 个 token。如果草稿在位置 1 就错了，你只得到 1 个 token。

### 精确性规则

推测解码**在分布上可证明等价于从 p 采样**。拒绝规则：

```
For each drafted token x_t:
    r ~ Uniform(0, 1)
    if r < p(x_t) / q(x_t):
        accept x_t
    else:
        sample replacement from residual: (p - q)+ / ||(p - q)+||_1
        stop
```

其中 `(p - q)+` 表示逐点差的正部分。当草稿和目标一致时（`p ≈ q`），接受率接近 1。当它们不一致时，残差分布被构造使整体样本仍然是精确的 `p`。

**贪心情况。** 对于 temperature=0 采样，只需检查 `argmax(p) == x_t`。如果是，接受；如果否，输出 `argmax(p)` 并停止。

### 期望加速比

如果草稿模型的 token 级接受率为 `α`，每个目标前向传递产生的期望 token 数为：

```
E[tokens] = (1 - α^{K+1}) / (1 - α)        # K = 草稿长度, α in [0, 1]
```

在 `α = 0.8, K = 4` 下：`(1 - 0.8^5)/(1 - 0.8) = 3.36` 个 token 每个前向。单次目标前向大约消耗 `cost_q * K + cost_p`（K 次草稿步骤加一次目标验证）。如果 `cost_p >> cost_q * K`，加速比在吞吐量上是 `3.36× / 1 = 3.36×`。

唯一真正的参数是 `α`，完全取决于草稿-目标的对齐。一个好的草稿就是一切。

### 训练草稿：蒸馏

一个随机的小模型产生糟糕的草稿。标准配方是从目标蒸馏：

1. 选择一个小架构（70B 目标约 1B，7B 目标约 500M）。
2. 在一个大文本语料库上运行目标模型；存储其下一 token 分布。
3. 用 KL 散度对目标分布（而非对真实 token）训练草稿。

结果：`α` 在编程上通常 0.6-0.8，在自然语言对话上 0.7-0.85。生产中加速比 2-3×。

### EAGLE：树草案 + 特征复用

Li、Wei、Zhang、Zhang（2024，"EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"）观察到标准推测解码中的两个低效：

1. 草稿执行 K 个串行步骤，每一步全栈。但草稿可以从最近的验证中复用目标的特征（隐藏状态）——目标已经计算了丰富的表示，而草稿正在从零重新推导它们。
2. 草稿输出一个线性链条。如果草稿能输出一个候选*树*（每个节点多个猜测），目标的单次前向传递可以通过一个树注意力掩码并行验证多个候选路径，并选择最长被接受的分支。

EAGLE-1 变化：
- 草稿输入 = 目标在位置 t 的最终隐藏状态，而非原始 token。
- 草稿架构 = 1 个 transformer decoder 层（非独立的小模型）。
- 输出 = 每深度 K = 4-8 个候选的树，深度 4-6。

EAGLE-2（2024）添加动态树拓扑：树在草稿不确定的地方变宽，在确信的地方保持窄。在不增加验证成本的情况下提高 `α_effective`。

EAGLE-3（Li 等人 2025，"EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test"）移除固定的顶层特征依赖，并用一个新的"测试时模拟"损失训练草稿——草稿在与目标测试时分布匹配的输出上训练，而非教师强迫的训练分布。接受率从 0.75（EAGLE-2）上升到 0.82（EAGLE-3），平均每个验证的 token 数从 3.0 到 4.5。

### 树注意力验证

当草稿输出一个树时，目标模型使用**树注意力掩码**在单次前向传递中验证它——一个编码树拓扑而非纯线的因果掩码。每个 token 仅关注其在树中的祖先。验证 pass 仍然是一次前向，一次矩阵乘法；拓扑掩码仅花费少量额外 KV 条目。

```
        root
       /    \
      a      b
     / \    / \
    c  d   e   f
```

如果 `a, b` 是竞争的第一 token 候选，`c, d, e, f` 是第二 token 候选，所有六个位置在单次前向传递中被验证。输出是沿任何被接受路径的最长前缀。

### 何时胜出，何时不胜

**胜出：**
- 具有可预测文本的对话/补全（代码、常见英语、结构化输出）。`α` 高。
- 解码时 GPU 计算未充分利用的设置（内存受限阶段）。树草案使用可用的 FLOPs。

**失败/无胜出：**
- 高随机性输出（高温创意写作）。`α` 掉向 `1/|vocab|`。
- 极高并发批量服务——批处理已经填满 FLOPs，树验证的空间很小。
- 非常小的目标模型，草稿并不比目标小多少。

生产部门通常在对话上报告 2-3× 挂钟加速，代码生成上 3-5×，创意写作上接近零。

```figure
speculative-decoding
```

## 构建它

`code/main.py`：

- 一个参考实现 `speculative_decode(target, draft, prompt, K, temperature)`，实现了精确拒绝规则并验证它保留了目标分布（经验 KL < 0.01 vs 普通目标采样）。
- 一个 EAGLE 风格树草案器，构建具有 top-p 分支的深度 K 树。
- 一个树注意力掩码构建器，为验证器产生正确的因果模式。
- 一个接受率测试框架，在一个小型 LM 上运行两者（从 GPT-2-medium 目标蒸馏一个 GPT-2-small）。

```python
def speculative_step(p_target, q_draft, K, temperature=1.0):
    """一轮推测解码。返回被接受的 token 列表。"""
    # 1. 草案 K 个 token
    draft_tokens = []
    q_probs = []
    state = draft_state_init()
    for _ in range(K):
        probs = softmax(q_draft(state) / temperature)
        t = np.random.choice(len(probs), p=probs)
        draft_tokens.append(t)
        q_probs.append(probs[t])
        state = draft_step(state, t)

    # 2. 目标在每个草案位置 + 1 个额外上计算 p
    p_probs_all = target_forward_batched(p_target, draft_tokens, temperature)

    # 3. 从左到右接受/拒绝
    accepted = []
    for k, tok in enumerate(draft_tokens):
        r = np.random.uniform()
        if r < p_probs_all[k][tok] / q_probs[k]:
            accepted.append(tok)
        else:
            residual = np.maximum(p_probs_all[k] - q_probs[k], 0)
            residual /= residual.sum()
            accepted.append(np.random.choice(len(residual), p=residual))
            return accepted
    # 4. 所有 K 个被接受 → 从目标采样奖励 token
    accepted.append(np.random.choice(len(p_probs_all[-1]), p=p_probs_all[-1]))
    return accepted
```

## 使用它

- **vLLM** 和 **SGLang** 搭载第一类推测解码。标志：`--speculative_model`, `--num_speculative_tokens`。EAGLE-2/3 支持通过 `--spec_decoding_algorithm eagle` 标志。
- **NVIDIA TensorRT-LLM** 原生支持 Medusa 和 EAGLE 树。
- **参考草稿模型**：`Qwen/Qwen3-0.6B-spec`（为 Qwen3-32B 起草），`meta-llama/Llama-3.2-1B-Instruct-spec`（为 70B 起草）。
- **Medusa heads**（Cai 等人 2024，"Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"）：不在草稿模型上，而是向目标自身添加 K 个并行预测头。部署更简单，接受率略低于 EAGLE。

## 产出

本课产出 `outputs/skill-speculative-tuning.md`——一个技能，剖析目标模型的工作负载并选择：草稿模型、K（草稿长度）、树宽度、温度，以及何时回退到普通解码。

## 练习

1. 实现精确拒绝规则并经验性验证它。通过 `speculative_decode` 和通过普通目标采样运行 10K 样本；计算两个输出分布之间的 TV 距离。应 < 0.01。

2. 计算加速公式。给定固定的 `α` 和 `K`，绘制每个目标前向的期望 token 数。为 α ∈ {0.5, 0.7, 0.9} 找到最优 K。

3. 训练一个小草稿。取一个 124M GPT-2 目标，用 KL 损失在 100M token 上蒸馏一个 30M GPT-2 草稿。在留出文本上测量 `α`。预期：0.6-0.7。

4. 实现 EAGLE 风格树草案。不采用链条，让草稿在每个深度输出 top-3 分支。构建树注意力掩码。验证目标接受最长正确分支。

5. 测量失败模式。在 temperature=1.5（高随机性）下运行推测解码。展示 α 崩塌，算法因草稿开销比普通解码更慢。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| 目标模型 | "大模型" | 你想要从中采样的慢、高质量模型（p 分布） |
| 草稿模型 | "推测器" | 小、快预测器（q 分布）；5-30x 更小 |
| K / 草稿长度 | "前瞻" | 每次验证 pass 的推测 token 数 |
| α / 接受率 | "命中率" | 草稿提案被接受的每 token 概率 |
| 精确拒绝规则 | "接受测试" | r < p/q 比较，保留目标分布 |
| 残差分布 | "修正后的 p-q" | (p - q)+ / ||(p - q)+||_1，拒绝时从中采样的分布 |
| 树草案 | "分支推测" | 草稿输出一个候选树，用树结构注意力掩码在一次 pass 中验证 |
| 树注意力掩码 | "拓扑掩码" | 编码树拓扑的因果掩码，使每个节点仅关注其祖先 |
| Medusa heads | "并行头" | 目标自身的 K 个额外预测头；无独立草稿模型 |
| EAGLE 特征复用 | "隐藏状态草稿" | 草稿输入是目标的最后隐藏状态，而非原始 token，缩小草稿 |
| 测试时模拟损失 | "EAGLE-3 训练" | 在与目标测试时分布匹配的输出上训练草稿，非教师强迫 |

## 延伸阅读

- [Leviathan, Kalai, Matias, 2023 — "Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/abs/2211.17192) —— 精确拒绝规则和理论加速分析
- [Chen, Borgeaud, Irving et al., 2023 — "Accelerating Large Language Model Decoding with Speculative Sampling"](https://arxiv.org/abs/2302.01318) —— DeepMind 的同期推测采样论文
- [Cai, Li, Geng, Wang, Wang, Zhu, Dao, 2024 — "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"](https://arxiv.org/abs/2401.10774) —— 草稿模型的并行头替代方案
- [Li, Wei, Zhang, Zhang, 2024 — "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"](https://arxiv.org/abs/2401.15077) —— 特征复用和树草案
- [Li et al., 2024 — "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees"](https://arxiv.org/abs/2406.16858) —— 动态树拓扑
- [Li et al., 2025 — "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test"](https://arxiv.org/abs/2503.01840) —— 训练时测试时匹配
- [Fu, Haotian, Peng et al., 2024 — "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding"](https://arxiv.org/abs/2402.02057) —— Jacobi/lookahead 解码，无推测器替代方案
