# 专家混合（MoE）

> 一个密集的 70B transformer 为每个 token 激活所有参数。一个 671B 的 MoE 每个 token 只激活 37B，并在每个基准上击败它。稀疏性是这十年最重要的扩展思想。

**类型：** 构建
**语言：** Python
**前置条件：** 第七阶段 · 05（完整 Transformer），第七阶段 · 07（GPT）
**时间：** 约 45 分钟

## 问题

密集 transformer 在推理时的 FLOPs 等于其参数数量（正向传播再乘 2）。扩大密集模型，每个 token 都支付全部账单。到 2024 年，前沿正在遇到计算墙：要有意义地变得更智能，你需要每个 token 呈指数级增长的计算量。

专家混合打破了这种联系。将每个 FFN 替换为 `E` 个独立专家 + 一个为每个 token 选择 `k` 个专家的路由器。总参数 = `E × FFN_size`。每个 token 的活跃参数 = `k × FFN_size`。典型的 2026 年配置：`E=256`，`k=8`。存储随 `E` 扩展，计算随 `k` 扩展。

2026 年的前沿几乎完全是 MoE：DeepSeek-V3（671B 总参数 / 37B 活跃参数）、Mixtral 8×22B、Qwen2.5-MoE、Llama 4、Kimi K2、gpt-oss。在 Artificial Analysis 的独立排行榜上，前 10 名开源模型都是 MoE。

## 概念

![MoE 层：路由器为每个 token 从 E 个中选择 k 个专家](../assets/moe.svg)

### FFN 替换

密集 transformer 模块：

```
h = x + attn(norm(x))
h = h + FFN(norm(h))
```

MoE 模块：

```
h = x + attn(norm(x))
scores = router(norm(h))              # (N_tokens, E)
top_k = argmax_k(scores)              # 每个 token 选择 E 中的 k 个
h = h + sum_{e in top_k}(
        gate(scores[e]) * Expert_e(norm(h))
    )
```

每个专家是一个独立的 FFN（通常是 SwiGLU）。路由器是一个单一的线性层。每个 token 选择自己的 `k` 个专家，并获得它们输出的门控混合。

### 负载均衡问题

如果路由器将 90% 的 token 通过专家 3，其他专家就饿死了。已尝试三种修复方案：

1. **辅助负载均衡损失**（Switch Transformer、Mixtral）。添加与专家使用方差成正比的惩罚。有效，但添加了超参数和第二个梯度信号。
2. **专家容量 + token 丢弃**（早期 Switch）。每个专家最多处理 `C × N/E` 个 token；溢出的 token 跳过该层。损害质量。
3. **无辅助损失均衡**（DeepSeek-V3）。添加一个学习的每个专家偏置，改变路由器的 top-k 选择。偏置在训练损失之外更新。不对主目标添加惩罚。2024 年的大解锁。

DeepSeek-V3 的方法：每个训练步骤后，对每个专家检查其使用是否高于或低于目标。用 `±γ` 推动偏置。选择使用 `scores + bias`。用于门控的专家概率是未改变的原始 `scores`。将路由与表达解耦。

### 共享专家

DeepSeek-V2/V3 还将专家分为*共享*和*路由*。每个 token 通过所有共享专家。路由专家通过 top-k 选择。共享专家捕获共同知识；路由专家专门化。V3 运行 1 个共享专家加上 256 个路由专家中的 top-8。

### 细粒度专家

经典 MoE（GShard、Switch）：每个专家与完整 FFN 一样宽。`E` 小（8–64），`k` 小（1–2）。

现代细粒度 MoE（DeepSeek-V3、Qwen-MoE）：每个专家更窄（1/8 FFN 大小）。`E` 大（256+），`k` 更大（8+）。相同的总参数，但组合扩展快得多。`C(256, 8) = 400 万亿` 种可能的"专家"组合。质量上升，延迟保持不变。

### 成本概况

每个 token，每层：

| 配置 | 活跃参数/token | 总参数 |
|--------|-----------------------|--------------|
| Mixtral 8×22B | ~39B | 141B |
| Llama 3 70B（密集） | 70B | 70B |
| DeepSeek-V3 | 37B | 671B |
| Kimi K2（MoE） | ~32B | 1T |

DeepSeek-V3 在几乎每个基准上击败 Llama 3 70B（密集），同时每个 token 做**更少的活跃 FLOPs**。更多参数 = 更多知识。更多活跃 FLOPs = 每个 token 更多计算。MoE 将它们解耦。

### 陷阱：内存

所有专家都驻留在 GPU 上，无论哪些被激活。一个 671B 模型需要 ~1.3 TB 的 VRAM 用于 fp16 权重。前沿 MoE 部署需要专家并行——在不同 GPU 上分片专家，通过网络路由 token。延迟由 all-to-all 通信主导，而不是矩阵乘法。

## 动手构建

参见 `code/main.py`。一个紧凑的纯标准库 MoE 层，包含：

- `n_experts=8` 个 SwiGLU 风格的专家（每个一个线性层，用于演示）
- top-k=2 路由
- softmax 归一化门控权重
- 通过每个专家偏置实现的无辅助损失均衡

### 步骤 1：路由器

```python
def route(hidden, W_router, top_k, bias):
    scores = [sum(h * w for h, w in zip(hidden, W_router[e])) for e in range(len(W_router))]
    biased = [s + b for s, b in zip(scores, bias)]
    top_idx = sorted(range(len(biased)), key=lambda i: -biased[i])[:top_k]
    # 对所选专家的原始分数进行 softmax
    chosen = [scores[i] for i in top_idx]
    m = max(chosen)
    exps = [math.exp(c - m) for c in chosen]
    s = sum(exps)
    gates = [e / s for e in exps]
    return top_idx, gates
```

偏置影响选择，不影响门控权重。这就是 DeepSeek-V3 的技巧——偏置纠正负载不均衡而不操纵模型的预测。

### 步骤 2：通过路由器运行 100 个 token

跟踪哪些专家被激活了多少次。没有偏置，使用是倾斜的。通过偏置更新循环（对过度使用的专家 `-γ`，对使用不足的专家 `+γ`），使用在几次迭代内收敛到均匀分布。

### 步骤 3：参数计数比较

打印 MoE 配置的"密集等效"。DeepSeek-V3 形状：256 路由 + 1 共享，8 活跃，d_model=7168。总参数数量令人瞠目。活跃数量是密集 Llama 3 70B 的七分之一。

## 使用它

HuggingFace 加载：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x22B-v0.1")
```

2026 年生产推理：vLLM 原生支持 MoE 路由。SGLang 具有最快的专家并行路径。两者都自动处理 top-k 选择和专家并行。

**何时选择 MoE：**
- 你想以每个 token 更低的推理成本获得前沿质量。
- 你有 VRAM / 专家并行基础设施。
- 你的工作负载是 token 密集型（聊天、代码），而不是上下文密集型（长文档）。

**何时不选择 MoE：**
- 边缘部署——你为任何活跃 FLOP 支付全部存储成本。
- 延迟关键的单一用户服务——专家路由增加了开销。
- 小型模型（<7B）——MoE 的质量优势仅出现在计算阈值之上（~6B 活跃参数）。

## 交付成果

参见 `outputs/skill-moe-configurator.md`。该技能根据参数预算、训练 token 和部署目标，为新的 MoE 选择 E、k 和共享专家布局。

## 练习

1. **简单。** 运行 `code/main.py`。观察无辅助损失偏置更新如何在 50 次迭代内均衡专家使用。
2. **中等。** 用基于哈希的路由器替换学习的路由器（确定性，无学习）。比较质量和均衡。为什么学习的路由器更好？
3. **困难。** 实现 GRPO 风格的"推出匹配路由"（DeepSeek-V3.2 技巧）：记录推理期间哪些专家被激活，在梯度计算期间强制执行相同的路由。在玩具策略梯度设置上测量效果。

## 关键术语

| 术语 | 人们怎么说 | 它实际上的含义 |
|------|-----------------|-----------------------|
| 专家 | "许多 FFN 中的一个" | 一个独立的前馈网络；专用于 FFN 计算稀疏切片的参数。 |
| 路由器 | "门控" | 一个微小的线性层，为每个 token 对每个专家打分；top-k 选择。 |
| Top-k 路由 | "每个 token k 个活跃专家" | 每个 token 的 FFN 计算恰好通过 k 个专家，由门控加权。 |
| 辅助损失 | "负载均衡惩罚" | 惩罚偏斜的专家使用的额外损失项。 |
| 无辅助损失 | "DeepSeek-V3 的技巧" | 仅通过路由器选择上的每个专家偏置来均衡；无额外梯度。 |
| 共享专家 | "始终开启" | 每个 token 都通过的额外专家；捕获共同知识。 |
| 专家并行 | "按专家分片" | 将不同的专家分布到不同的 GPU；通过网络路由 token。 |
| 稀疏性 | "活跃参数 < 总参数" | 比率 `k × expert_size / (E × expert_size)`；DeepSeek-V3 为 37/671 ≈ 5.5%。 |

## 延伸阅读

- [Shazeer 等人 (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) — 这个思想。
- [Fedus、Zoph、Shazeer (2022). Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) — Switch，经典 MoE。
- [Jiang 等人 (2024). Mixtral of Experts](https://arxiv.org/abs/2401.04088) — Mixtral 8×7B。
- [DeepSeek-AI (2024). DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — MLA + 无辅助损失 MoE + MTP。
- [Wang 等人 (2024). Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664) — 基于偏置的均衡论文。
- [Dai 等人 (2024). DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066) — 本课路由器使用的细粒度 + 共享专家分割。
- [Kim 等人 (2022). DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training](https://arxiv.org/abs/2201.05596) — 原始共享专家论文。
