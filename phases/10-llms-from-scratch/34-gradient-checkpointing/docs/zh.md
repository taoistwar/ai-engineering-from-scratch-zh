# 梯度检查点与激活重新计算

> 反向传播保留每个中间激活。在 70B 参数和 128K 上下文中，每个 rank 的激活为 3 TB。检查点用 FLOPs 交换内存：重新计算而非保存。问题在于丢弃哪些段，答案不是"全部丢弃"。

**Type:** Build
**Languages:** Python (with numpy, optional torch)
**Prerequisites:** Phase 10 Lesson 04 (Pre-Training Mini-GPT), Phase 10 Lesson 05 (Scaling & Distributed)
**Time:** ~70 minutes

## 问题

训练一个 transformer 为每个层存储在反向传播中被求导的每个操作的输入：注意力输入、Q/K/V 投影、softmax 输出、FFN 输入、归一化输出和残差流。对于具有隐藏大小 `d`、序列长度 `L`、批次 `B` 的层，每层约 `12 * B * L * d` 个浮点数。

对于 `d=8192, L=8192, B=1`，每层 800 MB（BF16）。一个 64 层模型是 51 GB 激活——这还是在乘以微批大小之前，在加上注意力 softmax 中间值（每头 `L^2`）之前，在计入张量并行部分副本之前。

双向账单：BF16 权重加优化器状态可能适合 80GB，但激活把你推出去。梯度检查点（又名激活重新计算）是标准修复。丢弃大部分激活；在反向传播期间重做前向以取回它们。成本：额外 FLOPs。收益：内存按检查点段与总层数的比例下降。

天真地做，检查点每步消耗约 33% 更多前向传递 FLOPs。做得好——按照 Korthikanti 等人的"智能选择"进行选择性检查点——你以不到 5% 的 FLOP 开销节省 5 倍内存。而对于使用 FP8 矩阵乘法、FSDP 卸载和专家并行 MoE 的情况，这真的很重要：你既负担不起内存，也负担不起浪费的计算。

## 概念

### 反向传播实际需要什么

`output = layer(input)`。反向传播想要 `grad_input` 和 `grad_params`。要计算它们，它需要：

- `input`（用于计算线性层的 `grad_params = input.T @ grad_output`）
- 一些激活导数的中间值（ReLU/GELU/softmax 的导数依赖激活值）

前向传递自动将这些存储在 autograd 图中。每个 `tensor.retain_grad()` 和每个需要其输入的操作都保留一个引用。

### 天真的完全检查点

将网络分割为 `N` 个段。在前向期间，只存储每个段的*输入*。当反向传播需要中间值时，重新运行该段的前向传递以具体化它们，然后求导。

例子：32 层 transformer 分割为 32 个段，每段 1 层。

- 内存：32 个层输入（小）vs 32 *（每层激活量）（巨大）。
- 额外计算：每段 1 次额外前向，即总共约 33% 更多前向 FLOPs（因为反向传播是 2 倍前向，完整步骤变为 1 + 1 + 2 = 4 单位，而不是 1 + 2 = 3）。

这是 Chen 等人 2016 的原始配方：每 `sqrt(L)` 层一个检查点以平衡内存和计算。对于 L=64，即 8 个检查点。

### 选择性检查点（Korthikanti 2022）

并非所有激活的成本相同。注意力 softmax 输出是 `B*L*L*heads`，随序列长度*平方*增长。FFN 隐藏激活是 `B*L*4d`，线性增长。对于长序列，softmax 占主导。

选择性检查点保留存储成本低的激活（线性投影、残差），只重新计算昂贵的那些（注意力）。你付出最少的 FLOPs 重新计算，但节省了 O(L^2) 内存。

Megatron-Core 将此实现为"选择性"激活重新计算。用于大多数 2024+ 前沿训练运行。

### 卸载

重新计算的替代方案：在前向和后向之间将激活传输到 CPU RAM。需要 PCIe 带宽；当空闲带宽超过重新具体化成本时有益。混合策略很常见：对某些层检查点，对其他层卸载。

FSDP2 将卸载作为第一类选项提供。卸载在 GPU 受内存瓶颈但 CPU-GPU 传输有余量时闪耀。

### 重新计算成本模型

每 `k` 层进行天真检查点的每步 FLOPs（`L` 层中）：

```
flops_fwd_normal = L * f_layer
flops_bwd_normal = 2 * L * f_layer
flops_total_normal = 3 * L * f_layer

flops_fwd_ckpt = L * f_layer
flops_recompute = L * f_layer  # 段中每层一次额外前向
flops_bwd_ckpt = 2 * L * f_layer
flops_total_ckpt = 4 * L * f_layer
overhead = 4 / 3 - 1 = 0.33 = 33%
```

使用选择性检查点，你只重新计算注意力 kernel，而非整个层：

```
flops_recompute_selective = L * f_attention ~= L * f_layer * 0.15
overhead_selective = (3 + 0.15) / 3 - 1 = 0.05 = 5%
```

### 内存节省模型

每层激活量：`A`。对于 `L` 层，总激活内存：`L * A`。

完全检查点（段大小 1）：只存储 `L * input_volume`（对于标准 transformer 约 `L * 1/10 A`）。节省约 `9 * L * A * 1/10`。

每 `k` 层检查点：存储 `L/k * A` 加上活动段内的 `k-1` 层份额。

在 `k = sqrt(L)` 时，内存和重新计算成本都随 `sqrt(L)` 扩展——均匀成本层的最优权衡。

### 何时不检查点

- 流水线阶段的最内层已在飞行中。它们无论如何都需要完成。
- 第一层和最后一层，如果它们主导阶段的 compute（在 transformer 中罕见）。
- 已经使用 FlashAttention 的注意力 kernel——Flash 已经快速重新计算 softmax，因此额外的层级检查点在其上增加很少。

### 实现模式

1. **函数包装器：** 将段包装在 `torch.utils.checkpoint.checkpoint(fn, input)` 中。PyTorch 只存储 `input`，在反向传播时重新计算其他一切。

2. **基于装饰器：** 标记层为可检查点；训练器在配置时决定哪些段被包装。

3. **手动显式重新计算：** 自己写反向传播传递，调用自定义的 `recompute_forward`，用存储的输入重复前向。

三种都给出相同的功能结果。包装器是标准惯用法。

### 与 TP / PP / FP8 的交互

- **张量并行：** 检查点输入必须在重新计算时被收集或重新分散；处理通信成本。
- **流水线并行：** 典型模式是检查点每个流水线阶段的前向，使逆序微批次可以复用激活内存。
- **FP8 重新计算：** 重新计算期间更新的 amax 历史必须匹配原始前向的，否则 FP8 缩放会漂移。大多数框架快照缩放因子。

## 构建它

### 步骤 1：一个具有段的玩具模型

```python
import numpy as np


def linear_forward(x, w, b):
    return x @ w + b


def relu(x):
    return np.maximum(x, 0)


def layer_forward(x, w1, b1, w2, b2):
    h = relu(linear_forward(x, w1, b1))
    return linear_forward(h, w2, b2)


def model_forward(x, params):
    activations = [x]
    h = x
    for w1, b1, w2, b2 in params:
        h = layer_forward(h, w1, b1, w2, b2)
        activations.append(h)
    return h, activations
```

### 步骤 2：需要所有激活的天真反向传播

```python
def model_backward(grad_output, activations, params):
    grads = [None] * len(params)
    g = grad_output
    for i in range(len(params) - 1, -1, -1):
        w1, b1, w2, b2 = params[i]
        x_in = activations[i]
        h_pre = linear_forward(x_in, w1, b1)
        h = relu(h_pre)
        gh = g @ w2.T
        gw2 = h.T @ g
        gb2 = g.sum(axis=0)
        g_pre = gh * (h_pre > 0)
        gx = g_pre @ w1.T
        gw1 = x_in.T @ g_pre
        gb1 = g_pre.sum(axis=0)
        grads[i] = (gw1, gb1, gw2, gb2)
        g = gx
    return g, grads
```

### 步骤 3：每 k 检查点内存

```python
def model_forward_checkpointed(x, params, k=4):
    saved_inputs = [x]
    h = x
    for i, (w1, b1, w2, b2) in enumerate(params):
        h = layer_forward(h, w1, b1, w2, b2)
        if (i + 1) % k == 0:
            saved_inputs.append(h)
    return h, saved_inputs


def model_backward_checkpointed(grad_output, saved_inputs, params, k=4):
    grads = [None] * len(params)
    g = grad_output
    segments = [(j * k, min((j + 1) * k, len(params))) for j in range(len(saved_inputs))]
    for seg_idx in range(len(saved_inputs) - 1, -1, -1):
        start, end = segments[seg_idx]
        if start >= end:
            continue
        x_in = saved_inputs[seg_idx]
        _, seg_acts = model_forward(x_in, params[start:end])
        g, seg_grads = model_backward(g, seg_acts, params[start:end])
        for j, gr in enumerate(seg_grads):
            grads[start + j] = gr
    return g, grads
```

### 步骤 4：成本模型

```python
def checkpoint_cost(n_layers, segment_size, flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }


def selective_checkpoint_cost(n_layers, attention_fraction=0.15,
                              flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * attention_fraction * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }
```

### 步骤 5：内存估算器

```python
def activation_memory_mb(n_layers, hidden=8192, seq=8192,
                        batch=1, bytes_per_value=2):
    per_layer = 12 * batch * seq * hidden * bytes_per_value
    return n_layers * per_layer / 1e6


def memory_after_checkpoint(n_layers, segment_size, hidden=8192,
                           seq=8192, batch=1, bytes_per_value=2):
    n_seg = max(1, n_layers // segment_size)
    saved = (n_seg + segment_size) * 1 * batch * seq * hidden * bytes_per_value
    return saved / 1e6
```

### 步骤 6：最优段大小

```python
def optimal_segment(n_layers):
    return int(round(np.sqrt(n_layers)))
```

### 步骤 7：选择性检查点决策

```python
def should_recompute(layer_type, activation_bytes, recompute_flops_ratio):
    if layer_type == "attention" and activation_bytes > 100 * 1e6:
        return True
    if layer_type == "ffn" and activation_bytes > 500 * 1e6:
        return recompute_flops_ratio < 0.1
    return False
```

## 使用它

- **torch.utils.checkpoint**：`from torch.utils.checkpoint import checkpoint`——PyTorch 中的经典包装器。包装一个函数；只存储输入，在反向传播时重新计算。
- **Megatron-Core 激活重新计算**：支持 `selective`、`full` 和 `block` 模式。2024+ 前沿训练的标准。
- **FSDP2 卸载**：`module.to_empty(device="cpu")` 配合 FSDP2 中的 `offload_policy`，将激活分片到 CPU 而非重新计算。
- **DeepSpeed ZeRO-Offload**：优化器状态和激活的 CPU 卸载，补充检查点。

## 产出

本课产出 `outputs/prompt-activation-recompute-policy.md`——一个提示，接收你的模型配置（层数、hidden、seq、batch）和可用 GPU 内存，并发出每层重新计算策略（none / selective / full / offload）。

## 练习

1. 验证正确性。运行 `model_forward` + `model_backward`（全激活）vs `model_forward_checkpointed` + `model_backward_checkpointed`（段）。参数梯度必须与机器精度相同。

2. 将段大小 `k` 从 1 扫描到 `L`。绘制 FLOP 开销和内存。找到曲线的拐点。

3. 实现选择性检查点：存储注意力模块输入但不存储其中间值。为 seq=8192 的 32 层模型测量 FLOP 开销 vs 全层检查点。

4. 添加卸载。将段输入保存到模拟的"CPU 缓冲区"（一个单独的列表）。将"PCIe 带宽"测量为 bytes/time，找到卸载和重新计算的盈亏平衡点。

5. 分别在有和没有 `torch.utils.checkpoint` 的情况下基准测试一个真实的 PyTorch transformer。测量内存（通过 `torch.cuda.max_memory_allocated`）和步骤时间。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|----------------------|
| 梯度检查点 | "通过重做前向节省内存" | 只存储段输入；在反向传播期间重新计算中间值以获取支持梯度的张量 |
| 激活重新计算 | "与检查点相同" | 同一技术的面向 HPC 的名称 |
| 段大小（k）| "每个检查点多少层" | 其激活被丢弃并一起重新具体化的层数 |
| 选择性检查点 | "Korthikanti 的技巧" | 只重新计算存储昂贵的激活（注意力 softmax）；保留便宜的 |
| 完全检查点 | "天真版本" | 在每个段中重新计算每层的激活 |
| 块检查点 | "粗粒度" | 检查点整个 transformer 块；最大粒度 |
| FLOP 开销 | "计算税" | 每步额外 FLOPs = (recompute FLOPs) / (fwd + bwd FLOPs)；天真 33%，选择性 5% |
| 激活卸载 | "传输到 CPU" | 在前向→后向之间将激活移动到 CPU RAM；重新计算的替代方案 |
| sqrt-L 规则 | "经典最优" | 对于均匀成本层，最优检查点间距为 sqrt(L) 层 |
| 注意力 softmax 量 | "O(L^2) 问题" | L^2 * heads * batch 个浮点数；在长上下文上主导激活内存 |

## 延伸阅读

- [Chen et al., 2016 -- "Training Deep Nets with Sublinear Memory Cost"](https://arxiv.org/abs/1604.06174) —— 形式化梯度检查点的原始论文
- [Korthikanti et al., 2022 -- "Reducing Activation Recomputation in Large Transformer Models"](https://arxiv.org/abs/2205.05198) —— 选择性激活重新计算和正式的成本分析
- [Pudipeddi et al., 2020 -- "Training Large Neural Networks with Constant Memory using a New Execution Algorithm"](https://arxiv.org/abs/2002.05645) —— 通过反向模式重新具体化的替代恒定内存方法
- [Ren et al., 2021 -- "ZeRO-Offload: Democratizing Billion-Scale Model Training"](https://arxiv.org/abs/2101.06840) —— 规模化激活卸载
- [PyTorch torch.utils.checkpoint docs](https://pytorch.org/docs/stable/checkpoint.html) —— 标准 API
- [Megatron-Core activation recomputation documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/memory_optimizations.html) —— 选择性、完全和块模式
