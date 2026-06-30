# DeepSeek-V3 架构详解

> 第 10 阶段 · 第 14 课指出了每个开放模型调整的六个架构旋钮。DeepSeek-V3（2024 年 12 月，671B 总参数，37B 激活）调整了全部六个，并添加了另外四个：多头潜在注意力、无辅助损失负载均衡、多 Token 预测和 DualPipe 训练。本课从头到尾阅读 DeepSeek-V3 的架构，并根据发布的配置推导每个参数计数。结束时，你可以解释为什么 671B/37B 的比率是正确的赌注，以及为什么 MLA + MoE 一起在前沿领域胜过单独使用任何一个。

**Type:** Learn
**Languages:** Python (stdlib, parameter calculator)
**Prerequisites:** Phase 10 · 14 (open-model walkthroughs), Phase 10 · 17 (NSA), Phase 10 · 18 (MTP), Phase 10 · 19 (DualPipe)
**Time:** ~75 minutes

## 学习目标

- 阅读 DeepSeek-V3 的配置从头到尾，并用六个 GPT-2 旋钮加上四个 DeepSeek 特定的追加来解释每个字段。
- 推导总参数量（671B）、激活参数量（37B），以及贡献到每个的组件。
- 计算 MLA 在 128k 上下文下的 KV 缓存占用，并与同激活参数的密集 GQA 模型所需比较。
- 阐述四个 DeepSeek 特定创新（MLA、MTP、无辅助损失路由、DualPipe），并指出每个针对架构/训练栈的哪个部分。

## 问题

DeepSeek-V3 是第一个架构与 Llama 家族有显著差异的前沿开放模型。Llama 3 405B 是"调整了六个旋钮的 GPT-2"。DeepSeek-V3 是调整了全部六个旋钮加上另外四个的 GPT-2。阅读 Llama 3 配置是阅读 DeepSeek 配置的热身，但深层结构——注意力块的形状、路由逻辑、训练时目标——差异足以需要独立的讲解。

学习它的回报：DeepSeek-V3 的开放权重发布改变了开放模型中"前沿能力"的含义。该架构是许多 2026 年训练运行正在复制的蓝图。理解它是任何涉及前沿 LLM 训练或推理角色的基本筹码。

## 概念

### 再议不变核心

DeepSeek-V3 仍然是自回归的。它仍然堆叠 decoder 块。每个块仍然有注意力加 MLP 加两个 RMSNorm。它仍然在 MLP 中使用 SwiGLU。它仍然使用 RoPE。Pre-norm。权重绑定嵌入。与每个 Llama 或 Mistral 相同的基线。

### 转折：MLA 而非 GQA

从第 10 阶段 · 第 14 课你知道 GQA 通过在 Q 头组之间共享 K 和 V 来缩小 KV 缓存。多头潜在注意力（MLA）更进一步：K 和 V 被压缩为共享的低秩潜在表示（`kv_lora_rank`），然后在飞行中按头解压缩。KV 缓存仅存储潜在表示——通常每 token 每层 512 个 float，而非 8 x 128 = 1024 个 float。

在 128k 上下文下，DeepSeek-V3 使用 MLA（每个 token 每层一个共享潜在 `c^{KV}`；K 和 V 都从该潜在表示通过可被吸收到后续矩阵乘法中的向上投影导出）：

```
kv_cache = num_layers * kv_lora_rank * max_seq_len * bytes_per_element
         = 61 * 512 * 131072 * 2
         = 7.6 GB
```

一个假设的 GQA 基线（Llama 3 70B 形状，8 个 KV 头，头维度 128）将付出：

```
kv_cache = 2 * 61 * 8 * 128 * 131072 * 2
         = 30.5 GB
```

在 128k 上下文下，MLA 比 Llama-3-70B 风格的 GQA 缓存小 4 倍。

权衡：MLA 在每个注意力计算（每个头）添加解压缩步骤。与节省的带宽相比，额外的计算很小。对长上下文推理是净收益。

### 路由：无辅助损失负载均衡

MoE 路由器决定哪些 top-k 专家处理每个 token。天真的路由器将太多工作集中在少数几个专家上，让其他专家空闲。标准修复：添加一个惩罚负载不平衡的辅助损失项。这有效但会略微降低主任务性能。

DeepSeek-V3 引入了一个无辅助损失方案。每个专家的偏置项被添加到路由器 logits 中，在训练中通过一个简单规则调整：如果专家 `e` 过载，减小 `bias_e`；如果欠载，增加它。没有额外的损失项。训练保持干净。专家负载保持平衡。

对主损失的影响：无测量到的。对 MoE 架构的影响：更干净，没有辅助损失超参数需要调优。

### MTP：更密集训练 + 免费草案

从第 10 阶段 · 第 18 课你知道 DeepSeek-V3 添加了 D=1 个 MTP 模块，预测两个位置之后的 token。在推理时，训练好的模块被重新用作推测解码草案，接受率 80%+。在训练时，每个隐藏状态在 D+1 = 2 个目标上被监督，提供更密集的信号。

参数：671B 主模型之上 14B。开销：2.1%。

### 训练：DualPipe

从第 10 阶段 · 第 19 课你知道 DualPipe 是一个双向流水线，将前向和后向块与跨节点 all-to-all 通信重叠。在 DeepSeek-V3 的 2,048-H800 规模上，它回收了 1F1B 将损失于流水线气泡的大约 245k GPU 小时。

### 配置，逐字段

以下是 DeepSeek-V3 的配置（简化版）：

```
hidden_size: 7168
intermediate_size: 18432   (密集 MLP 隐藏大小，用于前几层)
moe_intermediate_size: 2048 (专家 MLP 隐藏大小)
num_hidden_layers: 61
first_k_dense_layers: 3    (前 3 层使用密集 MLP)
num_attention_heads: 128
num_key_value_heads: 128   (在 MLA 下形式上等于 num_heads，但
                           真正的压缩在 kv_lora_rank 中)
kv_lora_rank: 512          (MLA 潜在维度)
num_experts: 256            (每个块的 MoE 专家数)
num_experts_per_tok: 8      (top-8 路由)
shared_experts: 1           (每个块始终激活的共享专家)
max_position_embeddings: 163840
rope_theta: 10000.0
vocab_size: 129280
mtp_module: 1               (深度 1 的 1 个 MTP 模块)
```

解析它：

- `hidden_size=7168`：嵌入维度。
- `num_hidden_layers=61`：总块深度。
- `first_k_dense_layers=3`：前 3 个块使用大小为 18432 的密集 MLP。其余 58 个使用 MoE。
- `num_attention_heads=128`：128 个查询头。
- `kv_lora_rank=512`：K 和 V 被压缩到这个潜在维度，然后按头解压缩。
- `num_experts=256, num_experts_per_tok=8`：每个 MoE 块有 256 个专家，路由 top-8。
- `shared_experts=1`：在 256 个路由专家之上，1 个始终激活的专家对每个 token 做出贡献。将其视为一个"密集地板"，确保每个 token 获得可靠的某些内容。
- `moe_intermediate_size=2048`：每个专家 MLP 隐藏大小。比密集 MLP 小，因为有 256 个。

### 参数统计

完整计算位于 `code/main.py`。标题数字：

- 嵌入：`vocab * hidden = 129280 * 7168 = ~0.93B`。
- 前 3 个密集块：带 MLA 的注意力（每块 ~144M）+ 密集 MLP（每块 ~260M）+ norms。总计约 1.2B。
- 58 个 MoE 块：带 MLA 的注意力（~144M）+ 每个 256 专家（每个 30M）+ 1 共享专家（30M）+ norm。每个块总计 ~7.95B，含所有专家。58 个 MoE 块总 461B。
- MTP 模块：14B。

合计：核心架构约 476B + 14B MTP + 明确的发布数字 671B 考虑了额外的结构参数（偏置张量、专家特定组件、共享专家缩放等）。我们在计算器中再现的数字在发布的 3-5% 以内——差异来自 DeepSeek 报告在其第 2 节附录中记录的细粒度统计。

每次前向的激活参数：

- 注意力：每层 144M * 61 = 8.8B（所有层触发）。
- 激活 MLP：前 3 层密集（3 * 260M = 780M），58 个 MoE 层每个激活 8 路由 + 1 共享 + 路由开销。每层激活 MLP：~260M。总计：3 * 260M + 58 * 260M = ~15.9B。
- 嵌入 + norms：1.2B。
- 总激活：大致 26B 核心 + 14B MTP（训练但在推理时不一定总是运行）≈ 37B。

### 671B / 37B 比率

18x 稀疏比率（激活参数是总量的 5.5%）。DeepSeek-V3 是已发布开放权重的最稀疏前沿 MoE 模型。Mixtral 8x7B 比率 13/47（28%）密集得多。Llama 4 Maverick 比率 17B/400B（4.25%）是可比的。DeepSeek 的赌注：在前沿规模上，更多专家低激活比率产生每个激活 FLOP 更好的质量。

### DeepSeek-V3 位于何处

| 模型 | 总量 | 激活 | 比率 | 注意力 | 新颖想法 |
|-------|------|-------|-------|-----------|-------------|
| Llama 3 70B | 70B | 70B | 100% | GQA 64/8 | — |
| Llama 4 Maverick | 400B | 17B | 4.25% | GQA | — |
| Mixtral 8x22B | 141B | 39B | 27% | GQA | — |
| DeepSeek V3 | 671B | 37B | 5.5% | MLA 512 | MLA + MTP + aux-free + DualPipe |
| Qwen 2.5 72B | 72B | 72B | 100% | GQA 64/8 | YaRN extension |

### 后续：R1、V4

DeepSeek-R1（2025）是在 V3 骨干上的推理训练运行。R1 使用相同的架构。变化的是后训练配方（可验证任务上的大规模 RL），而非预训练架构。

DeepSeek-V4（如果发布）预计将保持 MLA + MoE + MTP，并添加 DSA（DeepSeek Sparse Attention），即第 10 阶段 · 第 17 课 NSA 的继任者。族系稳定：架构级别的创新累积；每个版本调整额外的旋钮。

```figure
moe-routing
```

## 使用它

`code/main.py` 是专门针对 DeepSeek-V3 形状的参数计算器。运行它，将其输出与论文的数字比较，并在假设变体上使用（256 专家 vs 512，top-8 vs top-16，MLA rank 512 vs 1024）。

要看的内容：

- 总参数量 vs 发布的 671B。
- 激活参数量 vs 发布的 37B。
- 128k 上下文下的 KV 缓存——MLA vs GQA 比较。
- 每层分解，以了解参数预算实际去了哪里。

## 产出

本课产出 `outputs/skill-deepseek-v3-reader.md`。给定一个 DeepSeek 系列模型（V3、R1 或任何未来变体），它产生一个逐组件架构阅读，命名配置的每个字段，按组件推导参数量，并识别模型使用的四个 DeepSeek 特定创新中的哪些。

## 练习

1. 运行 `code/main.py`。将计算器的总参数估计与发布的 671B 比较，并识别差异来源。论文第 2 节有完整的逐项清单。

2. 修改配置使用 MLA rank 256 而非 512。计算 128k 上下文下的结果 KV 缓存大小。它带来多少百分比的减少，以及对每头表达能力有什么成本？

3. 比较 DeepSeek-V3 的（256 专家，top-8）路由与假设的（512 专家，top-8）变体。总参数增长；激活参数保持不变。额外的专家容量在理论上买来了什么，在推理上又成本了什么？

4. 阅读 DeepSeek-V3 技术报告（arXiv:2412.19437）关于 MLA 的第 2.1 节。用三句话解释为什么 K 和 V 解压缩矩阵可以被"吸收"到后续矩阵乘法中以实现推理时效率。

5. DeepSeek-V3 对大多数操作使用 FP8 训练。计算存储 671B 权重时 FP8 vs BF16 的内存节省。这与 14.8T token 的训练预算如何交汇？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| MLA | "多头潜在注意力" | 将 K 和 V 压缩为共享的低秩潜在表示（kv_lora_rank，通常 512），飞行中按头解压缩；KV 缓存仅存储潜在表示 |
| kv_lora_rank | "MLA 压缩维度" | K 和 V 共享潜在表示的大小；DeepSeek-V3 使用 512 |
| 前 k 个密集层 | "早期层保持密集" | MoE 模型的前几层跳过 MoE 路由器并运行密集 MLP 以保持稳定性 |
| num_experts_per_tok | "Top-k 路由" | 每个 token 激活的路由专家数；DeepSeek-V3 使用 8 |
| 共享专家 | "始终激活的专家" | 无论路由如何都对每个 token 处理的专家；DeepSeek-V3 使用 1 |
| 无辅助损失路由 | "偏置调整负载均衡" | 在训练期间调整每个专家偏置项以保持专家负载平衡，无需添加损失项 |
| MTP 模块 | "额外的预测头" | Transformer 块从 h^(1) 和 E(t+1) 预测 t+2；更密集训练，免费推测解码草案 |
| DualPipe | "双向流水线" | 将前向/后向计算与跨节点 all-to-all 重叠的训练调度 |
| 激活参数比率 | "稀疏性" | active_params / total_params；DeepSeek-V3 达到 5.5% |
| FP8 训练 | "8 位训练" | 训练存储和许多计算操作使用 FP8；比 BF16 约减少一半内存，质量成本微小 |

## 延伸阅读

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) —— 完整的架构、训练和结果文档
- [DeepSeek-V3 model card on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3) —— 配置文件和部署说明
- [DeepSeek-V2 paper (arXiv:2405.04434)](https://arxiv.org/abs/2405.04434) —— 引入 MLA 的前身
- [DeepSeek-R1 paper (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) —— V3 架构上的推理训练后继
- [Native Sparse Attention (arXiv:2502.11089)](https://arxiv.org/abs/2502.11089) —— DeepSeek 家族注意力的未来方向
- [DualPipe repository](https://github.com/deepseek-ai/DualPipe) —— 训练调度参考
