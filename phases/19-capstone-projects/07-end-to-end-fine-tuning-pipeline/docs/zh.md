# 实践项目 07 — 端到端微调管道（数据到 SFT 到 DPO 到服务）

> 一个在你自己的数据上训练的 8B 模型，在你自己的偏好上进行 DPO 对齐，量化，使用推测解码，并以可衡量的 $/1M token 提供服务。2026 年的开放栈是 Axolotl v0.8、TRL 0.15、用于迭代的 Unsloth、用于量化的 GPTQ/AWQ/GGUF、用于服务的 vLLM 0.7 with EAGLE-3。实践项目是以可复现的方式运行整个管道——YAML 输入，服务的端点输出——并根据 2026 年模型开放性框架发布模型卡片。

**类型:** 实践项目
**语言:** Python（管道），YAML（配置），Bash（脚本）
**前置条件:** 阶段 2（ML），阶段 3（DL），阶段 7（transformers），阶段 10（从头构建 LLM），阶段 11（LLM 工程），阶段 17（基础设施），阶段 18（安全）
**涉及的阶段:** P2 · P3 · P7 · P10 · P11 · P17 · P18
**时间:** 35 小时

## 问题

2026 年，每个严肃的 AI 团队都有一个随时可用的微调管道。不是因为他们发布前沿基础模型，而是因为下游适配——领域 SFT、针对标注偏好的 DPO、用于推测解码的精馏草稿、使用 EAGLE-3 进行服务——是可衡量收益所在。Axolotl v0.8 处理多 GPU SFT 配置。TRL 0.15 处理 DPO 和 GRPO。Unsloth 让你实现快速的单 GPU 迭代。vLLM 0.7 with EAGLE-3 将解码吞吐量推到 2-3 倍而不损失质量。工具可以用了；工艺在 YAML 文件、数据卫生和评估纪律中。

你将运行一个 8B 基础模型（Llama 3.3、Qwen3 或 Gemma 3）经过 SFT 然后 DPO 在特定任务的数据上，量化以进行服务，并在 lm-evaluation-harness、RewardBench-2、MT-Bench-v2 和 MMLU-Pro 上测量增益。你将根据 2026 年模型开放性框架制作一个模型卡片。重点在于可复现性——一键重新运行整个管道的端到端。

## 概念

管道有五个阶段。**数据**：去重（MinHash / Datatrove）、质量过滤（Nemotron-CC 风格分类器）、PII 脱敏、针对公共基准污染的拆分-卫生检查。**SFT**：Axolotl YAML、ZeRO-3 on 8xH100、余弦调度、打包序列、2-3 个 epochs。**DPO 或 GRPO**：TRL 配置、1 epoch、偏好对（人工标注或模型判断）、beta 调整。**量化**：GPTQ + AWQ + GGUF 用于部署灵活性。**服务**：vLLM 0.7 with EAGLE-3 推测头（或 SGLang with SpecForge）、K8s 部署、队列等待上的 HPA。

消融是可交付成果：在三个特定任务基准上的 SFT-only vs SFT+DPO vs SFT+GRPO。服务指标：批处理 1/8/32 下的 token/s、EAGLE-3 接受率、$/1M token。安全评估：Llama Guard 4 通过率。模型卡片：偏见评估、可复现性种子、数据许可。

## 架构

```
raw data (HF datasets + internal)
    |
    v
Datatrove dedup + Nemotron-CC quality filter + PII scrub
    |
    v
split hygiene (MMLU-Pro contamination check)
    |
    v
Axolotl SFT config (YAML)  ---> 8xH100, ZeRO-3
    |
    v
TRL DPO / GRPO config       ---> 4xH100, 1 epoch
    |
    v
GPTQ + AWQ + GGUF quantize
    |
    v
vLLM 0.7 + EAGLE-3 speculative decoding
    |
    v
K8s deployment, HPA on queue-wait
    |
    v
lm-eval-harness + RewardBench-2 + MT-Bench-v2 + MMLU-Pro
    |
    v
model card (2026 MOF) + safety eval (Llama Guard 4)
```

## 技术栈

- 数据: Datatrove 用于去重，Nemotron-CC 分类器用于质量，Presidio 用于 PII
- 基础模型: Llama 3.3 8B、Qwen3 14B 或 Gemma 3 12B
- SFT: Axolotl v0.8 带 ZeRO-3、Flash Attention 3、打包序列
- 偏好调优: TRL 0.15 用于 DPO 或 GRPO；Unsloth 用于单 GPU 迭代
- 量化: GPTQ（Marlin）、AWQ、GGUF via llama.cpp
- 服务: vLLM 0.7 with EAGLE-3 推测解码（或 SGLang 0.4 + SpecForge）
- 评估: lm-evaluation-harness、RewardBench-2、MT-Bench-v2、MMLU-Pro
- 安全评估: Llama Guard 4、ShieldGemma-2
- 基础设施: Kubernetes + NVIDIA device plugin，队列等待指标上的 HPA
- 可观测性: W&B 用于训练，Langfuse 用于推理

## 构建它

1. **数据管道。** 在原始语料库上运行 Datatrove 去重。应用 Nemotron-CC 风格的质量分类器。Presidio 脱敏 PII。使用显式种子写入训练/验证拆分。

2. **污染检查。** 对每个验证拆分，计算 MinHash 与 MMLU-Pro、MT-Bench-v2、RewardBench-2 测试集的比较。拒绝任何重叠。

3. **Axolotl SFT。** YAML 带 ZeRO-3、FA3、序列打包。在 8xH100 上 2-3 epochs。记录到 W&B。

4. **TRL DPO / GRPO。** 取 SFT 检查点，在偏好对上运行一个 epoch 的 DPO（或在数学/代码上使用可验证奖励的 GRPO）。扫描 beta。

5. **量化。** 产出三个量化：GPTQ-INT4-Marlin、AWQ-INT4、GGUF-Q4_K_M 用于 llama.cpp。记录大小和名义吞吐量。

6. **使用推测解码进行服务。** vLLM 0.7 配置，带 EAGLE-3 草稿头，通过 Red Hat Speculators 训练。测量批处理 1/8/32 下的接受率、尾部延迟。报告 $/1M token 与 Anthropic / OpenAI 在同一评估上的对比。

7. **评估矩阵。** 在基础、SFT-only、SFT+DPO、SFT+GRPO 上运行 lm-eval-harness、RewardBench-2、MT-Bench-v2、MMLU-Pro。生成一张表。

8. **安全评估。** 在开发集上 Llama Guard 4 通过率。ShieldGemma-2 输出过滤器。

9. **模型卡片。** MOF 2026 模板：数据、训练、评估、安全、许可、带有 YAML 文件和提交 SHA 的可复现性部分。

## 使用它

```
$ ./pipeline.sh config/llama3.3-8b-domainX.yaml
[data]    300k deduped, 12k filtered, 280k accepted (seed=7)
[SFT]     3 epochs, 8xH100, 6h12m, val loss 1.42 -> 1.03
[DPO]     1 epoch, beta=0.08, 4xH100, 1h40m
[quant]   GPTQ-INT4 4.6 GB, AWQ-INT4 4.8 GB, GGUF-Q4_K_M 5.1 GB
[serve]   vLLM 0.7, EAGLE-3 acceptance 0.74, p99 126ms @ bs=8
[eval]    MMLU-Pro +3.2, MT-Bench-v2 +0.41, RewardBench-2 +0.08
[card]    model-card.md generated under 2026 MOF
```

## 交付它

`outputs/skill-finetuning-pipeline.md` 描述了可交付成果。单一命令从数据运行到 SFT 到 DPO 到量化到服务到评估，并产出模型卡片 + 服务的端点。

| 权重 | 标准 | 如何衡量 |
|:-:|---|---|
| 25 | 评估值与基础模型的差值 | 在目标任务上的实测增益（MMLU-Pro、MT-Bench-v2、特定任务） |
| 20 | 管道可复现性 | 一键使用相同种子端到端重新运行 |
| 20 | 数据卫生 | 去重率、PII 脱敏覆盖率、污染检查绿灯 |
| 20 | 服务效率 | bs=1/8/32 下的 token/s、EAGLE-3 接受率、$/1M token |
| 15 | 模型卡片 + 安全评估 | 2026 MOF 完整性 + Llama Guard 4 通过率 |
| **100** | | |

## 练习

1. 在同一特定任务基准上运行 SFT-only vs SFT+DPO vs SFT+GRPO。报告哪个偏好方法胜出以及胜出多少。

2. 将 Llama 3.3 8B 替换为 Qwen3 14B。测量在匹配质量下的 $/1M token。

3. 测量领域数据 vs 通用 ShareGPT 上的 EAGLE-3 接受率。报告差值以及这对延迟预算意味着什么。

4. 注入 1% 的污染（将 MMLU-Pro 答案泄漏到训练数据中）并重新运行评估。观察 MMLU-Pro 准确度不切实际地跳跃。构建一个污染检查 CI 门来捕获这种情况。

5. 将 LoRA SFT 添加为全量微调的替代方案。测量在内存低 10 倍时的质量差距。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| Axolotl | "SFT 训练器" | 统一的 YAML 驱动训练器，用于 SFT、DPO 和精馏 |
| TRL | "偏好调优器" | Hugging Face 库，用于 LLM 上的 DPO、GRPO、PPO |
| GRPO | "组相对策略优化" | DeepSeek R1 的 RL 配方，带有可验证的奖励 |
| EAGLE-3 | "推测解码草稿" | 预测前方 N 个 token 的草稿头；vLLM 使用目标模型验证 |
| MOF | "模型开放性框架" | 对模型发布在数据、代码、许可证方面进行分级的 2026 年标准 |
| Contamination check | "拆分卫生" | 基于 MinHash 的训练数据中测试集泄漏检测 |
| Acceptance rate | "EAGLE / MTP 指标" | 目标模型接受的草稿 token 的比例 |

## 扩展阅读

- [Axolotl 文档](https://axolotl-ai-cloud.github.io/axolotl/) — 参考 SFT / DPO 训练器
- [TRL 文档](https://huggingface.co/docs/trl) — DPO 和 GRPO 参考实现
- [Unsloth](https://github.com/unslothai/unsloth) — 单 GPU 迭代参考
- [DeepSeek R1 论文 (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — GRPO 方法
- [vLLM + EAGLE-3 文档](https://docs.vllm.ai) — 参考服务栈
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) — 替代推测解码训练器
- [模型开放性框架 2026](https://isocpp.org/) — 开放发布分级标准
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — 规范评估运行器
