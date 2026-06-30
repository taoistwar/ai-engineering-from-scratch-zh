# 具身 VLA：RT-2、OpenVLA、π0、GR00T

> 模型第一次从网站读取食谱并在厨房机器人上执行是 RT-2（Google DeepMind，2023 年 7 月）。RT-2 将动作离散化为文本 token，在网页数据和机器人动作数据上协同微调 VLM，并证明了网络规模的视觉-语言知识可以迁移到机器人控制。OpenVLA（2024 年 6 月）发布了开源 7B 参考实现。Physical Intelligence 的 π0 系列（2024-2025）增加了流匹配动作专家。NVIDIA 的 GR00T N1（2025 年 3 月）为大规模人形机器人提供了双系统（System 1 / System 2）控制。VLA 原语——视觉-语言-动作，一个能看、能读、能动的单一模型——是本阶段理解模型与第 15 阶段自主系统之间的桥梁。

**Type:** Learn
**Languages:** Python (stdlib, action tokenizer + VLA inference skeleton)
**Prerequisites:** Phase 12 · 05 (LLaVA), Phase 15 (Autonomous Systems, referenced)
**Time:** ~180 minutes

## 学习目标

- 描述动作 token 化：离散区间编码（RT-2）、FAST 高效动作 token、连续流匹配动作（π0）。
- 解释为什么在网页 + 机器人数据上协同微调能保持通用知识向新任务的迁移。
- 在相同机器人任务上比较 OpenVLA（开源 7B Llama+VLM）、π0（流匹配）和 GR00T N1（双系统）。
- 列举 Open X-Embodiment 数据集及其作为 RT-X 训练语料的角色。

## 问题

一个能根据自然语言指令做家务的机器人自 1970 年代以来一直是一个研究目标。2020 年代的答案是：视觉-语言-动作（VLA）模型。与用于 VQA 的 VLM 架构相同，但输出是动作（关节扭矩、末端执行器姿态、离散命令）而不是文本。

VLA 特有的挑战：

1. 动作空间是连续的（关节角度、力）和高维的（7-DOF 手臂 + 3-DOF 抓手 = 10 维，30 Hz）。
2. 机器人特定的训练数据稀缺。Open X-Embodiment 约有 1M 条轨迹；网络文本-图像有 5B+。
3. 控制频率很重要。30 Hz 控制循环意味着每个动作 33ms 预算。
4. 安全性。错误的动作会损坏硬件、伤害人类或财产。

## 概念

### 动作 Token 化（RT-2）

RT-2 的技巧：将每个关节目标表示为一个量化的文本 token。将归一化的 [-1, 1] 范围离散化为 256 个区间，将每个区间映射到一个词表 ID。一个 10-DOF 动作在每个控制步骤变为 10 个 token。

在以下混合上协同微调 PaLM-X VLM：

- 网页图像-文本对（标题生成、VQA）。
- 机器人演示，动作为 token。

模型看到"捡起红色方块"（语言）→ 图像（视觉）→ 10 个 token 的动作序列（离散化的关节目标）。网页预训练保持了通用知识迁移：RT-2 可以遵循"向快速移动的物体移动"，即使"快速移动"不在训练数据中。

在 RT-2 论文中推理频率为 3-5 Hz，受限于 VLM 自回归解码。

### OpenVLA——开源 7B 参考实现

OpenVLA（Kim 等人，2024 年 6 月）是开源权重的 RT-2 等价实现。7B Llama 骨干网络，DINOv2 + SigLIP 双视觉编码器，256 区间的动作 token 化。

在 Open X-Embodiment（22 个机器人共 970k 条轨迹）上训练。附带 LoRA 微调支持，以适配新机器人。

推理：通过量化在 A100 上达到 4-5 Hz。足够用于慢速操作，不足以用于高频控制。

### FAST Token 化器——更快的动作解码

Pertsch 等人（2024）证明离散区间 token 化效率低下——大多数动作聚类在区间空间的一小部分。FAST（频域动作序列 Token 化器）通过 DCT 压缩动作序列并量化系数。

一个 30 步的动作轨迹变为约 10 个 FAST token，而不是 300 个离散区间 token。推理速度提升 3-5 倍，无质量损失。

### π0 与流匹配动作

Physical Intelligence 的 π0（Black 等人，2024 年 10 月）用流匹配动作专家替换离散动作 token：

- 一个小型动作 Transformer 读取 VLM 的隐藏状态，并通过整流流输出连续的 50 步动作序列。
- 动作头使用流匹配损失进行训练；VLM 预训练保持不变。
- 推理：完整动作序列在约 5 个去噪步骤中发出，实际为 50 Hz 控制。

π0 的声明：在广泛的操作任务套件上击败 OpenVLA 和 Octo。连续动作公式保留了离散化所破坏的平滑性。

π0.5 和 π0-FAST 是增量升级。π0-FAST 结合了 FAST token 化和流匹配。

### GR00T N1——人形机器人的双系统

NVIDIA 的 GR00T N1（2025 年 3 月）为人形机器人（>30 DOF，全身）构建：

- System 2：一个大型 VLM，读取场景 + 指令，以约 1 Hz 产生高级子目标。
- System 1：一个小型动作头 Transformer，以子目标为条件产生 50-100 Hz 的低级关节命令。

这种拆分映射到 Kahneman 的快慢思考：System 2 规划，System 1 执行。好处：缓慢的 VLM 规模规划不会阻碍快速控制；System 1 保持小巧以确保低延迟。

GR00T N1.7（2025 年底）改进了数据扩展。GR00T 使用来自 Omniverse 的仿真到真实数据进行微调。

### Open X-Embodiment

训练数据。RT-X（2023 年 10 月）汇集了覆盖 22 个机器人共 1M 条轨迹的 22 个数据集。Open X-Embodiment 是每个人都使用的语料：

- ALOHA / Bridge V2 / Droid / RT-2 Kitchen / Language Table。
- 每个样本：(机器人状态、摄像头视角、指令、动作序列)。
- 训练规范：统一动作空间、归一化关节范围、调整摄像头尺寸。

OpenVLA 和 π0 在 Open X-Embodiment 上训练。到特定机器人的领域差距通过在 100-1000 个任务特定演示上进行 LoRA 微调来弥合。

### 协同微调 vs 仅机器人

协同微调将网页 VQA 数据与机器人轨迹混合。比例很重要：太多 VQA，模型遗忘动作；太多机器人数据，模型丢失通用知识。

RT-2 的比例：约 1:1。OpenVLA：约 0.5:1 网页对机器人。π0：类似。精确比例是一个根据数据集大小调整的超参数。

仅机器人训练产生在分布外指令上失败的任务特定模型。协同微调是"捡起红色方块（在演示中）"和"从左边捡起第三大的物体（新措辞）"之间的区别。

### 安全性与动作限制

每个生产 VLA 都带有：

- 硬关节限制（不能超出规格扭矩）。
- 速度限制（软裁剪）。
- 工作空间边界（末端执行器不能离开桌子）。
- 对新颖任务的人类审批。

这些位于 VLA 外部，作为控制层检查。VLA 的输出是建议，不是命令。

## 使用

`code/main.py`：

- 实现 256 区间动作 token 化和反 token 化。
- 草图化基于 DCT + 量化的 FAST token 化器。
- 比较（离散区间、FAST、连续流）每个动作步骤的 token 数。
- 打印 RT-2 → OpenVLA → π0 → GR00T 的谱系摘要。

## 产出

本课产出 `outputs/skill-vla-action-format-picker.md`。给定一个机器人任务（操作、导航、人形全身），在离散区间 + RT-2、FAST + OpenVLA、流匹配 + π0 或双系统 + GR00T 之间做出选择。

## 练习

1. 一个 10-DOF 手臂以 30 Hz 控制率运行。256 区间的离散区间 token 化每秒发出多少个 token？一个 7B VLM 能跟上吗？

2. FAST token 化将 30 步轨迹压缩为约 10 个 token。如果轨迹有高频运动（例如击鼓），用户会丢失什么？

3. π0 的流匹配头在约 5 步中去噪。比较它与 OpenVLA 在 4-5 Hz 下的自回归解码的吞吐量。

4. GR00T 的 System 1 / System 2 拆分映射到 Kahneman。提出一种可能有助于双足行走的不同拆分（System 3？）。

5. 阅读 Open X-Embodiment 第 4 节关于数据集管理。列举防止领域泄漏的三条管理规则。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| VLA | "视觉-语言-动作" | 接受图像 + 指令并输出动作命令的模型 |
| 动作 Token 化 | "离散区间" | 将连续关节目标量化为每维 256 个区间，每个是一个词表 ID |
| FAST Token 化器 | "频率动作 token" | DCT + 量化以将 30 步轨迹压缩为约 10 个 token |
| 协同微调 | "混合网页 + 机器人" | 在网页 VQA 数据与机器人演示一同训练以保持通用知识 |
| 流匹配动作头 | "π0 连续输出" | 通过整流流输出 50 步动作序列的小型 Transformer |
| System 1 / System 2 | "双系统控制" | 大型 VLM 缓慢规划，小型动作头快速执行；GR00T 模式 |
| Open X-Embodiment | "RT-X 数据集" | 1M 轨迹跨机器人数据集；训练语料 |

## 拓展阅读

- [Brohan et al. — RT-2 (arXiv:2307.15818)](https://arxiv.org/abs/2307.15818)
- [Kim et al. — OpenVLA (arXiv:2406.09246)](https://arxiv.org/abs/2406.09246)
- [Black et al. — π0 (arXiv:2410.24164)](https://arxiv.org/abs/2410.24164)
- [NVIDIA — GR00T N1 (arXiv:2503.14734)](https://arxiv.org/abs/2503.14734)
- [Open X-Embodiment Collab — RT-X (arXiv:2310.08864)](https://arxiv.org/abs/2310.08864)
