# 直接偏好优化家族

> Rafailov et al. (2023) 表明 RLHF 的最优解在偏好数据方面有闭合形式，因此你可以跳过显式奖励模型直接优化策略。该洞察催生了一个家族 — IPO、KTO、SimPO、ORPO、BPO — 每个修复 DPO 的一个失败模式。2026 年，直接对齐算法比 PPO 发布更多的前沿后训练运行。但来自第 2 课的过度优化曲线仍然适用：DAA 不能逃脱 Goodhart，它们只是移动了它咬合的位置。

**Type:** Learn
**Languages:** Python (stdlib, six-variant preference-loss comparator)
**Prerequisites:** Phase 18 · 01 (InstructGPT), Phase 18 · 02 (Reward hacking), Phase 10 · 08 (DPO basics)
**Time:** ~75 minutes

## 学习目标

- 从 RLHF-with-KL 最优解推导 DPO 闭合形式。
- 陈述 IPO、KTO、SimPO、ORPO、BPO 各自修复的 DPO 中的失败模式。
- 区分"隐式奖励差距"和"偏好强度"，解释为什么 IPO 的恒等映射重要。
- 解释为什么 Rafailov et al. (NeurIPS 2024) 证明 DAA 尽管没有显式 RM 也会过度优化。

## 问题

RLHF 目标（第 1 课）：

```
max_pi E_{x,y~pi} [ r(x, y) ] - beta * KL(pi || pi_ref)
```

有一个已知的最优解：

```
pi*(y|x) = (1/Z(x)) * pi_ref(y|x) * exp(r(x, y) / beta)
```

因此奖励由最优策略与参考的比率隐式定义：

```
r(x, y) = beta * log(pi*(y|x) / pi_ref(y|x)) + beta * log Z(x)
```

将此代入 Bradley-Terry 偏好似然，分区函数 `Z(x)` 因仅依赖于 `x` 而抵消。剩下的仅是在策略参数中的损失 — 不需要奖励模型。这就是 DPO。

缺陷：推导假设最优解可达、偏好数据在分布内、且参考策略是真正的模式锚点。其中没有一个是完全成立的。每个家族成员修复一个被违反的不同假设。

## 概念

### DPO (Rafailov et al., 2023)

```
L_DPO = -log sigmoid(
  beta * log(pi(y_w | x) / pi_ref(y_w | x))
  - beta * log(pi(y_l | x) / pi_ref(y_l | x))
)
```

可能出错的地方：

- 隐式奖励差距 `beta * (log(pi/pi_ref)_w - log(pi/pi_ref)_l)` 是无界的。微小的偏好可以产生任意大的差距。
- 损失将选中和被拒绝对数概率推向相反方向。只要被拒绝下降更快，它可以将选中绝对对数概率下推。这是退化选中响应现象。
- 分布外偏好（罕见 vs 罕见对）产生任意隐式奖励。

### IPO (Azar et al., 2024)

恒等偏好优化将 log-sigmoid 替换为在偏好概率上的恒等映射。损失变为在有界目标上的平方误差：

```
L_IPO = (log(pi(y_w | x) / pi_ref(y_w | x)) - log(pi(y_l | x) / pi_ref(y_l | x)) - 1/(2 beta))^2
```

差距由 `1/(2 beta)` 约束。偏好强度和隐式奖励差距成比例。没有爆炸。

### KTO (Ethayarajh et al., 2024)

Kahneman-Tversky 优化完全放弃成对结构。给定单个标记输出和二元"可取的"或"不可取的"信号，它映射到前景理论效用：

```
v(x, y) = sigma(beta * log(pi(y|x) / pi_ref(y|x)) - z_ref)
```

对收益和损失具有不同的权重（损失厌恶）。好处：可以使用不成对数据，这丰富得多。

### SimPO (Meng et al., 2024)

简单偏好优化将训练信号与生成对齐。完全移除参考策略，按长度归一化对数似然：

```
L_SimPO = -log sigmoid(
  (beta / |y_w|) * log pi(y_w | x)
  - (beta / |y_l|) * log pi(y_l | x)
  - gamma
)
```

带有边界 `gamma` 以稳定。长度归一化移除了利用 DPO 长度偏差失败模式的动机（更长的 `y_w` 在构造上给出更大的对数概率差距）。

### ORPO (Hong et al., 2024)

赔率比偏好优化向标准 SFT 负对数似然添加偏好项：

```
L_ORPO = L_NLL(y_w) + lambda * L_OR
L_OR = -log sigmoid(log(odds(y_w) / odds(y_l)))
```

无参考策略 — SFT 项是正则化器。单个阶段从基础模型训练到对齐模型。无独立的 SFT 检查点。

### BPO (ICLR 2026 提交, OpenReview id=b97EwMUWu7)

识别退化选中响应问题：DPO 保留了 `y_w > y_l` 的排名但 `y_w` 的绝对对数概率可能下降。BPO 添加了惩罚向下移动选中响应的单行修正。报告在 Llama-3.1-8B-Instruct 上的数学推理准确率相比 DPO 提高 +10.1%。

### 通用结果：DAA 仍然过度优化

Rafailov et al. "Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms" (NeurIPS 2024) 在多个数据集上跨 KL 预算训练了 DPO、IPO、SLiC 策略。黄金奖励 vs KL 曲线具有相同的 Gao et al. 峰值和崩溃形状。隐式奖励在训练期间查询分布外样本；KL 正则化不能稳定这一点。

DAA 不能逃脱 Goodhart。它们将咬合的面从"奖励模型过度优化"改变为"参考策略比率过度优化"。通用修复 — 更好的数据、集成、提前停止 — 适用于两者。

### 2026 年在其中选择

- 如果你有大量成对偏好数据：DPO 带保守的 beta，如果长度偏差明显则用 SimPO。
- 如果你有不成对二元反馈：KTO。
- 如果你想要从基础模型开始的单阶段管线：ORPO。
- 如果你在 DPO 日志中看到退化的选中对数概率：BPO。
- 如果偏好强度变化很大且 DPO 饱和：IPO。

每个实验室在电池测试中运行所有五个，按任务选择赢家。没有理由数学推理和安全的最优是相同的。

```figure
dpo-margin
```

## 使用它

`code/main.py` 在玩具偏好数据集上比较六种损失（DPO、IPO、KTO、SimPO、ORPO、BPO），其中真实偏好强度因对而异。每种损失在相同 500 对样本上用小型 softmax 策略优化。绘制每种方法的最终胜率、选中对数概率漂移和隐式奖励扩展。

## 交付它

本课产出 `outputs/skill-preference-loss-selector.md`。给定数据集统计（成对 vs 不成对、可变 vs 均匀偏好强度、长度分布）和目标（单阶段或 SFT 后偏好），推荐偏好损失并报告它保护的失败模式。

## 练习

1. 运行 `code/main.py`。报告 DPO 和 BPO 的最终选中对数概率下降。BPO 应该保持更高的选中绝对概率 — 验证这一点。

2. 修改偏好数据使所有对有相等强度。六种方法中哪种最鲁棒？哪种退化？解释 IPO 在这里的优势。

3. 使被拒绝响应平均比选中的长 2 倍。不改变其他任何东西，数值展示 DPO 的长度利用和 SimPO 的修复。

4. Rafailov et al. (NeurIPS 2024) 声称 DAA 过度优化。复现单点版本：绘制选中减被拒绝 KL 散度并在大 beta 时观察 DPO 中的过度优化。

5. 阅读 BPO 论文摘要 (OpenReview b97EwMUWu7)。写下 BPO 添加到 DPO 的单行修正。对照 `code/main.py` 中的实现确认。

## 核心术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| DPO | "没有奖励模型的 RLHF" | 从闭合形式 RLHF 最优解推导的损失；仅策略参数 |
| 隐式奖励 | "对数比率" | `beta * log(pi(y\|x) / pi_ref(y\|x))` — DPO 隐含的奖励 |
| IPO | "有界 DPO" | 用恒等替换 log-sigmoid；隐式奖励差距由 `1/(2 beta)` 限制 |
| KTO | "不成对 DPO" | 在单个标签上基于前景理论效用，带损失厌恶 |
| SimPO | "无参考 DPO" | 长度归一化对数似然 + 边界；无参考策略 |
| ORPO | "单阶段 DPO" | NLL + 赔率比偏好项；从基础模型单次训练 |
| BPO | "保留选中的 DPO" | DPO 加上对降低选中响应绝对对数概率的惩罚 |
| Degraded Chosen | "选中下降" | 只要被拒绝下降更快，DPO 降低选中对数概率 |
| DAA | "直接对齐算法" | 任何跳过显式 RM 的偏好损失方法 |

## 进一步阅读

- [Rafailov et al. — 直接偏好优化 (NeurIPS 2023, arXiv:2305.18290)](https://arxiv.org/abs/2305.18290)
- [Azar et al. — 理解从人类偏好中学习的通用理论范式 (AISTATS 2024, arXiv:2310.12036)](https://arxiv.org/abs/2310.12036) — IPO
- [Ethayarajh et al. — KTO: 作为前景理论优化的模型对齐 (arXiv:2402.01306)](https://arxiv.org/abs/2402.01306)
- [Meng, Xia, Chen — SimPO (NeurIPS 2024, arXiv:2405.14734)](https://arxiv.org/abs/2405.14734)
- [Hong, Lee, Thorne — ORPO (EMNLP 2024, arXiv:2403.07691)](https://arxiv.org/abs/2403.07691)
- [BPO — 行为保留优化 (ICLR 2026 OpenReview b97EwMUWu7)](https://openreview.net/forum?id=b97EwMUWu7)
- [Rafailov et al. — DAA 中 RM 过度优化的缩放定律 (NeurIPS 2024, arXiv:2406.02900)](https://arxiv.org/abs/2406.02900)
