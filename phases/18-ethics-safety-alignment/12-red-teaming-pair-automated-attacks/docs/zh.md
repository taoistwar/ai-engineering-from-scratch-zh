# 红队：PAIR 和自动化攻击

> Chao, Robey, Dobriban, Hassani, Pappas, Wong (NeurIPS 2023, arXiv:2310.08419)。PAIR — 提示自动迭代优化 — 是规范的自动化黑盒越狱。带有红队系统提示的攻击者 LLM 迭代地为目标 LLM 提出越狱，在其自身聊天历史中作为上下文反馈积累尝试和响应。PAIR 通常在 20 次查询内成功，比 GCG（Zou et al. 的 token 级梯度搜索）高效数量级且无需白盒访问。PAIR 现在是 JailbreakBench (arXiv:2404.01318) 和 HarmBench 中的标准基线，与 GCG、AutoDAN、TAP 和有说服力对抗提示并列。

**Type:** Build
**Languages:** Python (stdlib, mock PAIR loop against a toy target)
**Prerequisites:** Phase 18 · 01 (instruction-following), Phase 14 (agent engineering)
**Time:** ~75 minutes

## 学习目标

- 描述 PAIR 算法：攻击者系统提示、迭代优化、上下文反馈。
- 解释当目标是黑盒时为什么 PAIR 严格比 GCG 高效。
- 说出四种其他自动化攻击基线（GCG、AutoDAN、TAP、PAP）并陈述每种的一个区分特征。
- 描述 JailbreakBench 和 HarmBench 评估协议以及每个下的"攻击成功率"含义。

## 问题

红队曾经是手动活动。少量专家测试员构造对抗提示并跟踪哪些有效。这不能扩展：攻击成功率需要一个统计样本，而目标是每次模型发布都在变化的目标。PAIR 将红队操作化为带黑盒目标的优化问题。

## 概念

### PAIR 算法

输入：
- 目标 LLM T（我们正在攻击的模型）。
- 裁判 LLM J（评分响应是否是越狱）。
- 攻击者 LLM A（红队优化器）。
- 目标字符串 G："respond with [harmful instruction]."
- 预算 K（通常 20 次查询）。

循环，对于 k 在 1..K：
1. A 被提示目标 G 和迄今的（提示、响应）对历史。
2. A 发出新提示 p_k。
3. 将 p_k 提交给 T；接收响应 r_k。
4. J 按目标评分 (p_k, r_k)。
5. 如果分数 >= 阈值，停止 — 已找到越狱。
6. 否则，将 (p_k, r_k) 附加到 A 的历史；继续。

经验结果 (NeurIPS 2023)：对 GPT-3.5-turbo、Llama-2-7B-chat 攻击成功率 >50%；平均成功查询数在 10-20 范围内。

### 为什么 PAIR 高效

GCG（Zou et al. 2023）通过梯度搜索对抗 token 后缀；它需要白盒模型访问并产生不可读后缀。PAIR 是黑盒并产生跨模型迁移的自然语言攻击。PAIR 的上下文反馈让攻击者从每次拒绝中学习；GCG 没有等价物（每个新 token 更新必须重新发现先前的进展）。

### 相关自动化攻击

- **GCG (Zou et al. 2023, arXiv:2307.15043)。** 对抗后缀的 token 级梯度搜索。白盒、可迁移、产生不可读字符串。
- **AutoDAN (Liu et al. 2023)。** 在提示上的进化搜索，由分层目标引导。
- **TAP (Mehrotra et al. 2024)。** 攻击树与剪枝 — 分支多个 PAIR 风格的推出。
- **PAP (Zeng et al. 2024)。** 有说服力对抗提示 — 将人类说服技术编码为提示模板。

### JailbreakBench 和 HarmBench

两者（2024）标准化评估：

- JailbreakBench (arXiv:2404.01318)。100 种有危害行为跨 10 个 OpenAI 策略类别。攻击成功率 (ASR) 作为主要指标。需要裁判（GPT-4-turbo、Llama Guard 或 StrongREJECT）。
- HarmBench (Mazeika et al. 2024)。510 种行为跨 7 个类别，具有语义和功能危害测试。比较 18 种攻击对 33 个模型。

ASR 通常在固定查询预算下报告。比较攻击需要匹配预算；200 次查询下的 90% ASR 与 20 次下的 85% ASR 不可比。

## 使用它

`code/main.py` 构建玩具 PAIR 循环。目标是一个基于规则的玩具分类器，拒绝包含禁止词汇的提示。攻击者是迭代重写以避免禁止词汇的 LLM 代理。裁判评分。观察 PAIR 找到替代措辞。

## 交付它

本课产出 `outputs/skill-pair-benchmark.md`。给定目标模型端点、裁判规范和危害类别，运行 PAIR 并报告固定预算下的 ASR。

## 练习

1. 运行 `code/main.py`。在 20 次查询预算上测量对玩具目标的 ASR。改变初始攻击者提示 — ASR 变化吗？

2. PAIR 使用同一攻击者进行迭代优化。如果你每次迭代使用不同攻击者模型（更弱/更强），会发生什么？

3. 比较 PAIR 和 GCG 在需要白盒 vs 黑盒访问上。为什么这对生产目标防御重要？

4. 阅读 JailbreakBench 论文。列出评估协议中的三个设计决策以及每个抵抗了什么陷阱。

5. 对相同目标运行 TAP（树，如果实现）vs PAIR。在什么条件下 TAP 在相同查询预算上击败 PAIR？

## 核心术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| PAIR | "迭代优化越狱" | 使用上下文反馈攻击黑盒模型的 LLM 驱动攻击 |
| GCG | "对抗 token 搜索" | 在白盒模型上对抗后缀的梯度搜索 |
| AutoDAN | "进化搜索" | 在候选提示上的分层目标驱动搜索 |
| TAP | "攻击树" | 分支 PAIR 推出，带剪枝的树搜索 |
| PAP | "有说服力提示" | 应用说服模板的少样本攻击 |
| ASR | "攻击成功率" | 越狱达到目标行为的攻击尝试占比 |
| JailbreakBench | "标准评估套件" | 100 种行为跨 10 类，固定预算，标准裁判 |

## 进一步阅读

- [Chao et al. — PAIR (NeurIPS 2023, arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — 规范的自动化红队论文
- [Zou et al. — 通用和可迁移对抗攻击 (NeurIPS 2023, arXiv:2307.15043)](https://arxiv.org/abs/2307.15043) — GCG
- [JailbreakBench (arXiv:2404.01318)](https://arxiv.org/abs/2404.01318) — 标准化评估协议
- [HarmBench (Mazeika et al. 2024, NeurIPS 2024)](https://arxiv.org/abs/2402.04249) — 广泛比较
- [TAP (Mehrotra et al. 2024, arXiv:2312.02119)](https://arxiv.org/abs/2312.02119)
