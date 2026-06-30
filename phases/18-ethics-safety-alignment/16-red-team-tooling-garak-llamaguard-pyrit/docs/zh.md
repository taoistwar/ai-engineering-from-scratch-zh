# 红队工具 — Garak、Llama Guard、PyRIT

> 三种生产工具框架 2026 红队栈。Llama Guard (Meta) — 在 14 个 MLCommons 危害类别上微调的 Llama-3.1-8B 分类器；2025 年 Llama Guard 4 是从 Llama 4 Scout 剪枝的 12B 原生多模态分类器。Garak (NVIDIA) — 开源 LLM 漏洞扫描器，具有针对幻觉、数据泄露、提示注入、毒性和越狱的静态、动态和自适应探针。PyRIT (Microsoft) — 多轮红队战役，包含 Crescendo、TAP 和自定义转换链用于深度利用。Llama Guard 3 记录在 Meta 的 "Llama 3 Herd of Models" (arXiv:2407.21783)；Llama Guard 3-1B-INT4 在 arXiv:2411.17713；Garak 探针架构在 github.com/NVIDIA/garak。这些工具是 2026 年红队研究（第 12-15 课）与部署（第 17+ 课）之间的生产接口。

**Type:** Build
**Languages:** Python (stdlib, tool-architecture simulator and Llama Guard-style classifier mock)
**Prerequisites:** Phase 18 · 12-15 (jailbreaks and IPI)
**Time:** ~75 minutes

## 学习目标

- 描述 Llama Guard 3/4 在安全栈中的位置：输入分类器、输出分类器、或两者。
- 说出 14 个 MLCommons 危害类别并陈述一个不显而易见的（Code Interpreter Abuse）。
- 描述 Garak 的探针架构：探针、检测器、测试工具。
- 描述 PyRIT 的多轮战役结构以及它如何与 Garak 探针组合。

## 问题

第 12-15 课呈现攻击面。生产部署需要可重复、可扩展的评估。三种工具主导 2026 年：Llama Guard（防御分类器）、Garak（扫描器）、PyRIT（战役编排器）。每种针对红队生命周期的不同层面。

## 概念

### Llama Guard (Meta)

Llama Guard 3 是为 MLCommons AILuminate 14 个类别的输入/输出分类微调的 Llama-3.1-8B 模型：
- 暴力犯罪、非暴力犯罪、性相关、CSAM、诽谤
- 专业建议、隐私、知识产权、无差别武器、仇恨
- 自杀/自残、性内容、选举、代码解释器滥用

支持 8 种语言。用法：放在 LLM 之前（输入审核）、LLM 之后（输出审核）、或两者。两种用法生成不同的训练分布 — Llama Guard 3 作为处理两者的单一模型发布。

Llama Guard 3-1B-INT4 (arXiv:2411.17713, 440MB, 移动 CPU 上约 30 tokens/s) 是量化的边缘变体。

Llama Guard 4（2025 年 4 月）为 12B，原生多模态，从 Llama 4 Scout 剪枝。用一个摄取文本+图像的分类器替换了 8B 文本和 11B 视觉前辈。

### Garak (NVIDIA)

开源漏洞扫描器。架构：
- **探针。** 针对幻觉、数据泄露、提示注入、毒性、越狱的攻击生成器。静态（固定提示）、动态（生成的提示）、自适应（响应目标输出）。
- **检测器。** 根据预期失败模式对输出评分 — 毒性、泄露、越狱。
- **测试工具。** 管理探针-检测器对，运行战役，生成报告。

TrustyAI 将 Garak 与 Llama-Stack 盾牌集成（Prompt-Guard-86M 输入分类器、Llama-Guard-3-8B 输出分类器）进行端到端的被保护目标评估。分层评分 (TBSA) 取代二元的通过/失败 — 模型可以在同一探针上在严重性层 3 通过，在严重性层 5 失败。

### PyRIT (Microsoft)

Python Risk Identification Toolkit。多轮红队战役。围绕以下构建：
- **转换器。** 转换种子提示 — 改写、编码、翻译、角色扮演。
- **编排器。** 运行战役：Crescendo（逐步升级）、TAP（分支）、RedTeaming（自定义循环）。
- **评分。** LLM 作为裁判或分类器作为裁判。

## 使用它

`code/main.py` 模拟玩具分类器，运行探针（静态、动态、自适应），并以分层评分（通过/失败每层）评分输出。你将对同一探针在不同层上看到通过持续而失败出现。

## 交付它

本课产出 `outputs/skill-red-team-runbook.md`。给定目标模型端点和危害类别，选择工具和战役计划，运行，并按分层评分生成红队报告。

## 练习

1. 运行 `code/main.py`。同一探针在层 3 通过，层 4 失败。分层评分揭示了二元通过/失败遗漏的什么？

2. Llama Guard 4 是 12B 多模态。为什么单个统一分类器是比分离的文本/视觉分类器更强大的防御？

3. 用 Garak 运行静态 vs 自适应探针。在什么条件下自适应探针捕获静态探针遗漏的漏洞？

4. PyRIT 的 Crescendo 编排器逐步升级。设计一个目标模型抵抗温和但屈服于升级的提示集。这种模式有多常见？

5. 组合 Garak（扫描）与 PyRIT（战役）。谁在何处传递结果，以及这种组合捕获了什么单独一个工具遗漏的？

## 核心术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Llama Guard | "Meta 的安全分类器" | 在 14 个 MLCommons 类别上微调的 Llama 分类器 |
| Garak | "NVIDIA 扫描器" | 开源 LLM 漏洞扫描器；静态/动态/自适应探针 |
| PyRIT | "Microsoft 红队" | 多轮红队战役编排器 |
| MLCommons 类别 | "危害分类" | 安全分类的 14 个标准类别 |
| TBSA | "分层评分" | 分层安全评估 — 替代二元通过/失败 |
| Probe | "攻击生成器" | Garak 中生成对抗性 LLM 输入的组件 |
| Crescendo | "逐步升级" | PyRIT 编排器，逐步升级攻击 |

## 进一步阅读

- [Meta — Llama Guard 3 (arXiv:2407.21783)](https://arxiv.org/abs/2407.21783) — 14 类安全分类器
- [Llama Guard 3-1B-INT4 (arXiv:2411.17713)](https://arxiv.org/abs/2411.17713) — 量化边缘变体
- [NVIDIA — Garak](https://github.com/NVIDIA/garak) — 开源扫描器
- [Microsoft — PyRIT](https://github.com/Azure/PyRIT) — 红队工具包
- [MLCommons — AILuminate](https://mlcommons.org/ailuminate/) — 危害分类标准
