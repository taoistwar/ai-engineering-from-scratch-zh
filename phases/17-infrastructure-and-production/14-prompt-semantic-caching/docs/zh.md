# 提示缓存与语义缓存经济学

> **定价快照日期 2026-04。** 以下数字声明反映本课发布时捕获的供应商费率卡；在下游引用前与链接文档核实。

> 缓存在两个层面发生。L2（供应商级）提示/前缀缓存为重复前缀重用注意力 KV — Anthropic 的提示缓存文档宣传在长提示上高达 90% 成本降低和 85% 延迟降低；对于 Claude 3.5 Sonnet，缓存读取为 $0.30/M vs 全新 $3.00/M，5 分钟 TTL 和 1 小时 TTL 选项有 2 倍写入溢价（docs.anthropic.com, 2026-04）。OpenAI 提示缓存自动应用于 ≥1024 token 的提示，并将缓存输入定价约为全新输入的 90% 折扣（platform.openai.com, 2026-04）；每个模型的具体缓存费率取决于实时的费率卡。L1（应用级）语义缓存在嵌入相似度命中时完全跳过 LLM。供应商"95% 准确率"指的是匹配正确性，而非命中率 — 报告的生产命中率从 10%（开放式聊天）到 70%（结构化 FAQ）；没有供应商发布官方基线，因此将这些视为社区遥测而非保证。生产陷阱：并行化杀死缓存（在第一次缓存写入完成前发出的 N 个并行请求可使支出膨胀数倍），以及前缀内的动态内容完全阻止缓存命中。ProjectDiscovery 报告通过将动态文本移出可缓存前缀，从 7% 移至 74% 命中率（2025-11）。

**Type:** Learn
**Languages:** Python (stdlib, toy two-layer cache simulator)
**Prerequisites:** Phase 17 · 04 (vLLM Serving Internals), Phase 17 · 06 (SGLang RadixAttention)
**Time:** ~60 minutes

## 学习目标

- 区分 L2 提示/前缀缓存（供应商处的 KV 重用）和 L1 语义缓存（相似提示跳过 LLM）。
- 解释 Anthropic 的 `cache_control` 显式标记和两个 TTL 选项（5 分钟 vs 1 小时）及其价格乘数。
- 给定命中率、提示/响应混合和 token 价格，计算预期月度节省。
- 说出使账单膨胀 5-10 倍的并行化反模式，以及使命中率崩溃的动态内容反模式。

## 问题

你为 RAG 服务添加提示缓存。账单保持不变。你测量命中率；是 7%。你的提示看起来是静态的但并非如此 — 系统提示包括格式化为分钟级的当前日期、一个请求 ID，以及为多样性随机重排的示例。每个请求写入一个新缓存条目，读取为零。

另外，你的代理每个用户问题运行十个并行工具调用。所有十个在第一次缓存写入完成前到达供应商。十次写入，零次读取。你的账单是"带缓存"本应花费的 5-10 倍。

缓存是一个协议，不是一个标志。两个层面，两种不同的失败模式。

## 概念

### L2 — 供应商提示/前缀缓存

供应商存储可缓存前缀的注意力 KV 并在匹配该前缀的下一个请求上重用它。你支付一次写入成本，读取几乎免费。

**Anthropic (Claude 3.5 / 3.7 / 4 系列)**：请求中的显式 `cache_control` 标记。你标记哪些块可缓存。TTL：5 分钟（写入成本为基价的 1.25 倍）或 1 小时（写入成本为基价的 2 倍）。缓存读取：Claude 3.5 Sonnet 上 $0.30/M vs 全新 $3.00/M — 便宜 10 倍（docs.anthropic.com，截至 2026-04）。费率因模型而异（Opus/Haiku 单独发布）；始终交叉检查实时定价页面。

**OpenAI**：提示 ≥1024 token 自动缓存（platform.openai.com，2026-04）。无显式标志。缓存输入在当前 gpt-4o/gpt-5 费率卡上约为全新输入的 10 倍便宜。文档和发布说明均未发布官方命中率基线；精心设计提示下社区报告集中在 30–60%。监控 `usage.cached_tokens` 测量你自己的。

**Google (Gemini)**：通过显式 API 进行上下文缓存；100 万 token 上下文意味着缓存更有回报。

**自托管 (vLLM, SGLang)**：Phase 17 · 06 涵盖 RadixAttention — 在你自己的计算上相同模式。

### L1 — 应用级语义缓存

在完全调用 LLM 之前，哈希提示、嵌入它，并寻找相似的缓存请求（余弦相似度高于阈值，通常 0.95+）。命中时返回缓存的响应。未命中时调用 LLM 并缓存结果。

开源：Redis Vector Similarity、GPTCache、Qdrant。商业：Portkey Cache、Helicone Cache。

供应商准确率声明指的是返回的缓存响应在语义上适当的频率 — 而不是你命中的频率。生产命中率：

- 开放式聊天：10-15%。
- 结构化 FAQ / 支持：40-70%。
- 代码问题：20-30%（小变体消除命中）。
- 重复提示的语音代理：50-80%（语音标准化固定集合）。

### 并行化反模式

你的代理并行进行 10 个工具调用。所有 10 个都有相同的 4K-token 系统提示。Anthropic 缓存写入是每请求的；第一次缓存写入在供应商看到提示后约 300 毫秒完成。请求 2-10 在同一毫秒窗口到达，每个都看到缓存未命中。你支付 10 次写入溢价，0 次读取折扣。

修复：用先序序列批处理 — 先单独发请求 1，然后在请求 1 的缓存填充后发送 2-10。第一个工具调用增加 300 毫秒；节省 5-10 倍账单。

### 动态内容反模式

你的系统提示看起来像：

```
You are a helpful assistant. The current time is 14:32:17.
User ID: abc123. Today is Tuesday...
```

每个请求都是唯一的。每个请求都写入。零命中。

修复：将所有真正静态的内容移到可缓存前缀；在缓存边界之后附加动态内容：

```
[cacheable]
You are a helpful assistant. [rules, examples, instructions]
[/cacheable]
[dynamic, not cached]
Current time: 14:32:17. User: abc123.
```

ProjectDiscovery 以这种方式从 7% 移到 74% 缓存命中率，并发布了剖析。

### 批处理 + 缓存叠加以用于过夜工作负载

批处理 API（Phase 17 · 15）在 24 小时周转下给 50% 折扣。叠加缓存输入在上面再给你约 10 倍。叠加后，过夜分类、标注和报告生成工作负载可降至同步未缓存成本的约 10%。

### 你应该记住的数字

定价点在 2026-04 从链接的供应商文档捕获，每几个月漂移 — 在依赖前重新检查。

- Anthropic 缓存读取：Claude 3.5 Sonnet 上 $0.30/M，约为全新输入的 10 倍便宜（docs.anthropic.com）。
- Anthropic 缓存写入溢价：1.25 倍（5 分钟 TTL）或 2 倍（1 小时 TTL）。
- OpenAI 自动缓存：应用于 ≥1024 token 的提示；缓存输入定价约为当前费率卡上全新输入的 10%（platform.openai.com）。
- 语义缓存命中率（社区报告）：约 10% 开放聊天；结构化 FAQ 最高约 70%。不是供应商记录的基线。
- ProjectDiscovery：通过将动态内容移出前缀从 7% → 74% 命中率（项目博客，2025-11）。
- 并行化反模式：当 N 个并行请求错过第一次缓存写入时，典型报告 5–10 倍账单膨胀。

## 使用它

`code/main.py` 在混合工作负载上模拟 L1 + L2 缓存。报告命中率、账单，并展示并行化惩罚。

## 交付它

本课产出 `outputs/skill-cache-auditor.md`。给定提示模板和流量，审计可缓存性并推荐重构。

## 练习

1. 运行 `code/main.py`。切换并行化标志。账单变化多少？
2. 你的系统提示有日期。移出它。展示前后命中率数学。
3. 给定你的请求到达率，计算 1 小时 TTL（2 倍写入）vs 5 分钟 TTL（1.25 倍写入）的盈亏平衡。
4. 0.95 阈值下的语义缓存命中 20%。0.85 下命中 50% 但你看到不正确的缓存响应。选择正确的阈值并论证。
5. 你每个用户问题批处理 10 个并行子查询。重写以实现缓存友好而不增加端到端延迟。

## 核心术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| L2 提示缓存 | "前缀缓存" | 供应商存储重复前缀的 KV |
| `cache_control` | "Anthropic 缓存标记" | 标记可缓存块的显式属性 |
| 缓存写入溢价 | "写入税" | 首次未命中到缓存的额外成本（1.25 倍或 2 倍）|
| L1 语义缓存 | "嵌入缓存" | 在调用 LLM 前的应用级哈希和嵌入 |
| GPTCache | "LLM 缓存库" | 流行的开源 L1 缓存库 |
| 缓存命中率 | "命中/总计" | 从缓存服务的请求占比 |
| 并行化反模式 | "N-写入陷阱" | N 个并行请求错过缓存 N 次 |
| 动态内容陷阱 | "提示中时间陷阱" | 前缀中的动态字节消除命中率 |
| RadixAttention | "副本内缓存" | SGLang 的前缀缓存实现 |

## 进一步阅读

- [Anthropic 提示缓存](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — 官方 `cache_control` 语义和 TTL。
- [OpenAI 提示缓存](https://platform.openai.com/docs/guides/prompt-caching) — 自动缓存行为和资格。
- [TianPan — LLM 生产中的语义缓存](https://tianpan.co/blog/2026-04-10-semantic-caching-llm-production)
- [ProjectDiscovery — 用提示缓存削减 LLM 成本 59%](https://projectdiscovery.io/blog/how-we-cut-llm-cost-with-prompt-caching)
- [DigitalOcean / Anthropic — 提示缓存](https://www.digitalocean.com/blog/prompt-caching-with-digital-ocean)
