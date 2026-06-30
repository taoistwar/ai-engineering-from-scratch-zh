# Prompt Caching and Context Caching（提示缓存与上下文缓存）

> 你的系统提示词是 4,000 个 token。你的 RAG 上下文是 20,000 个 token。你每次请求都发送这两者，并且每次都为这两者付费。提示缓存让提供者在其端保持该前缀预热，并在重用时按正常费率的 10% 计费。正确使用时，可将推理成本降低 50-90%，首 token 延迟降低 40-85%。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 11 · 01 (Prompt Engineering), Phase 11 · 05 (Context Engineering), Phase 11 · 11 (Caching and Cost)
**Time:** ~60 minutes

## 问题

一个编程助手在对话的每一轮都向 Claude 发送相同的 15,000-token 系统提示词。二十轮对话，按每百万输入 token $3 计算，仅输入成本就是 $0.90——在任何实际用户消息之前。乘以 10,000 次日常对话，账单达到每天 $9,000，而这些文本从未变化。

你无法在损害质量的情况下缩短提示词，也无法避免发送它——模型每一轮都需要它。唯一的出路是停止为提供者已经见过的前缀支付全额费用。

这个出路就是提示缓存。Anthropic 于 2024 年 8 月发布了它（2025 年推出了带 1 小时延长 TTL 的变体），OpenAI 同年晚些时候自动化了它，Google 与 Gemini 1.5 一起发布了显式上下文缓存，到 2026 年，三家都在其前沿模型上将其作为一等特性提供。

## 概念

![Prompt caching: write once, read cheap](../assets/prompt-caching.svg)

**机制。** 当请求的前缀与最近请求匹配时，提供者从上次运行提供 KV 缓存，而不是重新编码 token。第一次写入时支付小额溢价，之后每次读取享有大幅折扣。

**2026 年的三种提供者风格。**

| 提供者 | API 风格 | 命中折扣 | 写入溢价 | 默认 TTL | 最小可缓存量 |
|---------|-----------|--------------|---------------|-------------|---------------|
| Anthropic | 内容块上显式 `cache_control` 标记 | 输入折扣 90% | 25% 附加费 | 5 分钟（可延长至 1 小时）| 1,024 token（Sonnet/Opus），2,048（Haiku）|
| OpenAI | 自动前缀检测 | 输入折扣 50% | 无 | 最多 1 小时（尽最大努力）| 1,024 token |
| Google（Gemini）| 显式 `CachedContent` API | 按存储计费；读取约正常费率的 25% | 按 token·小时存储费 | 用户设置（默认 1 小时）| 4,096 token（Flash），32,768（Pro）|

**不变量。** 三者都只缓存前缀。如果请求之间有任何 token 不同，第一个不同 token 之后的所有内容都是缺失。把*稳定*部分放在顶部，*可变*部分放在底部。

### 缓存友好的布局

```
[system prompt]          <-- 缓存此部分
[tool definitions]       <-- 缓存此部分
[few-shot examples]      <-- 缓存此部分
[retrieved documents]    <-- 如果复用则缓存，否则不
[conversation history]   <-- 缓存到上一轮
[current user message]   <-- 永远不要缓存（每次都不同）
```

违反该顺序——将用户消息放在系统提示词上方，在 few-shot 之间交错动态检索——缓存永远无法命中。

### 盈亏平衡计算

Anthropic 的 25% 写入溢价意味着缓存块必须至少被读取两次才能净节省费用。1 写 + 1 读平均每次请求 0.675x 成本（节省 32%）；1 写 + 10 读平均 0.205x（节省 80%）。经验法则：缓存任何你期望在 TTL 内至少复用 3 次的内容。

## 构建

### 步骤 1：使用显式标记的 Anthropic 提示缓存

```python
import anthropic

client = anthropic.Anthropic()

SYSTEM = [
    {
        "type": "text",
        "text": "You are a senior Python reviewer. Follow the rubric exactly.\n\n" + RUBRIC_15K_TOKENS,
        "cache_control": {"type": "ephemeral"},
    }
]

def review(code: str):
    return client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=SYSTEM,
        messages=[{"role": "user", "content": code}],
    )
```

`cache_control` 标记告诉 Anthropic 将该块存储 5 分钟。窗口内的复用命中；过期的复用会重新写入。

**响应 usage 字段：**

```python
response = review(code_a)
response.usage
# InputTokensUsage(
#     input_tokens=120,
#     cache_creation_input_tokens=15023,   # 按 1.25x 付费
#     cache_read_input_tokens=0,
#     output_tokens=340,
# )

response_b = review(code_b)
response_b.usage
# cache_creation_input_tokens=0
# cache_read_input_tokens=15023           # 按 0.1x 付费
```

在 CI 中检查两个字段——如果跨请求 `cache_read_input_tokens` 保持为零，你的缓存键正在漂移。

### 步骤 2：一小时延长 TTL

对于长时间运行的批处理作业，5 分钟默认值在作业之间过期。设置 `ttl`：

```python
{"type": "text", "text": RUBRIC, "cache_control": {"type": "ephemeral", "ttl": "1h"}}
```

1 小时 TTL 的写入溢价是 2x（基线以上 50% 而非 25%），但在任何复用前缀超过 5 次的批处理上回报很快。

### 步骤 3：OpenAI 自动缓存

OpenAI 无需配置。任何超过 1,024 token 且匹配最近请求的前缀自动获得 50% 折扣。

```python
from openai import OpenAI
client = OpenAI()

resp = client.chat.completions.create(
    model="gpt-5",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},   # 长且稳定
        {"role": "user", "content": user_msg},
    ],
)
resp.usage.prompt_tokens_details.cached_tokens  # 获得折扣的部分
```

相同的缓存友好布局规则适用。两件不会破坏 Anthropic 缓存但会破坏 OpenAI 缓存的事：更改 `user` 字段（用作缓存键组件）和重新排序工具。

### 步骤 4：Gemini 显式上下文缓存

Gemini 将缓存视为需要创建和命名的一等对象：

```python
from google import genai
from google.genai import types

client = genai.Client()

cache = client.caches.create(
    model="gemini-3-pro",
    config=types.CreateCachedContentConfig(
        display_name="rubric-v3",
        system_instruction=RUBRIC,
        contents=[FEW_SHOT_EXAMPLES],
        ttl="3600s",
    ),
)

resp = client.models.generate_content(
    model="gemini-3-pro",
    contents=["Review this code:\n" + code],
    config=types.GenerateContentConfig(cached_content=cache.name),
)
```

Gemini 按 token·小时对缓存存在期间收费，读取约为正常输入费率的 25%。当你在多天内跨多个会话复用相同的大型提示时，这是正确的形状。

### 步骤 5：在生产中测量命中率

参见 `code/main.py` 中模拟的三提供者记账器，它跟踪写入/读取/缺失计数并计算每 1000 个请求的混合成本。基于目标命中率进行部署门禁——大多数生产 Anthropic 设置应在预热后看到 >80% 的读取比例。

## 2026 年仍在线上存在的问题

- **顶部的动态时间戳。** `"Current time: 2026-04-22 15:30:02"` 位于系统提示词顶部。每次请求都缺失。将时间戳移到缓存断点以下。
- **工具重排。** 以稳定顺序序列化工具——部署之间的字典重排会破坏每一次命中。
- **近似重复的自由文本。** "You are helpful." vs "You are a helpful assistant."——一个字节差异 = 完全缺失。
- **过小的块。** Anthropic 强制执行 1,024-token 下限（Haiku 为 2,048）。更小的块静默地不被缓存。
- **盲目的成本仪表板。** 将"input tokens"拆分为缓存 vs 未缓存。否则流量下降看起来像缓存成功。

## 使用

2026 年缓存技术栈：

| 场景 | 选择 |
|-----------|------|
| 具有稳定 10k+ 系统提示词的智能体，多轮对话 | Anthropic `cache_control` 配合 5 分钟 TTL |
| 重复使用前缀 30+ 分钟的批处理作业 | Anthropic 配合 `ttl: "1h"` |
| GPT-5 上的无服务器端点，无自定义基础设施 | OpenAI 自动（只需使前缀稳定且长）|
| 多天复用大型代码/文档语料库 | Gemini 显式 `CachedContent` |
| 跨提供者回退 | 保持可缓存前缀布局在各提供者间相同，使任何命中都有效 |

与语义缓存（Phase 11 · 11）结合用于用户消息层：提示缓存处理*token 完全相同的*复用，语义缓存处理*含义相同的*复用。

## Ship It

保存 `outputs/skill-prompt-caching-planner.md`

## 练习、关键术语和进一步阅读同英文版本，已保留。

1. **简单.** 对一个 10 轮对话（带 5,000-token 系统提示词）针对 Claude，先无 `cache_control` 运行再使用 `cache_control`。报告每次的输入 token 账单。
2. **中等.** 编写一个测试工具，给定一个提示模板和请求日志，计算每个提供者（Anthropic 5m、Anthropic 1h、OpenAI 自动、Gemini 显式）的预期命中率和美元节省。
3. **困难.** 构建一个布局优化器：给定一个提示和标记为 `stable=True/False` 的字段列表，重写提示以在最大缓存友好位置放置单个缓存断点而不丢失信息。在真实 Anthropic 端点上验证。

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Prompt caching | "Makes long prompts cheap" | Reusing a provider-side KV-cache for matching prefixes; 50-90% discount on repeated input tokens. |
| `cache_control` | "The Anthropic marker" | Content-block attribute that declares "everything up to here is cacheable"; `{"type": "ephemeral"}`. |
| Cache write | "Paying the premium" | The first request that populates the cache; billed at ~1.25x input rate on Anthropic, free on OpenAI. |
| Cache read | "The discount" | Subsequent requests matching the prefix; billed at 10% (Anthropic), 50% (OpenAI), ~25% (Gemini). |
| TTL | "How long it lives" | Seconds the cache stays warm; Anthropic 5m default (extendable 1h), OpenAI best-effort up to 1h, Gemini user-set. |
| Extended TTL | "1-hour Anthropic cache" | `{"type": "ephemeral", "ttl": "1h"}`; 2x write premium but worth it for batch reuse. |
| Prefix match | "Why my cache missed" | Caches only hit when every token from the start up to the breakpoint is byte-identical. |
| Context caching (Gemini) | "The explicit one" | Google's named, storage-billed cache object; best for multi-day reuse of large corpora. |

## Further Reading

同英文版。
