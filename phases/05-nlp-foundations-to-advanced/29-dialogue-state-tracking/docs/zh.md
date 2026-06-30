# 对话状态追踪

> "I want a cheap restaurant in the north... actually make it moderate... and add Italian."三轮，三次状态更新。DST 保持槽-值字典同步，让预订正常工作。

**类型：** 构建
**语言：** Python
**先修要求：** 第五阶段 · 17（聊天机器人），第五阶段 · 20（结构化输出）
**预计时间：** 约75分钟

## 问题

在一个面向任务的对话系统中，用户的目标被编码为一组槽-值对：`{cuisine: italian, area: north, price: moderate}`。每个用户轮次可以添加、更改或移除一个槽。系统必须读取整个对话并正确输出当前状态。

搞错一个槽，系统就预订了错误的餐厅、安排了错误的航班或收取了错误的卡费。DST 是用户所说和后台执行之间的铰链。

尽管有了 LLM，为什么它在 2026 年仍然重要：

- 合规敏感领域（银行、医疗、航空预订）需要确定性槽值，而不是自由形式的生成。
- 工具使用智能体在调用 API 之前仍然需要槽解析。
- 多轮修正比看起来更难："actually no, make it Thursday."

现代流程：经典 DST 概念 + LLM 提取器 + 结构化输出护栏。

## 概念

![DST: dialog history → slot-value state](../assets/dst.svg)

**任务结构。** 一个模式定义领域（restaurant、hotel、taxi）及其槽（cuisine、area、price、people）。每个槽可以为空、填充自封闭集合的值（price: {cheap, moderate, expensive}）或自由形式的值（name: "The Copper Kettle"）。

**两种 DST 公式。**

- **分类。** 对于每个（slot, candidate_value）对，预测 yes/no。适用于封闭词汇槽。2020 年前的标准。
- **生成。** 给定对话，生成槽值为自由文本。适用于开放词汇槽。现代的默认选择。

**指标。** 联合目标准确率（JGA）——*每个*槽都正确的轮次占比。全有或全无。2026 年 MultiWOZ 2.4 排行榜的顶部约 83%。

**架构。**

1. **基于规则（槽正则表达式 + 关键词）。** 狭窄领域的强基线。可调试。
2. **TripPy / BERT-DST。** 带 BERT 编码的基于复制的生成。LLM 之前的标准。
3. **LDST（LLaMA + LoRA）。** 带领域-槽提示的指令调优 LLM。在 MultiWOZ 2.4 上达到 ChatGPT 级别的质量。
4. **无本体论（2024–26）。** 跳过模式；直接生成槽名称和值。处理开放领域。
5. **提示 + 结构化输出（2024–26）。** LLM 带 Pydantic schema + 约束解码。5 行代码，生产就绪。

### 经典失败模式

- **跨轮次共指。** "Let's stay with the first option."需要消解哪个选项。
- **覆写 vs 追加。** 用户说"add Italian。"你是替换 cuisine 还是追加？
- **隐式确认。** "OK cool"——那接受了提供的预订吗？
- **修正。** "Actually make it 7 pm."必须更新时间而不清除其他槽。
- **对先前系统话语的共指。** "Yes, that one."哪个"that"？

## 构建它

### 步骤 1：基于规则的槽提取器

参见`code/main.py`。正则表达式 + 同义词字典覆盖狭窄领域中 70% 的规范话语：

```python
CUISINE_SYNONYMS = {
    "italian": ["italian", "pasta", "pizza", "italy"],
    "chinese": ["chinese", "chow mein", "noodles"],
}


def extract_cuisine(utterance):
    for canonical, synonyms in CUISINE_SYNONYMS.items():
        if any(syn in utterance.lower() for syn in synonyms):
            return canonical
    return None
```

在规范词汇之外脆弱。适用于确定性槽确认。

### 步骤 2：状态更新循环

```python
def update_state(state, utterance):
    new_state = dict(state)
    for slot, extractor in SLOT_EXTRACTORS.items():
        value = extractor(utterance)
        if value is not None:
            new_state[slot] = value
    for slot in NEGATION_CLEARS:
        if is_negated(utterance, slot):
            new_state[slot] = None
    return new_state
```

三个不变性：

- 永远不要重置用户未触及的槽。
- 显式否定（"never mind the cuisine"）必须清除。
- 用户修正（"actually..."）必须覆写，而不是追加。

### 步骤 3：带结构化输出的 LLM 驱动 DST

```python
from pydantic import BaseModel
from typing import Literal, Optional
import instructor

class RestaurantState(BaseModel):
    cuisine: Optional[Literal["italian", "chinese", "indian", "thai", "any"]] = None
    area: Optional[Literal["north", "south", "east", "west", "center"]] = None
    price: Optional[Literal["cheap", "moderate", "expensive"]] = None
    people: Optional[int] = None
    day: Optional[str] = None


def llm_dst(history, llm):
    prompt = f"""You track the slot values of a restaurant booking across turns.
Dialogue so far:
{render(history)}

Update the state based on the latest user turn. Output only the JSON state."""
    return llm(prompt, response_model=RestaurantState)
```

Instructor + Pydantic 保证有效的状态对象。无正则表达式、无模式不匹配、无幻觉槽。

### 步骤 4：JGA 评估

```python
def joint_goal_accuracy(predicted_states, gold_states):
    correct = sum(1 for p, g in zip(predicted_states, gold_states) if p == g)
    return correct / len(predicted_states)
```

校准：系统在多少比例的轮次中正确得到了所有槽？对于 MultiWOZ 2.4，2026 年顶尖系统：80-83%。你的领域内系统应在你的狭窄词汇上超过此值，否则 LLM 基线击败你。

### 步骤 5：处理修正

```python
CORRECTION_CUES = {"actually", "no wait", "on second thought", "change that to"}


def is_correction(utterance):
    return any(cue in utterance.lower() for cue in CORRECTION_CUES)
```

在检测到修正时，覆写最后更新的槽而不是追加。没有 LLM 帮助很难做对。现代模式：始终让 LLM 从历史重新生成整个状态，而不是增量更新——这自然处理修正。

## 陷阱

- **全历史重新生成成本。** 让 LLM 每轮重新生成状态总共花费 O(n²) token。限制历史或总结更早的轮次。
- **模式漂移。** 事后添加新槽破坏旧的训练数据。给你的模式标注版本。
- **大小写敏感性。** "Italian" vs "italian" vs "ITALIAN"——在任何地方归一化。
- **隐式继承。** 如果用户先前指定了"for 4 people"，对时间的新请求不应清除 people。始终传递完整的历史。
- **自由形式 vs 封闭集。** 名称、时间和地址需要自由形式槽；cuisines 和 areas 是封闭的。在模式中混合两者。

## 使用它

2026 年技术栈：

| 场景 | 方法 |
|-----------|----------|
| 狭窄领域（一个或两个意图） | 基于规则 + 正则表达式 |
| 宽广领域、有标注数据 | LDST（LLaMA + LoRA 在 MultiWOZ 风格数据上） |
| 宽广领域、无标签、生产就绪 | LLM + Instructor + Pydantic schema |
| 口语 / 语音 | ASR + 归一化器 + LLM-DST |
| 多领域预订流程 | 基于模式的 LLM 配合每个领域的 Pydantic 模型 |
| 合规敏感 | 基于规则为主，LLM 后备有确认流程 |

## 交付它

保存为 `outputs/skill-dst-designer.md`：

```markdown
---
name: dst-designer
description: 设计对话状态追踪器——模式、提取器、更新策略、评估。
version: 1.0.0
phase: 5
lesson: 29
tags: [nlp, dialogue, task-oriented]
---

给定一个用例（领域、语言、词汇开放性、合规需求），输出：

1. 模式。领域列表、每个领域的槽、每个槽的开放 vs 封闭词汇。
2. 提取器。基于规则 / seq2seq / LLM-with-Pydantic。理由。
3. 更新策略。重新生成整个状态 / 增量；修正处理；否定处理。
4. 评估。保留对话集上的联合目标准确率、槽级精确率/召回率、最难槽上的混淆。
5. 确认流程。何时显式要求用户确认（破坏性操作、低置信度提取）。

拒绝为合规敏感槽使用仅 LLM 的 DST 而没有基于规则的二级检查。拒绝任何不能在用户修正时回滚槽的 DST。标记没有版本标签的模式。
```

## 练习

1. **简单。** 在`code/main.py`中为 3 个槽（cuisine、area、price）构建基于规则的状态追踪器。在 10 个手工制作的对话上测试。测量 JGA。
2. **中等。** 使用 Instructor + Pydantic + 一个小型 LLM 处理相同数据集。比较 JGA。检查最难处理的轮次。
3. **困难。** 实现两者并路由：基于规则为主，当基于规则发出 <2 个有信心的槽时 LLM 后备。测量组合的 JGA 和每轮推理成本。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| DST | 对话状态追踪 | 在对话轮次间维护槽-值字典。 |
| 槽 | 用户意图的单元 | 后台需要的命名参数（cuisine、date）。 |
| 领域 | 任务区域 | Restaurant、hotel、taxi——槽的集合。 |
| JGA | 联合目标准确率 | 每个槽都正确的轮次占比。全有或全无。 |
| MultiWOZ | 基准测试 | 多领域 WOZ 数据集；标准的 DST 评估。 |
| 无本体论 DST | 无模式 | 直接生成槽名称和值，无固定列表。 |
| 修正 | "Actually..." | 覆写先前填充槽的轮次。 |

## 扩展阅读

- [Budzianowski et al. (2018). MultiWOZ — A Large-Scale Multi-Domain Wizard-of-Oz](https://arxiv.org/abs/1810.00278) — 规范基准。
- [Feng et al. (2023). Towards LLM-driven Dialogue State Tracking (LDST)](https://arxiv.org/abs/2310.14970) — LLaMA + LoRA 指令调优用于 DST。
- [Heck et al. (2020). TripPy — A Triple Copy Strategy for Value Independent Neural Dialog State Tracking](https://arxiv.org/abs/2005.02877) — 基于复制的 DST 主力。
- [King, Flanigan (2024). Unsupervised End-to-End Task-oriented Dialogue with LLMs](https://arxiv.org/abs/2404.10753) — 基于 EM 的无监督 TOD。
- [MultiWOZ leaderboard](https://github.com/budzianowski/multiwoz) — 规范 DST 结果。
