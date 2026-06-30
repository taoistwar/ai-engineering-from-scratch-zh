# 视觉-语言模型 — ViT-MLP-LLM 模式

> 视觉编码器将图像转换为 Token。MLP 投影器将这些 Token 映射到 LLM 的嵌入空间。语言模型完成其余工作。这个模式——ViT-MLP-LLM——是 2026 年每个生产 VLM。

**类型：** Learn + Use
**语言：** Python
**先修要求：** Phase 4 Lesson 14 (ViT), Phase 4 Lesson 18 (CLIP), Phase 7 Lesson 02 (Self-Attention)
**时间：** ~75 分钟

## 学习目标

- 陈述 ViT-MLP-LLM 架构并解释三个组件各自贡献什么
- 在参数数量、上下文长度和基准性能方面比较 Qwen3-VL、InternVL3.5、LLaVA-Next 和 GLM-4.6V
- 解释 DeepStack：为什么多层 ViT 特征比单个最后一层特征更好地收紧视觉-语言对齐
- 在生产中使用 Cross-Modal Error Rate (CMER) 测量 VLM 幻觉并对信号采取行动

## 问题

CLIP（Phase 4 Lesson 18）为图像和文本提供共享嵌入空间，这足以用于零样本分类和检索。它无法回答"这张图像中有多少辆红色汽车？"，因为 CLIP 不生成文本——它只给相似度打分。

视觉-语言模型（VLM）——Qwen3-VL、InternVL3.5、LLaVA-Next、GLM-4.6V——将 CLIP 家族图像编码器接入完整语言模型。模型看到一张图像加一个问题，生成答案。到 2026 年，开源 VLM 在多模态基准（MMMU、MMBench、DocVQA、ChartQA、MathVista、OSWorld）上匹敌或击败 GPT-5 和 Gemini-2.5-Pro。

三个组件的组（ViT、投影器、LLM）是标准。模型之间的差异在于哪个 ViT、哪个投影器、哪个 LLM、训练数据和对齐配方。一旦你理解了模式，交换任何组件都是机械的。

## 概念

### ViT-MLP-LLM 架构

```mermaid
flowchart LR
    IMG["图像<br/>(H x W x 3)"] --> ViT["视觉编码器<br/>(ViT, CLIP-L,<br/>SigLIP, DINOv3)"]
    ViT --> FEATS["图像 Token<br/>(N, d_vit)"]
    FEATS --> PROJ["投影器<br/>(2-4 层 MLP<br/>或 Q-former)"]
    PROJ --> VTOK["LLM 空间中的<br/>图像 Token<br/>(N, d_llm)"]
    TXT["文本 prompt"] --> TOK["LLM 分词器"]
    TOK --> TTOK["文本 Token<br/>(M, d_llm)"]
    VTOK --> CONCAT["交错<br/>或拼接"]
    TTOK --> CONCAT
    CONCAT --> LLM["解码器 LLM<br/>(Qwen3, LLaMA 等)"]
    LLM --> OUT["文本答案"]

    style ViT fill:#dbeafe,stroke:#2563eb
    style PROJ fill:#fef3c7,stroke:#d97706
    style LLM fill:#dcfce7,stroke:#16a34a
```

1. **视觉编码器**——预训练 ViT（CLIP-L/14、SigLIP、DINOv3 或微调变体）。产生 patch Token。
2. **投影器**——一个小模块（2-4 层 MLP 或 Q-former）将视觉 Token 映射到 LLM 的嵌入维度。大部分微调在这里发生。
3. **LLM**——仅解码器语言模型（Qwen3、Llama、Mistral、GLM、InternLM）。按顺序读取视觉 + 文本 Token，生成文本。

原则上三个组件都是可训练的。实践中，视觉编码器和 LLM 大部分保持冻结而投影器进行训练——几十亿参数的信号花很少的钱。

### DeepStack

原始投影仅使用最后一层 ViT。DeepStack（Qwen3-VL）从多个 ViT 深度采样特征并将其堆叠。深层携带高级语义；浅层携带细粒度空间和纹理信息。将两者都送入 LLM 缩小了"图像包含什么"（语义）和"确切在哪里"（空间 grounding）之间的差距。

### 三个训练阶段

现代 VLM 分阶段训练：

1. **对齐**——冻结 ViT 和 LLM。仅训练投影器在图像-字幕对上。教投影器将视觉空间映射到语言空间。
2. **预训练**——解冻一切。在大规模交错图像-文本数据上训练（5 亿+对）。构建模型的视觉知识。
3. **指令微调**——在精选的（图像，问题，答案）三元组上微调。教对话行为和任务格式。这是将"视觉感知 LM"变成可用助手的关键。

大多数 LoRA 微调针对阶段 3，使用小型标注数据集。

### 模型家族比较（2026 年初）

| 模型 | 参数 | 视觉编码器 | LLM | 上下文 | 优势 |
|-------|--------|----------------|-----|---------|-----------|
| Qwen3-VL-235B-A22B (MoE) | 235B (22B active) | custom ViT + DeepStack | Qwen3 | 256K | 通用 SOTA, GUI agent |
| Qwen3-VL-30B-A3B (MoE) | 30B (3B active) | custom ViT + DeepStack | Qwen3 | 256K | 较小的 MoE 替代 |
| Qwen3-VL-8B (dense) | 8B | custom ViT | Qwen3 | 128K | 生产稠密默认 |
| InternVL3.5-38B | 38B | InternViT-6B | Qwen3 + GPT-OSS | 128K | 强 MMBench / MMVet |
| InternVL3.5-241B-A28B | 241B (28B active) | InternViT-6B | Qwen3 | 128K | 与 GPT-4o 竞争 |
| LLaVA-Next 72B | 72B | SigLIP | Llama-3 | 32K | 开源，易于微调 |
| GLM-4.6V | ~70B | custom | GLM | 64K | 开源，强 OCR |
| MiniCPM-V-2.6 | 8B | SigLIP | MiniCPM | 32K | 边缘友好 |

### 视觉 agent

Qwen3-VL-235B 在 OSWorld 上达到顶级全局性能——OSWorld 是一个操作 GUI（桌面、移动、Web）的**视觉 agent**基准。模型看到截图，理解 UI，并发出动作（点击、键入、滚动）。结合工具，它闭合了常见桌面任务的循环。这是大多数 2026"AI PC"演示背后运行的机制。

### Agent 能力 + RoPE 变体

VLM 需要知道一帧**什么时候**在视频中。Qwen3-VL 从 T-RoPE（时序旋转位置嵌入）演进到**基于文本的时间对齐**——与视频帧交错显式时间戳文本 Token。模型看到"`<timestamp 00:32>` 帧, prompt"，并能推理时序关系。

### 对齐问题

爬取数据集中 12% 的图像-文本对包含不完全基于图像的描述。在此之上训练的 VLM 会静默学习到幻觉——虚构对象、误读数字、发明关系。在生产中这是主要失败模式。

Skywork.ai 引入 **Cross-Modal Error Rate (CMER)** 来跟踪它：

```
CMER = 文本置信度高但图像-文本相似度（通过 CLIP 家族检查器）低的输出比例
```

高 CMER 意味着模型在自信地说出未基于图像的内容。监控 CMER 并将其视为生产 KPI，他们的幻觉率在部署中削减了约 35%。技巧不是"修复模型"而是"将高 CMER 输出路由到人工审核。"

### 使用 LoRA / QLoRA 微调

完整微调 70B VLM 对大多数团队来说遥不可及。LoRA（rank 16-64）在注意力 + 投影器层上，或 QLoRA 带 4 位基础权重，可放入单个 A100 / H100。成本：5,000-50,000 示例，$100-$5,000 计算，2-10 小时训练。

### 空间推理仍然薄弱

当前 VLM 在空间推理基准上得分 50-60%（上-下、左-右、计数、距离）。如果你的用例依赖于"哪个物体在上面"，充分验证——通用 VLM 性能低于人类。纯空间任务的优于 VLM 替代方案：专用关键点/姿态估计器、深度模型或带框几何后处理的检测模型。

## Build It

### 步骤 1：投影器

你将最常训练的部分。2-4 层 MLP 带 GELU。

```python
import torch
import torch.nn as nn


class Projector(nn.Module):
    def __init__(self, vit_dim=768, llm_dim=4096, hidden=4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vit_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, llm_dim),
        )

    def forward(self, x):
        return self.net(x)
```

输入是 `(N_patches, d_vit)` Token 张量。输出是 `(N_patches, d_llm)`。LLM 将每个输出行视为另一个 Token。

### 步骤 2：端到端组装 ViT-MLP-LLM

最小 VLM 前向传播的骨架。真实代码使用 `transformers`；这是概念布局。

```python
class MinimalVLM(nn.Module):
    def __init__(self, vit, projector, llm, image_token_id):
        super().__init__()
        self.vit = vit
        self.projector = projector
        self.llm = llm
        self.image_token_id = image_token_id  # 文本 prompt 中的占位 Token

    def forward(self, image, input_ids, attention_mask):
        # 1. 视觉特征
        vision_tokens = self.vit(image)                     # (B, N_patches, d_vit)
        vision_embeds = self.projector(vision_tokens)       # (B, N_patches, d_llm)

        # 2. 文本嵌入
        text_embeds = self.llm.get_input_embeddings()(input_ids)  # (B, M, d_llm)

        # 3. 用视觉嵌入替换图像占位 Token
        merged = self._merge(text_embeds, vision_embeds, input_ids)

        # 4. 运行 LLM
        return self.llm(inputs_embeds=merged, attention_mask=attention_mask)

    def _merge(self, text_embeds, vision_embeds, input_ids):
        out = text_embeds.clone()
        expected = vision_embeds.size(1)
        for b in range(input_ids.size(0)):
            positions = (input_ids[b] == self.image_token_id).nonzero(as_tuple=True)[0]
            if len(positions) != expected:
                raise ValueError(
                    f"batch item {b} 有 {len(positions)} 个图像 Token 但 vision_embeds 有 {expected} 个 patches。"
                    "批次中的每个样本必须预填充到相同数量的图像占位 Token。")
            out[b, positions] = vision_embeds[b]
        return out
```

文本中的 `<image>` 占位 Token 将被替换为真实图像嵌入——LLaVA、Qwen-VL 和 InternVL 使用的相同模式。

### 步骤 3：CMER 计算

轻量级运行时检查。

```python
import torch.nn.functional as F


def cross_modal_error_rate(image_emb, text_emb, text_confidence, sim_threshold=0.25, conf_threshold=0.8):
    """
    image_emb, text_emb: 图像和生成文本的嵌入（内部归一化）
    text_confidence:     每 Token 平均概率，在 [0, 1] 中
    返回:                高置信度但图像-文本对齐度低的输出比例
    """
    image_emb = F.normalize(image_emb, dim=-1)
    text_emb = F.normalize(text_emb, dim=-1)
    sim = (image_emb * text_emb).sum(dim=-1)        # 余弦相似度
    high_conf_low_sim = (text_confidence > conf_threshold) & (sim < sim_threshold)
    return high_conf_low_sim.float().mean().item()
```

将 CMER 视为生产 KPI。按端点、按 prompt 类型、按客户监控它。上升的 CMER 表示模型开始在某个输入分布上产生幻觉。

### 步骤 4：玩具 VLM 分类器（可运行）

演示投影器训练。假的"ViT 特征"进入；一个微型 LLM 风格的 Token 预测类别。

```python
class ToyVLM(nn.Module):
    def __init__(self, vit_dim=32, llm_dim=64, num_classes=5):
        super().__init__()
        self.projector = Projector(vit_dim, llm_dim, hidden=64)
        self.head = nn.Linear(llm_dim, num_classes)

    def forward(self, vision_tokens):
        projected = self.projector(vision_tokens)
        pooled = projected.mean(dim=1)
        return self.head(pooled)
```

可以在合成（特征，类别）对上在 200 步内拟合——足以展示投影器模式的工作方式。

## Use It

2026 年生产团队使用 VLM 的三种方式：

- **托管 API**——OpenAI Vision、Anthropic Claude Vision、Google Gemini Vision。零基础设施，供应商风险。
- **开源自托管**——Qwen3-VL 或 InternVL3.5 通过 `transformers` 和 `vllm`。完全控制，更高的前期投入。
- **在领域上微调**——加载 Qwen2.5-VL-7B 或 LLaVA-1.6-7B，在 5k-50k 自定义示例上 LoRA，用 `vllm` 或 `TGI` serving。

```python
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
from PIL import Image

model_id = "Qwen/Qwen3-VL-8B-Instruct"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForVision2Seq.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": Image.open("plot.png")},
        {"type": "text", "text": "这张图表显示了什么？"},
    ],
}]
inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
generated = model.generate(**inputs, max_new_tokens=256)
answer = processor.decode(generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
```

`apply_chat_template` 隐藏了 `<image>` 占位 Token 的分词化；模型内部处理合并。

## Ship It

本课产出：

- `outputs/prompt-vlm-selector.md`——根据准确率、延迟、上下文长度和预算选择 Qwen3-VL / InternVL3.5 / LLaVA-Next / API。
- `outputs/skill-cmer-monitor.md`——生成代码以用跨模态错误率、逐端点仪表板和告警阈值对生产 VLM 端点进行仪表化。

## 练习

1. **（简单）** 在五张图像上通过任何开源 VLM 运行三个 prompt（"这是什么？"、"数物体"、"描述场景"）。手动将每个答案评分为正确 / 部分正确 / 幻觉。计算首个 CMER 类似的率。
2. **（中等）** 在目标领域 500 张带字幕的图像上用 LoRA（rank 16）微调 Qwen2.5-VL-3B 或 LLaVA-1.6-7B。比较零样本与微调的 MMBench 风格准确率。
3. **（困难）** 将 VLM 的图像编码器替换为 DINOv3 而非其默认的 SigLIP/CLIP。只重新训练投影器（冻结 LLM + 冻结 DINOv3）。测量稠密预测任务（计数、空间推理）是否改善。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|----------------------|
| ViT-MLP-LLM | "VLM 模式" | 视觉编码器 + 投影器 + 语言模型；每个 2026 年 VLM |
| 投影器 | "桥梁" | 2-4 层 MLP（或 Q-former）将视觉 Token 映射到 LLM 嵌入空间 |
| DeepStack | "Qwen3-VL 特征技巧" | 堆叠多级 ViT 特征而非仅最后一层 |
| 图像 Token | "<image> 占位符" | 文本流中的特殊 Token，被投影的视觉嵌入替换 |
| CMER | "幻觉 KPI" | Cross-Modal Error Rate；文本置信度高但图像-文本相似度低时为高 |
| 视觉 agent | "会点击的 VLM" | 操作 GUI（OSWorld、移动、Web）并带有工具调用的 VLM |
| Q-former | "固定数量 Token 桥" | BLIP-2 风格投影器，产生固定数量的视觉查询 Token |
| 对齐 / 预训练 / 指令微调 | "三阶段" | 标准 VLM 训练 pipeline |

## 延伸阅读

- [Qwen3-VL Technical Report (arXiv 2511.21631)](https://arxiv.org/abs/2511.21631)
- [InternVL3.5 Advancing Open-Source Multimodal Models (arXiv 2508.18265)](https://arxiv.org/html/2508.18265v1)
- [LLaVA-Next 系列](https://llava-vl.github.io/blog/2024-05-10-llava-next-stronger-llms/)
- [BentoML: Best Open-Source VLMs 2026](https://www.bentoml.com/blog/multimodal-ai-a-guide-to-open-source-vision-language-models)
- [MMMU: Multi-discipline Multimodal Understanding benchmark](https://mmmu-benchmark.github.io/)
- [VLMs in manufacturing (Robotics Tomorrow, March 2026)](https://www.roboticstomorrow.com/story/2026/03/when-machines-learn-to-see-like-experts-the-rise-of-vision-language-models-in-manufacturing/26335/)
