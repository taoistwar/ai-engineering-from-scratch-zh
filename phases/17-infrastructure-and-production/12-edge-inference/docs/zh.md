# 边缘推理 — Apple Neural Engine、Qualcomm Hexagon、WebGPU/WebLLM、Jetson

> 核心边缘约束是内存带宽，而非计算。移动 DRAM 在 50-90 GB/s；数据中心 HBM3 达到 2-3 TB/s — 30-50 倍差距。解码是内存受限的，因此这一差距是决定性的。2026 年格局分四路。Apple M4/A18 Neural Engine 峰值 38 TOPS，统一内存（无 CPU↔NPU 拷贝）。Qualcomm Snapdragon X Elite / 8 Gen 4 Hexagon 达到 45 TOPS。WebGPU + WebLLM 在 M3 Max 上以约 41 tok/s 运行 Llama 3.1 8B (Q4)（约 70-80% 的原生速度）；17.6k GitHub 星星，兼容 OpenAI 的 API，约 70-75% 移动端覆盖率。NVIDIA Jetson Orin Nano Super (8GB) 可容纳 Llama 3.2 3B / Phi-3；AGX Orin 通过 vLLM 以约 40 tok/s 运行 gpt-oss-20b；Jetson T4000 (JetPack 7.1) 是 AGX Orin 的 2 倍。TensorRT Edge-LLM 支持 EAGLE-3、NVFP4、分块预填充 — 在 CES 2026 由 Bosch、ThunderSoft、MediaTek 展示。

**Type:** Learn
**Languages:** Python (stdlib, toy bandwidth-bound decode simulator)
**Prerequisites:** Phase 17 · 04 (vLLM Serving Internals), Phase 17 · 09 (Production Quantization)
**Time:** ~60 minutes

## 学习目标

- 解释为什么移动 LLM 推理是内存带宽受限的，计算是次要的。
- 列举四个边缘目标（Apple ANE、Qualcomm Hexagon、WebGPU/WebLLM、NVIDIA Jetson）并将每个匹配到一个用例。
- 说出 2026 年 WebGPU 覆盖率缺口（Firefox Android 赶上中）和 Safari iOS 26 的上线。
- 为每个目标选择量化格式（ANE 的 Core ML INT4 + FP16、Hexagon 的 QNN INT8/INT4、浏览器的 WebGPU Q4、Jetson Thor 的 NVFP4）。

## 问题

客户想要一个设备端聊天机器人：语音优先、默认私密、离线工作。在 MacBook Pro M3 Max 上，Llama 3.1 8B Q4 以约 55 tok/s 运行 — 可以。在 iPhone 16 Pro 上，同一模型以 3 tok/s 运行 — 不行。在配备 Snapdragon 8 Gen 3 的中端 Android 上，7 tok/s。在 Chrome Android v121+ 浏览器中通过 WebGPU，4-8 tok/s 取决于设备。

吞吐量差异不是移植问题。是带宽差距乘以量化格式再乘以 NPU 是否可从用户空间访问。2026 年的边缘推理是四个不同的问题，有四种不同的解决方案。

## 概念

### 带宽才是真正的天花板

解码为每个 token 读取完整的权重集合。一个 Q4 的 7B 模型是 3.5 GB。以 50 GB/s 读取 3.5 GB 需要 70 毫秒 — 理论天花板约为 14 tok/s。在 90 GB/s（高端移动 DRAM）天花板升至约 25 tok/s。再多的计算也无法帮助低于这个数字。

数据中心 HBM3 在 3 TB/s 下以 1.2 毫秒清除同样的 3.5 GB — 天花板是 830 tok/s。相同模型，相同权重。不同的内存子系统。

### Apple Neural Engine (M4 / A18)

- 最高 38 TOPS。统一内存（CPU 和 ANE 共享同一内存池）— 无拷贝开销。
- 通过 Core ML + 编译的 `.mlmodel` 模型访问，或通过 PyTorch 的 Metal Performance Shaders (MPS)。
- Llama.cpp Metal 后端使用 MPS，而非直接使用 ANE；原生 ANE 需要 Core ML 转换。
- 2026 年 iOS 应用的最佳实用路径：Core ML 使用 INT4 权重 + FP16 激活值。

### Qualcomm Hexagon (Snapdragon X Elite / 8 Gen 4)

- 最高 45 TOPS。在 SoC 中与 CPU 和 GPU 集成但分离的内存域。
- QNN (Qualcomm Neural Network) SDK 和 AI Hub 提供从 PyTorch/ONNX 的转换。
- 聊天模板、Llama 3.2、Phi-3 都作为一等制品在 AI Hub 上提供。

### Intel / AMD NPUs (Lunar Lake、Ryzen AI 300)

- 40-50 TOPS。软件落后于 Apple/Qualcomm；OpenVINO 在改进但小众。
- 最适合 Windows ARM Copilot 应用；AMD/Intel 桌面上原生用于本地优先。

### WebGPU + WebLLM

- 通过 WebGPU 计算着色器在浏览器中运行模型；无需安装。
- Llama 3.1 8B Q4 在 M3 Max 上约 41 tok/s — 约 70-80% 的原生速度。
- WebLLM 上 17.6k GitHub 星星；兼容 OpenAI 的 JS API；Apache 2.0。
- 2026 年覆盖率：Chrome Android v121+、Safari iOS 26 GA、Firefox Android 仍在追赶。总体约 70-75% 移动端覆盖率。

### NVIDIA Jetson 家族

- Orin Nano Super (8GB)：适合 Llama 3.2 3B、Phi-3 以不错的 tok/s。
- AGX Orin：通过 vLLM 以约 40 tok/s 运行 gpt-oss-20b。
- Thor / T4000 (JetPack 7.1)：2 倍 AGX Orin 性能，支持 EAGLE-3 和 NVFP4。
- TensorRT Edge-LLM (2026) 支持 EAGLE-3 推测解码、NVFP4 权重、分块预填充 — 数据中心优化移植到边缘。

### 每种目标的量化选择

| 目标 | 格式 | 备注 |
|--------|--------|-------|
| Apple ANE | INT4 权重 + FP16 激活值 | Core ML 转换路径 |
| Qualcomm Hexagon | QNN INT8 / INT4 | AI Hub 转换器 |
| WebGPU / WebLLM | Q4 MLC (q4f16_1) | 使用 `mlc_llm convert_weight` + 编译的 `.wasm`；不支持 GGUF |
| Jetson Orin Nano | Q4 GGUF 或 TRT-LLM INT4 | 内存受限 |
| Jetson AGX / Thor | NVFP4 + FP8 KV | Edge-LLM 路径 |

### 边缘上的長上下文陷阱

Llama 3.1 的 128K 上下文是数据中心特性。在配备 8 GB RAM 的手机上，4 GB 模型 + 32K token 的 2 GB KV 缓存 + OS 开销 = OOM。边缘部署保持上下文在 4K-8K，除非接受激进的 KV 量化（Q4 KV）。

### 语音是杀手级应用

语音代理对延迟敏感（首 token < 500 毫秒）。本地推理完全消除网络延迟。结合语音转文本（Whisper Turbo 变体在边缘运行），边缘推理成为生产质量的语音循环。

### 你应该记住的数字

- Apple M4 / A18 ANE：38 TOPS。
- Qualcomm Hexagon SD X Elite：45 TOPS。
- WebLLM M3 Max：Llama 3.1 8B Q4 上约 41 tok/s。
- AGX Orin：通过 vLLM 在 gpt-oss-20b 上约 40 tok/s。
- 数据中心-边缘带宽差距：30-50 倍。
- WebGPU 移动覆盖率：~70-75%（Firefox Android 滞后）。

## 使用它

`code/main.py` 从带宽受限的数学计算出跨边缘目标的理论解码吞吐量天花板。与观测的基准比较，突出带宽而非计算才是瓶颈的地方。

## 交付它

本课产出 `outputs/skill-edge-target-picker.md`。给定平台（iOS/Android/浏览器/Jetson）、模型和延迟/内存预算，选择量化格式和转换管线。

## 练习

1. 运行 `code/main.py`。对于 Snapdragon 8 Gen 3（约 77 GB/s 带宽）上 Q4 的 7B 模型，计算解码天花板。与观测到的 6-8 tok/s 比较 — 运行时高效吗？
2. Android 上的 WebGPU 需要 Chrome v121+。为旧浏览器设计降级方案 — 通过相同兼容 OpenAI 的 API 的服务器端。
3. 你的 iOS 应用需要 4K 上下文流式传输。哪种模型/格式组合让你在 iPhone 16 上保持在 4 GB 活动内存以下？
4. Jetson AGX Orin 以 40 tok/s 运行 gpt-oss-20b。Jetson Nano 只能容纳 3B。如果你的产品同时针对两者，你如何统一推理栈？
5. 论证"WebLLM 在 2026 年是否生产就绪"。引用覆盖率、性能和 Firefox Android 缺口。

## 核心术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| ANE | "Apple 神经引擎" | M 系列和 A 系列中的设备端 NPU；统一内存 |
| Hexagon | "Qualcomm NPU" | Snapdragon NPU；QNN SDK 访问 |
| WebGPU | "浏览器 GPU" | W3C 标准化的浏览器 GPU API；Chrome/Safari 2026 |
| WebLLM | "浏览器 LLM 运行时" | MLC-LLM 项目；Apache 2.0；兼容 OpenAI 的 JS |
| Jetson | "NVIDIA 边缘" | Orin Nano / AGX / Thor / T4000 家族 |
| TRT Edge-LLM | "边缘 TensorRT" | 2026 年 TensorRT-LLM 的边缘移植；EAGLE-3 + NVFP4 |
| 统一内存 | "共享池" | CPU 和 NPU 看到相同的 RAM；无拷贝开销 |
| 带宽受限 | "内存限制" | 解码受每秒读取权重的字节数制约 |
| Core ML | "Apple 转换" | 用于 ANE 原生模型的 Apple 框架 |
| QNN | "Qualcomm 栈" | Qualcomm Neural Network SDK |

## 进一步阅读

- [设备端 LLM 2026 年现状](https://v-chandra.github.io/on-device-llms/) — 格局和基准。
- [NVIDIA Jetson 边缘 AI](https://developer.nvidia.com/blog/getting-started-with-edge-ai-on-nvidia-jetson-llms-vlms-and-foundation-models-for-robotics/) — Orin / AGX / Thor。
- [NVIDIA TensorRT Edge-LLM](https://developer.nvidia.com/blog/accelerating-llm-and-vlm-inference-for-automotive-and-robotics-with-nvidia-tensorrt-edge-llm/) — 2026 年边缘移植公告。
- [WebLLM (arXiv:2412.15803)](https://arxiv.org/html/2412.15803v2) — 设计和基准。
- [Apple Core ML](https://developer.apple.com/documentation/coreml) — ANE 原生转换。
- [Qualcomm AI Hub](https://aihub.qualcomm.com/) — Hexagon 的预转换模型。
