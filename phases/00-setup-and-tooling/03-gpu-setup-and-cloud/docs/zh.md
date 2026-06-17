# GPU 设置与云服务

> 用 CPU 学习没问题，真正训练需要 GPU。

**类型:** Build
**语言:** Python
**前置要求:** Phase 0, Lesson 01
**时间:** ~45 分钟

## 学习目标

- 使用 `nvidia-smi` 和 PyTorch 的 CUDA API 验证本地 GPU 是否可用
- 配置 Google Colab 使用 T4 GPU 进行免费的云端实验
- 对比 CPU 和 GPU 上的矩阵乘法性能，测量加速比
- 使用 fp16 经验法则估算你的显存能装下的最大模型

## 问题所在

阶段 1-3 的大多数课程在 CPU 上运行即可。但一旦开始训练 CNN、Transformer 或 LLM（阶段 4+），就需要 GPU 加速。一个在 CPU 上需要 8 小时的训练任务，在 GPU 上只需 10 分钟。

你有三个选择：本地 GPU、云 GPU 或 Google Colab（免费）。

## 概念

```
你的选择：

1. 本地 NVIDIA GPU
   费用：$0（你已经有了）
   配置：安装 CUDA + cuDNN
   最适合：日常使用、大型数据集

2. Google Colab（免费版）
   费用：$0
   配置：无需配置
   最适合：快速实验、家里没有 GPU

3. 云 GPU（Lambda、RunPod、Vast.ai）
   费用：$0.20-2.00/小时
   配置：SSH + 安装
   最适合：正式训练、大型模型
```

## 动手做

### 选项 1：本地 NVIDIA GPU

检查你是否有一块：

```bash
nvidia-smi
```

安装支持 CUDA 的 PyTorch：

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### 选项 2：Google Colab

1. 访问 [colab.research.google.com](https://colab.research.google.com)
2. Runtime > Change runtime type > T4 GPU
3. 运行 `!nvidia-smi` 验证

将本课程中的 notebook 直接上传到 Colab 即可。

### 选项 3：云 GPU

对于 Lambda Labs、RunPod 或 Vast.ai：

```bash
ssh user@your-gpu-instance

pip install torch torchvision torchaudio
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### 没有 GPU？没问题。

大多数课程在 CPU 上也能运行。需要 GPU 的课程会注明并附上 Colab 链接。

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
```

## 动手做：GPU vs CPU 基准测试

```python
import torch
import time

size = 5000

a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

start = time.time()
c_cpu = a_cpu @ b_cpu
cpu_time = time.time() - start
print(f"CPU: {cpu_time:.3f}s")

if torch.cuda.is_available():
    a_gpu = a_cpu.to("cuda")
    b_gpu = b_cpu.to("cuda")

    torch.cuda.synchronize()
    start = time.time()
    c_gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU: {gpu_time:.3f}s")
    print(f"Speedup: {cpu_time / gpu_time:.0f}x")
```

## 练习

1. 运行上面的基准测试，对比 CPU 和 GPU 的时间
2. 如果没有 GPU，在 Google Colab 上运行并对比
3. 查看你有多少 GPU 显存，估算能装下的最大模型（经验法则：fp16 下每个参数 2 字节）

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| CUDA | "GPU 编程" | NVIDIA 的并行计算平台，让你能在 GPU 上运行代码 |
| VRAM | "显存" | GPU 上的视频内存，独立于系统内存，决定了能装多大的模型 |
| fp16 | "半精度" | 16 位浮点数，内存占用是 fp32 的一半，精度损失极小 |
| Tensor Core | "快速矩阵硬件" | GPU 上专门用于矩阵运算的核心，比普通核心快 4-8 倍 |
