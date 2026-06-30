# 权重初始化与训练稳定性

> 初始化错误，训练永远不会开始。初始化正确，50 层也能像 3 层一样顺利训练。

**类型：** 构建
**语言：** Python
**前置条件：** 第 03.04 课（激活函数），第 03.07 课（正则化）
**时间：** 约 90 分钟

## 学习目标

- 实现零初始化、随机初始化、Xavier/Glorot 初始化和 Kaiming/He 初始化策略，并测量它们对 50 层激活幅度的影响
- 推导为什么 Xavier 初始化使用 Var(w) = 2/(fan_in + fan_out)，而 Kaiming 使用 Var(w) = 2/fan_in
- 演示零初始化的对称性问题，并解释为什么仅靠随机缩放是不够的
- 将正确的初始化策略与激活函数匹配：Xavier 用于 sigmoid/tanh，Kaiming 用于 ReLU/GELU

## 问题

将所有权重初始化为零。什么都学不到。每个神经元计算相同的函数，接收相同的梯度，并以相同的方式更新。经过 10,000 个 epoch，你的 512 神经元隐藏层仍然是同一个神经元的 512 个副本。你为 512 个参数付费，只得到了 1 个。

将权重初始化为太大。激活值在网络上爆炸。到第 10 层，值达到 1e15。到第 20 层，它们溢出到无穷。梯度在反向中遵循相同的轨迹。

将权重从标准正态分布随机初始化。对 3 层有效。到 50 层，信号根据不同情况（随机缩放稍小或稍大）坍缩到零或引爆到无穷。"有效"和"破损"之间的边界薄如剃刀。

权重初始化是深度学习中最被低估的决策。架构有论文。优化器有博客文章。初始化只是一个脚注。但搞错了，其他一切都不重要——你的网络在训练开始前就已经死了。

## 概念

### 对称性问题

层中的每个神经元具有相同的结构：将输入乘以权重，加偏置，应用激活。如果所有权重从相同的值开始（零是极端情况），每个神经元计算相同的输出。在反向传播期间，每个神经元接收相同的梯度。在更新步骤中，每个神经元改变相同的量。

你被困住了。网络有数百个参数，但它们全部步调一致地移动。这被称为对称性，随机初始化是打破它的蛮力方式。每个神经元在权重空间中从不同的点开始，因此每个学习不同的特征。

但"随机"是不够的。随机性的*尺度*决定网络是否能够训练。

### 方差在层间的传播

考虑一个具有 fan_in 个输入的单层：

```
z = w1*x1 + w2*x2 + ... + w_n*x_n
```

如果每个权重 wi 从方差为 Var(w) 的分布中抽取，每个输入 xi 具有方差 Var(x)，则输出方差为：

```
Var(z) = fan_in * Var(w) * Var(x)
```

如果 Var(w) = 1 且 fan_in = 512，输出方差是输入方差的 512 倍。经过 10 层：512^10 = 1.2e27。你的信号已经爆炸。

如果 Var(w) = 0.001，每层输出方差收缩 0.001 * 512 = 0.512 倍。经过 10 层：0.512^10 = 0.00013。你的信号已经消失。

目标：选择 Var(w) 使 Var(z) = Var(x)。信号幅度在层间保持恒定。

### Xavier/Glorot 初始化

Glorot 和 Bengio（2010）为 sigmoid 和 tanh 激活函数推导了解法。为了在前向和反向传播中保持方差不变：

```
Var(w) = 2 / (fan_in + fan_out)
```

在实践中，权重从以下分布中抽取：

```
w ~ Uniform(-limit, limit)  其中 limit = sqrt(6 / (fan_in + fan_out))
```

或：

```
w ~ Normal(0, sqrt(2 / (fan_in + fan_out)))
```

这有效是因为 sigmoid 和 tanh 在零附近大致是线性的，而正确初始化的激活值就处于该区域。方差在数十层内保持稳定。

### Kaiming/He 初始化

ReLU 杀死一半的输出（所有负值变为零）。有效 fan_in 减半，因为平均一半的输入被归零。Xavier 初始化没有考虑到这一点——它低估了所需的方差。

He 等人（2015）调整了公式：

```
Var(w) = 2 / fan_in
```

权重从以下分布中抽取：

```
w ~ Normal(0, sqrt(2 / fan_in))
```

因子 2 补偿了 ReLU 将一半激活值归零。没有它，信号每层收缩约 0.5 倍。经过 50 层：0.5^50 = 8.8e-16。Kaiming 初始化防止了这一点。

### Transformer 初始化

GPT-2 引入了一种不同的模式。残差连接将每个子层的输出加到其输入上：

```
x = x + sublayer(x)
```

每次加法增加方差。有 N 个残差层时，方差与 N 成比例增长。GPT-2 将残差层的权重缩放 1/sqrt(2N)，其中 N 是层数。这保持了累积信号幅度的稳定。

Llama 3（4050 亿参数，126 层）使用类似的方案。没有这种缩放，残差流将在 126 层注意力和前馈块中无界增长。

```mermaid
flowchart TD
    subgraph "零初始化"
        Z1["第 1 层<br/>所有权重 = 0"] --> Z2["第 2 层<br/>所有神经元相同"]
        Z2 --> Z3["第 3 层<br/>仍然相同"]
        Z3 --> ZR["结果：无论多宽<br/>只有 1 个有效神经元"]
    end

    subgraph "Xavier 初始化"
        X1["第 1 层<br/>Var = 2/(fan_in+fan_out)"] --> X2["第 2 层<br/>信号稳定"]
        X2 --> X3["第 50 层<br/>信号稳定"]
        X3 --> XR["结果：使用<br/>sigmoid/tanh 可训练"]
    end

    subgraph "Kaiming 初始化"
        K1["第 1 层<br/>Var = 2/fan_in"] --> K2["第 2 层<br/>信号稳定"]
        K2 --> K3["第 50 层<br/>信号稳定"]
        K3 --> KR["结果：使用<br/>ReLU/GELU 可训练"]
    end
```

### 50 层后的激活幅度

```mermaid
graph LR
    subgraph "平均激活幅度"
        direction LR
        L1["第 1 层"] --> L10["第 10 层"] --> L25["第 25 层"] --> L50["第 50 层"]
    end

    subgraph "结果"
        R1["Random N(0,1)：第 5 层爆炸"]
        R2["Random N(0,0.01)：第 10 层消失"]
        R3["Xavier + Sigmoid：第 50 层 ~1.0"]
        R4["Kaiming + ReLU：第 50 层 ~1.0"]
    end
```

### 选择正确的初始化

```mermaid
flowchart TD
    Start["什么激活函数？"] --> Act{"激活类型？"}

    Act -->|"Sigmoid / Tanh"| Xavier["Xavier/Glorot<br/>Var = 2/(fan_in + fan_out)"]
    Act -->|"ReLU / Leaky ReLU"| Kaiming["Kaiming/He<br/>Var = 2/fan_in"]
    Act -->|"GELU / Swish"| Kaiming2["Kaiming/He<br/>（同 ReLU）"]
    Act -->|"Transformer 残差"| GPT["缩放 1/sqrt(2N)<br/>N = 层数"]

    Xavier --> Check["验证：激活幅度<br/>在所有层中保持<br/>在 0.5 到 2.0 之间"]
    Kaiming --> Check
    Kaiming2 --> Check
    GPT --> Check
```

```figure
weight-init-variance
```

## 构建它

### 步骤 1：初始化策略

初始化权重矩阵的四种方式。每种返回一个 list of lists（2D 矩阵），具有 fan_in 列和 fan_out 行。

```python
import math
import random


def zero_init(fan_in, fan_out):
    return [[0.0 for _ in range(fan_in)] for _ in range(fan_out)]


def random_init(fan_in, fan_out, scale=1.0):
    return [[random.gauss(0, scale) for _ in range(fan_in)] for _ in range(fan_out)]


def xavier_init(fan_in, fan_out):
    std = math.sqrt(2.0 / (fan_in + fan_out))
    return [[random.gauss(0, std) for _ in range(fan_in)] for _ in range(fan_out)]


def kaiming_init(fan_in, fan_out):
    std = math.sqrt(2.0 / fan_in)
    return [[random.gauss(0, std) for _ in range(fan_in)] for _ in range(fan_out)]
```

### 步骤 2：激活函数

我们需要 sigmoid、tanh 和 ReLU 来测试每种初始化策略与其对应的激活函数。

```python
def sigmoid(x):
    x = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-x))


def tanh_act(x):
    return math.tanh(x)


def relu(x):
    return max(0.0, x)
```

### 步骤 3：通过 50 层的前向传播

将随机数据通过深度网络，并测量每层的平均激活幅度。

```python
def forward_deep(init_fn, activation_fn, n_layers=50, width=64, n_samples=100):
    random.seed(42)
    layer_magnitudes = []

    inputs = [[random.gauss(0, 1) for _ in range(width)] for _ in range(n_samples)]

    for layer_idx in range(n_layers):
        weights = init_fn(width, width)
        biases = [0.0] * width

        new_inputs = []
        for sample in inputs:
            output = []
            for neuron_idx in range(width):
                z = sum(weights[neuron_idx][j] * sample[j] for j in range(width)) + biases[neuron_idx]
                output.append(activation_fn(z))
            new_inputs.append(output)
        inputs = new_inputs

        magnitudes = []
        for sample in inputs:
            magnitudes.append(sum(abs(v) for v in sample) / width)
        mean_mag = sum(magnitudes) / len(magnitudes)
        layer_magnitudes.append(mean_mag)

    return layer_magnitudes
```

### 步骤 4：实验

运行所有组合：零初始化、随机 N(0,1)、随机 N(0,0.01)、Xavier + sigmoid、Xavier + tanh、Kaiming + ReLU。打印关键层的幅度。

```python
def run_experiment():
    configs = [
        ("Zero init + Sigmoid", lambda fi, fo: zero_init(fi, fo), sigmoid),
        ("Random N(0,1) + ReLU", lambda fi, fo: random_init(fi, fo, 1.0), relu),
        ("Random N(0,0.01) + ReLU", lambda fi, fo: random_init(fi, fo, 0.01), relu),
        ("Xavier + Sigmoid", xavier_init, sigmoid),
        ("Xavier + Tanh", xavier_init, tanh_act),
        ("Kaiming + ReLU", kaiming_init, relu),
    ]

    print(f"{'策略':<30} {'L1':>10} {'L5':>10} {'L10':>10} {'L25':>10} {'L50':>10}")
    print("-" * 80)

    for name, init_fn, act_fn in configs:
        mags = forward_deep(init_fn, act_fn)
        row = f"{name:<30}"
        for idx in [0, 4, 9, 24, 49]:
            val = mags[idx]
            if val > 1e6:
                row += f" {'爆炸':>10}"
            elif val < 1e-6:
                row += f" {'消失':>10}"
            else:
                row += f" {val:>10.4f}"
        print(row)
```

### 步骤 5：对称性演示

展示零初始化产生相同的神经元。

```python
def symmetry_demo():
    random.seed(42)
    weights = zero_init(2, 4)
    biases = [0.0] * 4

    inputs = [0.5, -0.3]
    outputs = []
    for neuron_idx in range(4):
        z = sum(weights[neuron_idx][j] * inputs[j] for j in range(2)) + biases[neuron_idx]
        outputs.append(sigmoid(z))

    print("\n对称性演示（4 个神经元，零初始化）:")
    for i, out in enumerate(outputs):
        print(f"  神经元 {i}: output = {out:.6f}")
    all_same = all(abs(outputs[i] - outputs[0]) < 1e-10 for i in range(len(outputs)))
    print(f"  全部相同: {all_same}")
    print(f"  有效参数: 1 (不是 {len(weights) * len(weights[0])})")
```

### 步骤 6：逐层幅度报告

打印 50 层激活幅度的可视化柱状图。

```python
def magnitude_report(name, magnitudes):
    print(f"\n{name}:")
    for i, mag in enumerate(magnitudes):
        if i % 5 == 0 or i == len(magnitudes) - 1:
            if mag > 1e6:
                bar = "X" * 50 + " 爆炸"
            elif mag < 1e-6:
                bar = "." + " 消失"
            else:
                bar_len = min(50, max(1, int(mag * 10)))
                bar = "#" * bar_len
            print(f"  第 {i+1:3d} 层: {bar} ({mag:.6f})")
```

## 使用它

PyTorch 以内置函数的形式提供这些：

```python
import torch
import torch.nn as nn

layer = nn.Linear(512, 256)

nn.init.xavier_uniform_(layer.weight)
nn.init.xavier_normal_(layer.weight)

nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')

nn.init.zeros_(layer.bias)
```

当你调用 `nn.Linear(512, 256)` 时，PyTorch 默认使用 Kaiming 均匀初始化。这就是为什么大多数简单网络"直接能用"——PyTorch 已经做了正确的选择。但当你构建自定义架构或超过 20 层时，你需要理解正在发生的事情并可能覆盖默认值。

对于 transformers，HuggingFace 模型通常在其 `_init_weights` 方法中处理初始化。GPT-2 的实现将残差投影缩放 1/sqrt(N)。如果你从零开始构建一个 transformer，你需要自己添加这个。

## 交付物

本课产出：
- `outputs/prompt-init-strategy.md`——一个诊断权重初始化问题并推荐正确策略的提示词

## 练习

1. 添加 LeCun 初始化（Var = 1/fan_in，为 SELU 激活设计）。使用 LeCun init + tanh 运行 50 层实验，并与 Xavier + tanh 进行比较。

2. 实现 GPT-2 残差缩放：在添加到残差流之前，将每层输出乘以 1/sqrt(2*N)。运行 50 层，有和没有缩放，测量残差幅度增长的速度。

3. 创建一个"初始化健康检查"函数，接收网络的层维度和激活类型，然后推荐正确的初始化，并在当前初始化会导致问题时发出警告。

4. 使用 fan_in = 16 vs fan_in = 1024 运行实验。Xavier 和 Kaiming 会适应 fan_in，但随机初始化不会。展示随着更大层，"有效"和"破损"之间的差距如何扩大。

5. 实现正交初始化（生成随机矩阵，计算其 SVD，使用正交矩阵 U）。在 50 层 ReLU 网络中与 Kaiming 比较。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|----------|
| 权重初始化 | "随机设置起始权重" | 选择初始权重值的策略，决定了网络是否能够训练 |
| 对称性打破 | "让神经元不同" | 使用随机初始化确保神经元学习不同的特征，而不是计算相同的函数 |
| Fan-in | "神经元的输入数量" | 传入连接的数量，决定输入方差如何在加权和中累积 |
| Fan-out | "神经元的输出数量" | 传出连接的数量，与在反向传播期间保持梯度方差相关 |
| Xavier/Glorot 初始化 | "Sigmoid 初始化" | Var(w) = 2/(fan_in + fan_out)，设计用于通过 sigmoid 和 tanh 激活保持方差 |
| Kaiming/He 初始化 | "ReLU 初始化" | Var(w) = 2/fan_in，考虑了 ReLU 将一半激活值归零的因素 |
| 方差传播 | "信号如何在层间增长或收缩" | 激活方差如何基于权重尺度逐层变化的数学分析 |
| 残差缩放 | "GPT-2 的初始化技巧" | 将残差连接权重缩放 1/sqrt(2N) 以防止方差通过 N 个 transformer 层增长 |
| 死亡网络 | "什么都无法训练" | 一个网络，由于不当的初始化导致所有梯度为零或所有激活值饱和 |
| 激活值爆炸 | "值趋于无穷" | 当权重方差过高时，导致激活幅度在层间指数级增长 |

## 延伸阅读

- Glorot & Bengio, "Understanding the difficulty of training deep feedforward neural networks" (2010) -- 原始 Xavier 初始化论文，包含方差分析
- He et al., "Delving Deep into Rectifiers" (2015) -- 引入了 ReLU 网络的 Kaiming 初始化
- Radford et al., "Language Models are Unsupervised Multitask Learners" (2019) -- GPT-2 论文，包含残差缩放初始化
- Mishkin & Matas, "All You Need is a Good Init" (2016) -- 层序列单位方差初始化，分析公式的经验替代方案
