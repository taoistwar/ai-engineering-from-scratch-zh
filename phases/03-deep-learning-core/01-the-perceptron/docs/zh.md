# 感知机

> 感知机是神经网络的原子。拆开它，你会发现权重、偏置和一个决策。

**类型：** 构建
**语言：** Python
**前置条件：** 第一阶段（线性代数直觉）
**时间：** 约 60 分钟

## 学习目标

- 从零开始用 Python 实现一个感知机，包括权重更新规则和阶跃激活函数
- 解释为什么单个感知机只能解决线性可分问题，并演示 XOR 的失败案例
- 通过组合 OR、NAND 和 AND 门来构建多层感知机以求解 XOR
- 训练一个带 sigmoid 激活和反向传播的双层网络，自动学习 XOR

## 问题

你已经了解了向量和点积。你知道矩阵将输入转换为输出。但机器如何*学习*使用哪种变换呢？

感知机回答了这个问题。它是最简单的学习机器：接收一些输入，乘以权重，加上偏置，做出二值决策。然后调整。仅此而已。有史以来构建的每一个神经网络都是这一思想的层层堆叠。

理解感知机意味着理解代码中"学习"的真正含义：调整数值直到输出与现实匹配。

## 概念

### 一个神经元，一个决策

一个感知机接收 n 个输入，每个输入乘以一个权重，求和，加上偏置，然后将结果通过激活函数。

```mermaid
graph LR
    x1["x1"] -- "w1" --> sum["Σ(wi*xi) + b"]
    x2["x2"] -- "w2" --> sum
    x3["x3"] -- "w3" --> sum
    bias["偏置"] --> sum
    sum --> step["step(z)"]
    step --> out["输出 (0 或 1)"]
```

阶跃函数很粗暴：如果加权和加偏置 >= 0，输出 1。否则，输出 0。

```
step(z) = 1  若 z >= 0
          0  若 z < 0
```

这是一个线性分类器。权重和偏置定义了一条将输入空间一分为二的直线（或高维空间中的超平面）。

### 决策边界

对于两个输入，感知机在二维空间中画出一条直线：

```
  x2
  ┤
  │  类别 1        /
  │    (0)        /
  │              /
  │             / w1·x1 + w2·x2 + b = 0
  │            /
  │           /     类别 2
  │          /        (1)
  ┼─────────/────────── x1
```

直线一侧的所有点输出 0。另一侧的所有点输出 1。训练就是移动这条线，直到它正确地将类别分开。

### 学习规则

感知机的学习规则很简单：

```
对于每个训练样本 (x, y_true)：
    y_pred = predict(x)
    error = y_true - y_pred

    对于每个权重：
        w_i = w_i + learning_rate * error * x_i
    bias = bias + learning_rate * error
```

如果预测正确，error = 0，没有任何改变。如果预测 0 但应该是 1，权重增加。如果预测 1 但应该是 0，权重减小。学习率控制每次调整的幅度。

### XOR 问题

这里是它失败的地方。看看这些逻辑门：

```
AND 门：            OR 门：             XOR 门：
x1  x2  out       x1  x2  out       x1  x2  out
0   0   0         0   0   0         0   0   0
0   1   0         0   1   1         0   1   1
1   0   0         1   0   1         1   0   1
1   1   1         1   1   1         1   1   0
```

AND 和 OR 是线性可分的：你可以画一条直线将 0 和 1 分开。XOR 不是。没有一条直线能同时将 [0,1] 和 [1,0] 与 [0,0] 和 [1,1] 分开。

```
AND（可分）：              XOR（不可分）：

  x2                        x2
  1 ┤  0     1              1 ┤  1     0
    │     /                   │
  0 ┤  0 / 0                0 ┤  0     1
    ┼──/──────── x1           ┼────────── x1
      直线可行！              没有直线可行！
```

这是一个根本性的限制。单个感知机只能解决线性可分的问题。Minsky 和 Papert 在 1969 年证明了这一点，这几乎让神经网络研究停滞了十年。

解决办法：将感知机堆叠成层。多层感知机可以通过组合两个线性决策来解决 XOR。

```figure
perceptron-boundary
```

## 构建它

### 步骤 1：感知机类

```python
class Perceptron:
    def __init__(self, n_inputs, learning_rate=0.1):
        self.weights = [0.0] * n_inputs
        self.bias = 0.0
        self.lr = learning_rate

    def predict(self, inputs):
        total = sum(w * x for w, x in zip(self.weights, inputs))
        total += self.bias
        return 1 if total >= 0 else 0

    def train(self, training_data, epochs=100):
        for epoch in range(epochs):
            errors = 0
            for inputs, target in training_data:
                prediction = self.predict(inputs)
                error = target - prediction
                if error != 0:
                    errors += 1
                    for i in range(len(self.weights)):
                        self.weights[i] += self.lr * error * inputs[i]
                    self.bias += self.lr * error
            if errors == 0:
                print(f"在第 {epoch + 1} 轮收敛")
                return
        print(f"经过 {epochs} 轮后未收敛")
```

### 步骤 2：在逻辑门上训练

```python
and_data = [
    ([0, 0], 0),
    ([0, 1], 0),
    ([1, 0], 0),
    ([1, 1], 1),
]

or_data = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 1),
]

not_data = [
    ([0], 1),
    ([1], 0),
]

print("=== AND 门 ===")
p_and = Perceptron(2)
p_and.train(and_data)
for inputs, _ in and_data:
    print(f"  {inputs} -> {p_and.predict(inputs)}")

print("\n=== OR 门 ===")
p_or = Perceptron(2)
p_or.train(or_data)
for inputs, _ in or_data:
    print(f"  {inputs} -> {p_or.predict(inputs)}")

print("\n=== NOT 门 ===")
p_not = Perceptron(1)
p_not.train(not_data)
for inputs, _ in not_data:
    print(f"  {inputs} -> {p_not.predict(inputs)}")
```

### 步骤 3：观察 XOR 失败

```python
xor_data = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0),
]

print("\n=== XOR 门（单个感知机） ===")
p_xor = Perceptron(2)
p_xor.train(xor_data, epochs=1000)
for inputs, expected in xor_data:
    result = p_xor.predict(inputs)
    status = "OK" if result == expected else "错误"
    print(f"  {inputs} -> {result} (期望 {expected}) {status}")
```

它永远不会收敛。这是单个感知机无法学习 XOR 的有力证明。

### 步骤 4：用两层网络解决 XOR

诀窍：XOR = (x1 OR x2) AND NOT (x1 AND x2)。组合三个感知机：

```mermaid
graph LR
    x1["x1"] --> OR["OR 神经元"]
    x1 --> NAND["NAND 神经元"]
    x2["x2"] --> OR
    x2 --> NAND
    OR --> AND["AND 神经元"]
    NAND --> AND
    AND --> out["输出"]
```

```python
def xor_network(x1, x2):
    or_neuron = Perceptron(2)
    or_neuron.weights = [1.0, 1.0]
    or_neuron.bias = -0.5

    nand_neuron = Perceptron(2)
    nand_neuron.weights = [-1.0, -1.0]
    nand_neuron.bias = 1.5

    and_neuron = Perceptron(2)
    and_neuron.weights = [1.0, 1.0]
    and_neuron.bias = -1.5

    hidden1 = or_neuron.predict([x1, x2])
    hidden2 = nand_neuron.predict([x1, x2])
    output = and_neuron.predict([hidden1, hidden2])
    return output


print("\n=== XOR 门（多层网络） ===")
for inputs, expected in xor_data:
    result = xor_network(inputs[0], inputs[1])
    print(f"  {inputs} -> {result} (期望 {expected})")
```

四种情况全部正确。将感知机堆叠成层，能够产生单个感知机无法创建的决策边界。

### 步骤 5：训练一个双层网络

步骤 4 手动设定了权重。这对 XOR 有效，但对于你不知道正确权重的实际问题则不适用。解决办法：将阶跃函数替换为 sigmoid，并通过反向传播自动学习权重。

```python
class TwoLayerNetwork:
    def __init__(self, learning_rate=0.5):
        import random
        random.seed(0)
        self.w_hidden = [[random.uniform(-1, 1), random.uniform(-1, 1)] for _ in range(2)]
        self.b_hidden = [random.uniform(-1, 1), random.uniform(-1, 1)]
        self.w_output = [random.uniform(-1, 1), random.uniform(-1, 1)]
        self.b_output = random.uniform(-1, 1)
        self.lr = learning_rate

    def sigmoid(self, x):
        import math
        x = max(-500, min(500, x))
        return 1.0 / (1.0 + math.exp(-x))

    def forward(self, inputs):
        self.inputs = inputs
        self.hidden_outputs = []
        for i in range(2):
            z = sum(w * x for w, x in zip(self.w_hidden[i], inputs)) + self.b_hidden[i]
            self.hidden_outputs.append(self.sigmoid(z))
        z_out = sum(w * h for w, h in zip(self.w_output, self.hidden_outputs)) + self.b_output
        self.output = self.sigmoid(z_out)
        return self.output

    def train(self, training_data, epochs=10000):
        for epoch in range(epochs):
            total_error = 0
            for inputs, target in training_data:
                output = self.forward(inputs)
                error = target - output
                total_error += error ** 2

                d_output = error * output * (1 - output)

                saved_w_output = self.w_output[:]
                hidden_deltas = []
                for i in range(2):
                    h = self.hidden_outputs[i]
                    hd = d_output * saved_w_output[i] * h * (1 - h)
                    hidden_deltas.append(hd)

                for i in range(2):
                    self.w_output[i] += self.lr * d_output * self.hidden_outputs[i]
                self.b_output += self.lr * d_output

                for i in range(2):
                    for j in range(len(inputs)):
                        self.w_hidden[i][j] += self.lr * hidden_deltas[i] * inputs[j]
                    self.b_hidden[i] += self.lr * hidden_deltas[i]
```

```python
net = TwoLayerNetwork(learning_rate=2.0)
net.train(xor_data, epochs=10000)
for inputs, expected in xor_data:
    result = net.forward(inputs)
    predicted = 1 if result >= 0.5 else 0
    print(f"  {inputs} -> {result:.4f} (四舍五入: {predicted}, 期望 {expected})")
```

与步骤 4 有两个关键区别。第一，sigmoid 替换了阶跃函数——它是平滑的，因此梯度存在。第二，`train` 方法将误差从输出层反向传播到隐藏层，按每个权重对误差的贡献比例调整权重。这就是 20 行代码中的反向传播。

这是通往第 03 课的桥梁。`d_output` 和 `hidden_deltas` 背后的数学是应用于网络图的链式法则。我们将在那里正式推导。

## 使用它

你刚刚从零开始构建的一切，都可以用一个导入来完成：

```python
from sklearn.linear_model import Perceptron as SkPerceptron
import numpy as np

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0, 0, 0, 1])

clf = SkPerceptron(max_iter=100, tol=1e-3)
clf.fit(X, y)
print([clf.predict([x])[0] for x in X])
```

五行代码。你写的 30 行 `Perceptron` 类做的事情完全一样。sklearn 版本增加了收敛检查、多种损失函数和稀疏输入支持——但其核心循环是相同的：加权求和、阶跃函数、根据误差更新权重。

真正的差距体现在规模上。在生产网络中改变的是：

- 阶跃函数变成了 sigmoid、ReLU 或其他平滑激活函数
- 权重通过反向传播自动学习（第 03 课）
- 层数变得更深：3 层、10 层、100+ 层
- 同样的原理依然成立：每一层从上一层的输出中创建新特征

单个感知机只能画直线。把它们堆叠起来，你就可以画出任何形状。

## 交付物

本课产出：
- `outputs/skill-perceptron.md` - 一个技能，涵盖何时需要单层架构 vs 多层架构

## 练习

1. 在 NAND 门（通用门——任何逻辑电路都可以用 NAND 构建）上训练感知机。验证其权重和偏置形成了一个有效的决策边界。
2. 修改 Perceptron 类，在每个 epoch 跟踪决策边界（w1*x1 + w2*x2 + b = 0）。打印在 AND 门上训练时这条线如何移动。
3. 构建一个 3 输入感知机，只有当 3 个输入中至少有 2 个为 1 时才输出 1（多数表决函数）。这是线性可分的吗？为什么？

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|----------|
| 感知机 | "一个假神经元" | 一个线性分类器：输入与权重的点积，加上偏置，通过阶跃函数 |
| 权重 | "输入有多重要" | 一个乘数，缩放每个输入对决策的贡献 |
| 偏置 | "阈值" | 一个常数，移动决策边界，使感知机即使在零输入时也能激活 |
| 激活函数 | "压缩数值的东西" | 在加权和之后应用的函数——感知机用阶跃函数，现代网络用 sigmoid/ReLU |
| 线性可分 | "可以画一条线分开" | 一个数据集，其中一个超平面可以完美地将类别分开 |
| XOR 问题 | "感知机做不到的事情" | 证明单层网络无法学习非线性可分函数的例子 |
| 决策边界 | "分类器切换的地方" | 将输入空间分成两个类别的超平面 w*x + b = 0 |
| 多层感知机 | "真正的神经网络" | 堆叠成层的感知机，每层的输出馈送到下一层的输入 |

## 延伸阅读

- Frank Rosenblatt, "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain" (1958) -- 开创一切的原始论文
- Minsky & Papert, "Perceptrons" (1969) -- 证明 XOR 无法被单层网络解决、使感知机研究停滞十年的那本书
- Michael Nielsen, "Neural Networks and Deep Learning", Chapter 1 (http://neuralnetworksanddeeplearning.com/) -- 免费在线阅读，关于感知机如何组合成网络的最佳可视化解释
