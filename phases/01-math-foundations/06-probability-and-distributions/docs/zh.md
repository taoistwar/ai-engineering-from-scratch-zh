# 概率与分布

> 概率是 AI 用来表达不确定性的语言。

**类型：** 学习
**语言：** Python
**前置条件：** 第 1 阶段，第 01-04 课
**时间：** 约 75 分钟

## 学习目标

- 从零实现伯努利、分类、泊松、均匀和正态分布的 PMF 和 PDF
- 计算期望值、方差，并使用中心极限定理解释为什么高斯分布占主导地位
- 构建 softmax 和 log-softmax 函数，使用数值稳定性技巧（减去最大 logit）
- 从 logits 计算交叉熵损失，并将其与负对数似然联系起来

## 问题所在

分类器输出 `[0.03, 0.91, 0.06]`。语言模型从 50,000 个候选词中选择下一个词。扩散模型通过从学习的分布中采样来生成图像。这些都是概率的实际应用。

模型做出的每个预测都是一个概率分布。每个损失函数衡量预测分布与真实分布之间的距离。每个训练步骤调整参数，使一个分布更像另一个分布。没有概率，你无法阅读任何一篇机器学习论文，无法调试任何模型，也无法理解为什么你的训练损失是 NaN。

## 核心概念

### 事件、样本空间与概率

样本空间 S 是所有可能结果的集合。事件是样本空间的子集。概率将事件映射到 0 到 1 之间的数字。

```
抛硬币：
  S = {H, T}
  P(H) = 0.5,  P(T) = 0.5

掷一次骰子：
  S = {1, 2, 3, 4, 5, 6}
  P(偶数) = P({2, 4, 6}) = 3/6 = 0.5
```

三个公理定义了所有概率：
1. P(A) >= 0，对于任何事件 A
2. P(S) = 1（总会有事情发生）
3. 当 A 和 B 不能同时发生时，P(A 或 B) = P(A) + P(B)

其他所有内容（贝叶斯定理、期望、分布）都遵循这三条规则。

### 条件概率与独立性

P(A|B) 是在 B 发生的情况下 A 的概率。

```
P(A|B) = P(A 且 B) / P(B)

示例：一副牌
  P(国王 | 人头牌) = P(国王 且 人头牌) / P(人头牌)
                    = (4/52) / (12/52)
                    = 4/12 = 1/3
```

当一个事件的发生不影响另一个事件时，两个事件是独立的：

```
独立：   P(A|B) = P(A)
等价于： P(A 且 B) = P(A) * P(B)
```

抛硬币是独立的。不放回地抽牌则不是。

### 概率质量函数 vs 概率密度函数

离散随机变量有概率质量函数（PMF）。每个结果都有可以直接读取的特定概率。

```
PMF: P(X = k)

公平骰子：
  P(X = 1) = 1/6
  P(X = 2) = 1/6
  ...
  P(X = 6) = 1/6

  所有概率之和 = 1
```

连续随机变量有概率密度函数（PDF）。单个点的密度不是概率。概率来自于在区间上对密度进行积分。

```
PDF: f(x)

P(a <= X <= b) = 从 a 到 b 对 f(x) 的积分

f(x) 可以大于 1（密度，不是概率）
从 -inf 到 +inf 对 f(x) dx 的积分 = 1
```

这个区别在机器学习中很重要。分类输出是 PMF（离散选择）。VAE 潜在空间使用 PDF（连续）。

### 常见分布

**伯努利分布：** 一次试验，两种结果。建模二分类。

```
P(X = 1) = p
P(X = 0) = 1 - p
均值 = p,  方差 = p(1-p)
```

**分类分布：** 一次试验，k 种结果。建模多分类（softmax 输出）。

```
P(X = i) = p_i,  其中 p_i 之和 = 1
示例： P(猫) = 0.7,  P(狗) = 0.2,  P(鸟) = 0.1
```

**均匀分布：** 所有结果等可能。用于随机初始化。

```
离散： P(X = k) = 1/n，k 属于 {1, ..., n}
连续： f(x) = 1/(b-a)，x 属于 [a, b]
```

**正态分布（高斯分布）：** 钟形曲线。由均值（mu）和方差（sigma^2）参数化。

```
f(x) = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - mu)^2 / (2*sigma^2))

标准正态分布： mu = 0, sigma = 1
  68% 的数据在 1 个 sigma 内
  95% 在 2 个 sigma 内
  99.7% 在 3 个 sigma 内
```

**泊松分布：** 固定区间内稀有事件的计数。建模事件速率。

```
P(X = k) = (lambda^k * e^(-lambda)) / k!
均值 = lambda,  方差 = lambda
```

### 期望值与方差

期望值是加权平均结果。

```
离散：   E[X] = x_i * P(X = x_i) 之和
连续： E[X] = x * f(x) dx 的积分
```

方差衡量围绕均值的分散程度。

```
Var(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2
标准差 = sqrt(Var(X))
```

在机器学习中，期望值以损失函数的形式出现（数据分布上的平均损失）。方差告诉你模型的稳定性。梯度中的高方差意味着训练嘈杂。

### 联合分布与边缘分布

联合分布 P(X, Y) 一起描述两个随机变量。

联合 PMF 示例（X = 天气，Y = 雨伞）：

| | Y=0（无雨伞） | Y=1（有雨伞） | 边缘 P(X) |
|---|---|---|---|
| X=0（晴） | 0.40 | 0.10 | P(X=0) = 0.50 |
| X=1（雨） | 0.05 | 0.45 | P(X=1) = 0.50 |
| **边缘 P(Y)** | P(Y=0) = 0.45 | P(Y=1) = 0.55 | 1.00 |

边缘分布将另一个变量求和消去：

```
P(X = x) = 对所有 y 求和 P(X = x, Y = y)
```

上表中的行和列总计就是边缘分布。

### 为什么正态分布无处不在

中心极限定理：许多独立随机变量的和（或平均值）收敛到正态分布，无论原始分布是什么。

```
掷 1 个骰子：  均匀分布（平坦）
2 个骰子的平均值：  三角形（ peaked）
30 个骰子的平均值： 几乎完美的钟形曲线

这对任何起始分布都有效。
```

这就是为什么：
- 测量误差近似正态（许多小的独立来源）
- 神经网络中的权重初始化使用正态分布
- SGD 中的梯度噪声近似正态（许多样本梯度的和）
- 正态分布是在给定均值和方差下的最大熵分布

### 对数概率

原始概率会导致数值问题。将许多小概率相乘很快就会下溢到零。

```
P(句子) = P(word1) * P(word2) * ... * P(word_n)
            = 0.01 * 0.003 * 0.02 * ...
            -> 0.0（约 30 项后下溢）
```

对数概率解决了这个问题。乘法变成加法。

```
log P(句子) = log P(word1) + log P(word2) + ... + log P(word_n)
                = -4.6 + -5.8 + -3.9 + ...
                -> 有限数字（无下溢）
```

规则：
- log(a * b) = log(a) + log(b)
- 对数概率总是 <= 0（因为 0 < P <= 1）
- 更负 = 不太可能
- 交叉熵损失是正确类别的负对数概率

### Softmax 作为概率分布

神经网络输出原始分数（logits）。Softmax 将它们转换为有效的概率分布。

```
softmax(z_i) = exp(z_i) / sum(exp(z_j) for all j)

性质：
  - 所有输出在 (0, 1) 之间
  - 所有输出之和为 1
  - 保留输入的相对排序
  - exp() 放大 logits 之间的差异
```

softmax 技巧：在指数化之前减去最大 logit 以防止溢出。

```
z = [100, 101, 102]
exp(102) = 溢出

z_shifted = z - max(z) = [-2, -1, 0]
exp(0) = 1  （安全）

相同结果，无溢出。
```

Log-softmax 结合 softmax 和 log 以获得数值稳定性。PyTorch 在内部使用这个来计算交叉熵损失。

### 采样

采样意味着从分布中抽取随机值。在机器学习中：
- Dropout 随机采样哪些神经元要置零
- 数据增强采样随机变换
- 语言模型从预测分布中采样下一个 token
- 扩散模型采样噪声并逐步去噪

从任意分布采样需要技术，如逆变换采样、拒绝采样或重参数化技巧（用于 VAE）。

```figure
gaussian-pdf
```

## 构建它

### 第 1 步：概率基础

```python
import math
import random

def factorial(n):
    """计算阶乘"""
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def combinations(n, k):
    """计算组合数"""
    return factorial(n) // (factorial(k) * factorial(n - k))

def conditional_probability(p_a_and_b, p_b):
    """计算条件概率"""
    return p_a_and_b / p_b

p_king_given_face = conditional_probability(4/52, 12/52)
print(f"P(国王 | 人头牌) = {p_king_given_face:.4f}")
```

### 第 2 步：从零实现 PMF 和 PDF

```python
def bernoulli_pmf(k, p):
    """伯努利分布的 PMF"""
    return p if k == 1 else (1 - p)

def categorical_pmf(k, probs):
    """分类分布的 PMF"""
    return probs[k]

def poisson_pmf(k, lam):
    """泊松分布的 PMF"""
    return (lam ** k) * math.exp(-lam) / factorial(k)

def uniform_pdf(x, a, b):
    """均匀分布的 PDF"""
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0

def normal_pdf(x, mu, sigma):
    """正态分布的 PDF"""
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)
```

### 第 3 步：期望值和方差

```python
def expected_value(values, probabilities):
    """计算期望值"""
    return sum(v * p for v, p in zip(values, probabilities))

def variance(values, probabilities):
    """计算方差"""
    mu = expected_value(values, probabilities)
    return sum(p * (v - mu) ** 2 for v, p in zip(values, probabilities))

die_values = [1, 2, 3, 4, 5, 6]
die_probs = [1/6] * 6
mu = expected_value(die_values, die_probs)
var = variance(die_values, die_probs)
print(f"骰子： E[X] = {mu:.4f}, Var(X) = {var:.4f}, SD = {var**0.5:.4f}")
```

### 第 4 步：从分布中采样

```python
def sample_bernoulli(p, n=1):
    """从伯努利分布采样"""
    return [1 if random.random() < p else 0 for _ in range(n)]

def sample_categorical(probs, n=1):
    """从分类分布采样"""
    cumulative = []
    total = 0
    for p in probs:
        total += p
        cumulative.append(total)
    samples = []
    for _ in range(n):
        r = random.random()
        for i, c in enumerate(cumulative):
            if r <= c:
                samples.append(i)
                break
    return samples

def sample_normal_box_muller(mu, sigma, n=1):
    """使用 Box-Muller 变换从正态分布采样"""
    samples = []
    for _ in range(n):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(mu + sigma * z)
    return samples
```

### 第 5 步：Softmax 和对数概率

```python
def softmax(logits):
    """计算 softmax"""
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]

def log_softmax(logits):
    """计算 log-softmax"""
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]

def cross_entropy_loss(logits, target_index):
    """计算交叉熵损失"""
    log_probs = log_softmax(logits)
    return -log_probs[target_index]
```

### 第 6 步：中心极限定理演示

```python
def demonstrate_clt(dist_fn, n_samples, n_averages):
    """演示中心极限定理"""
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))
    return averages
```

### 第 7 步：可视化

```python
import matplotlib.pyplot as plt

xs = [mu + sigma * (i - 500) / 100 for i in range(1001)]
ys = [normal_pdf(x, mu, sigma) for x, mu, sigma in ...]
plt.plot(xs, ys)
```

完整实现及所有可视化在 `code/probability.py` 中。

## 使用它

使用 NumPy 和 SciPy，上面的所有内容都是一行代码：

```python
import numpy as np
from scipy import stats

normal = stats.norm(loc=0, scale=1)
samples = normal.rvs(size=10000)
print(f"均值： {np.mean(samples):.4f}, 标准差： {np.std(samples):.4f}")
print(f"P(X < 1.96) = {normal.cdf(1.96):.4f}")

logits = np.array([2.0, 1.0, 0.1])
from scipy.special import softmax, log_softmax
probs = softmax(logits)
log_probs = log_softmax(logits)
print(f"Softmax: {probs}")
print(f"Log-softmax: {log_probs}")
```

你从零构建了这些。现在你知道库调用在做什么了。

## 练习

1. 为指数分布实现逆变换采样。通过采样 10,000 个值并将直方图与真实 PDF 进行比较来验证。

2. 为两个灌铅骰子构建联合分布表。计算边缘分布并检查骰子是否独立。

3. 计算一个 5 类分类器的交叉熵损失，该分类器输出 logits `[2.0, 0.5, -1.0, 3.0, 0.1]`，正确类别是索引 3。然后用 PyTorch 的 `nn.CrossEntropyLoss` 验证你的答案。

4. 编写一个函数，接收对数概率列表并返回最可能的序列、总对数概率和等效的原始概率。用 50 个词的句子测试它，每个词的概率为 0.01。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 样本空间 | "所有可能性" | 实验所有可能结果的集合 S |
| PMF | "概率函数" | 给出每个离散结果确切概率的函数，总和为 1 |
| PDF | "概率曲线" | 连续变量的密度函数。在区间上积分得到概率 |
| 条件概率 | "给定某事的概率" | P(A\|B) = P(A 且 B) / P(B)。贝叶斯思维和贝叶斯定理的基础 |
| 独立性 | "互不影响" | P(A 且 B) = P(A) * P(B)。知道一个事件不能告诉你关于另一个事件的任何信息 |
| 期望值 | "平均值" | 所有结果的概率加权和。损失函数是一个期望值 |
| 方差 | "分散程度" | 与均值的期望平方偏差。高方差 = 嘈杂、不稳定的估计 |
| 正态分布 | "钟形曲线" | f(x) = (1/sqrt(2*pi*sigma^2)) * exp(-(x-mu)^2/(2*sigma^2))。由于 CLT 而无处不在 |
| 中心极限定理 | "平均值变成正态" | 许多独立样本的均值收敛到正态分布，无论来源是什么 |
| 联合分布 | "两个变量一起" | P(X, Y) 描述 X 和 Y 结果每种组合的概率 |
| 边缘分布 | "将另一个变量求和消去" | P(X) = sum_y P(X, Y)。从联合分布中恢复一个变量的分布 |
| 对数概率 | "概率的对数" | log P(x)。将乘积转换为和，防止长序列中的数值下溢 |
| Softmax | "将分数转换为概率" | softmax(z_i) = exp(z_i) / sum(exp(z_j))。将实值 logits 映射到有效的概率分布 |
| 交叉熵 | "损失函数" | -sum(p_true * log(p_predicted))。衡量两个分布的差异。越低越好 |
| Logits | "原始模型输出" | softmax 之前的未归一化分数。以 logistic 函数命名 |
| 采样 | "抽取随机值" | 根据概率分布生成值。模型如何生成输出 |

## 延伸阅读

- [3Blue1Brown: 但什么是中心极限定理？](https://www.youtube.com/watch?v=zeJD6dqJ5lo) - 为什么平均值变成正态的可视化证明
- [Stanford CS229 概率复习](https://cs229.stanford.edu/section/cs229-prob.pdf) - 涵盖本文所有内容及以上的简明参考
- [Log-Sum-Exp 技巧](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/) - 为什么数值稳定性很重要以及如何实现它
