# 信息论

> 信息论衡量意外程度。损失函数建立在它之上。

**类型：** 学习
**语言：** Python
**前置课程：** 阶段 1, 课程 06（概率）
**时长：** 约 60 分钟

## 学习目标

- 从零计算熵、交叉熵和 KL 散度，并解释它们之间的关系
- 推导为什么最小化交叉熵损失等价于最大化对数似然
- 计算特征与目标之间的互信息以排列特征重要性
- 解释困惑度作为语言模型有效选择词汇量的含义

## 问题

你在训练的每个分类模型中调用 `CrossEntropyLoss()`。你在每篇语言模型论文中看到"困惑度"。你在 VAE、蒸馏和 RLHF 中读到 KL 散度。这些不是互不相关的概念。它们是同一个想法戴着不同的帽子。

信息论为你提供了推理不确定性、压缩和预测的语言。Claude Shannon 在 1948 年发明它来解决通信问题。事实证明，训练神经网络也是一个通信问题：模型试图通过学习到的权重的噪声信道传输正确的标签。

本课程从零构建每个公式，让你看到它们从何而来以及为何有效。

## 概念

### 信息量（意外程度）

当不太可能的事情发生时，它携带更多信息。一枚硬币正面朝上？不意外。中彩票？非常意外。

概率为 p 的事件的信息量为：

```
I(x) = -log(p(x))
```

使用以 2 为底的对数得到比特。使用自然对数得到纳特。同一个想法，不同的单位。

```
Event              Probability    Surprise (bits)
Fair coin heads    0.5            1.0
Rolling a 6        0.167          2.58
1-in-1000 event    0.001          9.97
Certain event      1.0            0.0
```

确定事件携带零信息。你已经知道它们会发生。

### 熵（平均意外程度）

熵是分布中所有可能结果的期望意外程度。

```
H(P) = -sum( p(x) * log(p(x)) )  for all x
```

公平硬币是二元变量的最大熵：1 比特。有偏硬币（99% 正面）有低熵：0.08 比特。你已经知道会发生什么，所以每次抛掷几乎不告诉你任何信息。

```
Fair coin:    H = -(0.5 * log2(0.5) + 0.5 * log2(0.5)) = 1.0 bit
Biased coin:  H = -(0.99 * log2(0.99) + 0.01 * log2(0.01)) = 0.08 bits
```

熵衡量分布中不可约的不确定性。你无法压缩到它以下。

### 交叉熵（你每天使用的损失函数）

交叉熵衡量当你使用分布 Q 来编码实际来自分布 P 的事件时的平均意外程度。

```
H(P, Q) = -sum( p(x) * log(q(x)) )  for all x
```

P 是真实分布（标签）。Q 是你模型的预测。如果 Q 与 P 完美匹配，交叉熵等于熵。任何不匹配都会使它更大。

在分类中，P 是一个 one-hot 向量（真实类别的概率为 1，其他为 0）。这将交叉熵简化为：

```
H(P, Q) = -log(q(true_class))
```

这就是分类的整个交叉熵损失公式。最大化正确类别的预测概率。

### KL 散度（分布之间的距离）

KL 散度衡量你使用 Q 而非 P 时获得的额外意外程度。

```
D_KL(P || Q) = sum( p(x) * log(p(x) / q(x)) )  for all x
             = H(P, Q) - H(P)
```

交叉熵等于熵加上 KL 散度。由于真实分布的熵在训练期间是常数，最小化交叉熵就等于最小化 KL 散度。你在将模型的分布推向真实分布。

KL 散度不是对称的：D_KL(P || Q) != D_KL(Q || P)。它不是一个真正的距离度量。

### 互信息

互信息衡量知道一个变量能告诉你关于另一个变量的多少信息。

```
I(X; Y) = H(X) - H(X|Y)
        = H(X) + H(Y) - H(X, Y)
```

如果 X 和 Y 独立，互信息为零。知道一个不能告诉你关于另一个的任何信息。如果它们完全相关，互信息等于任一变量的熵。

在特征选择中，特征与目标之间的高互信息意味着该特征是有用的。低互信息意味着它是噪声。

### 条件熵

H(Y|X) 衡量在观察到 X 之后关于 Y 还剩多少不确定性。

```
H(Y|X) = H(X,Y) - H(X)
```

两个极端：
- 如果 X 完全确定 Y，那么 H(Y|X) = 0。知道 X 消除了关于 Y 的所有不确定性。示例：X = 摄氏温度，Y = 华氏温度。
- 如果 X 不能告诉你关于 Y 的任何信息，那么 H(Y|X) = H(Y)。知道 X 完全不减少你的不确定性。示例：X = 抛硬币，Y = 明天的天气。

条件熵始终非负且不超过 H(Y)：

```
0 <= H(Y|X) <= H(Y)
```

在机器学习中，条件熵出现在决策树中。在每次分裂时，算法选择特征 X 来最小化 H(Y|X) —— 即消除关于标签 Y 最多不确定性的特征。

### 联合熵

H(X,Y) 是 X 和 Y 联合分布的熵。

```
H(X,Y) = -sum sum p(x,y) * log(p(x,y))   for all x, y
```

关键性质：

```
H(X,Y) <= H(X) + H(Y)
```

当 X 和 Y 独立时取等号。如果它们共享信息，联合熵小于各个熵的和。"缺失"的熵恰好是互信息。

```mermaid
graph TD
    subgraph "Information Venn Diagram"
        direction LR
        HX["H(X)"]
        HY["H(Y)"]
        MI["I(X;Y)<br/>Mutual<br/>Information"]
        HXgY["H(X|Y)<br/>= H(X) - I(X;Y)"]
        HYgX["H(Y|X)<br/>= H(Y) - I(X;Y)"]
        HXY["H(X,Y) = H(X) + H(Y) - I(X;Y)"]
    end

    HXgY --- MI
    MI --- HYgX
    HX -.- HXgY
    HX -.- MI
    HY -.- MI
    HY -.- HYgX
    HXY -.- HXgY
    HXY -.- MI
    HXY -.- HYgX
```

这些关系：
- H(X,Y) = H(X) + H(Y|X) = H(Y) + H(X|Y)
- I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)
- H(X,Y) = H(X) + H(Y) - I(X;Y)

### 互信息（深入）

互信息 I(X;Y) 量化知道一个变量能减少关于另一个变量的多少不确定性。

```
I(X;Y) = H(X) - H(X|Y)
       = H(Y) - H(Y|X)
       = H(X) + H(Y) - H(X,Y)
       = sum sum p(x,y) * log(p(x,y) / (p(x) * p(y)))
```

性质：
- I(X;Y) >= 0 始终成立。通过观察某事你永远不会丢失信息。
- I(X;Y) = 0 当且仅当 X 和 Y 独立。
- I(X;Y) = I(Y;X)。它是对称的，不像 KL 散度。
- I(X;X) = H(X)。一个变量与自身共享其所有信息。

**互信息用于特征选择。** 在机器学习中，你需要关于目标有信息量的特征。互信息为你提供了一种原则性的方法来排列特征：

1. 对每个特征 X_i，计算 I(X_i; Y)，其中 Y 是目标变量。
2. 按互信息得分排列特征。
3. 保留前 k 个特征。

这对特征和目标之间的任何关系都有效 —— 线性、非线性、单调或非单调。相关性只能捕捉线性关系。互信息能捕捉一切。

| 方法 | 检测 | 计算成本 | 处理分类变量？ |
|------|------|---------|--------------|
| Pearson 相关 | 线性关系 | O(n) | 否 |
| Spearman 相关 | 单调关系 | O(n log n) | 否 |
| 互信息 | 任何统计依赖 | O(n log n) 使用分箱 | 是 |

### 标签平滑与交叉熵

标准分类使用硬目标：[0, 0, 1, 0]。真实类别的概率为 1，其他为 0。标签平滑用软目标替换它们：

```
soft_target = (1 - epsilon) * hard_target + epsilon / num_classes
```

当 epsilon = 0.1 且 4 个类别时：
- 硬目标：[0, 0, 1, 0]
- 软目标：[0.025, 0.025, 0.925, 0.025]

从信息论的角度来看，标签平滑增加了目标分布的熵。硬 one-hot 目标的熵为 0 —— 没有不确定性。软目标有正熵。

为什么这有帮助：
- 防止模型将 logit 推向极端值（在交叉熵下完美匹配 one-hot 目标需要无穷大的 logit）
- 充当正则化：模型不能 100% 确信
- 改善校准：预测概率更好地反映真实不确定性
- 缩小训练和推理行为之间的差距

带标签平滑的交叉熵损失变为：

```
L = (1 - epsilon) * CE(hard_target, prediction) + epsilon * H_uniform(prediction)
```

第二项惩罚远离均匀的预测 —— 对置信度的直接正则化。

### 为什么交叉熵是分类的首选损失

三个视角，同一个结论。

**信息论视角。** 交叉熵衡量你使用模型的分布而非真实分布时浪费了多少比特。最小化它使你的模型成为现实的最有效编码器。

**最大似然视角。** 对于 N 个训练样本，真实类别为 y_i：

```
Likelihood     = product( q(y_i) )
Log-likelihood = sum( log(q(y_i)) )
Negative log-likelihood = -sum( log(q(y_i)) )
```

最后一行就是交叉熵损失。最小化交叉熵 = 最大化你的模型下训练数据的似然。

**梯度视角。** 交叉熵相对于 logit 的梯度就是（预测 - 真实）。干净、稳定且计算快速。这就是为什么它与 softmax 完美搭配。

### 比特 vs 纳特

唯一的区别是对数的底。

```
log base 2   -> bits      (information theory tradition)
log base e   -> nats      (machine learning convention)
log base 10  -> hartleys  (rarely used)
```

1 纳特 = 1/ln(2) 比特 = 1.4427 比特。PyTorch 和 TensorFlow 默认使用自然对数（纳特）。

### 困惑度

困惑度是交叉熵的指数。它告诉你模型不确定地在多少个等概率选项之间做选择。

```
Perplexity = 2^H(P,Q)   (if using bits)
Perplexity = e^H(P,Q)   (if using nats)
```

困惑度为 50 的语言模型平均来说，就像它必须从 50 个可能的下一个 token 中均匀选择一样困惑。越低越好。

GPT-2 在常见基准上达到了约 30 的困惑度。现代模型在表示良好的领域中是个位数。

```figure
entropy-kl
```

## 构建它

### 步骤 1：信息量和熵

```python
import math

def information_content(p, base=2):
    if p <= 0 or p > 1:
        return float('inf') if p <= 0 else 0.0
    return -math.log(p) / math.log(base)

def entropy(probs, base=2):
    return sum(
        p * information_content(p, base)
        for p in probs if p > 0
    )

fair_coin = [0.5, 0.5]
biased_coin = [0.99, 0.01]
fair_die = [1/6] * 6

print(f"Fair coin entropy:   {entropy(fair_coin):.4f} bits")
print(f"Biased coin entropy: {entropy(biased_coin):.4f} bits")
print(f"Fair die entropy:    {entropy(fair_die):.4f} bits")
```

### 步骤 2：交叉熵和 KL 散度

```python
def cross_entropy(p, q, base=2):
    total = 0.0
    for pi, qi in zip(p, q):
        if pi > 0:
            if qi <= 0:
                return float('inf')
            total += pi * (-math.log(qi) / math.log(base))
    return total

def kl_divergence(p, q, base=2):
    return cross_entropy(p, q, base) - entropy(p, base)

true_dist = [0.7, 0.2, 0.1]
good_model = [0.6, 0.25, 0.15]
bad_model = [0.1, 0.1, 0.8]

print(f"Entropy of true dist:     {entropy(true_dist):.4f} bits")
print(f"CE (good model):          {cross_entropy(true_dist, good_model):.4f} bits")
print(f"CE (bad model):           {cross_entropy(true_dist, bad_model):.4f} bits")
print(f"KL divergence (good):     {kl_divergence(true_dist, good_model):.4f} bits")
print(f"KL divergence (bad):      {kl_divergence(true_dist, bad_model):.4f} bits")
```

### 步骤 3：交叉熵作为分类损失

```python
def softmax(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def cross_entropy_loss(true_class, logits):
    probs = softmax(logits)
    return -math.log(probs[true_class])

logits = [2.0, 1.0, 0.1]
true_class = 0

probs = softmax(logits)
loss = cross_entropy_loss(true_class, logits)

print(f"Logits:      {logits}")
print(f"Softmax:     {[f'{p:.4f}' for p in probs]}")
print(f"True class:  {true_class}")
print(f"Loss:        {loss:.4f} nats")
print(f"Perplexity:  {math.exp(loss):.2f}")
```

### 步骤 4：交叉熵等于负对数似然

```python
import random

random.seed(42)

n_samples = 1000
n_classes = 3
true_labels = [random.randint(0, n_classes - 1) for _ in range(n_samples)]
model_logits = [[random.gauss(0, 1) for _ in range(n_classes)] for _ in range(n_samples)]

ce_loss = sum(
    cross_entropy_loss(label, logits)
    for label, logits in zip(true_labels, model_logits)
) / n_samples

nll = -sum(
    math.log(softmax(logits)[label])
    for label, logits in zip(true_labels, model_logits)
) / n_samples

print(f"Cross-entropy loss:      {ce_loss:.6f}")
print(f"Negative log-likelihood: {nll:.6f}")
print(f"Difference:              {abs(ce_loss - nll):.2e}")
```

### 步骤 5：互信息

```python
def mutual_information(joint_probs, base=2):
    rows = len(joint_probs)
    cols = len(joint_probs[0])

    margin_x = [sum(joint_probs[i][j] for j in range(cols)) for i in range(rows)]
    margin_y = [sum(joint_probs[i][j] for i in range(rows)) for j in range(cols)]

    mi = 0.0
    for i in range(rows):
        for j in range(cols):
            pxy = joint_probs[i][j]
            if pxy > 0:
                mi += pxy * math.log(pxy / (margin_x[i] * margin_y[j])) / math.log(base)
    return mi

independent = [[0.25, 0.25], [0.25, 0.25]]
dependent = [[0.45, 0.05], [0.05, 0.45]]

print(f"MI (independent): {mutual_information(independent):.4f} bits")
print(f"MI (dependent):   {mutual_information(dependent):.4f} bits")
```

## 使用它

使用 NumPy 的相同概念，这是你在实践中使用的方式：

```python
import numpy as np

def np_entropy(p):
    p = np.asarray(p, dtype=float)
    mask = p > 0
    result = np.zeros_like(p)
    result[mask] = p[mask] * np.log(p[mask])
    return -result.sum()

def np_cross_entropy(p, q):
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    mask = p > 0
    return -(p[mask] * np.log(q[mask])).sum()

def np_kl_divergence(p, q):
    return np_cross_entropy(p, q) - np_entropy(p)

true = np.array([0.7, 0.2, 0.1])
pred = np.array([0.6, 0.25, 0.15])
print(f"Entropy:    {np_entropy(true):.4f} nats")
print(f"Cross-ent:  {np_cross_entropy(true, pred):.4f} nats")
print(f"KL div:     {np_kl_divergence(true, pred):.4f} nats")
```

你从零构建了 `torch.nn.CrossEntropyLoss()` 内部所做的事情。现在你知道为什么损失在训练期间下降了：你模型预测的分布正在接近真实分布，以浪费信息的纳特来衡量。

## 练习

1. 假设均匀分布（26 个字母），计算英文字母表的熵。然后使用实际字母频率估计它。哪个更高，为什么？

2. 一个模型对真实类别为 1 的样本输出 logit [5.0, 2.0, 0.5]。手动计算交叉熵损失，然后用你的 `cross_entropy_loss` 函数验证。什么 logit 会给出零损失？

3. 证明 KL 散度不是对称的。选取两个分布 P 和 Q，计算 D_KL(P || Q) 和 D_KL(Q || P)。解释为什么它们不同。

4. 构建一个函数来计算 token 预测序列的困惑度。给定 (true_token_index, predicted_logits) 对的列表，返回序列的困惑度。

## 关键术语

| 术语 | 人们怎么说 | 它实际意味着什么 |
|------|-----------|----------------|
| 信息量 | "意外程度" | 编码一个事件所需的比特（或纳特）数：-log(p) |
| 熵 | "随机性" | 分布中所有结果的平均意外程度。衡量不可约的不确定性。 |
| 交叉熵 | "那个损失函数" | 使用模型分布 Q 编码来自真实分布 P 的事件时的平均意外程度。 |
| KL 散度 | "分布之间的距离" | 使用 Q 而非 P 时浪费的额外比特。等于交叉熵减去熵。不对称。 |
| 互信息 | "X 和 Y 有多相关" | 通过知道 Y 减少的关于 X 的不确定性。零表示独立。 |
| Softmax | "将 logit 转为概率" | 指数化并归一化。将任何实值向量映射为有效的概率分布。 |
| 困惑度 | "模型有多困惑" | 交叉熵的指数。模型在每一步中做选择的有效词汇量。 |
| 比特 | "Shannon 的单位" | 用以 2 为底的对数衡量的信息。一比特解决一次公平硬币抛掷。 |
| 纳特 | "机器学习的单位" | 以自然对数衡量的信息。PyTorch 和 TensorFlow 默认使用。 |
| 负对数似然 | "NLL 损失" | 对 one-hot 标签与交叉熵损失相同。最小化它最大化正确预测的概率。 |

## 延伸阅读

- [Shannon 1948: A Mathematical Theory of Communication](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) - 原始论文，仍然易读
- [Visual Information Theory (Chris Olah)](https://colah.github.io/posts/2015-09-Visual-Information/) - 熵和 KL 散度的最佳可视化解释
- [PyTorch CrossEntropyLoss docs](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) - 框架如何实现你刚刚构建的内容
