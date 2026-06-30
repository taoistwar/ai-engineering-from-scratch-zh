# 数值稳定性

> 浮点数是漏洞百出的抽象。它会在训练中反咬你一口，而你不会提前察觉。

**类型：** 构建
**语言：** Python
**先决条件：** 阶段 1，第 01-04 课
**时间：** ~120 分钟

## 学习目标

- 使用最大值减法技巧实现数值稳定的 softmax 和 log-sum-exp
- 识别浮点计算中的上溢、下溢和灾难性抵消
- 使用中心有限差分将解析梯度与数值梯度进行验证
- 解释为什么 bfloat16 在训练中优于 float16，以及损失缩放如何防止梯度下溢

## 问题

你的模型训练了三个小时，然后损失变成了 NaN。你添加了一条打印语句。第 9,000 步时 logits 正常。第 9,001 步时它们变成了 `inf`。到第 9,002 步，每个梯度都是 `nan`，训练已死。

或者：你的模型训练到完成，但准确率比论文声称的低 2%。你检查了所有东西。架构匹配。超参数匹配。数据匹配。问题是论文使用了 float32 而你使用了 float16，没有正确的缩放。三十二位累积舍入误差悄悄吞噬了你的准确率。

或者：你从零实现了交叉熵损失。它在小 logits 上工作正常。当 logits 超过 100 时，它返回 `inf`。softmax 溢出，因为 `exp(100)` 超过了 float32 的表示范围。每个 ML 框架通过一个两行技巧来处理这个问题。你不知道这个技巧的存在。

数值稳定性不是一个理论问题。它是成功的训练运行与默默失败的训练运行之间的差别。你将来调试的每个严重 ML bug 最终都会归结为浮点数。

## 概念

### IEEE 754：计算机如何存储实数

计算机按照 IEEE 754 标准将实数存储为浮点值。浮点数有三个部分：一个符号位，一个指数，和一个尾数（有效数）。

```
Float32 布局（总共 32 位）：
[1 符号] [8 指数] [23 尾数]

值 = (-1)^符号 * 2^(指数 - 127) * 1.尾数
```

尾数决定精度（有多少有效数字）。指数决定范围（数字可以多大或多小）。

```
格式     位数   指数  尾数  十进制位数  范围（约）
float64   64     11       52        ~15-16          +/- 1.8e308
float32   32     8        23        ~7-8            +/- 3.4e38
float16   16     5        10        ~3-4            +/- 65,504
bfloat16  16     8        7         ~2-3            +/- 3.4e38
```

float32 给你大约 7 位十进制精度。这意味着它能分辨 1.0000001 和 1.0000002，但分辨不了 1.00000001 和 1.00000002。7 位之后，一切都是舍入噪声。

float16 给你大约 3 位精度。它能表示的最大数字是 65,504。这对 ML 来说令人不安地小，因为 logits、梯度和激活值经常会超过这个值。

bfloat16 是 Google 对 float16 范围问题的回答。它具有与 float32 相同的 8 位指数（相同的范围，最高 3.4e38）但只有 7 位尾数（比 float16 精度低）。对于训练神经网络，范围比精度更重要，因此 bfloat16 通常会胜出。

### 为什么 0.1 + 0.2 != 0.3

数字 0.1 无法在二进制浮点中精确表示。在基 2 中，它是一个循环小数：

```
二进制中的 0.1 = 0.0001100110011001100110011...（永远循环）
```

Float32 将其截断为 23 位尾数。存储的值约为 0.100000001490116。类似地，0.2 被存储为约 0.200000002980232。它们的和为 0.300000004470348，而不是 0.3。

```
在 Python 中：
>>> 0.1 + 0.2
0.30000000000000004

>>> 0.1 + 0.2 == 0.3
False
```

这对 ML 很重要，因为：

1. 像 `if loss < threshold` 这样的损失比较可能会给出错误答案
2. 累积许多小值（数千步的梯度更新）会偏离真实和
3. 如果你用 `==` 比较浮点数，校验和和可重现性测试会失败

修复方法：永远不要用 `==` 比较浮点数。使用 `abs(a - b) < epsilon` 或 `math.isclose()`。

### 灾难性抵消

当你减去两个几乎相等的浮点数时，有效数字相互抵消，只剩下被提升为前导数字的舍入噪声。

```
a = 1.0000001    （在 float32 中存储为 1.00000011920929）
b = 1.0000000    （在 float32 中存储为 1.00000000000000）

真实差值：  0.0000001
计算值：    0.00000011920929

相对误差：19.2%
```

单次减法就有 19% 的相对误差。在 ML 中，每当你：

- 计算具有大均值的数据方差：`E[x^2] - E[x]^2` 且 E[x] 很大时
- 减去几乎相等的对数概率
- 使用过小的 epsilon 计算有限差分梯度

修复方法：重新排列公式以避免减去大的几乎相等的数。对于方差，使用 Welford 算法或先中心化数据。对于对数概率，全程在对数空间中工作。

### 上溢和下溢

当结果太大而无法表示时发生上溢。当结果太小（比可表示的最小正数更接近零）时发生下溢。

```
Float32 边界：
  最大值：  3.4028235e+38
  最小正值（正规）：1.175e-38
  最小正值（非正规）：1.401e-45
  上溢：  任何 > 3.4e38 变成 inf
  下溢：任何 < 1.4e-45 变成 0.0
```

`exp()` 函数是 ML 中上溢的主要来源：

```
exp(88.7)  = 3.40e+38   （刚好适合 float32）
exp(89.0)  = inf         （上溢）
exp(-87.3) = 1.18e-38   （刚好高于下溢）
exp(-104)  = 0.0         （下溢到零）
```

`log()` 函数会遇到另一方向：

```
log(0.0)   = -inf
log(-1.0)  = nan
log(1e-45) = -103.3      （正常）
log(1e-46) = -inf        （输入下溢到 0，然后 log(0) = -inf）
```

在 ML 中，`exp()` 出现在 softmax、sigmoid 和概率计算中。`log()` 出现在交叉熵、对数似然和 KL 散度中。组合 `log(exp(x))` 没有正确的技巧就是一个雷区。

### Log-Sum-Exp 技巧

直接计算 `log(sum(exp(x_i)))` 在数值上是危险的。如果任何 `x_i` 很大，`exp(x_i)` 会溢出。如果所有 `x_i` 都非常负，每个 `exp(x_i)` 下溢到零，`log(0)` 是 `-inf`。

技巧：在取指数前减去最大值。

```
log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
```

为什么有效：减去 `max(x)` 后，最大的指数是 `exp(0) = 1`。不会发生上溢。和中至少有一项是 1，所以和至少为 1，`log(1) = 0`。不会下溢到 `-inf`。

证明：

```
log(sum(exp(x_i)))
= log(sum(exp(x_i - c + c)))                    （加减 c）
= log(sum(exp(x_i - c) * exp(c)))               （exp(a+b) = exp(a)*exp(b)）
= log(exp(c) * sum(exp(x_i - c)))               （提出 exp(c)）
= c + log(sum(exp(x_i - c)))                    （log(a*b) = log(a) + log(b)）
```

设 `c = max(x)`，上溢就被消除了。

这个技巧在 ML 中随处可见：
- Softmax 归一化
- 交叉熵损失计算
- 序列模型中的对数概率求和
- 高斯混合模型
- 变分推断

### 为什么 Softmax 需要最大值减法技巧

Softmax 将 logits 转换为概率：

```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

没有技巧，logits [100, 101, 102] 会导致上溢：

```
exp(100) = 2.69e43
exp(101) = 7.31e43
exp(102) = 1.99e44
sum      = 2.99e44

这些溢出 float32（最大 ~3.4e38）？不，2.69e43 < 3.4e38？实际上：
exp(88.7) 已经在 float32 的极限了。
exp(100) 在 float32 中 = inf。
```

使用技巧，减去 max(x) = 102：

```
exp(100 - 102) = exp(-2) = 0.135
exp(101 - 102) = exp(-1) = 0.368
exp(102 - 102) = exp(0)  = 1.000
sum = 1.503

softmax = [0.090, 0.245, 0.665]
```

概率完全相同。计算是安全的。这不是一种优化，而是正确性的必要条件。

### NaN 和 Inf：检测和预防

`nan`（非数字）和 `inf`（无穷大）在计算中像病毒一样传播。梯度更新中的一个 `nan` 使权重变成 `nan`，进而使后续每个输出变成 `nan`。训练在一步之内就死了。

`inf` 如何出现：
- 对大正数的 `exp()`
- 除以零：`1.0 / 0.0`
- 在累积中 `float32` 溢出

`nan` 如何出现：
- `0.0 / 0.0`
- `inf - inf`
- `inf * 0`
- 负数的 `sqrt()`
- 负数的 `log()`
- 涉及已有 `nan` 的任何算术

检测：

```python
import math

math.isnan(x)       # 如果 x 是 nan 则为 True
math.isinf(x)       # 如果 x 是 +inf 或 -inf 则为 True
math.isfinite(x)    # 如果 x 既不是 nan 也不是 inf 则为 True
```

预防策略：

1. 对 `exp()` 的输入进行截断：`exp(clamp(x, -80, 80))`
2. 对分母添加 epsilon：`x / (y + 1e-8)`
3. 在 `log()` 内部添加 epsilon：`log(x + 1e-8)`
4. 使用稳定的实现（log-sum-exp、稳定 softmax）
5. 梯度裁剪以防止权重爆炸
6. 调试期间在每个前向传播后检查 `nan`/`inf`

### 数值梯度检查

解析梯度（来自反向传播）可能存在 bug。数值梯度检查通过有限差分计算梯度来验证它们。

中心差分公式：

```
df/dx ~= (f(x + h) - f(x - h)) / (2h)
```

这是 O(h^2) 精度的，比仅 O(h) 的前向差分 `(f(x+h) - f(x)) / h` 好得多。

选择 h：太大会使近似错误。太小则灾难性抵消会破坏答案。`h = 1e-5` 到 `1e-7` 是典型值。

检查：计算解析梯度和数值梯度之间的相对差。

```
relative_error = |grad_analytical - grad_numerical| / max(|grad_analytical|, |grad_numerical|, 1e-8)
```

经验法则：
- relative_error < 1e-7：完美，梯度是正确的
- relative_error < 1e-5：可接受，可能正确
- relative_error > 1e-3：有问题
- relative_error > 1：梯度完全错误

在实现新层或损失函数时始终检查梯度。PyTorch 为此提供了 `torch.autograd.gradcheck()`。

### 混合精度训练

现代 GPU 有专门的硬件（Tensor Cores），可以以比 float32 快 2-8 倍的速度计算 float16 矩阵乘法。混合精度训练利用了这一点：

```
1. 保持 float32 主副本权重
2. 在 float16 中进行前向传播（快）
3. 在 float32 中计算损失（防止溢出）
4. 在 float16 中进行反向传播（快）
5. 将梯度缩放到 float32
6. 更新 float32 主权重
```

纯 float16 训练的问题：梯度通常非常小（1e-8 或更小）。Float16 将任何低于约 6e-8 的值下溢到零。你的模型停止学习，因为所有梯度更新都是零。

修复方法是损失缩放：

```
1. 将损失乘以一个大的缩放因子（例如 1024）
2. 反向传播计算 (loss * 1024) 的梯度
3. 所有梯度都大了 1024 倍（被推到 float16 下溢阈值以上）
4. 在更新权重前将梯度除以 1024
5. 净效应：相同的更新，但没有下溢
```

动态损失缩放自动调整缩放因子。从一个大值开始（65536）。如果梯度溢出到 `inf`，将其减半。如果 N 步没有溢出，将其加倍。

### bfloat16 vs float16：为什么 bfloat16 在训练中获胜

```
float16:   [1 符号] [5 指数]  [10 尾数]
bfloat16:  [1 符号] [8 指数]  [7 尾数]
```

float16 有更多精度（10 位尾数 vs 7），但范围有限（最大 ~65,504）。bfloat16 精度较低但范围与 float32 相同（最大 ~3.4e38）。

对于训练神经网络：

- 激活值和 logits 在训练峰值期间经常超过 65,504。float16 会溢出；bfloat16 可以处理。
- float16 需要损失缩放，而 bfloat16 通常不需要，因为其范围覆盖了梯度量级谱。
- bfloat16 是 float32 的简单截断：丢弃尾数底部 16 位。转换是简单的，且在指数上无损失。

float16 在推理中更受青睐，因为值有界且精度更重要。bfloat16 在训练中更受青睐，因为范围更重要。这就是为什么 TPU 和现代 NVIDIA GPU（A100、H100）具有原生 bfloat16 支持。

### 梯度裁剪

当梯度通过许多层呈指数增长时会发生梯度爆炸（在 RNN、深度网络和 transformer 中常见）。单个大梯度可以一步破坏所有权重。

两种裁剪类型：

**按值裁剪：** 独立截断每个梯度元素。

```
grad = clamp(grad, -max_val, max_val)
```

简单但可能改变梯度向量的方向。

**按范数裁剪：** 缩放整个梯度向量使其范数不超过阈值。

```
if ||grad|| > max_norm:
    grad = grad * (max_norm / ||grad||)
```

保留梯度方向。这就是 `torch.nn.utils.clip_grad_norm_()` 所做的。它是标准选择。

典型值：transformers 用 `max_norm=1.0`，RL 用 `max_norm=0.5`，简单网络用 `max_norm=5.0`。

梯度裁剪不是一个 hack。它是一种安全机制。没有它，一个异常批次可能产生足以破坏数周训练的大梯度。

### 归一化层作为数值稳定器

批量归一化、层归一化和 RMS 归一化通常被呈现为帮助训练收敛的正则化器。它们也是数值稳定器。

没有归一化，激活值可能在各层之间指数级地增长或缩小：

```
第 1 层：值在 [0, 1] 中
第 5 层：值在 [0, 100] 中
第 10 层：值在 [0, 10,000] 中
第 50 层：值在 [0, inf] 中
```

归一化在每一层重新中心化和重新缩放激活值：

```
LayerNorm(x) = (x - mean(x)) / (std(x) + epsilon) * gamma + beta
```

`epsilon`（通常是 1e-5）防止所有激活值相同时的除零错误。学习参数 `gamma` 和 `beta` 让网络恢复所需的任何缩放。

这使值在整个网络中都保持在数值安全的范围内，防止前向传播中的上溢和反向传播中的梯度爆炸。

### 常见的 ML 数值 Bug

**Bug：损失在几个 epoch 后变成 NaN。**
原因：logits 变得太大，softmax 溢出。或者学习率太高，权重发散。
修复：使用稳定 softmax（最大值减法），降低学习率，添加梯度裁剪。

**Bug：损失卡在 log(num_classes)。**
原因：模型输出接近均匀概率。通常意味着梯度消失或模型根本没有学习。
修复：检查数据标签是否正确，验证损失函数，检查是否有 dead ReLU。

**Bug：验证准确率比预期低 1-3%。**
原因：混合精度没有适当的损失缩放。梯度下溢悄悄将小更新置零。
修复：启用动态损失缩放，或切换到 bfloat16。

**Bug：某些层的梯度范数为 0.0。**
原因：dead ReLU 神经元（所有输入为负），或 float16 下溢。
修复：使用 LeakyReLU 或 GELU，使用梯度缩放，检查权重初始化。

**Bug：模型在一个 GPU 上工作但在另一个上给出不同结果。**
原因：非确定性的浮点累积顺序。GPU 并行归约在不同硬件上以不同顺序求和，而浮点加法不满足结合律。
修复：接受小差异（1e-6），或设置 `torch.use_deterministic_algorithms(True)` 并接受速度损失。

**Bug：`exp()` 在损失计算中返回 `inf`。**
原因：原始 logits 在没有最大值减法技巧的情况下传给 `exp()`。
修复：使用 `torch.nn.functional.log_softmax()`，它在内部实现了 log-sum-exp。

**Bug：从 float32 切换到 float16 后训练发散。**
原因：float16 不能表示低于 6e-8 的梯度量级或高于 65,504 的激活值。
修复：使用带损失缩放的混合精度（AMP），或改用 bfloat16。

```figure
logsumexp-stability
```

## 构建它

### 步骤 1：演示浮点精度极限

```python
print("=== 浮点精度 ===")
print(f"0.1 + 0.2 = {0.1 + 0.2}")
print(f"0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")
print(f"差值: {(0.1 + 0.2) - 0.3:.2e}")
```

### 步骤 2：实现朴素 vs 稳定 softmax

```python
import math

def softmax_naive(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def softmax_stable(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

safe_logits = [2.0, 1.0, 0.1]
print(f"朴素:  {softmax_naive(safe_logits)}")
print(f"稳定: {softmax_stable(safe_logits)}")

dangerous_logits = [100.0, 101.0, 102.0]
print(f"稳定: {softmax_stable(dangerous_logits)}")
# softmax_naive(dangerous_logits) 会返回 [nan, nan, nan]
```

### 步骤 3：实现稳定 log-sum-exp

```python
def logsumexp_naive(values):
    return math.log(sum(math.exp(v) for v in values))

def logsumexp_stable(values):
    c = max(values)
    return c + math.log(sum(math.exp(v - c) for v in values))

safe = [1.0, 2.0, 3.0]
print(f"朴素:  {logsumexp_naive(safe):.6f}")
print(f"稳定: {logsumexp_stable(safe):.6f}")

large = [500.0, 501.0, 502.0]
print(f"稳定: {logsumexp_stable(large):.6f}")
# logsumexp_naive(large) 返回 inf
```

### 步骤 4：实现稳定交叉熵

```python
def cross_entropy_naive(true_class, logits):
    probs = softmax_naive(logits)
    return -math.log(probs[true_class])

def cross_entropy_stable(true_class, logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = math.log(sum(math.exp(s) for s in shifted))
    log_prob = shifted[true_class] - log_sum_exp
    return -log_prob

logits = [2.0, 5.0, 1.0]
true_class = 1
print(f"朴素:  {cross_entropy_naive(true_class, logits):.6f}")
print(f"稳定: {cross_entropy_stable(true_class, logits):.6f}")
```

### 步骤 5：梯度检查

```python
def numerical_gradient(f, x, h=1e-5):
    grad = []
    for i in range(len(x)):
        x_plus = x[:]
        x_minus = x[:]
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad

def check_gradient(analytical, numerical, tolerance=1e-5):
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        denom = max(abs(a), abs(n), 1e-8)
        rel_error = abs(a - n) / denom
        status = "OK" if rel_error < tolerance else "FAIL"
        print(f"  参数 {i}: 解析={a:.8f} 数值={n:.8f} "
              f"相对误差={rel_error:.2e} [{status}]")

def f(params):
    x, y = params
    return x**2 + 3*x*y + y**3

def f_grad(params):
    x, y = params
    return [2*x + 3*y, 3*x + 3*y**2]

point = [2.0, 1.0]
analytical = f_grad(point)
numerical = numerical_gradient(f, point)
check_gradient(analytical, numerical)
```

## 使用它

### 混合精度模拟

```python
import struct

def float32_to_float16_round(x):
    packed = struct.pack('f', x)
    f32 = struct.unpack('f', packed)[0]
    packed16 = struct.pack('e', f32)
    return struct.unpack('e', packed16)[0]

def simulate_bfloat16(x):
    packed = struct.pack('f', x)
    as_int = int.from_bytes(packed, 'little')
    truncated = as_int & 0xFFFF0000
    repacked = truncated.to_bytes(4, 'little')
    return struct.unpack('f', repacked)[0]
```

### 梯度裁剪

```python
def clip_by_norm(gradients, max_norm):
    total_norm = math.sqrt(sum(g**2 for g in gradients))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        return [g * scale for g in gradients]
    return gradients

grads = [10.0, 20.0, 30.0]
clipped = clip_by_norm(grads, max_norm=5.0)
print(f"原始范数: {math.sqrt(sum(g**2 for g in grads)):.2f}")
print(f"裁剪后范数:  {math.sqrt(sum(g**2 for g in clipped)):.2f}")
print(f"方向保持: {[c/clipped[0] for c in clipped]} == {[g/grads[0] for g in grads]}")
```

### NaN/Inf 检测

```python
def check_tensor(name, values):
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)
    if has_nan or has_inf:
        print(f"警告 {name}: nan={has_nan} inf={has_inf}")
        return False
    return True

check_tensor("good", [1.0, 2.0, 3.0])
check_tensor("bad",  [1.0, float('nan'), 3.0])
check_tensor("ugly", [1.0, float('inf'), 3.0])
```

完整实现及所有边界情况演示见 `code/numerical.py`。

## 输出成果

本课产出：
- `code/numerical.py`，包含稳定 softmax、log-sum-exp、交叉熵、梯度检查和混合精度模拟
- `outputs/prompt-numerical-debugger.md`，用于诊断训练中的 NaN/Inf 和数值问题

这些稳定实现将在阶段 3 构建训练循环时和阶段 4 实现注意力机制时再次出现。

## 练习

1. **灾难性抵消。** 使用 float32 中的朴素公式 `E[x^2] - E[x]^2` 计算 [1000000.0, 1000001.0, 1000002.0] 的方差。然后使用 Welford 在线算法计算。将误差与真实方差 (0.6667) 进行比较。

2. **精度探测。** 找出最小的正 float32 值 `x`，使得在 Python 中 `1.0 + x == 1.0`。这就是机器精度。验证它是否匹配 `numpy.finfo(numpy.float32).eps`。

3. **Log-sum-exp 边界情况。** 测试你的 `logsumexp_stable` 函数：(a) 所有值相等，(b) 一个值远大于其他值，(c) 所有值非常负（-1000）。验证它在朴素版本失败的地方给出正确结果。

4. **神经网络层梯度检查。** 实现单个线性层 `y = Wx + b` 及其解析反向传播。使用 `numerical_gradient` 验证一个 3x2 权重矩阵的正确性。

5. **损失缩放实验。** 模拟 float16 训练：创建范围在 [1e-9, 1e-3] 内的随机梯度，转换为 float16，测量有多少比例变成零。然后应用损失缩放（乘以 1024），转换为 float16，缩放回来，再次测量零的比例。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------------|----------------------|
| IEEE 754 | "浮点标准" | 定义二进制浮点格式、舍入规则和特殊值（inf、nan）的国际标准。每个现代 CPU 和 GPU 都实现它。 |
| 机器精度 | "精度极限" | 在给定浮点格式中使 1.0 + e != 1.0 的最小值 e。对于 float32，约为 1.19e-7。 |
| 灾难性抵消 | "减法导致的精度损失" | 当减去几乎相等的浮点数时，有效数字相互抵消，舍入噪声主导结果。 |
| 上溢 | "数字太大" | 结果超过最大可表示值，变成 inf。exp(89) 在 float32 中上溢。 |
| 下溢 | "数字太小" | 结果比最小可表示正数更接近零，变成 0.0。exp(-104) 在 float32 中下溢。 |
| Log-sum-exp 技巧 | "先减最大值" | 通过提取 exp(max(x)) 来计算 log(sum(exp(x)))，以防止上溢和下溢。用于 softmax、交叉熵和对数概率数学。 |
| 稳定 softmax | "不会爆炸的 softmax" | 在取指数前减去 max(logits)。结果数值上完全相同，不会有上溢。 |
| 梯度检查 | "验证你的反向传播" | 将反向传播的解析梯度与有限差分的数值梯度进行比较，以捕获实现 bug。 |
| 混合精度 | "Float16 前向，float32 后向" | 对速度关键的操作使用低精度浮点数，对数值敏感的操作使用高精度浮点数。典型加速为 2-3 倍。 |
| 损失缩放 | "防止梯度下溢" | 在反向传播前将损失乘以一个大常数，使梯度保持在 float16 的可表示范围内，然后在权重更新前除以相同常数。 |
| bfloat16 | "Brain 浮点" | Google 的 16 位格式，具有 8 位指数（与 float32 相同范围）和 7 位尾数（比 float16 精度低）。在训练中更受青睐。 |
| 梯度裁剪 | "限制梯度范数" | 缩放梯度向量使其范数不超过阈值。防止梯度爆炸破坏权重。 |
| NaN | "非数字" | 由未定义运算（0/0、inf-inf、sqrt(-1)）产生的特殊浮点值。传播到所有后续算术中。 |
| Inf | "无穷大" | 由上溢或除零产生的特殊浮点值。可以组合产生 NaN（inf - inf、inf * 0）。 |
| 数值梯度 | "暴力求导" | 通过计算 f(x+h) 和 f(x-h) 并除以 2h 来近似导数。慢但可靠用于验证。 |

## 进一步阅读

- [每个计算机科学家应该了解的浮点运算知识 (Goldberg 1991)](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html)——权威参考文献，内容密实但完整
- [混合精度训练 (Micikevicius et al., 2018)](https://arxiv.org/abs/1710.03740)——介绍 float16 训练损失缩放的 NVIDIA 论文
- [AMP: 自动混合精度 (PyTorch 文档)](https://pytorch.org/docs/stable/amp.html)——PyTorch 中混合精度的实用指南
- [bfloat16 格式 (Google Cloud TPU 文档)](https://cloud.google.com/tpu/docs/bfloat16)——为什么 Google 为 TPU 选择这种格式
- [Kahan 求和 (维基百科)](https://en.wikipedia.org/wiki/Kahan_summation_algorithm)——减少浮点求和舍入误差的算法
