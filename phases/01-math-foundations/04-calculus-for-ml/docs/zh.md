# 机器学习中的微积分

> 导数告诉你哪个方向是下坡。这就是神经网络学习所需的全部。

**Type:** Learn
**Language:** Python
**Prerequisites:** Phase 1, Lessons 01-03
**Time:** ~60 分钟

## 学习目标

- 为常见的机器学习函数（x^2、sigmoid、交叉熵）计算数值导数和解析导数
- 从零实现梯度下降，在一维和二维中最小化损失函数
- 推导线性回归模型的梯度，并通过手动权重更新进行训练
- 解释 Hessian 矩阵、Taylor 级数近似及其与优化方法的联系

## 问题所在

你有一个包含数百万个权重的神经网络。每个权重都是一个旋钮。你需要弄清楚每个旋钮该往哪个方向转，才能让模型的错误稍微少一点。微积分给你指明了方向。

没有微积分，训练神经网络就意味着随机尝试改动，然后祈祷最好的结果。有了导数，你就能确切知道每个权重如何影响误差。你每次都能以正确的方式转动每个旋钮。

## 概念

### 什么是导数？

导数衡量的是变化率。对于函数 y = f(x)，导数 f'(x) 告诉你：如果你将 x 微调一点点，y 会变化多少？

在几何上，导数是某一点处切线的斜率。

**f(x) = x^2：**

| x | f(x) | f'(x)（斜率） |
|---|------|---------------|
| 0 | 0    | 0（平坦，在底部） |
| 1 | 1    | 2 |
| 2 | 4    | 4（该点处切线的斜率） |
| 3 | 9    | 6 |

在 x=2 处，斜率是 4。如果你将 x 向右移动一点点，y 大约会增加那个量的 4 倍。在 x=0 处，斜率是 0。你处于碗底。

形式化定义：

```
f'(x) = lim   f(x + h) - f(x)
        h->0  -----------------
                     h
```

在代码中，你跳过极限，直接使用一个非常小的 h。这就是数值导数。

### 偏导数：一次一个变量

实际函数有很多输入。神经网络的损失取决于数千个权重。偏导数保持所有变量不变，只对其中一个求导。

```
f(x, y) = x^2 + 3xy + y^2

df/dx = 2x + 3y     (将 y 视为常数)
df/dy = 3x + 2y     (将 x 视为常数)
```

每个偏导数回答的问题是：如果我只微调这一个权重，损失会如何变化？

### 梯度：所有偏导数的向量

梯度将每个偏导数收集到一个向量中。对于函数 f(x, y, z)，梯度是：

```
grad f = [ df/dx, df/dy, df/dz ]
```

梯度指向最陡上升的方向。要最小化一个函数，就朝相反方向走。

**f(x,y) = x^2 + y^2 的等高线图：**

该函数形成一个碗形，等高线是同心圆。最小值在 (0, 0)。

| 点 | grad f | -grad f（下降方向） |
|-------|--------|----------------------------|
| (1, 1) | [2, 2]（指向上坡，远离最小值） | [-2, -2]（指向下坡，朝向最小值） |
| (0, 0) | [0, 0]（平坦，在最小值处） | [0, 0] |

这就是图片中的梯度下降。计算梯度，取反，走一步。

### 与优化的联系

训练神经网络就是优化。你有一个损失函数 L(w1, w2, ..., wn) 来衡量模型有多大的错误。你想要最小化它。

```
梯度下降更新规则：

  w_new = w_old - learning_rate * dL/dw

对于每个权重：
  1. 计算损失对该权重的偏导数
  2. 从权重中减去它的一个小的倍数
  3. 重复
```

学习率控制步长。太大你会冲过头。太小你会爬行。

**损失地形（一维切片）：**

损失函数 L(w) 随着权重 w 的变化形成一条有峰有谷的曲线。

| 特征 | 描述 |
|---------|-------------|
| 全局最小值 | 整条曲线上最低的点——最优解 |
| 局部最小值 | 一个比邻居低但并非整体最低的谷 |
| 斜率 | 梯度下降从任意起点沿斜率下坡 |

梯度下降沿斜率下坡。它可能陷入局部最小值，但在高维空间（数百万个权重）中，这在实际中很少成为问题。

### 数值导数 vs 解析导数

计算导数有两种方法。

解析法：手动应用微积分规则。对于 f(x) = x^2，导数是 f'(x) = 2x。精确。快速。

数值法：用定义近似。为一个极小的 h 计算 f(x+h) 和 f(x-h)，然后用差值。

```
数值法（中心差分）：

f'(x) ~= f(x + h) - f(x - h)
          -----------------------
                  2h

h = 0.0001 在实践中效果很好
```

数值导数更慢，但适用于任何函数。解析导数更快，但需要你推导公式。神经网络框架使用第三种方法：自动微分，它机械地计算精确导数。你将在 Phase 3 中看到。

### 简单函数的导数推导

这些是你在机器学习中反复看到的导数。

```
函数              导数                    用于
--------          ----------              -------
f(x) = x^2        f'(x) = 2x             损失函数（MSE）
f(x) = wx + b     f'(w) = x              线性层（对权重的梯度）
                  f'(b) = 1              线性层（对偏置的梯度）
                  f'(x) = w              线性层（对输入的梯度）
f(x) = e^x        f'(x) = e^x           Softmax，注意力
f(x) = ln(x)      f'(x) = 1/x           交叉熵损失
f(x) = 1/(1+e^-x) f'(x) = f(x)(1-f(x))  Sigmoid 激活
```

对于 f(x) = x^2：

```
f(x) = x^2    f'(x) = 2x

  x    f(x)   f'(x)   含义
  -2    4      -4      斜率向左倾斜（递减）
  -1    1      -2      斜率向左倾斜（递减）
   0    0       0      平坦（最小值！）
   1    1       2      斜率向右倾斜（递增）
   2    4       4      斜率向右倾斜（递增）
```

对于 f(w) = wx + b，其中 x=3, b=1：

```
f(w) = 3w + 1    f'(w) = 3

对 w 的导数就是 x。
如果 x 很大，w 的微小变化会导致输出的巨大变化。
```

### 链式法则

当函数复合时，链式法则告诉你如何求导。

```
如果 y = f(g(x))，则 dy/dx = f'(g(x)) * g'(x)

示例：y = (3x + 1)^2
  外层：f(u) = u^2       f'(u) = 2u
  内层：g(x) = 3x + 1    g'(x) = 3
  dy/dx = 2(3x + 1) * 3 = 6(3x + 1)
```

神经网络是函数的链式组合：输入 -> 线性 -> 激活 -> 线性 -> 激活 -> 损失。反向传播就是从输出到输入反复应用链式法则。这就是整个算法。

### Hessian 矩阵

梯度告诉你斜率。Hessian 告诉你曲率。

Hessian 是二阶偏导数的矩阵。对于函数 f(x1, x2, ..., xn)，Hessian 的第 (i, j) 个元素是：

```
H[i][j] = d^2f / (dx_i * dx_j)
```

对于两变量函数 f(x, y)：

```
H = | d^2f/dx^2    d^2f/dxdy |
    | d^2f/dydx    d^2f/dy^2 |
```

**在临界点（梯度 = 0）处 Hessian 告诉你的信息：**

| Hessian 性质 | 含义 | 示例曲面 |
|-----------------|---------|-----------------|
| 正定（所有特征值 > 0） | 局部最小值 | 朝上的碗 |
| 负定（所有特征值 < 0） | 局部最大值 | 朝下的碗 |
| 不定（混合特征值） | 鞍点 | 马鞍形状 |

**示例：** f(x, y) = x^2 - y^2（鞍函数）

```
df/dx = 2x       df/dy = -2y
d^2f/dx^2 = 2    d^2f/dy^2 = -2    d^2f/dxdy = 0

H = | 2   0 |
    | 0  -2 |

特征值：2 和 -2（一正一负）
--> (0, 0) 处为鞍点
```

与 f(x, y) = x^2 + y^2（碗形）对比：

```
H = | 2  0 |
    | 0  2 |

特征值：2 和 2（都为正）
--> (0, 0) 处为局部最小值
```

**为什么 Hessian 在机器学习中重要：**

Newton 法使用 Hessian 来比梯度下降做出更好的优化步骤。它不仅跟随斜率，还考虑曲率：

```
Newton 更新：    w_new = w_old - H^(-1) * gradient
梯度下降：       w_new = w_old - lr * gradient
```

Newton 法收敛更快，因为 Hessian "重新缩放"了梯度——陡峭的方向获得更小的步长，平坦的方向获得更大的步长。

问题是：对于有 N 个参数的神经网络，Hessian 是 N x N 的。一个有 100 万参数的模型需要一个 1 万亿个元素的矩阵。这就是为什么我们使用近似方法。

| 方法 | 使用什么 | 代价 | 收敛性 |
|--------|-------------|------|-------------|
| 梯度下降 | 仅一阶导数 | 每步 O(N) | 慢（线性） |
| Newton 法 | 完整 Hessian | 每步 O(N^3) | 快（二次） |
| L-BFGS | 从梯度历史近似 Hessian | 每步 O(N) | 中等（超线性） |
| Adam | 每参数自适应学习率（对角 Hessian 近似） | 每步 O(N) | 中等 |
| 自然梯度 | Fisher 信息矩阵（统计 Hessian） | 每步 O(N^2) | 快 |

在实践中，Adam 是深度学习的默认优化器。它通过跟踪每个参数梯度的运行均值和方差，廉价地近似二阶信息。

### Taylor 级数近似

任何光滑函数都可以在局部用多项式近似：

```
f(x + h) = f(x) + f'(x)*h + (1/2)*f''(x)*h^2 + (1/6)*f'''(x)*h^3 + ...
```

你包含的项越多，近似就越好——但只在点 x 附近。

**为什么 Taylor 级数对机器学习重要：**

- **一阶 Taylor = 梯度下降。** 当你使用 f(x + h) ~ f(x) + f'(x)*h 时，你在做线性近似。梯度下降最小化这个线性模型来选择 h = -lr * f'(x)。

- **二阶 Taylor = Newton 法。** 使用 f(x + h) ~ f(x) + f'(x)*h + (1/2)*f''(x)*h^2，你得到一个二次模型。最小化它得到 h = -f'(x)/f''(x)——Newton 步。

- **损失函数设计。** MSE 和交叉熵是光滑的，这意味着它们的 Taylor 展开表现良好。这不是巧合。光滑的损失使优化可预测。

```
近似阶数              捕获的内容            优化方法
-------------------    -----------------   -------------------
0 阶（常数）           仅值                  随机搜索
1 阶（线性）           斜率                  梯度下降
2 阶（二次）           曲率                  Newton 法
更高阶                 更精细的结构          在机器学习中很少使用
```

关键洞察：所有基于梯度的优化实际上都是在局部近似损失函数，并走向该近似的最小值。

### 机器学习中的积分

导数告诉你变化率。积分计算累积量——曲线下的面积。

在机器学习中，你很少手动计算积分，但这个概念无处不在：

**概率。** 对于具有密度 p(x) 的连续随机变量：
```
P(a < X < b) = integral from a to b of p(x) dx
```
概率密度曲线在 a 和 b 之间的面积就是落在该范围内的概率。

**期望值。** 按概率加权的平均结果：
```
E[f(X)] = integral of f(x) * p(x) dx
```
数据分布上的期望损失是一个积分。训练最小化的是它的经验近似。

**KL 散度。** 衡量两个分布的差异程度：
```
KL(p || q) = integral of p(x) * log(p(x) / q(x)) dx
```
用于 VAE、知识蒸馏和贝叶斯推断。

**归一化常数。** 在贝叶斯推断中：
```
p(w | data) = p(data | w) * p(w) / integral of p(data | w) * p(w) dw
```
分母是对所有可能参数值的积分。它通常是难以处理的，这就是为什么我们使用 MCMC 和变分推断等近似方法。

| 积分概念 | 在机器学习中的出现 |
|-----------------|----------------------|
| 曲线下面积 | 从密度函数得到概率 |
| 期望值 | 损失函数、风险最小化 |
| KL 散度 | VAE、策略优化、蒸馏 |
| 归一化 | 贝叶斯后验、softmax 分母 |
| 边缘似然 | 模型比较、证据下界（ELBO） |

### 计算图中的多变链式法则

链式法则不仅适用于排成一线的标量函数。在神经网络中，变量会分叉和合并。以下是导数如何在简单的前向传播中流动：

```mermaid
graph LR
    x["x（输入）"] -->|"*w"| z1["z1 = w*x"]
    z1 -->|"+b"| z2["z2 = w*x + b"]
    z2 -->|"sigmoid"| a["a = sigmoid(z2)"]
    a -->|"损失函数"| L["L = -(y*log(a) + (1-y)*log(1-a))"]
```

反向传播从右到左计算梯度：

```mermaid
graph RL
    dL["dL/dL = 1"] -->|"dL/da"| da["dL/da = -y/a + (1-y)/(1-a)"]
    da -->|"da/dz2 = a(1-a)"| dz2["dL/dz2 = dL/da * a(1-a)"]
    dz2 -->|"dz2/dw = x"| dw["dL/dw = dL/dz2 * x"]
    dz2 -->|"dz2/db = 1"| db["dL/db = dL/dz2 * 1"]
```

每条箭头乘以局部导数。任何参数的梯度是从损失到该参数路径上所有局部导数的乘积。当路径分叉和合并时，你将贡献求和（多变链式法则）。

这就是反向传播的全部：通过计算图从输出到输入系统地应用链式法则。

### Jacobian 矩阵

当一个函数将向量映射到向量时（如神经网络层），它的导数是一个矩阵。Jacobian 包含每个输出对每个输入的所有偏导数。

对于 f: R^n -> R^m，Jacobian J 是一个 m x n 矩阵：

| | x1 | x2 | ... | xn |
|---|---|---|---|---|
| f1 | df1/dx1 | df1/dx2 | ... | df1/dxn |
| f2 | df2/dx1 | df2/dx2 | ... | df2/dxn |
| ... | ... | ... | ... | ... |
| fm | dfm/dx1 | dfm/dx2 | ... | dfm/dxn |

你不会为神经网络手动计算 Jacobian。PyTorch 会处理。但知道它的存在有助于你理解反向传播中的形状：如果一层将 R^n 映射到 R^m，它的 Jacobian 是 m x n。梯度通过这个矩阵的转置反向流动。

### 为什么这对神经网络重要

神经网络中的每个权重都会得到一个梯度。梯度告诉你如何调整该权重以减少损失。

```mermaid
graph LR
    subgraph Forward["前向传播"]
        I["输入"] --> W1["W1"] --> R["relu"] --> W2["W2"] --> S["softmax"] --> L["损失"]
    end
```

```mermaid
graph RL
    subgraph Backward["反向传播"]
        dL["dL/dloss"] --> dW2["dL/dW2"] --> d2["..."] --> dW1["dL/dW1"]
    end
```

每次权重更新：
- `W1 = W1 - lr * dL/dW1`
- `W2 = W2 - lr * dL/dW2`

前向传播计算预测和损失。反向传播计算损失对每个权重的梯度。然后每个权重向下坡迈出一小步。重复数百万步。这就是深度学习。

```figure
derivative-tangent
```

## 构建它

### 步骤 1：从零开始计算数值导数

```python
def numerical_derivative(f, x, h=1e-7):
    return (f(x + h) - f(x - h)) / (2 * h)

def f(x):
    return x ** 2

for x in [-2, -1, 0, 1, 2]:
    numerical = numerical_derivative(f, x)
    analytical = 2 * x
    print(f"x={x:2d}  f'(x) 数值={numerical:.6f}  解析={analytical:.1f}")
```

数值导数与解析导数在小数点后很多位上匹配。

### 步骤 2：偏导数和梯度

```python
def numerical_gradient(f, point, h=1e-7):
    gradient = []
    for i in range(len(point)):
        point_plus = list(point)
        point_minus = list(point)
        point_plus[i] += h
        point_minus[i] -= h
        partial = (f(point_plus) - f(point_minus)) / (2 * h)
        gradient.append(partial)
    return gradient

def f_multi(point):
    x, y = point
    return x**2 + 3*x*y + y**2

grad = numerical_gradient(f_multi, [1.0, 2.0])
print(f"(1,2) 处的数值梯度：{[f'{g:.4f}' for g in grad]}")
print(f"(1,2) 处的解析梯度：[2*1+3*2, 3*1+2*2] = [{2*1+3*2}, {3*1+2*2}]")
```

### 步骤 3：梯度下降求 f(x) = x^2 的最小值

```python
x = 5.0
lr = 0.1
for step in range(20):
    grad = 2 * x
    x = x - lr * grad
    print(f"步骤 {step:2d}  x={x:8.4f}  f(x)={x**2:10.6f}")
```

从 x=5 开始，每一步都更接近 x=0（最小值）。

### 步骤 4：二维函数上的梯度下降

```python
def f_2d(point):
    x, y = point
    return x**2 + y**2

point = [4.0, 3.0]
lr = 0.1
for step in range(30):
    grad = numerical_gradient(f_2d, point)
    point = [p - lr * g for p, g in zip(point, grad)]
    loss = f_2d(point)
    if step % 5 == 0 or step == 29:
        print(f"步骤 {step:2d}  点=({point[0]:7.4f}, {point[1]:7.4f})  f={loss:.6f}")
```

### 步骤 5：比较数值导数和解析导数

```python
import math

test_functions = [
    ("x^2",      lambda x: x**2,          lambda x: 2*x),
    ("x^3",      lambda x: x**3,          lambda x: 3*x**2),
    ("sin(x)",   lambda x: math.sin(x),   lambda x: math.cos(x)),
    ("e^x",      lambda x: math.exp(x),   lambda x: math.exp(x)),
    ("1/x",      lambda x: 1/x,           lambda x: -1/x**2),
]

x = 2.0
print(f"{'函数':<12} {'数值':>12} {'解析':>12} {'误差':>12}")
print("-" * 50)
for name, f, df in test_functions:
    num = numerical_derivative(f, x)
    ana = df(x)
    err = abs(num - ana)
    print(f"{name:<12} {num:12.6f} {ana:12.6f} {err:12.2e}")
```

### 步骤 6：数值计算 Hessian

```python
def hessian_2d(f, x, y, h=1e-5):
    fxx = (f(x + h, y) - 2 * f(x, y) + f(x - h, y)) / (h ** 2)
    fyy = (f(x, y + h) - 2 * f(x, y) + f(x, y - h)) / (h ** 2)
    fxy = (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h ** 2)
    return [[fxx, fxy], [fxy, fyy]]

def saddle(x, y):
    return x ** 2 - y ** 2

def bowl(x, y):
    return x ** 2 + y ** 2

H_saddle = hessian_2d(saddle, 0.0, 0.0)
H_bowl = hessian_2d(bowl, 0.0, 0.0)
print(f"鞍点 Hessian: {H_saddle}")  # [[2, 0], [0, -2]] -- 混合符号
print(f"碗形 Hessian:  {H_bowl}")    # [[2, 0], [0, 2]]  -- 都为正
```

鞍函数的 Hessian 有特征值 2 和 -2（混合符号，确认鞍点）。碗形的特征值为 2 和 2（都为正，确认最小值）。

### 步骤 7：Taylor 近似实战

```python
import math

def taylor_approx(f, f_prime, f_double_prime, x0, h, order=2):
    result = f(x0)
    if order >= 1:
        result += f_prime(x0) * h
    if order >= 2:
        result += 0.5 * f_double_prime(x0) * h ** 2
    return result

x0 = 0.0
for h in [0.1, 0.5, 1.0, 2.0]:
    true_val = math.sin(h)
    t1 = taylor_approx(math.sin, math.cos, lambda x: -math.sin(x), x0, h, order=1)
    t2 = taylor_approx(math.sin, math.cos, lambda x: -math.sin(x), x0, h, order=2)
    print(f"h={h:.1f}  sin(h)={true_val:.4f}  1阶={t1:.4f}  2阶={t2:.4f}")
```

在 x0=0 附近，sin(x) ~ x（一阶 Taylor）。对于小的 h，近似非常好，但对于大的 h 会失效。这就是为什么梯度下降在小的学习率下效果最好——每一步都假设线性近似是准确的。

### 步骤 8：为什么这对神经网络重要

```python
import random

random.seed(42)

w = random.gauss(0, 1)
b = random.gauss(0, 1)
lr = 0.01

xs = [1.0, 2.0, 3.0, 4.0, 5.0]
ys = [3.0, 5.0, 7.0, 9.0, 11.0]

for epoch in range(200):
    total_loss = 0
    dw = 0
    db = 0
    for x, y in zip(xs, ys):
        pred = w * x + b
        error = pred - y
        total_loss += error ** 2
        dw += 2 * error * x
        db += 2 * error
    dw /= len(xs)
    db /= len(xs)
    total_loss /= len(xs)
    w -= lr * dw
    b -= lr * db
    if epoch % 40 == 0 or epoch == 199:
        print(f"epoch {epoch:3d}  w={w:.4f}  b={b:.4f}  loss={total_loss:.6f}")

print(f"\n学到：y = {w:.2f}x + {b:.2f}")
print(f"实际：y = 2x + 1")
```

每个基于梯度的训练循环都遵循这个模式：预测，计算损失，计算梯度，更新权重。

## 使用它

使用 NumPy，同样的操作更快更简洁：

```python
import numpy as np

x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([3, 5, 7, 9, 11], dtype=float)

w, b = np.random.randn(), np.random.randn()
lr = 0.01

for epoch in range(200):
    pred = w * x + b
    error = pred - y
    loss = np.mean(error ** 2)
    dw = np.mean(2 * error * x)
    db = np.mean(2 * error)
    w -= lr * dw
    b -= lr * db

print(f"学到：y = {w:.2f}x + {b:.2f}")
```

你刚刚从零构建了梯度下降。PyTorch 自动化了梯度计算，但更新循环是完全相同的。

## 练习

1. 实现 `numerical_second_derivative(f, x)`，调用两次 `numerical_derivative`。验证 x^3 在 x=2 处的二阶导数是 12。
2. 使用梯度下降求 f(x, y) = (x - 3)^2 + (y + 1)^2 的最小值。从 (0, 0) 开始。答案应收敛到 (3, -1)。
3. 为梯度下降循环添加动量：维护一个累积过去梯度的速度向量。比较在 f(x) = x^4 - 3x^2 上有和没有动量的收敛速度。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|----------------------|
| 导数 | "斜率" | 函数在某一点的变化率。告诉你输出随输入单位变化变化多少。 |
| 偏导数 | "对一个变量求导" | 对其他变量保持不变时，对一个变量的导数。 |
| 梯度 | "最陡上升方向" | 所有偏导数的向量。指向函数增长最快的方向。 |
| 梯度下降 | "向下走" | 从参数中减去梯度（乘以学习率）以减少损失。神经网络训练的核心。 |
| 学习率 | "步长" | 控制每次梯度下降步长大小的标量。太大：发散。太小：收敛缓慢。 |
| 链式法则 | "导数相乘" | 复合函数的求导规则：df/dx = df/dg * dg/dx。反向传播的数学基础。 |
| Jacobian | "导数矩阵" | 当函数将向量映射到向量时，Jacobian 是输出对输入的所有偏导数的矩阵。 |
| 数值导数 | "有限差分" | 通过在两个相邻点求值并计算它们之间的斜率来近似导数。 |
| 反向传播 | "反向自动微分" | 使用链式法则从输出到输入逐层计算梯度。神经网络学习的方式。 |
| Hessian | "二阶导数矩阵" | 所有二阶偏导数的矩阵。描述函数的曲率。临界点处正定的 Hessian 意味着局部最小值。 |
| Taylor 级数 | "多项式近似" | 使用导数在某点附近近似函数：f(x+h) ~ f(x) + f'(x)h + (1/2)f''(x)h^2 + ... 理解梯度下降和 Newton 法为何有效的基础。 |
| 积分 | "曲线下面积" | 一个量在某个范围内的累积。在机器学习中，积分定义概率、期望值和 KL 散度。 |

## 延伸阅读

- [3Blue1Brown: 微积分的本质](https://www.3blue1brown.com/topics/calculus) - 导数、积分和链式法则的直观理解
- [Stanford CS231n: 反向传播](https://cs231n.github.io/optimization-2/) - 梯度如何在神经网络层中流动
