# 线性代数直觉

> 每个 AI 模型不过是戴着华丽帽子的矩阵数学。

**类型：** 学习
**语言：** Python, Julia
**前置知识：** Phase 0
**时间：** 约 60 分钟

## 学习目标

- 在 Python 中从零实现向量和矩阵运算（加法、点积、矩阵乘法）
- 从几何角度解释点积、投影和 Gram-Schmidt 过程的作用
- 使用行约简确定一组向量的线性无关性、秩和基
- 将线性代数概念与其 AI 应用联系起来：嵌入、注意力分数和 LoRA

## 问题所在

打开任何机器学习论文。在第一页内，你就会看到向量、矩阵、点积和变换。没有线性代数直觉，这些只是符号。有了它，你能看到神经网络实际上在做什么——在空间中移动点。

你不需要成为数学家。你需要从几何上理解这些运算的含义，然后自己编写代码。

## 核心概念

### 向量是点（也是方向）

向量只是一个数字列表。但这些数字有含义——它们是空间中的坐标。

**二维向量 [3, 2]：**

| x | y | 点 |
|---|---|-------|
| 3 | 2 | 向量从原点 (0,0) 指向平面上的 (3, 2) |

该向量的大小为 sqrt(3^2 + 2^2) = sqrt(13)，指向右上方。

在 AI 中，向量代表一切：
- 一个词 → 一个 768 个数字的向量（其在嵌入空间中的"含义"）
- 一张图像 → 一个数百万像素值的向量
- 一个用户 → 一个偏好向量

### 矩阵是变换

矩阵将一个向量变换为另一个向量。它可以旋转、缩放、拉伸或投影。

```mermaid
graph LR
    subgraph Before
        A["点 A"]
        B["点 B"]
    end
    subgraph Matrix["矩阵乘法"]
        M["M（变换）"]
    end
    subgraph After
        A2["点 A'"]
        B2["点 B'"]
    end
    A --> M
    B --> M
    M --> A2
    M --> B2
```

在 AI 中，矩阵就是模型本身：
- 神经网络权重 → 将输入变换为输出的矩阵
- 注意力分数 → 决定关注什么的矩阵
- 嵌入 → 将词映射为向量的矩阵

### 点积衡量相似性

两个向量的点积告诉你它们有多相似。

```
a · b = a₁×b₁ + a₂×b₂ + ... + aₙ×bₙ

同方向：      a · b > 0  （相似）
垂直：        a · b = 0  （无关）
反方向：      a · b < 0  （不相似）
```

这正是搜索引擎、推荐系统和 RAG 的工作方式——找到点积高的向量。

### 线性无关性

如果集合中没有任何向量可以表示为其他向量的组合，则这些向量是线性无关的。如果 v1、v2、v3 是无关的，它们张成一个三维空间。如果其中一个是其他向量的组合，它们只张成一个平面。

为什么这对 AI 很重要：你的特征矩阵应该具有线性无关的列。如果两个特征完全相关（线性相关），模型无法区分它们的影响。这会导致回归中的多重共线性——权重矩阵变得不稳定，微小的输入变化会产生剧烈的输出波动。

**具体示例：**

```
v1 = [1, 0, 0]
v2 = [0, 1, 0]
v3 = [2, 1, 0]   # v3 = 2*v1 + v2
```

v1 和 v2 是无关的——没有一个可以表示为另一个的标量倍数或组合。但 v3 = 2*v1 + v2，所以 {v1, v2, v3} 是一个相关集合。这三个向量都位于 xy 平面内。无论你怎么组合它们，都无法到达 [0, 0, 1]。你有三个向量但只有两个维度的自由度。

在数据集中：如果 feature_3 = 2*feature_1 + feature_2，添加 feature_3 不会给模型提供新信息。更糟糕的是，它使正规方程奇异——权重没有唯一解。

### 基和秩

基是一组最小的线性无关向量，它们张成整个空间。基向量的数量就是空间的维度。

三维空间的标准基是 {[1,0,0], [0,1,0], [0,0,1]}。但三维空间中任意三个无关向量都构成一个有效的基。基的选择就是坐标系的选择。

矩阵的秩 = 线性无关列的数量 = 线性无关行的数量。如果秩 < min(行数, 列数)，矩阵是秩亏的。这意味着：
- 系统有无穷多解（或无解）
- 变换中丢失了信息
- 矩阵不可逆

| 情况 | 秩 | 对机器学习的意义 |
|-----------|------|---------------------|
| 满秩（秩 = min(m, n)） | 最大可能 | 存在唯一的最小二乘解。模型是良态的。 |
| 秩亏（秩 < min(m, n)） | 低于最大 | 特征是冗余的。有无穷多权重解。需要正则化。 |
| 秩 1 | 1 | 每一列都是一个向量的缩放副本。所有数据都在一条线上。 |
| 接近秩亏（小奇异值） | 数值上低 | 矩阵是病态的。微小的输入噪声会导致大的输出变化。使用 SVD 截断或岭回归。 |

### 投影

将向量 **a** 投影到向量 **b** 上给出 **a** 在 **b** 方向上的分量：

```
proj_b(a) = (a dot b / b dot b) * b
```

残差（a - proj_b(a)）垂直于 b。这种正交分解是最小二乘拟合的基础。

投影在机器学习中无处不在：
- 线性回归最小化从观测值到列空间的距离——解就是一个投影
- PCA 将数据投影到最大方差方向上
- Transformer 中的注意力计算查询在键上的投影

```mermaid
graph LR
    subgraph Projection["a 在 b 上的投影"]
        direction TB
        O["原点"] --> |"b（方向）"| B["b"]
        O --> |"a（原始）"| A["a"]
        O --> |"proj_b(a)"| P["投影"]
        A -.-> |"残差（垂直）"| P
    end
```

**示例：** a = [3, 4], b = [1, 0]

proj_b(a) = (3*1 + 4*0) / (1*1 + 0*0) * [1, 0] = 3 * [1, 0] = [3, 0]

投影丢弃了 y 分量。这是最简单的降维形式——丢弃你不关心的方向。

### Gram-Schmidt 过程

将任意一组无关向量转换为正交归一基。正交归一意味着每个向量长度为 1，每对向量都互相垂直。

算法：
1. 取第一个向量，归一化
2. 取第二个向量，减去它在第一个向量上的投影，归一化
3. 取第三个向量，减去它在所有先前向量上的投影，归一化
4. 对剩余向量重复

```
输入：v1, v2, v3, ...（线性无关）

u1 = v1 / |v1|

w2 = v2 - (v2 dot u1) * u1
u2 = w2 / |w2|

w3 = v3 - (v3 dot u1) * u1 - (v3 dot u2) * u2
u3 = w3 / |w3|

输出：u1, u2, u3, ...（正交归一基）
```

这就是 QR 分解内部的工作方式。Q 是正交归一基，R 捕获投影系数。QR 分解用于：
- 求解线性系统（比高斯消元更稳定）
- 计算特征值（QR 算法）
- 最小二乘回归（标准数值方法）

```figure
eigen-directions
```

## 动手构建

### 第 1 步：从零开始构建向量（Python）

```python
class Vector:
    def __init__(self, components):
        self.components = list(components)
        self.dim = len(self.components)

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.components, other.components)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.components, other.components)])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.components, other.components))

    def magnitude(self):
        return sum(x**2 for x in self.components) ** 0.5

    def normalize(self):
        mag = self.magnitude()
        return Vector([x / mag for x in self.components])

    def cosine_similarity(self, other):
        return self.dot(other) / (self.magnitude() * other.magnitude())

    def __repr__(self):
        return f"Vector({self.components})"


a = Vector([1, 2, 3])
b = Vector([4, 5, 6])

print(f"a + b = {a + b}")
print(f"a · b = {a.dot(b)}")
print(f"|a| = {a.magnitude():.4f}")
print(f"余弦相似度 = {a.cosine_similarity(b):.4f}")
```

### 第 2 步：从零开始构建矩阵（Python）

```python
class Matrix:
    def __init__(self, rows):
        self.rows = [list(row) for row in rows]
        self.shape = (len(self.rows), len(self.rows[0]))

    def __matmul__(self, other):
        if isinstance(other, Vector):
            return Vector([
                sum(self.rows[i][j] * other.components[j] for j in range(self.shape[1]))
                for i in range(self.shape[0])
            ])
        rows = []
        for i in range(self.shape[0]):
            row = []
            for j in range(other.shape[1]):
                row.append(sum(
                    self.rows[i][k] * other.rows[k][j]
                    for k in range(self.shape[1])
                ))
            rows.append(row)
        return Matrix(rows)

    def transpose(self):
        return Matrix([
            [self.rows[j][i] for j in range(self.shape[0])]
            for i in range(self.shape[1])
        ])

    def __repr__(self):
        return f"Matrix({self.rows})"


rotation_90 = Matrix([[0, -1], [1, 0]])
point = Vector([3, 1])

rotated = rotation_90 @ point
print(f"原始：{point}")
print(f"旋转 90°：{rotated}")
```

### 第 3 步：为什么这对 AI 很重要

```python
import random

random.seed(42)
weights = Matrix([[random.gauss(0, 0.1) for _ in range(3)] for _ in range(2)])
input_vector = Vector([1.0, 0.5, -0.3])

output = weights @ input_vector
print(f"输入（3D）：{input_vector}")
print(f"输出（2D）：{output}")
print("这就是神经网络层所做的——矩阵乘法。")
```

### 第 4 步：Julia 版本

```julia
a = [1.0, 2.0, 3.0]
b = [4.0, 5.0, 6.0]

println("a + b = ", a + b)
println("a · b = ", a ⋅ b)       # Julia 支持 Unicode 运算符
println("|a| = ", √(a ⋅ a))
println("余弦值 = ", (a ⋅ b) / (√(a ⋅ a) * √(b ⋅ b)))

# 矩阵-向量乘法
W = [0.1 -0.2 0.3; 0.4 0.5 -0.1]
x = [1.0, 0.5, -0.3]
println("Wx = ", W * x)
println("这就是一个神经网络层。")
```

### 第 5 步：线性无关性和投影从零构建（Python）

```python
def is_linearly_independent(vectors):
    n = len(vectors)
    dim = len(vectors[0].components)
    mat = Matrix([v.components[:] for v in vectors])
    rows = [row[:] for row in mat.rows]
    rank = 0
    for col in range(dim):
        pivot = None
        for row in range(rank, len(rows)):
            if abs(rows[row][col]) > 1e-10:
                pivot = row
                break
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        scale = rows[rank][col]
        rows[rank] = [x / scale for x in rows[rank]]
        for row in range(len(rows)):
            if row != rank and abs(rows[row][col]) > 1e-10:
                factor = rows[row][col]
                rows[row] = [rows[row][j] - factor * rows[rank][j] for j in range(dim)]
        rank += 1
    return rank == n


def project(a, b):
    scalar = a.dot(b) / b.dot(b)
    return Vector([scalar * x for x in b.components])


def gram_schmidt(vectors):
    orthonormal = []
    for v in vectors:
        w = v
        for u in orthonormal:
            proj = project(w, u)
            w = w - proj
        if w.magnitude() < 1e-10:
            continue
        orthonormal.append(w.normalize())
    return orthonormal


v1 = Vector([1, 0, 0])
v2 = Vector([1, 1, 0])
v3 = Vector([1, 1, 1])
basis = gram_schmidt([v1, v2, v3])
for i, u in enumerate(basis):
    print(f"u{i+1} = {u}")
    print(f"  |u{i+1}| = {u.magnitude():.6f}")

print(f"u1 · u2 = {basis[0].dot(basis[1]):.6f}")
print(f"u1 · u3 = {basis[0].dot(basis[2]):.6f}")
print(f"u2 · u3 = {basis[1].dot(basis[2]):.6f}")
```

## 实际应用

现在用 NumPy 做同样的事——这是你在实践中真正会用的：

```python
import numpy as np

a = np.array([1, 2, 3], dtype=float)
b = np.array([4, 5, 6], dtype=float)

print(f"a + b = {a + b}")
print(f"a · b = {np.dot(a, b)}")
print(f"|a| = {np.linalg.norm(a):.4f}")
print(f"余弦值 = {np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)):.4f}")

W = np.random.randn(2, 3) * 0.1
x = np.array([1.0, 0.5, -0.3])
print(f"Wx = {W @ x}")
```

### 使用 NumPy 计算秩、投影和 QR

```python
import numpy as np

A = np.array([[1, 2], [2, 4]])
print(f"秩：{np.linalg.matrix_rank(A)}")

a = np.array([3, 4])
b = np.array([1, 0])
proj = (np.dot(a, b) / np.dot(b, b)) * b
print(f"{a} 在 {b} 上的投影：{proj}")

Q, R = np.linalg.qr(np.random.randn(3, 3))
print(f"Q 是正交的：{np.allclose(Q @ Q.T, np.eye(3))}")
print(f"R 是上三角的：{np.allclose(R, np.triu(R))}")
```

### PyTorch —— 张量是带自动微分的向量

```python
import torch

x = torch.randn(3, requires_grad=True)
y = torch.tensor([1.0, 0.0, 0.0])

similarity = torch.dot(x, y)
similarity.backward()

print(f"x = {x.data}")
print(f"y = {y.data}")
print(f"点积 = {similarity.item():.4f}")
print(f"d(dot)/dx = {x.grad}")
```

点积关于 x 的梯度就是 y。PyTorch 自动计算了这个。神经网络中的每个运算都是由这样的运算构建的——矩阵乘法、点积、投影——自动微分跟踪所有这些运算的梯度。

你刚刚从零构建了 NumPy 一行代码就能完成的功能。现在你知道底层发生了什么了。

## 发布它

本课产出：
- `outputs/prompt-linear-algebra-tutor.md` —— 一个通过几何直觉教授线性代数的 AI 助手提示词

## 联系

本课中的每个概念都连接到现代 AI 的具体部分：

| 概念 | 出现位置 |
|---------|------------------|
| 点积 | Transformer 中的注意力分数，RAG 中的余弦相似度 |
| 矩阵乘法 | 每个神经网络层，每个线性变换 |
| 线性无关性 | 特征选择，避免多重共线性 |
| 秩 | 判断系统是否可解，LoRA（低秩自适应） |
| 投影 | 线性回归（投影到列空间），PCA |
| Gram-Schmidt / QR | 数值求解器，特征值计算 |
| 正交归一基 | 稳定的数值计算，白化变换 |

LoRA 值得特别说明。它通过将权重更新分解为低秩矩阵来微调大型语言模型。LoRA 不是更新一个 4096x4096 的权重矩阵（1600 万参数），而是更新两个大小为 4096x16 和 16x4096 的矩阵（13.1 万参数）。秩 16 约束意味着 LoRA 假设权重更新存在于完整 4096 维空间的 16 维子空间中。这就是线性代数在做实际工作。

## 练习

1. 实现 `Vector.angle_between(other)`，返回两个向量之间的角度（以度为单位）
2. 创建一个二维缩放矩阵，将 x 坐标加倍，y 坐标变为三倍，然后将其应用于向量 [1, 1]
3. 给定 5 个随机词类向量（维度 50），使用余弦相似度找到最相似的两个
4. 验证 Gram-Schmidt 输出确实是正交归一的：检查每对向量的点积为 0，每个向量的大小为 1
5. 创建一个秩为 2 的 3x3 矩阵。使用 `rank()` 方法验证。然后解释列张成的几何对象是什么。
6. 将向量 [1, 2, 3] 投影到 [1, 1, 1] 上。结果在几何上代表什么？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|----------------------|
| 向量 | "一个箭头" | 一个表示 n 维空间中点或方向的数字列表 |
| 矩阵 | "一个数字表" | 一个将向量从一个空间映射到另一个空间的变换 |
| 点积 | "乘完求和" | 两个向量对齐程度的度量——相似性搜索的核心 |
| 嵌入 | "某种 AI 魔法" | 一个表示某物（词、图像、用户）含义的向量 |
| 线性无关性 | "它们不重叠" | 集合中没有任何向量可以表示为其他向量的组合 |
| 秩 | "有多少维度" | 矩阵中线性无关列（或行）的数量 |
| 投影 | "影子" | 一个向量在另一个向量方向上的分量 |
| 基 | "坐标轴" | 一组张成空间的最小无关向量集 |
| 正交归一 | "垂直的单位向量" | 互相垂直且每个长度都为 1 的向量 |
