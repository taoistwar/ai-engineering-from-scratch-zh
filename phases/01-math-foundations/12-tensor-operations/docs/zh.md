# 张量运算

> 张量是数据和深度学习之间的共同语言。每一张图像、每一个句子、每一个梯度都流经它们。

**类型：** 构建
**语言：** Python
**先决条件：** 阶段 1，第 01 课（线性代数直觉），第 02 课（向量、矩阵与运算）
**时间：** ~90 分钟

## 学习目标

- 从头实现一个张量类，包含形状（shape）、步长（strides）、重塑（reshape）、转置（transpose）和逐元素运算
- 应用广播规则对不同形状的张量进行操作，无需复制数据
- 编写 einsum 表达式实现点积、矩阵乘法、外积和批量运算
- 追踪多头注意力机制中每一步的精确张量形状

## 问题

你构建了一个 transformer。前向传播看起来很整洁。运行后得到：`RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x768 and 512x768)`。你盯着这些形状。尝试转置。现在它说 `Expected 4D input (got 3D input)`。你添加了一个 unsqueeze。然后又出错。

形状错误是深度学习代码中最常见的 bug。它们在概念上并不难——每个操作都有一个形状约定——但它们会迅速叠加。一个 transformer 有数十个重塑、转置和广播串联在一起。一个错误的轴，错误就会级联扩散。更糟的是，有些形状错误根本不会抛出异常。它们通过沿错误维度广播或在错误轴上求和，悄无声息地产生垃圾结果。

矩阵处理两组事物之间的成对关系。真实数据无法容纳在两个维度中。一批 32 张 224x224 的 RGB 图像是一个 4D 张量：`(32, 3, 224, 224)`。具有 12 个头的自注意力机制也是 4D：`(batch, heads, seq_len, head_dim)`。你需要一种可推广到任意维度的数据结构，并对所有维度进行清晰组合的运算。这种结构就是张量。掌握其运算后，形状错误将变得轻松可调试。

## 概念

### 张量是什么

张量是一个具有统一数据类型的多维数字数组。维度的数量称为**秩**（rank，或**阶**）。每个维度是一个**轴**（axis）。**形状**（shape）是一个元组，列出每个轴上的大小。

```mermaid
graph LR
    S["标量<br/>秩 0<br/>形状: ()"] --> V["向量<br/>秩 1<br/>形状: (3,)"]
    V --> M["矩阵<br/>秩 2<br/>形状: (2,3)"]
    M --> T3["3D 张量<br/>秩 3<br/>形状: (2,2,2)"]
    T3 --> T4["4D 张量<br/>秩 4<br/>形状: (B,C,H,W)"]
```

总元素数 = 所有大小的乘积。形状 `(2, 3, 4)` 包含 `2 * 3 * 4 = 24` 个元素。

### 深度学习中的张量形状

不同的数据类型按照惯例映射到特定的张量形状。

```mermaid
graph TD
    subgraph 视觉
        V1["(B, C, H, W)<br/>32, 3, 224, 224"]
    end
    subgraph NLP
        N1["(B, T, D)<br/>16, 128, 768"]
    end
    subgraph 注意力
        A1["(B, H, T, D)<br/>16, 12, 128, 64"]
    end
    subgraph 权重
        W1["Linear: (out, in)<br/>Conv2D: (out_c, in_c, kH, kW)<br/>Embedding: (vocab, dim)"]
    end
```

PyTorch 使用 NCHW（通道优先）。TensorFlow 默认使用 NHWC（通道最后）。布局不匹配会导致隐性的减速或错误。

### 内存布局如何工作

内存中的 2D 数组是一维字节序列。**步长**（strides）告诉你在每个轴上移动一步需要跳过多少个元素。

```mermaid
graph LR
    subgraph "行优先（C 序）"
        R["a b c d e f<br/>步长: (3, 1)"]
    end
    subgraph "列优先（F 序）"
        C["a d b e c f<br/>步长: (1, 2)"]
    end
```

转置不会移动数据。它交换步长，使张量变为**非连续的**——一行中的元素在内存中不再相邻。

### 广播规则

广播允许你对不同形状的张量进行操作，无需复制数据。从右边对齐形状。两个维度在相等或其中一个是 1 时是兼容的。维度较少的张量会在左边用 1 填充。

```
张量 A:     (8, 1, 6, 1)
张量 B:        (7, 1, 5)
填充后的 B: (1, 7, 1, 5)
结果:       (8, 7, 6, 5)
```

### Einsum：通用的张量运算

爱因斯坦求和（Einstein summation）为每个轴标注一个字母。在输入中出现但不在输出中出现的轴会被求和。在两者中都出现的轴会被保留。

```mermaid
graph LR
    subgraph "矩阵乘法: ik,kj -> ij"
        A["A(I,K)"] --> |"对 k 求和"| C["C(I,J)"]
        B["B(K,J)"] --> |"对 k 求和"| C
    end
```

关键模式：`i,i->`（点积），`i,j->ij`（外积），`ii->`（迹），`ij->ji`（转置），`bij,bjk->bik`（批量矩阵乘法），`bhtd,bhsd->bhts`（注意力分数）。

```figure
tensor-broadcast
```

## 构建它

代码位于 `code/tensors.py`。每个步骤引用其中的实现。

### 步骤 1：张量存储和步长

张量存储一个扁平数字列表加上形状元数据。步长告诉索引逻辑如何将多维索引映射到扁平位置。

```python
class Tensor:
    def __init__(self, data, shape=None):
        if isinstance(data, (list, tuple)):
            self._data, self._shape = self._flatten_nested(data)
        elif isinstance(data, np.ndarray):
            self._data = data.flatten().tolist()
            self._shape = tuple(data.shape)
        else:
            self._data = [data]
            self._shape = ()

        if shape is not None:
            total = reduce(lambda a, b: a * b, shape, 1)
            if total != len(self._data):
                raise ValueError(
                    f"Cannot reshape {len(self._data)} elements into shape {shape}"
                )
            self._shape = tuple(shape)

        self._strides = self._compute_strides(self._shape)

    @staticmethod
    def _compute_strides(shape):
        if len(shape) == 0:
            return ()
        strides = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]
        return tuple(strides)
```

对于形状 `(3, 4)`，步长为 `(4, 1)`——前进一行跳过 4 个元素，前进一列跳过 1 个元素。

### 步骤 2：重塑（Reshape）、压缩（Squeeze）、扩展（Unsqueeze）

重塑改变形状而不改变元素顺序。元素总数必须保持不变。使用 `-1` 让一个维度自动推断其大小。

```python
t = Tensor(list(range(12)), shape=(2, 6))
r = t.reshape((3, 4))
r = t.reshape((-1, 3))
```

Squeeze 移除大小为 1 的轴。Unsqueeze 插入一个。Unsqueezing 对广播至关重要——将偏置向量 `(D,)` 添加到批量 `(B, T, D)` 需要 unsqueeze 到 `(1, 1, D)`。

```python
t = Tensor(list(range(6)), shape=(1, 3, 1, 2))
s = t.squeeze()
v = Tensor([1, 2, 3])
u = v.unsqueeze(0)
```

### 步骤 3：转置（Transpose）和维度重排（Permute）

转置交换两个轴。Permute 重新排列所有轴。这就是在 NCHW 和 NHWC 之间转换的方法。

```python
mat = Tensor(list(range(6)), shape=(2, 3))
tr = mat.transpose(0, 1)

t4d = Tensor(list(range(24)), shape=(1, 2, 3, 4))
perm = t4d.permute((0, 2, 3, 1))
```

转置或 permute 后，张量在内存中是非连续的。在 PyTorch 中，`view` 在非连续张量上会失败——请使用 `reshape` 或先调用 `.contiguous()`。

### 步骤 4：逐元素运算和归约

逐元素运算（加、乘、减）独立应用于每个元素并保持形状。归约（求和、均值、最大值）压缩一个或多个轴。

```python
a = Tensor([[1, 2], [3, 4]])
b = Tensor([[10, 20], [30, 40]])
c = a + b
d = a * 2
s = a.sum(axis=0)
```

CNN 中的全局平均池化：`(B, C, H, W).mean(axis=[2, 3])` 产生 `(B, C)`。NLP 中的序列平均池化：`(B, T, D).mean(axis=1)` 产生 `(B, D)`。

### 步骤 5：使用 NumPy 进行广播

`tensors.py` 中的 `demo_broadcasting_numpy()` 函数展示了核心模式。

```python
activations = np.random.randn(4, 3)
bias = np.array([0.1, 0.2, 0.3])
result = activations + bias

images = np.random.randn(2, 3, 4, 4)
scale = np.array([0.5, 1.0, 1.5]).reshape(1, 3, 1, 1)
result = images * scale

a = np.array([1, 2, 3]).reshape(-1, 1)
b = np.array([10, 20, 30, 40]).reshape(1, -1)
outer = a * b
```

通过广播计算成对距离：将 `(M, 2)` 重塑为 `(M, 1, 2)`，将 `(N, 2)` 重塑为 `(1, N, 2)`，相减，平方，沿最后一个轴求和，取平方根。结果：`(M, N)`。

### 步骤 6：Einsum 运算

`demo_einsum()` 和 `demo_einsum_gallery()` 函数遍历了每个常见模式。

```python
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])
dot = np.einsum("i,i->", a, b)

A = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
B = np.array([[7, 8, 9], [10, 11, 12]], dtype=float)
matmul = np.einsum("ik,kj->ij", A, B)

batch_A = np.random.randn(4, 3, 5)
batch_B = np.random.randn(4, 5, 2)
batch_mm = np.einsum("bij,bjk->bik", batch_A, batch_B)
```

缩并的计算成本是所有索引大小（保留的和被求和的）的乘积。对于 `bij,bjk->bik`，其中 B=32, I=128, J=64, K=128：`32 * 128 * 64 * 128 = 33,554,432` 次乘加运算。

### 步骤 7：通过 einsum 实现的注意力机制

`demo_attention_einsum()` 函数端到端地实现了多头注意力。

```python
B, H, T, D = 2, 4, 8, 16
E = H * D

X = np.random.randn(B, T, E)
W_q = np.random.randn(E, E) * 0.02

Q = np.einsum("bte,ek->btk", X, W_q)
Q = Q.reshape(B, T, H, D).transpose(0, 2, 1, 3)

scores = np.einsum("bhtd,bhsd->bhts", Q, K) / np.sqrt(D)
weights = softmax(scores, axis=-1)
attn_output = np.einsum("bhts,bhsd->bhtd", weights, V)

concat = attn_output.transpose(0, 2, 1, 3).reshape(B, T, E)
output = np.einsum("bte,ek->btk", concat, W_o)
```

每一步都是一个张量运算：投影（通过 einsum 的矩阵乘法），头拆分（reshape + transpose），注意力分数（通过 einsum 的批量矩阵乘法），加权和（通过 einsum 的批量矩阵乘法），头合并（transpose + reshape），输出投影（通过 einsum 的矩阵乘法）。

## 使用它

### 从零实现 vs NumPy

| 操作 | 从零实现（Tensor 类） | NumPy |
|---|---|---|
| 创建 | `Tensor([[1,2],[3,4]])` | `np.array([[1,2],[3,4]])` |
| 重塑 | `t.reshape((3,4))` | `a.reshape(3,4)` |
| 转置 | `t.transpose(0,1)` | `a.T` 或 `a.transpose(0,1)` |
| 压缩 | `t.squeeze(0)` | `np.squeeze(a, 0)` |
| 求和 | `t.sum(axis=0)` | `a.sum(axis=0)` |
| Einsum | 不支持 | `np.einsum("ij,jk->ik", a, b)` |

### 从零实现 vs PyTorch

```python
import torch

t = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
t.shape
t.stride()
t.is_contiguous()

t.reshape(3, 2)
t.unsqueeze(0)
t.transpose(0, 1)
t.transpose(0, 1).contiguous()

torch.einsum("ik,kj->ij", A, B)
```

PyTorch 增加了自动求导、GPU 支持和优化的 BLAS 内核。形状语义完全相同。如果你理解了从零实现，PyTorch 形状错误将变得可读。

### 每个神经网络层作为一种张量运算

| 运算 | 张量形式 | Einsum |
|---|---|---|
| 线性层 | `Y = X @ W.T + b` | `"bd,od->bo"` + 偏置 |
| 注意力 QKV | `Q = X @ W_q` | `"btd,dh->bth"` |
| 注意力分数 | `Q @ K.T / sqrt(d)` | `"bhtd,bhsd->bhts"` |
| 注意力输出 | `softmax(scores) @ V` | `"bhts,bhsd->bhtd"` |
| 批量归一化 | `(X - mu) / sigma * gamma` | 逐元素 + 广播 |
| Softmax | `exp(x) / sum(exp(x))` | 逐元素 + 归约 |

## 输出成果

本课产生两个可复用的提示词：

1. **`outputs/prompt-tensor-shapes.md`**——一个用于调试张量形状不匹配的系统化提示词。包含每种常见运算（matmul、broadcast、cat、Linear、Conv2d、BatchNorm、softmax）的决策表和修复查找表。

2. **`outputs/prompt-tensor-debugger.md`**——一个逐步调试提示词，当形状错误阻碍你时，可以粘贴到任意 AI 助手中。将错误信息和你的张量形状输入进去，获得精确的修复方案。

## 练习

1. **简单——重塑往返。**取一个形状为 `(2, 3, 4)` 的张量。将其重塑为 `(6, 4)`，然后重塑为 `(24,)`，再重塑回 `(2, 3, 4)`。通过打印扁平数据来验证每一步的元素顺序保持不变。

2. **中等——实现广播。**为 `Tensor` 类扩展一个 `broadcast_to(shape)` 方法，将大小为 1 的维度扩展以匹配目标形状。然后修改 `_elementwise_op` 使其在操作前自动广播。用形状 `(3, 1)` 和 `(1, 4)` 进行测试，产生 `(3, 4)`。

3. **困难——从头构建 einsum。**实现一个基本的 `einsum(subscripts, *tensors)` 函数，至少处理：点积（`i,i->`）、矩阵乘法（`ij,jk->ik`）、外积（`i,j->ij`）和转置（`ij->ji`）。解析下标字符串，识别缩并索引，并遍历所有索引组合。将你的结果与 `np.einsum` 进行比较。

4. **困难——注意力形状追踪器。**编写一个函数，接受 `batch_size`、`seq_len`、`embed_dim` 和 `num_heads` 作为输入，打印多头注意力每一步的精确形状：输入、Q/K/V 投影、头拆分、注意力分数、softmax 权重、加权和、头合并、输出投影。与 `demo_attention_einsum()` 的输出进行验证。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|---|---|---|
| 张量 | "比矩阵多几个维度" | 具有统一类型和定义好的形状、步长及运算的多维数组 |
| 秩 | "维度的数量" | 轴的数量。矩阵的秩为 2，而不是其矩阵秩 |
| 形状 | "张量的大小" | 列出每个轴大小的元组。`(2, 3)` 表示 2 行 3 列 |
| 步长 | "内存如何布局" | 沿每个轴前进一个位置需要跳过的元素数量 |
| 广播 | "形状不同时它自动处理" | 一组严格的规则：从右边对齐，维度必须相等或其中一个为 1 |
| 连续 | "张量是正常的" | 元素在内存中按逻辑布局顺序无间隙地存储 |
| Einsum | "一种优雅的矩阵乘法写法" | 一种通用记号，用一行表达任何张量缩并、外积、迹或转置 |
| View | "和 reshape 一样" | 与重塑共享相同内存缓冲但具有不同形状/步长元数据的张量。在非连续数据上会失败 |
| 缩并 | "对一个索引求和" | 两个张量之间的共享索引进行乘加运算，产生较低秩的结果的通用运算 |
| NCHW / NHWC | "PyTorch vs TensorFlow 格式" | 图像张量的内存布局约定。NCHW 将通道放在空间维度之前，NHWC 放在之后 |

## 进一步阅读

- [NumPy 广播](https://numpy.org/doc/stable/user/basics.broadcasting.html)——带有视觉示例的规范规则
- [PyTorch 张量视图](https://pytorch.org/docs/stable/tensor_view.html)——视图何时有效、何时复制
- [einops](https://github.com/arogozhnikov/einops)——让张量重塑变得可读且安全的库
- [图解 Transformer](https://jalammar.github.io/illustrated-transformer/)——可视化流经注意力的张量形状
- [NumPy 中的爱因斯坦求和](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html)——完整的 einsum 文档及示例
