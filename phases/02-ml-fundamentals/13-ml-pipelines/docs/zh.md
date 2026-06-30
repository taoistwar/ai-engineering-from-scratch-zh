# ML 管道

> 一个模型不是一个产品。一个管道才是。管道是从原始数据到部署预测的一切，每一步都必须可复现。

**类型：** 构建
**语言：** Python
**先决条件：** 阶段 2，第 12 课（超参数调优）
**时间：** ~120 分钟

## 学习目标

- 从头构建一个 ML 管道，将插补、缩放、编码和模型训练串联成一个单一的可复现对象
- 识别数据泄漏场景，并解释管道如何通过仅在训练数据上拟合变换来防止它
- 构建一个 ColumnTransformer，对数值和分类特征应用不同的预处理
- 实现管道序列化，并演示相同的拟合管道在训练和生产中产生相同的结果

## 问题

你有一个笔记本，加载数据、用中位数填充缺失值、缩放特征、训练模型并打印准确率。它能工作。你发布它。

一个月后，有人重新训练模型并得到不同的结果。中位数是在包括测试数据的整个数据集上计算的（数据泄漏）。缩放参数没有被保存，因此推理使用不同的统计量。特征工程代码在训练和服务之间被复制粘贴，副本分道扬镳。一个分类列在生产中获得了一个编码器从未见过的新值。

这些不是假设的。它们是 ML 系统在生产中失败的最常见原因。管道通过将每个变换步骤打包成一个单一的、有序的、可复现的对象来解决所有这些问题。

## 概念

### 管道是什么

管道是一个有序的数据变换序列，后跟一个模型。每一步以前一步的输出作为输入。整个管道在训练数据上拟合一次。在推理时，相同的拟合管道变换新数据并产生预测。

管道保证：
- 变换仅在训练数据上拟合（无泄漏）
- 在推理时应用相同的变换
- 整个对象可以序列化并作为一个完整制品部署
- 交叉验证按折应用管道，防止微妙的泄漏

### 数据泄漏：沉默的杀手

当来自测试集或未来数据的信息污染训练时，发生数据泄漏。管道防止最常见的形式。

**有泄漏（错误）**：在分割前缩放整个数据集。缩放器看到了测试数据。
**正确**：先分割，然后在训练数据上拟合缩放器，仅对测试数据进行 transform。

使用管道，你不需要考虑这些。管道自动处理。

### sklearn Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression()),
])

pipe.fit(X_train, y_train)
predictions = pipe.predict(X_test)
```

### ColumnTransformer：不同列不同的管道

```python
from sklearn.compose import ColumnTransformer

numeric_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])

categorical_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, ["age", "income", "score"]),
    ("cat", categorical_pipe, ["city", "gender", "plan"]),
])

full_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", GradientBoostingClassifier()),
])
```

`handle_unknown="ignore"` 在生产中至关重要。当出现新类别时，它产生一个零向量而不是崩溃。

## 构建它

从头实现简单的 `Pipeline` 和 `ColumnTransformer`。代码见 `code/ml_pipeline.py`。

## 使用它

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

# 保存
joblib.dump(full_pipeline, "pipeline.joblib")

# 在生产中加载
pipe = joblib.load("pipeline.joblib")
predictions = pipe.predict(new_data)
```

## 练习

1. 构建一个管道，执行：插补缺失值、标准化、独热编码、逻辑回归。应用 5 折交叉验证。确认数据泄漏不会发生。
2. 在有和没有 `handle_unknown="ignore"` 的情况下，将具有生产中所见类别的测试数据输入 OneHotEncoder。解释差异。

## 关键术语

| 术语 | 实际含义 |
|------|----------------------|
| 管道 | 变换和最终估计器的有序序列 |
| ColumnTransformer | 对不同列组应用不同变换 |
| 数据泄漏 | 来自验证/测试集的信息污染训练 |
| handle_unknown | OneHotEncoder 参数，控制如何处理未见类别 |
| 序列化 | 将拟合的管道保存到磁盘并稍后加载 |
