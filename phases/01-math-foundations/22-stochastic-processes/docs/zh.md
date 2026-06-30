# 随机过程

> 有结构的随机性。随机游走、马尔可夫链和扩散模型背后的数学。

**类型：** 学习
**语言：** Python
**先决条件：** 阶段 1，第 06-07 课（概率论、贝叶斯）
**时间：** ~75 分钟

## 学习目标

- 模拟 1D 和 2D 随机游走并验证位移的 sqrt(n) 缩放
- 构建马尔可夫链模拟器并通过特征分解计算其平稳分布
- 实现 Metropolis-Hastings MCMC 和 Langevin 动力学从目标分布中采样
- 将前向扩散过程与布朗运动连接起来，并解释反向过程如何生成数据

## 问题

许多 AI 系统涉及随时间演化的随机性。不是静态的随机性——结构化的、序列化的随机性，其中每一步依赖于之前的。

语言模型每次生成一个 token。每个 token 依赖于先前的上下文。模型输出一个概率分布，从中采样，然后继续。这是一个随机过程。

扩散模型逐步向图像添加噪声，直到它变成纯静态。然后它们反转过程，逐步去噪，直到新图像出现。前向过程是一个马尔可夫链。反向过程是一个运行反向的学习的马尔可夫链。

强化学习智能体在环境中采取行动。每个行动以某种概率导致新状态。智能体在随机世界中遵循随机策略。整个过程是一个马尔可夫决策过程。

MCMC 采样——贝叶斯推断的骨干——构建一个其平稳分布是你要采样的后验的马尔可夫链。

所有这些都建立在四个基础思想上：
1. 随机游走——最简单的随机过程
2. 马尔可夫链——具有转移矩阵的结构化随机性
3. Langevin 动力学——带噪声的梯度下降
4. Metropolis-Hastings——从任意分布中采样

## 概念

### 随机游走

从位置 0 开始。在每一步，抛一枚公平硬币。正面：向右移动（+1）。反面：向左移动（-1）。

n 步后，你的位置是 n 个随机 +/-1 值的和。期望位置是 0（游走是无偏的）。但距原点的期望距离按 sqrt(n) 增长。

这与直觉相反。游走是公平的——没有向任一方向漂移。但随时间推移，它离起点越来越远。n 步后的标准差是 sqrt(n)。

```
步骤 0：位置 = 0
步骤 1：位置 = +1 或 -1
步骤 2：位置 = +2、0 或 -2
...
步骤 100：距原点期望距离 ~ 10 (sqrt(100))
步骤 10000：距原点期望距离 ~ 100 (sqrt(10000))
```

**在 2D 中**，游走上、下、左、右移动，概率相等。相同的 sqrt(n) 缩放适用于距原点的距离。路径描绘出分形般的图案。

**为什么是 sqrt(n)？** 每一步是 +1 或 -1，概率相等。n 步后，位置 S_n = X_1 + X_2 + ... + X_n 其中每个 X_i 是 +/-1。每一步的方差是 1，步骤是独立的，所以 Var(S_n) = n。标准差 = sqrt(n)。根据中心极限定理，S_n / sqrt(n) 收敛到标准正态分布。

这个 sqrt(n) 缩放在 ML 中到处都是。SGD 噪声按 1/sqrt(batch_size) 缩放。嵌入维度按 sqrt(d) 缩放。平方根是独立随机加法的签名。

**与布朗运动的联系。** 取步长 1/sqrt(n) 且每单位时间 n 步的随机游走。当 n 趋于无穷时，游走收敛到布朗运动 B(t)——一个连续时间过程，其中 B(t) 是均值为 0、方差为 t 的正态分布。

布朗运动是扩散的数学基础。它模拟流体中粒子的随机颤动、股票价格的波动，以及——至关重要——扩散模型中的噪声过程。

**赌徒破产。** 一个随机游走者从位置 k 开始，在 0 和 N 处有吸收壁。在 0 之前到达 N 的概率是多少？对于公平游走：P(到达 N) = k/N。这出奇地简单而优雅。它与鞅理论相连——公平随机游走是一个鞅（期望未来值 = 当前值）。

### 马尔可夫链

马尔可夫链是一个根据固定概率在状态之间转移的系统。关键性质：下一状态仅取决于当前状态，不取决于历史。

```
P(X_{t+1} = j | X_t = i, X_{t-1} = ...) = P(X_{t+1} = j | X_t = i)
```

这是马尔可夫性质。它意味着你可以用一个转移矩阵 P 描述整个动力学：

```
P[i][j] = 从状态 i 到状态 j 的概率
```

P 的每一行和为 1（你必须去某个地方）。

**例子——天气：**

```
状态：晴天 (0)、雨天 (1)、多云 (2)

P = [[0.7, 0.1, 0.2],    （如果是晴天：70% 晴天，10% 雨天，20% 多云）
     [0.3, 0.4, 0.3],    （如果是雨天：30% 晴天，40% 雨天，30% 多云）
     [0.4, 0.2, 0.4]]    （如果是多云：40% 晴天，20% 雨天，40% 多云）
```

从任意状态开始。多次转移后，状态的分布收敛到平稳分布 pi，其中 pi * P = pi。这是 P 的特征值为 1 的左特征向量。

对于天气链，平稳分布可能是 [0.53, 0.18, 0.29]——长期来看，无论起始状态如何，53% 的时间是晴天。

```mermaid
graph LR
    S["晴天"] -->|0.7| S
    S -->|0.1| R["雨天"]
    S -->|0.2| C["多云"]
    R -->|0.3| S
    R -->|0.4| R
    R -->|0.3| C
    C -->|0.4| S
    C -->|0.2| R
    C -->|0.4| C
```

**计算平稳分布。** 有两种方法：

1. **幂方法**：将任意初始分布反复乘以 P。足够次迭代后，它收敛。
2. **特征值方法**：找到 P 的特征值为 1 的左特征向量。这是 P^T 的特征值为 1 的特征向量。

两种方法都要求链满足收敛条件。

**收敛条件。** 马尔可夫链收敛到唯一平稳分布，如果它是：
- **不可约的**：每个状态都可以从每个其他状态到达
- **非周期的**：链不会以固定周期循环

你在 ML 中遇到的大多数链满足这两个条件。

**吸收状态。** 如果一旦进入就永不离开（P[i][i] = 1），该状态是吸收的。吸收马尔可夫链模拟具有终端状态的过程——结束的游戏、流失的客户、遇到文本结束 token 的 token 序列。

**混合时间。** 链运行多少步后"接近"平稳分布？形式上，到平稳距离的总变分距离降到低于某个阈值的步数。快混合 = 需要少量步数。P 的谱间隙（1 减第二大的特征值）控制混合时间。更大的间隙 = 更快的混合。

### 与语言模型的联系

语言模型中的 token 生成近似是一个马尔可夫过程。给定当前上下文，模型输出下一 token 的分布。温度控制锐度：

```
P(token_i) = exp(logit_i / temperature) / sum(exp(logit_j / temperature))
```

- 温度 = 1.0：标准分布
- 温度 < 1.0：更锐（更确定）
- 温度 > 1.0：更平坦（更随机）
- 温度 -> 0：argmax（贪婪）

Top-k 采样截断到 k 个最高概率的 token。Top-p（核）采样截断到累积概率超过 p 的最小 token 集合。两者修改马尔可夫转移概率。

### 布朗运动

随机游走的连续时间极限。位置 B(t) 有三个性质：
1. B(0) = 0
2. B(t) - B(s) 是均值为 0、方差为 t - s 的正态分布（对 t > s）
3. 不重叠区间上的增量独立

布朗运动连续但处处不可微——它在每个尺度上颤动。路径在平面中具有分形维度 2。

在离散模拟中，你通过以下方式近似布朗运动：

```
B(t + dt) = B(t) + sqrt(dt) * z，    其中 z ~ N(0, 1)
```

sqrt(dt) 缩放很重要。它来自应用于随机游走的中心极限定理。

### Langevin 动力学

梯度下降找到函数的最小值。Langevin 动力学找到与 exp(-U(x)/T) 成比例的概率分布，其中 U 是能量函数，T 是温度。

```
x_{t+1} = x_t - dt * gradient(U(x_t)) + sqrt(2 * T * dt) * z_t
```

两个力作用在粒子上：
1. **梯度力** (-dt * gradient(U))：推向低能量（像梯度下降）
2. **随机力** (sqrt(2*T*dt) * z)：推向随机方向（探索）

在温度 T = 0，这是纯梯度下降。在高温下，它近乎随机游走。在合适的温度下，粒子探索能量曲面并在低能量区域花费更多时间。

**与扩散模型的联系。** 扩散模型的前向过程是：

```
x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * 噪声
```

这是一个逐步将数据与噪声混合的马尔可夫链。足够步数后，x_T 是纯高斯噪声。

反向过程——从噪声回到数据——也是一个马尔可夫链，但其转移概率由神经网络学习。网络学习预测每一步添加的噪声，然后减去它。

```mermaid
graph LR
    subgraph "前向过程（添加噪声）"
        X0["x_0 (数据)"] -->|"+ 噪声"| X1["x_1"]
        X1 -->|"+ 噪声"| X2["x_2"]
        X2 -->|"..."| XT["x_T (纯噪声)"]
    end
    subgraph "反向过程（去噪）"
        XT2["x_T (噪声)"] -->|"神经网络"| XR2["x_{T-1}"]
        XR2 -->|"神经网络"| XR1["x_{T-2}"]
        XR1 -->|"..."| XR0["x_0 (生成的数据)"]
    end
```

### MCMC：马尔可夫链蒙特卡洛

有时你需要从一个你能评估（最多差一个常数）但不能直接采样的分布 p(x) 中采样。贝叶斯后验是经典例子——你知道似然乘以先验，但归一化常数难以处理。

**Metropolis-Hastings** 构建一个其平稳分布是 p(x) 的马尔可夫链：

1. 从某个位置 x 开始
2. 从提议分布 Q(x'|x) 提议一个新位置 x'
3. 计算接受比：a = p(x') * Q(x|x') / (p(x) * Q(x'|x))
4. 以概率 min(1, a) 接受 x'。否则留在 x。
5. 重复。

如果 Q 是对称的（例如，Q(x'|x) = Q(x|x') = N(x, sigma^2)），比率简化为 a = p(x') / p(x)。你只需要概率的比率——归一化常数抵消。

链保证在温和条件下收敛到 p(x)。但如果提议太小（随机游走）或太大（高拒绝率），收敛可能很慢。调整提议是 MCMC 的艺术。

**为什么有效。** 接受率确保细致平衡：在 x 并移动到 x' 的概率等于在 x' 并移动到 x 的概率。细致平衡意味着 p(x) 是链的平稳分布。因此足够的步数后，样本来自 p(x)。

**实际考虑：**
- **预热期**：丢弃前 N 个样本。链需要时间从其起点到达平稳分布。
- **稀疏化**：保留每第 k 个样本以减少自相关。
- **多链**：从不同起点运行几条链。如果它们收敛到相同分布，你有收敛的证据。
- **接受率**：对于 d 维中的高斯提议，最优接受率约为 23%（Roberts & Rosenthal, 2001）。太高意味着链几乎不移动。太低意味着它拒绝一切。

### 随机过程在 AI 中的应用

| 过程 | AI 应用 |
|---------|---------------|
| 随机游走 | RL 中的探索、Node2Vec 嵌入 |
| 马尔可夫链 | 文本生成、MCMC 采样 |
| 布朗运动 | 扩散模型（前向过程） |
| Langevin 动力学 | 基于分数的生成模型、SGLD |
| 马尔可夫决策过程 | 强化学习 |
| Metropolis-Hastings | 贝叶斯推断、后验采样 |

```figure
random-walk-diffusion
```

## 构建它

### 步骤 1：随机游走模拟器

```python
import numpy as np

def random_walk_1d(n_steps, seed=None):
    rng = np.random.RandomState(seed)
    steps = rng.choice([-1, 1], size=n_steps)
    positions = np.concatenate([[0], np.cumsum(steps)])
    return positions


def random_walk_2d(n_steps, seed=None):
    rng = np.random.RandomState(seed)
    directions = rng.choice(4, size=n_steps)
    dx = np.zeros(n_steps)
    dy = np.zeros(n_steps)
    dx[directions == 0] = 1   # 右
    dx[directions == 1] = -1  # 左
    dy[directions == 2] = 1   # 上
    dy[directions == 3] = -1  # 下
    x = np.concatenate([[0], np.cumsum(dx)])
    y = np.concatenate([[0], np.cumsum(dy)])
    return x, y
```

1D 游走存储累积和。每一步是 +1 或 -1。n 步后，位置是和。方差随 n 线性增长，因此标准差按 sqrt(n) 增长。

### 步骤 2：马尔可夫链

```python
class MarkovChain:
    def __init__(self, transition_matrix, state_names=None):
        self.P = np.array(transition_matrix, dtype=float)
        self.n_states = len(self.P)
        self.state_names = state_names or [str(i) for i in range(self.n_states)]

    def step(self, current_state, rng=None):
        if rng is None:
            rng = np.random.RandomState()
        probs = self.P[current_state]
        return rng.choice(self.n_states, p=probs)

    def simulate(self, start_state, n_steps, seed=None):
        rng = np.random.RandomState(seed)
        states = [start_state]
        current = start_state
        for _ in range(n_steps):
            current = self.step(current, rng)
            states.append(current)
        return states

    def stationary_distribution(self):
        eigenvalues, eigenvectors = np.linalg.eig(self.P.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, idx])
        stationary = stationary / stationary.sum()
        return np.abs(stationary)
```

平稳分布是 P 的特征值为 1 的左特征向量。我们通过计算 P^T 的特征向量来找到它（转置将左特征向量变为右特征向量）。

### 步骤 3：Langevin 动力学

```python
def langevin_dynamics(grad_U, x0, dt, temperature, n_steps, seed=None):
    rng = np.random.RandomState(seed)
    x = np.array(x0, dtype=float)
    trajectory = [x.copy()]
    for _ in range(n_steps):
        noise = rng.randn(*x.shape)
        x = x - dt * grad_U(x) + np.sqrt(2 * temperature * dt) * noise
        trajectory.append(x.copy())
    return np.array(trajectory)
```

梯度将 x 推向低能量。噪声防止它卡住。在平衡态，样本的分布与 exp(-U(x)/temperature) 成比例。

### 步骤 4：Metropolis-Hastings

```python
def metropolis_hastings(target_log_prob, proposal_std, x0, n_samples, seed=None):
    rng = np.random.RandomState(seed)
    x = np.array(x0, dtype=float)
    samples = [x.copy()]
    accepted = 0
    for _ in range(n_samples - 1):
        x_proposed = x + rng.randn(*x.shape) * proposal_std
        log_ratio = target_log_prob(x_proposed) - target_log_prob(x)
        if np.log(rng.rand()) < log_ratio:
            x = x_proposed
            accepted += 1
        samples.append(x.copy())
    acceptance_rate = accepted / (n_samples - 1)
    return np.array(samples), acceptance_rate
```

算法提议一个新点，检查它是否有更高概率（或以与比率成比例的概率接受），并重复。接受率应在 23-50% 之间以获得良好混合。

## 使用它

在实践中，你使用已建立的库来实现这些算法。但理解机制对调试和调优很重要。

```python
import numpy as np

rng = np.random.RandomState(42)
walk = np.cumsum(rng.choice([-1, 1], size=10000))
print(f"最终位置: {walk[-1]}")
print(f"期望距离: {np.sqrt(10000):.1f}")
print(f"实际距离: {abs(walk[-1])}")
```

### numpy 用于转移矩阵

```python
import numpy as np

P = np.array([[0.7, 0.1, 0.2],
              [0.3, 0.4, 0.3],
              [0.4, 0.2, 0.4]])

distribution = np.array([1.0, 0.0, 0.0])
for _ in range(100):
    distribution = distribution @ P

print(f"平稳分布: {np.round(distribution, 4)}")
```

反复将初始分布乘以 P。足够迭代后，无论你从哪开始，它都收敛到平稳分布。这是寻找主导左特征向量的幂方法。

### 与真实框架的联系

- **PyTorch 扩散：** Hugging Face `diffusers` 中的 `DDPMScheduler` 实现了前向和反向马尔可夫链
- **NumPyro / PyMC：** 使用 MCMC（NUTS 采样器，它改进了 Metropolis-Hastings）进行贝叶斯推断
- **Gymnasium (RL)：** 环境步骤函数定义了一个马尔可夫决策过程

### 验证马尔可夫链收敛

```python
import numpy as np

P = np.array([[0.9, 0.1], [0.3, 0.7]])

eigenvalues = np.linalg.eigvals(P)
spectral_gap = 1 - sorted(np.abs(eigenvalues))[-2]
print(f"特征值: {eigenvalues}")
print(f"谱间隙: {spectral_gap:.4f}")
print(f"近似混合时间: {1/spectral_gap:.1f} 步")
```

谱间隙告诉你链多快忘记其初始状态。0.2 的间隙意味着大约 5 步混合。0.01 的间隙意味着大约 100 步。在运行长时间模拟前始终检查这一点——慢速混合的链浪费计算。

## 输出成果

本课产生：
- `outputs/prompt-stochastic-process-advisor.md`——一个帮助识别哪种随机过程框架适用于给定问题的提示

## 联系

| 概念 | 出现在哪里 |
|---------|------------------|
| 随机游走 | Node2Vec 图嵌入、RL 中的探索 |
| 马尔可夫链 | LLM 中的 token 生成、MCMC 采样 |
| 布朗运动 | DDPM 中的前向扩散过程、基于 SDE 的模型 |
| Langevin 动力学 | 基于分数的生成模型、随机梯度 Langevin 动力学 (SGLD) |
| 平稳分布 | MCMC 收敛目标、PageRank |
| Metropolis-Hastings | 贝叶斯后验采样、模拟退火 |
| 温度 | LLM 采样、RL 中的 Boltzmann 探索、模拟退火 |
| 混合时间 | MCMC 的收敛速度、谱间隙分析 |
| 吸收状态 | 序列结束 token、RL 中的终端状态 |
| 细致平衡 | MCMC 采样器的正确性保证 |

扩散模型值得特别关注。DDPM (Ho et al., 2020) 定义了一个前向马尔可夫链：

```
q(x_t | x_{t-1}) = N(x_t; sqrt(1-beta_t) * x_{t-1}, beta_t * I)
```

其中 beta_t 是噪声调度。T 步后，x_T 近似为 N(0, I)。反向过程由一个预测噪声的神经网络参数化：

```
p_theta(x_{t-1} | x_t) = N(x_{t-1}; mu_theta(x_t, t), sigma_t^2 * I)
```

生成的每一步都是学习的马尔可夫链中的一步。理解马尔可夫链意味着理解扩散模型如何以及为什么生成数据。

SGLD（随机梯度 Langevin 动力学）将小批量梯度下降与 Langevin 噪声结合。你不是计算完整梯度，而是使用随机估计并添加校准噪声。随着学习率衰减，SGLD 从优化过渡到采样——你免费获得近似的贝叶斯后验样本。这是从神经网络获得不确定性估计的最简单方式之一。

跨越所有这些联系的关键洞察：随机过程不仅仅是理论工具。它们是现代 AI 系统内部的计算机制。当你调整 LLM 的温度时，你正在调整一个马尔可夫链。当你训练扩散模型时，你正在学习反转一个类布朗运动过程。当你运行贝叶斯推断时，你正在构建一个收敛到后验的链。

## 练习

1. **模拟 1000 次 10000 步的随机游走。** 绘制最终位置的分布。验证它近似是高斯分布，均值为 0，标准差为 sqrt(10000) = 100。

2. **使用马尔可夫链构建文本生成器。** 在小型语料库上训练：对每个词，统计到下一个词的转移。构建转移矩阵。通过从链中采样生成新句子。

3. **使用 Metropolis-Hastings 实现模拟退火。** 从高温开始（几乎接受一切）并逐渐冷却（仅接受改进）。用它找到具有许多局部最小值的函数的最小值。

4. **在不同温度下比较 Langevin 动力学。** 从双阱势 U(x) = (x^2 - 1)^2 中采样。在低温下，样本聚集在一个阱中。在高温下，它们分布在两者之间。找到链在阱之间混合的临界温度。

5. **实现前向扩散过程。** 从 1 维信号开始（例如，正弦波）。使用线性噪声调度在 100 步中逐步添加噪声。展示信号如何退化为纯噪声。然后实现一个简单的去噪器反转该过程（即使只是减去估计噪声的天真版本也行）。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------------|----------------------|
| 随机游走 | "抛硬币运动" | 每一步位置按随机增量变化的过程 |
| 马尔可夫性质 | "无记忆" | 未来仅取决于当前状态，不取决于历史 |
| 转移矩阵 | "概率表" | P[i][j] = 从状态 i 移动到状态 j 的概率 |
| 平稳分布 | "长期平均" | 分布 pi 其中 pi*P = pi——链的平衡态 |
| 布朗运动 | "随机颤动" | 随机游走的连续时间极限，B(t) ~ N(0, t) |
| Langevin 动力学 | "带噪声的梯度下降" | 结合确定梯度和随机扰动的更新规则 |
| MCMC | "走向目标" | 构建一个其平稳分布是你要的分布的马尔可夫链 |
| Metropolis-Hastings | "提议并接受/拒绝" | 使用接受率确保收敛的 MCMC 算法 |
| 温度 | "随机性旋钮" | 控制探索和利用之间权衡的参数 |
| 扩散过程 | "噪声进来，噪声出去" | 前向：逐步添加噪声。反向：逐步移除它。生成数据。 |

## 进一步阅读

- **Ho, Jain, Abbeel (2020)**——"Denoising Diffusion Probabilistic Models." 启动扩散模型革命的 DDPM 论文。对前向和反向马尔可夫链的清晰推导。
- **Song & Ermon (2019)**——"Generative Modeling by Estimating Gradients of the Data Distribution." 使用 Langevin 动力学进行采样的基于分数的方法。
- **Roberts & Rosenthal (2004)**——"General state space Markov chains and MCMC algorithms." MCMC 何时以及为什么有效的理论。
- **Norris (1997)**——"Markov Chains." 标准教科书。涵盖收敛、平稳分布和击中时间。
- **Welling & Teh (2011)**——"Bayesian Learning via Stochastic Gradient Langevin Dynamics." 将 SGD 与 Langevin 动力学结合用于可扩展的贝叶斯推断。
