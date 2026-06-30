# 3D Gaussian Splatting from Scratch

> 一个场景是数百万个 3D 高斯组成的云。每个都有位置、朝向、尺度和依赖于观察方向的不透明度和颜色。光栅化它们，通过光栅化反向传播，完成。

**类型：** Build
**语言：** Python
**先修要求：** Phase 4 Lesson 13 (3D Vision & NeRF), Phase 1 Lesson 12 (Tensor Operations), Phase 4 Lesson 10 (Diffusion basics optional)
**时间：** ~90 分钟

## 学习目标

- 解释为什么 3D Gaussian Splatting 在 2026 年取代了 NeRF 成为照片级逼真 3D 重建的生产默认方案
- 陈述六个逐高斯参数（位置、旋转四元数、尺度、不透明度、球谐颜色、可选特征）以及每个贡献多少浮点数
- 从头实现使用 `alpha` 合成的 2D Gaussian Splatting 光栅化器，然后展示 3D 情况如何投影到相同循环
- 使用 `nerfstudio`、`gsplat` 或 `SuperSplat` 从 20-50 张照片重建场景，并导出为 `KHR_gaussian_splatting` glTF 扩展或 OpenUSD 26.03 `UsdVolParticleField3DGaussianSplat` schema

## 问题

NeRF 将场景存储为 MLP 的权重。每个渲染像素是沿光线进行数百次 MLP 查询。训练需要数小时，渲染需要数秒，并且权重无法编辑——如果你想在场景中移动一把椅子，你必须重新训练。

3D Gaussian Splatting（Kerbl, Kopanas, Leimkühler, Drettakis, SIGGRAPH 2023）替换了这一切。场景是一组显式的 3D 高斯。渲染是 GPU 光栅化，达到 100+ fps。训练需要数分钟。编辑是直接的：平移一组高斯，你就移动了椅子。到 2026 年，Khronos Group 已批准了 Gaussian splat 的 glTF 扩展，OpenUSD 26.03 提供了 Gaussian splat schema，Zillow 和 Apartments.com 用它们渲染房产，大多数新的 3D 重建研究论文都是核心 3DGS 思想的变体。

思维模型很简单，但数学有足够多的移动部件，大多数导论从光栅化开始，跳过了投影和球谐。本课构建整个东西——先做 2D 版本，然后扩展到 3D。

## 概念

### 高斯携带什么

一个 3D 高斯是空间中一个参数化的斑点，具有以下属性：

```
位置             mu         (3,)    世界坐标中的中心
旋转             q          (4,)    编码朝向的单位四元数
尺度             s          (3,)    每轴 log-尺度（渲染时指数化）
不透明度         alpha      (1,)    sigmoid 之后的不透明度 [0, 1]
SH 系数          c_lm       (3 * (L+1)^2,)   视角相关颜色
```

旋转 + 尺度构建 3x3 协方差：`Sigma = R S S^T R^T`。这是高斯在 3D 中的形状。球谐函数让颜色随观察方向变化——镜面高光、微妙光泽、视角相关的辉光——无需存储逐视图纹理。SH 阶数 3 为每个颜色通道提供 16 个系数，每个高斯仅颜色就需要 48 个浮点数。

一个场景通常有 100 万到 500 万个高斯。每个存储大约 60 个浮点数（3 + 4 + 3 + 1 + 48 + 杂项）。这相当于一个五百万高斯场景的 240 MB——远小于具有逐点纹理的等效点云，并且比高分辨率下重新渲染的 NeRF MLP 权重小一个数量级。

### 光栅化，而非光线步进

```mermaid
flowchart LR
    SCENE["数百万个 3D 高斯<br/>(position, rotation, scale,<br/>opacity, SH colour)"] --> PROJ["投影到 2D<br/>(相机外参 + 内参)"]
    PROJ --> TILES["分配到 tiles<br/>(16x16 屏幕空间)"]
    TILES --> SORT["每 tile<br/>深度排序"]
    SORT --> ALPHA["从前到后<br/>Alpha 合成"]
    ALPHA --> PIX["像素颜色"]

    style SCENE fill:#dbeafe,stroke:#2563eb
    style ALPHA fill:#fef3c7,stroke:#d97706
    style PIX fill:#dcfce7,stroke:#16a34a
```

五步，全部 GPU 友好。没有每个像素的 MLP 查询。单个 RTX 3080 Ti 以 147 fps 渲染 600 万个 splats。

### 投影步骤

在世界位置 `mu` 具有 3D 协方差 `Sigma` 的 3D 高斯投影到屏幕位置 `mu'` 具有 2D 协方差 `Sigma'` 的 2D 高斯：

```
mu' = project(mu)
Sigma' = J W Sigma W^T J^T          (2 x 2)

W = 视图变换（相机的旋转 + 平移）
J = 视角投影在 mu' 处的 Jacobian
```

2D 高斯的足迹是一个椭圆，其轴是 `Sigma'` 的特征向量。该椭圆内的每个像素接收高斯的贡献，用 `exp(-0.5 * (p - mu')^T Sigma'^-1 (p - mu'))` 加权。

### Alpha 合成规则

对于一个像素，覆盖它的高斯从后到前排序（或等价的从前到后带反向公式）。颜色用与自从 1980 年代以来每个半透明光栅化器相同的公式合成：

```
C_pixel = sum_i alpha_i * T_i * c_i

T_i = prod_{j < i} (1 - alpha_j)       到 i 的透射率
alpha_i = opacity_i * exp(-0.5 * d^T Sigma'^-1 d)   局部贡献
c_i = eval_SH(SH_i, view_direction)    视角相关颜色
```

这是**与 NeRF 体积渲染相同的公式**，只是在显式的稀疏高斯集合上而不是沿光线的稠密样品上。这种等式是渲染质量匹敌 NeRF 的原因——两者都在积分相同的辐射场方程。

### 为什么可微分

每一步——投影、tile 分配、alpha 合成、SH 评估——都对高斯参数可微分。给定一张真实图像，计算渲染像素损失，通过光栅化器反向传播，用梯度下降更新所有 `(mu, q, s, alpha, c_lm)`。超过约 30,000 次迭代，高斯找到它们正确的位置、尺度和颜色。

### 密集化和剪枝

固定的一组高斯无法覆盖复杂场景。训练包括两种自适应机制：

- **克隆**一个位置上的高斯当其梯度幅度高但尺度小时——重建需要此处更多细节。
- **分割**一个大尺度高斯成两个较小的当其梯度高时——一个大高斯太平滑无法适应区域。
- **剪枝**不透明度降到阈值以下的高斯——它们不贡献。

密集化每 N 次迭代运行。场景通常从约 100k 初始高斯（从 SfM 点播种）增长到训练结束时的 1-5M。

### 球谐函数一句话

视角相关颜色是单位球面上的函数 `c(direction)`。球谐函数是球面的傅里叶基。截断在阶数 `L` 每个通道获得 `(L+1)^2` 个基函数。评估新视角的颜色是已学习的 SH 系数与在观察方向上评估的基之间的点积。阶数 0 = 一个系数 = 恒定颜色。阶数 3 = 16 个系数 = 足以捕捉 Lambert 着色、镜面反射和轻微反射。标准的 Gaussian Splatting 论文默认使用阶数 3。

### 2026 年生产技术栈

```
1. 采集             智能手机 / DJI 无人机 / 手持扫描仪
2. SfM / MVS        COLMAP 或 GLOMAP 推导相机位姿 + 稀疏点
3. 训练 3DGS        nerfstudio / gsplat / inria 官方 / PostShot（RTX 4090 上约 10-30 分钟）
4. 编辑              SuperSplat / SplatForge（清理浮点、分割）
5. 导出              .ply -> glTF KHR_gaussian_splatting 或 .usd（OpenUSD 26.03）
6. 查看              Cesium / Unreal / Babylon.js / Three.js / Vision Pro
```

### 4D 和生成变体

- **4D Gaussian Splatting**——高斯是时间的函数；用于体积视频（Superman 2026, A$AP Rocky's "Helicopter"）。
- **生成 splats**——文本到 splat 模型（World Labs 的 Marble），幻觉出整个场景。
- **3D Gaussian Unscented Transform**——NVIDIA NuRec 用于自动驾驶模拟的变体。

## Build It

### 步骤 1：2D 高斯

我们首先构建 2D 光栅化器。3D 情况在投影后简化为它。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def eval_2d_gaussian(means, covs, points):
    """
    means:  (G, 2)      中心
    covs:   (G, 2, 2)   协方差矩阵
    points: (H, W, 2)   像素坐标
    返回:   (G, H, W)   每个高斯在每个像素的密度
    """
    G = means.size(0)
    H, W, _ = points.shape
    flat = points.view(-1, 2)
    inv = torch.linalg.inv(covs)
    diff = flat[None, :, :] - means[:, None, :]
    d = torch.einsum("gpi,gij,gpj->gp", diff, inv, diff)
    density = torch.exp(-0.5 * d)
    return density.view(G, H, W)
```

`einsum` 对每个（高斯, 像素）对执行二次型 `diff^T Sigma^-1 diff`。

### 步骤 2：2D splatting 光栅化器

从前到后 Alpha 合成。2D 中的深度无意义，所以我们使用每个高斯的已学习标量来排序。

```python
def rasterise_2d(means, covs, colours, opacities, depths, image_size):
    """
    means:     (G, 2)
    covs:      (G, 2, 2)
    colours:   (G, 3)
    opacities: (G,)     在 [0, 1] 中
    depths:    (G,)     用于排序的每个高斯标量
    image_size: (H, W)
    返回:      (H, W, 3) 渲染图像
    """
    H, W = image_size
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=means.device),
        torch.arange(W, dtype=torch.float32, device=means.device),
        indexing="ij",
    )
    points = torch.stack([xx, yy], dim=-1)

    densities = eval_2d_gaussian(means, covs, points)
    alphas = opacities[:, None, None] * densities
    alphas = alphas.clamp(0.0, 0.99)

    order = torch.argsort(depths)
    alphas = alphas[order]
    colours_sorted = colours[order]

    T = torch.ones(H, W, device=means.device)
    out = torch.zeros(H, W, 3, device=means.device)
    for i in range(means.size(0)):
        a = alphas[i]
        out += (T * a)[..., None] * colours_sorted[i][None, None, :]
        T = T * (1.0 - a)
    return out
```

不快——真正的实现使用基于 tile 的 CUDA 内核——但数学完全正确且完全可微分。

### 步骤 3：可训练的 2D splat 场景

```python
class Splats2D(nn.Module):
    def __init__(self, num_splats=128, image_size=64, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        H, W = image_size, image_size
        self.means = nn.Parameter(torch.rand(num_splats, 2, generator=g) * torch.tensor([W, H]))
        self.log_scale = nn.Parameter(torch.ones(num_splats, 2) * math.log(2.0))
        self.rot = nn.Parameter(torch.zeros(num_splats))  # 2D 中单角度
        self.colour_logits = nn.Parameter(torch.randn(num_splats, 3, generator=g) * 0.5)
        self.opacity_logit = nn.Parameter(torch.zeros(num_splats))
        self.depth = nn.Parameter(torch.rand(num_splats, generator=g))

    def covs(self):
        s = torch.exp(self.log_scale)
        c, si = torch.cos(self.rot), torch.sin(self.rot)
        R = torch.stack([
            torch.stack([c, -si], dim=-1),
            torch.stack([si, c], dim=-1),
        ], dim=-2)
        S = torch.diag_embed(s ** 2)
        return R @ S @ R.transpose(-1, -2)

    def forward(self, image_size):
        covs = self.covs()
        colours = torch.sigmoid(self.colour_logits)
        opacities = torch.sigmoid(self.opacity_logit)
        return rasterise_2d(self.means, covs, colours, opacities, self.depth, image_size)
```

`log_scale`、`opacity_logit` 和 `colour_logits` 都是无约束参数，渲染时通过正确的激活函数映射。这是每个 3DGS 实现的标准模式。

### 步骤 4：将 2D 高斯拟合到目标图像

```python
import math
import numpy as np

def make_target(size=64):
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    img = np.zeros((size, size, 3), dtype=np.float32)
    # 红色圆
    mask = (xx - 20) ** 2 + (yy - 20) ** 2 < 10 ** 2
    img[mask] = [1.0, 0.2, 0.2]
    # 蓝色正方形
    mask = (np.abs(xx - 45) < 8) & (np.abs(yy - 40) < 8)
    img[mask] = [0.2, 0.3, 1.0]
    return torch.from_numpy(img)


target = make_target(64)
model = Splats2D(num_splats=64, image_size=64)
opt = torch.optim.Adam(model.parameters(), lr=0.05)

for step in range(200):
    pred = model((64, 64))
    loss = F.mse_loss(pred, target)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 40 == 0:
        print(f"步骤 {step:3d}  mse {loss.item():.4f}")
```

200 步后，64 个高斯沉降到两个形状中。这就是整个想法——对显式几何原语进行梯度下降。

### 步骤 5：从 2D 到 3D

3D 扩展保持相同的循环。添加的内容：

1. 每个高斯旋转用四元数而非单角度。
2. 协方差是 `R S S^T R^T`，`R` 从四元数构建，`S = diag(exp(log_scale))`。
3. 投影 `(mu, Sigma) -> (mu', Sigma')` 使用相机外参和 `mu` 处透视投影的 Jacobian。
4. 颜色变成球谐展开；在观察方向上评估。
5. 深度排序用实际相机空间 z 而非已学习标量。

每个生产实现（`gsplat`、`inria/gaussian-splatting`、`nerfstudio`）在 GPU 上用基于 tile 的 CUDA 内核精确执行此操作。

### 步骤 6：球谐评估

阶数最多为 3 的 SH 基每个通道有 16 项。评估：

```python
def eval_sh_degree_3(sh_coeffs, dirs):
    """
    sh_coeffs: (..., 16, 3)   最后一维是 RGB 通道
    dirs:      (..., 3)       单位向量
    返回:      (..., 3)
    """
    C0 = 0.282094791773878
    C1 = 0.488602511902920
    C2 = [1.092548430592079, 1.092548430592079,
          0.315391565252520, 1.092548430592079,
          0.546274215296039]
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    x2, y2, z2 = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z

    result = C0 * sh_coeffs[..., 0, :]
    result = result - C1 * y[..., None] * sh_coeffs[..., 1, :]
    result = result + C1 * z[..., None] * sh_coeffs[..., 2, :]
    result = result - C1 * x[..., None] * sh_coeffs[..., 3, :]

    result = result + C2[0] * xy[..., None] * sh_coeffs[..., 4, :]
    result = result + C2[1] * yz[..., None] * sh_coeffs[..., 5, :]
    result = result + C2[2] * (2.0 * z2 - x2 - y2)[..., None] * sh_coeffs[..., 6, :]
    result = result + C2[3] * xz[..., None] * sh_coeffs[..., 7, :]
    result = result + C2[4] * (x2 - y2)[..., None] * sh_coeffs[..., 8, :]

    # 阶数 3 项此处省略以求简洁；完整 16 系数版本见代码文件
    return result
```

已学习的 `sh_coeffs` 为该高斯存储"每个方向上的颜色"。渲染时你根据当前视图方向评估并得到一个 3 向量 RGB。

## Use It

对于真正的 3DGS 工作，使用 `gsplat`（Meta）或 `nerfstudio`：

```bash
pip install nerfstudio gsplat
ns-download-data example
ns-train splatfacto --data path/to/data
```

`splatfacto` 是 nerfstudio 的 3DGS 训练器。典型场景在 RTX 4090 上运行 10-30 分钟。

2026 年重要的导出选项：

- `.ply`——原始高斯云（可移植，最大文件）。
- `.splat`——PlayCanvas / SuperSplat 量化格式。
- glTF `KHR_gaussian_splatting`——Khronos 标准，跨查看器可移植（2026 年 2 月 RC）。
- OpenUSD `UsdVolParticleField3DGaussianSplat`——USD 原生，用于 NVIDIA Omniverse 和 Vision Pro pipeline。

对于 4D / 动态场景，`4DGS` 和 `Deformable-3DGS` 用时变均值和透明度扩展相同机制。

## Ship It

本课产出：

- `outputs/prompt-3dgs-capture-planner.md`——一个为给定场景类型规划采集会话（照片数量、相机路径、光照）的 prompt。
- `outputs/skill-3dgs-export-router.md`——一个根据下游查看器或引擎选择正确导出格式（`.ply` / `.splat` / glTF / USD）的 skill。

## 练习

1. **（简单）** 在不同合成图像上运行上述 2D splat 训练器。在 `[16, 64, 256]` 中变化 `num_splats` 并绘制每个的 MSE vs 步骤。识别收益递减的点。
2. **（中等）** 扩展 2D 光栅化器以支持依赖于标量"视角角度"的每个高斯 RGB 颜色，通过阶数 2 谐波。在一对目标图像上训练并验证模型重建两者。
3. **（困难）** 克隆 `nerfstudio` 并在你拥有的任何场景（桌面、植物、人脸、房间）的 20 张照片采集上训练 `splatfacto`。导出到 glTF `KHR_gaussian_splatting` 并在查看器（Three.js `GaussianSplats3D`、SuperSplat、Babylon.js V9）中打开。报告训练时间、高斯数量和渲染 fps。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|----------------------|
| 3DGS | "高斯 splats" | 将场景显式表示为数百万个 3D 高斯，每个具有 position, rotation, scale, opacity, SH colour |
| 协方差 | "高斯的形状" | `Sigma = R S S^T R^T`；一个高斯的朝向和各向异性尺度 |
| Alpha 合成 | "从后到前混合" | 与 NeRF 体积渲染相同的公式，现在在显式稀疏集合上 |
| 密集化 | "克隆和分割" | 在重建欠拟合的地方自适应添加新高斯 |
| 剪枝 | "删除低透明度" | 移除训练期间透明度崩溃到接近零的高斯 |
| 球谐函数 | "视角相关颜色" | 球面上的傅里叶基；将颜色存储为观察方向的函数 |
| Splatfacto | "nerfstudio 的 3DGS" | 2026 年训练 3DGS 的最简单路径 |
| `KHR_gaussian_splatting` | "glTF 标准" | Khronos 2026 扩展，使 3DGS 跨查看器和引擎可移植 |

## 延伸阅读

- [3D Gaussian Splatting for Real-Time Radiance Field Rendering (Kerbl et al., SIGGRAPH 2023)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)——原始论文
- [gsplat (Meta/nerfstudio)](https://github.com/nerfstudio-project/gsplat)——生产品质 CUDA 光栅化器
- [nerfstudio Splatfacto](https://docs.nerf.studio/nerfology/methods/splat.html)——参考训练配方
- [Khronos KHR_gaussian_splatting 扩展](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_gaussian_splatting/README.md)——2026 年可移植格式
- [OpenUSD 26.03 release notes](https://openusd.org/release/)——`UsdVolParticleField3DGaussianSplat` schema
- [THE FUTURE 3D State of Gaussian Splatting 2026](https://www.thefuture3d.com/blog-0/2026/4/4/state-of-gaussian-splatting-2026)——行业概述
