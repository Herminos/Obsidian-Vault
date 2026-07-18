# 优化失败时的诊断与对策 (Optimization: Diagnosis and Solutions)

当[[梯度下降 Gradient Descent]]停在某个梯度为零的点，但损失值仍不理想时，说明优化遇到了困难。本笔记讨论如何诊断和处理优化失败的问题。

---

## 1. 临界点 (Critical Points)

### 1.1 什么是临界点

在优化过程中，当梯度为零时，参数停止更新：

$$
\nabla L(\boldsymbol{\theta}') = \mathbf{0}
$$

这样的点 $\boldsymbol{\theta}'$ 称为**临界点 (Critical Point)**。临界点可能是：

| 类型 | 含义 | 梯度 | 是否可逃离 |
|------|------|:--:|:--:|
| **局部最小值 (Local Minima)** | 局部区域内的最低点 | $0$ | 很难 |
| **鞍点 (Saddle Point)** | 某些方向是极小值，某些方向是极大值 | $0$ | ✅ 可以 |
| **全局最小值 (Global Minima)** | 整个定义域的最低点 | $0$ | —（目标） |

### 1.2 泰勒级数近似 (Taylor Series Approximation)

要判断临界点的类型，需要分析损失函数 $L(\boldsymbol{\theta})$ 在临界点附近的形状。利用**二阶泰勒展开 (Second-Order Taylor Expansion)**：

在临界点 $\boldsymbol{\theta}'$ 附近，对损失函数做泰勒展开：

$$
L(\boldsymbol{\theta}) \approx L(\boldsymbol{\theta}') + \underbrace{(\boldsymbol{\theta} - \boldsymbol{\theta}')^{T} \nabla L(\boldsymbol{\theta}')}_{\text{一阶项 = 0}} + \frac{1}{2} (\boldsymbol{\theta} - \boldsymbol{\theta}')^{T} \mathbf{H} (\boldsymbol{\theta} - \boldsymbol{\theta}')
$$

由于在临界点处梯度为零，一阶项消失，简化为：

$$
L(\boldsymbol{\theta}) \approx L(\boldsymbol{\theta}') + \frac{1}{2} (\boldsymbol{\theta} - \boldsymbol{\theta}')^{T} \mathbf{H} (\boldsymbol{\theta} - \boldsymbol{\theta}')
$$

其中 **$\mathbf{H}$ 是 Hessian 矩阵 (Hessian Matrix)**，即损失函数对所有参数的二阶偏导矩阵：

$$
\mathbf{H}_{ij} = \frac{\partial^{2} L}{\partial \theta_{i} \partial \theta_{j}}
$$

### 1.3 用 Hessian 的特征值判断临界点类型

临界点附近的形状由二次型 $\mathbf{v}^{T} \mathbf{H} \mathbf{v}$ 决定。令 $\mathbf{v} = \boldsymbol{\theta} - \boldsymbol{\theta}'$：

- **若对所有方向 $\mathbf{v}$ 都有 $\mathbf{v}^{T} \mathbf{H} \mathbf{v} > 0$**：
  → 所有方向都是「上坡」→ **局部最小值 (Local Minima)**

- **若对所有方向 $\mathbf{v}$ 都有 $\mathbf{v}^{T} \mathbf{H} \mathbf{v} < 0$**：
  → 所有方向都是「下坡」→ **局部最大值 (Local Maxima)**

- **若某些方向为正、某些方向为负**：
  → 有「上坡」也有「下坡」→ **鞍点 (Saddle Point)**

用 Hessian 矩阵的**特征值 (Eigenvalues)** 来表述：

| Hessian 特征值 | 临界点类型 |
|------|------|
| **全部为正** ($\lambda_{i} > 0$ for all $i$) | 局部最小值 (Local Minima) |
| **全部为负** ($\lambda_{i} < 0$ for all $i$) | 局部最大值 (Local Maxima) |
| **有正有负** | 鞍点 (Saddle Point) |

> 📐 **直觉理解**：在一维中，二阶导数 $> 0$ 表示极小值，$< 0$ 表示极大值。在高维中，Hessian 矩阵的特征值就是各个「主方向」上的二阶导数，判断方法完全一致。

### 1.4 Hessian 指导参数更新

当确认某个临界点是鞍点后，可以利用 Hessian 的**负特征值**对应的**特征向量 (Eigenvector)** 来逃离。

具体来说，令 $\lambda < 0$ 为 Hessian 的一个负特征值，$\mathbf{u}$ 为对应的特征向量：

$$
\mathbf{H} \mathbf{u} = \lambda \mathbf{u}
$$

沿 $\mathbf{u}$ 方向移动时：

$$
\mathbf{u}^{T} \mathbf{H} \mathbf{u} = \mathbf{u}^{T} (\lambda \mathbf{u}) = \lambda \|\mathbf{u}\|^{2} < 0
$$

即沿 $\mathbf{u}$ 方向损失会**下降**。因此可以沿该方向更新参数来逃离鞍点：

$$
\boldsymbol{\theta} \leftarrow \boldsymbol{\theta}' - \eta \cdot \mathbf{u}
$$

> ⚠️ 实际中很少显式计算 Hessian（维度太大、计算量太大），但这一理论帮助我们理解：梯度为零不一定意味着无路可走。

### 1.5 高维空间中的临界点：鞍点远多于局部最小值

在参数量极大的深度学习模型中，**临界点绝大多数是鞍点，而非局部最小值**。

**原因：** Hessian 矩阵的维度等于参数数量（可能达数百万甚至数十亿）。该矩阵的**所有**特征值同时为正的概率极低——只要有一个特征值为负，该点就是鞍点。

这意味着：
- 在实际训练中，梯度为零时更可能是碰到了鞍点，而非局部最小值
- 优化算法的核心挑战是**逃离鞍点**，而非避开局部最小值

---

## 2. 批次训练 (Batch Training)

### 2.1 基本概念回顾

实际训练中，每次不会用全部 $N$ 笔数据计算损失，而是将数据分成若干个**批次 (Batch)**：

- 每个批次大小为 $B$（**超参数 (Hyperparameter)**）
- 每算完一个 Batch 就更新一次参数 → 一次 **更新 (Update)**
- 遍历全部数据一次 → 一个 **轮次 (Epoch)**

详见 [[Guide.md#6-批次训练-batch-training]]。

### 2.2 为什么要分批次？

1. **硬件限制**：GPU 显存 (VRAM) 或内存 (RAM) 装不下整个数据集
2. **计算效率**：现代 GPU 的并行计算能力使得处理一个 Batch 和处理单个样本的时间几乎相同
3. **优化效果**：小批次的噪声梯度有助于探索损失平面、逃离鞍点

### 2.3 每个 Epoch 后洗牌 (Shuffle)

每完成一个 Epoch 后，应**随机打乱 (Shuffle)** 数据的顺序并重新划分 Batch。这样做的目的是：
- 使每个 Epoch 的 Batch 组合不同，避免模型看到固定的 Batch 顺序
- 增加训练的随机性，有助于泛化

### 2.4 小批次 vs 大批次

| | 小批次 (Small Batch) | 大批次 (Large Batch) |
|------|------|------|
| **单次更新计算时间** | 短 | 长（但 GPU 并行后差异不大） |
| **完成一个 Epoch 的时间** | **更⻓**（更新次数多） | **更短**（更新次数少） |
| **梯度噪声** | 大（每次更新噪声大） | 小（梯度估计更准确） |
| **泛化能力** | ✅ 通常更好 | ⚠️ 可能较差 |
| **收敛到的解** | 倾向于 **Flat Minima** | 倾向于 **Sharp Minima** |
| **硬件利用率** | 较低 | 较高 |

#### 为什么 GPU 下 Batch Size 差异不大？

对于现代 GPU 的高度并行架构，计算 $B=1$ 的梯度和 $B=1000$ 的梯度的**时间几乎相同**（直到 Batch 大到超出 GPU 的并行处理极限）。但是，小 Batch 意味着一个 Epoch 内需要更多次更新（$N/B$ 次），所以：

$$
\text{小 Batch 的总训练时间} = \text{更多更新次数} \times \text{每次更新时间相似}
$$

→ 小 Batch 完成一个 Epoch 的时间显著更长。

#### 为什么小 Batch 的泛化效果更好？

关键在于**噪声梯度 (Noisy Gradient)** 和**解的平坦度**：

- **小 Batch**：每次更新的梯度噪声大，优化路径「跳跃」，更容易跳过 Sharp Minima，最终收敛到**平坦最小值 (Flat Minima)**
- **大 Batch**：梯度比较精确，优化路径平滑，容易陷入**尖锐最小值 (Sharp Minima)**

**Flat Minima vs Sharp Minima：**

```
Flat Minima:                    Sharp Minima:
   损失                           损失
    ↑         ___                  ↑           /\
    |       _/   \_                |         _/  \_
    |     _/       \_              |       _/      \_
    |   _/           \_            |     _/  ↑       \_
    | _/               \_          |   _/ 测试误差大    \_
    +---------------------→       +---------------------→
         参数                              参数
```

- **Flat Minima**：参数在小范围内抖动时损失变化不大 → 泛化能力强
- **Sharp Minima**：参数稍有偏移损失就剧烈上升 → 测试时表现差

这是小 Batch 训练在测试集上往往表现更好的重要原因。

---

## 3. 动量 (Momentum)

### 3.1 动量的动机

普通的梯度下降在遇到以下情况时表现不佳：
- **鞍点**：梯度为零，参数停止更新
- **平坦区域**：梯度极小，收敛极其缓慢
- **峡谷地形**：在一个方向上震荡，另一个方向上进展缓慢

**动量 (Momentum)** 通过模拟物理中的惯性来缓解这些问题。

### 3.2 动量的数学表述

引入动量后，参数更新不再是简单的「当前梯度 × 学习率」，而是结合了前一步的移动方向：

$$
\begin{aligned}
\mathbf{v}^{t+1} &= \beta \mathbf{v}^{t} - \eta \cdot \nabla L(\boldsymbol{\theta}^{t}) \\
\boldsymbol{\theta}^{t+1} &\leftarrow \boldsymbol{\theta}^{t} + \mathbf{v}^{t+1}
\end{aligned}
$$

其中：
- $\mathbf{v}^{t}$ — 第 $t$ 步的**动量 (Momentum)**（即上一步的移动向量）
- $\beta$ — **动量系数**，通常设为 $0.9$，控制「惯性」的大小
- $\eta$ — 学习率

### 3.3 动量如何工作

可以分两步理解动量的效果：

**第一步：结合历史方向**

$$
\underbrace{\mathbf{v}^{t+1}}_{\text{新移动方向}} = \underbrace{\beta \mathbf{v}^{t}}_{\text{上一步的惯性}} + \underbrace{(-\eta \cdot \nabla L)}_{\text{当前梯度的修正}}
$$

- 上一步的方向被「记忆」下来（以 $\beta$ 的比例衰减）
- 当前梯度在此基础上做修正
- 如果连续几步的梯度方向一致 → 动量累积，加速前进
- 如果梯度方向交替变化 → 动量互相抵消，减少震荡

**第二步：更新参数**

$$
\boldsymbol{\theta}^{t+1} \leftarrow \boldsymbol{\theta}^{t} + \mathbf{v}^{t+1}
$$

参数沿累积的动量方向移动。

### 3.4 动量如何帮助逃离临界点

在鞍点处，梯度为零（$\nabla L = \mathbf{0}$），普通梯度下降会完全停滞。但动量仍然保留了**之前累积的移动方向**：

$$
\mathbf{v}^{t+1} = \beta \mathbf{v}^{t} - \eta \cdot \mathbf{0} = \beta \mathbf{v}^{t} \neq \mathbf{0}
$$

也就是说，即便当前梯度为零，动量仍然可以推动参数继续移动，从而**滚过鞍点**，继续下降。

> 🏀 **物理类比**：就像一个小球从山坡上滚下来。小球获得动量后，即使经过一个平坦的小平台（鞍点），也会由于惯性继续向前滚动，而不会停在平台上。

---

## 4. 关键结论 (Key Takeaways)

- **临界点 ≠ 死胡同**：梯度为零可能是鞍点而非局部最小值，高维空间中鞍点占绝大多数
- **Hessian 矩阵**是判断临界点类型的数学工具：看特征值的正负
  - 全正 → Local Minima
  - 有正有负 → Saddle Point
- **小 Batch 更有助于泛化**：噪声梯度帮助找到 Flat Minima，但训练时间更长
- **动量**给优化增加了「惯性」，是逃离鞍点和加速收敛的关键技术
- 现代优化器（如 Adam）同时结合了动量和自适应学习率

---

## 5. 相关链接 (Related)

- [[Guide.md]] — 梯度下降和批次训练入门
- [[梯度下降 Gradient Descent]] — 梯度下降的详细推导和变体
- [[损失函数 Loss Function]]
- [[过拟合 Overfitting]]
- [[ml.md]] — 训练诊断流程中关于优化问题的部分
