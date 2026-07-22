---
tags:
  - ml/concept
  - ml/optimization
aliases:
  - Gradient Descent
  - 梯度下降
created: 2026-07-20
---

# 梯度下降 (Gradient Descent)

## 1. 定义

**梯度下降 (Gradient Descent)** 是一种通过迭代求解函数最小值的优化算法，是训练机器学习模型最核心的优化方法。

## 2. 核心思想

顺着函数下降最快的方向（即**负梯度方向**）逐步移动，直到到达某个最小值点。

## 3. 数学推导

设损失函数为 $L(\boldsymbol{\theta})$，其中 $\boldsymbol{\theta}$ 是可训练参数的集合。

### 3.1 单参数情况

对单个参数 $w$：

$$
w^{t+1} \leftarrow w^{t} - \eta \cdot \frac{\partial L}{\partial w}\Big|_{w=w^{t}}
$$

### 3.2 多参数情况

对于参数向量 $\boldsymbol{\theta}$：

$$
\boldsymbol{\theta}^{t+1} \leftarrow \boldsymbol{\theta}^{t} - \eta \cdot \nabla L(\boldsymbol{\theta}^{t})
$$

其中：
- $\eta$ — **学习率 (Learning Rate)**，控制更新步长
- $\nabla L$ — **梯度 (Gradient)**，即所有偏导组成的向量

## 4. 训练步骤

1. （随机）初始化参数 $\boldsymbol{\theta}^{0}$
2. 计算当前参数的梯度 $\mathbf{g} = \nabla L(\boldsymbol{\theta}^{t})$
3. 更新参数：$\boldsymbol{\theta}^{t+1} \leftarrow \boldsymbol{\theta}^{t} - \eta \cdot \mathbf{g}$
4. 重复步骤 2-3，直到收敛

## 5. 学习率 (Learning Rate)

**学习率 $\eta$** 是最重要的**超参数 (Hyperparameter)** 之一：

| 学习率 | 效果 |
|--------|------|
| **太大** | 震荡或发散，可能跳过最小值 |
| **太小** | 收敛太慢，耗时长 |
| **适中** | 平稳收敛到最小值 |

### 常见学习率调度策略

- **固定学习率**：最简单的策略
- **学习率衰减 (Learning Rate Decay)**：随训练进行逐渐减小
- **余弦退火 (Cosine Annealing)**：按余弦曲线周期性变化
- **预热 (Warmup)**：先从小学习率开始逐渐增大，再衰减

## 6. 梯度下降的变体

### 6.1 批量梯度下降 (Batch GD)

每次使用**全部**训练数据计算梯度：

$$
\nabla L = \frac{1}{N} \sum_{n=1}^{N} \nabla L_{n}
$$

- 优点：梯度估计精确
- 缺点：计算量大，内存需求高

### 6.2 随机梯度下降 (SGD)

每次仅使用**1 个**样本计算梯度：

$$
\nabla L \approx \nabla L_{n}
$$

- 优点：计算快，能逃离局部最小值
- 缺点：梯度估计噪声大，收敛不稳定

### 6.3 小批量梯度下降 (Mini-batch GD)

取折中——每次使用**一小批 (Batch)** 样本：

$$
\nabla L \approx \frac{1}{B} \sum_{b=1}^{B} \nabla L_{b}
$$

这是实践中**最常用**的方式。详见 [[Guide.md#6-批次训练-batch-training]]。

### 6.4 带动量 (Momentum)

引入动量项，模拟物理中的惯性：

$$
\begin{aligned}
\mathbf{v}^{t+1} &= \beta \mathbf{v}^{t} - \eta \cdot \nabla L(\boldsymbol{\theta}^{t}) \\
\boldsymbol{\theta}^{t+1} &\leftarrow \boldsymbol{\theta}^{t} + \mathbf{v}^{t+1}
\end{aligned}
$$

- 有助于加速收敛和逃离局部最小值
- $\beta$ 为动量系数，通常设为 $0.9$

### 6.5 Adam

**Adam (Adaptive Moment Estimation)** 结合了动量和自适应学习率，是目前最流行的优化器之一：

- 对每个参数使用不同的自适应学习率
- 结合了一阶动量（均值）和二阶动量（方差）
- 收敛快，对超参数不敏感

## 7. 局限性

### 7.1 局部最小值 (Local Minima)

梯度下降可能收敛到局部最小值而非全局最小值：

$$
\nabla L(\boldsymbol{\theta}^{*}) = 0 \quad\text{但}\quad L(\boldsymbol{\theta}^{*}) > L(\boldsymbol{\theta}_{\text{global}})
$$

**缓解方法**：随机初始化、动量、SGD 的噪声、学习率调度。

### 7.2 鞍点 (Saddle Point)

鞍点处梯度也为零，但不是最小值。高维空间中鞍点比局部最小值更常见。

## 8. 超参数一览

| 超参数 | 含义 | 典型值 |
|--------|------|--------|
| $\eta$ | 学习率 | $0.1$, $0.01$, $0.001$ |
| $B$ | 批次大小 (Batch Size) | $32$, $64$, $128$ |
| $\beta$ | 动量系数 | $0.9$ |

## 9. 相关链接

- [[Guide.md]] — 梯度下降的入门推导
- [[损失函数 Loss Function]]
- [[ml.md]]
