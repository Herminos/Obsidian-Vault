# 损失函数 (Loss Function)

## 1. 定义

**损失函数 (Loss Function)** 是衡量模型预测值与真实值之间差异的函数。它是关于模型参数 $\boldsymbol{\theta}$ 的函数，值越小表示模型预测越准确。

$$
L(\boldsymbol{\theta}) = \text{衡量 } \hat{y} \text{ 与 } y \text{ 的差距}
$$

其中 $\hat{y} = f_{\boldsymbol{\theta}}(x)$ 是模型预测值，$y$ 是真实标签。

## 2. 回归问题中的损失函数

### 2.1 均方误差 (MSE, Mean Squared Error)

$$
\text{MSE} = \frac{1}{N} \sum_{n=1}^{N} (y_n - \hat{y}_n)^{2}
$$

**特点：**
- 对**离群值 (Outliers)** 敏感——大误差被平方放大
- 梯度随误差线性变化，误差大时梯度也大
- 常用于线性回归和数值预测

### 2.2 平均绝对误差 (MAE, Mean Absolute Error)

$$
\text{MAE} = \frac{1}{N} \sum_{n=1}^{N} |y_n - \hat{y}_n|
$$

**特点：**
- 对离群值**更鲁棒**——不会像 MSE 那样被放大
- 梯度恒为常数（$\pm 1$），在最小值附近可能震荡
- 在 $y = \hat{y}$ 处不可导

### 2.3 Huber 损失

MSE 和 MAE 的折中方案：

$$
\text{Huber} =
\begin{cases}
\frac{1}{2}(y - \hat{y})^{2} & \text{if } |y - \hat{y}| \leq \delta \\[4pt]
\delta \cdot (|y - \hat{y}| - \frac{1}{2}\delta) & \text{otherwise}
\end{cases}
$$

- 小误差时用 MSE（可导、平滑）
- 大误差时用 MAE（鲁棒）

## 3. 分类问题中的损失函数

### 3.1 交叉熵损失 (Cross-Entropy Loss)

对于二分类：

$$
L = -\frac{1}{N} \sum_{n=1}^{N} \big[y_n \log(\hat{y}_n) + (1 - y_n) \log(1 - \hat{y}_n)\big]
$$

对于多分类（$K$ 个类别）：

$$
L = -\frac{1}{N} \sum_{n=1}^{N} \sum_{k=1}^{K} y_{n,k} \log(\hat{y}_{n,k})
$$

其中 $\hat{y}_{n,k}$ 是模型预测样本 $n$ 属于类别 $k$ 的概率（通常经过 softmax）。

**特点：**
- 与最大似然估计等价
- 配合 softmax 使用时梯度简洁
- 是分类问题的标准选择

### 3.2 铰链损失 (Hinge Loss)

$$
L = \frac{1}{N} \sum_{n=1}^{N} \max(0, 1 - y_n \cdot \hat{y}_n)
$$

- 用于 SVM (Support Vector Machine)
- 当预测正确且置信度足够时不产生损失

## 4. 损失函数的性质

一个好的损失函数通常应满足：

| 性质 | 说明 |
|------|------|
| **连续可导** | 便于使用[[梯度下降 Gradient Descent]]优化 |
| **凸性 (Convexity)** | 保证全局最优解的唯一性（线性模型 + MSE/MAE = 凸优化） |
| **对任务敏感** | 回归用平方类，分类用交叉熵 |
| **鲁棒性** | 不易受离群值影响 |

## 5. 整体损失 vs 单点损失

| | 单点损失 | 整体损失 |
|------|------|------|
| 记号 | $e_n$ 或 $\ell(y_n, \hat{y}_n)$ | $L$ |
| 范围 | 单个样本 | 整个数据集 |
| 关系 | — | $L = \frac{1}{N} \sum_{n=1}^{N} e_n$ |

## 6. 相关链接

- [[Guide.md]] — 损失函数的入门定义
- [[梯度下降 Gradient Descent]]
- [[过拟合 Overfitting]]
- [[正则化 Regularization]]
