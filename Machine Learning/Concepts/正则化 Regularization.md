# 正则化 (Regularization)

## 1. 定义

**正则化 (Regularization)** 是一类通过对模型参数施加约束来防止**过拟合 (Overfitting)** 的技术。其核心思想是在损失函数中添加一个**惩罚项 (Penalty Term)**，抑制参数变得过大，从而限制模型的复杂度。

## 2. 数学形式

正则化后的损失函数为：

$$
L_{\text{reg}}(\boldsymbol{\theta}) = L(\boldsymbol{\theta}) + \lambda \cdot R(\boldsymbol{\theta})
$$

其中：
- $L(\boldsymbol{\theta})$ — 原始损失函数（如 MSE、交叉熵）
- $R(\boldsymbol{\theta})$ — 正则化项（惩罚项）
- $\lambda$ — 正则化系数，控制惩罚力度（**超参数 (Hyperparameter)**）

## 3. 常见正则化方法

### 3.1 L1 正则化 (Lasso)

$$
R(\boldsymbol{\theta}) = \|\boldsymbol{\theta}\|_{1} = \sum_{i} |\theta_{i}|
$$

**特点：**
- 倾向于产生**稀疏解**——部分参数被压缩为精确的 $0$
- 等价于在优化时对参数施加 Laplace 先验
- 可用于**特征选择 (Feature Selection)**

### 3.2 L2 正则化 (Ridge / Weight Decay)

$$
R(\boldsymbol{\theta}) = \|\boldsymbol{\theta}\|_{2}^{2} = \sum_{i} \theta_{i}^{2}
$$

**特点：**
- 倾向于让所有参数都**较小且均匀**，但不为 $0$
- 等价于在优化时对参数施加 Gaussian 先验
- 在深度学习中通常称为**权重衰减 (Weight Decay)**
- 梯度下降中的更新形式：

$$
\theta_{t+1} \leftarrow (1 - \eta \lambda) \cdot \theta_{t} - \eta \cdot \nabla L(\theta_{t})
$$

可以看到，每次更新时参数先衰减 $1 - \eta \lambda$ 的比例，这也是「权重衰减」名称的由来。

### 3.3 弹性网 (Elastic Net)

同时使用 L1 和 L2 正则化：

$$
R(\boldsymbol{\theta}) = \alpha \|\boldsymbol{\theta}\|_{1} + (1 - \alpha) \|\boldsymbol{\theta}\|_{2}^{2}
$$

兼具 L1 的稀疏性和 L2 的稳定性。

## 4. 深度学习中的隐式正则化

除了显式添加惩罚项，以下技术也有正则化效果：

| 方法 | 正则化原理 |
|------|------|
| **早停 (Early Stopping)** | 限制优化步数，等效于限制参数空间大小 |
| **丢弃法 (Dropout)** | 训练时随机丢弃神经元，等效于训练子网络集成 |
| **批归一化 (Batch Normalization)** | 引入噪声，有轻微的隐式正则化效果 |
| **数据增强** | 增加有效数据量，间接抑制过拟合 |
| **小批量训练 (Mini-batch)** | 梯度中的噪声也有轻微正则化效果 |

## 5. 如何选择正则化强度？

- $\lambda$ 太大 → 模型过于简单 → **欠拟合 (Underfitting)**
- $\lambda$ 太小 → 正则化无效 → **过拟合 (Overfitting)**
- 最优 $\lambda$ 通常通过**交叉验证 (Cross Validation)** 在验证集上调参确定

## 6. 相关链接

- [[过拟合 Overfitting]]
- [[损失函数 Loss Function]]
- [[梯度下降 Gradient Descent]]
- [[ml.md]]
