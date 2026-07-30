---
tags:
  - ml/guide
aliases:
  - 深度学习入门
  - ML Guide
created: 2026-07-20
---

# 深度学习入门 (Introduction to Deep Learning)

深度学习（Deep Learning）的核心思想约等于**寻找一个函数（Find a Function）**。

## 专有名词

1. **回归 (Regression)**：函数输出一个标量 (scalar)。
2. **分类 (Classification)**：给定若干选项 (classes)，函数输出正确的那一个。
3. **结构化学习 (Structured Learning)**：创造具有结构的内容（如图像、文档等）。

---

## 1. 寻找函数：含有未知参数的模型

第一步是找到一个**带有未知参数 (Unknown Parameters)** 的函数：

$$
y = b + w x_{1}
$$

其中：
- $y$ 为输出（即**特征 Feature**）
- $x_{1}$ 为输入 (Input)
- $b$ 为**偏置 (Bias)**，$w$ 为**权重 (Weight)**，两者均为未知参数
- $y = b + w x_{1}$ 整体称为**模型 (Model)**

---

## 2. 从训练数据定义损失函数

**损失函数 (Loss Function)** 是关于参数 (parameters) 的函数：

$$
L(b, w)
$$

损失函数衡量一组参数值的好坏：**损失越大，参数越差**。

例如，给定参数 $L(0.5k, 1)$：

$$
y = 0.5k + 1 \cdot x_{1}
$$

若预测值为 $5.3k$ 而真实值为 $4.9k$，则单点误差为：

$$
e_{1} = |y - \hat{y}|
$$

其中 $\hat{y}$ 表示真实值 (ground truth)。

整体损失为所有误差的均值：

$$
L = \frac{1}{N} \sum_{n=1}^{N} e_{n}
$$

常见误差函数：
- $e = |y - \hat{y}|$ — **平均绝对误差 (MAE, Mean Absolute Error)**
- $e = (y - \hat{y})^{2}$ — **均方误差 (MSE, Mean Squared Error)**

---

## 3. 优化

目标是找到一组 $w$ 和 $b$，使损失函数最小化。

### 3.1 梯度下降 (Gradient Descent)

梯度下降的步骤：

1. （随机）选取初始值 $w^{0}$
2. 计算 $\frac{\partial L}{\partial w}\big|_{w=w^{0}}$ 即 $L$ 对 $w$ 的偏导数
3. 导数为负 $\rightarrow$ 增大 $w$；导数为正 $\rightarrow$ 减小 $w$
4. $\eta$ 为**学习率 (Learning Rate)**，属于**超参数 (Hyperparameter)**，由人为设定
5. 迭代更新：$w^{1} \leftarrow w^{0} - \eta \cdot \frac{\partial L}{\partial w}$
6. 若某步 $w^{t}$ 处导数为 $0$，则参数不再更新
7. 可能收敛到**局部最小值 (Local Minima)** 而非**全局最小值 (Global Minima)**

#### 双参数情况

对于 $L(b, w)$，同时更新两个参数：

$$
\begin{aligned}
w^{t+1} &\leftarrow w^{t} - \eta \cdot \frac{\partial L}{\partial w}\Big|_{(b^{t}, w^{t})} \\[6pt]
b^{t+1} &\leftarrow b^{t} - \eta \cdot \frac{\partial L}{\partial b}\Big|_{(b^{t}, w^{t})}
\end{aligned}
$$

### 3.2 训练 (Training)

以上三步（找模型、定义损失、优化参数）合称为**训练 (Training)**。

---

## 4. 线性模型的局限

$$
y = wx + b
$$

称为**线性模型 (Linear Model)**。

线性模型的表达能力有很大限制，这种来自模型本身的限制称为**模型偏差 (Model Bias)**。后续需要引入更复杂的非线性模型来解决这一问题。

---

## 5. 超越线性模型：Sigmoid 函数

线性模型 $y = wx + b$ 表达能力有限，无法拟合复杂的数据分布。我们可以用**常数 + Sigmoid 函数**的组合来逼近任意连续曲线。

### 5.1 Sigmoid 函数

**Sigmoid 函数 (Sigmoid Function)** 是一条 S 形曲线，定义为：

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

在实际建模中，我们使用带参数的 Sigmoid：

$$
y = c \cdot \sigma(b + w x_{1}) = \frac{c}{1 + e^{-(b + w x_{1})}}
$$

其中三个参数各自控制曲线的形态：
- $w$ → 改变曲线的**斜率 (slope)**
- $b$ → 改变曲线的**水平位移 (shift)**
- $c$ → 改变曲线的**高度 (height)**

### 5.2 多特征 + 多 Sigmoid

当有多个特征时，模型扩展为：

$$
y = b + \sum_{i} c_{i} \cdot \sigma\left(b_{i} + \sum_{j} w_{ij} x_{j}\right)
$$

符号说明：
- $j = 1, 2, \dots$ — 特征的编号 (number of features)
- $i = 1, 2, \dots$ — Sigmoid 的编号 (number of sigmoids)

对于 3 个特征、3 个 Sigmoid 的具体展开：

$$
\begin{aligned}
r_{1} &= b_{1} + w_{11}x_{1} + w_{12}x_{2} + w_{13}x_{3} \\
r_{2} &= b_{2} + w_{21}x_{1} + w_{22}x_{2} + w_{23}x_{3} \\
r_{3} &= b_{3} + w_{31}x_{1} + w_{32}x_{2} + w_{33}x_{3}
\end{aligned}
$$

### 5.3 矩阵形式

令 $\mathbf{x}$ 为特征向量，$\mathbf{W}$ 为权重矩阵，$\mathbf{b}$ 为偏置向量：

$$
\mathbf{r} = \mathbf{b} + \mathbf{W} \mathbf{x}
$$

经过 Sigmoid 激活后再与系数向量 $\mathbf{c}$ 组合：

$$
\begin{aligned}
a_{i} &= \sigma(r_{i}) \\[4pt]
y &= b + \sum_{i} c_{i} a_{i} = b + \mathbf{c}^{T} \mathbf{a}
\end{aligned}
$$

写成紧凑的线性代数形式：

$$
y = b + \mathbf{c}^{T} \sigma(\mathbf{b} + \mathbf{W} \mathbf{x})
$$

其中：
- $\mathbf{x}$ — 输入特征 (feature)
- $\mathbf{W}, \mathbf{b}, b, \mathbf{c}^{T}$ — 未知参数 (Unknown Parameters)

将所有可训练参数（$\mathbf{W}$ 的所有行/列展开、$\mathbf{b}$、$\mathbf{c}^{T}$、$b$）拼接为一个长向量 $\boldsymbol{\theta}$，损失函数即为参数的函数 $L(\boldsymbol{\theta})$。

### 5.4 参数优化

在统一的参数向量 $\boldsymbol{\theta}$ 下，优化目标为：

$$
\boldsymbol{\theta}^{*} = \arg\min_{\boldsymbol{\theta}} L(\boldsymbol{\theta})
$$

训练步骤：
1. （随机）选取初始值 $\boldsymbol{\theta}^{0}$
2. 计算梯度 $\mathbf{g} = \nabla L(\boldsymbol{\theta}^{0})$
3. 更新：$\boldsymbol{\theta}^{1} \leftarrow \boldsymbol{\theta}^{0} - \eta \cdot \mathbf{g}$
4. 重复直到收敛

---

## 6. 批次训练 (Batch Training)

实际训练中，不会一次性用全部 $N$ 笔资料计算损失。而是将资料分成若干**批次 (Batch)**：

- 每个批次大小为 $B$（**批次大小 (Batch Size)**，属于超参数）
- 每算完一个 Batch 的 Loss 就更新一次参数，称为一次 **更新 (Update)**
- 跑完所有 $N$ 笔资料（即所有 Batch）称为一个**轮次 (Epoch)**

**实例：**

$$
N = 10000 \text{ (总样本数)}, \quad B = 10 \text{ (批次大小)}
$$

则：

$$
\text{1 Epoch 中的更新次数} = \frac{N}{B} = 1000 \text{ updates}
$$

| 术语 | 含义 |
|------|------|
| 更新 (Update) | 每算一个 Batch 的 Loss，更新一次参数 |
| 轮次 (Epoch) | 遍历完所有训练资料一次 |
| 批次大小 (Batch Size) | 每个 Batch 含有的样本数，属于超参数 |

---

## 7. 激活函数：从 Sigmoid 到 ReLU

### 7.1 Sigmoid 的替代方案

Sigmoid 并非唯一的选择。**ReLU (Rectified Linear Unit)** 是另一种常用的激活函数：

$$
\text{ReLU}(z) = c \cdot \max(0, \; b + w x_{1})
$$

扩展到多特征：

$$
y = b + \sum_{i} c_{i} \cdot \max\!\left(0, \; b_{i} + \sum_{j} w_{ij} x_{j}\right)
$$

### 7.2 Sigmoid vs ReLU

两者统称为**激活函数 (Activation Function)**：

| 激活函数 | 公式 | 特点 |
|----------|------|------|
| Sigmoid | $\sigma(z) = \frac{1}{1 + e^{-z}}$ | S 形，输出在 $(0, 1)$ 之间 |
| ReLU | $\text{ReLU}(z) = \max(0, z)$ | 计算简单，实践中效果通常更好 |

> 在实践中，**ReLU 通常比 Sigmoid 表现更好**。

---

## 8. 神经网络与深度学习

### 8.1 从神经元到网络

将上述结构不断堆叠：

$$
\begin{aligned}
\mathbf{a} &= f(\mathbf{b} + \mathbf{W} \mathbf{x}) \\
\mathbf{a}' &= f(\mathbf{b}' + \mathbf{W}' \mathbf{a}) \\
&\vdots
\end{aligned}
$$

核心概念：

| 术语 | 定义 |
|------|------|
| **神经元 (Neuron)** | 单个计算单元：$a = f(b + \mathbf{W} \mathbf{x})$ |
| **神经网络 (Neural Network)** | 多个神经元的组合 |
| **层 (Layer)** | 同一排的神经元称为一层 |
| **深度学习 (Deep Learning)** | 包含很多层的网络（"深" = 层数多） |

### 8.2 过拟合 (Overfitting)

当模型在**训练资料上表现越来越好，但在未见过的资料上表现反而变差**时，称为**过拟合 (Overfitting)**。

这意味着模型过度记忆了训练数据中的噪声，而非学习到真正的模式。解决方法包括：
- 增加训练数据
- [[正则化 Regularization]]
- 早停 (Early Stopping)
- 降低模型复杂度

---

## 相关笔记

- [[线性模型 Linear Model]]
- [[梯度下降 Gradient Descent]]
- [[损失函数 Loss Function]]
- [[模型偏差 Model Bias]]
- [[激活函数 Activation Function]]
- [[过拟合 Overfitting]]
- [[神经网络 Neural Network]]
- [[深度学习 Deep Learning]]
