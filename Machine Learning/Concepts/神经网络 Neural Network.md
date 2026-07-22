---
tags:
  - ml/concept
  - ml/neural-net
aliases:
  - Neural Network
  - 神经网络
created: 2026-07-20
---

# 神经网络 (Neural Network)

## 1. 定义

**神经网络 (Neural Network)** 是由大量**神经元 (Neuron)** 相互连接构成的计算模型，灵感来源于生物神经系统。每个神经元执行简单的计算，大量神经元的组合可以表示极其复杂的函数。

## 2. 神经元 (Neuron)

单个神经元的计算：

$$
a = f\left(b + \sum_{j=1}^{d} w_{j} x_{j}\right) = f(b + \mathbf{w}^{T} \mathbf{x})
$$

其中：
- $\mathbf{x} \in \mathbb{R}^{d}$ — 来自前一层的 $d$ 个输入
- $\mathbf{w}$ — **权重 (Weights)**
- $b$ — **偏置 (Bias)**
- $f(\cdot)$ — **激活函数 (Activation Function)**
- $a$ — 输出（**激活值 (Activation)**）

## 3. 层 (Layer)

多个神经元并排组成一个**层 (Layer)**：

### 3.1 输入层 (Input Layer)

接收原始特征 $\mathbf{x}$，不进行任何计算。

### 3.2 隐藏层 (Hidden Layer)

位于输入层和输出层之间的层，执行非线性变换。一个网络可以有多个隐藏层。

### 3.3 输出层 (Output Layer)

产生最终预测。激活函数根据任务选择：
- 回归 → 恒等映射（无激活）
- 二分类 → Sigmoid
- 多分类 → Softmax

## 4. 前向传播 (Forward Propagation)

从输入到输出的计算流程。对于一个 $L$ 层网络：

$$
\begin{aligned}
\mathbf{a}^{(0)} &= \mathbf{x} \\
\mathbf{z}^{(l)} &= \mathbf{b}^{(l)} + \mathbf{W}^{(l)} \mathbf{a}^{(l-1)} \quad (l = 1, 2, \dots, L) \\
\mathbf{a}^{(l)} &= f^{(l)}(\mathbf{z}^{(l)}) \\
\hat{\mathbf{y}} &= \mathbf{a}^{(L)}
\end{aligned}
$$

其中：
- $\mathbf{z}^{(l)}$ — 第 $l$ 层的线性组合结果（激活前）
- $\mathbf{a}^{(l)}$ — 第 $l$ 层的激活输出
- $\mathbf{W}^{(l)}, \mathbf{b}^{(l)}$ — 第 $l$ 层的权重和偏置
- $f^{(l)}$ — 第 $l$ 层的激活函数

## 5. 全连接层 (Fully Connected Layer)

最基本的层类型：**每个神经元与上一层的所有神经元都相连**。

参数数量：
- 权重：$d_{\text{in}} \times d_{\text{out}}$
- 偏置：$d_{\text{out}}$

其中 $d_{\text{in}}$ 和 $d_{\text{out}}$ 分别为输入和输出维度。

## 6. 通用逼近定理 (Universal Approximation Theorem)

> 一个包含至少一个隐藏层且使用非线性激活函数的前馈神经网络，只要有足够多的神经元，就可以**以任意精度逼近任意连续函数**。

这一定理说明了神经网络的表达能力，但：
- 没有说明需要多少神经元（可能指数级多）
- 没有说明如何找到这些参数（优化问题）
- 深度网络可能比宽而浅的网络更参数高效

## 7. 训练过程

1. **前向传播**：输入 → 逐层计算 → 输出预测
2. **计算损失**：对比预测 $\hat{\mathbf{y}}$ 和真实标签 $\mathbf{y}$
3. **反向传播 (Backpropagation)**：利用链式法则从输出层逐层回传梯度
4. **参数更新**：使用[[梯度下降 Gradient Descent]]或其变体更新 $\mathbf{W}$ 和 $\mathbf{b}$

## 8. 网络架构设计

### 决定因素

| 因素 | 影响 |
|------|------|
| **层数 (Depth)** | 更深的网络能学习更抽象/层次化的特征 |
| **每层宽度 (Width)** | 更宽的层有更多神经元，表达能力更强 |
| **激活函数** | 决定非线性类型，ReLU 是默认选择 |
| **连接模式** | 全连接 / 卷积 (CNN) / 自注意力 (Transformer) |
| **正则化** | 防止过拟合 |

### 经验法则

- 从较浅、较小的网络开始
- 逐步增加复杂度，观察训练和验证损失
- 更多层通常比更宽的层更有效（深度学习）

## 9. 相关链接

- [[Guide.md]] — 神经网络的入门定义
- [[深度学习 Deep Learning]]
- [[激活函数 Activation Function]]
- [[梯度下降 Gradient Descent]]
- [[损失函数 Loss Function]]
- [[过拟合 Overfitting]]
