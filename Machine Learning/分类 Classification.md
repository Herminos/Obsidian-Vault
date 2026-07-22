# 分类 (Classification)

**分类 (Classification)** 是机器学习的核心任务之一：给定若干选项（类别），模型输出正确的那一个。

---

## 1. 分类与回归的区别

| | 回归 (Regression) | 分类 (Classification) |
|------|------|------|
| **输出类型** | 连续标量 (Scalar) | 离散类别 (Class) |
| **模型输出** | 直接输出数值 | 输出各类别的概率 |
| **输出层激活函数** | 恒等映射 / 无激活 | **Softmax**（多分类）/ **Sigmoid**（二分类） |
| **常用损失函数** | MSE、MAE | **交叉熵 (Cross-Entropy)** |
| **示例** | 预测房价、温度 | 识别猫/狗、垃圾邮件检测 |

回归的模型形式：

$$
y = b + \mathbf{c}^{T} \sigma(\mathbf{b} + \mathbf{W} \mathbf{x})
$$

分类则需要在此基础上加入 Softmax 将输出转为概率分布（见 [§3](#3-softmax从-logits-到概率)）。

---

## 2. 类别表示：独热编码 (One-Hot Encoding)

当类别之间没有天然的顺序或数值关系时（如猫、狗、鸟），使用**独热编码 (One-Hot Encoding)** 来表示类别标签。

对于一个 $K$ 分类问题，每个标签表示为一个长度为 $K$ 的向量：

$$
\mathbf{y} = [0, 0, \dots, 1, \dots, 0]^{T}
$$

其中只有正确类别对应的位置为 $1$，其余为 $0$。

**示例（3 分类：猫、狗、鸟）：**

$$
\begin{aligned}
\text{猫} &\rightarrow [1, 0, 0]^{T} \\
\text{狗} &\rightarrow [0, 1, 0]^{T} \\
\text{鸟} &\rightarrow [0, 0, 1]^{T}
\end{aligned}
$$

> ⚠️ 如果类别之间有自然的顺序关系（如「差、中、良、优」），则不适合用 One-Hot，而应使用**序数编码 (Ordinal Encoding)**。

---

## 3. Softmax：从 Logits 到概率

### 3.1 定义

神经网络最后一层输出的是原始分数（称为 **Logits**），记作 $y_1, y_2, \dots, y_K$。这些值可以是任意实数，需要转换为一个概率分布。

**Softmax 函数**将 $K$ 个任意实数映射为 $K$ 个概率值：

$$
\hat{y}_i = \frac{e^{y_i}}{\sum_{k=1}^{K} e^{y_k}} \quad (i = 1, 2, \dots, K)
$$

### 3.2 性质

| 性质 | 说明 |
|------|------|
| **归一化** | $\sum_{i=1}^{K} \hat{y}_i = 1$，构成合法概率分布 |
| **正值** | $\hat{y}_i \in (0, 1)$（指数函数的输出恒正） |
| **保序** | 若 $y_i > y_j$，则 $\hat{y}_i > \hat{y}_j$（指数函数单调递增） |
| **放大差异** | 指数运算使较大的值更突出，较小的值被压制 |

### 3.3 直观理解

Softmax 可以看作以下步骤的组合：

1. **取指数** $e^{y_i}$：将任意实数变为正数，同时放大数值间差异
2. **归一化**：除以总和，使所有输出之和为 $1$

最终每个 $\hat{y}_i$ 可以解释为「模型认为样本属于类别 $i$ 的概率」。

### 3.4 二分类特例：Sigmoid = Softmax with $K=2$

当 $K=2$ 时，Softmax 退化为 Sigmoid：

$$
\begin{aligned}
\hat{y}_1 &= \frac{e^{y_1}}{e^{y_1} + e^{y_2}} = \frac{1}{1 + e^{-(y_1 - y_2)}} = \sigma(y_1 - y_2) \\[6pt]
\hat{y}_2 &= 1 - \hat{y}_1
\end{aligned}
$$

因此，二分类时使用 Sigmoid 即可，多分类必须使用 Softmax。

---

## 4. 分类的损失函数

### 4.1 交叉熵损失 (Cross-Entropy Loss)

**交叉熵 (Cross-Entropy)** 是分类问题的标准损失函数。它衡量两个概率分布之间的差异——模型预测的分布 $\hat{\mathbf{y}}$ 与真实分布 $\mathbf{y}$。

对于**多分类**（$K$ 个类别）：

$$
L = -\sum_{i=1}^{K} y_i \log(\hat{y}_i)
$$

由于 $\mathbf{y}$ 是 One-Hot 向量（只有正确类别 $c$ 处 $y_c = 1$，其余为 $0$），上式简化为：

$$
L = -\log(\hat{y}_c)
$$

其中 $\hat{y}_c$ 是模型对**正确类别**的预测概率。

对于 $N$ 个样本的整体损失：

$$
L = -\frac{1}{N} \sum_{n=1}^{N} \sum_{i=1}^{K} y_{n,i} \log(\hat{y}_{n,i})
$$

### 4.2 交叉熵与最大似然估计

最小化交叉熵损失**等价于最大化似然 (Maximum Likelihood)**：

$$
\begin{aligned}
\arg\min_{\boldsymbol{\theta}} \left( -\sum_{n} \log P_{\boldsymbol{\theta}}(y_n \mid \mathbf{x}_n) \right)
= \arg\max_{\boldsymbol{\theta}} \prod_{n} P_{\boldsymbol{\theta}}(y_n \mid \mathbf{x}_n)
\end{aligned}
$$

即：交叉熵最小化 = 让模型对正确类别的预测概率尽可能大。

### 4.3 为什么交叉熵优于 MSE？

对于分类任务，**交叉熵远优于均方误差 (MSE)**。原因在于梯度行为：

**交叉熵 + Softmax 的梯度：**

$$
\frac{\partial L}{\partial y_i} = \hat{y}_i - y_i
$$

梯度等于「预测值 - 真实值」，线性且简洁：

- 预测正确时（$\hat{y}_c \approx 1$）→ 梯度接近 $0$，参数几乎不更新 ✅
- 预测错误时（$\hat{y}_c \approx 0$）→ 梯度较大，参数大幅更新，快速修正 ✅

**MSE + Softmax 的梯度：**

$$
\frac{\partial L}{\partial y_i} = 2(\hat{y}_i - y_i) \cdot \hat{y}_i (1 - \hat{y}_i)
$$

多了一个因子 $\hat{y}_i (1 - \hat{y}_i)$，导致**梯度饱和**问题：

- 当模型预测错误且非常自信（$\hat{y}_{\text{wrong}} \approx 1$，$\hat{y}_c \approx 0$）时：
  - $\hat{y}_c (1 - \hat{y}_c) \approx 0$，梯度极小
  - 参数几乎不更新 → **学不动** ❌

这正是交叉熵成为分类任务标准损失函数的根本原因。

| | 交叉熵 (Cross-Entropy) | 均方误差 (MSE) |
|------|:--:|:--:|
| **梯度形式** | $\hat{y}_i - y_i$ | $2(\hat{y}_i - y_i) \cdot \hat{y}_i (1 - \hat{y}_i)$ |
| **错误时的梯度大小** | 大 ✅ | 可能极小（饱和）❌ |
| **收敛速度** | 快 | 慢 |
| **与 Softmax 配合** | ✅ 天然良配 | ❌ 梯度消失 |

---

## 5. 关键结论 (Key Takeaways)

- **分类 vs 回归**：分类输出离散概率分布（需 Softmax + Cross-Entropy），回归输出连续标量
- **One-Hot 编码**是当类别无顺序关系时的标准标签表示法
- **Softmax** 将任意实数向量转换为概率分布：$\hat{y}_i = e^{y_i} / \sum_k e^{y_k}$
- **二分类用 Sigmoid，多分类用 Softmax**——前者是后者的特例
- **Cross-Entropy** 是分类的标准损失函数，等价于最大化似然
- **Cross-Entropy 优于 MSE** 的根本原因：梯度不会在错误时饱和，始终能有效更新

---

## 6. 相关链接 (Related)

- [[入门 Guide]] — 入门：回归、分类、结构化学习的定义
- [[损失函数 Loss Function]] — MSE、MAE、Cross-Entropy 等损失函数详解
- [[激活函数 Activation Function]] — Sigmoid 和 Softmax 的详细对比
- [[梯度下降 Gradient Descent]]
- [[优化 Optimization]]
