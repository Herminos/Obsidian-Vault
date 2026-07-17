# 激活函数 (Activation Function)

## 1. 定义

**激活函数 (Activation Function)** 是神经网络中引入**非线性**的组件。没有激活函数，多层线性变换的组合仍然是线性的——深度网络将退化为等效的单层线性模型。

## 2. 为什么需要非线性？

考虑一个两层「网络」：

$$
\begin{aligned}
\mathbf{h} &= \mathbf{W}_{1} \mathbf{x} + \mathbf{b}_{1} \\
\mathbf{y} &= \mathbf{W}_{2} \mathbf{h} + \mathbf{b}_{2}
\end{aligned}
$$

代入可得：

$$
\mathbf{y} = \mathbf{W}_{2} (\mathbf{W}_{1} \mathbf{x} + \mathbf{b}_{1}) + \mathbf{b}_{2} = \underbrace{\mathbf{W}_{2} \mathbf{W}_{1}}_{\mathbf{W}'} \mathbf{x} + \underbrace{(\mathbf{W}_{2}\mathbf{b}_{1} + \mathbf{b}_{2})}_{\mathbf{b}'}
$$

多个线性变换的复合**仍然是线性变换**。加入激活函数后：

$$
\mathbf{y} = \mathbf{W}_{2} \cdot f(\mathbf{W}_{1} \mathbf{x} + \mathbf{b}_{1}) + \mathbf{b}_{2}
$$

$f$ 的非线性使得这个复合变换不再是线性的，网络因此获得了表示非线性模式的能力。

## 3. 常见激活函数

### 3.1 Sigmoid

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

| 属性 | 值 |
|------|------|
| 输出范围 | $(0, 1)$ |
| 可导性 | 处处可导 |
| 参数控制 | $y = c \cdot \sigma(b + w x)$ |

**优点：**
- 输出有界，可作为概率解释
- 平滑、单调

**缺点：**
- **梯度消失 (Vanishing Gradient)**：输入绝对值大时梯度趋近于 $0$
- 输出不以 $0$ 为中心
- 涉及指数运算，计算量大

### 3.2 Tanh (双曲正切)

$$
\tanh(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}} = 2\sigma(2z) - 1
$$

| 属性 | 值 |
|------|------|
| 输出范围 | $(-1, 1)$ |
| 中心化 | 以 $0$ 为中心（优于 Sigmoid） |

**缺点**：同样存在梯度消失问题。

### 3.3 ReLU (Rectified Linear Unit)

$$
\text{ReLU}(z) = \max(0, z)
$$

| 属性 | 值 |
|------|------|
| 输出范围 | $[0, +\infty)$ |
| 计算复杂度 | 极低（仅需比较） |

**优点：**
- 计算简单高效
- 在 $z > 0$ 区域梯度恒为 $1$，**缓解了梯度消失问题**
- 实践中通常比 Sigmoid/Tanh 表现更好
- 产生稀疏激活（部分神经元输出为 $0$）

**缺点：**
- **Dying ReLU**：当 $z \leq 0$ 时梯度为 $0$，若神经元持续落入该区域则「死亡」，不再更新

### 3.4 ReLU 的变体

| 变体 | 公式 | 改进 |
|------|------|------|
| **Leaky ReLU** | $\max(0.01z, z)$ | 负区域有微小梯度，缓解 Dying ReLU |
| **Parametric ReLU (PReLU)** | $\max(\alpha z, z)$, $\alpha$ 可学习 | 负区域的斜率由数据学习 |
| **ELU** | $z > 0$: $z$; $z \leq 0$: $\alpha(e^{z} - 1)$ | 负区域平滑、以 $0$ 为中心 |
| **GELU** | $z \cdot \Phi(z)$ ($\Phi$ 为标准正态 CDF) | Transformer 中广泛使用 |

### 3.5 Softmax

$$
\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{k=1}^{K} e^{z_k}}
$$

- 将 $K$ 个实数转换为 $K$ 个概率值（和为 $1$）
- 通常用于多分类的**最后一层**

## 4. 激活函数一览表

| 激活函数 | 公式 | 输出范围 | 梯度消失 | 推荐场景 |
|------|------|------|:--:|------|
| Sigmoid | $\frac{1}{1 + e^{-z}}$ | $(0,1)$ | ⚠️ 严重 | 二分类输出层 |
| Tanh | $\tanh(z)$ | $(-1,1)$ | ⚠️ 严重 | 较少使用 |
| ReLU | $\max(0, z)$ | $[0,\infty)$ | ✅ 缓解 | **隐藏层首选** |
| Leaky ReLU | $\max(0.01z, z)$ | $(-\infty,\infty)$ | ✅ 缓解 | Dying ReLU 场景 |
| Softmax | $\frac{e^{z_i}}{\sum e^{z_k}}$ | $(0,1)$ | — | **多分类输出层** |

## 5. 选择建议

- **隐藏层**：默认使用 ReLU 或其变体
- **二分类输出层**：Sigmoid
- **多分类输出层**：Softmax
- **回归输出层**：不使用激活函数（或恒等映射 $f(z)=z$）

## 6. 相关链接

- [[Guide.md]] — Sigmoid 和 ReLU 的入门介绍
- [[神经网络 Neural Network]]
- [[线性模型 Linear Model]]
- [[深度学习 Deep Learning]]
