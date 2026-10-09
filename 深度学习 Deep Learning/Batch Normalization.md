# 批归一化 (Batch Normalization)

**批归一化 (Batch Normalization, BN)** 是深度学习中广泛使用的训练加速与稳定技术。它通过对每一层的激活值做归一化，使**误差曲面 (Error Surface)** 更加平滑，从而可以使用更大的学习率、更快收敛。

---

## 1. 问题：输入尺度影响训练难度

在线性模型或神经网络中，如果某个输入特征的数值远远大于其他特征，那么该特征对应的权重参数会对输出产生不成比例的影响：

$$
y = b + \sum_{j} w_j x_j
$$

当某些 $x_j$ 很大时，$w_j$ 的微小变化会导致输出剧变 → **误差曲面在该方向极度陡峭**，梯度下降训练困难。

**直观理解：** 在一个方向上学习率太小（震荡），另一个方向上学习率太大（进展缓慢），很难找到统一合适的学习率。

---

## 2. 特征归一化 (Feature Normalization)

### 2.1 基本做法

对输入特征做**标准化 (Standardization)**，使所有特征处于相同的数值尺度：

$$
\tilde{x}_j = \frac{x_j - \mu_j}{\sigma_j}
$$

其中 $\mu_j$ 和 $\sigma_j$ 分别是特征 $j$ 在训练集上的**均值**和**标准差**。

归一化后，每个特征的均值为 $0$，标准差为 $1$，误差曲面在各个方向上的曲率更加均匀。

### 2.2 仅在输入层归一化不够

即使对网络输入做了归一化，经过隐藏层的线性变换 + 激活函数后，中间层的激活值分布可能再次变得不均匀。因此需要在**每一层之后**都做归一化。

---

## 3. 批归一化机制 (Batch Normalization)

### 3.1 为什么按批次做？

理论上应该在**整个训练集**上统计每层的均值和方差。但训练数据量巨大时，GPU 显存无法一次加载全部数据。

**解决方案：** 就像用小批量梯度下降一样，在每一个**批次 (Batch)** 内统计均值和方差：

$$
\begin{aligned}
\mu_{\mathcal{B}} &= \frac{1}{B} \sum_{i=1}^{B} \mathbf{z}^{(i)} \\[4pt]
\sigma_{\mathcal{B}}^{2} &= \frac{1}{B} \sum_{i=1}^{B} (\mathbf{z}^{(i)} - \mu_{\mathcal{B}})^{2}
\end{aligned}
$$

其中 $\mathbf{z}^{(i)}$ 是当前层对批次中第 $i$ 个样本的激活前输出。

### 3.2 归一化

对每个神经元的输出做标准化：

$$
\hat{\mathbf{z}}^{(i)} = \frac{\mathbf{z}^{(i)} - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^{2} + \epsilon}}
$$

$\epsilon$ 是一个极小常数（如 $10^{-5}$），防止除零。

### 3.3 可学习的缩放与偏移

仅仅标准化为 $N(0, 1)$ 可能会限制模型的表达能力。因此 BN 引入两个**可学习的参数**：

$$
\mathbf{y}^{(i)} = \gamma \odot \hat{\mathbf{z}}^{(i)} + \beta
$$

|    参数    | 作用         | 含义             |
| :------: | ---------- | -------------- |
| $\gamma$ | 恢复或调整输出的方差 | **缩放 (Scale)** |
| $\beta$  | 恢复或调整输出的均值 | **偏移 (Shift)** |

- 如果 $\gamma = \sigma_{\mathcal{B}}$，$\beta = \mu_{\mathcal{B}}$，则 BN 退化为恒等映射
- 这两个参数让网络在需要时可以**完全恢复原始分布**而不损失表达能力
- $\gamma$ 和 $\beta$ 与 $\mathbf{W}$、$\mathbf{b}$ 一样通过反向传播学习

### 3.4 完整流程

```
输入 batch: z^(1), z^(2), ..., z^(B)
          ↓
   计算 μ_B, σ²_B
          ↓
   归一化: ẑ = (z - μ_B) / √(σ²_B + ε)
          ↓
   缩放偏移: y = γ · ẑ + β
          ↓
   送入激活函数: a = f(y)
```

---

## 4. 训练与推理的差异

### 4.1 训练阶段

每个 Batch 独立计算 $\mu_{\mathcal{B}}$ 和 $\sigma_{\mathcal{B}}^{2}$，按上述流程完成 BN。

### 4.2 推理阶段

推理时没有 Batch 的概念（可能只有一张图）。因此 BN 使用训练期间累积的**全局统计量**：

- **运行均值 (Running Mean)**：$\mu_{\text{running}}$ — 训练中所有 Batch 均值的指数移动平均
- **运行方差 (Running Variance)**：$\sigma^{2}_{\text{running}}$ — 训练中所有 Batch 方差的指数移动平均

推理时的 BN 变为：

$$
\hat{\mathbf{z}} = \frac{\mathbf{z} - \mu_{\text{running}}}{\sqrt{\sigma^{2}_{\text{running}} + \epsilon}}, \quad \mathbf{y} = \gamma \hat{\mathbf{z}} + \beta
$$

> ⚠️ **坑点：** 训练/推理的 BN 行为不同，忘记切换模式会导致性能骤降。PyTorch 中用 `model.train()` / `model.eval()` 自动处理。

---

## 5. 内部协变量偏移 (Internal Covariate Shift)

### 5.1 理论解释

BN 原作者（Ioffe & Szegedy, 2015）提出的动机是解决**内部协变量偏移 (Internal Covariate Shift)**：

> 网络训练过程中，前面层参数更新会改变后面层输入的分布。后面的层不断需要适应新的输入分布，导致训练变慢。

BN 通过强制每层输入分布稳定（均值 $0$、方差 $1$），减少了层间分布的漂移，从而加速训练。

### 5.2 后续研究的不同观点

近年研究发现 BN 的真正收益可能更多来自：

- **平滑优化景观 (Smooth Optimization Landscape)**：BN 使损失函数曲面更平滑，梯度更可预测
- **允许更大的学习率**：平滑曲面下大学习率也不易发散
- **轻微的隐式正则化**：Batch 统计量的噪声有类似 Dropout 的效果

---

## 6. BN 的实际效果

| 效果 | 说明 |
|------|------|
| **加速收敛** | 可以用大学习率，减少训练 Epoch |
| **降低对初始化的敏感度** | 即使初始权重不太理想，BN 也能稳定训练 |
| **缓解梯度消失/爆炸** | 激活值被归一化，不会落入 Sigmoid/Tanh 的饱和区 |
| **轻微正则化** | Batch 噪声引入随机性，减少对 Dropout 的依赖 |

---

## 7. 其他归一化方法

BN 不是唯一的归一化选择。根据归一化维度不同，有不同的变体：

| 方法 | 归一化维度 | 适用场景 | 特点 |
|------|------|------|------|
| **Batch Norm** | 沿 Batch 维度 | CNN、大批次训练 | 依赖 Batch Size，小 Batch 时不稳定 |
| **Layer Norm** | 沿 Feature 维度（每个样本独立） | Transformer、RNN、NLP | 不依赖 Batch Size，序列模型首选 |
| **Instance Norm** | 沿空间维度（每个样本每个通道） | 图像风格迁移 | 保留个体样本的风格差异 |
| **Group Norm** | 通道分组归一化 | 小 Batch 的 CNN | BN 的小 Batch 替代品 |

```
Batch Norm:    对每个神经元，在一个 Batch 内归一化
Layer Norm:    对每个样本，在所有神经元上归一化
Instance Norm: 对每个样本的每个通道，在空间上归一化
Group Norm:    对每个样本，在一组通道上归一化
```

> 💡 **Transformer 为什么用 Layer Norm 而不用 Batch Norm？** 序列长度可变，不同位置的 Batch 统计不稳定，Layer Norm 每个样本独立归一化更鲁棒。

---

## 8. 关键结论 (Key Takeaways)

- **归一化使误差曲面平滑**，训练更稳定、收敛更快
- **BN = Batch 内标准化 + 可学习缩放偏移**：$\mathbf{y} = \gamma \cdot (\mathbf{z} - \mu_{\mathcal{B}}) / \sigma_{\mathcal{B}} + \beta$
- **$\gamma$ 和 $\beta$ 是关键**：让网络在需要时可以恢复原始分布，不损失表达力
- **训练 vs 推理**：训练用 Batch 统计，推理用全局 Running Mean/Variance
- **BN 适合 CNN，LN 适合 Transformer**：选归一化方法要看架构和数据特性

---

## 9. 相关链接 (Related)

- [[入门 Guide]] — 深度学习入门
- [[优化 Optimization]] — 梯度下降、学习率、优化器
- [[神经网络 Neural Network]] — 层、前向传播
- [[深度学习 Deep Learning]]
- [[过拟合 Overfitting]] — BN 的隐式正则化效果
