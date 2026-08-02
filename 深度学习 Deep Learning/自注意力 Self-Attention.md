# 自注意力 (Self-Attention)

**自注意力 (Self-Attention)** 是 Transformer 架构的核心组件，解决了一个关键问题：**如何让模型理解变长序列中每个元素与整个序列的关系？**

---

## 1. 向量集合作为输入 (Vector Set as Input)

很多场景下，输入不是固定维度的向量，而是一个**变长的向量集合 (Vector Set)**。

### 1.1 从 One-Hot 到词嵌入

最初的表示方法：**独热编码 (One-Hot Vector)**——每个词表示为一个长度等于词汇表大小的向量，只有对应位置为 $1$。

**弊端：** 完全抛弃了词之间的语义关系。「猫」和「狗」的 One-Hot 向量距离与「猫」和「桌子」完全一样——模型无法感知语义相似性。

**词嵌入 (Word Embedding)** 解决了这个问题：每个词映射到一个低维稠密向量，语义相近的词向量距离也相近。

### 1.2 向量集合的多样性

向量集合不仅限于文本：

| 数据类型 | 向量化方式 |
|------|------|
| **文本** | 词嵌入 (Word Embedding) |
| **语音** | 每一帧的声学特征向量 |
| **图 (Graph)** | 每个节点的特征向量（人际关系、分子结构） |
| **图像** | 将图像切分为 Patch，每个 Patch 视为一个向量 |

---

## 2. 输出类型

根据任务不同，Self-Attention 之后可以有多种输出方式：

| 输出方式 | 说明 | 示例 |
|------|------|------|
| **每个向量一个 Label** | 序列标注 | 词性标注 (POS Tagging) |
| **整个序列一个 Label** | 序列分类 | 情感分析 |
| **输出另一个序列** | 序列到序列 (Seq2Seq) | 机器翻译、GPT 文本生成 |

---

## 3. 序列标注与上下文问题

### 3.1 全连接网络的局限

**序列标注 (Sequence Labeling)** 中，全连接网络独立处理每个输入向量——**完全无法理解上下文**。

例如：「我吃苹果的时候很甜」——处理「苹果」时，FC 网络不知道前面有「吃」、后面有「甜」，无法判断「苹果」是被吃的食物还是被画的苹果。

### 3.2 滑动窗口的局限

用滑动窗口（CNN 思路）可以捕获**局部**上下文，但窗口大小有限，仍然无法理解**整个序列**的全局依赖关系。

### 3.3 Self-Attention 的方案

Self-Attention 让序列中的**每一个向量都与整个序列的所有向量交互**，从而获得全局上下文理解。

---

## 4. Self-Attention 的核心机制

### 4.1 整体流程

```
输入序列 → Self-Attention → 融合全局信息的输出 → FC → Self-Attention → FC → ... → 最终输出
```

Self-Attention 和全连接层（FC）交替堆叠，每一层 Self-Attention 让每个向量「看一眼」整个序列。

### 4.2 Q、K、V：查询、键、值

Self-Attention 的核心是三个可学习的线性投影：

对于序列中的每个向量 $\mathbf{a}_i$，分别乘以三个参数矩阵得到：

$$
\begin{aligned}
\mathbf{q}_i &= \mathbf{W}^{Q} \mathbf{a}_i \quad &\text{(Query, 查询)} \\
\mathbf{k}_i &= \mathbf{W}^{K} \mathbf{a}_i \quad &\text{(Key, 键)} \\
\mathbf{v}_i &= \mathbf{W}^{V} \mathbf{a}_i \quad &\text{(Value, 值)}
\end{aligned}
$$

其中 $\mathbf{W}^{Q}, \mathbf{W}^{K}, \mathbf{W}^{V}$ 是**可训练的参数矩阵**，所有位置共享。

> 📐 **直觉类比**：数据库查询中，Query 表示「我要找什么」，Key 表示「我是什么」，通过匹配 Query 和 Key 的相似度决定从 Value 中取多少信息。

### 4.3 注意力分数的计算

向量 $\mathbf{a}_1$ 对 $\mathbf{a}_2$ 的**注意力分数 (Attention Score)** 计算方式：

**点积法 (Dot-Product)**（最常用）：

$$
\alpha_{1,2} = \mathbf{q}_1 \cdot \mathbf{k}_2 = \mathbf{q}_1^{T} \mathbf{k}_2
$$

**加性法 (Additive)**：

$$
\alpha_{1,2} = \mathbf{v}^{T} \tanh(\mathbf{W}_1 \mathbf{q}_1 + \mathbf{W}_2 \mathbf{k}_2)
$$

### 4.4 Softmax 归一化

对每个查询向量 $\mathbf{q}_i$，将其与所有 $\mathbf{k}_j$ 的注意力分数做 Softmax：

$$
\alpha'_{i,j} = \frac{\exp(\alpha_{i,j})}{\sum_{t=1}^{n} \exp(\alpha_{i,t})}
$$

得到归一化后的注意力权重 $\alpha'_{i,j}$，满足 $\sum_j \alpha'_{i,j} = 1$。

### 4.5 加权求和输出

向量 $\mathbf{a}_i$ 经过 Self-Attention 后的输出 $\mathbf{b}_i$：

$$
\mathbf{b}_i = \sum_{j=1}^{n} \alpha'_{i,j} \cdot \mathbf{v}_j
$$

每个位置的输出是所有位置 Value 的**加权和**——权重由 Query-Key 的匹配程度决定。

### 4.6 直观例子

> 「**我** 吃 **苹果** 的时候 很 **甜**」

处理「苹果」时：
- Query（苹果）与 Key（吃）匹配 → 高分
- Query（苹果）与 Key（甜）匹配 → 高分
- Query（苹果）与 Key（我/的时候/很）匹配 → 低分

模型通过「吃」和「甜」理解了苹果是被食用的，而非被画在纸上。

---

## 5. 矩阵形式

所有向量并行计算，Self-Attention 本质上就是**矩阵乘法**：

### 5.1 生成 Q、K、V

将输入序列的 $n$ 个向量拼成矩阵 $\mathbf{I} = [\mathbf{a}_1, \mathbf{a}_2, \dots, \mathbf{a}_n]$：

$$
\mathbf{Q} = \mathbf{W}^{Q} \mathbf{I}, \quad \mathbf{K} = \mathbf{W}^{K} \mathbf{I}, \quad \mathbf{V} = \mathbf{W}^{V} \mathbf{I}
$$

### 5.2 一步计算注意力

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}\!\left(\frac{\mathbf{Q} \mathbf{K}^{T}}{\sqrt{d_k}}\right) \mathbf{V}
$$

其中 $\sqrt{d_k}$ 是**缩放因子**（$d_k$ 为 Key 的维度），防止点积过大导致 Softmax 梯度消失。

### 5.3 计算流程

$$
\underbrace{\mathbf{I}}_{d \times n}
\xrightarrow{\mathbf{W}^{Q}, \mathbf{W}^{K}, \mathbf{W}^{V}}
\underbrace{\mathbf{Q}, \mathbf{K}, \mathbf{V}}_{d_k \times n}
\xrightarrow{\mathbf{Q}^{T}\mathbf{K}}
\underbrace{\text{Scores}}_{n \times n}
\xrightarrow{\text{Softmax}}
\underbrace{\text{Weights}}_{n \times n}
\xrightarrow{\times \mathbf{V}}
\underbrace{\text{Output}}_{d_v \times n}
$$

---

## 6. 多头自注意力 (Multi-Head Self-Attention)

### 6.1 动机

单头注意力只能从**一个角度**看序列。多头注意力让模型同时从**多个角度**（多个子空间）关注序列的不同方面。

### 6.2 机制

1. 输入分别通过 $h$ 组独立的 $\mathbf{W}^{Q}, \mathbf{W}^{K}, \mathbf{W}^{V}$ 参数矩阵
2. 每组独立计算 Self-Attention，得到 $h$ 个输出
3. 将所有头的输出**拼接 (Concat)** 后再做一次线性投影，恢复原始维度

$$
\begin{aligned}
\text{head}_i &= \text{Attention}(\mathbf{Q}_i, \mathbf{K}_i, \mathbf{V}_i) \\
\text{MultiHead} &= \mathbf{W}^{O} \cdot \text{Concat}(\text{head}_1, \dots, \text{head}_h)
\end{aligned}
$$

- 不同头可能学到不同模式：一个头关注语法关系，另一个关注语义关系
- 每个头的维度通常是 $d_{\text{model}} / h$

---

## 7. 位置编码 (Positional Encoding)

### 7.1 问题

原始的 Self-Attention **对位置没有感知**——打乱输入顺序，输出值不变（只是排列不同）。但序列的位置信息至关重要。

### 7.2 绝对位置编码 (Absolute Positional Encoding)

给每个位置的输入向量叠加一个**位置指纹 (Positional Encoding)**：

$$
\mathbf{a}_i^{\text{pos}} = \mathbf{a}_i + \mathbf{p}_i
$$

其中 $\mathbf{p}_i$ 是位置 $i$ 的编码向量（如正弦/余弦函数），确保不同位置有不同的表示。

### 7.3 相对位置编码 (Relative Positional Encoding)

不关心绝对位置，只关心**两个位置的相对距离**。在计算 Attention Score 时叠加位置偏置：

- **旋转位置编码 (RoPE)**：对 Q、K 向量按位置进行旋转，使点积结果自然包含位置差信息。现代 LLM（LLaMA 等）广泛使用

### 7.4 线性偏置 (Linear Bias / ALiBi)

最简单的方法：根据两个位置的距离**直接衰减**注意力分数——距离越远，分数越低，模型越不关注。

---

## 8. Self-Attention 在各领域的应用

### 8.1 语音 (Speech)

- 语音帧非常短，序列极长（10 秒语音 ≈ 1000 帧）
- 直接对整个序列做 Self-Attention 计算量太大（$O(n^2)$）
- **截断自注意力 (Truncated Self-Attention)**：限制每个向量只看附近的有限窗口

### 8.2 图像 (Image)

- 将图像切分为若干 **Patch**（如 $16 \times 16$ 像素块），每个 Patch 视作一个向量
- Self-Attention 让每个 Patch 关注所有其他 Patch（ViT 架构）

### 8.3 图 (Graph)

- 图的节点天然就是向量集合
- Self-Attention 相当于图神经网络 (GNN) 的一种形式：节点通过注意力机制聚合邻居信息

---

## 9. Self-Attention vs CNN vs RNN

| | Self-Attention | CNN | RNN |
|------|:--:|:--:|:--:|
| **感受野** | 全局，**机器自动学习** | 局部，**人工设定**（Kernel Size） | 逐步传递，距离远时信息衰减 |
| **并行计算** | ✅ 完全并行 | ✅ 并行 | ❌ 必须串行（每步依赖前一步） |
| **长距离依赖** | ✅ 一步直达 | ⚠️ 需要堆叠多层 | ❌ 需经过多步传递 |
| **数据效率** | 需要大量数据 | 数据效率高 | 中等 |
| **本质关系** | CNN 是 Self-Attention 的简化特例 | ← | — |

> 💡 **关键洞察：** CNN 本质上是**感受野受限的简化版 Self-Attention**——CNN 的感受野是人类手工指定的局部窗口，Self-Attention 的感受野是机器从数据中学出的全局关注。因此 CNN 在小数据集上更稳定（有强归纳偏置），Self-Attention 在大数据集上更强大（可学习任意关注模式）。

---

## 10. 关键结论 (Key Takeaways)

- **Self-Attention 让序列中每个元素关注所有元素**，解决 FC 无上下文、CNN 窗口有限的痛点
- **Q、K、V 机制**：Query 查、Key 匹配、Value 取信息，三条投影矩阵可学习
- **计算本质是矩阵乘法**：$\text{Softmax}(\mathbf{Q}\mathbf{K}^{T}/\sqrt{d_k})\mathbf{V}$，高度并行
- **Multi-Head** 让模型从多个子空间同时关注，丰富表示能力
- **位置编码**弥补 Self-Attention 对位置的天然盲区（绝对/相对/线性偏置）
- **CNN 是 Self-Attention 的简化版**：手工感受野 vs 学习感受野
- **Transformer = "Attention is all you need"**：用 Self-Attention + FC 堆叠替代 RNN

---

## 11. 相关链接 (Related)

- [[入门 Guide]] — 深度学习入门
- [[神经网络 Neural Network]]
- [[深度学习 Deep Learning]]
- [[卷积神经网络 CNN]] — CNN vs Self-Attention 对比
- [[优化 Optimization]]
