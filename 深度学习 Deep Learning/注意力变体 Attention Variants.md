# 注意力机制变体 (Attention Variants)

标准 [[自注意力 Self-Attention]] 的计算复杂度为 $O(n^2)$（$n$ 为序列长度）。当序列极长（如高分辨率图像、长文档）时，Attention 成为计算瓶颈。此外，Attention 层可能出现在隐藏层中，导致输入长度 $n$ 激增。

本笔记梳理了各类降低复杂度或改进机制的注意力变体。

---

## 1. 稀疏注意力 (Sparse Attention)

核心思想：**不是所有 Token 对都需要计算注意力**——有选择地只计算重要的 Token 对。

### 1.1 局部注意力 (Local / Truncated Attention)

每个 Token 只计算与相邻一段窗口内 Token 的注意力：

$$
\alpha_{i,j} = 0 \quad \text{if } |i - j| > w
$$

- 复杂度从 $O(n^2)$ 降至 $O(n \cdot w)$
- 适合**语音识别**（相邻帧高度相关）
- 本质与 CNN 类似——感受野从学习变成了手工设定

### 1.2 全局注意力 (Global Attention)

在序列中加入**特殊 Token**，这些 Token 与所有位置互相计算注意力：

- **方式一**：标记原序列中的某些 Token 为全局 Token
- **方式二**：新增可学习的特殊 Token 作为序列的「浓缩摘要」

全局 Token 负责收集和分发全局信息，普通 Token 保持局部注意力。

### 1.3 随机注意力 (Random Attention)

随机采样若干 Token 对计算注意力，其余不计算。简单粗暴，但在某些场景下出人意料地有效。

### 1.4 聚类注意力 (Clustering Attention)

将 Token 分组（聚类），只计算**同一 Cluster 内** Token 之间的注意力：

- 一种方式是使用**可学习的路由网络**（如 Sinkhorn Sorting Network）来判断哪些 Token 对需要计算注意力
- 某个 Attention Score 是否被计算，本身可以是一个**可学习的决策**

---

## 2. 低秩近似：Linformer

### 2.1 核心发现

Linformer (2020) 发现：Self-Attention 的注意力矩阵**秩很低 (Low Rank)**——意味着有大量冗余空间可被压缩。

### 2.2 机制

将原来的 $n$ 个 Key 向量通过线性投影压缩为 $k$ 个（$k \ll n$）：

$$
\tilde{\mathbf{K}} = \mathbf{E} \mathbf{K}, \quad \tilde{\mathbf{V}} = \mathbf{E} \mathbf{V}
$$

其中 $\mathbf{E} \in \mathbb{R}^{k \times n}$ 是投影矩阵。

| 原 Attention | Linformer |
|------|------|
| $\text{Softmax}(\mathbf{Q}\mathbf{K}^{T})\mathbf{V}$，形状 $n \times n$ | $\text{Softmax}(\mathbf{Q}\tilde{\mathbf{K}}^{T})\tilde{\mathbf{V}}$，形状 $n \times k$ |
| 复杂度 $O(n^2)$ | 复杂度 $O(nk)$ |

### 2.3 Compressed Attention（卷积版 Linformer）

使用**卷积层 (CNN)** 代替线性投影来压缩 Key 序列，称为 **Convolutional Linformer**。CNN 的局部归纳偏置使压缩保留了空间连续性。

---

## 3. 线性注意力 (Linear Attention)

### 3.1 核心技巧：改变计算顺序

标准 Self-Attention：

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}(\mathbf{Q}\mathbf{K}^{T}) \mathbf{V}
$$

计算 $\mathbf{Q}\mathbf{K}^{T}$ 的复杂度为 $O(n^2)$。

Linear Attention 的 insight：**改变乘法顺序**。如果能将 Softmax 分解为 $\phi(\mathbf{Q})\phi(\mathbf{K})^{T}$，则：

$$
\text{LinearAttn} = \phi(\mathbf{Q}) \cdot (\phi(\mathbf{K})^{T} \mathbf{V})
$$

先算 $\phi(\mathbf{K})^{T}\mathbf{V}$（大小 $d \times d$），再算 $\phi(\mathbf{Q})$ 乘它：

| 计算顺序 | 复杂度 |
|------|:--:|
| $(\mathbf{Q}\mathbf{K}^{T})\mathbf{V}$ | $O(n^2 d)$ |
| $\phi(\mathbf{Q})(\phi(\mathbf{K})^{T}\mathbf{V})$ | $O(nd^2)$ |

当 $d \ll n$ 时（通常如此），Linear Attention 将复杂度从**平方级降到线性级**。

### 3.2 关键挑战

如何找到合适的 $\phi$ 函数，使得 $\phi(\mathbf{Q})\phi(\mathbf{K})^{T}$ 能够近似 Softmax 的效果——这是 Linear Attention 研究的核心问题。

---

## 4. Synthesizer：学习生成注意力矩阵

### 4.1 动机

每次通过 $\mathbf{Q}\mathbf{K}^{T}$ 点积计算注意力矩阵既**耗时**，又**不一定必要**。

### 4.2 机制

Synthesizer 用**可学习的神经网络**直接生成注意力矩阵，跳过了 Q·K 点积：

- **Dense Synthesizer**：用全连接网络从输入生成注意力矩阵
- **Random Synthesizer**：注意力矩阵是**可训练的参数**（不依赖输入），类似随机初始化的固定注意力模式

核心洞察：注意力矩阵的质量不一定依赖于 Q·K 交互——有时直接学习注意力模式也能达到类似效果。

---

## 5. 潜在注意力 (Latent Attention)

### 5.1 动机

前面各类方法的核心困境在于：注意力矩阵本身的大小是 $n \times n$，无论如何优化计算方式，总需要某种形式的 Token 间交互。能不能**从根本上消除 Token 间的直接交互**？

**潜在注意力 (Latent Attention)** 的答案是：引入一个固定大小的**潜在瓶颈 (Latent Bottleneck)**，让所有 Token 通过这个瓶颈间接交互。

### 5.2 机制

引入 $m$ 个**潜在向量 (Latent Vectors)**，$m \ll n$，作为信息交换的中介：

$$
\begin{aligned}
\text{第 1 步（压缩）：} &\quad \tilde{\mathbf{V}} = \text{CrossAttn}(\text{Latents} \to \text{Input}) \quad \text{输入序列压缩进 } m \text{ 个向量} \\
\text{第 2 步（自注意）：} &\quad \tilde{\mathbf{V}} = \text{SelfAttn}(\text{Latents}) \quad \text{潜在向量之间交互} \\
\text{第 3 步（解压）：} &\quad \text{Output} = \text{CrossAttn}(\text{Input} \to \text{Latents}) \quad \text{将信息分发回每个 Token}
\end{aligned}
$$

**核心思想：** Token 不再直接相互关注，而是各自与 $m$ 个潜在向量交互。就像所有 Token 把自己的信息写入一个固定大小的「公告板」，再从公告板上读取其他 Token 的信息。

### 5.3 复杂度

| 步骤 | 操作 | 复杂度 |
|------|------|:--:|
| Token → Latents | Cross-Attention | $O(n \cdot m)$ |
| Latents 自交互 | Self-Attention | $O(m^2)$ |
| Latents → Token | Cross-Attention | $O(n \cdot m)$ |
| **总计** | | $O(nm + m^2)$ |

当 $m \ll n$ 时，总复杂度近似线性 $O(n)$，而注意力矩阵从 $n \times n$ 压缩为 $m \times m$。

### 5.4 代表工作：DeepSeek MLA (Multi-head Latent Attention)

**MLA** 是 DeepSeek-V2 (2024) 提出的潜在注意力实现，其动机非常务实——**压缩 KV 缓存 (KV Cache)**。

**背景问题：** 在 LLM 推理时，每个已生成的 Token 的 Key 和 Value 向量需要缓存在 GPU 显存中（KV Cache），以便后续 Token 做 Self-Attention。序列越长，KV Cache 越大，很快就耗尽显存。

**MLA 的核心思想：** 不缓存完整的 Key 和 Value，而是缓存一个**低秩潜在表示 (Low-Rank Latent)**：

$$
\begin{aligned}
\mathbf{c}_{\text{KV}} &= \mathbf{W}_{\text{down}} \cdot \mathbf{h} \quad &\text{(输入 → 潜在向量，降维)} \\
\mathbf{k} &= \mathbf{W}_{\text{K}}^{\text{up}} \cdot \mathbf{c}_{\text{KV}} \quad &\text{(潜在 → Key，升维)} \\
\mathbf{v} &= \mathbf{W}_{\text{V}}^{\text{up}} \cdot \mathbf{c}_{\text{KV}} \quad &\text{(潜在 → Value，升维)}
\end{aligned}
$$

其中 $\mathbf{h}$ 是当前 Token 的隐藏状态，$\mathbf{c}_{\text{KV}}$ 是压缩后的潜在向量（维度远小于 $\mathbf{k}$ 和 $\mathbf{v}$）。

**为什么这能节省显存？**

| | 标准 MHA | DeepSeek MLA |
|------|:--:|:--:|
| **缓存内容** | 完整的 $\mathbf{k}$ 和 $\mathbf{v}$（$2 \times d \times n$） | 仅压缩的 $\mathbf{c}_{\text{KV}}$（$d_c \times n$，$d_c \ll 2d$） |
| **计算时** | 直接用缓存的 K、V | 从 $\mathbf{c}_{\text{KV}}$ 升维还原 K、V |
| **Query** | 同样做低秩压缩 | Q 也经压缩后升维（对称设计） |

**关键权衡：** 用少量的**额外计算**（升维矩阵乘法）换取**大幅 KV Cache 显存压缩**。在长序列推理中，显存往往是瓶颈而非计算——MLA 用计算换空间，性价比极高。

MLA 已成功部署在 **DeepSeek-V2 / V3** 中，支持超长上下文（128K+ tokens），是潜在注意力在工业级 LLM 中落地的标杆案例。

### 5.5 与其他方法的对比

| 对比 | 稀疏注意力 | 潜在注意力 |
|------|------|------|
| **交互方式** | Token 间直接交互（部分） | Token 通过瓶颈间接交互 |
| **注意力矩阵** | 稀疏 $n \times n$ | 压缩为 $m \times m$，$m \ll n$ |
| **信息流** | 局部为主 + 少量全局 | 全局压缩后统一交互 |
| **长序列** | 需精心设计稀疏模式 | 天然适合（瓶颈固定大小） |

---

## 6. 各变体总览

| 方法                    | 核心思想              |      复杂度       | 解决的问题         |
| --------------------- | ----------------- | :------------: | ------------- |
| **Local / Truncated** | 只算窗口内注意力          | $O(n \cdot w)$ | 长序列计算量        |
| **Global Attention**  | 特殊 Token 收集全局信息   | $O(n^2)$ (少量)  | 局部注意力的全局信息缺失  |
| **Clustering**        | Token 分组，组内计算     | $O(n \cdot c)$ | 长序列 + 保持相关性   |
| **Linformer**         | 低秩压缩 Key/Value    |    $O(nk)$     | 注意力矩阵冗余       |
| **Linear Attention**  | 改变乘法顺序            |     $O(n)$     | $O(n^2)$ 瓶颈   |
| **Synthesizer**       | 神经网络直接生成注意力       |     取决于实现      | Q·K 点积的计算开销   |
| **Latent Attention**  | 低秩潜在瓶颈压缩 KV Cache | $O(nm + m^2)$  | KV Cache 显存瓶颈 |

---

## 7. 关键结论 (Key Takeaways)

- 标准 Self-Attention 的 $O(n^2)$ 是长序列处理的根本瓶颈
- **稀疏化**（局部、聚类、随机）是最直接的降低复杂度方式
- **Linformer** 利用低秩特性压缩，实现 $O(nk)$
- **Linear Attention** 改变计算顺序，实现真正的 $O(n)$——关键在 $\phi$ 函数设计
- **Synthesizer** 提出激进观点：注意力不一定需要 Q·K 点积
- **潜在注意力**从根本上消除 Token 间直接交互，用固定大小瓶颈实现 $O(nm + m^2)$，Perceiver 是代表性实践

---

## 8. 相关链接 (Related)

- [[自注意力 Self-Attention]] — 标准 Self-Attention 完整机制
- [[Transformer]] — Encoder/Decoder 架构
- [[卷积神经网络 CNN]]
- [[深度学习 Deep Learning]]
