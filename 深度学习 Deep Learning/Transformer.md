# Transformer

**Transformer** 是 2017 年由 Vaswani 等人在论文 *"Attention is All You Need"* 中提出的架构，彻底改变了深度学习——尤其是在**自然语言处理 (NLP)** 领域。其核心思想是用**自注意力 (Self-Attention)** 替代 RNN 的串行计算，实现高度并行化。

---

## 1. 序列到序列 (Sequence-to-Sequence, Seq2Seq)

Transformer 是一种 **Seq2Seq** 模型：输入一个序列，输出另一个序列。应用非常广泛：

| 任务 | 输入 | 输出 |
|------|------|------|
| **机器翻译** | 英文句子 | 中文句子 |
| **语音辨识** | 语音信号 | 文字 |
| **语音翻译** | 外语语音 | 中文文字 |
| **聊天机器人 (Chatbot)** | 用户消息 | 回复 |
| **句法分析 (Syntactic Parsing)** | 英文句子 | 树状语法结构 |

大多数 NLP 任务都可以用 Transformer 或其变体解决。

---

## 2. Encoder（编码器）

### 2.1 职责

Encoder 的职责：**输入一个向量序列，输出一个等长的、融合了全局上下文信息的向量序列**。

### 2.2 核心组件

Encoder 的一个 Block 包含：

```
输入 → Self-Attention → Add & Norm → Feed-Forward → Add & Norm → 输出
```

具体拆解：

| 组件 | 作用 |
|------|------|
| **Multi-Head Self-Attention** | 让每个向量关注整个序列的所有向量 |
| **残差连接 (Residual Connection)** | 将输入直接加到输出上：$\text{Output} = \text{Layer}(x) + x$，防止梯度消失 |
| **层归一化 (Layer Normalization)** | 对每个样本独立归一化，稳定训练 |
| **前馈网络 (Feed-Forward Network)** | 两层全连接 + 激活函数（通常是 ReLU），对每个位置的向量独立做非线性变换 |

### 2.3 Norm 的位置演变

原始论文：Norm **后置**（Post-Norm）

```
x → MHA → Add → Norm → FFN → Add → Norm
```

现代实践：Norm **前置**（Pre-Norm）

```
x → Norm → MHA → Add → Norm → FFN → Add
```

Pre-Norm 训练更稳定，梯度流动更顺畅，是现在的主流选择。

### 2.4 完整 Encoder

$$
\text{Encoder} = \text{Embedding} + \text{Positional Encoding} + N \times (\text{MHA} + \text{Add} + \text{Norm} + \text{FFN} + \text{Add} + \text{Norm})
$$

$N$ 层 Encoder Block 堆叠，逐层提取更深层的语义表示。

---

## 3. Decoder（解码器）

### 3.1 职责

Decoder 的职责是**生成**输出序列，而非仅仅分析。它与 Encoder 的核心区别在于它是**自回归 (Autoregressive)** 的。

### 3.2 自回归生成 (Autoregressive, AT)

Decoder 一次只输出一个 Token：

1. 输入一个起始符号 `BEGIN`
2. 输出第一个 Token
3. 将 `BEGIN + Token1` 作为新输入
4. 输出第二个 Token
5. 重复，直到输出终止符号 `END`

像一个**推文接龙**游戏——每次基于已有的输出，预测下一个词。

### 3.3 掩码多头注意力 (Masked MHA)

Decoder 的 Self-Attention 是**带掩码的 (Masked)**。在生成位置 $i$ 的输出向量 $\mathbf{b}_i$ 时：

- 只能看到位置 $1, 2, \dots, i$ 的信息
- 位置 $i+1, i+2, \dots$ 的 Attention Score 被设为 $-\infty$（Softmax 后为 $0$）

$$
\alpha'_{i,j} =
\begin{cases}
\frac{\exp(\alpha_{i,j})}{\sum_{t=1}^{i} \exp(\alpha_{i,t})} & j \leq i \\[6pt]
0 & j > i
\end{cases}
$$

这确保了**生成过程是因果的 (Causal)**——当前输出不能「偷看」未来的信息。

### 3.4 Encoder 与 Decoder 的对比

| | Encoder | Decoder |
|------|------|------|
| **角色** | 「阅读并理解」输入 | 「生成」输出 |
| **Self-Attention** | 无掩码，双向 | **掩码 (Masked)**，单向（因果） |
| **Cross-Attention** | 无 | **有**：Q 来自 Decoder，K、V 来自 Encoder |
| **输出** | 对输入序列的编码表示 | 自回归生成的目标序列 |
| **并行性** | 完全并行 | 训练时并行（Teacher Forcing），推理时串行 |

### 3.5 非自回归解码器 (Non-Autoregressive, NAT)

**NAT Decoder** 输入一个序列，一次性地并行输出完整输出序列。

| | AT Decoder | NAT Decoder |
|------|:--:|:--:|
| **输出方式** | 逐 Token 串行生成 | 一次性并行生成 |
| **速度** | 慢（串行） | 快（并行） |
| **性能** | ✅ 更好 | ⚠️ 通常不如 AT |
| **可控性** | 难以控制输出长度 | 输出长度需提前指定 |

NAT 适用于对速度要求极高的场景（但性能有损失）。

---

## 4. Cross-Attention：Encoder 与 Decoder 的桥梁

### 4.1 机制

Decoder 的某一层中插入了 **Cross-Attention**（或称 Encoder-Decoder Attention），让 Decoder 在生成时「回顾」Encoder 的输出：

$$
\begin{aligned}
\mathbf{Q} &\leftarrow \text{Decoder 的当前表示} \\
\mathbf{K}, \mathbf{V} &\leftarrow \text{Encoder 的最终输出}
\end{aligned}
$$

### 4.2 直觉理解

在生成翻译结果时，Decoder 每输出一个词，Cross-Attention 让它回头看一眼原文中与当前词最相关的部分——就像翻译员翻译时不断参考原文。

> **类比：** Encoder 写了一份「输入序列的大纲」，Decoder 在写输出时不断查阅这份大纲，确保不偏题。

---

## 5. 训练

### 5.1 Encoder-Decoder 架构

训练时将 Encoder 的输入和 Decoder 的正确输出同时喂给模型，用 **Teacher Forcing** 训练下一个 Token 的预测。

### 5.2 Decoder-Only 架构 (GPT 系列)

只有 Decoder，没有独立的 Encoder。给定前文作为输入，用 Masked Self-Attention 预测下一个 Token。GPT、LLaMA 等现代 LLM 均采用此架构。

| 架构 | 代表模型 | 适用场景 |
|------|------|------|
| **Encoder-Decoder** | 原始 Transformer、T5、BART | 翻译、摘要等 Seq2Seq 任务 |
| **Encoder-Only** | BERT | 文本理解、分类 |
| **Decoder-Only** | GPT、LLaMA | 文本生成、对话 |

---

## 6. 训练技巧

### 6.1 复制机制 (Copy Mechanism)

对于某些任务（如对话），模型需要**直接从输入中复制**某些内容，而非从参数中「回忆」。

**示例：** 用户说「我叫张三」，模型回复「你好张三！」——模型需要学会复制「张三」这个名字，而不是试图从训练参数中预测一个名字。

- 在**摘要 (Summarization)** 中，模型需要从原文中提取关键信息的精确措辞
- **引导注意力 (Guided Attention)** 强制模型关注输入的特定部分

### 6.2 束搜索 (Beam Search)

Decoder 生成时，每一步有 $V$（词表大小）个候选 Token。**贪心解码 (Greedy Decoding)** 每步选最高分的 Token——但局部最优 ≠ 全局最优。

**束搜索 (Beam Search)** 每步保留 $k$ 条**最优的部分序列**（$k$ 为 Beam Size），而非仅保留一条：

| | 贪心解码 | Beam Search |
|------|:--:|:--:|
| **每步保留** | 1 条序列 | $k$ 条序列 |
| **全局最优** | 不一定 | 更接近 |
| **计算量** | 小 | 大（$k \times V$） |
| **多样性** | 差 | 可能趋于「安全」回答 |

> ⚠️ **随机性很重要：** 对于需要创造力和想象力的任务（如写诗、讲故事），分数最高的序列不一定是最好的。这时需要引入**采样 (Sampling)** 和**温度 (Temperature)** 来控制随机性。

### 6.3 计划采样 (Scheduled Sampling)

**问题：** 训练时用的 Teacher Forcing（喂正确答案）和推理时用的自回归生成（喂自己的输出）存在**训练-推理不一致 (Exposure Bias)**。

**Scheduled Sampling** 在训练过程中逐渐用模型自己的预测替代正确答案输入，缓解这种不一致性，使模型在推理时更鲁棒。

---

## 7. 关键结论 (Key Takeaways)

- **Transformer = Self-Attention + FFN + Residual + Layer Norm** 的堆叠
- **Encoder** 双向理解输入，**Decoder** 单向自回归生成
- **Masked MHA** 保证因果性——生成时不偷看未来
- **Cross-Attention** 是 Encoder → Decoder 的信息桥梁
- **Decoder-Only** (GPT 等) 是现代 LLM 的主流架构
- **Pre-Norm** 比 Post-Norm 训练更稳定，是当前默认选择
- **Beam Search** 和 **Scheduled Sampling** 是提升生成质量的关键技巧

---

## 8. 相关链接 (Related)

- [[自注意力 Self-Attention]] — Self-Attention 完整机制、Q/K/V、Multi-Head、位置编码
- [[Batch Normalization]] — 归一化（BN vs LN）
- [[神经网络 Neural Network]] — 前馈网络、残差连接
- [[深度学习 Deep Learning]]
- [[优化 Optimization]]
