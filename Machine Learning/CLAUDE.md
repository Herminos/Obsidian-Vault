# CLAUDE.md — 李宏毅深度学习 2026 笔记整理规范

## 项目概述

本目录是学习李宏毅 (Hung-yi Lee) 2026 年深度学习课程的 Obsidian 笔记库。笔记以中文为主，使用 Obsidian 双向链接组织知识图谱。

## 笔记整理规范

### 1. 语言规范

- **正文使用中文**书写，确保语句通顺、逻辑清晰。
- **专有名词首次出现时**，在中文译名后用圆括号标注英文原文，格式：
  - `梯度下降 (Gradient Descent)`
  - `损失函数 (Loss Function)`
  - `学习率 (Learning Rate)`
- 后续同一笔记中再次出现该术语时，可直接使用中文或英文缩写。

### 2. 公式排版（严格 LaTeX）

- **行内公式**使用 `$...$`，如 `$y = wx + b$`
- **独立公式块**使用 `$$...$$`，如：
  $$
  L(w,b) = \frac{1}{N} \sum_{n=1}^{N} e_n
  $$
- 多字符下标用花括号包裹：`x_{1}` 而非 `x_1`
- 使用 `\hat{y}` 表示预测值，`\bar{y}` 表示均值
- 偏导数使用 `\frac{\partial L}{\partial w}`
- 花体/特殊字体：`\mathcal{L}`, `\mathbb{E}`
- 常见符号对照：
  - `\eta` — 学习率
  - `\theta` — 参数集合
  - `\nabla` — 梯度算子
  - `\sum` / `\prod` — 求和 / 求积

### 3. Obsidian 双向链接

- 使用 `[[笔记名]]` 创建到其他笔记的链接
- 使用 `[[笔记名#标题]]` 链接到特定章节
- 使用 `[[笔记名|显示文本]]` 自定义链接显示文字
- 核心概念应建立独立笔记并互相链接，例如：
  - `[[梯度下降 Gradient Descent]]`
  - `[[损失函数 Loss Function]]`
  - `[[线性模型 Linear Model]]`
  - `[[过拟合 Overfitting]]`

### 4. 笔记结构模板

每节课的笔记建议如下结构：

```markdown
# 课程标题 (Lecture Title)

## 1. 核心概念 (Core Concepts)

概念解释，专有名词标注英文。

## 2. 数学推导 (Mathematical Derivation)

$$
公式块
$$

## 3. 关键结论 (Key Takeaways)

- 要点列表
- ...

## 4. 相关链接 (Related)

- [[相关概念1]]
- [[相关概念2]]
```

### 5. 行内指令标记 (For Claude)
 
笔记中可能出现以 `(For Claude)` 或 `(TODO)` 开头的行内指令，这些是对我的直接指示。**遇到此类标记时，必须立即执行后面的指令**，而非仅将其视为普通文本保留。

常见指令类型：
- `(For Claude) 补全 xxx` → 根据上下文补全缺失的推导、定义或公式
- `(For Claude) 展开 xxx` → 将简略写出但未详细推导的内容展开完整
- `(For Claude) 修正 xxx` → 检查并修正某处的错误
- `(For Claude) 添加 xxx 的图示说明` → 用文字补充可视化描述
- `(For Claude) 补充例子` → 为该概念添加具体的数值例子

执行原则：
- 指令完成的内容应**融入笔记正文**，与原有风格一致（中文 + 英文注释 + LaTeX）
- 执行完毕后，可以**删除或注释掉**该 `(For Claude)` 标记行，表示任务已完成
- 若指令不明确，应在执行前向用户确认意图

### 6. 已有笔记索引

当前已有笔记：
- `Guide.md` — 课程入门：机器学习基本概念、损失函数、梯度下降、线性模型
- `ml.md` — 训练流程与诊断：过拟合、模型偏差、交叉验证、不匹配
- `Concepts/` — 概念详解目录：
  - `过拟合 Overfitting.md`
  - `正则化 Regularization.md`
  - `模型偏差 Model Bias.md`
  - `梯度下降 Gradient Descent.md`
  - `损失函数 Loss Function.md`
  - `线性模型 Linear Model.md`
  - `激活函数 Activation Function.md`
  - `神经网络 Neural Network.md`
  - `深度学习 Deep Learning.md`

### 7. 整理流程

当用户请求整理笔记时：
1. 读取目标笔记文件
2. 将混杂的中英文统一为中文叙述 + 专有名词英文注释
3. 检查并修正所有公式的 LaTeX 语法
4. 识别核心概念，建议或创建双向链接 `[[...]]`
5. 保持原有的数学推导逻辑不变，仅优化表达

### 8. 常用术语对照表

| 中文 | English |
|------|---------|
| 机器学习 | Machine Learning |
| 深度学习 | Deep Learning |
| 回归 | Regression |
| 分类 | Classification |
| 结构化学习 | Structured Learning |
| 模型 | Model |
| 参数 | Parameter |
| 特征 | Feature |
| 权重 | Weight |
| 偏置 | Bias |
| 损失函数 | Loss Function |
| 梯度下降 | Gradient Descent |
| 学习率 | Learning Rate |
| 超参数 | Hyperparameter |
| 局部最小值 | Local Minima |
| 全局最小值 | Global Minima |
| 训练 | Training |
| 线性模型 | Linear Model |
| 模型偏差 | Model Bias |
| 过拟合 | Overfitting |
| 欠拟合 | Underfitting |
| 验证集 | Validation Set |
| 测试集 | Test Set |
| 激活函数 | Activation Function |
| 反向传播 | Backpropagation |
| 神经网络 | Neural Network |
| 卷积神经网络 | Convolutional Neural Network (CNN) |
| 循环神经网络 | Recurrent Neural Network (RNN) |
|  transformer | Transformer |
| 自注意力 | Self-Attention |
| 预训练 | Pre-training |
| 微调 | Fine-tuning |
| 批量 | Batch |
| 轮次 | Epoch |
| 正则化 | Regularization |
| 归一化 | Normalization |
