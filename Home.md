---
tags:
  - moc
aliases:
  - 首页
  - Home
created: 2026-07-20
---

# 🏠 知识库首页

欢迎来到我的 Obsidian 知识库。这里目前主要收录**李宏毅 2026 深度学习课程**的学习笔记。

## 离散数学

- [[离散数学/离散数学-第1章-集合与逻辑-知识总结]] — 集合运算、基数、命题逻辑与谓词逻辑
- [[离散数学/离散数学-第2章-映射-知识总结]] — 映射、抽屉原理、置换、运算与有限自动机
- [[离散数学/离散数学-第3章-关系-知识总结]] — 关系性质、闭包、等价关系与偏序集

---

## 📖 机器学习笔记

> 笔记语言：中文 + 英文术语注释 | 课程：Hung-yi Lee 2026 Deep Learning

### 🗺️ 入门与总览

- [[深度学习 Deep Learning/入门 Guide|深度学习入门 (Guide)]] — 从回归到深度学习的基础概念全览
- [[深度学习 Deep Learning/训练流程与诊断 ml|训练诊断 (ML Diagnosis)]] — 模型表现不佳时的系统化排查流程
- [[深度学习 Deep Learning/优化 Optimization|优化诊断 (Optimization)]] — 临界点、鞍点、动量、自适应学习率

### 📚 概念详解

| 概念 | 说明 |
|------|------|
| [[深度学习 Deep Learning/Concepts/线性模型 Linear Model\|线性模型]] | 最简单的 ML 模型，所有非线性模型的基石 |
| [[深度学习 Deep Learning/Concepts/损失函数 Loss Function\|损失函数]] | 衡量预测与真实值差距的函数 |
| [[深度学习 Deep Learning/Concepts/梯度下降 Gradient Descent\|梯度下降]] | 通过迭代寻找最优参数的优化算法 |
| [[深度学习 Deep Learning/Concepts/激活函数 Activation Function\|激活函数]] | 为神经网络引入非线性能力 |
| [[深度学习 Deep Learning/Concepts/神经网络 Neural Network\|神经网络]] | 模拟神经元连接的计算模型 |
| [[深度学习 Deep Learning/Concepts/深度学习 Deep Learning\|深度学习]] | 多层神经网络的训练与应用 |
| [[深度学习 Deep Learning/Concepts/模型偏差 Model Bias\|模型偏差]] | 模型容量不足 vs 统计偏差 |
| [[深度学习 Deep Learning/Concepts/过拟合 Overfitting\|过拟合]] | 训练集表现好但测试集表现差 |
| [[深度学习 Deep Learning/Concepts/正则化 Regularization\|正则化]] | 防止过拟合的核心技术 |

---

## 📋 快速导航

```dataview
TABLE file.cday AS "创建日期", tags AS "标签"
FROM "深度学习 Deep Learning"
WHERE file.name != "CLAUDE"
SORT file.name ASC
```

---

*最后更新：2026-07-20*
