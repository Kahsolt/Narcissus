---
title: 最小生成集问题
date: 2019-10-05 10:04:29
updated:
categories: 解命题
tags:
  - 算法
  - 代数结构
---

<br/>

# 起源

七段数码管是这样的东西……

# 描述

给定数字0~9的二进制表示集合`N`，给定标准算子集合`OP = { NOT, OR, AND, XOR }`，求`N`在`OP`下的最小生成集。

扩展问题：给定任意n个数码管字符的二进制表示集合`N = {N1, N2, ..., Nn}`，给定k个合法算子的集合`OP = { OP1, OP2, ..., OPk}`，求`N`在`OP`下的最小生成集。

# 暴力解

{% include_code lang:python MGS_bf.py %}
