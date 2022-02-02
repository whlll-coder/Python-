# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np


### 创建带有偏差的 y = 3x^2 -2 的数据

x = np.random.rand(100, 1)  # 制作100个0到1的随机数
x = x * 4 - 2               # 值的范围变更为-2～2

y = 3 * x**2 - 2  # y = 3x^2 - 2

y += np.random.randn(100, 1)  # 加上标准正态分布（均值0、标准偏差1）的随机数


### 学习

from sklearn import linear_model


model = linear_model.LinearRegression()
model.fit(x**2, y)  # 把x平方之后给出


### 显示系统、截距和决定系数

print('系数', model.coef_)
print('截距', model.intercept_)

print('决定系数', model.score(x**2, y))


### 显示图表

plt.scatter(x, y, marker ='+')
plt.scatter(x, model.predict(x**2), marker='o')  # 将x平方交给predict
plt.show()
