# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np


### 创建y = 3x -2 的数据

x = np.random.rand(100, 1)  # 制作100个0到1的随机数
x = x * 4 - 2               # 值的范围变更为-2～2

y = 3 * x - 2  # y = 3x - 2


### 学习

from sklearn import linear_model


model = linear_model.LinearRegression()
model.fit(x, y)


### 显示系数和截距

print('系数', model.coef_)
print('截距', model.intercept_)


### 显示图表

plt.scatter(x, y, marker='+')
plt.show()
