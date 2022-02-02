# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np


### 创建带有偏差的 y = 3x_1 - 2x_2 + 1 的数据

x1 = np.random.rand(100, 1)  # 制作100个0到1的随机数
x1 = x1 * 4 - 2              # 值的范围变更为-2～2

x2 = np.random.rand(100, 1)  #  关于x2也一样
x2 = x2 * 4 - 2

y = 3 * x1 - 2 * x2 + 1

y += np.random.randn(100, 1)  # 加上标准正态分布（均值0、标准偏差1）的随机数


### 学习

from sklearn import linear_model


x1_x2 = np.c_[x1, x2]  # 转换为[[x1_1, x2_1], [x1_2, x2_2], ..., [x1_100, x2_100]]的形式
                       

model = linear_model.LinearRegression()
model.fit(x1_x2, y)


### 显示系数、截距和决定系数

print('系数', model.coef_)
print('截距', model.intercept_)

print('决定系数', model.score(x1_x2, y))


### 显示图表

y_ = model.predict(x1_x2)  # 用求出的回归式进行预测

plt.subplot(1, 2, 1)
plt.scatter(x1, y, marker='+')
plt.scatter(x1, y_, marker='o')
plt.xlabel('x1')
plt.ylabel('y')

plt.subplot(1, 2, 2)
plt.scatter(x2, y, marker='+')
plt.scatter(x2, y_, marker='o')
plt.xlabel('x2')
plt.ylabel('y')

plt.tight_layout()
plt.show()
