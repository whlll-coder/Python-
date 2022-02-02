# -*- coding: utf-8 -*-

import math

import numpy as np
import matplotlib.pyplot as plt


### 创建带有偏差的正弦波数据

x = np.random.rand(1000, 1)  # 做出1000个0-1之间的随机数
x = x * 20 - 10              # 值的范围变更为-10～10

y = np.array([math.sin(v) for v in x])  # 正弦波曲线
y += np.random.randn(1000)  # 加上标准正态分布（均值0、标准偏差1）的随机数


### 学习: Random Forest

from sklearn import ensemble


model = ensemble.RandomForestRegressor()
model.fit(x, y)


### 显示决定系数

r2 = model.score(x, y)
print('决定系数', r2)


### 显示图表

plt.scatter(x, y, marker='+')
plt.scatter(x, model.predict(x), marker='o')
plt.show()
