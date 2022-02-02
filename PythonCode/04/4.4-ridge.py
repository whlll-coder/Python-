# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np


### 创建带有偏差的y = 4x^3 - 3x^2 + 2x - 1 的数据

x = np.random.rand(100, 1)  # 做出100个0-1之间的随机数
x = x * 2 - 1               # 值的范围变更为-2～2

y = 4 * x**3 - 3 * x**2 + 2 * x - 1

y += np.random.randn(100, 1)  # 加上标准正态分布（均值0、标准偏差1）的随机数

# 学习数据30个
x_train = x[:30]
y_train = y[:30]

# 测试数据70个
x_test = x[70:]
y_test = y[70:]


### 使用Ridge作为9次式回归

from sklearn import linear_model

# 学习输入数据
X_TRAIN = np.c_[x_train**9, x_train**8, x_train**7, x_train**6, x_train**5,
                x_train**4, x_train**3, x_train**2, x_train]

model = linear_model.Ridge()
model.fit(X_TRAIN, y_train)


### 显示由系数、截距和决定系数决定的学习数据

print('系数（学习数据）', model.coef_)
print('截距（学习数据）', model.intercept_)

print('决定系数（学习数据）', model.score(X_TRAIN, y_train))


### 显示测试数据的决定系数

X_TEST = np.c_[x_test**9, x_test**8, x_test**7, x_test**6, x_test**5,
               x_test**4, x_test**3, x_test**2, x_test]

print('决定系数（测试数据）', model.score(X_TEST, y_test))


### 显示图表

plt.subplot(2, 2, 1)
plt.scatter(x, y, marker='+')
plt.title('all')

plt.subplot(2, 2, 2)
plt.scatter(x_train, y_train, marker='+')
plt.scatter(x_train, model.predict(X_TRAIN), marker='o')
plt.title('train')

plt.subplot(2, 2, 3)
plt.scatter(x_test, y_test, marker='+')
plt.scatter(x_test, model.predict(X_TEST), marker='o')
plt.title('test')

plt.tight_layout()
plt.show()
