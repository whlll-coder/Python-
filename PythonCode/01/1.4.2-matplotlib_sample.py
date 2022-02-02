#为了设定各方程式导入Numpy
import numpy as np
# 通过plt导入matplotlib的pyplot
import matplotlib.pyplot as plt

# 设定x轴的范围和精度，准备x值
x = np.arange(-3, 3, 0.1)

#准备各方程式的y值
y_sin = np.sin(x)
x_rand = np.random.rand(100) * 6 - 3
y_rand = np.random.rand(100) * 6 - 3

# 创建figure对象
plt.figure()

# 设定为用1个图表表示
plt.subplot(1, 1, 1)

#设定各方程式的线性和标记，标签，plot
##线形图
plt.plot(x, y_sin, marker='o', markersize=5, label='line')

##散点图
plt.scatter(x_rand, y_rand, label='scatter')

#显示图例
plt.legend()
#显示网格线
plt.grid(True)

#显示图表
plt.show()
