# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn import datasets

# 加载iris数据
iris = datasets.load_iris()
data = iris["data"]

# 学习→生成簇
model = cluster.AffinityPropagation().fit(data)

# 取得学习结果的标签
labels = model.labels_

### 图表的绘制

# 因为簇数不同，记号用排列表示

markers = ["o", "^", "*","v", "+", "x", "d", "p", "s", "1", "2"]

# 数据定义
x_index = 2 
y_index = 3

data_x=data[:,x_index]
data_y=data[:,y_index]

x_label = iris["feature_names"][x_index]
y_label = iris["feature_names"][y_index]

# 绘制每个集群的散布图
for idx in range(labels.max() + 1):
    plt.scatter(data_x[labels==idx], data_y[labels==idx],
                c='black' ,alpha=0.3,s=100, marker=markers[idx],
                label="cluster {0:d}".format(idx))

# 设置轴标签和标题
plt.xlabel(x_label,fontsize='xx-large')
plt.ylabel(y_label,fontsize='xx-large')
plt.title("AffinityPropagation",fontsize='xx-large')

# 显示图例
plt.legend( loc="upper left" )

plt.show()
