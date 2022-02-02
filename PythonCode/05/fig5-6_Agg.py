# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn import datasets

# 加载iris数据
iris = datasets.load_iris()
data = iris["data"]

# 学习→生成簇
model = cluster.AgglomerativeClustering(n_clusters=3, linkage="ward")
model.fit(data)

# 取得学习结果的标签
labels = model.labels_

### 图表的绘制

# 数据定义
x_index = 2 
y_index = 3

data_x=data[:,x_index]
data_y=data[:,y_index]

x_label = iris["feature_names"][x_index]
y_label = iris["feature_names"][y_index]
   
# 绘制散布图
plt.scatter(data_x[labels==0], data_y[labels==0],c='black' ,alpha=0.3,s=100, marker="o")
plt.scatter(data_x[labels==1], data_y[labels==1],c='black' ,alpha=0.3,s=100, marker="^")
plt.scatter(data_x[labels==2], data_y[labels==2],c='black' ,alpha=0.3,s=100, marker="*")

# 设置轴标签和标题
plt.xlabel(x_label,fontsize='xx-large')
plt.ylabel(y_label,fontsize='xx-large')
plt.title("AgglomerativeClustering(ward)",fontsize='xx-large')

plt.show()
