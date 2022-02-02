import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn import datasets

# 加载iris数据
iris = datasets.load_iris()
data = iris['data']

# 学习 → 生成簇
model = cluster.KMeans(n_clusters=3)
model.fit(data)

# 取得学习结果的标签
labels = model.labels_

# 图表的绘制
ldata = data[labels == 0]
plt.scatter(ldata[:, 2], ldata[:, 3],
                  c='black' ,alpha=0.3,s=100 ,marker="^")

ldata = data[labels == 1]
plt.scatter(ldata[:, 2], ldata[:, 3],
                  c='black' ,alpha=0.3,s=100 ,marker="*")

ldata = data[labels == 2]
plt.scatter(ldata[:, 2], ldata[:, 3],
                  c='black' ,alpha=0.3,s=100 ,marker="o")

# 设置轴标签
plt.xlabel(iris["feature_names"][2],fontsize='large')
plt.ylabel(iris["feature_names"][3],fontsize='large')

plt.show()
