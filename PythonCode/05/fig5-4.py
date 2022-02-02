import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn import datasets


# 加载iris数据
iris = datasets.load_iris()
data = iris['data']

# 学习→生成簇
model = cluster.KMeans(n_clusters=3)
model.fit(data)

# 取得学习结果的标签
labels = model.labels_


### 图表的绘制
MARKERS = ["o", "^" , "*" , "v", "+", "x", "d", "p", "s", "1", "2"]

# 用于在指定索引的feature值中创建散布图的函数
def scatter_by_features(feat_idx1, feat_idx2):
    for lbl in range(labels.max() + 1):
        clustered = data[labels == lbl]
        plt.scatter(clustered[:, feat_idx1], clustered[:, feat_idx2],
                    c='black' ,alpha=0.3,s=100,
                    marker=MARKERS[lbl], label='label {}'.format(lbl))

    plt.xlabel(iris["feature_names"][feat_idx1],fontsize='xx-large')
    plt.ylabel(iris["feature_names"][feat_idx2],fontsize='xx-large')


plt.figure(figsize=(16, 16))

# feature "sepal length" 和 "sepal width"
plt.subplot(3, 2, 1)
scatter_by_features(0, 1)

# feature "sepal length" 和 "petal length"
plt.subplot(3, 2, 2)
scatter_by_features(0, 2)

# feature "sepal length" 和 "petal width"
plt.subplot(3, 2, 3)
scatter_by_features(0, 3)

# feature "sepal width" 和 "petal length"
plt.subplot(3, 2, 4)
scatter_by_features(1, 2)

# feature "sepal width" 和 "petal width"
plt.subplot(3, 2, 5)
scatter_by_features(1, 3)

# feature "petal length" 和 "petal width"
plt.subplot(3, 2, 6)
scatter_by_features(2, 3)

plt.tight_layout()
plt.show()
