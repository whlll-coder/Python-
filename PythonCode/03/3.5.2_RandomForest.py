# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import ensemble
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

# 加载digits数据
digits = datasets.load_digits()

# 在第2行第5列显示图像
for label, img in zip(digits.target[:10], digits.images[:10]):
    plt.subplot(2, 5, label + 1)
    plt.axis('off')
    plt.imshow(img, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Digit: {0}'.format(label))
plt.show()

# 求出3和8的数据位置
flag_3_8 = (digits.target == 3) + (digits.target == 8)

# 获取数据3和8
images = digits.images[flag_3_8]
labels = digits.target[flag_3_8]

# 将3和8的图像数据一维化
images = images.reshape(images.shape[0], -1)

# 分类器的生成
n_samples = len(flag_3_8[flag_3_8])
train_size = int(n_samples * 3 / 5)
classifier = ensemble.RandomForestClassifier(n_estimators=20, max_depth=3, criterion="gini")
classifier.fit(images[:train_size], labels[:train_size])

# 分类器性能的确认
expected = labels[train_size:]
predicted = classifier.predict(images[train_size:])

print('Accuracy:\n',
      accuracy_score(expected, predicted))
print('Confusion matrix:\n',
      confusion_matrix(expected, predicted))
print('Precision:\n',
      precision_score(expected, predicted, pos_label=3))
print('Recall:\n',
      recall_score(expected, predicted, pos_label=3))
print('F-measure:\n',
      f1_score(expected, predicted, pos_label=3))
