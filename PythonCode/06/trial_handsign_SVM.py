# -*- coding: utf-8 -*- 
import os
import sys
import glob
import numpy as np
from skimage import io
from sklearn.datasets import _base

IMAGE_SIZE = 40
COLOR_BYTE = 3
CATEGORY_NUM = 6

## 读入分类到带有标签名称(0~)的目录中的图像文件
## 输入路径是标签名称上级的目录
def load_handimage(path):

    # 取得文件一览
    files = glob.glob(os.path.join(path, '*/*.png'))

    # 确保图像和标签的领域
    images = np.ndarray((len(files), IMAGE_SIZE, IMAGE_SIZE,
                            COLOR_BYTE), dtype = np.uint8)
    labels = np.ndarray(len(files), dtype=np.int)

    # 读取图像和标签
    for idx, file in enumerate(files):
       # 读取图像
       image = io.imread(file)
       images[idx] = image

       # 通过目录名取得标签
       label = os.path.split(os.path.dirname(file))[-1]
       labels[idx] = int(label)

    # 匹配scikit-learn其他数据集的格式
    flat_data = images.reshape((-1, IMAGE_SIZE * IMAGE_SIZE * COLOR_BYTE))
    images = flat_data.view()
    return _base.Bunch(data=flat_data,
                 target=labels.astype(np.int),
                 target_names=np.arange(CATEGORY_NUM),
                 images=images,
                 DESCR=None)

#####################################
from sklearn import svm, metrics

## 对学习数据的目录，测试数据的目录进行指定
if __name__ == '__main__':
    argvs  = sys.argv
    train_path = argvs[1]
    test_path = argvs[2]

    # 学习数据的读取
    train = load_handimage(train_path)

    # 算法：线性SVM
    classifier = svm.LinearSVC()

    # 学习
    classifier.fit(train.data, train.target)

    # 测试数据的读取
    test = load_handimage(test_path)

    # 测试
    predicted = classifier.predict(test.data)

    # 结果表示
    print("Accuracy:\n%s" % metrics.accuracy_score(test.target, predicted))
