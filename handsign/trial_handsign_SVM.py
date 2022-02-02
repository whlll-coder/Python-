# -*- coding: utf-8 -*-
from cProfile import label
import os
import sys
import glob
from hypothesis import target
from importlib_metadata import files
from matplotlib import image
import numpy as np
from skimage import io
from sklearn.datasets import _base
from skimage.feature import hog
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [1, 10, 100],
    'loss': ['hinge', 'squared_hinge']
}

#IMAGE_SIZE = 40
#COLOR_BYTE = 3
CATEGORY_NUM = 6

## 读入分类到带有标签名称(0~)的目录中的图像文件
## 输入路径是标签名称上级的目录
def load_handimage(path):

    # 取得文件一览
    files = glob.glob(os.path.join(path, '*/*.png'))

    # 确保图像和标签的领域
    #images = np.ndarray((len(files), IMAGE_SIZE, IMAGE_SIZE, COLOR_BYTE), dtype=np.uint8)
    hogs = np.ndarray((len(files), 3600), dtype=np.float)
    labels = np.ndarray(len(files), dtype=np.int)

    # 读取图像和标签
    for idx, file in enumerate(files):
        # 读取图像
        image = io.imread(file, as_gray=True)
        h = hog(image, orientations=9, pixels_per_cell=(5, 5), cells_per_block=(5, 5))
        hogs[idx] = h

        # 通过目录名取得标签
        label = os.path.split(os.path.dirname(file))[-1]
        labels[idx] = int(label)

    # 匹配 scikt-learn其他数据集的格式
    #flat_data = images.reshape((-1, IMAGE_SIZE * IMAGE_SIZE * COLOR_BYTE))
    #images = flat_data.view()
    return _base.Bunch(data=hogs, target=labels.astype(np.int), target_names = np.arange(CATEGORY_NUM), DESCR=None)

#####################################
from sklearn import svm, metrics

## 对学习数据的目录，测试数据的目录进行指定
if __name__ == '__main__':
    argvs = sys.argv
    train_path = argvs[1]
    test_path = argvs[2]

    # 学习数据的读取
    train = load_handimage(train_path)

    # 算法: 线性SVM
    classfier = GridSearchCV(svm.LinearSVC(), param_grid)

    # 学习
    classfier.fit(train.data, train.target)

    # 网格搜索结果显示
    print("Best Estimator:\n%s\n", classfier.best_estimator_)
    means = classfier.cv_results_['mean_test_score']
    stds = classfier.cv_results_['std_test_score']
    for params, mean, std in zip(classfier.cv_results_['params'], means, stds):
        print("{:.3f} (+/- {:.3f}) for {}".format(mean, std/ 2, params))

    # 测试数据的读取
    test = load_handimage(test_path)

    # 测试
    predicted = classfier.predict(test.data)

    # 结果显示
    print("Accuracy:\n%s" % metrics.accuracy_score(test.target, predicted))

