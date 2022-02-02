# -*- coding: utf-8 -*- 
import os
import sys
import glob
import numpy as np
from skimage import io
from sklearn.datasets import _base
from skimage.feature import hog

CATEGORY_NUM = 6

## 读入分类到带有标签名称(0~)的目录中的图像文件
## 输入路径是标签名称上级的目录
def load_handimage(path):

    # 取得文件一览
    files = glob.glob(os.path.join(path, '*/*.png'))

    # 确保图像和标签的领域
    hogs = np.ndarray((len(files), 3600), dtype = np.float)
    labels = np.ndarray(len(files), dtype=np.int)

    # 读取标签和图像
    for idx, file in enumerate(files):
        # 读取图像
        image = io.imread(file, as_gray=True)
        h = hog(image, orientations=9, pixels_per_cell=(5, 5),
            cells_per_block=(5, 5))
        hogs[idx] = h

        # 通过目录名取得标签
        label = os.path.split(os.path.dirname(file))[-1]
        labels[idx] = int(label)

    return _base.Bunch(data=hogs,
                 target=labels.astype(np.int),
                 target_names=np.arange(CATEGORY_NUM),
                 DESCR=None)

#####################################
from sklearn import svm, metrics

## usage:
##    python classify_handsign_1.py <n> <dir_1> <dir_2> ... <dir_m>
##      n          测试数据目录数
##      dir_1      数据目录1
##      dir_m      数据目录m

if __name__ == '__main__':
    argvs = sys.argv
    
    # 取得测试用目录
    paths_for_test = argvs[2:2+int(argvs[1])]
    paths_for_train = argvs[2+int(argvs[1]):]
    
    print('test ', paths_for_test)
    print('train', paths_for_train)

    # 学习数据的读取
    data = []
    label = []
    for i in range(len(paths_for_train)):
        path = paths_for_train[i]
        d = load_handimage(path)
        data.append(d.data)
        label.append(d.target)
    train_data = np.concatenate(data)
    train_label = np.concatenate(label)

    # 算法：线性SVM
    classifier = svm.LinearSVC()
    
    # 学习
    classifier.fit(train_data, train_label)

    for path in paths_for_test:
        # 测试数据的读取
        d = load_handimage(path)
        
        # 测试
        predicted = classifier.predict(d.data)

        # 結果表示
        print("### %s ###" % path)
        print("Accuracy:\n%s"
            % metrics.accuracy_score(d.target, predicted))
        print("Classification report:\n%s\n"
            % metrics.classification_report(d.target, predicted))
