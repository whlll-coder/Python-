# -*- coding: utf-8 -*- 
import matplotlib.pyplot as plt
import sys
from skimage.feature import hog
from skimage import data, color, exposure, io


if __name__ == '__main__':
    argvs = sys.argv

    ## 读取数据（灰度）
    image = io.imread(argvs[1], as_gray=True)

    ## 计算HOG特征量
    fd, hog_image = hog(image, orientations=9, pixels_per_cell=(5, 5),
                    cells_per_block=(5, 5), visualize=True)

    ## 绘制原始图像
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True,
                    sharey=True)

    ax1.axis('off')
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Input image')
    ax1.set_adjustable('box-forced')

    ## 绘制HOG特征量

    hog_image_rescaled = exposure.rescale_intensity(hog_image, 
                    in_range=(0, 0.02))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    ax1.set_adjustable('box-forced')
    plt.show()
