
# -*- coding: utf-8 -*-

import time
import numpy as np

# 进行数据取得和结果可视化的方法是
# 外部模块get_data.py中定义。
from get_data import get_all, visualize_result


def main():
    # 设置实验参数
    # 请改变dimension, nonlinear, num_of_samples来比较结果
    # NOTE: 因为这些是为了实验而变更数据集本身的东西
    #       与定义算法动作的超参数不同

    # 特征向量的次元
    dimension = 100
    # 非线性标记
    # True  -> 超平面
    # False -> 超曲面
    # 线性回归是超平面的模型，当然False会给出更好的估计结果
    nonlinear = False
    # 所有数据的数量
    num_of_samples = 1000
    # 噪声幅度
    noise_amplitude = 0.01

    # 获取全部数据
    # NOTE: 测试数据上有记号“test”、
    #       关于学习用数据，为了让公式在执行计算的代码中更容易识别
    #       没有标记'train’
    (A, Y), (A_test, Y_test) = get_all(
        dimension, nonlinear, num_of_samples, noise_amplitude
    )

    # 逆矩阵估计
    start = time.time()
    # 直接计算(A^tA)^(-1) A^t Y 
    D_hat_inv = (np.linalg.inv(A.T.dot(A)).dot(A.T)).dot(Y)
    print("D_hat_inv: {0:.16f}[s]".format(time.time() - start))

    # 联立方程的求解估计
    start = time.time()
    # 用A.tA * D = A.t Y 解 D 
    D_hat_slv = np.linalg.solve(A.T.dot(A), A.T.dot(Y))
    print("D_hat_slv: {0:.16f}[s]".format(time.time() - start))

    # 两个解的差
    dD = np.linalg.norm(D_hat_inv - D_hat_slv)
    print("difference of two solutions: {0:.4e}".format(dD))

    # NOTE: 我们可以确定两个解之间没有太大的差异，、
    #       在下面的图中，我们只使用D_hat_slv
    # 对测试数据的模拟
    Y_hat = A_test.dot(D_hat_slv)
    mse = np.linalg.norm(Y_test-Y_hat) / dimension
    print("test error: {:.4e}".format(mse))

    # 实验记录
    parameters = {
        "linearity": "nonlinear" if nonlinear else "linear",
        "dimension": dimension,
        "num_of_samples": num_of_samples,
    }
    # 结果的显示
    # NOTE: 只显示二维结果
    visualize_result(
        "linear_regression_analytic_solution",
        A_test[:, :2], Y_test, Y_hat, parameters
    )

if __name__ == "__main__":
    main()
