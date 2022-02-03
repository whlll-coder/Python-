
# -*- coding: utf-8 -*-


import os
import json
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt


def visualize_result(
    experiment_name,
    X_test, Y_test, Y_hat, parameters,
    losses=None, save_dir="results"
):
    """
    结果可视化
    """
    # 没有保存目录时创建
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    save_dir += "_" + experiment_name + os.sep + now
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 测试数据适用（仅前2轴）
    # 创建显示区域
    plt.figure()
    # 为了同时显示估计值和真值，设定为hold＝“on”
    #plt.hold("on")
    # x_0 vs y 的显示
    plt.subplot(211)
    plt.plot(X_test[:, 0], Y_test, "+", label="True")
    plt.plot(X_test[:, 0], Y_hat, "x", label="Estimate")
    plt.xlabel("x_0")
    plt.ylabel("y")
    plt.legend()
    # x_1 vs y 的显示
    plt.subplot(212)
    plt.plot(X_test[:, 1], Y_test, "+")
    plt.plot(X_test[:, 1], Y_hat, "x")
    plt.xlabel("x_1")
    plt.ylabel("y")

    # 保存参数到文件
    # NOTE:json形式是设定文件等数据记述方便的形式
    #       其实质是结构化文本文件
    #       阅读时请使用适当的文本编辑器
    #       使用Python时，标准具备处理json的模块
    #       （名称也是json模块）
    #       其他数据记述形式有yaml、xml等
    fn_param = "parameters.json"
    with open(save_dir + os.sep + fn_param, "w") as fp:
        json_str = json.dumps(parameters, indent=4)
        fp.write(json_str)

    # 保存图像到文件
    fn_fit = "fitting.png"  # 各种条件
    plt.savefig(save_dir + os.sep + fn_fit)

    # 表示损失
    if losses is not None:
        train_losses, test_losses = losses
        # NOTE：损失的推移通常是指数的、
        #       多以对数比例显示
        x_train = range(len(train_losses))
        x_test = range(len(test_losses))
        plt.figure()
        plt.plot(
            x_train, np.log(train_losses),
            x_test, np.log(test_losses)
        )
        plt.xlabel("steps")
        plt.ylabel("ln(loss)")
        plt.legend(["training loss", "test loss"])

        fn_loss = "loss.png"
        plt.savefig(save_dir + os.sep + fn_loss)


def flat_nd(xs):
    """
    返回numpy.array
    """
    return np.c_[tuple([x.flatten() for x in xs])]


def genearate_original_data(
    dimension=2, nonlinear=False, num_of_samples=10000, noise_amplitude=0.1
):
    """
    用其他方法生成返回变量的来源数据
    """
    # 次元は最低でも2とします
    if dimension < 2:
        raise ValueError("'dimension' must be larger than 2")

    # NOTE：输入值x的范围为规定值[0,1]。
    #       但是，采样的点决定为均匀随机数。
    x_sample = np.random.rand(num_of_samples, dimension)
    # NOTE: 返回显示时均匀无噪声的数据
    #       即使显示多维数据也不知道、
    #       为了方便起见，只动了最初的二维、
    #       其他次元全部固定为常数
    grid_1d = np.arange(0.0, 1.0, 0.01)
    fixed_coeff = 0.0
    x_grid = flat_nd(np.meshgrid(grid_1d, grid_1d))

    # NOTE: “正确答案”的关系式是
    #         f(x) = -1.0 + x_1 + 0.5 * x_2 + Σ_{i>=3} 1/i * x_i
    #                + sin(2πx_1) + cos(2πx_2)
    #                  + Σ_{i>=3, odd} sin(2πx_i)
    #                  + Σ_{i>=4, even} cos(2πx_i)
    #       
    #       不是特别有意义的式子。
    def f(x):
        # 3有可能没有以下项
        higher_terms = x[:, 2:] / np.arange(2, x.shape[1])
        if len(higher_terms) == 0:
            ht_sum = 0.0
        else:
            ht_sum = np.sum(higher_terms, axis=1)

        # 首先加入线性项
        y = -1.0 + 1.0 * x[:, 0] + 0.5 * x[:, 1] + ht_sum

        # 非线性标记
        if nonlinear:
            if len(higher_terms) == 0:
                ht_sum = 0.0
            else:
                PI2 = np.pi*2
                sin = np.sin(PI2*x[:, 2::2])
                cos = np.cos(PI2*x[:, 3::2])
                ht_sum = np.sum(sin) + np.sum(cos)
            y += np.sin(PI2*x[:, 0]) + np.cos(PI2*x[:, 1]) + ht_sum

        return y

    # 计算输出值。
    # NOTE: 对样本数据添加正常噪声。
    noise = noise_amplitude * np.random.randn(x_sample.shape[0])
    y_sample = f(x_sample) + noise

    y_grid = f(x_grid).reshape(x_grid.shape[0])
    # 添加固定值
    fixed_columns = fixed_coeff * np.ones((x_grid.shape[0], dimension-2))
    x_grid = np.concatenate((x_grid, fixed_columns), axis=1)
    return (
        (x_sample, y_sample),
        (x_grid, y_grid),
    )


def coeff(x):
    """
    将原始数据x整形成系数矩阵并返回
    """
    return np.c_[x, np.ones(x.shape[0])]


def get_all(
    dimension, nonlinear, num_of_samples, noise_amplitude,
    return_coefficient_matrix=True
):
    """
    输入值x为线性回归系数矩阵、
    输出矢量为y、
    全部数据一起返回
    """

    # 获取原始数据
    # NOTE: 不需要原始数据获取网格点上的值
    #       以通常不可见的变量名接受并忽略
    #       只是带有惯用的意思、
    #       请注意，这是一个实际可访问的普通变量
    data_sample, _ = genearate_original_data(
        dimension, nonlinear, num_of_samples, noise_amplitude
    )
    X, Y = data_sample

    # 选择随机索引以确定学习/测试数据
    N = X.shape[0]
    perm_indices = np.random.permutation(range(N))
    train = perm_indices[:N//2]  # 因为是整数，所以向下舍入
    test = perm_indices[N//2:]

    # 是否作为系数矩阵返回
    if return_coefficient_matrix:
        X = coeff(X)

    return (X[train], Y[train]), (X[test], Y[test])


def get_batch(data, batch_size):
    """
    输入值x、输出值y的元组数据
    以分批方式返回
    """
    X, Y = data
    N = len(X)

    # 用permutation方法打乱升序排列的非负整数
    indices = np.random.permutation(np.arange(N))
    # 以批量大小剪切打乱的整数列
    # 用于X和Y的索引
    data_batch = [
        (X[indices[i: i+batch_size]], Y[indices[i: i+batch_size]])
        for i in range(0, N, batch_size)
    ]

    return data_batch


def main():
    """
    单独执行此文件时
    显示所有获取的数据。
    """

    # 数据数量
    num_of_samples = 1000

    # 非线性标记
    # True - > 平面
    # False -> 曲面
    nonlinear = False

    # 数据生成
    data_sample, data_grid = genearate_original_data(
        nonlinear=nonlinear, num_of_samples=num_of_samples
    )
    x_sample, y_sample = data_sample
    x_grid, y_grid = data_grid

    # 表示用整形
    num_of_grid_points = int(np.sqrt(len(y_grid)))
    x_grid_0 = x_grid[:, 0].reshape((num_of_grid_points,)*2)
    x_grid_1 = x_grid[:, 1].reshape((num_of_grid_points,)*2)
    y_grid = y_grid.reshape((num_of_grid_points,)*2)

    # NOTE: 看图的话，等高线的缓慢的地方和急的地方
    #       可以看出是以同样的密度得分的
    #       （看不清楚的情况下请减少分数试着实行）。
    #       实际上，要想抓住函数形式，紧急的地方的信息很重要、
    #       明白并不是说得分均一就好。
    plt.figure()
    plt.contour(x_grid_0, x_grid_1, y_grid, levels=np.arange(-2.0, 2.0, 0.1))
    plt.xlabel("x_0")
    plt.ylabel("x_1")
    plt.title("countour of f(x)")
    plt.scatter(x_sample[:, 0], x_sample[:, 1], color="k", marker="+")

    plt.savefig("original_data.png")

if __name__ == "__main__":
    main()
