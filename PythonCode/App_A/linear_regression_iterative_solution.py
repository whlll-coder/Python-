
# -*- coding: utf-8 -*-

import numpy as np
from get_data import get_all, get_batch, visualize_result, coeff


def raw_gradient(A, E):
    """
   计算梯度
    """
    # NOTE: 线性回归的最小二乘法为数据块计算一个梯度、
    #       用迷你分批法的说明延长了
    #       没有处理“取每个样品的坡度之和”
    return A.T.dot(E)


def momentum_method(
    A, E, current_parameter,
    learning_rate, momentum, regularize_coeff, prev_difference,
):
    """
    运用动量法
    """
    # Δw := -α * ∇L + β Δw + γ w
    return (
        - learning_rate * raw_gradient(A, E) +  # 勾配
        momentum * prev_difference -  # 力矩
        regularize_coeff * current_parameter  # 罚则项
    )


def train_epoch(data, D_hat, learning_rate, momentum, regularize_coeff):
    """
    实行一个时代的学习
    """
    difference = 0.0
    losses = []
    for step, (X, Y) in enumerate(data):
        # 变形为系数矩阵
        A = coeff(X)

        # 损失的计算
        E = A.dot(D_hat) - Y
        loss = E.T.dot(E)
        losses.append(loss)

        # 梯度和更新量的计算
        difference = momentum_method(
            A, E, D_hat,  # 数据
            learning_rate, momentum, regularize_coeff,  # ハイパーパラメータ
            difference,  # 上次的更新量
        )

        # 更新参数
        D_hat += difference

        # 定期显示中途经过
        if step % 100 == 0:
            print("step {0:8}: loss = {1:.4e}".format(step, loss))

    # 返回损失的平均值和在此环比的最终估计值
    return np.mean(losses), D_hat


def main():
    # 与线性回归相似的参数
    # 特征向量的维度的设定
    dimension = 10
    # 非线性标志
    nonlinear = False
    # 所有数据的数量
    num_of_samples = 1000
    # 噪声的幅度
    noise_amplitude = 0.01

    # 超参数的设置
    batch_size = 10
    max_epoch = 10000
    learning_rate = 1e-3
    momentum = 0.9  # 将该值设为正则为动量法
    regularize_coeff = 0.0  # 如果该值为正则会有L2范数的惩罚

    # 获取全部数据
    # NOTE: 只看小批量行为、
    #       一旦取得全部数据后，以批处理为单位进行剪切并返还
    #       但是在需要小批量处理的情况下，
    #       一般情况下，我们无法读完所有数据，、
    #       应该读入几批，缓冲后只返回一批
    #       在这种情况下，请活用python的功能
    #       创建要依次加载的生成器是有效的
    data_train, (X_test, Y_test) = get_all(
        dimension, nonlinear, num_of_samples, noise_amplitude,
        return_coefficient_matrix=False
    )
    A_test = coeff(X_test)

    # 损失历史
    train_losses = []
    test_losses = []
    # 参数的初始值
    D_hat = np.zeros(dimension+1)
    # 关于时代的循环
    for epoch in range(max_epoch):
        print("epoch: {0} / {1}".format(epoch, max_epoch))
        # 分批
        data_train_batch = get_batch(data_train, batch_size)
        # 1分阶段学习
        loss, D_hat = train_epoch(
            data_train_batch, D_hat,
            learning_rate, momentum, regularize_coeff
        )

        # 将损失存储在历史记录中
        train_losses.append(loss)

        # 在典型的代码中，每隔几次测试一次，、
        # 确认一下中途经过会显示出多大的泛化性能、
        # 在这里进行测试
        Y_hat = A_test.dot(D_hat)
        E = Y_hat - Y_test
        test_loss = E.T.dot(E)
        test_losses.append(test_loss)

    # 实验记录
    parameters = {
        "linearity": "nonlinear" if nonlinear else "linear",
        "dimension": dimension,
        "num_of_samples": num_of_samples,
        "batch_size": batch_size,
        "max_epoch": max_epoch,
        "learning_rate": learning_rate,
        "momentum": momentum,
        "regularize_coeff": regularize_coeff,
    }
    # 显示结果
    visualize_result(
        "linear_regression_iterative_solution",
        A_test[:, 0:2], Y_test, Y_hat, parameters,
        losses=(train_losses, test_losses)
    )

if __name__ == "__main__":
    main()
