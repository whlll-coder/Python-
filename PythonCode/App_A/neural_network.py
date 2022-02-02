
# -*- coding: utf-8 -*-

import numpy as np
from get_data import get_all, get_batch, visualize_result


# 信号计算时允许的最小值
FLOOR = np.log(np.spacing(1))


def backward(
    Z, E, W, learning_rate, momentum, regularize_coeff, prev_difference
):
    """
    计算误差逆传播
    """
    # 嵌入超参数
    # 创建本地函数对象
    def dW(e, z, w, pd):
        """
        将矩法应用于简单的梯度、
        返回批量内的总和
        """
        # NOTE: 因为样品表示第0分量、
        #       无法很好地写出矢量之间的积
        #       相反，使用广播取得总的积
        g = e[:, :, np.newaxis] * z[:, np.newaxis, :]
        dW_batch = momentum_method(
            g, w, learning_rate, momentum, regularize_coeff, pd
        )
        return np.sum(dW_batch, axis=0)

    # NOTE: 因为没有给最终层打信号、
    #       在公式中没有更新的地方
    #       在这里打信号的时候
    #         E = grad_sigmoid(Z[-1]) * E
    #      然后插入

    # 更新量
    d = [dW(E, Z[-2], W[-1], prev_difference[-1])]
    # 反向输出和重量。
    # NOTE: _Zp 相当于 f(u^(k)) , _Zn 相当于 z_k 
    for _Wp, _Wn, _Zp, _Zn, pd in zip(
        W[-1::-1], W[-2::-1], Z[-2::-1], Z[-3::-1], prev_difference[-2::-1]
    ):
        # 根据前一层的误差计算简单梯度法的更新量
        E = (_Zp*(1-_Zp)) * E.dot(_Wp)
        # 使用力矩法存储值
        d.insert(0, dW(E, _Zn, _Wn, pd))

    # NOTE: 用线性回归的代码来返回更新量、
    #       你可以在这个方法中进行更新。
    return d


def forward(X, W):
    """
    进行正向计算。
    """
    Z = [X]
    for _W in W[:-1]:
        # NOTE: 因为每个批次都存储在第0成分中，、
        #       它不是一个等式，而是一个转置的表达式。
        Z.append(sigmoid(Z[-1].dot(_W.T)))
    # 为了解决回归问题，最后一层不发信号
    # 信号的范围是[0,1]，不能输出任意的实数
    Z.append(Z[-1].dot(W[-1].T))
    return Z


def sigmoid(X):
    """
    计算每个元素的信号
    """
    # 就这样使用X的话，会产生负数较大的溢出
    # 为了避免这种情况，所有的要素都初始化为零、
    # 十分大きなXのみ実際の計算に利用します
    out = np.zeros(X.shape)
    stable = (X > FLOOR)  # 稳定区域
    out[stable] = 1/(1+np.exp(-X[stable]))
    return out


def momentum_method(
    raw_gradient, current_parameter,
    learning_rate, momentum, regularize_coeff, prev_difference,
):
    """
    运用动量法。
    """
    # Δw := -α * ∇L + β Δw - γ w
    return (
        - learning_rate * raw_gradient +  # 梯度
        momentum * prev_difference -   # 力矩
        regularize_coeff * current_parameter  # 罚则项
    )


def train_epoch(data, W_hat, learning_rate, momentum, regularize_coeff):
    """
    进行一个阶段的学习。
    """
    difference = [0.0]*len(W_hat)
    losses = []
    for step, (X, Y) in enumerate(data):
        # 损失的计算
        # 正向计算
        Z = forward(X, W_hat)

        # 最終層の誤差
        # NOTE: 为了与Z[-1]的维度(m, 1)对齐
        #       Yにも次元を加えています
        E = Z[-1] - Y[:, np.newaxis]
        loss = E[:, 0].T.dot(E[:, 0])
        losses.append(loss)

        # 梯度和更新量的计算
        difference = backward(
            Z, E,  # 数据和中间层输出和误差
            W_hat,  # 参数
            learning_rate, momentum, regularize_coeff,  # 超参数
            difference,  # 上次的更新量
        )

        # 更新参数
        for _W_hat, _difference in zip(W_hat, difference):
            _W_hat += _difference

        # 定期显示中途经过
        if step % 100 == 0:
            print("step {0:8}: loss = {1:.4e}".format(step, loss))

    # 返回损失的平均值和在此环比的最终估计值
    return np.mean(losses), W_hat


def init_weights(num_units, prev_num_unit):
    W = []
    # NOTE: 最后一层是一维的、在num_units中追加了[1]
    for num_unit in num_units+[1]:
        # 权重的大小（当前层的单元数，上一层的单元数）
        # NOTE: 因为误差是被加权传播的、
        #       如果初始值为零则不更新
        #       这里根据正规随机数初始化
        #       此时，标准偏差为
        #         √(2.0/prev_num_unit)
        #      这样更容易收敛
        #       如果以python 2系列来执行的话，如果是2的话，会变成整数的除法，请注意
        random_weigts = np.random.randn(num_unit, prev_num_unit)
        normalized_weights = np.sqrt(2.0/prev_num_unit) * random_weigts
        W.append(normalized_weights)
        prev_num_unit = num_unit
    return W


def main():
    # 与线性回归相同的参数
    # 特征向量的维度设置
    dimension = 10
    # 非线性标记
    nonlinear = True
    # 所有数据的数量
    num_of_samples = 1000
    # 噪声幅度
    noise_amplitude = 0.01

    # 线性回归和共同的超参数
    batch_size = 10
    max_epoch = 10000
    learning_rate = 1e-3
    momentum = 0.0
    regularize_coeff = 0.0

    # 神经网络特有的超参数
    # 中间层的单元数（通道数）
    # NOTE: 这里只有一层、
    #       在列表中添加值可以增加层次
    #       但是，请注意内存的使用量！
    num_units = [
        50,
        100,
    ]

    # 获取全部数据
    data_train, (X_test, Y_test) = get_all(
        dimension, nonlinear, num_of_samples, noise_amplitude,
        return_coefficient_matrix=False
    )

    # 损失历史
    train_losses = []
    test_losses = []
    # 参数的初始值
    W_hat = init_weights(num_units, dimension)
    for epoch in range(max_epoch):
        print("epoch: {0}/{1}".format(epoch, max_epoch))
        # 分批
        data_train_batch = get_batch(data_train, batch_size)
        # 1分阶段学习
        train_loss, W_hat = train_epoch(
            data_train_batch,  W_hat,
            learning_rate, momentum, regularize_coeff
        )

        # 将结果保存在历史记录中
        train_losses.append(train_loss)

        # 对测试数据的拟合
        Y_hat = forward(X_test, W_hat)[-1][:, 0]
        E = Y_hat - Y_test
        test_loss = E.T.dot(E)
        test_losses.append(test_loss)

    # 用于实验记录
    parameters = {
        "linearity": "nonlinear" if nonlinear else "linear",
        "dimension": dimension,
        "num_of_samples": num_of_samples,
        "batch_size": batch_size,
        "max_epoch": max_epoch,
        "learning_rate": learning_rate,
        "momentum": momentum,
        "regularize_coeff": regularize_coeff,
        "num_units": num_units,
    }
    # 表示结果
    visualize_result(
        "neural_network",
        X_test[:, 0:2], Y_test, Y_hat, parameters,
        losses=(train_losses, test_losses)
    )


if __name__ == "__main__":
    main()
