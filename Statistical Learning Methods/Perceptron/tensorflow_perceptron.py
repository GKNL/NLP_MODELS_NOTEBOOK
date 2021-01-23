# -*- coding: utf-8 -*-
# @Time    : 2021/1/23 22:25
# @Author  : Peng Miao
# @File    : tensorflow_perceptron.py
# @Intro   : 使用tensorflow实现感知机


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def generate_data(a, b, n):
    '''
    产生样本 [x, y, 1] 其中y = a * x + b 加上随机噪声
    :param n: 样本个数
    :return: X: 数据 label：类别标号{-1, +1}
    '''
    X = np.ndarray((n, 3))
    X[:, 0] = np.linspace(0, 10, n)
    X[:, 1] = a * X[:, 0] + b + np.random.randn(n) * 3
    X[:, 2] = 1  # 相当于偏置b
    label = np.array([1 if a * x[0] + b < x[1] else -1 for x in X])
    return X, label


def plot_data(X, label, a, b):
    '''
    绘制
    '''
    plt.scatter(X[label == 1][:, 0], X[label == 1][:, 1])
    plt.scatter(X[label == -1][:, 0], X[label == -1][:, 1])
    x_end = np.array([X[0, 0], X[-1, 0]])
    y_end = a * x_end + b
    plt.plot(x_end, y_end)
    plt.show()


def learn(X, label):
    sess = tf.Session()

    A = tf.Variable(tf.random_normal(shape=[3, 1]))
    x_data = tf.placeholder(shape=[1, 3], dtype=tf.float32)
    label_data = tf.placeholder(shape=[1, 1], dtype=tf.float32)

    loss = tf.reduce_mean(tf.matmul(tf.matmul(x_data, A), label_data))  # 计算单个样本数据的loss
    my_opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train_step = my_opt.minimize(loss)  # 梯度计算和参数更新

    init = tf.global_variables_initializer()
    sess.run(init)
    steps = 3000
    for i in range(steps):
        flag = True
        idx = 0
        # 找一个错误分类样本
        for j in range(len(X)):
            t = sess.run(loss, feed_dict={x_data: [X[j]], label_data: [[label[j]]]})
            if t > 0:
                idx = j  # 第idx个样本被错误分类
                flag = False
                break
        if flag:
            print('all classed correctly!')
            break
        else:
            if (i + 1) % 10 == 0:
                # 计算当前loss
                print('loss:', sess.run(loss, feed_dict={x_data: [X[idx]], label_data: [[label[idx]]]}))
            # 随机梯度下降更新权重
            sess.run(train_step, feed_dict={x_data: [X[idx]], label_data: [[label[idx]]]})

    A_ = sess.run(A)
    plot_data(X, label, -A_[0, 0] / A_[1, 0], - A_[2, 0] / A_[1, 0])
    print('a_ = ', -A_[0, 0] / A_[1, 0], ' b_ = ', - A_[2, 0] / A_[1, 0])


if __name__ == '__main__':
    a = 1.0
    b = 2.0
    X, label = generate_data(a, b, 100)
    plot_data(X, label, a, b)
    learn(X, label)