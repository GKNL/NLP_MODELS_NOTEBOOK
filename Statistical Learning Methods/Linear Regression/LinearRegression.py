# -*- coding: utf-8 -*-
# @Time    : 2021/2/3 23:00
# @Author  : Peng Miao
# @File    : LinearRegression.py
# @Intro   : Pytorch实现线性回归

import torch
import numpy as np
import random


"""
生成数据集
"""
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.randn(num_examples, num_inputs,dtype=torch.float32)  # 生成特征矩阵
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
# 随机噪声
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float32)
print(features[0], labels[0])

"""
批量读取数据
"""
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)]) # 最后一次可能不足一个batch
        yield  features.index_select(0, j), labels.index_select(0, j)


"""
初始化模型参数
"""
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32, requires_grad=True)  # 均值为0、标准差为0.01的正态随机数
b = torch.zeros(1, dtype=torch.float32, requires_grad=True)

"""
定义模型
"""
def linreg(X, w, b):
    return torch.mm(X, w) + b


"""
定义损失函数
"""
def squared_loss(y_hat, y):
    return 1/2 * ((y_hat - y.view(y_hat.size()))**2)


"""
定义优化算法
"""
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size  # 注意这里更改param时用的param.data


"""
训练模型
"""
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss
batch_size = 10

for epoch in range(num_epochs):  # 训练模型一共需要num_epochs个迭代周期
    # 在每一个迭代周期中，会使用训练数据集中所有样本一次（假设样本数能够被批量大小整除）。X
    # 和y分别是小批量样本的特征和标签
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y).sum()  # l是有关小批量X和y的损失
        l.backward()  # 小批量的损失对模型参数求梯度
        sgd([w, b], lr, batch_size)  # 使用小批量随机梯度下降迭代模型参数

        # 梯度清零!!!!否则会累加
        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))

# 比较真实的w、b和学习到的w、b
print(true_w, '\n', w)
print(true_b, '\n', b)