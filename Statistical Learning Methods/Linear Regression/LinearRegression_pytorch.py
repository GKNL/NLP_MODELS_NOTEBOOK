# -*- coding: utf-8 -*-
# @Time    : 2021/2/4 0:18
# @Author  : Peng Miao
# @File    : LinearRegression_pytorch.py
# @Intro   : 线性回归的简洁实现

import torch
import numpy as np
from torch import nn

"""
生成数据集
"""
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

"""
读取数据
"""
import torch.utils.data as Data

batch_size = 10
# 将训练数据的特征和标签组合
dataset = Data.TensorDataset(features, labels)
# 随机读取小批量
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)

"""
!!定义模型!!
"""
class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1, bias=True)  # 定义线性模型(两个参数：输入特征数、输出特征数)
    # forward 定义前向传播
    def forward(self, x):
        y = self.linear(x)
        return y

net = LinearNet(num_inputs)
# # 写法二
# net = nn.Sequential()
# net.add_module('linear', nn.Linear(num_inputs, 1))
# # net.add_module ......
print(net) # 使用print可以打印出网络的结构

# 查看模型参数（可以看到有两个Tensor：一个是w，一个是偏置b）
for param in net.parameters():
    print(param.size())

"""
初始化模型参数
"""
from torch.nn import init

init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)

"""
定义损失函数
"""
loss = nn.MSELoss()

"""
定义优化算法
"""
import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr=0.03)
print(optimizer)

"""
训练模型
"""
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad() # 梯度清零，等价于net.zero_grad()
        l.backward()
        # 对optimizer实例调用step函数，从而更新权重和偏差
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))
