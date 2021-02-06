# -*- coding: utf-8 -*-
# @Time    : 2021/2/3 15:11
# @Author  : Peng Miao
# @File    : LSTM_Pytorch.py
# @Intro   : 使用Pytorch实现RNN（LSTM）

import torch
from torch import nn
from torch.nn import init

class MyDictDense(nn.Module):
    def __init__(self):
        super(MyDictDense, self).__init__()
        self.params = nn.ParameterDict({
                'linear1': nn.Parameter(torch.randn(4, 4)),
                'linear2': nn.Parameter(torch.randn(4, 1))
        })
        self.params.update({'linear3': nn.Parameter(torch.randn(4, 2))}) # 新增

    def forward(self, x, choice='linear1'):
        return torch.mm(x, self.params[choice])

net = MyDictDense()
print(net)

x = torch.ones(1, 4)
print(net(x, 'linear2'))
