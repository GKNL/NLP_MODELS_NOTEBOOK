# -*- coding: utf-8 -*-
# @Time    : 2021/2/3 15:11
# @Author  : Peng Miao
# @File    : LSTM_Pytorch.py
# @Intro   : 使用Pytorch实现RNN（BiLSTM）

import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F

class TextRNN(nn.Module):
    """文本分类，RNN模型"""

    def __init__(self):
        super(TextRNN, self).__init__()
        # 三个待输入的数据
        self.embedding = nn.Embedding(5000, 64)  # 使用随机初始化的方式进行词嵌入[这里正好字典中有5000个字]
        self.rnn = nn.LSTM(input_size=64, hidden_size=128, bidirectional=True)  # hidden_size　隐层状态的维数：（每个LSTM单元或者时间步的输出的ht的维度，单元内部有权重与偏差计算）
        self.f1 = nn.Sequential(nn.Linear(256, 10),
                                nn.Softmax())  # 全连接层

    def forward(self, x):
        x = self.embedding(x) # batch_size(一次性喂给网络多少条句子) x text_len(每个句子的长度[词的个数]) x embedding_size 64*600*64
        x= x.permute(1, 0, 2) # text_len x batch_size x embedding_size 600*64*64
        """
        output(seq_len, batch, hidden_size * num_directions)  600*64*(128*2)
        hn(num_layers * num_directions, batch, hidden_size)   (1*2)*64*128
        cn(num_layers * num_directions, batch, hidden_size)   (1*2)*64*128
        """
        output, (h_n, c_n)= self.rnn(x)
        final_feature_map = F.dropout(h_n, 0.8)
        # 按列拼接
        feature_map = torch.cat([final_feature_map[i, :, :] for i in range(final_feature_map.shape[0])], dim=1) #64*256 Batch_size * (hidden_size * hidden_layers * 2)
        final_out = self.f1(feature_map) #64*10 batch_size * class_num
        return final_out
