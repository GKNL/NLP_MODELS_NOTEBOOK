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
        self.embedding = nn.Embedding(5000, 64)  # 进行词嵌入
        self.rnn = nn.LSTM(input_size=64, hidden_size=128, bidirectional=True)  # hidden_size　LSTM中隐层的维度
        self.f1 = nn.Sequential(nn.Linear(256, 10),
                                nn.Softmax())  # 全连接层

    def forward(self, x):
        x = self.embedding(x) # batch_size(一次性喂给网络多少条句子) x text_len(每个句子的长度[词的个数]) x embedding_size 64*600*64
        x= x.permute(1, 0, 2) # text_len x batch_size x embedding_size 600*64*64
        # x为600*64*256, h_n为2*64*128 lstm_out       Sentence_length * Batch_size * (hidden_layers * 2 [bio-direct]) h_n           （num_layers * 2） * Batch_size * hidden_layers
        x, (h_n, c_n)= self.rnn(x)
        final_feature_map = F.dropout(h_n, 0.8)
        feature_map = torch.cat([final_feature_map[i, :, :] for i in range(final_feature_map.shape[0])], dim=1) #64*256 Batch_size * (hidden_size * hidden_layers * 2)
        final_out = self.f1(feature_map) #64*10 batch_size * class_num
        return final_out
