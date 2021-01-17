# -*- coding: utf-8 -*-
# @Time    : 2021/1/15 22:22
# @Author  : Peng Miao
# @File    : LogisticRegression.py
# @Intro   : 使用逻辑斯蒂回归实现文本分类【Tensorflow实现】

import tensorflow as tf

class LogisticRegression(object):
    def __init__(self, config, seq_len):
        self.config = config
        self.seq_len = seq_len
        self.lr()

    def lr(self):
        self.x = tf.placeholder(tf.float32, [None, self.seq_len])  # n行 seq_len列
        self.y_ = tf.placeholder(tf.float32, [None, self.config.num_classes]) # 训练集标签

        # 变量，使用tf.Variable进行存储
        w = tf.Variable(tf.zeros([self.seq_len, self.config.num_classes]))  # seq_len行 num_classes列
        b = tf.Variable(tf.zeros([self.config.num_classes]))

        y = tf.nn.softmax(tf.matmul(self.x, w) + b)  # 使用softmax作为激活函数实现多分类

        self.y_pred_cls = tf.argmax(y, 1)  # 每行最大值的索引位置（即预测的分类标签）

        cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(y), reduction_indices=[1]))
        self.loss = tf.reduce_mean(cross_entropy)  # 交叉熵loss

        self.train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

        # 计算准确率
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))