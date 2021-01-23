# -*- coding: utf-8 -*-
# @Time    : 2021/1/22 20:30
# @Author  : Peng Miao
# @File    : dual_form_perceptron.py
# @Intro   : 感知器学习算法的对偶形式

"""
输入：T={(x1,y1),(x2,y2)...(xN,yN)}（其中xi∈X=Rn，yi∈Y={-1, +1}，i=1,2...N，学习速率为η）
输出：alpha, b;感知机模型f(x)=sign(w·x+b)
(1) 初始化w0,b0
(2) 在训练数据集中选取（xi, yi）
(3) 如果yi((alpha_j*yj*xj)的和*xi+b)≤0
           alpha = alpha + η
           b = b + ηyi
(4) 转至（2）
"""

import os
import numpy as np
from matplotlib import pyplot as plt

# 初始化参数
training_set = np.array([
        [[3, 3], 1],
        [[4, 3], 1],
        [[1, 1], -1]
])

a = np.zeros(len(training_set), np.float)
b = 0
n = 1  # 学习率
Gram = None
y = np.array(training_set[:, 1])
x = np.empty((len(training_set), 2), np.float)
for i in range(len(training_set)):
    x[i] = training_set[i][0]


def generate_gram(x):
    """
    生成Gram矩阵
    :param x: input数据集
    :return: gram矩阵
    """
    sampleLen = len(x)
    gram = np.zeros((sampleLen,sampleLen),np.float)
    for i in range(sampleLen):
        for j in range(sampleLen):
            gram[i][j] = np.dot(x[i],x[j])
    return gram

# print(generate_gram(x))


def update(i):
    """
    更新alpha和b
    :param i: 第i个数据，对应更新alpha_i
    :return:
    """
    global a, b, n
    a[i] = a[i] + n
    b += n*y[i]
    print("alpha: " + str(a) + " b: " + str(b))


def calculate(i):
    """
    计算是否需要对第i个样本进行更新
    :param i:对第i个样本数据进行计算
    :return:
    """
    global a, b, x, y
    res = np.dot(a * y, Gram[i])
    res = y[i]*(res + b)
    return res


def check():
    global a, b, x, y
    flag = False
    for i in range(len(training_set)):
        if calculate(i) <= 0:
            flag = True
            update(i)
    if not flag:
        w = np.dot(a * y, x)
        print("RESULT: w: " + str(w) + " b: " + str(b))
        return False
    return True

if __name__ == "__main__":
    Gram = generate_gram(x)
    for i in range(1000):
        if not check():
            break