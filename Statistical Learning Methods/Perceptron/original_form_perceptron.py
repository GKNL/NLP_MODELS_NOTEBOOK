# -*- coding: utf-8 -*-
# @Time    : 2021/1/22 17:14
# @Author  : Peng Miao
# @File    : original_form_perceptron.py
# @Intro   : 感知器学习算法的原始形式

"""
输入：T={(x1,y1),(x2,y2)...(xN,yN)}（其中xi∈X=Rn，yi∈Y={-1, +1}，i=1,2...N，学习速率为η）
输出：w, b;感知机模型f(x)=sign(w·x+b)
(1) 初始化w0,b0
(2) 在训练数据集中选取（xi, yi）
(3) 如果yi(w xi+b)≤0
           w = w + ηyixi
           b = b + ηyi
(4) 转至（2）
"""

import os
import sys

# An example in that book, the training set and parameters' sizes are fixed
training_set = []

w = []
b = 0
lens = 0
n = 0


# update parameters using stochastic gradient descent
def update(item):
    global w, b, lens, n
    for i in range(lens):
        w[i] = w[i] + n * item[1] * item[0][i]
    b = b + n * item[1]
    print(w, b)  # you can uncomment this line to check the process of stochastic gradient descent


# calculate the functional distance between 'item' an the dicision surface
def cal(item):
    global w, b
    res = 0
    for i in range(len(item[0])):
        res += item[0][i] * w[i]
    res += b
    res *= item[1]
    return res


# check if the hyperplane can classify the examples correctly
def check():
    flag = False
    for item in training_set:
        if cal(item) <= 0:
            flag = True
            update(item)
    if not flag:  # False
        print("RESULT: w: " + str(w) + " b: " + str(b))
        tmp = ''
        for keys in w:
            tmp += str(keys) + ' '
        tmp = tmp.strip()

        os._exit(0)
    flag = False
