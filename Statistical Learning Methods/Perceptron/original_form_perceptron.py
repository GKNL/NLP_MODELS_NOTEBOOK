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
import numpy as np
from matplotlib import pyplot as plt

# 初始化参数
training_set = []

w = []
b = 0
lens = 0  # 权重向量维度（输入数据的维度，即网络的结点数）
n = 0


# 通过SGD更新参数
def update(item,k):
    global w, b, lens, n
    for i in range(lens):
        w[i] = w[i] + n * item[1] * item[0][i]
    b = b + n * item[1]
    print("第"+str(k)+"次更新权重: w: " + str(w) + " b: " + str(b))


# 计算item与决策面之间的距离（不考虑1/||w||）：-y(wx+b)
def cal(item):
    global w, b
    res = 0
    for i in range(len(item[0])):
        res += item[0][i] * w[i]
    res += b
    res *= item[1]
    return res


# 检查超平面是否可以正确地分类这些数据
def check(k):
    flag = False  # 是否需要更新
    for item in training_set:
        if cal(item) <= 0:
            flag = True
            update(item,k)
    if not flag:  # False
        print("RESULT: w: " + str(w) + " b: " + str(b))

        # 绘制结果
        datas = np.array(training_set)
        plt.title("Perceptron Result")
        plt.xlabel("x1 axis")
        plt.ylabel("x2 axis")
        x_end = []
        for i in datas[:, 0]:
            plt.scatter(i[0], i[1])
            x_end.append(i[0])

        x_end = np.array(x_end)
        y_end = (1/w[0])*(-b - w[1]*x_end)
        plt.plot(x_end, y_end)
        plt.show()

        os._exit(0)


if __name__ == "__main__":
    training_set = [[[3, 3], 1],
                    [[4, 3], 1],
                    [[1, 1], -1]]
    lens = 2
    n = 0.5
    for i in range(lens):
        w.append(0)

    for i in range(1000):
        check(i+1)
