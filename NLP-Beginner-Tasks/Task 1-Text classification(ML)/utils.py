# -*- coding: utf-8 -*-
# @Time    : 2021/1/17 11:02
# @Author  : Peng Miao
# @File    : utils.py
# @Intro   : 一些常用的激活函数

import numpy as np


def sigmoid(Z):
    """ sigmoid(x) = 1/[1+e^(-x)] """
    A = 1 / (1 + np.exp(-Z))
    return A


def tanh(Z):
    """ tanh(x) = [e^x-e^(-x)]/[e^x+e^(-x)] """
    A = np.sinh(Z) / np.cosh(Z)
    return A


def relu(Z):
    """  relu(x) = x * (x > 0) """
    A = np.maximum(0, Z)
    assert (A.shape == Z.shape)
    return A


def leakyRelu(Z, lambda1=0.1):
    """  leakyRelu(x) = x(x>=0时); lambda*x(x<0时) """
    A = list(map(lambda x: x if x > 0 else lambda1 * x, Z))
    return A


def softmax(Z):
    """  softmax(x) = e^i/(e^j的和) """
    Z_shift = Z - np.max(Z, axis=0)
    A = np.exp(Z_shift) / np.sum(np.exp(Z_shift), axis=0)
    return A


if __name__ == "__main__":
    X = np.array([-3, -2, -1, 0, 1, 2, 3])
    print(relu(X))
