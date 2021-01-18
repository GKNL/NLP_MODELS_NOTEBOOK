# -*- coding: utf-8 -*-
# @Time    : 2021/1/17 18:11
# @Author  : Peng Miao
# @File    : train.py
# @Intro   : 载入数据，训练logisticRegression模型


def getData(dataProvider):
    """
    获取划分好的数据集
    :param dataProvider: Preprocessing.py中DataProcess类的实例化对象
    :return:X_train, X_test, y_train, y_test
    """
    print("Loading training and validation data...")
    X_train, X_test, y_train, y_test = dataProvider.provide_data()
    X_train = X_train.toarray()
    X_test = X_test.toarray()
    return X_train, X_test, y_train, y_test