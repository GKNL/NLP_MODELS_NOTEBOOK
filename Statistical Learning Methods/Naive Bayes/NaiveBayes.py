# -*- coding: utf-8 -*-
# @Time    : 2021/1/30 0:33
# @Author  : Peng Miao
# @File    : NaiveBayes.py
# @Intro   : 朴素贝叶斯算法【贝叶斯估计（拉普拉斯平滑）】

import numpy as np
import pandas as pd


class NaiveBayes():
    def __init__(self):
        self.modelDic = {}  # 使用字典来存储先验概率及条件概率分布。key 为类别名 val 为字典（PClass表示类别y的概率，PFeature:{}表示Px^(i)_y的概率）
        self.laplace = 1  # 拉普拉斯平滑系数

    def fit(self, x_train, y_train):
        """
        通过训练集计算先验概率分布和条件概率分布
        :param x_train:训练数据集
        :param y_train:训练标记集
        :return:
        """
        # 频次汇总 得到各个特征对应的概率
        yTrainCounts = y_train.value_counts()
        yTrainCounts = yTrainCounts.apply(lambda x: (x + self.laplace) / (y_train.size + yTrainCounts.size*self.laplace))  # 使用了拉普拉斯平滑
        retModel = {}
        for nameClass, val in yTrainCounts.items():
            retModel[nameClass] = {'PClass': val, 'PFeature': {}}
        # print(retModel)

        propNamesAll = x_train.columns[:-1]
        allPropByFeature = {}  # 对于每一种属性，有几种可取的值（用于后续的laplace平滑）
        for nameFeature in propNamesAll:
            allPropByFeature[nameFeature] = list(x_train[nameFeature].value_counts().index)

        # 按标签值进行分组
        for nameClass, group in x_train.groupby(x_train.columns[-1]):
            # 对训练集的每个x^(i)计算条件概率Px_y
            for nameFeature in propNamesAll:
                eachClassPFeature = {}
                propDatas = group[nameFeature]  # 该group样本中该属性对应的values
                propClassSummary = propDatas.value_counts()  # 例:'否'的group中，对于'色泽'属性，{'乌黑':2,'青绿':3,...}
                # 如果有属性没有出现在该组group的某个feature中，那么自动补0
                for propName in allPropByFeature[nameFeature]:
                    if not propClassSummary.get(propName):
                        propClassSummary[propName] = 0
                Si = len(allPropByFeature[nameFeature])  # x^(i)的可取值有Si个
                # 求Px_y，并进行平滑处理
                propClassSummary = propClassSummary.apply(lambda x: (x + self.laplace) / (propDatas.size + Si*self.laplace))
                for nameFeatureProp, valP in propClassSummary.items():
                    eachClassPFeature[nameFeatureProp] = valP
                retModel[nameClass]['PFeature'][nameFeature] = eachClassPFeature

        self.modelDic = retModel
        return retModel

    def predict(self, data):
        curMaxRate = None
        curClassSelect = None
        for yClass, classInfo in self.modelDic.items():
            Py_x = 0
            Py = np.log(classInfo['PClass'])
            Px_y = classInfo['PFeature']
            for nameFeature, val in data.items():
                propsRate = Px_y.get(nameFeature)
                if not propsRate:
                    continue
                Py_x += np.log(propsRate.get(val, 0))  # 使用log加法避免很小的小数连续乘，接近零
            Py_x += Py
            if curMaxRate == None or Py_x > curMaxRate:
                curMaxRate = Py_x
                curClassSelect = yClass
        return curClassSelect


if __name__ == "__main__":
    dataTrain = pd.read_csv("train.csv", encoding="gbk",sep='\t')
    yTrain = dataTrain.iloc[:, -1]
    model = NaiveBayes()
    model.fit(dataTrain, yTrain)
    print(model.modelDic)
    # 对data的每一行（即每个样本）进行预测
    preResult = dataTrain.apply(lambda d: model.predict(d), axis=1)
    pd = pd.DataFrame({'预测值': preResult, '正取值': dataTrain.iloc[:, -1]})
    print(pd)
    print('正确率:%f%%' % (pd[pd['预测值'] == pd['正取值']].shape[0] * 100.0 / pd.shape[0]))

