# -*- coding: utf-8 -*-
# @Time    : 2021/1/16 0:03
# @Author  : Peng Miao
# @File    : LR_Config.py
# @Intro   : 模型训练用到的相关路径

import os

pwd_path = os.path.abspath(os.path.dirname(__file__))
print(pwd_path)


class LrConfig(object):
    #  训练模型用到的路径
    dataset_path = os.path.join(pwd_path + '/data' + "/cnews.train.txt")
    stopwords_path = os.path.join(pwd_path + '/data' + "/stopwords.txt")
    # tfidf_model_save_path = os.path.join(pwd_path + '/result' + "/tfidf_model.m")
    categories_save_path = os.path.join(pwd_path + '/result' + '/categories.txt')

    lr_save_dir = os.path.join(pwd_path + '/result' + "/checkpoints")
    lr_save_path = os.path.join(lr_save_dir, 'best_validation')
    #  变量
    num_epochs = 100  # 总迭代轮次
    num_classes = 10  # 类别数
    print_per_batch = 10  # 每多少轮输出一次结果