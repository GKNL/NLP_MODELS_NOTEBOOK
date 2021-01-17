# -*- coding: utf-8 -*-
# @Time    : 2021/1/16 10:10
# @Author  : Peng Miao
# @File    : Preprocessing.py
# @Intro   : 读取数据集，划分训练集、测试集、验证集等

from .LR_Config import LrConfig
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.externals import joblib
import jieba
import numpy as np

config = LrConfig()


class DataProcess(object):
    def __init__(self, dataset_path=None, stopwords_path=None, model_save_path=None):
        self.dataset_path = dataset_path
        self.stopwords_path = stopwords_path
        self.model_save_path = model_save_path

    def read_data(self):
        """读取数据"""
        stopwords = list()
        with open(self.dataset_path, encoding='utf-8') as f1:
            data = f1.readlines()
        with open(self.stopwords_path, encoding='utf-8') as f2:
            temp_stopwords = f2.readlines()
        for word in temp_stopwords:
            stopwords.append(word[:-1])
        return data, stopwords

    def save_categories(self, data, save_path):
        """将文本的类别写到本地"""
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write('|'.join(data))

    def pre_data(self, data, stopwords, test_size=0.2):
        """数据预处理"""
        label_list = list()
        text_list = list()
        for line in data:
            label, text = line.split('\t', 1)
            # print(label)
            seg_text = [word for word in jieba.cut(text) if word not in stopwords]
            text_list.append(' '.join(seg_text))
            label_list.append(label)
        # 标签转化为one-hot格式
        encoder_nums = LabelEncoder()
        label_nums = encoder_nums.fit_transform(label_list)
        categories = list(encoder_nums.classes_)
        self.save_categories(categories, config.categories_save_path)
        label_nums = np.array([label_nums]).T
        encoder_one_hot = OneHotEncoder()
        label_one_hot = encoder_one_hot.fit_transform(label_nums)
        label_one_hot = label_one_hot.toarray()
        return model_selection.train_test_split(text_list, label_one_hot, test_size=test_size, random_state=1024)

    # TODO:使用word2vec来代替one-hot进行文本表示
    def get_word2vec(self):
        """提取word2vec特征"""
        pass

    def provide_data(self):
        """提供数据"""
        data, stopwords = self.read_data()
        X_train, X_test, y_train, y_test = self.pre_data(data, stopwords, test_size=0.2)
        return X_train, X_test, y_train, y_test

    def batch_iter(self, x, y, batch_size=64):
        """迭代器，将数据分批传给模型"""
        data_len = len(x)
        num_batch = int((data_len-1)/batch_size)+1
        indices = np.random.permutation(np.arange(data_len))
        x_shuffle = x[indices]
        y_shuffle = y[indices]
        for i in range(num_batch):
            start_id = i*batch_size
            end_id = min((i+1)*batch_size, data_len)
            yield x_shuffle[start_id: end_id], y_shuffle[start_id: end_id]