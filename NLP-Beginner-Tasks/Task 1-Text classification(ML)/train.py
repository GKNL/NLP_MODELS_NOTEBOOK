# -*- coding: utf-8 -*-
# @Time    : 2021/1/17 18:11
# @Author  : Peng Miao
# @File    : train.py
# @Intro   : 载入数据，训练logisticRegression模型

import tensorflow as tf
from LR_Config import LrConfig
from Preprocessing import DataProcess
from LogisticRegression import LrModel
import time
from datetime import timedelta

def getData():
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


def evaluate(sess, x_, y_):
    """
    测试集上准曲率评估
    :param sess: sess会话
    :param x_: 测试集样本
    :param y_: 测试集标签
    :return: total_loss、total_acc
    """
    data_len = len(x_)
    batch_eval = dataProvider.batch_iter(x_, y_, 128)
    total_loss = 0
    total_acc = 0
    for batch_xs, batch_ys in batch_eval:
        batch_len = len(batch_xs)
        loss, acc = sess.run([model.loss, model.accuracy], feed_dict={model.x: batch_xs, model.y_: batch_ys})
        total_loss += loss * batch_len  # 乘上batch_len是为了保证这个batch的loss在整体中的占比。最后在return的时候进行归一化
        total_acc += acc * batch_len
    return total_loss/data_len, total_acc/data_len


def get_time_dif(start_time):
    """获取已经使用的时间"""
    end_time = time.time()
    time_dif = end_time-start_time
    return timedelta(seconds=int(round(time_dif)))


def train(X_train, X_test, y_train, y_test):
    # 配置Saver
    saver = tf.train.Saver()
    # 训练模型
    print("Training and evaluating...")
    start_time = time.time()
    total_batch = 0  # 总批次
    best_acc_val = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一批次提升
    require_improvement = 1000  # 如果超过1000轮未提升，提前结束训练
    flag = False
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(config.num_epochs):
            # 获取每个batch的数据并训练
            batch_train = dataProvider.batch_iter(X_train, y_train)
            for batch_xs, batch_ys in batch_train:
                # 每print_per_batch轮输出一次结果
                if total_batch % config.print_per_batch == 0:
                    # 计算loss和accuracy
                    loss_train, acc_train = sess.run([model.loss, model.accuracy], feed_dict={model.x: X_train, model.y_: y_train})
                    loss_val, acc_val = evaluate(sess, X_test, y_test)

                    if acc_val > best_acc_val:
                        # 保存最好结果
                        best_acc_val = acc_val
                        last_improved = total_batch
                        saver.save(sess=sess, save_path=config.lr_save_path)
                        improve_str = "*"
                    else:
                        improve_str = ""
                    time_dif = get_time_dif(start_time)
                    msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%}, '\
                           + 'Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                    print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improve_str))
                sess.run(model.train_step, feed_dict={model.x: batch_xs, model.y_: batch_ys})
                total_batch += 1

                if total_batch - last_improved > require_improvement:
                    #  验证集准确率长期不提升，提前结束训练
                    print("No optimization for a long time, auto-stopping...")
                    flag = True
                    break
            if flag:
                break

if __name__ == "__main__":
    config = LrConfig()
    dataProvider = DataProcess(config.dataset_path, config.stopwords_path, config.tfidf_model_save_path)
    X_train, X_test, y_train, y_test = getData()
    seq_length = len(X_train[0])
    model = LrModel(config, seq_length)
    train(X_train, X_test, y_train, y_test)