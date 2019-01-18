# coding=utf-8

import os
# 图像读取库
from PIL import Image
# 矩阵运算库
import numpy as np
import tensorflow as tf
import random

abspath = os.path.split(os.path.realpath(__file__))[0]
# 数据文件夹
data_dir = abspath + "\\data"
# 模型文件路径
model_path = abspath + "\\model\\image_model"
# 数据文件夹
train_data_dir = abspath + "\\traindata"
test_data_dir = abspath + "\\testdata"


# 从文件夹读取图片和标签到numpy数组中
# 标签信息在文件名中，例如1_40.jpg表示该图片的标签为1
def read_data2(data_dir):
    datas = []
    labels = []
    fpaths = []
    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        fpaths.append(fpath)
    random.shuffle(fpaths)

    for fpath in fpaths:
        image = Image.open(fpath)
        data = np.array(image) / 255.0
        fname = os.path.split(fpath)[1]
        label = int(fname.split("_")[0])
        datas.append(data)
        labels.append(label)

    train_datas = np.array(datas)[0:3800]
    train_labels = np.array(labels)[0:3800]
    test_datas = np.array(datas)[3800:]
    test_labels = np.array(labels)[3800:]

    print("shape of training datas: {}\tshape of training labels: {}".format(
        train_datas.shape, train_labels.shape))
    print("shape of test datas: {}\tshape of test labels: {}".format(
        test_datas.shape, test_datas.shape))

    return train_datas, train_labels, test_datas, test_labels


def read_data(train_dir, test_dir):
    train_datas = []
    train_labels = []
    train_fpaths = []
    test_datas = []
    test_labels = []
    test_fpaths = []
    for fname in os.listdir(train_dir):
        fpath = os.path.join(train_dir, fname)
        train_fpaths.append(fpath)
    random.shuffle(train_fpaths)

    for fpath in train_fpaths:
        image = Image.open(fpath)
        data = np.array(image) / 255.0
        fname = os.path.split(fpath)[1]
        label = int(fname.split("_")[0])
        train_datas.append(data)
        train_labels.append(label)

    train_datas = np.array(train_datas)
    train_labels = np.array(train_labels)

    for fname in os.listdir(test_dir):
        fpath = os.path.join(test_dir, fname)
        test_fpaths.append(fpath)

    for fpath in test_fpaths:
        image = Image.open(fpath)
        data = np.array(image) / 255.0
        fname = os.path.split(fpath)[1]
        label = int(fname.split("_")[0])
        test_datas.append(data)
        test_labels.append(label)

    test_datas = np.array(test_datas)
    test_labels = np.array(test_labels)

    print("shape of training datas: {}\tshape of training labels: {}".format(
        train_datas.shape, train_labels.shape))
    print("shape of test datas: {}\tshape of test labels: {}".format(
        test_datas.shape, test_datas.shape))

    return train_datas, train_labels, test_datas, test_labels


with tf.device('/cpu:0'):
    train_datas, train_labels, test_datas, test_labels = read_data(
        train_data_dir, test_data_dir)

    # 计算有多少类图片
    num_classes = len(set(train_labels))
    num_train_datas = len(train_datas)

    # 定义Placeholder，存放输入和标签
    datas_placeholder = tf.placeholder(tf.float32, [None, 32, 32, 3])
    labels_placeholder = tf.placeholder(tf.int32, [None])

    # 存放DropOut参数的容器，训练时为0.25，测试时为0
    dropout_placeholdr = tf.placeholder(tf.float32)

    # 定义卷积层, 20个卷积核, 卷积核大小为5，用Relu激活
    conv0 = tf.layers.conv2d(datas_placeholder, 10, 3, activation=tf.nn.relu)
    # 定义max-pooling层，pooling窗口为2x2，步长为2x2
    pool0 = tf.layers.max_pooling2d(conv0, [2, 2], [2, 2])

    # 定义卷积层, 40个卷积核, 卷积核大小为4，用Relu激活
    conv1 = tf.layers.conv2d(pool0, 20, 5, activation=tf.nn.relu)
    # 定义max-pooling层，pooling窗口为2x2，步长为2x2
    pool1 = tf.layers.max_pooling2d(conv1, [2, 2], [2, 2])

    conv2 = tf.layers.conv2d(pool1, 20, 1, activation=tf.nn.relu)

    # 将3维特征转换为1维向量
    flatten = tf.layers.flatten(conv2)

    # 全连接层，转换为长度为100的特征向量
    fc = tf.layers.dense(flatten, 100, activation=tf.nn.relu)

    # 加上DropOut，防止过拟合
    dropout_fc = tf.layers.dropout(fc, dropout_placeholdr)

    # 未激活的输出层
    logits = tf.layers.dense(dropout_fc, num_classes)

    summary_logits = tf.summary.histogram('logits', logits)

    correct_prediction = tf.equal(
        tf.argmax(logits, 1), tf.argmax(tf.one_hot(labels_placeholder, num_classes), 1))

    training_set_accuracy = tf.reduce_mean(
        tf.cast(correct_prediction, tf.float32))
    summary_training_set_accuracy = tf.summary.scalar(
        "training_set_accuracy", training_set_accuracy)

    test_set_accuracy = tf.reduce_mean(
        tf.cast(correct_prediction, tf.float32))
    summary_test_set_accuracy = tf.summary.scalar(
        "test_set_accuracy", test_set_accuracy)

    # 利用交叉熵定义损失
    losses = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.one_hot(labels_placeholder, num_classes),
        logits=logits
    )
    # 平均损失
    mean_loss = tf.reduce_mean(losses)
    summary_mean_loss = tf.summary.scalar("mean_loss", mean_loss)

    # 定义优化器，指定要优化的损失函数
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(losses)

    # 用于保存和载入模型
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # try:
        #     saver.restore(sess, model_path+"-2000")
        # except:
        #     pass
        # 如果是训练，初始化参数
        write = tf.summary.FileWriter('D:\\tflog\\log_simple', sess.graph)

        p_feed_dict = {
            datas_placeholder: test_datas,
            labels_placeholder: test_labels,
            dropout_placeholdr: 0
        }

        for step in range(3001):
            pre_index = int((step % (num_train_datas/100)) * 100)
            batch_x = train_datas[pre_index:pre_index+100]
            batch_y = train_labels[pre_index:pre_index+100]

            train_feed_dict = {
                datas_placeholder: batch_x,
                labels_placeholder: batch_y,
                dropout_placeholdr: 0.25
            }

            _, mean_loss_val, summary_mean_loss_val,summary_logits_val = sess.run(
                [optimizer, mean_loss, summary_mean_loss,summary_logits], feed_dict=train_feed_dict)

            if step % 10 == 0:
                test_feed_dict = {
                    datas_placeholder: batch_x,
                    labels_placeholder: batch_y,
                    dropout_placeholdr: 0
                }
                accuracy_val, summary_test_set_accuracy_val = sess.run(
                    [test_set_accuracy, summary_test_set_accuracy], feed_dict=p_feed_dict)

                train_accuracy_val, summary_training_set_accuracy_val = sess.run(
                    [training_set_accuracy, summary_training_set_accuracy], feed_dict=test_feed_dict)

                write.add_summary(summary_mean_loss_val, step)
                write.add_summary(summary_training_set_accuracy_val, step)
                write.add_summary(summary_test_set_accuracy_val, step)
                write.add_summary(summary_logits_val, step)

                print("step = {}\tmean loss = {}\ttrain accuracy = {:.2f}\tglobal accuracy = {:.2f}".format(
                    step, mean_loss_val, train_accuracy_val, accuracy_val))
            # if step % 1000 == 0:
            #     saver.save(sess, model_path, global_step=step)
        saver.save(sess, model_path, global_step=3000)
        print("training is over,save model into:{}".format(model_path))
