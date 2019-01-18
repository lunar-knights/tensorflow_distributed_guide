# coding=utf-8

import time
import tempfile
import math
import random
import tensorflow as tf
import numpy as np
from PIL import Image
import os


flags = tf.app.flags
IMAGE_PIXELS = 32

abspath = os.path.split(os.path.realpath(__file__))[0]
# 数据文件夹
data_dir = abspath + "/data"
# 模型文件路径
log_dir = "/d/tflog/log_tmp"
train_data_dir = abspath + "/traindata"
test_data_dir = abspath + "/testdata"

flags.DEFINE_string('data_dir', data_dir,
                    'Directory  for storing mnist data')
flags.DEFINE_string('log_dir', log_dir,
                    'Directory  for storing mnist data')

flags.DEFINE_integer('train_steps', 3000,
                     'Number of training steps to perform')
# 定义分布式参数
# 参数服务器parameter server节点
flags.DEFINE_string('ps_hosts', '172.17.0.2:2333,172.17.0.5:2333',
                    'Comma-separated list of hostname:port pairs')
# 两个worker节点
flags.DEFINE_string('worker_hosts', '172.17.0.3:2333,172.17.0.4:2333',
                    'Comma-separated list of hostname:port pairs')
# 设置job name参数
flags.DEFINE_string('job_name', None, 'job name: worker or ps')
# 设置任务的索引
flags.DEFINE_integer('task_index', None, 'Index of task within the job')

FLAGS = flags.FLAGS

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


def main(unused_argv):
    if FLAGS.job_name is None or FLAGS.job_name == '':
        raise ValueError('Must specify an explicit job_name !')
    else:
        print('job_name : %s' % FLAGS.job_name)
    if FLAGS.task_index is None or FLAGS.task_index == '':
        raise ValueError('Must specify an explicit task_index!')
    else:
        print('task_index : %d' % FLAGS.task_index)

    ps_spec = FLAGS.ps_hosts.split(',')
    worker_spec = FLAGS.worker_hosts.split(',')

    # 创建集群
    cluster = tf.train.ClusterSpec({'ps': ps_spec, 'worker': worker_spec})
    server = tf.train.Server(
        cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    if FLAGS.job_name == u'ps':
        with tf.device("/cpu:0"):
            server.join()

    is_chief = (FLAGS.task_index == 0)

    worker_device = '/job:worker/task:%d' % FLAGS.task_index
    with tf.device(tf.train.replica_device_setter(worker_device=worker_device,
                                                  cluster=cluster
                                                  )):
        train_datas, train_labels, test_datas, test_labels = read_data(
            train_data_dir, test_data_dir)

        # train_datas, train_labels, test_datas, test_labels = read_data2(data_dir)

        global_step = tf.Variable(
            0, name='global_step', trainable=False)

        # 计算有多少类图片
        num_classes = len(set(train_labels))
        num_train_datas = len(train_datas)

        # 定义Placeholder，存放输入和标签
        datas_placeholder = tf.placeholder(tf.float32, [None, 32, 32, 3])
        labels_placeholder = tf.placeholder(tf.int32, [None])

        # 存放DropOut参数的容器，训练时为0.25，测试时为0
        dropout_placeholdr = tf.placeholder(tf.float32)

        # 定义卷积层, 20个卷积核, 卷积核大小为5，用Relu激活
        conv0 = tf.layers.conv2d(
            datas_placeholder, 10, 3, activation=tf.nn.relu)
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

        correct_prediction_test = tf.equal(
            tf.argmax(logits, 1), tf.argmax(tf.one_hot(labels_placeholder, num_classes), 1))

        training_set_accuracy = tf.reduce_mean(
            tf.cast(correct_prediction_test, tf.float32))
        summary_training_set_accuracy = tf.summary.scalar(
            "training_set_accuracy", training_set_accuracy)

        test_set_accuracy = tf.reduce_mean(
            tf.cast(correct_prediction_test, tf.float32))
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

    # worker_device = '/job:worker/task:%d/cpu:1' % FLAGS.task_index
    # with tf.device(tf.train.replica_device_setter(worker_device=worker_device,
    #                                               cluster=cluster
    #                                               )):
        # 定义优化器，指定要优化的损失函数
        optimizer = tf.train.AdamOptimizer(
            learning_rate=1e-2).minimize(losses, global_step=global_step)

        # 用于保存和载入模型
        # saver = tf.train.Saver()
        # summary_op = tf.summary.merge_all()
        # init_op = tf.global_variables_initializer()
        # log_dir = FLAGS.log_dir
        if is_chief:
            print('Worker %d: Initailizing session...' % FLAGS.task_index)
        else:
            print('Worker %d: Waiting for session to be initaialized...' %
                  FLAGS.task_index)

        sess_config = tf.ConfigProto(
            allow_soft_placement=False, log_device_placement=False)

        hooks = [tf.train.StopAtStepHook(last_step=FLAGS.train_steps)]

        mts = tf.train.MonitoredTrainingSession(
            master=server.target, is_chief=is_chief,
            hooks=hooks,
            # checkpoint_dir=log_dir,
            save_checkpoint_secs=30, config=sess_config)

        # sv = tf.train.Supervisor(is_chief=is_chief, logdir=log_dir, init_op=init_op,
        #                          global_step=global_step, saver=saver, summary_op=summary_op,
        #                          save_summaries_secs=30,
        #                          save_model_secs=30)
        # sess = sv.prepare_or_wait_for_session(
        #     server.target, config=sess_config)

        print('Worker %d: Session initialization  complete.' % FLAGS.task_index)
        with mts as sess:
            if is_chief:
                write = tf.summary.FileWriter(
                    '/d/tflog/log_write_by_hxl', sess.graph)

            local_step = 0
            p_feed_dict = {
                datas_placeholder: test_datas,
                labels_placeholder: test_labels,
                dropout_placeholdr: 0
            }
            step = 0
            while not sess.should_stop():
                pre_index = int((step % (num_train_datas/100)) * 100)
                batch_x = train_datas[pre_index:pre_index+100]
                batch_y = train_labels[pre_index:pre_index+100]

                train_feed_dict = {
                    datas_placeholder: batch_x,
                    labels_placeholder: batch_y,
                    dropout_placeholdr: 0.25
                }

                # _, mean_loss_val, step, summary_mean_loss_val = sess.run(
                #     [optimizer, mean_loss, global_step, summary_mean_loss], feed_dict=train_feed_dict)

                if (not is_chief and step>100) or is_chief:
                    _, mean_loss_val, step, summary_mean_loss_val = sess.run(
                        [optimizer, mean_loss, global_step, summary_mean_loss], feed_dict=train_feed_dict)
                else:
                    step = sess.run(global_step, feed_dict=train_feed_dict)
                    continue

                if local_step % 10 == 0 and step < FLAGS.train_steps-10:
                    test_feed_dict = {
                        datas_placeholder: batch_x,
                        labels_placeholder: batch_y,
                        dropout_placeholdr: 0
                    }
                    accuracy_val, summary_test_set_accuracy_val = sess.run(
                        [test_set_accuracy, summary_test_set_accuracy], feed_dict=p_feed_dict)
                    train_accuracy_val, summary_training_set_accuracy_val = sess.run(
                        [training_set_accuracy, summary_training_set_accuracy], feed_dict=test_feed_dict)
                    print('Worker %d: traing step %d dome (global step:%d)' %
                          (FLAGS.task_index, local_step, step))
                    print("mean loss = {}\ttrain accuracy = {:.2f}\tglobal accuracy = {:.2f}".format(
                        mean_loss_val, train_accuracy_val, accuracy_val))

                    if is_chief:
                        write.add_summary(summary_mean_loss_val, step)
                        write.add_summary(
                            summary_training_set_accuracy_val, step)
                        write.add_summary(summary_test_set_accuracy_val, step)

                local_step += 1
                # if step >= FLAGS.train_steps:
                #     print("training is over,save model into:{}".format('/d/tflog/log_write_by_hxl'))
                #     break
            print("training is over,save model into:{}".format('/d/tflog/log_write_by_hxl'))
            # sess.close()


if __name__ == "__main__":
    tf.app.run()
