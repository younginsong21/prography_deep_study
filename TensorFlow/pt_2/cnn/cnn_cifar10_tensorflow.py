# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from tensorflow.python.keras._impl.keras.datasets.cifar10 import load_data
tf.set_random_seed(777)

# Hyper parameter
learning_rate = 1e-3

# Input placeholders
X = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
Y = tf.placeholder(tf.int32, shape=[None, 1])
Y_one_hot = tf.one_hot(Y, 10)
Y_one_hot = tf.reshape(Y_one_hot, [-1, 10])
is_training = tf.placeholder(tf.bool)

 # CNN 설계
'''
Layer 1
    image input: (?, 32, 32, 3)
    filter shape: 4*4, 6장
    conv activation: relu
    conv output: (32, 32, 6)
    pool output: (16, 16, 6)
    keep_prob = 0.7
'''
conv1 = tf.layers.conv2d(inputs=X,
                             filters=6,
                             kernel_size=[4, 4],
                             padding='SAME',
                             activation=tf.nn.relu)

pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                    pool_size=[2, 2],
                                    padding='SAME',
                                    strides=2)
dropout1 = tf.layers.dropout(inputs=pool1,
                                 rate=0.7,
                                 training=is_training)  # training 시에 True로

'''
Layer 2
    image input: (?, 16, 16, 6)
    filter shape: 4*4, 12
    conv activation: relu
    conv output: (16, 16, 12)
    pool output: (8, 8, 12)
    keep_prob = 0.7
'''
conv2 = tf.layers.conv2d(inputs=dropout1,
                             filters=12,
                             kernel_size=[4, 4],
                             padding='SAME',
                             activation=tf.nn.relu)
pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                    pool_size=[2, 2],
                                    padding='SAME',
                                    strides=2)
dropout2 = tf.layers.dropout(inputs=pool2,
                                 rate=0.7,
                                 training=is_training)

'''
Layer 3
    image input: (?, 8, 8, 12)
    filter shape: 4*4, 24
    conv activation: relu
    conv output: (8, 8, 24)
    pool output: (4, 4, 24)
    flatten output = (?, 4*4*24)
    keep_prob = 0.7
'''
conv3 = tf.layers.conv2d(inputs=dropout2,
                             filters=24,
                             kernel_size=[4, 4],
                             padding='SAME',
                             activation=tf.nn.relu)

pool3 = tf.layers.max_pooling2d(inputs=conv3,
                                    pool_size=[2, 2],
                                    padding='SAME',
                                    strides=2)
dropout3 = tf.layers.dropout(inputs=pool3,
                                 rate=0.7,
                                 training=is_training)
flat = tf.reshape(dropout3, [-1, 4 * 4 * 24])

'''
FC Layer 1
    input : (?, 4*4*24)
    activation = relu
    output = 100
    keep_prob = 0.5
'''
dense4 = tf.layers.dense(inputs=flat,
                             units=100,
                             activation=tf.nn.relu)

dropout4 = tf.layers.dropout(inputs=dense4,
                                 rate=0.7,
                                 training=is_training)

'''
FC layer 2
input: (?, 100)
activation: XXXXXXX
output: (?, 10)
'''
hypothesis = tf.layers.dense(inputs=dropout4, units=10)

# cost, optimizer 설정
inner_cost = tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis,
                                                         labels=Y_one_hot)
cost = tf.reduce_mean(inner_cost)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# accuracy 설정
is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))


# CIFAR-10 데이터를 다운로드
(x_train, y_train), (x_test, y_test) = load_data()


# 세션 열고 학습 진행
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 학습 시작
    for epoch in range(500):
        # batch를 총 100개, 즉 500개씩 작은 파일로 쪼개서 학습 시작
        for batch_num in range(100):
            batch_x_train = np.asarray(x_train[500*batch_num: 500*batch_num+500])
            batch_y_train = np.asarray(y_train[500*batch_num: 500*batch_num+500])
            c, _ = sess.run([cost, optimizer], feed_dict={X:batch_x_train,
                                                        Y:batch_y_train,
                                                        is_training:True})

        if epoch % 10 == 0:
            print("Epoch:", epoch, " cost:", c)

    # 학습 끝

    # Test set 전체에 대한 정확도 출력
    acc = sess.run(accuracy, feed_dict={X: x_test,
                                        Y: y_test,
                                        is_training: False})
    print("Accuracy:", acc)  # test set에 대한 정확도 출력


'''
initializer 명시 안했을 때:
Accuracy: 0.5397

initializer 명시했을때: (kernel_initializer: he)-> 시간 없어서 다 못돌려봄 ㅜㅜ 나중에 돌려보기

'''