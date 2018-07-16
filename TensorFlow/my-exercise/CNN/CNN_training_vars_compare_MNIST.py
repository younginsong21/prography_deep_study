"""
weights, bias 초기화 방법 & dropout에 따라 정확도가 어떻게 달라지는지 궁금해서 쓴 코드

초기화 방식
weights: random_normal, truncated_normal, xavier, he
bias: random_normal, zero
"""

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
tf.set_random_seed(777)

# 데이터셋 로딩
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# weight 만들어주는 함수
def get_weights(weights_type, weight_shape, name):
    """
    :param weights_type: 어떤 초기화 방식을 이용한 weights인지
    :param weight_shape: weights의 shape
    :return: 인자에 맞는 weight variable
    """
    if weights_type == "random":
        return tf.Variable(tf.random_normal(weight_shape, stddev=0.01), name="random"+name)
    elif weights_type == "truncated":
        return tf.Variable(tf.truncated_normal(weight_shape, stddev=0.01), name="trunc"+name)
    elif weights_type == "xavier":
        return tf.get_variable("xa_weight"+name, shape=weight_shape,
                               initializer=tf.contrib.layers.xavier_initializer())
    # cs231에서 relu activation function에서는 he initializer가 잘 작동한다고..
    # 출처: https://smist08.wordpress.com/tag/he-initialization/
    elif weights_type == "he":
        return tf.get_variable("he_weight"+name, shape=weight_shape,
                               initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
    else:
        return None

def get_bias(bias_type, bias_shape, name):
    """
    :param bias_type: 어떤 초기화 방식을 이용한 bias인지
    :param bias_shape: bias의 shape
    :return: 인자에 맞는 bias variable
    """
    if bias_type == "random":
        return tf.Variable(tf.random_normal(bias_shape, stddev=0.01), name="random"+name)
    elif bias_type == "zero":
        return tf.Variable(tf.zeros(bias_shape), name="zero"+name)
    else:
        return None

def build_net(weights_type, bias_type):
    # input placeholders
    X = tf.placeholder(tf.float32, [None, 28*28*1])
    X_img = tf.reshape(X, [-1, 28, 28, 1])
    Y = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32) #for dropout

    """
    네트워크 구조는 Conv-Pooling-Dropout 3계층, FC 2계층으로 이루어짐.
    CP layer1 (28, 28, 1) -> (28, 28, 32) -> (14, 14, 32)
    CP layer2 (14, 14, 32) -> (14, 14, 64) -> (7, 7, 64)
    CP layer3 (7, 7, 64) -> (7, 7, 128) -> (4, 4, 128)
    FC layer4 (?, 4*4*128) -> (?, 625)
    FC layer5 (?, 625) -> (?, 10)
    """
    # Layer1
    W1 = get_weights(weights_type, [3, 3, 1, 32], "1")
    L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
    L1 = tf.nn.relu(L1)
    L1 = tf.nn.max_pool(L1 , ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1],
                                padding='SAME')
    L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

    # Layer2
    W2 = get_weights(weights_type, [3, 3, 32, 64], "2")
    L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
    L2 = tf.nn.relu(L2)
    L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1],
                                padding='SAME')
    L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

    # Layer3
    W3 = get_weights(weights_type, [3, 3, 64, 128], "3")
    L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
    L3 = tf.nn.relu(L3)
    L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME')
    L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
    L3 = tf.reshape(L3, [-1, 4*4*128]) # FC layer에 넣기 위해 flatten

    # FC layer 4
    W4 = get_weights(weights_type, [4*4*128, 625], "w4")
    b4 = get_bias(bias_type, [625], "b4")
    L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
    L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

    # FC layer 5
    W5 = get_weights(weights_type, [625, 10], "w5")
    b5 = get_bias(bias_type, [10], "b5")
    logits = tf.matmul(L4, W5) + b5

    return X, Y, logits, keep_prob

user_weight = input("weight 초기화 방법 입력>")
user_bias = input("bias 초기화 방법 입력>")
X, Y, logits, keep_prob = build_net(user_weight, user_bias)

inner_cost = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                     labels=Y)
cost = tf.reduce_mean(inner_cost)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

# Launch graph and start training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('Learning started. It takes sometime.')
    for epoch in range(15):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / 100)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            c, _ = sess.run([cost, optimizer],
                            feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.7})
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

    print('Learning Finished!')

    # test set으로 학습된 모델을 검증 및 정확도 출력
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print('\n\nAccuracy:', sess.run(accuracy,
                                    feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1}))


"""
sungkim- Lab11의 deepCNN과 동일한 구조
(기존 random, random의 경우 0.9938)
--------------------------------
weight 초기화 방법 입력>he
bias 초기화 방법 입력>zero

Accuracy: 0.9944 
--------------------------------
weight 초기화 방법 입력>he
bias 초기화 방법 입력>random

Accuracy: 0.9943
--------------------------------
weight 초기화 방법 입력>xavier
bias 초기화 방법 입력>zero

Accuracy: 0.9935
--------------------------------
weight 초기화 방법 입력>xavier
bias 초기화 방법 입력>random

Accuracy: 0.9926
--------------------------------
weight 초기화 방법 입력>random
bias 초기화 방법 입력>zero

Accuracy: 0.9931
--------------------------------
weight 초기화 방법 입력>random
bias 초기화 방법 입력>random

Accuracy: 0.9927
--------------------------------
weight 초기화 방법 입력>truncated
bias 초기화 방법 입력>zero

Accuracy: 0.9931
--------------------------------
weight 초기화 방법 입력>truncated
bias 초기화 방법 입력>random

Accuracy: 0.9941
--------------------------------
"""