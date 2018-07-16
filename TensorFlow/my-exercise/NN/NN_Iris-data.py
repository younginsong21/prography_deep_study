import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers

tf.set_random_seed(777)

training_epoch = 10000
num_classes = 3

# 데이터 로딩, 셔플링
dataXY = np.loadtxt('data-Iris.csv', delimiter=',', dtype=np.float32)
np.random.shuffle(dataXY)

# XY 분리, column 재정리
dataX = dataXY[:, :-1]
dataY = dataXY[:, [-1]]

# 데이터 정규화 과정
# zero-centered
dataX -= np.mean(dataX, axis=0)
# normalization
dataX /= np.std(dataX, axis=0)

# training set, test set 분리 (비율은 9:1)
train_len = int(len(dataX) * 0.9)
trainX, testX = dataX[:train_len], dataX[train_len:]
trainY, testY = dataY[:train_len], dataY[train_len:]

# input placeholders
X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.int32, shape=[None, 1])
Y_one_hot = tf.one_hot(Y, num_classes)
Y_one_hot = tf.reshape(Y_one_hot, [-1, num_classes])
keep_prob = tf.placeholder(tf.float32)

# feed_dict 설정
training_dict = {X: trainX, Y: trainY, keep_prob:0.7}
testing_dict = {X: testX, Y: testY, keep_prob:1.0}

# Buliding Layers

# Layer 1
W1 = tf.get_variable(name="W1", shape=[4, 12], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([12]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

# Layer 2
W2 = tf.get_variable(name="W2", shape=[12, 12], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([12]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

# Layer 3
W3 = tf.get_variable(name="W3", shape=[12, num_classes],
                     initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([num_classes]))
hypothesis = tf.matmul(L2, W3) + b3

# Cost, optimizer setting
inner_cost = tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis,
                                                     labels=Y_one_hot)
cost = tf.reduce_mean(inner_cost)
optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.1).minimize(cost)

# Accuracy setting
is_equal = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(is_equal, tf.float32))

# Launch graphs
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Start training
    for epoch in range(training_epoch):
        c, _ = sess.run([cost, optimizer],
                        feed_dict=training_dict)
        if epoch % 2000 == 0:
            print("Epoch:", epoch, " Cost:", c)


    # Get accuracy
    acc = sess.run(accuracy, feed_dict=testing_dict)
    print("Accuracy:", acc)

'''
excel shuffle 매번 다르게 되어서 매번 다른 결과값이 나옴 --> 이런 데이터 어떻게 전처리하는 것이 좋은지?
Epoch: 0  Cost: 2.0723093
Epoch: 2000  Cost: 1.0372391
Epoch: 4000  Cost: 0.71424663
Epoch: 6000  Cost: 0.6085352
Epoch: 8000  Cost: 0.48824555
Accuracy: 0.93333334
'''

'''
Normalization 이후에
Epoch: 0  Cost: 1.4570078
Epoch: 2000  Cost: 0.7989205
Epoch: 4000  Cost: 0.5634061
Epoch: 6000  Cost: 0.43790233
Epoch: 8000  Cost: 0.29800516
Accuracy: 1.0
'''