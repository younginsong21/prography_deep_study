import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers

tf.set_random_seed(777)

# 데이터 로딩, 셔플링
dataXY = np.loadtxt('data-glass.csv', delimiter=',', dtype=np.float32)
np.random.shuffle(dataXY)

# XY 분리
dataX = dataXY[:, :-1]
dataY = dataXY[:, [-1]]

# 실제 csv 파일은 label이 1, 2, 3, 5, 6, 7으로 총 6개
#          변경뒤 label  0, 1, 2, 3, 4, 5
# 이를 one_hot 처리하기 적합한 형태로 바꿔주기
for i in range(len(dataY)):
    if dataY[i] in [1, 2, 3]:
        dataY[i] = dataY[i] - 1
    elif dataY[i] in [5, 6, 7]:
        dataY[i] = dataY[i] - 2

# 데이터 정규화 과정
# zero-centered
dataX -= np.mean(dataX, axis=0)
# normalization
dataX /= np.std(dataX, axis=0)

# training set, test set 분리 (비율은 9:1)
train_len = int(len(dataX) * 0.9)
trainX = dataX[:train_len]
trainY = dataY[:train_len]
testX = dataX[train_len:]
testY = dataY[train_len:]

# hyper parameter
learning_rate = 0.03
training_epoch = 20000
keep_prob = 0.7

# parameter
num_classes = 6

# input placeholders
X = tf.placeholder(tf.float32, shape=[None, 9])
Y = tf.placeholder(tf.int32, shape=[None, 1])
Y_one_hot = tf.one_hot(Y, num_classes)             # one hot 처리
Y_one_hot = tf.reshape(Y_one_hot, [-1, num_classes])
keep_prob = tf.placeholder(tf.float32)

# feed_dict 설정
training_dict = {X: trainX, Y: trainY, keep_prob:0.7}
testing_dict = {X: testX, Y: testY, keep_prob:1.0}

# Buliding Layers

# Layer 1
W1 = tf.get_variable(name="W1", shape=[9, 72], initializer=layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([72]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

# Layer 2
W2 = tf.get_variable(name="W2", shape=[72, 72], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([72]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

# Layer 3
W3 = tf.get_variable(name="W3", shape=[72, num_classes],
                     initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([num_classes]))
hypothesis = tf.matmul(L2, W3) + b3  # 여기선 activation function 없음!

# Cost, optimizer setting
inner_cost = tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis,
                                                     labels=Y_one_hot)
cost = tf.reduce_mean(inner_cost)
optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(cost)

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
        # Get accuracy
        acc = sess.run(accuracy, feed_dict=testing_dict)
        if epoch % 2000 == 0:
            print("Epoch:", epoch, " Cost:", c, " Validating accuracy:", acc)

    # Get accuracy
    acc = sess.run(accuracy, feed_dict=testing_dict)
    print("Accuracy:", acc)



'''
Normalization 이후에
Epoch: 0  Cost: 2.943504  Validating accuracy: 0.36363637
Epoch: 2000  Cost: 1.3645476  Validating accuracy: 0.6363636
Epoch: 4000  Cost: 1.0814546  Validating accuracy: 0.6818182
Epoch: 6000  Cost: 0.96277237  Validating accuracy: 0.59090906
Epoch: 8000  Cost: 0.87541026  Validating accuracy: 0.72727275
Epoch: 10000  Cost: 0.86397934  Validating accuracy: 0.72727275
Epoch: 12000  Cost: 0.73296934  Validating accuracy: 0.72727275
Epoch: 14000  Cost: 0.7230455  Validating accuracy: 0.72727275
Epoch: 16000  Cost: 0.7382713  Validating accuracy: 0.77272725
Epoch: 18000  Cost: 0.70665425  Validating accuracy: 0.72727275
Accuracy: 0.72727275
'''