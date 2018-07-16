import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

training_epoch = 10000
num_classes = 6

# 데이터 로딩, 셔플링
dataXY = np.loadtxt('data-winequality.csv', delimiter=',', dtype=np.float32)
np.random.shuffle(dataXY) #데이터 셔플링

# XY 분리, X 데이터 부분만 normalization
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
X = tf.placeholder(tf.float32, shape=[None, 11])
Y = tf.placeholder(tf.int32, shape=[None, 1])      # 0-5의 총 6개 class
Y_one_hot = tf.one_hot(Y, num_classes)             # one hot 처리
Y_one_hot = tf.reshape(Y_one_hot, [-1, num_classes])
keep_prob = tf.placeholder(tf.float32)

# feed_dict 설정
training_dict = {X: trainX, Y: trainY, keep_prob:0.7}
testing_dict = {X: testX, Y: testY, keep_prob:1.0}

# Buliding Layers

# Layer 1
W1 = tf.get_variable(name="W1", shape=[11, 22], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([22]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

# Layer 2
W2 = tf.get_variable(name="W2", shape=[22, 22], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([22]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

# Layer 3
W3 = tf.get_variable(name="W3", shape=[22, num_classes],
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
Epoch: 0  Cost: 4.71321
Epoch: 2000  Cost: 1.2718583
Epoch: 4000  Cost: 1.1753225
Epoch: 6000  Cost: 1.106722
Epoch: 8000  Cost: 1.0943114
Accuracy: 0.6
'''

'''
데어터 전처리 거친 후
Epoch: 0  Cost: 4.5973577
Epoch: 2000  Cost: 1.2292452
Epoch: 4000  Cost: 1.0741922
Epoch: 6000  Cost: 1.0107135
Epoch: 8000  Cost: 0.99584484
Accuracy: 0.60625
'''