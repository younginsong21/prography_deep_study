import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

# class 구조화
class Model():

    def __init__(self, sess):
        self.sess = sess

    # Neural network 구성
    def build_net(self):
        learning_rate = 0.001
        num_classes = 3

        # input placeholders
        self.X = tf.placeholder(tf.float32, shape=[None, 4])
        self.Y = tf.placeholder(tf.int32, shape=[None, 1])  # Iris 종류: 0-2의 총 3개 class
        Y_one_hot = tf.one_hot(self.Y, num_classes)  # one hot 처리
        self.Y_one_hot = tf.reshape(Y_one_hot, [-1, num_classes])
        self.keep_prob = tf.placeholder(tf.float32)

        # layer 1
        W1 = tf.Variable(tf.random_normal([4, 12]), name="W1")
        b1 = tf.Variable(tf.random_normal([12]))
        L1 = tf.nn.relu(tf.matmul(self.X, W1) + b1)
        L1 = tf.nn.dropout(L1, keep_prob=self.keep_prob)

        # layer 2
        W2 = tf.Variable(tf.random_normal([12, 12]), name="W2")
        b2 = tf.Variable(tf.random_normal([12]))
        L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
        L2 = tf.nn.dropout(L2, keep_prob=self.keep_prob)

        # layer 3
        W3 = tf.Variable(tf.random_normal([12, num_classes]), name="W3")
        b3 = tf.Variable(tf.random_normal([num_classes]))
        self.hypothesis = tf.matmul(L2, W3) + b3

        self.sess.run(tf.initialize_all_variables())

        # cost, optimizer 정의
        cost_inner = tf.nn.softmax_cross_entropy_with_logits(logits=self.hypothesis,
                                                             labels=self.Y_one_hot)

        self.cost = tf.reduce_mean(cost_inner)
        self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(self.cost)

        is_equal = tf.equal(tf.arg_max(self.hypothesis, 1),
                            tf.arg_max(self.Y_one_hot, 1))
        self.accuracy = tf.reduce_mean(tf.cast(is_equal, tf.float32))


    def training(self, x_data, y_data):
        return self.sess.run([self.cost, self.optimizer],
                             feed_dict={self.X: x_data,
                                        self.Y: y_data,
                                        self.keep_prob:0.7})

    def predict(self, x_data):
        return self.sess.run(self.hypothesis, feed_dict={self.X: x_data,
                                                         self.keep_prob:1.0})

    def get_accuracy(self, x_data, y_data):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_data,
                                                       self.Y: y_data,
                                                       self.keep_prob:1.0})

training_epoch = 15

# 데이터 로딩, 셔플링
dataXY = np.loadtxt('Iris.csv', delimiter=',', dtype=np.float32)
np.random.shuffle(dataXY)

# XY 분리, column 재정리
dataX = dataXY[:, :-1]
dataY = dataXY[:, [-1]]

# training set, test set 분리 (비율은 8:2)
train_len = int(len(dataX) * 0.8)
trainX, testX = dataX[:train_len], dataX[train_len:]
trainY, testY = dataY[:train_len], dataY[train_len:]

sess = tf.Session()
mymodel = Model(sess)

# building neural network
mymodel.build_net()

# training
print("Learning started")
for epoch in range(training_epoch):
    c, _ = mymodel.training(trainX, trainY)
    print("Epoch:", epoch, " Cost:", c)
print("Learning finished")

# get accuracy
print("Average accuracy:", mymodel.get_accuracy(testX, testY))




