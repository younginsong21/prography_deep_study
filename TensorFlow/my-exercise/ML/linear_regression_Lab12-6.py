# lab12-6 stock data RNN --> linear regression implement


import tensorflow as tf
import numpy as np
tf.set_random_seed(777)  # reproducibility

# normalize function
def MinMaxScalar(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-7)

# Load data
xy = np.loadtxt('data-02-stock_daily.csv', delimiter=',')
xy = MinMaxScalar(xy)
xy = xy[::-1] # reverse order
print("xy.shape:", xy.shape)

dataX = xy[:, 0:-1]  # x = (open, high, low, volume)
dataY = xy[:, [-1]]    # y = (close)
print("dataX.shape:", dataX.shape)
print("dataY.shape:", dataY.shape)

# training set / test set 분리
# 전체 데이터의 70%을 training set으로, 나머지를 test set으로 사용
train_size = int(len(dataY)*0.7)
test_size = len(dataY) - train_size
trainX, testX = np.array(dataX[:train_size]), np.array(dataX[train_size:])
print("trainX.shape:", trainX.shape)
print("testX.shape:", testX.shape)
trainY, testY = np.array(dataY[:train_size]), np.array(dataY[train_size:])
print("trainY.shape:", trainY.shape)
print("testY.shape:", testY.shape)

# input placeholders
X = tf.placeholder(tf.float32, [None, 4])
Y = tf.placeholder(tf.float32, [None, 1])

# weight, bias
W = tf.Variable(tf.random_normal([4, 1]))
b = tf.Variable(tf.random_normal([1]))

# hypothesis, cost setting
hypothesis = tf.matmul(X, W) + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# session launch
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# training start
for i in range(1000):
    c, _ = sess.run([cost, optimizer],
                    feed_dict={X:trainX, Y:trainY})
    print(i, c)

# validating start
pred = sess.run(hypothesis, feed_dict={X:testX})
for i in range(len(pred)):
    print("Pred:", pred[i], " Actual:", testY[i])
