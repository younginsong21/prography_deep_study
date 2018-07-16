import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import matplotlib
import os

tf.set_random_seed(777)  # reproducibility

'''
input dim = 5(each day's open, high, low, volume, close value)
sequence = 7 (7days)
output dim = 1 (8day's close value)
'''

# normalize function
def MinMaxScalar(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-7)

# hyper parameter
seq_length = 7
input_dim = 5
hidden_dim = 10   # 내 마음대로 세팅
output_dim = 1
learning_rate = 0.01
iteration = 500

# Load data
xy = np.loadtxt('data-02-stock_daily.csv', delimiter=',')
xy = xy[::-1] # reverse order

xy = MinMaxScalar(xy)                       # value normalize
x = xy                                      # x: (open, high, low, volume, close)
y = xy[:, [-1]] # close value as label      # y: (close)

dataX = []
dataY = []

# window 방식으로 7개씩 끊어서 가져오기
for i in range(0, len(y) -seq_length):
    _x = x[i: i+seq_length]
    _y = y[i + seq_length]                  # 다음 날의 close 값
    print(_x, '->', _y)
    dataX.append(_x)
    dataY.append(_y)

# dataX와 dataY의 길이는 동일해야함
#print("len(dataX)", int(len(dataX)))
#print("len(dataY)", int(len(dataY)))


# training set / test set 분리
# 전체 데이터의 70%을 training set으로, 나머지를 test set으로 사용
train_size = int(len(dataY)*0.7)
test_size = len(dataY) - train_size
trainX, testX = np.array(dataX[:train_size]), np.array(dataX[train_size:])
trainY, testY = np.array(dataY[:train_size]), np.array(dataY[train_size:])

# input placeholder
X = tf.placeholder(tf.float32, [None, seq_length, input_dim])
Y = tf.placeholder(tf.float32, [None, output_dim])

# Setting LSTM cell
cell = rnn.BasicLSTMCell(num_units=hidden_dim,
                         state_is_tuple=True)       # 바로 output으로 쓸게 아니고 FC에 넣을거임

outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

# outputs[:, -1] --> 마지막 cell의 output만 FC에 넣을거기 때문에
# --> 이것도 내 마음대로 인지?
Y_pred = tf.contrib.layers.fully_connected(outputs[:, -1],
                                           output_dim,
                                           activation_fn=None)

# cost & optimizer setting
loss = tf.reduce_sum(tf.square(Y_pred-Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    _, l = sess.run([train, loss],
                    feed_dict={X:trainX, Y:trainY})
    print(i, l)

testPredict = sess.run(Y_pred,
                       feed_dict={X:testX})

import matplotlib.pyplot as plt
plt.plot(testY)
plt.plot(testPredict)
plt.show()