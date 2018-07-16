import tensorflow as tf
import numpy as np
tf.set_random_seed(777)  # for reproducibility

# 데이터 로딩, X-Y 분리
dataXY = np.loadtxt('data-01-test-score.csv', delimiter=',', dtype=np.float32)
dataX = dataXY[:, 0:-1]
dataY = dataXY[:, [-1]]

# train data(80%), test data(20%) 분리
train_len = int(len(dataX) * 0.8)
trainX, testX = dataX[:train_len], dataX[train_len:]
print("trainX.shape:", trainX.shape)
print("testX.shape:", testX.shape)
trainY, testY = dataY[:train_len], dataY[train_len:]
print("trainY.shape:", trainY.shape)
print("testY.shape:", testY.shape)

# input placeholders
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

# training variables
W = tf.Variable(tf.random_normal([3, 1]))
b = tf.Variable(tf.random_normal([1]))

# hypothesis & cost
hypothesis = tf.matmul(X, W) + b
cost = tf.reduce_mean(tf.square(hypothesis-Y))

# optimizer setting
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(cost)

# launch session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# training
for step in range(6001):
    c, _ = sess.run([cost, optimizer],
                    feed_dict={X: trainX, Y: trainY})
    if step % 500 == 0:
        print("step:", step, " cost:", c)

# validating
pred = sess.run(hypothesis,
                feed_dict={X: testX})

for i in range(len(testX)):
    print("pred:", pred[i], " actual:", testY[i])
