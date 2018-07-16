import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

# data normalization function
def normalization(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)

# 데이터 로딩, 셔플링
dataXY = np.loadtxt('winequality-red-pre.csv', delimiter=',', dtype=np.float32)
np.random.shuffle(dataXY) #데이터 셔플링

# XY 분리, X 데이터 부분만 normalization
dataX = dataXY[:, :-1]
dataX = normalization(dataX)
dataY = dataXY[:, [-1]]

# training set, test set 분리 (비율은 9:1)
train_len = int(len(dataX) * 0.9)
trainX, testX = dataX[:train_len], dataX[train_len:]
trainY, testY = dataY[:train_len], dataY[train_len:]

num_classes = 6

# input placeholders
X = tf.placeholder(tf.float32, shape=[None, 11])
Y = tf.placeholder(tf.int32, shape=[None, 1])      # 0-5의 총 6개 class
Y_one_hot = tf.one_hot(Y, num_classes)             # one hot 처리
Y_one_hot = tf.reshape(Y_one_hot, [-1, num_classes])

# train variables
W = tf.Variable(tf.random_normal([11, num_classes]))
b = tf.Variable(tf.random_normal([num_classes]))

# logit, hypothesis, loss 정의
logit = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logit)
inner_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logit,
                                                     labels=Y_one_hot)
loss = tf.reduce_mean(inner_loss)

# optimizer
learning_rate = 0.015
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
print("learing rate:", learning_rate, "\n")

# prediction, accuracy 정의
prediction = tf.argmax(hypothesis, 1)
is_correct = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# launch graph, training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # start training
    for step in range(10000):
        l, _ = sess.run([loss, optimizer],
                        feed_dict={X: trainX, Y: trainY})
        if step % 1000 == 0:
            print("step:", step, " cost:", l)
    # end of training

    # accuray report with testing set
    acc = sess.run(accuracy, feed_dict={X:testX, Y:testY})
    print("accuracy: ", acc)



'''
step: 0  cost: 2.930546
step: 1000  cost: 0.96249616
step: 2000  cost: 0.94169956
step: 3000  cost: 0.9342867
step: 4000  cost: 0.92977023
step: 5000  cost: 0.92662036
step: 6000  cost: 0.92413765
step: 7000  cost: 0.92197275
step: 8000  cost: 0.9201315
step: 9000  cost: 0.9187862
accuracy:  0.65625
'''
