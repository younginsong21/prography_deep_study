import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Hyper parameter
learning_rate = 0.001
training_epochs = 15
batch_size = 100

# Input placeholders
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)     #training시에는 0.7, testing시에는 1.0

# setting dict
train_dict = {X: mnist.train.images, Y: mnist.train.labels, keep_prob: 0.7}
test_dict = {X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1.0}

# Building layers
# Layer1
W1 = tf.get_variable("W1", shape=[784, 512],
                     initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([512]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

# Layer2
W2 = tf.get_variable("W2", shape=[512, 512],
                     initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([512]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

# Layer3
W3 = tf.get_variable("W3", shape=[512, 512],
                     initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([512]))
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

# Layer4
W4 = tf.get_variable("W4", shape=[512, 512],
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([512]))
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

# Final layer5
W5 = tf.get_variable("W5", shape=[512, 10],
                     initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(L4, W5) + b5  #여기에선 activation 안함!!

# cost, optimizer 설정
inner_cost = tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis,
                                                     labels=Y)
cost = tf.reduce_mean(inner_cost)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Launch graphs
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Learning start
print("Learning started!")
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size) #총 batch loading 횟수

    for i in range(total_batch):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        c, _ = sess.run([cost, optimizer],
                        feed_dict=train_dict)
        avg_cost += c/total_batch

    print("Epoch:", epoch, "cost:", avg_cost)
# Learning finish

# Accuracy report
is_equal = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_equal, tf.float32))
acc = sess.run(accuracy, feed_dict=test_dict)
print("Accuracy:", acc)


'''
Accuracy: 93 정도
'''




