import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)  # reproducibility
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# hyper parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100

keep_prob = tf.placeholder(tf.float32)

def training():
    # train my model
    print('Learning started. It takes sometime.')
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, optimizer],
                            feed_dict={X: batch_xs, Y: batch_ys, keep_prob:0.7})
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

    print('Learning Finished!')


def accuracy_report():
    # Test model and check accuracy
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('\n\nAccuracy:', sess.run(accuracy,
                                feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob:1}))


# input placeholders
X = tf.placeholder(tf.float32, [None, 28*28])
X_img = tf.reshape(X, [-1, 28, 28, 1]) #image input 28*28*1(black or white)
Y = tf.placeholder(tf.float32, [None, 10])

# network 구조는 3개의 c-p 계층, fully-connected layer 2개

## Layer 1 ##
# filter setting : 3*3 size, # = 32
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))

# conv output = (?, 28, 28, 32)
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME') #원본 이미지와 동일한 사이즈
L1 = tf.nn.relu(L1)

# pool output--> (?, 14, 14, 32)
# stride가 1*1일때 padding='SAME'인 경우 원본 이미지와 동일한 사이즈
# stride가 2*2이면 원본 이미지의 절반 사이즈가 된다.
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')

L1 = tf.nn.dropout(L1, keep_prob=keep_prob) #drop out

## layer 2 ##
# image input = (?, 14, 14, 32)
# filter setting : 3*3 size, # = 64
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))

# conv output = (?, 14, 14, 64)
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)

# pool output =  (?, 7, 7, 64)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
L2 = tf.nn.dropout(L2, keep_prob=keep_prob) #drop out


## layer 3 ##
# image input = (?, 7, 7, 64)
# filter setting = 3*3 size, #=128
W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))

# conv output = (?, 7, 7, 128)
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(L3)

# pool output = (?, 4, 4, 128)
L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')

L3 = tf.nn.dropout(L3, keep_prob=keep_prob) #drop out


# flatten image
L3 = tf.reshape(L3, [-1, 4*4*128])


## fully connected layer 1 ##
W4 = tf.get_variable("W4", shape=[4*4*128, 625],
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([625]))
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob) #drop out

## fully connected layer 2 ##
W5 = tf.get_variable("W5", shape=[625, 10],
                     initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]))
logits = tf.matmul(L4, W5) + b5

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

training()

accuracy_report()




