import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)  # reproducibility


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# hyper parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100

def training():
    # train my model
    print('Learning started. It takes sometime.')
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            feed_dict = {X: batch_xs, Y: batch_ys}
            c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

    print('Learning Finished!')


def accuracy_report():
    # Test model and check accuracy
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Accuracy:', sess.run(accuracy, feed_dict={
        X: mnist.test.images, Y: mnist.test.labels}))


# input placeholders
X = tf.placeholder(tf.float32, [None, 28*28])
X_img = tf.reshape(X, [-1, 28, 28, 1]) #image input 28*28*1(black or white)
Y = tf.placeholder(tf.float32, [None, 10])

## Layer 1 ##
# filter setting : 3*3 size, # = 32
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))

# conv output--> (?, 28, 28, 32)
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME') #원본 이미지와 동일한 사이즈
L1 = tf.nn.relu(L1)

# pool output--> (?, 14, 14, 32)
# stride가 1*1일때 padding='SAME'인 경우 원본 이미지와 동일한 사이즈
# stride가 2*2이면 원본 이미지의 절반 사이즈가 된다.
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')

## layer 2 ##
# filter setting : 3*3 size, # = 364
# input image : (?, 14, 14, 32)
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))

# conv output --> (?, 14, 14, 64)
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)

# pool output --> (?, 7, 7, 64)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')

# f-c layer에 넣기 전에 일렬로 변환
L2 = tf.reshape(L2, [-1, 7*7*64])

## fully connected layer##
W3 = tf.get_variable("W3", shape=[7*7*64, 10],
                     initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]))
logits = tf.matmul(L2, W3) + b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

training()

accuracy_report()




