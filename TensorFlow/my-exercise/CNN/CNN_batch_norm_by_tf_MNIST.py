"""
CS231n에서 소개된 batch normalization 기법 연습 코드
--> Keras 제공 함수 사용하지 않고 구현
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
tf.set_random_seed(777)

# input placeholders
X = tf.placeholder(tf.float32, [None, 28*28*1])
X_img = tf.reshape(X, [-1, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])
training = tf.placeholder(tf.bool)

"""
네트워크 구조는 Conv-Pooling-Dropout 3계층, FC 2계층으로 이루어짐.
CP layer1 (28, 28, 1) -> (28, 28, 32) -> (14, 14, 32)
CP layer2 (14, 14, 32) -> (14, 14, 64) -> (7, 7, 64)
CP layer3 (7, 7, 64) -> (7, 7, 128) -> (4, 4, 128)
FC layer4 (?, 4*4*128) -> (?, 625)
FC layer5 (?, 625) -> (?, 10)
"""

###### CP layer1 ######
conv1 = tf.layers.conv2d(inputs=X_img,
                             filters=32,
                             kernel_size=[3, 3],
                             padding='SAME',
                             activation=tf.nn.relu)                     # convolutional layer
conv1 = tf.layers.batch_normalization(conv1, training=training)     # batch normalization
pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                    pool_size=[2, 2],
                                    padding='SAME',
                                    strides=2)                          # pooling layer
dropout1 = tf.layers.dropout(inputs=pool1, rate=0.75, training=training) # dropout

###### CP layer2 ######
conv2 = tf.layers.conv2d(inputs=dropout1,
                             filters=64,
                             kernel_size=[3, 3],
                             padding='SAME',
                             activation=tf.nn.relu)  # convolutional layer
conv2 = tf.layers.batch_normalization(conv2, training=training)  # batch normalization
pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                    pool_size=[2, 2],
                                    padding='SAME',
                                    strides=2)  # pooling layer
dropout2 = tf.layers.dropout(inputs=pool2, rate=0.75, training=training)  # dropout

###### CP layer3 ######
conv3 = tf.layers.conv2d(inputs=dropout2,
                             filters=128,
                             kernel_size=[3, 3],
                             padding='SAME',
                             activation=tf.nn.relu)  # convolutional layer
conv3 = tf.layers.batch_normalization(conv3, training=training)  # batch normalization
pool3 = tf.layers.max_pooling2d(inputs=conv3,
                                    pool_size=[2, 2],
                                    padding='SAME',
                                    strides=2)  # pooling layer
dropout3 = tf.layers.dropout(inputs=pool3, rate=0.75, training=training)  # dropout

flatten = tf.reshape(dropout3, shape=[-1, 4 * 4 * 128])

###### FC layer1 ######
dense4 = tf.layers.dense(inputs=flatten,
                             units=625,
                             activation=tf.nn.relu)
dense4 = tf.layers.batch_normalization(dense4, training=training)  # batch normalization
dropout4 = tf.layers.dropout(inputs=dense4, rate=0.5, training=training)

###### FC layer2 ######
logits = tf.layers.dense(inputs=dropout4, units=10)

# hyper parameter
lr = 0.001
epochs = 20

# 데이터셋 로딩
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# cost 정의
inner_cost = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
cost = tf.reduce_mean(inner_cost)

# optimizer 정의
optimizer = tf.train.AdamOptimizer(learning_rate=lr)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(cost)

# launch graph & 학습 시작
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('Learning started. It takes sometime.')
    for epoch in range(epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / 100)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            c, _ = sess.run([cost, train_op],
                            feed_dict={X: batch_xs, Y: batch_ys, training:True})
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

    print('Learning Finished!')

    # test set으로 학습된 모델을 검증 및 정확도 출력
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    acc  = sess.run([accuracy],
                      feed_dict={X: mnist.test.images, Y: mnist.test.labels,
                                                       training: False})
    print('\n\nAccuracy:', acc)