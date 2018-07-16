import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)  # reproducibility

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset

# hyper parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100

class Model:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()


    # network building
    def _build_net(self):
        with tf.variable_scope(self.name):
            # for drop-out
            self.training = tf.placeholder(tf.bool)

            # input X placeholder
            self.X = tf.placeholder(tf.float32, [None, 784])

            # X --> image (28*28, gray color)
            self.X_img = tf.reshape(self.X, [-1, 28, 28, 1])

            # label Y placeholder
            self.Y = tf.placeholder(tf.float32, [None, 10])

            '''
            layer 1:
            image input = (?, 28, 28, 1)
            filter shape = 3 X 3, 32장
            conv activation = relu
            conv output = (?, 28, 28, 32)
            pool output = (?, 14, 14, 32)
            keep_prob=0.7
            '''
            conv1 = tf.layers.conv2d(inputs=self.X_img,
                                     filters=32,
                                     kernel_size=[3, 3],
                                     padding='SAME',
                                     activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                            pool_size=[2, 2],
                                            padding='SAME',
                                            strides=2)
            dropout1 = tf.layers.dropout(inputs=pool1, rate=0.7, training=self.training)

            '''
            layer 2:
            image input = (?, 14, 14, 32)
            conv activation = relu
            filter shape = 3 X 3, 64장
            conv output = (?, 14, 14, 64)
            pool output = (?, 7, 7, 64)
            keep_prob=0.7
            '''
            conv2 = tf.layers.conv2d(inputs=dropout1,
                                     kernel_size=[3, 3],
                                     filters=64,
                                     padding='SAME',
                                     activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                            pool_size=[2, 2],
                                            padding='SAME',
                                            strides=2)
            dropout2 = tf.layers.dropout(inputs=pool2, rate=0.7, training=self.training)
            '''
            layer 3:
            image input = (?, 7, 7, 64)
            conv activation = relu
            filter shape = 3 X 3, 128장
            conv output = (?, 7, 7, 128)
            pool output = (?, 4, 4, 128)
            flatten output = (?, 4*4*128)
            keep_prob=0.7
            '''
            conv3 = tf.layers.conv2d(inputs=dropout2,
                                     kernel_size=[3, 3],
                                     filters=128,
                                     padding='SAME',
                                     activation=tf.nn.relu)
            pool3 = tf.layers.max_pooling2d(inputs=conv3,
                                            pool_size=[2, 2],
                                            padding='SAME',
                                            strides=2)
            dropout3 = tf.layers.dropout(inputs=pool3, rate=0.7, training=self.training)
            flat = tf.reshape(dropout3, [-1, 4*4*128])

            '''
            fully connected layer1:
            input = (?, 4*4*,128)
            activation = relu
            output = 625
            keep_prob=0.5
            '''
            dense4 = tf.layers.dense(inputs=flat,
                                     units=625,
                                     activation=tf.nn.relu)
            dropout4 = tf.layers.dropout(inputs=dense4, rate=0.5, training=self.training)

            '''
            fully connected layer:
            input = (?, 625)
            activation = X
            output = (?, 10)
            '''
            self.logits = tf.layers.dense(inputs=dropout4, units=10)


        #define cost, optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,
                                                                           labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    def get_predict(self, x_test):
        return self.sess.run(self.logits, feed_dict={self.X:x_test, self.training:False})


    def get_accuracy(self, x_test, y_test):
        return self.sess.run(self.accuracy, feed_dict={self.X:x_test, self.Y:y_test, self.training:False})

    def train(self):
        print('Learning Started!')
        # train my model
        for epoch in range(training_epochs):
            avg_cost = 0
            total_batch = int(mnist.train.num_examples / batch_size)

            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                c, _ = self.sess.run([self.cost, self.optimizer],
                                     feed_dict={self.X:batch_xs, self.Y:batch_ys, self.training:True})
                avg_cost += c / total_batch

            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

        print('Learning Finished!')


sess = tf.Session()
model = Model(sess, "m1")
sess.run(tf.global_variables_initializer())

model.train() #이거 그냥 내가 수정함

# Test model and check accuracy
print('Accuracy:', model.get_accuracy(mnist.test.images, mnist.test.labels))

