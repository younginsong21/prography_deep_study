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
            conv output = (?, 28, 28, 32)
            pool output = (?, 14, 14, 32)
            '''
            W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
            L1 = tf.nn.conv2d(self.X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
            L1 = tf.nn.relu(L1)
            L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME')

            '''
            layer 2:
            image input = (?, 14, 14, 32)
            filter shape = 3 X 3, 64장
            conv output = (?, 14, 14, 64)
            pool output = (?, 7, 7, 64)
            flatten output = (?, 7*7*64)
            '''
            W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
            L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
            L2 = tf.nn.relu(L2)
            L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME')
            L2 = tf.reshape(L2, [-1, 7*7*64])

            '''
            fully connected layer:
            input = (?, 7*7*64)
            output = (?, 10)
            '''
            W3 = tf.get_variable("W3", shape=[7 * 7 * 64, 10],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.random_normal([10]))
            self.logits = tf.matmul(L2, W3) + b


        #define cost, optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,
                                                                           labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    def get_predict(self, x_test):
        return self.sess.run(self.logits, feed_dict={self.X:x_test})


    def get_accuracy(self, x_test, y_test):
        return self.sess.run(self.accuracy, feed_dict={self.X:x_test, self.Y:y_test})

    def train(self):
        print('Learning Started!')
        # train my model
        for epoch in range(training_epochs):
            avg_cost = 0
            total_batch = int(mnist.train.num_examples / batch_size)

            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                c, _ = self.sess.run([self.cost, self.optimizer],
                                     feed_dict={self.X:batch_xs, self.Y:batch_ys})
                avg_cost += c / total_batch

            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

        print('Learning Finished!')


sess = tf.Session()
model = Model(sess, "m1")
sess.run(tf.global_variables_initializer())

model.train() #이거 그냥 내가 수정함

# Test model and check accuracy
print('Accuracy:', model.get_accuracy(mnist.test.images, mnist.test.labels))

