import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.contrib import seq2seq
tf.set_random_seed(777)  # reproducibility

# Data Creation
sentence = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")
char_set = list(set(sentence))
char_dic = {w: i for i, w in enumerate(char_set)}

# Hyper parameter
dic_size = len(char_set)    # RNN input size
hidden_size = len(char_set) # RNN output size
num_classes = len(char_set) # final output size
seq_length = 10             # 내 마음대로 세팅
learning_rate = 0.1

dataX = []
dataY = []

# seq_length만큼 window 옮겨가면서 sentence에서 뽑아냄
for i in range(0, len(sentence)-seq_length):
    x_str = sentence[i:i+seq_length]
    y_str = sentence[i+1:i+seq_length+1]
    print(i, x_str, '->', y_str)

    x = [char_dic[c] for c in x_str]
    y = [char_dic[c] for c in y_str]

    dataX.append(x)
    dataY.append(y)

# 중간에 자르지 않고 가지고 있는 X 전부 한꺼번에 넣음
# 문장의 개수가 곧 batch size
batch_size = len(dataX)

# Tensor Placeholder
X = tf.placeholder(tf.int32, [None, seq_length])
Y = tf.placeholder(tf.int32, [None, seq_length])

X_one_hot = tf.one_hot(X, num_classes)

# RNN cell
cell = rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, X_one_hot,
                                     initial_state=initial_state,
                                     dtype=tf.float32)
weights = tf.ones([batch_size, seq_length])

# Loss setting
sequence_loss = seq2seq.sequence_loss(logits=outputs,
                                      targets=Y,
                                      weights=weights)
loss = tf.reduce_mean(sequence_loss)

# training
train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(3000):
        l, _ = sess.run([loss, train], feed_dict={X:dataX, Y:dataY})
        print("step:", i, " loss:", l)
