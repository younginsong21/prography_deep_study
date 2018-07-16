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

# Stacked RNN cell
cell = rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
cell = rnn.MultiRNNCell([cell]*2, state_is_tuple=True)  # stacked RNN cell (2층)
                                                        # 100층으로 쌓고 싶으면 [cell]*100

initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, X_one_hot,
                                     initial_state=initial_state,
                                     dtype=tf.float32)  # output of RNN cells
#print(outputs.shape)

# input of softmax
X_for_softmax = tf.reshape(outputs, [-1, hidden_size])

# softmax setting
softmax_w = tf.get_variable("sotfmax_w", [hidden_size, num_classes])
sotfmax_b = tf.get_variable("softmax_b", [num_classes])
outputs = tf.matmul(X_for_softmax, softmax_w) + sotfmax_b              #이 output은 activation 없음

# output of softmax reshape
outputs = tf.reshape(outputs, [batch_size, seq_length, num_classes])   #RNN의 output과 동일한 shape
#print(outputs.shape)

weights = tf.ones([batch_size, seq_length])

# Loss setting
sequence_loss = seq2seq.sequence_loss(logits=outputs,
                                      targets=Y,
                                      weights=weights)
loss = tf.reduce_mean(sequence_loss)

# Training
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(500):
    _, l, results = sess.run(
        [train_op, loss, outputs], feed_dict={X: dataX, Y: dataY})
    for j, result in enumerate(results):
        index = np.argmax(result, axis=1)
        print(i, j, ''.join([char_set[t] for t in index]), l)

# Let's print the last char of each result to check it works
results = sess.run(outputs, feed_dict={X: dataX})
for j, result in enumerate(results):
    index = np.argmax(result, axis=1)
    if j is 0:  # print all for the first result to make a sentence
        print(''.join([char_set[t] for t in index]), end='')
    else:
        print(char_set[index[-1]], end='')