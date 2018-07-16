import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import numpy as np
tf.set_random_seed(777)  # reproducibility

# Data creation
sample = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")

idx2char = list(set(sample)) # index -> char 접근
char2idx = {c:i for i, c in enumerate(idx2char)} # char -> index 접근

# Define rnn input, output size and final output size
rnn_input_size = len(char2idx)
rnn_output_size = len(char2idx)
final_output_size = len(char2idx) # num of classes

# Hyper parameters
sequence_length = 10 # 한번에 들어가는 문자열 길이
learning_rate = 0.1

x_data = []
y_data = []
for i in range(0, len(sample) - sequence_length):
    x_str = sample[i:i + sequence_length]
    y_str = sample[i + 1: i + sequence_length + 1]
    print(i, x_str, '->', y_str)

    x = [char2idx[c] for c in x_str]  # x str to index
    y = [char2idx[c] for c in y_str]  # y str to index

    x_data.append(x)
    y_data.append(y)

batch_size = len(x_data)

# Define input placeholders
x = tf.placeholder(tf.int32, [None, sequence_length])
y = tf.placeholder(tf.int32, [None, sequence_length])

x_one_hot = tf.one_hot(x, final_output_size)

# LSTM cell 생성하는 함수
def lstm_cell():
    cell = rnn.BasicLSTMCell(rnn_output_size, state_is_tuple=True)
    return cell

# 여러개의 stacked RNN cell
cells = rnn.MultiRNNCell([lstm_cell() for _ in range(2)], state_is_tuple=True)

initial_state = cells.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cells, x_one_hot,
                                     initial_state=initial_state,
                                     dtype=tf.float32)

# FC layer
x_for_fc = tf.reshape(outputs, [-1, rnn_output_size]) # reshape for FC layer
outputs = tf.contrib.layers.fully_connected(x_for_fc, final_output_size, activation_fn=None)
outputs = tf.reshape(outputs, [batch_size, sequence_length, final_output_size]) # reshape for seq loss

# Define sequence loss
weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs,
                                                 targets=y,
                                                 weights=weights)
loss = tf.reduce_mean(sequence_loss)

# Define optimizer and prediction
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
print("output's shape:", outputs.shape) # (170=batch size, 10=sequence length, 25=num of classes)
                                        # axis=0 : batch size, axis=1: sequence length, axis=2: num of classes
prediction = tf.argmax(outputs, axis=2) # classes(set of characters) 중 가장 max인 값 찾는거

# Launch graphs and start learning
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(500):
        _, l, results = sess.run(
            [train, loss, outputs], feed_dict={x: x_data, y: y_data})
        for j, result in enumerate(results):
            index = np.argmax(result, axis=1)
            print(i, j, ''.join([idx2char[t] for t in index]), l)

    # Let's print the last char of each result to check it works
    results = sess.run(outputs, feed_dict={x: x_data})
    for j, result in enumerate(results):
        index = np.argmax(result, axis=1)
        if j is 0:  # print all for the first result to make a sentence
            print(''.join([idx2char[t] for t in index]), end='')
        else:
            print(idx2char[index[-1]], end='')
