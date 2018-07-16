import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import numpy as np
tf.set_random_seed(777)  # reproducibility

# Data creation
sample = " if you want you"
idx2char = list(set(sample)) # index -> char 접근
char2idx = {c:i for i, c in enumerate(idx2char)} # char -> index 접근

# Define rnn input, output size and final output size
rnn_input_size = len(char2idx)
rnn_output_size = len(char2idx)
final_output_size = len(char2idx) # num of classes

# Hyper parameters
batch_size = 1 # 한번에 한 덩어리의 문자열만
sequence_length = len(sample) - 1 # 한번에 들어가는 문자열 길이
learning_rate = 0.1

sample_idx = [char2idx[c] for c in sample] # sample sentence -> index
x_data = [sample_idx[:-1]]
y_data = [sample_idx[1:]]

# Define input placeholders
x = tf.placeholder(tf.int32, [None, sequence_length])
y = tf.placeholder(tf.int32, [None, sequence_length])
x_one_hot = tf.one_hot(x, final_output_size)

# Define RNN cell
cell = rnn.BasicLSTMCell(num_units=rnn_output_size,
                         state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, x_one_hot,
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
prediction = tf.argmax(outputs, axis=2)

# Launch graphs and start learning
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(50):
        l, _ = sess.run([loss, train], feed_dict={x:x_data, y:y_data})
        result = sess.run(prediction, feed_dict={x:x_data})

        # print result string using idx2char list
        result_str = [idx2char[c] for c in np.squeeze(result)]

        print(i, "loss:", l, "prediction:", ''.join(result_str))