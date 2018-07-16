import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.contrib import seq2seq
tf.set_random_seed(777)  # reproducibility

# Data Creation
sample = " if you want you"
idx2char = list(set(sample))
char2idx = {c: i for i, c in enumerate(idx2char)}
#print(char2idx)

sample_idx = [char2idx[c] for c in sample]
x_data = [sample_idx[:-1]]   # "hello"에서는 "hell"가 x_data가 됨.
#print(x_data)
y_data = [sample_idx[1:]]    # "hello"에서는 "ello"가 y_data가 됨.
#print(y_data)

# Hyper parameter
dic_size = len(char2idx)    # RNN input size
hidden_size = len(char2idx) # RNN output size
num_classes = len(char2idx) # final output size
batch_size = 1              # one sentence --> one batch
seq_length = len(sample)-1  # unit
learning_rage = 0.1

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
train = tf.train.GradientDescentOptimizer(learning_rate=learning_rage).minimize(loss)
prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(3000):
        l, _ = sess.run([loss, train], feed_dict={X:x_data, Y:y_data})
        result = sess.run(prediction, feed_dict={X:x_data})

        #print character
        result_str = [idx2char[c] for c in np.squeeze(result)]
        print("step:", i, " loss:", l, " prediction:", ''.join(result_str))
