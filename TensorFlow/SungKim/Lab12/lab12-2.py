import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
tf.set_random_seed(777)  # reproducibility
# --> 이거 추가 안하면 잘못된 결과 나오고 추가 하면 제대로 결과 나오던데 그 이유는?

# 1. Voca setting
# input: hihell -> output: ihello
idx2char = ['h', 'i', 'e', 'l', 'o']

# x_ont_hot.shape = (None, seq_length, input_dim)
x_one_hot = [[[1, 0, 0, 0, 0],   # h 0
              [0, 1, 0, 0, 0],   # i 1
              [1, 0, 0, 0, 0],   # h 0
              [0, 0, 1, 0, 0],   # e 2
              [0, 0, 0, 1, 0],   # l 3
              [0, 0, 0, 1, 0]]]  # l 3

# input: hihell -> output: ihello
x_data = [[0, 1, 0, 2, 3, 3]]
y_data = [[1, 0, 2, 3, 3, 4]]

# hyper parameter
num_classes = 5
input_dim = 5           #one-hot size
hidden_size = 5         #output dim
batch_size = 1          #one input sentence
seq_length = 6          #output('ihello')의 길이가 6
learning_rate = 0.1

X = tf.placeholder(tf.float32, [None, seq_length, input_dim]) #X one-hot
Y = tf.placeholder(tf.int32, [None, seq_length])              #Y label

#1. Creating RNN cell
cell = rnn.BasicLSTMCell(num_units=hidden_size,
                         state_is_tuple=True) #num_units = dim of output
initial_state = cell.zero_state(batch_size, tf.float32)


#2. Execute RNN cell
outputs, _states = tf.nn.dynamic_rnn(cell, X,
                                     initial_state=initial_state,
                                     dtype=tf.float32)

#3. computing sequence_loss & avg_loss
weights = tf.ones([batch_size, seq_length])
seq_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs,
                                            targets=Y,
                                            weights=weights)
avg_loss = tf.reduce_mean(seq_loss)

#4. training
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(avg_loss)
prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(200):
        loss, _ = sess.run([avg_loss, train],
                           feed_dict={X: x_one_hot, Y:y_data})
        result = sess.run(prediction,
                          feed_dict={X: x_one_hot})
        print("step:", i, " loss:", loss, " prediction:", result, " true Y:", y_data)

        #print character using dict.
        result_str = [idx2char[c] for c in np.squeeze(result)]
        print("\tPrediction str: ", ''.join(result_str))

