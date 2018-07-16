'''
대상 dataset: Cifar-10
데이터셋을 통해서 간단한 Multi-label classification 구현
이미지는 32*32*3(RGB)으로 이루어져 있음
레이블은 10개
train 50000, test 10000
'''

import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Hyper parameters
learning_rate = 0.01
training_epoch = 30
num_classes = 10

# Load dataset
batch_1 = unpickle('CIFAR_data\data_batch_1')
batch1_x = batch_1[b'data']
batch1_x = np.array(batch1_x, dtype=np.float32)
batch1_x -= np.mean(batch1_x, axis=0) #데이터 전처리, Zero-centered
print(batch1_x)
batch1_y = batch_1[b'labels']
batch1_y = np.reshape(batch1_y, [-1, 1])
#print(batch1_x.shape)
#print(batch1_y.shape)
#print(batch1_y)

batch_2 = unpickle('CIFAR_data\data_batch_2')
batch2_x = batch_2[b'data']
batch2_x = np.array(batch2_x, dtype=np.float32)
batch2_x -= np.mean(batch2_x, axis=0) #데이터 전처리, Zero-centered
batch2_y = batch_2[b'labels']
batch2_y = np.reshape(batch2_y, [-1, 1])

batch_3 = unpickle('CIFAR_data\data_batch_3')
batch3_x = batch_3[b'data']
batch3_x = np.array(batch3_x, dtype=np.float32)
batch3_x -= np.mean(batch3_x, axis=0) #데이터 전처리, Zero-centered
batch3_y = batch_3[b'labels']
batch3_y = np.reshape(batch3_y, [-1, 1])

batch_4 = unpickle('CIFAR_data\data_batch_4')
batch4_x = batch_4[b'data']
batch4_x = np.array(batch4_x, dtype=np.float32)
batch4_x -= np.mean(batch4_x, axis=0) #데이터 전처리, Zero-centered
batch4_y = batch_4[b'labels']
batch4_y = np.reshape(batch4_y, [-1, 1])

batch_5 = unpickle('CIFAR_data\data_batch_5')
batch5_x = batch_5[b'data']
batch5_x = np.array(batch5_x, dtype=np.float32)
batch5_x -= np.mean(batch5_x, axis=0) #데이터 전처리, Zero-centered
batch5_y = batch_5[b'labels']
batch5_y = np.reshape(batch5_y, [-1, 1])

dataX = [batch1_x, batch2_x, batch3_x, batch4_x, batch5_x]
dataY = [batch1_y, batch2_y, batch3_y, batch4_y, batch5_y]

# Test dataset load
batch_test = unpickle(r'CIFAR_data\test_batch')
batch_test_x = batch_test[b'data']
batch_test_x = np.array(batch_test_x, dtype=np.float32)
batch_test_x -= np.mean(batch_test_x, axis=0) #데이터 전처리, Zero-centered
batch_test_y = batch_test[b'labels']
batch_test_y = np.reshape(batch_test_y, [-1, 1])
#print(batch_test_x.shape)
#print(batch_test_y.shape)

# Input placeholders
X = tf.placeholder(tf.float32, shape=[None, 3072])   # 32*32*3=3072
Y = tf.placeholder(tf.int32, shape=[None, 1])
Y_one_hot = tf.one_hot(Y, num_classes)
Y_one_hot = tf.reshape(Y_one_hot, [-1, num_classes])
keep_prob = tf.placeholder(tf.float32)

# Building layers
# Layer 1
W1 = tf.get_variable("W1", shape=[3072, 4000],
                     initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([4000]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

# Layer 2
W2 = tf.get_variable("W2", shape=[4000, 4000],
                     initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([4000]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

# layer 3 (final, without activation)
W3 = tf.get_variable("W5", shape=[4000, 10],
                     initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(L2, W3) + b3

# cost, optimizer 설정
inner_cost = tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis,
                                                     labels=Y_one_hot)
cost = tf.reduce_mean(inner_cost)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Launch graphs
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Learning started
print("Learning started!")
for epoch in range(training_epoch):
    avg_cost = 0
    for batch_num in range(len(dataX)):
        c, _ = sess.run([cost, optimizer],
                        feed_dict={X: dataX[batch_num],
                                   Y: dataY[batch_num],
                                   keep_prob: 0.7})
        avg_cost += c / len(dataX)

    print("Epoch:", epoch, " Cost:", avg_cost)

# Accuracy setting
is_equal = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(is_equal, tf.float32))
acc = sess.run(accuracy, feed_dict={X: batch_test_x,
                                    Y: batch_test_y,
                                    keep_prob: 1.0})
print("Accuracy:", acc)

'''
MNIST때랑 다르게 완전 안좋은 결과 ㅜㅜ 왜일까
Learning started!
Epoch: 0  Cost: 4350.900234985352
Epoch: 1  Cost: 35.72701034545898
Epoch: 2  Cost: 7.064052295684814
Epoch: 3  Cost: 3.7177125930786135
Epoch: 4  Cost: 2.902018117904663
Epoch: 5  Cost: 2.5955101013183595
Epoch: 6  Cost: 2.4642531871795654
Epoch: 7  Cost: 2.4108148574829102
Epoch: 8  Cost: 2.384822988510132
Epoch: 9  Cost: 2.3680652141571046
Epoch: 10  Cost: 2.35433087348938
Epoch: 11  Cost: 2.344518804550171
Epoch: 12  Cost: 2.336577033996582
Epoch: 13  Cost: 2.339858627319336
Epoch: 14  Cost: 2.330372142791748
Epoch: 15  Cost: 2.327707624435425
Epoch: 16  Cost: 2.3238707065582274
Epoch: 17  Cost: 2.3208670616149902
Epoch: 18  Cost: 2.3229032516479493
Epoch: 19  Cost: 2.3203804969787596
Epoch: 20  Cost: 2.3187976837158204
Epoch: 21  Cost: 2.3186397552490234
Epoch: 22  Cost: 2.3191415309906005
Epoch: 23  Cost: 2.317008876800537
Epoch: 24  Cost: 2.3159940242767334
Epoch: 25  Cost: 2.3131515502929685
Epoch: 26  Cost: 2.3101784229278564
Epoch: 27  Cost: 2.3135985374450683
Epoch: 28  Cost: 2.3094850540161134
Epoch: 29  Cost: 2.311448001861572
Accuracy: 0.1348
'''

'''
데이터 전처리 거친 후에 --> 더 안좋아짐.... 내가 Cifar10 데이터셋을 애초에 잘못 가져왔나?
Learning started!
Epoch: 0  Cost: 102868.18532562254
Epoch: 1  Cost: 34952.3466796875
Epoch: 2  Cost: 1024.303271484375
Epoch: 3  Cost: 12.575697708129884
Epoch: 4  Cost: 3.208060693740845
Epoch: 5  Cost: 2.798537540435791
Epoch: 6  Cost: 2.7238635540008542
Epoch: 7  Cost: 2.6647048473358153
Epoch: 8  Cost: 2.631215047836304
Epoch: 9  Cost: 2.599806070327759
Epoch: 10  Cost: 2.5794387340545657
Epoch: 11  Cost: 2.5541398525238037
Epoch: 12  Cost: 2.5339558124542236
Epoch: 13  Cost: 2.515631771087647
Epoch: 14  Cost: 2.493239402770996
Epoch: 15  Cost: 2.477343702316284
Epoch: 16  Cost: 2.461604928970337
Epoch: 17  Cost: 2.453989458084106
Epoch: 18  Cost: 2.4338149547576906
Epoch: 19  Cost: 2.422135257720947
Epoch: 20  Cost: 2.4120198249816895
Epoch: 21  Cost: 2.400856781005859
Epoch: 22  Cost: 2.3921602725982662
Epoch: 23  Cost: 2.3823194026947023
Epoch: 24  Cost: 2.3743212699890135
Epoch: 25  Cost: 2.3675845146179197
Epoch: 26  Cost: 2.3626921653747557
Epoch: 27  Cost: 2.3552093505859375
Epoch: 28  Cost: 2.3495872020721436
Epoch: 29  Cost: 2.3459199905395507
Accuracy: 0.1
'''



