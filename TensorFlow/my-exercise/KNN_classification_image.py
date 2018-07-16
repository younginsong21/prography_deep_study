import tensorflow as tf
import random
import numpy as np
tf.set_random_seed(777)  # for reproducibility
from tensorflow.examples.tutorials.mnist import input_data
sess = tf.Session()
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

class model():

    # X, Y 데이터셋을 메모리에 단순히 기억시키는 과정
    # batch size => 사용자에게 입력 받음
    def train(self, batch_size):
        self.batch_size = batch_size
        dataX, dataY = mnist.train.next_batch(self.batch_size)
        self.X = dataX
        self.Y = dataY

    # L1 distance 이용해서 KNN 구현
    def predict_l1(self, inputX, inputY):
        self.inputX = inputX
        self.inputY = inputY
        # L1 distance 계산
        diff_mat = np.abs(self.X - self.inputX)
        row_sum_diff = diff_mat.sum(axis=1)

        # 오름차순 정렬후 idx만 배열에 저장
        sort_dist_idx = row_sum_diff.argsort()

        # 최상위 k개 선정
        self.k = int(input("k값을 입력하세요:"))

        for i in range(self.k):
            result_label = self.Y[sort_dist_idx[i], :]
            result_label = tf.argmax(result_label)
            print(i+1 , "번째 KNN 결과 Label:", sess.run(result_label))
            print("실제 Label:", sess.run(tf.argmax(self.inputY)))
            print("=================================")

    # L2 distance 이용해서 KNN 구현
    def predict_l2(self):
        # L2 distance 계산 제곱->합산->루트
        diff_mat = np.abs(self.X - self.inputX)
        sq_diff_mat = diff_mat ** 2
        row_sum_diff = diff_mat.sum(axis=1)
        row_sum_diff = np.sqrt(row_sum_diff)

        # 오름차순 정렬후 idx만 배열에 저장
        sort_dist_idx = row_sum_diff.argsort()

        # 최상위 k개 선정
        for i in range(self.k):
            result_label = self.Y[sort_dist_idx[i], :]
            result_label = tf.argmax(result_label)
            print(i + 1, "번째 KNN 결과 Label:", sess.run(result_label))
            print("실제 Label:", sess.run(tf.argmax(self.inputY)))
            print("=================================")


mymodel = model()
batch_size = int(input("batch size를 입력하세요:"))

r = random.randint(0, mnist.test.num_examples-1)
inputX = mnist.test.images[r]
inputY = mnist.test.labels[r]
inputX = np.tile(inputX, (batch_size, 1))

mymodel.train(batch_size)

print("L1 distance를 이용한 KNN 구현")
print("=================================")
mymodel.predict_l1(inputX, inputY)
print("================================")
print("L2 distance를 이용한 KNN 구현")
print("=================================")
mymodel.predict_l2()

'''
batch size를 입력하세요:500
L1 distance를 이용한 KNN 구현
=================================
k값을 입력하세요:7
1 번째 KNN 결과 Label: 8
실제 Label: 8
=================================
2 번째 KNN 결과 Label: 8
실제 Label: 8
=================================
3 번째 KNN 결과 Label: 2
실제 Label: 8
=================================
4 번째 KNN 결과 Label: 8
실제 Label: 8
=================================
5 번째 KNN 결과 Label: 8
실제 Label: 8
=================================
6 번째 KNN 결과 Label: 3
실제 Label: 8
=================================
7 번째 KNN 결과 Label: 8
실제 Label: 8
=================================
================================
L2 distance를 이용한 KNN 구현
=================================
1 번째 KNN 결과 Label: 8
실제 Label: 8
=================================
2 번째 KNN 결과 Label: 8
실제 Label: 8
=================================
3 번째 KNN 결과 Label: 2
실제 Label: 8
=================================
4 번째 KNN 결과 Label: 8
실제 Label: 8
=================================
5 번째 KNN 결과 Label: 8
실제 Label: 8
=================================
6 번째 KNN 결과 Label: 3
실제 Label: 8
=================================
7 번째 KNN 결과 Label: 8
실제 Label: 8
=================================
'''



