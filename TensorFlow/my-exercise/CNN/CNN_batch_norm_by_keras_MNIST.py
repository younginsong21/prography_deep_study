"""
CS231n에서 소개된 batch normalization 기법 연습 코드
--> Keras 라이브러리에서 제공하는 함수 BatchNormalization() 이용해서 구현
--> 졸프 때 쓰려고 weight, bias 추출 확인코드 추가
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras import optimizers
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

# 데이터셋 로딩
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x_train = mnist.train.images
y_train = mnist.train.labels
x_test = mnist.test.images
y_test = mnist.test.labels

# 데이터 preprocessing
# x에 해당하는 데이터들만 0~1 사이의 범위로 만들어줌
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
# conv. layer에 input으로 넣기 위해 image 형식으로 변환
x_train = np.reshape(x_train, [-1, 28, 28, 1])
x_test = np.reshape(x_test, [-1, 28, 28, 1])

# CNN 계층 구성
model = Sequential()

# conv+relu -> conv+relu -> batch norm. -> pooling -> dropout의 구조로 구성
model.add(Conv2D(32, (3, 3), padding='same', input_shape=[28, 28, 1]))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten()) # flatten 작업

model.add(Dense(625))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()

# 모델 학습 설정
adam = optimizers.adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# 학습
hist = model.fit(x_train, y_train, batch_size=100, epochs=20,
          validation_data=(x_test, y_test), shuffle=False)

# 모델의 weight, bias 추출해서 저장
print(model.layers) # layers 구조 파악

# 학습과정 그래프로 나타내기
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

acc_ax.plot(hist.history['acc'], 'b', label='train acc')
acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()

# 정확도 계산
acc = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy:", acc[1] * 100)