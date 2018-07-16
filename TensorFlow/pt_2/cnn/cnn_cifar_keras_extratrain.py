from keras.models import load_model
from tensorflow.python.keras._impl.keras.datasets.cifar10 import load_data
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils

batch_sz = 32
num_classes = 10
epochs = 70

(x_train, y_train), (x_test, y_test) = load_data()
# x data preprocessing (0-1 사이의 값으로 만들어주기)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# y data preprocessing (one-hot encoding)
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

model = load_model('cnn_cifar10_keras_6layers_ep30.h5')
model.summary()

# 추가학습
# Train cnn model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
hist = model.fit(x_train, y_train, batch_size=batch_sz, epochs=epochs,
                validation_data=(x_test, y_test), shuffle=True)

# 정확도 계산
acc = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy:", acc[1] * 100)

model.save('cnn_cifar10_keras_6layers_ep100.h5')
print("save model complete!")

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