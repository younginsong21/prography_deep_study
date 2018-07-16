import tensorflow as tf
import numpy as np
from tensorflow.python.keras._impl.keras.datasets.cifar10 import load_data

# CIFAR-10 데이터를 다운로드
(x_train, y_train), (x_test, y_test) = load_data()

print("x_train.shape:", x_train.shape)
print("y_train.shape:", y_train.shape)
print("x_test.shape:", x_test.shape)
print("y_test.shape:", y_test.shape)

data2arr = np.asarray(x_train[0:500])
print(data2arr.shape)