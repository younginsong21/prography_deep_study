import numpy as np
from sklearn import svm
from tensorflow.python.keras._impl.keras.datasets.cifar10 import load_data  # cifar 데이터셋 로드

class svm_classifier_model():

    # 클래스 생성자
    # gamma, 학습 데이터 개수, 테스트 데이터 개수 지정
    def __init__(self, gamma, train_len, test_len):
        self.gammma = gamma

        #hyper parameter 지정
        self.train_len = train_len  # training에 쓸 데이터 개수
        self.test_len = test_len  # test에 쓸 데이터 개수
        self.nb_classes = 10  # class의 개수 --> 우리는 cifar-10 데이터셋을 사용하므로 10
        self.acc_sum = 0  # 정확도 계산에 사용할 변수. true label과 pred label이 일치할 경우 1씩 더해짐.

    # 데이터 전처리 과정
    # normalization과 reshape이 수행
    def data_processing(self):
        # 데이터 세팅
        (x_train, y_train), (x_test, y_test) = load_data()

        # x data preprocessing (0-1 사이의 값으로 만들어주기)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        # svm에 적합한 형태로 reshape (data, label 생성)
        # 시간이 너무 오래 걸려서 우선은 앞에서 train_len만큼의 데이터를 가져옴
        x_train = x_train[0:self.train_len]
        y_train = y_train[0:self.train_len]
        self.data = np.reshape(x_train, [self.train_len, 32 * 32 * 3])
        label = np.reshape(y_train, [1, self.train_len])
        self.label = label[0]

        # test set 형태 변경
        x_test = x_test[:self.test_len]
        y_test = y_test[:self.test_len]
        self.x_test = np.reshape(x_test, [self.test_len, 32 * 32 * 3])
        y_test = np.reshape(y_test, [1, self.test_len])
        self.y_test = y_test[0]

    # rbf kernel을 사용하는 svm model을 정의하고 학습
    def train_svm(self):
        self.data_processing()
        # svm classifier 정의
        # kernel = rbf kernel
        svm_clf = svm.SVC(kernel='rbf', gamma=self.gammma)
        # svm 학습
        svm_clf.fit(self.data, self.label)

        return svm_clf #학습된 svm 모델 리턴

    # 테스트 데이터 대상으로 prediction 진행, 정확도 계산함수
    def predict(self):
        svm_clf = self.train_svm()
        acc_sum = 0
        # test set에 대해 predict 과정
        for i in range(self.test_len):
            true = self.y_test[i]
            pred = svm_clf.predict([self.x_test[i]])[0]  # prediction

            if true == pred:
                acc_sum += 1

        print("Accuracy with SVM classifier:", float(acc_sum) / self.test_len)


print("SVM model #1 with gamma=0.01")
svm1 = svm_classifier_model(gamma=0.01, train_len=5000, test_len=500)
svm1.predict()

print("\nSVM model #2 with gamma=0.001")
svm2 = svm_classifier_model(gamma=0.001, train_len=5000, test_len=500)
svm2.predict()

print("\nSVM model #3 with gamma=0.0001")
svm3 = svm_classifier_model(gamma=0.0001, train_len=5000, test_len=500)
svm3.predict()