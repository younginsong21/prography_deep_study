"""
multi classification을 이진분류기인 SVM으로 해결
--> one vs. rest 방식으로 구현
--> class가 M개인 경우 : 각각의 클래스에 속하는 점/ 속하지 않는 점을 분류하는
    svm 분류기를 총 M개 만들기
--> M개의 분류기에서 voting 해서 가장 많은 score 가진 label로 결정.
"""
import numpy as np
from sklearn import svm
from tensorflow.python.keras._impl.keras.datasets.cifar10 import load_data  # cifar 데이터셋 로드

train_len = 2000
test_len = 500
nb_classes = 10

# 데이터 세팅
(x_train, y_train), (x_test, y_test) = load_data()

# x data preprocessing (0-1 사이의 값으로 만들어주기)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

"""
svm에 적합한 형태로 reshape (data, label 생성)
시간 너무 오래 걸려서 일단 x_train에서 앞의 2000 가져옴
                        y_train에서 앞의 2000 가져옴
"""
x_train = x_train[0:train_len]
y_train = y_train[0:train_len]
data = np.reshape(x_train, [train_len, 32 * 32 * 3])
label = np.reshape(y_train, [1, train_len])
label = label[0]

'''
#데이터 시각화
color = [str(item/255.) for item in label]
plt.scatter(data, label, c=color)
plt.show()
'''

# 총 10개의 binary classify label 생성
bn_label_lists = []
for i in range(nb_classes):
    bn_label_lists.append([])

for idx in range(len(label)):
    lb = label[idx]
    for i in range(nb_classes):
        if i == lb:
            bn_label_lists[i].append(1)
        else:
            bn_label_lists[i].append(-1)


# 10개의 svm classifier 생성
# i번째 svm model은 i번째 label으로 fit
svms = []
for i in range(nb_classes):
    svms.append(svm.SVC(kernel='rbf', C=10.0, random_state=0, gamma=0.19)) #kernel: linear->rbf으로 변경
    svms[i].fit(data, bn_label_lists[i])


# 새로운 데이터로 prediction 과정
zero_list = np.zeros([1, nb_classes])
test =  x_test[0:test_len]
test = np.reshape(test, [test_len, 32 * 32 * 3])

new = []
vote = []
for i in range(test_len):
    new.append([test[i]])
    vote.append(zero_list[0])

acc_sum = 0
for data_idx in range(test_len):
    for svm_idx in range(nb_classes):
        vote[data_idx][svm_idx] += svms[svm_idx].predict(new[data_idx])[0]  # prediction
    vote[data_idx] = np.argsort(vote[data_idx])
    pred = vote[data_idx][len(svms) - 1] # 가장 많이 투표된 label로 결정
    real = y_test[data_idx][0]           # 실제 label

    if pred == real :
        acc_sum += 1

# 정확도 출력
print("accuracy:", float(acc_sum) / test_len)

"""
kernel: polynomial으로 변경
데이터 전처리 코드 추가
accuracy: 0.112
"""

"""
kernel: rbf으로 변경
데이터 전처리 코드 추가
accuracy: 0.112
"""