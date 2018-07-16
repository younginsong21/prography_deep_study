import tensorflow as tf
# RNN sequence가 정해지지 않은 경우 --> 문자열의 길이가 가변
'''
기존에는 정해진 sequence 길이에다가 빈 공간에 padding 끼워넣음.
--> weight이 있기 때문에 loss 함수 계산 시 안좋은 영향
--> 개선방법: 각 batch sentence의 길이를 sequence_length를 list 형태로 전달
'''

