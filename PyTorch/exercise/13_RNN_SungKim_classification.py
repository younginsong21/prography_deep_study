import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from name_dataset import NameDataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Hyper parameters
HIDDEN_SIZE = 100
N_LAYERS = 2
BATCH_SIZE = 256
N_EPOCHS = 100

# Data loaders
test_dataset = NameDataset(is_train_set=False)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE, shuffle=True)


train_dataset = NameDataset(is_train_set=True)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE, shuffle=True)

N_COUNTRIES = len(train_dataset.get_countries())
print(N_COUNTRIES, "countries")
N_CHARS = 128  # ASCII

# 소요 시간 구하는 함수
def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# Tensor => Variable 변환 함수
def create_variable(tensor):
    # Do cuda() before wrapping with variable
    if torch.cuda.is_available():
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)

# 여러개의 input string을 처리할 때 sequence_length를 맞추기 위한 zero padding
# 기능 추가: Tensor 정렬
# 함수의 인자 input_seq: padding 처리하고 싶은 input 문자열들 (여러개)
#             seq_lengths: 각 문자열들의 길이 (여러개)
def zero_padding_sequences(vectorized_seqs, seq_lengths, countries):
    seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long()
    for idx, (seq, seq_len) in enumerate(zip(vectorized_seqs, seq_lengths)):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)  # 앞에서부터 원래 seq 길이까지는 원래 seq들로 채우고 뒤에는 초기화한대로 0이 됨

    # Tensor를 길이에 따라 정렬
    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    seq_tensor = seq_tensor[perm_idx]
