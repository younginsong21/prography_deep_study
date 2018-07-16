import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import csv
import gzip

# Name Dataset 클래스 정의
class NameDataset(Dataset):
    # 데이터 파일 가져와서 추출
    def __init__(self, is_train_set=False):
        if is_train_set:  # training set인 경우
            filename = '../data/names_train.csv.gz'
        else:  # test set인 경우
            filename = '../data/names_test.csv.gz'
        with gzip.open(filename, "rt") as f:
            reader = csv.reader(f)
            rows = list(reader)

        self.names = [row[0] for row in rows]
        self.countries = [row[1] for row in rows]
        self.len = len(self.countries)

        # 데이터셋에 포함된 나라를 중복없이 담은 리스트
        self.country_list = list(sorted(set(self.countries)))

    # row index가 주어졌을 때 해당 row의 item 가져오는 함수
    def __getitem__(self, idx):
        return self.names[idx], self.countries[idx]

    # 전체 데이터셋의 길이
    def __len__(self):
        return self.len

    # country_list 가져오는 함수
    def get_countries(self):
        return self.country_list

    # country_list에서 특정 idx의 country만 가져오는 함수
    # idx로 country에 접근
    def idx2country(self, idx):
        return self.country_list[idx]

    # country_list에서 특정 country의 idx만 가져오는 함수
    # country로 idx에 접근
    def country2idx(self, country):
        return self.country_list.index(country)

if __name__ == "__main__":
    dataset = NameDataset(False) #test set loading
    print(dataset.get_countries())
    print(dataset.idx2country(3))
    print(dataset.country2idx('Korean'))

    train_loader = DataLoader(dataset=dataset,
                              batch_size=10,
                              shuffle=True)

    print(len(train_loader.dataset))
    for epoch in range(2):
        for i, (names, countries) in enumerate(train_loader):
            print(epoch, i, "names", names, "countries", countries)
