"""
Kaggle dataset
features: 11개
labels: 0-5 총 6개
"""
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

# Hyper parameters
input_size = 11
num_classes = 6
num_epochs = 200
batch_size = 200
learning_rate = 0.005

# Some directories
data_dir = "C:/Users/young/Desktop/young_local/dataset"
ckpt_dir = "C:/Users/young/Desktop/young_local/PyTorch/exercise/ckpt"

# Wine Quality dataset class implement
class WineDataSet(data.Dataset):
    def __init__(self):
        alldata = np.loadtxt(data_dir + "/data-winequality.csv", delimiter=',')
        feature = np.ndarray.astype(alldata[:, :-1], np.float32)
        label = np.ndarray.astype(alldata[:, [-1]], np.int32)
        self.feature = feature
        self.label = label

    # feature, label -> Tensor
    def __getitem__(self, idx):
        return torch.Tensor(self.feature[idx]), self.label[idx]

    # 데이터셋 길이
    def __len__(self):
        return self.feature.shape[0]

# Training dataset loading
Wine_train = WineDataSet()
train_loader = torch.utils.data.DataLoader(dataset=Wine_train,
                                           batch_size=batch_size,
                                           shuffle=True)
# Logistic model
model = nn.Linear(input_size, num_classes)
print(model)

# Loss and optimizer
cost_func = nn.CrossEntropyLoss()  # internally softmax
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (features, labels) in enumerate(train_loader):
        # Some preprocessing in labels
        labels = labels.long().squeeze(1)

        # Forward pass
        outputs = model(features)
        loss = cost_func(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

# Test the model
with torch.no_grad():
    correct = 0
    total = 0
    for features, labels in train_loader:
        # Some preprocessing in labels
        labels = labels.long().squeeze(1)

        # prediction
        outputs = model(features)
        _, predicted = torch.max(outputs.data, 1)

        # Compute accuracy
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy: %.2f %%' % (100 * (correct) / total))

# Save the model checkpoint
torch.save(model.state_dict(), ckpt_dir+'/04_Wine_softmax_model.ckpt')

"""
Accuracy: 45.00 %
"""