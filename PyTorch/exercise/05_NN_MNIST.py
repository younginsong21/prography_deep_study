import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Hyper parameters
input_size = 28*28
hidden_size = 512
num_classes = 10
learning_rate = 0.001
num_epochs = 5
batch_size = 100

# Some directories
data_dir = "C:/Users/young/Desktop/young_local/dataset"
ckpt_dir = "C:/Users/young/Desktop/young_local/PyTorch/exercise/ckpt"

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='../../data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data',
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# Neural Network model
class NNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NNModel, self).__init__()
        # Define Neural network
        self.layer = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes, bias=True)
        )
    def forward(self, x):
        out = self.layer(x)
        return out

# Define model
model = NNModel(input_size, hidden_size, num_classes)
print(model)

# Cost function and optimizer
cost_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Reshape
        images = images.reshape(-1, 28 * 28)

        # Forward pass
        outputs = model(images)
        loss = cost_func(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

# Test the model
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        # reshape
        images = images.reshape(-1, 28 * 28)

        # predict
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        # Compute accuracy
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the model on the 10000 test images: %.2f %%' % (100*(correct)/total))

# Save the model checkpoint
torch.save(model.state_dict(), ckpt_dir+'/05_MNIST_nn_model.ckpt')

"""
Accuracy of the model on the 10000 test images: 97.00 %
"""