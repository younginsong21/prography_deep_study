import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 100
num_classes = 2
batch_size = 100
learning_rate = 0.001

# Define transform
transform = transforms.Compose(
                   [transforms.Resize((32,32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Define training dataset/ test dataset loader
trainset = dset.ImageFolder(root="data_dir/train", transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=batch_size,
                                          shuffle=True)

testset = dset.ImageFolder(root='data_dir/test',transform=transform)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=10,
                                         shuffle=True)

# Define my dataset's classes
classes=('bookcase', 'chair')

# Define function to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Define Convolutional Neural Network
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # Layer1
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 6, 3),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # Layer2
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # FC layer1
        self.fc1 = nn.Sequential(
            nn.Linear(576, 120), nn.ReLU()
        )
        # FC layer2
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84), nn.ReLU()
        )
        # FC layer3
        self.fc3 = nn.Sequential(
            nn.Linear(84, 21), nn.ReLU()
        )
        # FC layer4 (final layer)
        self.fc4 = nn.Linear(21, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)  # flatten
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        return out

# Safe DataLoader multiprocessing with Windows
if __name__ == '__main__':
    # Define my CNN model
    model = CNNModel().to(device)
    print(model)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    print("Learning started!!!")
    for epoch in range(num_epochs):
        avg_cost = 0
        for i, (images, labels) in enumerate(testloader):
            images = images.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + Backward + Optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Compute average cost
            avg_cost += loss.item() / batch_size

        if epoch % 10 == 0:
            print("Epoch [%d/%d] , Loss: %.8f" % (epoch+1, num_epochs, avg_cost))
    print("Learning finished!!!")

    # Test the model
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)

            # Prediction
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            # Compute accuracy
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print("Test Accuracy:", 100*(float(correct)) / total)

    # Save the model checkpoint
    torch.save(model.state_dict(), 'mini_pj_model.ckpt')
