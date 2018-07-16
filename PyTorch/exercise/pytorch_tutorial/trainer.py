import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

##### Load your own model
import models.cnn as cnn


class Trainer(object):
    def __init__(self, config, data_loader):
        self.config = config

        # training hyper-parameters
        self.niter = config.niter
        self.batch_size = config.batch_size
        self.image_size = config.image_size
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        # model hyper-parameters
        self.nc = int(config.nc)
        self.nclass = int(config.nclass)

        # misc
        self.data_loader = data_loader[0]
        self.test_loader = data_loader[1]
        self.ngpu = int(config.ngpu)
        self.cuda = config.cuda
        self.outf = config.outf

        self.build_model()

        if self.cuda:
            self.cnn.cuda()

    def build_model(self):
        self.cnn = cnn._CNN(self.ngpu, self.nc, self.nclass)
        print("Build Model:")
        print(self.cnn)
        if self.config.model_path != '':
            self.cnn.load_state_dict(torch.load(self.config.model_path))

    def train(self):
        # Your LOSS function.
        # criterion = nn.BCELoss() # for binary
        criterion = nn.CrossEntropyLoss()

        if self.cuda:
            criterion.cuda()

        # setup optimizer
        optimizer = optim.Adam(self.cnn.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))

        for epoch in range(self.niter):
            for i, (images, labels) in enumerate(self.data_loader, 0):
                if self.cuda:
                    images = Variable(images).cuda()
                    labels = Variable(labels).cuda()
                else:
                    images = Variable(images)
                    labels = Variable(labels)

                optimizer.zero_grad()
                outputs = self.cnn(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                if (i + 1) % self.config.log_step == 0:
                    print('Epoch [%d/%d], Iter [%d] Loss: %.4f'
                          % (epoch + 1, self.niter, i + 1, loss.data[0]))

                if (i + 1) % self.config.sample_step == 0:
                    print("Model saved")
                    torch.save(self.cnn.state_dict(), '%s/CNN_epoch_%03d.pth' % (self.outf, epoch))

        # Eval
        correct = 0
        total = 0
        for images, labels in self.test_loader:
            images = Variable(images).cuda()
            outputs = self.cnn(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels).sum()

        # do checkpointing
        torch.save(self.cnn.state_dict(), '%s/CNN_epoch_%03d.pth' % (self.outf, epoch))

        print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))
