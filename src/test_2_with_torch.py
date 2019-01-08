import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt
import mnist_loader
import random
from torch.autograd import Variable

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.out = torch.nn.Linear(32*7*7,10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)
        output = self.out(x)
        return output

training_data, validation_data, test_data = mnist_loader.load_data_2d()
print(type(training_data))
print(type(validation_data))
print(type(test_data))
training_data_x = []
training_data_y = []
for x ,y in list(training_data):
    training_data_x.append(x)
    training_data_y.append(y)

net = Net()
print(net)
training_data_x = torch.FloatTensor(training_data_x)
training_data_x = torch.unsqueeze(training_data_x,dim=1)
training_data_y = torch.LongTensor(training_data_y)
optimizer = torch.optim.Adam(net.parameters(),lr=0.2)
loss_func = torch.nn.CrossEntropyLoss()

#training_data_x ,y = Variable(training_data_y),Variable(training_data_y)
for t in range(10):
    out = net(training_data_x)
    #out = torch.unsqueeze(out,dim=1)
    ttt = training_data_y[:, :, 0]
    # print(out.size())
    # print(training_data_y.size())
    # print(ttt.size())
    # tt0 = torch.max(ttt, 1)
    # print(tt0)
    # torch.max(ttt,1)  max 返回最大值和索引  1:返回每行最大值和索引，0:返回没列最大值和索引
    loss = loss_func(out,torch.max(ttt,1)[1])


