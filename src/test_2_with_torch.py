import torch
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt
import mnist_loader
import random

class Net(torch.nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Net,self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1,padding=2)
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1,padding=2)
        self.out = torch.nn.Linear(32*7*7,10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.ReLU(x)
        x = torch.nn.MaxPool2d(2)

        x = self.conv2(x)
        x = torch.nn.ReLU(x)
        x = torch.nn.MaxPool2d(2)

        #x = x.view(x.size(0),-1)
        out = self.out(x)

        return out,x

training_data, validation_data, test_data = mnist_loader.load_data_2d()
print(type(training_data))
print(type(validation_data))
print(type(test_data))

for index , item in list(training_data):
    print(index)
    print(item)
    print("this")
    print()
#(training_data_x,training_data_y) = [(x,y) for x,y in enumerate(list(training_data))]
net = Net()
print(net)
print(training_data_x[0])

# optimizer = torch.optim.Adam(net.parameters(),lr=0.2)
# loss_func = torch.nn.CrossEntropyLoss()
#
# for t in range(200):
#     [training_data_x,training_data_y] = [[x,y] for x,y in training_data]
#     out = net(training_data_x)
#     loss = loss_func(out,training_data_y)


