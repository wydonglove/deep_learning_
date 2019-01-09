import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt
import mnist_loader
import random
from torch.autograd import Variable
from torch.utils.data import TensorDataset

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

test_data_x = []
test_data_y = []
for x ,y in list(test_data):
    test_data_x.append(x)
    test_data_y.append(y)

net = Net()
net.cuda()
print(net)
training_data_x = torch.FloatTensor(training_data_x)
training_data_x = torch.unsqueeze(training_data_x,dim=1)
training_data_y = torch.LongTensor(training_data_y)
training_data_y = training_data_y[:, :, 0]

optimizer = torch.optim.Adam(net.parameters(),lr=0.01)
loss_func = torch.nn.CrossEntropyLoss()

test_data_x = torch.FloatTensor(test_data_x)
test_data_x = torch.unsqueeze(test_data_x,dim=1)
test_data_y = torch.LongTensor(test_data_y)


torch_dataset = TensorDataset(training_data_x,training_data_y)

train_loader = Data.DataLoader(dataset=torch_dataset,batch_size=100,shuffle=True)


#training_data_x ,y = Variable(training_data_y),Variable(training_data_y)
EPOCH = 100
for epoch in range(4):
    print("----")
    for step,(b_x,b_y) in enumerate(train_loader):
        b_x = b_x.cuda()
        b_y = b_y.cuda()

        out = net(b_x)
        loss = loss_func(out,torch.max(b_y,1)[1]).cuda()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 20 == 0:
            test_out = net(test_data_x.cuda())
            pred_y = torch.max(test_out, 1)[1].cpu().data.numpy()
            accuracy = float((pred_y == test_data_y.data.numpy()).astype(int).sum()) / float(test_data_y.size(0))
            print("Loss=%.4f,Accuracy=%.4f" % (loss.cpu().data.numpy(), accuracy))



# for t in range(100):
#     out = net(training_data_x)
#     #out = torch.unsqueeze(out,dim=1)
#     # print(out.size())
#     # print(training_data_y.size())
#     # print(ttt.size())
#     # tt0 = torch.max(ttt, 1)
#     # print(tt0)
#     # torch.max(ttt,1)  max 返回最大值和索引  1:返回每行最大值和索引，0:返回没列最大值和索引
#     loss = loss_func(out,torch.max(training_data_y,1)[1])
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     if t % 5 == 0 :
#         test_out = net(test_data_x)
#         pred_y = torch.max(test_out,1)[1].data.numpy()
#         accuracy = float((pred_y==test_data_y.data.numpy()).astype(int).sum())/ float(test_data_y.size(0))
#         print("Loss=%.4f,Accuracy=%.4f" % (loss.data.numpy(),accuracy))


