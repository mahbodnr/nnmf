import torch
from nnmf.modules import NNMFConv2d
# from torch.nn import Conv2d as NNMFConv2d
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import Adam
from tqdm import tqdm
from matplotlib import pyplot as plt

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        iterations = 100
        backward_method = "david"
        self.nnmf1 = NNMFConv2d(
            in_channels= 3,
            out_channels = 32,
            kernel_size = 3,
            n_iterations = iterations,
            backward_method=backward_method,
        )
        self.nnmf2 = NNMFConv2d(
            in_channels= 32,
            out_channels = 64,
            kernel_size = 3,
            n_iterations=iterations,
            backward_method=backward_method,
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        self.out = nn.Linear(64 * 6 * 6, 10)

    def forward(self, x):
        x = self.nnmf1(x)
        x = self.pool2(x)
        x = self.nnmf2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
    
        return x

# load cifar data
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR10(root='~/data', train=True, download=False, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testset = torchvision.datasets.CIFAR10(root='~/data', train=False, download=False, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# create model
net = Net().cuda()

# define loss function
criterion = nn.CrossEntropyLoss()
optimizer = Adam(net.parameters(), lr=0.001)

corrects = 0
items = 0

def evaluate():
    corrects = 0
    items = 0
    for i, data in tqdm(enumerate(testloader), total=len(testloader)):
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        corrects += (predicted == labels).sum().item()
        items += labels.size(0)
    accuracy = corrects / items
    print(f'* Test Accuracy: {accuracy}')

for epoch in range(10):
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs, 1)
        corrects += (predicted == labels).sum().item()
        items += labels.size(0)
        if i % 100 == 99:
            accuracy = corrects / items
            print(f'Epoch: {epoch}, Iteration: {i}, Loss: {loss.item()}, Accuracy: {accuracy}')
            corrects = 0
            items = 0
    evaluate()
