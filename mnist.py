import torch
from implicit_backprop.modules import NNMFDense
# from nnmf.modules import NNMFDense
# from torch.nn import Linear as NNMFDense
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
        self.nnmf1 = NNMFDense(
            in_features=784,
            out_features=100,
            n_iterations=iterations,
        )
        self.nnmf2 = NNMFDense(
            in_features=100,
            out_features=10,
            n_iterations=iterations,
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.nnmf1(x)
        x = self.nnmf2(x)
        return x

# load mnist data
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.MNIST(root='~/data', train=True, download=False, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testset = torchvision.datasets.MNIST(root='~/data', train=False, download=False, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# create model
net = Net().cuda()

# define loss function
criterion = nn.CrossEntropyLoss()
optimizer = Adam(net.parameters(), lr=0.01)
# optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
corrects = 0
items = 0

def evaluate():
    corrects = 0
    items = 0
    for i, data in tqdm(enumerate(testloader), total=len(testloader)):
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        inputs = inputs.view(inputs.size(0), -1)
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
        inputs = inputs.view(inputs.size(0), -1)
        optimizer.zero_grad()
        # inputs = inputs + 0.1
        outputs = net(inputs)
        # print(net.nnmf1.weight)
        # print(net.nnmf1.weight.sum(1))
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
            # plt.hist(net.nnmf1.weight.cpu().detach().numpy().flatten(), bins=100, alpha=0.5, label="NNMF1")
            # plt.hist(net.nnmf2.weight.cpu().detach().numpy().flatten(), bins=100, alpha=0.5, label="NNMF2")
            # plt.legend()
            # plt.show()
    evaluate()
