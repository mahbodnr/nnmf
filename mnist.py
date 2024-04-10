import torch
from implicit_backprop.modules import NNMFDense
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import Adam


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        iterations = 1
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

# create model
net = Net().cuda()

# define loss function
criterion = nn.CrossEntropyLoss()
optimizer = Adam(net.parameters(), lr=0.001)

for epoch in range(10):
    for i, data in enumerate(trainloader):
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        inputs = inputs.view(inputs.size(0), -1)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f'Epoch: {epoch}, Iteration: {i}, Loss: {loss.item()}')
