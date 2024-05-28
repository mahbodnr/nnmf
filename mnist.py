import torch
# from implicit_backprop.modules import NNMFDense
from nnmf.modules import NNMFDense
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
            backward_method="all_grads",
        )
        self.nnmf2 = NNMFDense(
            in_features=100,
            out_features=10,
            n_iterations=iterations,
            backward_method="all_grads",
        )
        # self.out = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.nnmf1(x)
        x = self.nnmf2(x)
        # x = self.out(x)
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
optimizer = Adam(net.parameters(), lr=0.00001)
# optimizer = torch.optim.AdamW(net.parameters(), lr=0.001)
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
        # print(net.nnmf1.reconstruction_mse)
        # print(net.nnmf2.reconstruction_mse)
        # print(net.nnmf1.weight.sum(1))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs, 1)
        corrects += (predicted == labels).sum().item()
        items += labels.size(0)
        if i % 100 == 0:
            accuracy = corrects / items
            print(f'Epoch: {epoch}, Iteration: {i}, Loss: {loss.item()}, Accuracy: {accuracy}')
            corrects = 0
            items = 0
            # plt.hist(net.nnmf1.weight.cpu().detach().numpy().flatten(), bins=100, alpha=0.5, label="NNMF1")
            # plt.hist(net.nnmf2.weight.cpu().detach().numpy().flatten(), bins=100, alpha=0.5, label="NNMF2")
            # plt.legend()
            # plt.show()
            plt.figure(figsize=(20, 10))
            plt.suptitle(f"add - Iteration: {i}")
            plt.subplot(2, 2, 1)
            plt.plot(torch.tensor(net.nnmf1.reconstruction_mse).detach().cpu().numpy(),)
            plt.title("NNMF1 Reconstruction MSE")
            plt.subplot(2, 2, 2)
            plt.plot(torch.tensor(net.nnmf2.reconstruction_mse).detach().cpu().numpy(),)
            plt.title("NNMF2 Reconstruction MSE")
            plt.subplot(2, 2, 3)
            plt.plot(torch.tensor(net.nnmf1.convergence).detach().cpu().numpy(), )
            plt.title("NNMF1 Convergence")
            plt.subplot(2, 2, 4)
            plt.plot(torch.tensor(net.nnmf2.convergence).detach().cpu().numpy(),)
            plt.title("NNMF2 Convergence")
            plt.show()
    evaluate()
