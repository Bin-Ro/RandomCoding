import torch
from torch import nn
import torchvision
from torch.utils import data
from torchvision import transforms

def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root='/tmp/dive_into_deeplearning/ch3/data', train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root='/tmp/dive_into_deeplearning/ch3/data', train=False, transform=trans, download=True)
    return data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=4), data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=4)

batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)

def accuracy(y_hat, y):
    y_hat = y_hat.argmax(dim=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter):
    with torch.inference_mode():
        metric = torch.zeros(2, dtype=torch.float32, device='cuda')
        for X, y in data_iter:
            X, y = X.to('cuda'), y.to('cuda')
            metric += torch.tensor([accuracy(net(X), y), y.numel()], device='cuda')
    return metric[0] / metric[1]

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.W = nn.Parameter(torch.randn(28 * 28, 10))
        self.b = nn.Parameter(torch.zeros(10))

    def forward(self, X):
        return X.reshape(-1, 28 * 28) @ self.W + self.b

net = MyNet()
net.to('cuda')

num_epochs = 10
class MyLoss:
    def __init__(self):
        pass

    def __call__(self, y, y_hat):
        y_exp = y.exp()
        partition = y_exp.sum(dim=1, keepdim=True)
        y_normalize = y_exp / partition
        return -y_normalize[range(len(y_normalize)), y_hat].log().mean()

loss = MyLoss()

class MyTrainer:
    def __init__(self, parameters):
        self.parameters = list(parameters)
        self.lr = 1e-1

    def zero_grad(self):
        with torch.inference_mode():
            for parameter in self.parameters:
                if parameter.grad is not None:
                    parameter.grad.zero_()

    def step(self):
        with torch.inference_mode():
            for parameter in self.parameters:
                if parameter.grad is not None:
                    parameter -= self.lr * parameter.grad

trainer = MyTrainer(net.parameters())

net.eval()
with torch.inference_mode():
    print(f'train_acc: {evaluate_accuracy(net, train_iter)}')
    print(f'test_acc: {evaluate_accuracy(net, test_iter)}\n')

for epoch in range(num_epochs):
    net.train()
    for X, y in train_iter:
        X, y = X.to('cuda'), y.to('cuda') 
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    net.eval()
    with torch.inference_mode():
        print(f'epoch: {epoch + 1}, train_acc: {evaluate_accuracy(net, train_iter)}')
        print(f'epoch: {epoch + 1}, test_acc: {evaluate_accuracy(net, test_iter)}\n')
