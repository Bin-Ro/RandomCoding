import torch
from torch import nn
from torch.nn import functional as F

class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, mod in enumerate(args):
            self._modules[str(idx)] = mod

    def forward(self, X):
        for mod in self._modules.values():
            X = mod(X)
        return X

class MyLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, X):
        return X @ self.weight + self.bias

class MyReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return F.relu(X)

net = MySequential(MyLinear(2, 2), MyReLU(), MyLinear(2, 1))
print(f'net: {net}')
print(f'net.state_dict(): {net.state_dict()}')
print(f'net.named_parameters(): {list(net.named_parameters())}')
