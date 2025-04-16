import torch
from torch import nn

class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, mod in enumerate(args):
            self._modules[str(idx)] = mod
    
    def forward(self, X):
        for mod in self._modules.values():
            X = mod(X)
        return X

net = MySequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 1))
print(f'net: {net}')
