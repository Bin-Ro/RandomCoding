import torch
from torch import nn

x = torch.tensor(1.0, requires_grad=False)
param = nn.Parameter(x)
print(f'x: {x}')
print(f'param: {param}')
print(f'param.requires_grad: {param.requires_grad}')

class SimpleModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))
        self.bias = torch.randn(output_dim, requires_grad=True)

    def forward(self, X):
        return X @ self.weight + self.bias

net = SimpleModel(3, 2)
print(f'net.parameters(): {list(net.parameters())}')
