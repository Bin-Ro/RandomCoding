import torch
from torch import nn

x = torch.arange(27.0).reshape(1, 3, 3, 3)

net = nn.AdaptiveAvgPool2d((1, 1))

print(f'x: {x}')
print(f'net: {net}')
print(f'net(x): {net(x)}')
