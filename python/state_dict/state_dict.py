import torch
from torch import nn

net = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 1))

print(f'net: {net}')
print(f'net.state_dict(): {net.state_dict()}')
print(f'net.named_parameters(): {list(net.named_parameters())}')
