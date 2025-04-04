import torch
from torch import nn

conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)

X = torch.rand(1, 1, 8, 8)

print(f'X.shape: {X.shape}')
print(f'conv2d(X).shape: {conv2d(X).shape}')

conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
print(f'X.shape: {X.shape}')
print(f'conv2d(X).shape: {conv2d(X).shape}')
