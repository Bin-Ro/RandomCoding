import torch
from torch import nn

X = torch.arange(32.0).reshape(2, 4, 4)

pool2d = nn.MaxPool2d(3)

print(f'X: {X}, {X.shape}')
res = pool2d(X)
print(f'pool2d(X): {res}, {res.shape}')
