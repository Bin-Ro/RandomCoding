import torch
from torch import nn

loss = nn.MSELoss()

y = torch.arange(72.0).reshape(4, 2, 3, 3)
y_hat = torch.arange(10.0, 82.0).reshape(4, 2, 3, 3)
print(f'y: {y}')
print(f'y_hat: {y_hat}')

l = loss(y, y_hat)
print(f'l: {l}')
