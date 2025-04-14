import torch
from torch import nn

X = torch.arange(50.0).reshape(10, 5)
y = torch.randint(0, 5, size=(10,))

print(f'X: {X}')
print(f'y: {y}')

loss = nn.CrossEntropyLoss()
l = loss(X, y)
print(f'l: {l}')

loss = nn.CrossEntropyLoss(reduction='none')
l = loss(X, y)
print(f'l: {l}')

loss = nn.CrossEntropyLoss(reduction='sum')
l = loss(X, y)
print(f'l: {l}')
