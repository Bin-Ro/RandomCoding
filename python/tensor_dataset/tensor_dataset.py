import torch
from torch.utils import data

X = torch.arange(30.0).reshape(10, 3)
y = torch.arange(10.0)
print(f'X: {X}')
print(f'y: {y}')

dataset = data.TensorDataset(X, y)
data_iter = data.DataLoader(dataset, batch_size=3, shuffle=True)

for i, (X, y) in enumerate(data_iter):
    print(f'\ni: {i}')
    print(f'X: {X}')
    print(f'y: {y}\n')

X = torch.arange(180.0).reshape(10, 2, 3, 3)
y = torch.arange(10.0)
print(f'X: {X}')
print(f'y: {y}')

dataset = data.TensorDataset(X, y)
data_iter = data.DataLoader(dataset, batch_size=3, shuffle=True)

for i, (X, y) in enumerate(data_iter):
    print(f'\ni: {i}')
    print(f'X: {X}')
    print(f'y: {y}\n')
