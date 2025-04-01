import torch

x = torch.arange(6)
y = x.view(2, 3)
z = x.view(3, 2)
print(f'x: {x}')
print(f'y: {y}')
print(f'z: {z}')

x = torch.arange(4).reshape(2, 2)
y = x.view(-1)
print(f'x: {x}')
print(f'y: {y}')
y[0] = -99
print(f'x: {x}')
print(f'y: {y}')

x = torch.arange(4).reshape(2, 2).t()
y = x.contiguous().view(-1)
z = x.reshape(-1)
print(f'x: {x}')
print(f'y: {y}')
print(f'z: {z}')
