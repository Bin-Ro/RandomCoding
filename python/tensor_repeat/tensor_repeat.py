import torch

x = torch.tensor([1, 2, 3])
y = x.repeat(3)
print(f'x: {x}')
print(f'y: {y}')

x = torch.arange(1, 5).reshape(2, 2)
y = x.repeat(2, 3)
print(f'x: {x}')
print(f'y: {y}')
