import torch

x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5])
print(f'x: {x}')
print(f'y: {y}')

xx, yy = torch.meshgrid(x, y)
print(f'xx: {xx}')
print(f'yy: {yy}')

xx, yy = torch.meshgrid(x, y, indexing='xy')
print(f'xx: {xx}')
print(f'yy: {yy}')
