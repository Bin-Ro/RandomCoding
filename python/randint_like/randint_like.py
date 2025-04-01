import torch

x = torch.zeros(2, 3)
y = torch.randint_like(x, low=-2, high=5)
z = torch.randint_like(x, low=-2, high=5, dtype=torch.int32)
print(f'x: {x}')
print(f'y: {y}')
print(f'z: {z}')
