import torch

a = torch.arange(9.0).reshape(3, 3)
b = torch.arange(9.0).reshape(3, 3) + 1
print(f'a: {a}')
print(f'b: {b}')
print(f'torch.stack((a, b), dim=0): {torch.stack((a, b), dim=0)}')
print(f'torch.stack((a, b), dim=1): {torch.stack((a, b), dim=1)}')
