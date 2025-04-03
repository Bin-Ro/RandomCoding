import torch

a = torch.arange(6.0).reshape(2, 3)
b = torch.arange(6.0).reshape(2, 3) + 1
print(f'a: {a}')
print(f'b: {b}')
print(f'torch.cat((a, b), dim=0): {torch.cat((a, b), dim=0)}')
print(f'torch.cat((a, b), dim=1): {torch.cat((a, b), dim=1)}')
