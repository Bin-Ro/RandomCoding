import torch

x = torch.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
print(f'x: {x}')
print(f'x.argmax(dim=1): {x.argmax(dim=1)}')

y = torch.tensor([[0, 0, 1, 0, 0]])
print(f'y: {y}')
print(f'y.argmax(dim=1): {y.argmax(dim=1)}')
print(f'y.argmax(dim=1).reshape(1): {y.argmax(dim=1).reshape(1)}')
