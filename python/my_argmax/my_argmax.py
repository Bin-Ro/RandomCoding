import torch

x = torch.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
print(f'x: {x}')
print(f'x.argmax(dim=1): {x.argmax(dim=1)}')
