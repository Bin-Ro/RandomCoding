import torch

x = torch.rand(size=(2, 3, 4))
print(f'x.shape: {x.shape}')
print(f'x.T.shape: {x.T.shape}')
print(f'x.transpose(0, 1).shape: {x.transpose(0, 1).shape}')
print(f'x.transpose(1, 2).shape: {x.transpose(1, 2).shape}')
print(f'x.permute(1, 0, 2).shape: {x.permute(1, 0, 2).shape}')
print(f'x.permute(0, 2, 1).shape: {x.permute(0, 2, 1).shape}')
