import torch
from torch import nn

model = nn.Linear(2, 1)
optimizer = torch.optim.SGD(model.parameters())
print(f'model: {model}')
print(f'optimizer: {optimizer}')

print(f'optimizer.param_groups:\n')
for i, param_group in enumerate(optimizer.param_groups):
    print(f'param_group {i}:\n')
    for k in param_group:
        print(f'{k}: {param_group[k]}')
