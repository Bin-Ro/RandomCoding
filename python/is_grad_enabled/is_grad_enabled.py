import torch

print(f'torch.is_grad_enabled(): {torch.is_grad_enabled()}')

with torch.inference_mode():
    print(f'torch.is_grad_enabled(): {torch.is_grad_enabled()}')
