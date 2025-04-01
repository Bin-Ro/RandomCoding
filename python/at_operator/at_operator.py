import torch

a = torch.arange(12).reshape(2, 2, 3)
b = torch.arange(18).reshape(2, 3, 3)

print(f'a: {a}')
print(f'b: {b}')
print(f'a @ b: {a @ b}')

a = torch.arange(12).reshape(2, 2, 3)
b = torch.arange(9).reshape(3, 3)

print(f'a: {a}')
print(f'b: {b}')
print(f'a @ b: {a @ b}')

a = torch.arange(36).reshape(2, 3, 2, 3)
b = torch.arange(54).reshape(2, 3, 3, 3)

print(f'a: {a}')
print(f'b: {b}')
print(f'a @ b: {a @ b}')

a = torch.arange(12).reshape(2, 1, 2, 3)
b = torch.arange(12).reshape(2, 3, 2)

print(f'a: {a}')
print(f'b: {b}')
print(f'a @ b: {a @ b}')

# error dimension
a = torch.arange(8).reshape(2, 2, 2)
b = torch.arange(12).reshape(3, 2, 2)

print(f'a: {a}')
print(f'b: {b}')
print(f'a @ b: {a @ b}')
