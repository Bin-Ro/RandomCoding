import torch

a = torch.tensor([1, 2, 3])
b = torch.tensor([-1, -2, -3])

condition = torch.tensor([True, False, True])

result = torch.where(condition, a, b)
print(result)

condition = torch.tensor([[True], [False]])
x = torch.tensor([1, 2, 3])
y = torch.tensor(-1)

result = torch.where(condition, x, y)
print(result)

x = torch.tensor([0, 1, 1, 0])
result = torch.where(x == 1)
print(result)

x = torch.tensor([-1, 2, -3])
result = torch.where(x < 0, x, torch.zeros_like(x))
print(result)

x = torch.tensor([-1, 2, -3])
result = torch.where(x < 0, x**2, x)
print(result)

x = torch.tensor([[1, 0], [0, 1]])
row, col = torch.where(x != 0)
print(row, col)
