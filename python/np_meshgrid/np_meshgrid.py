import numpy as np

x = np.array([1, 2, 3])
y = np.array([4, 5])
print(f'x: {x}')
print(f'y: {y}')

xx, yy = np.meshgrid(x, y, sparse=True)
print(f'xx: {xx}')
print(f'yy: {yy}')

# broadcast
print(f'xx + yy: {xx + yy}')
