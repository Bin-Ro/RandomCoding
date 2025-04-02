import numpy as np

print(f'np.random.choice([10, 20, 30, 40]): {np.random.choice([10, 20, 30, 40])}')
print(f'np.random.choice([10, 20, 30, 40], size=3): {np.random.choice([10, 20, 30, 40], size=3)}')
print(f'np.random.choice([10, 20, 30, 40], size=3, replace=False): {np.random.choice([10, 20, 30, 40], size=3, replace=False)}')
print(f'np.random.choice(10, size=5, replace=False): {np.random.choice(10, size=5, replace=False)}')
print(f'np.random.choice([10, 20, 30, 40], size=3, p=[.1, .2, .3, .4]): {np.random.choice([10, 20, 30, 40], size=3, p=[.1, .2, .3, .4])}')
