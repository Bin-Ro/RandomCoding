import torch
import matplotlib.pyplot as plt

x = torch.arange(-8., 8., .1)
y = x.sigmoid()

plt.plot(x, y)

plt.xlabel('x')
plt.ylabel('y')
plt.title('sigmoid')
plt.grid(True)
plt.show()
