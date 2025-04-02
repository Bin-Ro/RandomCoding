from scipy import interpolate
import numpy as np

x = np.array([-1, 0, 1.0])
y = np.array([-1, 0, 1.0])
print(f'x: {x}')
print(f'y: {y}')

z = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0.]])
print(f'z: {z}')

f = interpolate.interp2d(x, y, z, kind='linear')

xnew = np.array([-1, -.5, 0, .5, 1])
ynew = np.array([-1, -.5, 0, .5, 1])

znew = f(xnew, ynew)
print(f'znew: {znew}')
