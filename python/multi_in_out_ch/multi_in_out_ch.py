import torch
from torch import nn

def corr2d(X, K):
    h, w = K.shape
    Y = torch.empty(size=(X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y

def corr2d_multi_in(X, K):
    return sum(corr2d(x, k) for x, k in zip(X, K))

def corr2d_multi_in_out(X, K):
    return torch.stack([corr2d_multi_in(X, k) for k in K], dim=0)

X = torch.empty(2, 3, 3)
X[0] = torch.arange(9.0).reshape(3, 3)
X[1] = X[0] + 1

K = torch.empty(2, 2, 2)
K[0] = torch.arange(4.0).reshape(2, 2)
K[1] = K[0] + 1
K = torch.stack([K, K + 1, K + 2], dim=0)

print(f'X: {X}')
print(f'K: {K}')
print(f'corr2d_multi_in_out(X, K): {corr2d_multi_in_out(X, K)}')
