import torch
from torch import nn

def corr2d(X, K):
    h, w = K.shape
    Y = torch.empty(size=(X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y

X = torch.ones(6, 8)
X[:, 2:6] = 0.
print(f'X: {X}')

K = torch.tensor([[1.0, -1.0]])
print(f'K: {K}')

Y = corr2d(X, K)
print(f'Y: {Y}')

X = X.reshape((1, 1) + X.shape)
Y = Y.reshape((1, 1) + Y.shape)

net = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)

trainer = torch.optim.AdamW(net.parameters(), lr=.1)
loss = nn.MSELoss()

for epoch in range(100):
    l = loss(net(X), Y)
    trainer.zero_grad()
    l.backward()
    trainer.step()
    print(f'epoch: {epoch + 1}, loss: {l}')

print(f'net.state_dict(): {net.state_dict()}')
