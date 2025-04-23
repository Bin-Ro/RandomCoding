import torch
from torch import nn
from torch.utils import data
import matplotlib.pyplot as plt

T = 1000
time = torch.arange(1, T + 1, dtype=torch.float32, device='cuda')
x = torch.sin(.01 * time) + torch.normal(0, .1, (T,), device='cuda')

tau = 4
n_train = 600
features = torch.empty(size=(T - tau, tau), device='cuda')
for i in range(tau):
    features[:, i] = x[i: T - tau + i]
labels = x[tau:].reshape(-1, 1)

net = nn.Sequential(nn.Linear(tau, 512),
        nn.ELU(),
        nn.Linear(512, 256),
        nn.ELU(),
        nn.Linear(256, 128),
        nn.ELU(),
        nn.Linear(128, 1))
net.to('cuda')

loss = nn.MSELoss()
trainer = torch.optim.AdamW(net.parameters())

def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

train_iter = load_array((features[:n_train], labels[:n_train]), batch_size=16)

net.eval()
with torch.inference_mode():
    print(f'loss: {loss(net(features), labels)}')

for epoch in range(200):
    net.train()
    for X, y in train_iter:
        X, y = X.to('cuda'), y.to('cuda')
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    net.eval() 
    with torch.inference_mode():
        print(f'epoch: {epoch + 1}, loss: {loss(net(features), labels)}')

onestep_preds = net(features)

plt.plot(time.detach().cpu(), x.detach().cpu(), label='cos')
plt.plot(time[tau:].detach().cpu(), onestep_preds.detach().cpu(), label='predicted cos')
plt.title('cos')
plt.xlabel('time')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.show()
