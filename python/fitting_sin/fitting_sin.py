import torch
from torch import nn
from torch.utils import data
import matplotlib.pyplot as plt

features = torch.arange(-10, 10, .001).reshape(-1, 1)
labels = features.sin() + torch.randn_like(features)
features, labels = features.to('cuda'), labels.to('cuda')
shuffle_idx = torch.randperm(len(features))
shuffle_features = features[shuffle_idx]
shuffle_labels = labels[shuffle_idx]
shuffle_features, shuffle_labels = shuffle_features.to('cuda'), shuffle_labels.to('cuda')

def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

train_features = shuffle_features[:len(shuffle_features) // 2]
train_labels = shuffle_labels[:len(shuffle_labels) // 2]

test_features = shuffle_features[len(shuffle_features) // 2:]
test_labels = shuffle_labels[len(shuffle_labels) // 2:]

train_iter = load_array((train_features, train_labels), batch_size=256)

net = nn.Sequential(nn.Linear(1, 512),
        nn.ELU(),
        nn.Linear(512, 256),
        nn.ELU(),
        nn.Linear(256, 128),
        nn.ELU(),
        nn.Linear(128, 1))
net = net.to('cuda')
print(f'net: {net}')

loss = nn.MSELoss()
trainer = torch.optim.AdamW(net.parameters())

net.eval()
with torch.inference_mode():
    print(f'train_loss: {loss(net(train_features), train_labels)}')
    print(f'test_loss: {loss(net(test_features), test_labels)}\n')

for epoch in range(200):
    net.train()
    for X, y in train_iter:
        X, y = X.to('cuda'), y.to('cuda')
        trainer.zero_grad()
        l = loss(net(X), y)
        l.backward()
        trainer.step()
    net.eval()
    with torch.inference_mode():
        print(f'epoch: {epoch + 1}, train_loss: {loss(net(train_features), train_labels)}')
        print(f'epoch: {epoch + 1}, test_loss: {loss(net(test_features), test_labels)}\n')

net.eval()
with torch.inference_mode():
    plt.plot(features.detach().cpu(), labels.detach().cpu(), label='noise sin')
    plt.plot(features.detach().cpu(), features.sin().detach().cpu(), label='sin')
    plt.plot(features.detach().cpu(), net(features).detach().cpu(), label='fitting sin')
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('fitting sin')
    plt.legend()
    plt.show()
