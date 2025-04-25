import os 
import requests
import re
import hashlib
import collections
import random
import torch
from torch import nn
from torch.nn import functional as F

DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'
DATA_HUB['time_machine'] = (DATA_URL + 'timemachine.txt', '090b5e7e70c295757f55df93cb0a180b9691891a')

def download(name, cache_dir='data'):
    assert name in DATA_HUB, f'{name} does not exist in {DATA_HUB}'
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname

def read_time_machine():
    with open(download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

#lines = read_time_machine()
#for line in lines:
#    print(line)

def tokenize(lines):
    return [list(line) for line in lines]

#tokens = tokenize(lines)
#for token in tokens:
#    print(token)

def count_corpus(tokens):
    if isinstance(tokens[0], (list, tuple)):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

class Vocab:
    def __init__(self, tokens):
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.idx_to_token = ['<unk>'] + [token for token, _ in self._token_freqs]
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[idx] for idx in indices]

    @property
    def unk(self):
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

#vocab = Vocab(tokens)
#print("vocab.to_tokens(3):", vocab.to_tokens(3))

def seq_data_iter_random(corpus, batch_size, num_steps):
    corpus = corpus[random.randint(0, num_steps - 1):]
    num_subseqs = (len(corpus) - 1) // num_steps
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    random.shuffle(initial_indices)

    def data(pos):
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)

def seq_data_iter_sequential(corpus, batch_size, num_steps):
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y

def load_corpus_time_machine():
    lines = read_time_machine()
    tokens = tokenize(lines)
    vocab = Vocab(tokens)
    corpus = [vocab[token] for line in tokens for token in line]
    return corpus, vocab

#corpus, vocab = load_corpus_time_machine()

class SeqDataLoader:
    def __init__(self, batch_size, num_steps, use_random_iter):
        self.data_iter_fn = seq_data_iter_random if use_random_iter else seq_data_iter_sequential
        self.corpus, self.vocab = load_corpus_time_machine()
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)

def load_data_time_machine(batch_size, num_steps, use_random_iter=False):
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter)
    return data_iter, data_iter.vocab

#for i, (X, Y) in enumerate(train_iter):
#    print(f'i: {i}')
#    print(f'X: {X}')
#    print(f'Y: {Y}\n')
#print(f'len(train_iter.corpus): {len(train_iter.corpus)}')

#my_seq = list(range(35))
#for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
#    print(f'X: {X}')
#    print(f'Y: {Y}\n')

#for X, Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
#    print(f'X: {X}')
#    print(f'Y: {Y}\n')

def get_params(vocab_size, num_hiddens):
    num_inputs = num_ouputs = vocab_size

    W_xh = torch.normal(0, .01, size=(num_inputs, num_hiddens), requires_grad=True)
    W_hh = torch.normal(0, .01, size=(num_hiddens, num_hiddens), requires_grad=True)
    b_h = torch.zeros(num_hiddens, requires_grad=True)

    W_hq = torch.normal(0, .01, size=(num_hiddens, num_ouputs), requires_grad=True)
    b_q = torch.zeros(num_ouputs, requires_grad=True)

    return [W_xh, W_hh, b_h, W_hq, b_q]

def init_rnn_state(batch_size, num_hiddens):
    return torch.zeros(size=(batch_size, num_hiddens))

def rnn(inputs, state, params):
    W_xh, W_hh, b_h, W_hq, b_q = params
    H = state
    outputs = []
    for X in inputs:
        H = torch.tanh(X @ W_xh + H @ W_hh + b_h)
        Y = H @ W_hq + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), H

class RNNModelScratch:
    def __init__(self, vocab_size, num_hiddens, get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens 
        self.params = get_params(vocab_size, num_hiddens)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size):
        return self.init_state(batch_size, self.num_hiddens)

train_iter, vocab = load_data_time_machine(batch_size=32, num_steps=35)
num_hiddens = 512
net = RNNModelScratch(len(vocab), num_hiddens, get_params, init_rnn_state, rnn)

X, _ = next(iter(train_iter))

state = net.begin_state(X.shape[0])
Y, new_state = net(X, state)
print(f'Y.shape: {Y.shape}')
print(f'new_state.shape: {new_state.shape}')
