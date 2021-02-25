#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.features = nn.ModuleList([
            nn.Linear(2, 3),
            nn.Sigmoid(),
            nn.Linear(3, 1),
            nn.Sigmoid()
        ]).eval()

    def forward(self, x):
        for feature in self.features:
            x = feature(x)
        return x

    def fit(self, x, y, epochs, verbose=True):
        optimizer = optim.AdamW(self.parameters())
        criterion = nn.MSELoss()
        history = {'loss': []}
        for i in range(epochs):
            optimizer.zero_grad()
            output = self(x)
            history['loss'].append(criterion(output, y))
            if verbose:
                print(f"Epoch {i + 1}/{epochs} loss: {history['loss'][-1]}", end="\r")
            history['loss'][-1].backward()
            optimizer.step()
        if verbose:
            print()
        return history

    def copy_params(self, params):
        for p, t in zip(params, self.parameters()):
            t.data.copy_(p)


if __name__ == '__main__':
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
    x = torch.tensor([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ], device=device)
    y = torch.tensor([
        [0.0],
        [1.0],
        [1.0],
        [0.0]
    ], device=device)
    net = Net()
    net.to(device)
    print(net)
    history = net.fit(x, y, 1_000)
    print(net(x))
