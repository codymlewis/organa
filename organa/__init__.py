'''
A modular HTTP based FL system.

Author: Cody Lewis
'''

from abc import abstractmethod

import bottle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset

from organa import client, server


def create_client(client_id, addr, net, data, **kwargs):
    '''
    Create and return a client object ready to interact with a server.

    param client_id: ID of the client
    param addr: Resolvable address of the server
    param net: Model to start with
    param data: Data that the client will hold
    param kwargs: Addition arguments passed to the client constructor
    '''
    return client.Client(client_id, addr, net, data, **kwargs)


def fed_avg(net, grads, **kwargs):
    """
    Perform federated averaging across the client gradients.

    param net: The global model
    param grads: Gradients from the clients
    """
    with torch.no_grad():
        total_dc = sum([g["data_count"] for g in grads.values()])
        for g in grads.values():
            alpha = g["data_count"] / total_dc
            for k, p in enumerate(net.parameters()):
                p.data.add_(alpha * g["grads"][k])


def start_server(ip, port, server_name, net, k, fit_fun=fed_avg):
    '''
    Create a FL server, set up the routes, and start it.

    param ip: IP of the host
    param port: Port of the host
    param server_name: Name of the chosen server as per
        http://bottlepy.org/docs/dev/deployment.html#switching-the-server-backend
    param net: Model to start with
    param k: Minimum number of clients to use for an update
    param fit_fun: Fitting function to use
    '''
    print("Organa federated learning server")
    s = server.Server(net, k, fit_fun)
    bottle.route('/<epoch>')(s.send)
    bottle.route('/', method="POST")(s.get)
    bottle.run(server=server_name, host=ip, port=port)


class Model(nn.Module):
    '''Abstract model that is compatible with this package'''
    def __init__(self, lr, lr_changes, device):
        '''
        param lr: List of learning rates to use
        param lr_changes: List of times to change to the next learning rate
        param device: Device to run ML operations on
        '''
        super().__init__()
        self.lr = lr[0]
        self.learning_rates = lr.copy()
        del self.learning_rates[0]
        self.lr_changes = lr_changes.copy()
        self.device = device
        self.epoch_count = 0

    @abstractmethod
    def forward(self, *x):
        """
        The torch prediction function.

        param x: Input vector
        """
        pass

    def fit(self, data, epochs=1, scaling=1, verbose=True):
        """
        Fit the model for some epochs, return history of loss values and the
        gradients of the changed parameters

        param data: Iterable x, y pairs
        param epochs: number of epochs to train for
        param verbose: output training stats if True
        """
        optimizer = optim.SGD(
            self.parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=0.0001
        )
        criterion = nn.CrossEntropyLoss()
        data_count = 0
        for i in range(epochs):
            optimizer.zero_grad()
            x, y = next(iter(data))
            x = x.to(self.device)
            y = y.to(self.device)
            output = self(x)
            loss = criterion(output, y)
            if verbose:
                print(
                    f"Epoch {i + 1}/{epochs} loss: {loss}",
                    end="\r"
                )
            loss.backward()
            optimizer.step()
            data_count += len(y)
        self.epoch_count += 1
        if self.lr_changes and self.epoch_count > self.lr_changes[0]:
            self.lr = self.learning_rates[0]
            del self.learning_rates[0]
            del self.lr_changes[0]
        if verbose:
            print()
        return loss, {
            "grads": [scaling * -self.lr * p.grad for p in self.parameters()],
            "data_count": data_count
        }

    def get_params(self):
        """Get the tensor form parameters of this model."""
        return [p.data for p in self.parameters()]

    def copy_params(self, params):
        """
        Copy input parameters into self.

        param params: Global model paramters list
        """
        for p, t in zip(params, self.parameters()):
            t.data.copy_(p)


class DatasetWrapper(Dataset):
    """Wrapper class for torch datasets to allow for easy non-iid splitting"""
    def __init__(self):
        self.targets = torch.tensor([])
        self.y_dim = 0

    def __len__(self):
        return len(self.targets)

    @abstractmethod
    def __getitem__(self, i):
        pass

    def get_dims(self):
        """Get the x and y dimensions of the dataset"""
        if len(self) < 1:
            return (0, 0)
        x, _ = self[0]
        return (x.shape[0], self.y_dim)

    def get_idx(self, classes):
        """Get the ids of data belong to the specified classes"""
        return torch.arange(len(self.targets))[
            sum([(self.targets == i).long() for i in classes]).bool()
        ]

    def assign_to_classes(self, classes):
        """Leave only data belonging to the classes within this set"""
        idx = self.get_idx(classes)
        self.data = self.data[idx]
        self.targets = self.targets[idx]
