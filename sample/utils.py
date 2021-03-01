import torch
import torch.nn as nn
import torchvision

from organa import DatasetWrapper
from organa import Model


class MNIST(DatasetWrapper):
    """The MNIST dataset in torch readable form"""
    def __init__(self, ds_path, train=True, download=False):
        super().__init__()
        ds = torchvision.datasets.MNIST(
            ds_path,
            train=train,
            download=download
        )
        self.data = ds.data.flatten(1).float()
        self.targets = ds.targets
        self.y_dim = len(self.targets.unique())

    def __getitem__(self, i):
        return (self.data[i], self.targets[i])


def load_data(batch_size, train=True, shuffle=True):
    """
    Load the specified dataset in a form suitable for the model

    Keyword arguments:
    options -- options for the simulation
    train -- load the training dataset if true otherwise load the validation
    classes -- use only the classes in list, use all classes if empty list or
    None
    """
    data = MNIST(
        "./data/mnist",
        train=train,
        download=True,
    )
    x_dim, y_dim = data.get_dims()
    return {
        "dataloader": torch.utils.data.DataLoader(
            data,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
        ),
        "x_dim": x_dim,
        "y_dim": y_dim,
    }


class SoftMaxModel(Model):
    """The softmax perceptron class"""
    def __init__(self, params):
        super().__init__([0.01], [], params['device'])
        self.features = nn.ModuleList([
            nn.Linear(
                params['num_in'], params['num_in'] * params['params_mul']
            ),
            nn.Sigmoid(),
            nn.Linear(
                params['num_in'] * params['params_mul'], params['num_out']
            ),
            nn.Softmax(dim=1)
        ]).eval()

    def forward(self, x):
        for feature in self.features:
            x = feature(x)
        return x



def load_model(params):
    """Load the model specified in params"""
    return SoftMaxModel(params).to(params['device'])
