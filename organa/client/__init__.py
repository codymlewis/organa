'''
The client/user piece of the FL system

Author: Cody Lewis
'''

import pickle

import requests


class Client:
    '''Client/user class for the FL system.'''
    def __init__(self, client_id, addr, net, data, train_epochs=1,
                 grad_scaling=1):
        '''
        param client_id: ID of the client
        param addr: Resolvable address of the server
        param net: Model to start with
        param data: Data that the client will hold
        param train_epochs: Amount to epochs to before before sending grads
        param grad_scaling: Value to scale the gradients by
        '''
        self.id = client_id
        self.addr = addr
        self.net = net
        self.data = data['dataloader']
        self.epoch = 0
        self.train_epochs = train_epochs
        self.grad_scaling = grad_scaling

    def get(self):
        '''
        Get the current model from the server, copy its parameters.
        '''
        if (r := requests.get(f"http://{self.addr}/{self.epoch}")):
            self.net.copy_params(pickle.loads(r.content))
            return True
        return False

    def post(self):
        '''Train and submit the resulting gradients to the server.'''
        loss, grads = self.net.fit(
            self.data,
            epochs=self.train_epochs,
            scaling=self.grad_scaling,
            verbose=False
        )
        if (r := requests.post(f"http://{self.addr}/",
            data=pickle.dumps({'grads': grads, 'id': self.id}))):
            self.epoch += 1
        return loss
