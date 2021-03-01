'''
An HTTP based federated learning server

Author: Cody Lewis
'''

import pickle

import bottle


class Server:
    '''FL Server object.'''
    def __init__(self, net, k, fit_fun):
        '''
        param net: Model to start with
        param k: Minimum number of clients to use for an update
        param fit_fun: Fitting function to use
        '''
        self.net = net
        self.grads = dict()
        self.grad_count = 0
        self.epoch = 0
        self.k = k
        self.fit_fun = fit_fun

    def send(self, epoch):
        '''
        Send the current global model parameters if the user is on time.

        param epoch: Epoch the user is up to
        '''
        if int(epoch) == self.epoch:
            return bottle.HTTPResponse(
                status=200,
                body=pickle.dumps([p for p in self.net.parameters()])
            )
        return bottle.HTTPResponse(status=425, body="Too Early")

    def get(self):
        '''Recieve a gradient from the user.'''
        body = bottle.request.body.read()
        msg = pickle.loads(body)
        self.grads[msg['id']] = msg['grads']
        self.grad_count += 1
        if self.grad_count >= self.k:
            self.fit_fun(self.net, self.grads, None)
            self.epoch += 1
            self.grad_count = 0
        return bottle.HTTPResponse(
            status=200,
            body=f"Gradient Received from User {msg['id']}"
        )
