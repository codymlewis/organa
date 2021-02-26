import requests
import pickle
import torch
import time
from threading import Thread

class Client:
    def __init__(self, client_id, addr, net, data):
        self.id = client_id
        self.addr = addr
        self.net = net
        self.data = data['dataloader']
        self.epoch = 0

    def get(self):
        if (r := requests.get(f"http://{self.addr[0]}:{self.addr[1]}",
            params={"epoch": self.epoch})):
            self.net.copy_params(pickle.loads(r.content))
            return True
        return False

    def post(self):
        loss, grads = self.net.fit(self.data)
        if (r := requests.post(f"http://{self.addr[0]}:{self.addr[1]}",
            data=pickle.dumps({'grads': grads, 'id': self.id}))):
            self.epoch += 1

class ClientThread(Thread):
    def __init__(self, client_id, addr, options, data):
        super().__init__()
        self.client = Client(client_id, addr, load_model(options.model_params), data)
        self.options = options

    def run(self):
        while self.client.epoch < self.options.server_epochs:
            while not self.client.get():
                time.sleep(1)
            self.client.post()

if __name__ == '__main__':
    from utils import load_options, load_data, load_model
    addr = ("localhost", 50002)
    options = load_options()
    data = load_data(options)
    options.model_params['num_in'] = data['x_dim']
    options.model_params['num_out'] = data['x_dim']
    for i in range(options.users):
        t = ClientThread(
            i,
            addr,
            options,
            load_data(options, classes=options.class_shards[i])
        )
        t.start()
