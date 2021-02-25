import requests
import pickle
import torch
import Model

class Client:
    def __init__(self, addr, device):
        self.addr = addr
        self.device = device
        self.net = Model.Net().to(device)

    def get(self):
        if (r := requests.get(f"http://{self.addr[0]}:{self.addr[1]}")):
            self.net.copy_params(pickle.loads(r.content))


if __name__ == '__main__':
    addr = ("localhost", 50002)
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
    client = Client(addr, device)
    print([p for p in client.net.parameters()])
    client.get()
    print([p for p in client.net.parameters()])
