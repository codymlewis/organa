from threading import Thread
import time

import torch
import organa

from utils import load_data, load_model


class ClientThread(Thread):
    def __init__(self, client_id, addr, net, data, server_epochs):
        super().__init__()
        self.client = organa.create_client(
            client_id, addr, net, data
        )
        self.server_epochs = server_epochs

    def run(self):
        while self.client.epoch < self.server_epochs:
            while not self.client.get():
                time.sleep(0.5)
            loss = self.client.post()
            print(f"Client {self.client.id}: Epoch {self.client.epoch}, Loss: {loss}")


if __name__ == '__main__':
    ip, port = "localhost", 8080
    data = load_data(512)
    model_params = {"params_mul": 10, "device": torch.device("cuda:0")}
    model_params['num_in'] = data['x_dim']
    model_params['num_out'] = data['x_dim']
    for i in range(10):
        t = ClientThread(
            i,
            f"{ip}:{port}",
            load_model(model_params),
            load_data(512),
            3000
        )
        t.start()
