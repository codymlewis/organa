import torch

import organa

from utils import load_data, load_model


if __name__ == '__main__':
    ip, port = "localhost", 8080
    K = 10
    data = load_data(512)
    model_params = {"params_mul": 10, "device": torch.device("cuda:0")}
    model_params['num_in'] = data['x_dim']
    model_params['num_out'] = data['x_dim']
    organa.start_server(
        ip, port, "bjoern", load_model(model_params), K
    )
