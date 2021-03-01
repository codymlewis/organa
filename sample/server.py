import torch

import organa

from utils import load_data, load_model


def fed_avg(net, grads, _params):
    """Perform federated averaging across the client gradients"""
    with torch.no_grad():
        total_dc = sum([g["data_count"] for g in grads.values()])
        for g in grads.values():
            alpha = g["data_count"] / total_dc
            for k, p in enumerate(net.parameters()):
                p.data.add_(alpha * g["grads"][k])


if __name__ == '__main__':
    ip, port = "localhost", 8080
    K = 10
    data = load_data(512)
    model_params = {"params_mul": 10, "device": torch.device("cuda:0")}
    model_params['num_in'] = data['x_dim']
    model_params['num_out'] = data['x_dim']
    organa.start_server(ip, port, load_model(model_params), K, fed_avg)
