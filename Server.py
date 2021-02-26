import socket
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
import torch
import pickle
import torchvision

from io import BytesIO
from urllib.parse import urlparse
import utils

class RequestHandler(BaseHTTPRequestHandler):
    def __init__(self, net, options):
        self.net = net
        self.grads = dict()
        self.grad_count = 0
        self.epoch = 0
        self.fit_fun = load_fit_fun(options.fit_fun)
        self.options = options

    def server_init(self, *args):
        super().__init__(*args)
        return self

    def do_GET(self):
        query = urlparse(self.path).query
        params = dict(q.split('=') for q in query.split('&'))
        if (epoch := params.get('epoch')):
            if int(epoch) <= self.epoch:
                self.send_response(200)
                self.send_header("Content-type", "application/octet-stream")
                self.end_headers()
                self.wfile.write(pickle.dumps([p for p in self.net.parameters()]))
            else:
                self.send_response(425)
                self.send_header("Content-type", "text/plain")
                self.end_headers()
                self.wfile.write(b"Too Early")
        else:
            self.send_response(400)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Need to specify the current epoch")


    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)
        msg = pickle.loads(body)
        self.grads[msg['id']] = msg['grads']
        self.grad_count += 1
        if self.grad_count >= self.options.users:
            self.fit_fun(self.net, self.grads, None)
            self.epoch += 1
            self.grad_count = 0
        self.send_response(200)
        self.end_headers()
        response = BytesIO()
        response.write(b'This is POST request. ')
        response.write(b'Received: a grad')
        self.wfile.write(response.getvalue())

def load_fit_fun(fn_name):
    """Load the class of the specified adversary"""
    fit_funs = {
        "federated averaging": fed_avg,
    }
    if (chosen_fit_fun := fit_funs.get(fn_name)) is None:
        raise utils.errors.MisconfigurationError(
        f"Fitness function '{fn_name}' does not exist, " +
        f"possible options: {set(fit_funs.keys())}"
    )
    return chosen_fit_fun


def fed_avg(net, grads, _params):
    """Perform federated averaging across the client gradients"""
    with torch.no_grad():
        total_dc = sum([g["data_count"] for g in grads.values()])
        for g in grads.values():
            alpha = g["data_count"] / total_dc
            for k, p in enumerate(net.parameters()):
                p.data.add_(alpha * g["grads"][k])

class Server:
    def __init__(self, addr, net, options):
        self.net = net
        rh = RequestHandler(self.net, options)
        self.httpserver = HTTPServer(addr, rh.server_init)

    def serve_forever(self):
        self.httpserver.serve_forever()

    def server_close(self):
        self.httpserver.server_close()


if __name__ == '__main__':
    from utils import load_options, load_data, load_model
    addr = ("localhost", 50002)
    options = load_options()
    data = load_data(options)
    options.model_params['num_in'] = data['x_dim']
    options.model_params['num_out'] = data['x_dim']
    server = Server(addr, load_model(options.model_params), options)
    print(f"Serving Federated Learning at http://{addr[0]}:{addr[1]}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass

    server.server_close()
    print("Server stopped.")
