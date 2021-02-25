import socket
from http.server import BaseHTTPRequestHandler, HTTPServer
import torch
import pickle
import torchvision

import Model


class RequestHandler(BaseHTTPRequestHandler):
    def __init__(self, net):
        self.net = net

    def server_init(self, *args):
        super().__init__(*args)
        return self

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "application/octet-stream")
        self.end_headers()
        self.wfile.write(pickle.dumps([p for p in self.net.parameters()]))


class Server:
    def __init__(self, addr, device):
        self.net = Model.Net().to(device)
        rh = RequestHandler(self.net)
        self.httpserver = HTTPServer(addr, lambda *args: rh.server_init(*args))

    def serve_forever(self):
        self.httpserver.serve_forever()

    def server_close(self):
        self.httpserver.server_close()


if __name__ == '__main__':
    addr = ("localhost", 50002)
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
    server = Server(addr, device)
    print([p for p in server.net.parameters()])
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass

    server.server_close()
    print("Server stopped.")
