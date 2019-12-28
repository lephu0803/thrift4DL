# Copyright (c) 2019 congvm
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


import sys
import os
from .api import create_app
from multiprocessing import Process
from wsgiserver import WSGIServer


class HTTPServer(Process):
    def __init__(self, host, port, http_port):
        Process.__init__(self)
        self.host = host
        self.port = port
        self.http_port = http_port

    def start_http_server(self, host, port, http_port):
        app = create_app(host=host, port=port)
        http_server = WSGIServer(app, host='0.0.0.0', port=int(http_port))
        http_server.start()

    def run(self):
        print(f"Start HTTPServer on 0.0.0.0:{self.http_port}")
        self.start_http_server(self.host, self.port, self.http_port)
