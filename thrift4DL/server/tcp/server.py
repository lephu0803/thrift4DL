# Copyright (c) 2019 congvm
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from multiprocessing import Manager
import time
import multiprocessing
from thrift.transport import TSocket
import traceback
from thrift.transport.TTransport import TTransportException
from thrift.protocol import TBinaryProtocol
import warnings
import numpy as np
from thrift.Thrift import TType, TMessageType, TApplicationException
import random

from ..http import HTTPServer


IDLE_QUEUE_BLOCK_TIME_SEC = 10

__all__ = ['TModelPoolServer']


class TModelPoolServerBase():
    def __init__(self):
        pass

    def serve(self):
        pass


class TModelPoolServerV1(TModelPoolServerBase):
    def __init__(self, host, port, handler_cls,
                 model_path, gpu_ids, mem_fractions, http_port=None,
                 batch_infer_size=1, batch_group_timeout=1,
                 verbose=True, logger=None):
        self.handler_cls = handler_cls
        self.model_path = model_path
        self.gpu_ids = gpu_ids
        self.mem_fractions = mem_fractions
        self.host = host
        self.port = port
        self.socket = TSocket.TServerSocket(host=self.host, port=self.port)
        self.batch_infer_size = batch_infer_size
        self.batch_group_timeout = batch_group_timeout
        self.handlers = []
        self.is_running = False
        self.verbose = verbose
        self.client_queue = multiprocessing.Queue()
        self.http_port = http_port

    def print_server_info(self):
        import pprint
        pp = pprint.PrettyPrinter(indent=3)
        print("="*50)
        print("Server Information:\n")
        pp.pprint(self.__dict__)
        print("="*50)

    def prepare(self):
        for i in range(len(self.gpu_ids)):
            wrk = self.handler_cls(model_path=self.model_path,
                                   gpu_id=self.gpu_ids[i],
                                   mem_fraction=self.mem_fractions[i],
                                   client_queue=self.client_queue,
                                   batch_infer_size=self.batch_infer_size,
                                   batch_group_timeout=self.batch_group_timeout)
            wrk.daemon = True
            wrk.start()
            self.handlers.append(wrk)

        if self.http_port is not None:
            http_server = HTTPServer(host=self.host, port=self.port, http_port=self.http_port)
            http_server.daemon = True
            http_server.start()

        self.is_running = True
        if self.verbose:
            self.print_server_info()

    def serve(self):
        self.prepare()
        self.socket.listen()
        print("Service started")
        while self.is_running:
            try:
                client = self.socket.accept()
                if not client:
                    continue
                self.client_queue.put(client)
            except (SystemExit, KeyboardInterrupt):
                break
            except Exception as err:
                print(traceback.format_exc())


class TModelPoolServerV2(TModelPoolServerV1):
    def prepare(self):
        self.list_clients = []

        for i in range(len(self.gpu_ids)):
            client_queue = multiprocessing.Queue()
            wrk = self.handler_cls(model_path=self.model_path,
                                   gpu_id=self.gpu_ids[i],
                                   mem_fraction=self.mem_fractions[i],
                                   client_queue=client_queue,
                                   batch_infer_size=self.batch_infer_size,
                                   batch_group_timeout=self.batch_group_timeout)
            wrk.daemon = True
            wrk.start()
            self.handlers.append(wrk)
            self.list_clients.append(client_queue)

        if self.http_port is not None:
            http_server = HTTPServer(host=self.host, port=self.port, http_port=self.http_port)
            http_server.daemon = True
            http_server.start()

        self.is_running = True
        if self.verbose:
            self.print_server_info()

    def serve(self):
        self.prepare()
        self.socket.listen()
        print("Service started")
        while self.is_running:
            try:
                client = self.socket.accept()
                if not client:
                    continue
                rand_idx = random.randint(0, len(self.list_clients) - 1)
                self.list_clients[rand_idx].put(client)
            except (SystemExit, KeyboardInterrupt):
                break
            except Exception as err:
                print(traceback.format_exc())


class TModelPoolServer():
    def __init__(self, *args, **kwargs):
        self.__server = TModelPoolServerV2(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.__server, name)
