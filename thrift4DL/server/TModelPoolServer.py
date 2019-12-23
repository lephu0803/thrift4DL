from .ttypes import TResult
from .Thrift4DLService import Receiver, Deliver
import time
import multiprocessing
import logging
from zaailabcorelib.thrift.transport import TSocket
import traceback
from zaailabcorelib.thrift.transport.TTransport import TTransportException
from zaailabcorelib.thrift.protocol import TBinaryProtocol
import warnings
from thrift.Thrift import TType, TMessageType, TApplicationException
logger = logging.getLogger(__name__)


class TModelPoolServer():
    ''' A server runs a pool of multiple models to serve single request
        Written by CongVM
    '''

    def __init__(self, host, port, handler_cls, model_path, gpu_ids, mem_fractions, batch_infer_size=1, batch_group_timeout=10, logger=None):
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
        self.handler_cls = handler_cls
        self.model_path = model_path
        self.gpu_ids = gpu_ids
        self.mem_fractions = mem_fractions
        self.host = host
        self.port = port
        self.socket = TSocket.TServerSocket(host=self.host, port=self.port)
        self.client_queue = multiprocessing.JoinableQueue()
        self.args_queue = multiprocessing.JoinableQueue()
        self.result_queue = multiprocessing.JoinableQueue()
        self.batch_infer_size = batch_infer_size
        self.batch_group_timeout = batch_group_timeout
        self.handlers = []
        self.is_running = False

    def prepare(self):
        receiver = Receiver(client_queue=self.client_queue,
                            args_queue=self.args_queue)
        receiver.daemon = True
        receiver.start()

        deliver = Deliver(result_queue=self.result_queue)
        deliver.daemon = True
        deliver.start()

        for i in range(len(self.gpu_ids)):
            wrk = self.handler_cls(model_path=self.model_path,
                                   gpu_id=self.gpu_ids[i],
                                   mem_fraction=self.mem_fractions[i],
                                   args_queue=self.args_queue,
                                   result_queue=self.result_queue,
                                   batch_infer_size=self.batch_infer_size,
                                   batch_group_timeout=self.batch_group_timeout)
            wrk.daemon = True
            wrk.start()
            self.handlers.append(wrk)
        self.is_running = True

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
                tb = traceback.format_exc()
                self.logger.exception(tb)


class TModelPoolServerV2(TModelPoolServer):
    def __init__(self, host, port, handler_cls, model_path, gpu_ids, mem_fractions, batch_infer_size=1, batch_group_timeout=10, logger=None):
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
        self.handler_cls = handler_cls
        self.model_path = model_path
        self.gpu_ids = gpu_ids
        self.mem_fractions = mem_fractions
        self.host = host
        self.port = port
        self.socket = TSocket.TServerSocket(host=self.host, port=self.port)
        self.client_queue = multiprocessing.JoinableQueue()
        self.batch_infer_size = batch_infer_size
        self.batch_group_timeout = batch_group_timeout
        self.handlers = []
        self.is_running = False
        
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
        self.is_running = True
