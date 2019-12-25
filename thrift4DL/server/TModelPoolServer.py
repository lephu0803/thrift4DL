from multiprocessing import Manager
import time
import multiprocessing
import logging
from thrift.transport import TSocket
import traceback
from thrift.transport.TTransport import TTransportException
from thrift.protocol import TBinaryProtocol
import warnings
import numpy as np
from thrift.Thrift import TType, TMessageType, TApplicationException
logger = logging.getLogger(__name__)


class TModelPoolServerBase():
    def __init__(self):
        pass

    def serve(self):
        raise NotImplementedError


class TModelPoolServerV2():
    def __init__(self, host, port, handler_cls, model_path, gpu_ids, mem_fractions, batch_infer_size=1, batch_group_timeout=10, verbose=True, logger=None):
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
        # self.client_queue = Manager().Queue()
        self.batch_infer_size = batch_infer_size
        self.batch_group_timeout = batch_group_timeout
        self.handlers = []
        self.is_running = False
        if verbose:
            self.print_server_info()

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


class TModelPoolServerV3():
    def __init__(self, host, port, handler_cls, model_path, gpu_ids, mem_fractions, batch_infer_size=1, batch_group_timeout=10, verbose=True, logger=None):
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
        # self.client_queue = multiprocessing.JoinableQueue()
        self.manager = Manager()
        # self.client_queues = self.manager.dict()
        self.client_queues = []
        self.batch_infer_size = batch_infer_size
        self.batch_group_timeout = batch_group_timeout
        self.handlers = []
        self.is_running = False
        if verbose:
            self.print_server_info()

    def print_server_info(self):
        import pprint
        pp = pprint.PrettyPrinter(indent=3)
        print("="*50)
        print("Server Information:\n")
        pp.pprint(self.__dict__)
        print("="*50)

    def prepare(self):
        for i in range(len(self.gpu_ids)):
            # client_queue = Manager().Queue()
            client_queue = multiprocessing.JoinableQueue()
            self.client_queues.append(client_queue)
            wrk = self.handler_cls(model_path=self.model_path,
                                   gpu_id=self.gpu_ids[i],
                                   mem_fraction=self.mem_fractions[i],
                                   client_queue=client_queue,
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
                client_queue_idx = np.random.randint(len(self.gpu_ids))
                self.client_queues[client_queue_idx].put(client)
            except (SystemExit, KeyboardInterrupt):
                break
            except Exception as err:
                tb = traceback.format_exc()
                self.logger.exception(tb)


class Ventilator(multiprocessing.Process):
    def __init__(self, client_queues, connection_queue):
        multiprocessing.Process.__init__(self)
        self.client_queues = client_queues
        self.connection_queue = connection_queue
        self.n_process = len(self.client_queues.keys())

    def run(self):
        while True:
            client = self.connection_queue.get()
            client_queue_idx = np.random.randint(self.n_process)
            self.client_queues[client_queue_idx].put(client)


class TModelPoolServerV4():
    def __init__(self, host, port, handler_cls, model_path, gpu_ids, mem_fractions, batch_infer_size=1, batch_group_timeout=10, verbose=True, logger=None):
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
        # self.client_queue = multiprocessing.JoinableQueue()
        self.manager = Manager()
        self.client_queues = self.manager.dict()
        self.connection_queue = multiprocessing.Queue()
        self.batch_infer_size = batch_infer_size
        self.batch_group_timeout = batch_group_timeout
        self.handlers = []
        self.is_running = False
        if verbose:
            self.print_server_info()

    def print_server_info(self):
        import pprint
        pp = pprint.PrettyPrinter(indent=3)
        print("="*50)
        print("Server Information:\n")
        pp.pprint(self.__dict__)
        print("="*50)

    def get_mpqueue(self):
        q = Manager().Queue()
        # q = multiprocessing.JoinableQueue()
        return q

    def prepare(self):
        for i in range(len(self.gpu_ids)):
            client_queue = self.get_mpqueue()
            
            self.client_queues[i] = client_queue
            wrk = self.handler_cls(model_path=self.model_path,
                                   gpu_id=self.gpu_ids[i],
                                   mem_fraction=self.mem_fractions[i],
                                   client_queue=client_queue,
                                   batch_infer_size=self.batch_infer_size,
                                   batch_group_timeout=self.batch_group_timeout)

            wrk.daemon = True
            wrk.start()
            self.handlers.append(wrk)

        ven_wrk = Ventilator(self.client_queues, self.connection_queue)
        ven_wrk.daemon = True
        ven_wrk.start()
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
                self.connection_queue.put(client)
            except (SystemExit, KeyboardInterrupt):
                break
            except Exception as err:
                tb = traceback.format_exc()
                self.logger.exception(tb)

class TModelPoolServer():
    def __init__(self, *args, **kwargs):
        self.__server = TModelPoolServerV3(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.__server, name)
