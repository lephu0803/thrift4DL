from .Thrift4DLService import Receiver, Deliver
import time
import multiprocessing
import logging
from zaailabcorelib.thrift.transport import TSocket
import traceback
from zaailabcorelib.thrift.transport.TTransport import TTransportException
from zaailabcorelib.thrift.protocol import TBinaryProtocol
import warnings

logger = logging.getLogger(__name__)


class Handler():
    def __init__(self):
        pass

    def predict(self, input):
        pass


class TModelPoolServer():
    ''' A server runs a pool of multiple models to serve single request
        Written by CongVM
    '''

    def __init__(self, host, port, batch_infer_size=1, batch_group_timeout=10, n_models=2, logger=None):
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
        self.host = host
        self.port = port
        self.socket = TSocket.TServerSocket(host=self.host, port=self.port)
        self.n_models = int(n_models)
        self.client_queue = multiprocessing.JoinableQueue()
        self.args_queue = multiprocessing.JoinableQueue()
        self.result_queue = multiprocessing.JoinableQueue()
        self.batch_infer_size = batch_infer_size
        self.batch_group_timeout = batch_group_timeout
        self.list_handlers = []
        self.list_models = []

    def prepare(self):
        receiver = Receiver(client_queue=self.client_queue,
                            args_queue=self.args_queue)
        receiver.daemon = True
        receiver.start()

        deliver = Deliver(result_queue=self.result_queue)
        deliver.daemon = True
        deliver.start()

    def serve(self):
        self.prepare()
        # first bind and listen to the port
        self.socket.listen()
        print("Service started")
        while True:
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
