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

from .ttypes import TResult

class BaseHandler(multiprocessing.Process):
    def __init__(self, model_path, gpu_id, mem_fraction, args_queue, result_queue):
        multiprocessing.Process.__init__(self)
        print("Init Handler")
        self.result_queue = result_queue
        self.args_queue = args_queue
        self.gpu_id = gpu_id
        self.mem_fraction = mem_fraction
        self.model_path = model_path

    def get_env(self):
        raise NotImplementedError

    def get_model(self, model_path, gpu_id, mem_fraction):
        raise NotImplementedError

    def preprocessing(self, input):
        raise NotImplementedError

    def postprocessing(self, input):
        raise NotImplementedError

    def predict(self, model, input):
        raise NotImplementedError
        
    def run(self):
        model = self.get_model(self.model_path, self.gpu_id, self.mem_fraction)
        while True:
            args_dict = self.args_queue.get()
            try:
                args_request = args_dict['args_request']
                result = args_dict['result']
                args = self.preprocessing(args_request)
                pred_result = self.predict(model, args)
                pred_result = self.postprocessing(pred_result)
                result.success = TResult(error_code=0, response=str(pred_result))
                args_dict['result'] = result
                args_dict['msg_type'] = TMessageType.REPLY
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                print(tb)
                args_dict['msg_type'] = TMessageType.EXCEPTION
                args_dict['result'] = TApplicationException(TApplicationException.INTERNAL_ERROR, 'Internal error')
            self.result_queue.put(args_dict)


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
                                   result_queue=self.result_queue)
            wrk.daemon = True
            wrk.start()
            self.handlers.append(wrk)

    def serve(self):
        self.prepare()
        self.socket.listen()
        print("Service started")
        while True:
            try:
                client = self.socket.accept()
                if not client:
                    continue
                self.logger.info(client)
                self.client_queue.put(client)
            except (SystemExit, KeyboardInterrupt):
                break
            except Exception as err:
                tb = traceback.format_exc()
                self.logger.exception(tb)
