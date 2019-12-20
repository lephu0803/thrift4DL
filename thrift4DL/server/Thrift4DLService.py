from thrift.Thrift import TType, TMessageType, TException, TApplicationException
from thrift.protocol.TProtocol import TProtocolException
from thrift.TRecursive import fix_spec

import sys
import logging
from .ttypes import *
from thrift.Thrift import TProcessor
from thrift.transport import TTransport

from .Thrift4DLServiceBase import ProcessorBase, predict_result, predict_args, ping_args, ping_result
import traceback
from multiprocessing import Process
from threading import Thread
from thrift.transport import TTransport
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol


class Validator():
    def __init__(self):
        self.list_validators = []

    def set_validation(self, validation_func):
        self.list_validators.append(validation_func)

    def validate(self, inp):
        for func in self.list_validators:
            try:
                is_valid = func(inp)
                if not is_valid:
                    return False
            except Exception as e:
                tb = traceback.format_exc()
                raise ValueError(tb)
        return True


class Receiver(ProcessorBase, Thread):
    def __init__(self, client_queue, args_queue):
        Thread.__init__(self)
        self._client_queue = client_queue
        self._args_queue = args_queue
        self._validator = Validator()

        self._iptranfac = TTransport.TFramedTransportFactory()
        self._optranfac = TTransport.TFramedTransportFactory()
        self._iprotfac = TBinaryProtocol.TBinaryProtocolFactory()
        self._oprotfac = TBinaryProtocol.TBinaryProtocolFactory()

        self._processMap = {}
        self._processMap["predict"] = Receiver.process_predict
        self._processMap["ping"] = Receiver.process_ping

    def run(self):
        print("Start Receiver")
        while True:
            client = self._client_queue.get()
            self._client_queue.task_done()
            # get connection
            itrans = self._iptranfac.getTransport(client)
            otrans = self._optranfac.getTransport(client)
            iprot = self._iprotfac.getProtocol(itrans)
            oprot = self._oprotfac.getProtocol(otrans)
            # process
            self.process(iprot, oprot, itrans, otrans)

    def process(self, iprot, oprot, itrans, otrans):
        (name, type, seqid) = iprot.readMessageBegin()
        if name not in self._processMap:
            iprot.skip(TType.STRUCT)
            iprot.readMessageEnd()
            x = TApplicationException(
                TApplicationException.UNKNOWN_METHOD, 'Unknown function %s' % (name))
            oprot.writeMessageBegin(name, TMessageType.EXCEPTION, seqid)
            x.write(oprot)
            oprot.writeMessageEnd()
            oprot.trans.flush()
            itrans.close()
            otrans.close()
            return
        else:
            self._processMap[name](self, seqid, iprot, oprot, itrans, otrans)
        return True

    def parse_args(self, iprot):
        args = predict_args()
        args.read(iprot)
        iprot.readMessageEnd()
        return args.request

    def process_predict(self, seqid, iprot, oprot, itrans, otrans):
        args_request = self.parse_args(iprot)
        result = predict_result()
        args_dict = {
            'iprot': iprot,
            'oprot': oprot,
            'itrans': itrans,
            'otrans': otrans,
            'seqid': seqid,
            'args_request': args_request,
            'result': result,
            'msg_type': None,
        }
        self._args_queue.put(args_dict)


class Deliver(ProcessorBase, Thread):
    def __init__(self, result_queue):
        Thread.__init__(self)
        self._result_queue = result_queue
        self._validator = Validator()
        self._processMap = {}
        self._processMap["predict"] = Receiver.process_predict
        self._processMap["ping"] = Receiver.process_ping

    def run(self):
        print("Start Deliver")
        while True:
            result_dict = self._result_queue.get()
            self._result_queue.task_done()
            oprot = result_dict['oprot']
            itrans = result_dict['itrans']
            otrans = result_dict['otrans']
            seqid = result_dict['seqid']
            result = result_dict['result']
            msg_type = result_dict['msg_type']
            self.parse_result(result=result,
                              oprot=oprot, msg_type=msg_type, seqid=seqid)
            itrans.close()
            otrans.close()

    def parse_result(self, result, oprot, msg_type, seqid):
        oprot.writeMessageBegin("predict", msg_type, seqid)
        result.write(oprot)
        oprot.writeMessageEnd()
        oprot.trans.flush()
