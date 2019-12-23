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

class Receiver(ProcessorBase, Process):
    def __init__(self, client_queue, args_queue=None):
        Process.__init__(self)
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

    def process(self, client):
        # get connection
        itrans = self._iptranfac.getTransport(client)
        otrans = self._optranfac.getTransport(client)
        iprot = self._iprotfac.getProtocol(itrans)
        oprot = self._oprotfac.getProtocol(otrans)
        return iprot, oprot, itrans, otrans

    def run(self):
        print("Start Receiver")
        while True:
            try:
                client = self._client_queue.get()
                self._client_queue.task_done()
                iprot, oprot, itrans, otrans = self.process(client)
                self.validate(iprot, oprot, itrans, otrans)
            except Exception as e:
                print(traceback.format_exc())

    def validate(self, iprot, oprot, itrans, otrans):
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


class Deliver(ProcessorBase, Process):
    def __init__(self, result_queue):
        Process.__init__(self)
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
                              oprot=oprot,
                              msg_type=msg_type,
                              seqid=seqid)
            itrans.close()
            otrans.close()

    def parse_result(self, result, oprot, msg_type, seqid):
        oprot.writeMessageBegin("predict", msg_type, seqid)
        result.write(oprot)
        oprot.writeMessageEnd()
        oprot.trans.flush()


class ReceiverV2(Receiver):
    def __init__(self, client_queue):
        self._client_queue = client_queue
        self._iptranfac = TTransport.TFramedTransportFactory()
        self._optranfac = TTransport.TFramedTransportFactory()
        self._iprotfac = TBinaryProtocol.TBinaryProtocolFactory()
        self._oprotfac = TBinaryProtocol.TBinaryProtocolFactory()

        self._processMap = {}
        self._processMap["predict"] = ReceiverV2.process_predict
        self._processMap["ping"] = ReceiverV2.process_ping

    def get_connection(self, client):
        # get connection
        itrans = self._iptranfac.getTransport(client)
        otrans = self._optranfac.getTransport(client)
        iprot = self._iprotfac.getProtocol(itrans)
        oprot = self._oprotfac.getProtocol(otrans)
        connection_info = {
            'iprot': iprot,
            'oprot': oprot,
            'itrans': itrans,
            'otrans': otrans,
            'seqid': None,
            'args_request': None,
            'result': None,
            'msg_type': None,
        }
        return connection_info

    def validate(self, connection_info):
        iprot = connection_info['iprot']
        oprot = connection_info['oprot']
        itrans = connection_info['itrans']
        otrans = connection_info['otrans']
        (name, type, seqid) = iprot.readMessageBegin()
        connection_info['name'] = name
        connection_info['seqid'] = seqid
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
            return None
        else:
            return self._processMap[name](self, connection_info)

    def parse_args(self, iprot):
        args = predict_args()
        args.read(iprot)
        iprot.readMessageEnd()
        return args.request

    def process_ping(self, connection_info):
        pass

    def process_predict(self, connection_info):
        args_request = self.parse_args(connection_info['iprot'])
        result = predict_result()
        connection_info['args_request'] = args_request
        connection_info['result'] = result
        return connection_info

    def process(self, client):
        connection_info = None
        try:
            connection_info = self.get_connection(client)
            connection_info = self.validate(connection_info)
        except Exception as e:
            print(traceback.format_exc())
        return connection_info


class DeliverV2(Deliver):
    def __init__(self):
        pass

    def process(self, connection_info):
        oprot = connection_info['oprot']
        itrans = connection_info['itrans']
        otrans = connection_info['otrans']
        seqid = connection_info['seqid']
        result = connection_info['result']
        msg_type = connection_info['msg_type']
        self.parse_result(result=result,
                          oprot=oprot,
                          msg_type=msg_type,
                          seqid=seqid)
        itrans.close()
        otrans.close()

    def parse_result(self, result, oprot, msg_type, seqid):
        oprot.writeMessageBegin("predict", msg_type, seqid)
        result.write(oprot)
        oprot.writeMessageEnd()
        oprot.trans.flush()
