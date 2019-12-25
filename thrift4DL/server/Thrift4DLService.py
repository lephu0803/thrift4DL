from thrift.Thrift import TType, TMessageType, TException, TApplicationException
from thrift.protocol.TProtocol import TProtocolException
from thrift.TRecursive import fix_spec

import sys
import logging
from .ttypes import *
from thrift.Thrift import TProcessor
from .Thrift4DLServiceBase import predict_result, predict_args, ping_args, ping_result
import traceback
from multiprocessing import Process
from threading import Thread
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from queue import Empty


class ReceiverV1(Process):
    def __init__(self, client_queue, args_queue=None):
        Process.__init__(self)
        self._client_queue = client_queue
        self._args_queue = args_queue
        self._processMap = {}
        self._processMap["predict"] = ReceiverV1.process_predict
        self._iptranfac = TTransport.TFramedTransportFactory()
        self._optranfac = TTransport.TFramedTransportFactory()
        self._iprotfac = TBinaryProtocol.TBinaryProtocolFactory()
        self._oprotfac = TBinaryProtocol.TBinaryProtocolFactory()

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
        return args.image_binary

    def process_predict(self, seqid, iprot, oprot, itrans, otrans):
        image_binary = self.parse_args(iprot)
        result = predict_result()
        args_dict = {
            'iprot': iprot,
            'oprot': oprot,
            'itrans': itrans,
            'otrans': otrans,
            'seqid': seqid,
            'image_binary': image_binary,
            'result': result,
            'msg_type': None,
        }
        self._args_queue.put(args_dict)


class DeliverV1(Process):
    def __init__(self, result_queue):
        Process.__init__(self)
        self._result_queue = result_queue
        self._processMap = {}

    def run(self):
        print("Start Deliver")
        while True:
            try:
                result_dict = self._result_queue.get()  # block=True, timeout=0.5)
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
            except Empty:
                continue

    def parse_result(self, result, oprot, msg_type, seqid):
        oprot.writeMessageBegin("predict", msg_type, seqid)
        result.write(oprot)
        oprot.writeMessageEnd()
        oprot.trans.flush()


class ReceiverV2(ReceiverV1):
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
            'image_binary': None,
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
        return args.image_binary

    def process_predict(self, connection_info):
        image_binary = self.parse_args(connection_info['iprot'])
        result = predict_result()
        connection_info['image_binary'] = image_binary
        connection_info['result'] = result
        return connection_info

    def process_ping(self, connection_info):
        seqid = connection_info['seqid']
        iprot = connection_info['iprot']
        oprot = connection_info['oprot']
        itrans = connection_info['itrans']
        otrans = connection_info['otrans']
        args = ping_args()
        args.read(iprot)
        iprot.readMessageEnd()
        result = ping_result()
        try:
            msg_type = TMessageType.REPLY
        except (TTransport.TTransportException, KeyboardInterrupt, SystemExit):
            raise
        except Exception as ex:
            msg_type = TMessageType.EXCEPTION
            logging.exception(ex)
            result = TApplicationException(
                TApplicationException.INTERNAL_ERROR, 'Internal error')
        oprot.writeMessageBegin("ping", msg_type, seqid)
        result.write(oprot)
        oprot.writeMessageEnd()
        oprot.trans.flush()
        itrans.close()
        otrans.close()

    def process(self, client):
        connection_info = None
        try:
            connection_info = self.get_connection(client)
            connection_info = self.validate(connection_info)
        except Exception as e:
            print(traceback.format_exc())
        return connection_info


class DeliverV2(DeliverV1):
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
