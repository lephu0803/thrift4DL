
import sys
import logging
import traceback
from .ttypes import *
from .Thrift4DLServiceBase import predict_result, predict_args, ping_args, ping_result

from multiprocessing import Process
from thrift.Thrift import TProcessor
from thrift.transport import TTransport
from thrift.Thrift import TType, TMessageType, TException, TApplicationException
from thrift.protocol import TBinaryProtocol
from thrift.protocol.TProtocol import TProtocolException
from thrift.TRecursive import fix_spec

from queue import Empty


class Receiver():
    def __init__(self):
        self._iptranfac = TTransport.TFramedTransportFactory()
        self._optranfac = TTransport.TFramedTransportFactory()
        self._iprotfac = TBinaryProtocol.TBinaryProtocolFactory()
        self._oprotfac = TBinaryProtocol.TBinaryProtocolFactory()

        self._processMap = {}
        self._processMap["predict"] = Receiver.process_predict

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
            return self.process_predict(connection_info)

    def parse_args(self, iprot):
        args = predict_args()
        args.read(iprot)
        iprot.readMessageEnd()
        return args.image_binary

    def process(self, client):
        try:
            connection_info = self.get_connection(client)
            connection_info = self.validate(connection_info)
            return connection_info
        except Exception as e:
            print(traceback.format_exc())

    def process_predict(self, connection_info):
        image_binary = self.parse_args(connection_info['iprot'])
        result = predict_result()
        connection_info['image_binary'] = image_binary
        connection_info['result'] = result
        return connection_info


class Deliver():
    def process(self, connection_info):
        try:
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
        except Exception as e:
            print(traceback.format_exc())

    def parse_result(self, result, oprot, msg_type, seqid):
        try:
            oprot.writeMessageBegin("predict", msg_type, seqid)
            result.write(oprot)
            oprot.writeMessageEnd()
            oprot.trans.flush()
        except Exception as e:
            print(traceback.format_exc())
