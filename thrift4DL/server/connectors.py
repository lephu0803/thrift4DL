# Copyright (c) 2019 congvm
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


import sys
import logging
import traceback
from .ttypes import *
from .thriftbase import predict_result, predict_args, ping_args, ping_result

from multiprocessing import Process
from thrift.Thrift import TProcessor
from thrift.transport import TTransport
from thrift.Thrift import TType, TMessageType, TException, TApplicationException
from thrift.protocol import TBinaryProtocol
from thrift.protocol.TProtocol import TProtocolException
from thrift.TRecursive import fix_spec
from queue import Empty

INVALID_NAME = 'invalid'


class Validator(object):
    """
    This class is to check if connection is valid or not, then parse messages.
    """

    def __init__(self):
        self.func_names = {}
        self.func_names['predict'] = self.process_predict
        self.func_names['ping'] = self.process_ping

    def _parse_ping_args(self, iprot):
        args = ping_args()
        args.read(iprot)
        iprot.readMessageEnd()
        return args

    def _parse_predict_args(self, iprot):
        args = predict_args()
        args.read(iprot)
        iprot.readMessageEnd()
        return args.image_binary

    def _parse_invalid_args(self, iprot):
        iprot.skip(TType.STRUCT)
        iprot.readMessageEnd()
        return

    def process_ping(self, connection_info):
        self._parse_ping_args(connection_info['iprot'])
        connection_info['result'] = ping_result()
        return connection_info

    def process_invalid(self, connection_info):
        name = connection_info['name']
        self._parse_invalid_args(connection_info['iprot'])
        connection_info['result'] = TApplicationException(TApplicationException.UNKNOWN_METHOD,
                                                          'Unknown function %s' % (name))
        connection_info['msg_type'] = TMessageType.EXCEPTION
        return connection_info

    def process_predict(self, connection_info):
        image_binary = self._parse_predict_args(connection_info['iprot'])
        connection_info['image_binary'] = image_binary
        connection_info['result'] = predict_result()
        return connection_info

    def process(self, connection_info):
        assert len(self.func_names) > 0, ValueError(
            '`func_names` in `Validator` must be not empty')
        iprot = connection_info['iprot']
        (name, type, seqid) = iprot.readMessageBegin()
        connection_info['name'] = name
        connection_info['seqid'] = seqid
        connection_info['msg_type'] = TMessageType.REPLY
        if name in self.func_names:
            connection_info = self.func_names[name](connection_info)
        else:
            connection_info = self.process_invalid(connection_info)
        return connection_info


class Receiver():
    def __init__(self):
        self._iptranfac = TTransport.TFramedTransportFactory()
        self._optranfac = TTransport.TFramedTransportFactory()
        self._iprotfac = TBinaryProtocol.TBinaryProtocolFactory()
        self._oprotfac = TBinaryProtocol.TBinaryProtocolFactory()

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
            'name': None,
        }
        return connection_info

    def process(self, client):
        try:
            connection_info = self.get_connection(client)
            return connection_info
        except Exception as e:
            raise TException(traceback.format_exc())


class Deliver():
    def process(self, connection_info):
        oprot = connection_info['oprot']
        itrans = connection_info['itrans']
        otrans = connection_info['otrans']
        seqid = connection_info['seqid']
        result = connection_info['result']
        msg_type = connection_info['msg_type']
        name = connection_info['name']
        try:
            oprot.writeMessageBegin(name, msg_type, seqid)
            result.write(oprot)
            oprot.writeMessageEnd()
            oprot.trans.flush()
        except TTransport.TTransportException:
            print(traceback.format_exc())
        except Exception:
            print(traceback.format_exc())
        itrans.close()
        otrans.close()
