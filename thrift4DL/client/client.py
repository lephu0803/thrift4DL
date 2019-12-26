# Copyright (c) 2019 congvm
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import traceback
import json
from .thrift4DL.ttypes import *
from .thrift4DL import Thrift4DLService
from thrift.protocol.TJSONProtocol import TJSONProtocol
from thrift.protocol.TBinaryProtocol import TBinaryProtocol
from thrift.transport import TTransport, TSocket, TSSLSocket, THttpClient
import sys
if sys.version_info[0] > 2:
    from urllib.parse import urlparse
else:
    from urlparse import urlparse


class BaseClient():
    def predict(self, x):
        return None

    def ping(self):
        return None


class Client(BaseClient):
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.socket = TSocket.TSocket(host, port)
        self.transport = TTransport.TFramedTransport(self.socket)
        self.protocol = TBinaryProtocol(self.transport)
        self.client = Thrift4DLService.Client(self.protocol)

    def predict(self, x):
        self.transport.open()
        ret = None
        try:
            request_dict = {"value": x}
            request_json = json.dumps(request_dict)
            ret = self.client.predict(request_json,)
        except Exception as e:
            print(traceback.format_exc())
        self.transport.close()
        return ret

    def ping(self):
        self.transport.open()
        ret = None
        try:
            ret = self.client.ping()
        except Exception as e:
            print(traceback.format_exc())
        self.transport.close()
        return ret


class VisionClient(Client):

    def predict(self, image_binary):
        self.transport.open()
        ret = None
        try:
            ret = self.client.predict(image_binary)
        except Exception as e:
            print(traceback.format_exc())
        self.transport.close()
        return ret
