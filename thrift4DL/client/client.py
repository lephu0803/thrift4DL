import sys
if sys.version_info[0] > 2:
    from urllib.parse import urlparse
else:
    from urlparse import urlparse
from thrift.transport import TTransport, TSocket, TSSLSocket, THttpClient
from thrift.protocol.TBinaryProtocol import TBinaryProtocol
from thrift.protocol.TJSONProtocol import TJSONProtocol

from .thrift4DL import Thrift4DLService
from .thrift4DL.ttypes import *
import json
import traceback

class Client(object):
    def __init__(self, host, port):
        self.host = 'localhost'
        self.port = 9090
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

    def ping(self, x):
        self.transport.open()
        ret = None
        try:
            ret = self.client.ping()
        except Exception as e:
            print(traceback.format_exc())
        self.transport.close()
        return ret

class ClientDev(Client):
    def __init__(self, mode='json', *args, **kwargs):
        Client.__init__(self, *args, **kwargs)
        if mode == 'json':
            self.protocol = TJSONProtocol(self.transport)
        self.client = Thrift4DLService.Client(self.protocol)

    def predict(self, x):
        self.transport.open()
        ret = None
        try: 
            ret = self.client.predict(x)
        except Exception as e:
            print(traceback.format_exc())
        self.transport.close()
        return ret

class VisionClient(ClientDev):
    def predict(self, image_binary):
        self.transport.open()
        ret = None
        try: 
            ret = self.client.predict(image_binary)
        except Exception as e:
            print(traceback.format_exc())
        self.transport.close()
        return ret
