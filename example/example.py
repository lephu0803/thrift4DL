
import os
import sys
sys.path.append('..')
from thrift4DL.prototype.TModelPoolServer import TModelPoolServer

server = TModelPoolServer(host='localhost', port=8811)
server.serve()
