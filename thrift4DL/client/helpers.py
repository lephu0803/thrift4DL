# Copyright (c) 2019 congvm
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from thrift.TSerialization import serialize
from thrift.protocol.TJSONProtocol import TSimpleJSONProtocolFactory
import json


def thrift_to_json(thrift_object):
    return json.loads(serialize(thrift_object, protocol_factory=TSimpleJSONProtocolFactory()).decode("utf-8"))