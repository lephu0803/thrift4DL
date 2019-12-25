from thrift.TSerialization import serialize
from thrift.protocol.TJSONProtocol import TSimpleJSONProtocolFactory
import json


def thrift_to_json(thrift_object):
    return json.loads(serialize(thrift_object, protocol_factory=TSimpleJSONProtocolFactory()).decode("utf-8"))