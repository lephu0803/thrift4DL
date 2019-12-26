import os
import sys
import json
from thrift4DL.client import Client
client = Client('10.40.34.15', 9093)
url = "https://f10.group.zp.zdn.vn/1666445185667442009/846cb7955c5fa501fc4e.jpg"

result = client.predict(url)
json.loads(result.response)