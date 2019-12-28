# Copyright (c) 2019 congvm
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import requests
import numpy as np
import json

def test_api(image_hex, host, port):
    url = 'http://0.0.0.0:8181/v1/predict/'
    js = {"image": image_hex}
    response = requests.post(url, json=js)
    return response