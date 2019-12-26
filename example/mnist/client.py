# Copyright (c) 2019 congvm
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from thrift4DL.client import VisionClient
import numpy as np
from zaailabcorelib.ztools import encode_image, load_image

# Initialize a client
client = VisionClient(host='127.0.0.1', port='9090')

# Load and encode image to hex
img_arr = load_image('../example/mnist/num5.png')
img_hex = encode_image(img_arr)

# Request to server
result = client.predict(img_hex)
print(result)

# Ping to check server
client.ping()



