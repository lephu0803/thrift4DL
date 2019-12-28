# Copyright (c) 2019 congvm
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import json

def cvt_vision_result_proto(error_code, error_message, content):
    json_data = {
        "error_code": error_code,
        "error_message": error_message,
        "content": content
    }
    return json.dumps(json_data)