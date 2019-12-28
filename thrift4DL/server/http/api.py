# Copyright (c) 2019 congvm
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from flask import Flask
from flask import render_template, request
import json
from thrift4DL.client import VisionClient
APP_NAME = 'RESTful-Thrift4DL'

def create_app(host, port):
    app = Flask(APP_NAME)

    @app.route('/v1/predict/', methods=['POST'])
    def predict():
        json_data = request.get_json()
        hex_image = json_data['image']
        client = VisionClient(host=host, port=port)
        pred_result = client.predict(hex_image)
        response = json.dumps(pred_result.response)
        return response 
        
    return app


