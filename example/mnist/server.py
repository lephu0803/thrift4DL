from thrift4DL.server import TModelPoolServer
from thrift4DL.server import BatchingHandler
from model import MnistModel
import json
import os
import numpy as np
from thrift4DL.helpers import decode_image
import time

class ServerHandler(BatchingHandler):
    def get_env(self, gpu_id, mem_fraction):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        import tensorflow as tf
        return tf

    def get_model(self, model_path, env_params):
        tf = env_params
        model = MnistModel(tf, model_path)
        return model

    def preprocessing(self, model, input):
        """Preprocess for an input"""
        input = decode_image(input)
        input = model.preprocessing(input)
        assert input.shape == (1, 28, 28, 1), ValueError("Wrong input shape")
        return input

    def predict(self, model, input):
        """Given a batch of input to predict"""
        input = np.vstack(input)    
        result = model.predict(input)
        return result

    def postprocessing(self, model, input):
        """"Process single result after prediction"""
        pred_num = np.argmax(input)
        pred_score = np.max(input)
        return json.dumps({"pred_num": int(pred_num), "pred_score": float(pred_score)})


NUM_MODELS = 1
server = TModelPoolServer(host='localhost', port='9090',
                          handler_cls=ServerHandler,
                          model_path='mnist.pb', gpu_ids=[6]*NUM_MODELS,
                          mem_fractions=[0.1]*NUM_MODELS,
                          batch_infer_size=100, batch_group_timeout=1)
server.serve()