from thrift4DL.server import TModelPoolServer
from thrift4DL.server.base_handler import BatchingBaseHandlerV2
from model import MnistModel
import json
import os
import numpy as np
from thrift4DL.helpers import decode_image


class ServerHandler(BatchingBaseHandlerV2):
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
        input = decode_image(input)
        input = model.preprocessing(input)
        return input

    def predict(self, model, input):
        input = np.vstack(input)
        return model.predict(input)

    def postprocessing(self, model, input):
        pred_num = np.argmax(input)
        pred_score = np.max(input)
        return json.dumps({"pred_num": int(pred_num), "pred_score": float(pred_score)})


server = TModelPoolServer(host='localhost', port='9090',
                          handler_cls=ServerHandler,
                          model_path='mnist.pb', gpu_ids=[-1]*2,
                          mem_fractions=[0.1]*2,
                          batch_infer_size=2)
server.serve()


from multiprocessing import connection
import multiprocessing

multiprocessing.Pipe()