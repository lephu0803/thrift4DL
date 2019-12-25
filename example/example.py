
import os
import sys
sys.path.append('..')
from thrift4DL.server.TModelPoolServer import TModelPoolServer
from thrift4DL.server.base_handler import BatchingBaseHandlerV2
import json
import numpy as np

class Model():
    def __init__(self, tf, model_path, gpu_id, mem_fraction):
        print("Model initialized")
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            self.x = tf.placeholder(dtype=tf.float32)
            self.y = self.x + 10

    def predict(self, input):
        with self.graph.as_default():
            result = self.sess.run(self.y, feed_dict={self.x: input})
        return result


class Handler(BatchingBaseHandlerV2):
    def get_env(self, gpu_id, mem_fraction):
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        import tensorflow as tf
        return tf

    def get_model(self, model_path, gpu_id, mem_fraction):
        tf = self.get_env(gpu_id=-1, mem_fraction=0)
        model = Model(tf, model_path, gpu_id, mem_fraction)
        return model

    def preprocessing(self, input):
        input = json.loads(input)
        value = input['value']
        return value

    def postprocessing(self, input):
        result_dict = {"value": float(input)}
        input = json.dumps(result_dict)
        return input

    def predict(self, model, input):
        result = model.predict(input)
        print(result)
        return result

server = TModelPoolServer(host='localhost', port=9090,
                          handler_cls=Handler,
                          model_path='/',
                          gpu_ids=[-1]*3,
                          mem_fractions=[0.3]*3, 
                          batch_infer_size=5, 
                          batch_group_timeout=0.5)
server.serve()
