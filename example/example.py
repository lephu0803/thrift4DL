
import os
import sys
sys.path.append('..')
from thrift4DL.server.TModelPoolServer import BaseHandler, TModelPoolServer


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


class Handler(BaseHandler):
    def get_env(self):
        import tensorflow as tf
        return tf

    def get_model(self, model_path, gpu_id, mem_fraction):
        tf = self.get_env()
        model = Model(tf, model_path, gpu_id, mem_fraction)
        return model

    def preprocessing(self, input):
        value = input['value']
        return value

    def postprocessing(self, input):
        return input

    def predict(self, model, input):
        result = model.predict(input)
        return result


server = TModelPoolServer(host='localhost', port=8811,
                          handler_cls=Handler,
                          model_path='/',
                          gpu_ids=[1, 1, 1],
                          mem_fractions=[0.3, 0.3, 0.3])
server.serve()
