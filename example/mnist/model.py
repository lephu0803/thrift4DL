# Copyright (c) 2019 congvm
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import numpy as np
import cv2

class MnistModel():
    def __init__(self, tf, model_path, config=None):        
        self.graph = tf.Graph()
        with self.graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.input_tensor = self.graph.get_tensor_by_name(
                'conv2d_1_input:0')
            self.output_tensor = self.graph.get_tensor_by_name(
                'dense_2/Softmax:0')

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.gpu_options.per_process_gpu_memory_fraction = self.gpu_memory_fraction
        config.log_device_placement = False
        self.sess = tf.Session(graph=self.graph, config=config)

    def preprocessing(self, img_arr):
        img_arr = cv2.resize(img_arr, (28, 28))
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
        img_arr = img_arr.reshape((28, 28, 1)).astype(np.float32)
        img_arr = np.expand_dims(img_arr, 0)
        img_arr /= 255.
        return img_arr

    def predict(self, img_expanded):
        with self.graph.as_default():
            pred_classes = self.sess.run(
                self.output_tensor,
                feed_dict={self.input_tensor: img_expanded})
        return pred_classes

    def postprocessing(self, result):
        return result
