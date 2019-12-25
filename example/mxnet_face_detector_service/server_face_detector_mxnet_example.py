from face_detector.retinaface.core.utils import common
from face_detector.retinaface.core.face_detector import FaceDetector
from face_detector.retinaface.core.utils import constant as cons
import numpy as np
import cv2
from thrift4DL.server.TModelPoolServer import TModelPoolServerV2, TModelPoolServer
from thrift4DL.server.base_handler import BatchingBaseHandlerV2, BaseHandlerV2
from thrift4DL.helpers import decode_image
import json
import os
import time

class Handler(BatchingBaseHandlerV2):
    def get_env(self, gpu_id, mem_fraction):
        os.environ['HTTPS_PROXY'] = '10.40.34.14:81'
        os.environ['HTTP_PROXY'] = '10.40.34.14:81'
        os.environ['https_proxy'] = '10.40.34.14:81'
        os.environ['http_proxy'] = '10.40.34.14:81'
        os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = "0"

    def get_model(self, model_path, gpu_id, mem_fraction):
        self.get_env(gpu_id=gpu_id, mem_fraction=mem_fraction)
        self.detector_model = FaceDetector(model_path=cons.MODEL_PATH, gpu_id=gpu_id)
        return self.detector_model

    def preprocessing(self, input):
        img_arr = decode_image(input)
        img_arr, scale_ratio = self.detector_model.preprocessing(img_arr)
        return (img_arr, scale_ratio)

    def predict(self, input):
        input_arr = np.vstack([inp[0] for inp in input])
        scale_ratios = [inp[1] for inp in input]
        result = self.detector_model.forward(input_arr)
        return result

    def postprocessing(self, input):
        scale_ratio = 1
        bboxes, landmarks = self.detector_model.postprocessing(input)
        result_dict = {'bboxes': bboxes.tolist(),
                       'landmarks': landmarks.tolist(),
                       }
        input = json.dumps(result_dict)
        return input


N = 1
server = TModelPoolServer(host='10.40.34.15',
                            port=9093,
                            handler_cls=Handler,
                            model_path='/',
                            gpu_ids=[6]*N,
                            mem_fractions=[0.3]*N,
                            batch_infer_size=1)
print("Hello")
server.serve()
