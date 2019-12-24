from face_detector.retinaface.core.utils import common
from face_detector.retinaface.core.face_detector import FaceDetector
from face_detector.retinaface.core.utils import constant as cons
import numpy as np
import cv2
from thrift4DL.server.TModelPoolServer import TModelPoolServerV2
from thrift4DL.server.base_handler import BatchingBaseHandlerV2, BaseHandlerV2
import json
import os
from time import time


class Model():
    def __init__(self, model_path, gpu_id, mem_fraction):
        print("Model initialized")
        self.detector = FaceDetector(model_path=cons.MODEL_PATH, gpu_id=gpu_id)

    def predict(self, input):
        bboxes, landmarks, scale_ratio = self.detector.detect(input)
        return [(bboxes, landmarks, scale_ratio)]

class Handler(BatchingBaseHandlerV2):
    def get_env(self, gpu_id, mem_fraction):
        os.environ['HTTPS_PROXY'] = '10.40.34.14:81'
        os.environ['HTTP_PROXY'] = '10.40.34.14:81'
        os.environ['https_proxy'] = '10.40.34.14:81'
        os.environ['http_proxy'] = '10.40.34.14:81'
        os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = "0"

    def get_model(self, model_path, gpu_id, mem_fraction):
        self.get_env(gpu_id=gpu_id, mem_fraction=mem_fraction)
        model = Model(model_path, gpu_id, mem_fraction)
        return model

    def preprocessing(self, input):
        input = json.loads(input)
        url = input['value']
        img_arr = common.load_image_from_url(url)
        img_arr = cv2.resize(img_arr, (300, 300))
        return img_arr

    def postprocessing(self, input):
        bboxes, landmarks, scale_ratio = input
        result_dict = {'bboxes': bboxes.tolist(),
                       'landmarks': landmarks.tolist(),
                       'scale_ratio': float(scale_ratio)
                       }
        input = json.dumps(result_dict)
        return input

    def predict(self, model, input):
        input = np.vstack(input)
        result = model.predict(input)
        return result


server = TModelPoolServerV2(host='10.40.34.15',
                            port=9093,
                            handler_cls=Handler,
                            model_path='/',
                            gpu_ids=[7, 7, 7],
                            mem_fractions=[0.3, 0.3, 0.3],
                            batch_infer_size=1)
print("Hello")
server.serve()
