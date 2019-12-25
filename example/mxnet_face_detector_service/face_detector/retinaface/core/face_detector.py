from .main.detector_method.retinaface_detector import RetinaFaceDetector
from .preprocessing.face_pyramid_padding import FacePyramidPadding
from .utils import common
import os
from .utils import constant as cons
import cv2
import numpy as np
import time


class FaceDetector():
    def __init__(self, model_path, gpu_id, mem_fraction=0.3):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        self.gpuid = common.get_gpuids()[-1]
        self.model_path = model_path
        self.prefix, self.epoch = self.model_path.split(',')
        self.threshold = cons.RETINAFACE_THRESHOLD
        self.target_size = cons.SCALES_FIRST[0]
        self.max_size = cons.SCALES_FIRST[1]
        self.detector = RetinaFaceDetector(self.prefix, int(self.epoch),
                                           self.gpuid, cons.RETINAFACE_NETWORK)
        self.face_pyramid_padding = FacePyramidPadding()

    def preprocessing(self, img_arr):
        img_arr = cv2.resize(img_arr, (300, 300))
        img_arr, scale_ratio = self.face_pyramid_padding.preprocessing(img_arr)
        img_arr = img_arr[..., ::-1]
        img_arr = np.expand_dims(img_arr, 0)
        img_arr = np.transpose(img_arr, [0, 3, 1, 2])  # NCHW
        return img_arr, scale_ratio

    def forward(self, img_arr):
        out_net = self.detector.forward(img_arr)
        return out_net

    def postprocessing(self, out_net):
        bboxes, landmarks  = self.detector.postprocessing(out_net)
        if self.face_pyramid_padding.is_pyramid_image:
            bboxes, landmarks = self.face_pyramid_padding.postprocessing(bboxes, landmarks, self.threshold)
        return bboxes, landmarks
