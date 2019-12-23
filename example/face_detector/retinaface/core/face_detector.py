from .main.detector_method.retinaface_detector import RetinaFaceDetector
from .preprocessing.face_pyramid_padding import FacePyramidPadding
from .utils import common
import os
from .utils import constant as cons
import cv2

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


    def detect(self, img_arr):
        padding_img_arr, scale_ratio = self.face_pyramid_padding.preprocessing(img_arr)
        bboxes, landmarks = self.detector.detect(padding_img_arr,
                                                 self.threshold,
                                                 do_flip=cons.RETINAFACE_FLIP)
        return bboxes, landmarks, scale_ratio
