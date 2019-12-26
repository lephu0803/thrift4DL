import os
import cv2
import numpy as np
from .core.utils import constant as cons
from .core.face_detector import FaceDetector
from .core.utils import common

if __name__ == "__main__":
    os.environ['HTTPS_PROXY'] = '10.40.34.14:81'
    os.environ['HTTP_PROXY'] = '10.40.34.14:81'
    os.environ['https_proxy'] = '10.40.34.14:81'
    os.environ['http_proxy'] = '10.40.34.14:81'
    os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] ="0"
    # export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
    MODEL_PATH = cons.MODEL_PATH
    detector = FaceDetector(model_path=MODEL_PATH, gpu_id=2)

    # The detector automatically keep image's ratio while inference so 
    # no need to resize or rescale the original image
    list_url_to_detect = ["https://f10.group.zp.zdn.vn/1666445185667442009/846cb7955c5fa501fc4e.jpg"]
    for url in list_url_to_detect:
        img_arr = common.load_image_from_url(url)
        bboxes, landmarks, scale_ratio = detector.detect(img_arr)
        print(bboxes, landmarks, scale_ratio)
        """ @bboxes: x1, y1, x2, y2, predict_score
        """
        
