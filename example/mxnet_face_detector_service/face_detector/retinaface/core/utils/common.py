import numpy as np
import os
import random
import cv2
import urllib
from . import constant as cons
from zaailabcorelib.ztools.image import load_image


def get_gpuids():
    gpuids = []
    cvd = os.environ['CUDA_VISIBLE_DEVICES'].strip()
    if len(cvd) > 0:
        for i in range(len(cvd.split(','))):
            gpuids.append(i)
    return gpuids

def calculate_image_scale_ratio(img, desired_size):
    target_size = desired_size[0]
    max_size = desired_size[1]
    img_shape = img.shape
    img_size_min = img_shape[0]
    img_size_max = img_shape[1]
    img_scale = float(target_size) / float(img_size_min)
    if np.round(img_scale * img_size_max) > max_size:
        img_scale = float(max_size) / float(img_size_max)
    return img_scale

def resize_image(img, img_scale):
    return cv2.resize(img, None, None, fx=img_scale, fy=img_scale, interpolation=cv2.INTER_LINEAR)

def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret

def load_image_from_url(url, proxies=None):
    return load_image(url, proxy=proxies)