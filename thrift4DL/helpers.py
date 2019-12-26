# Copyright (c) 2019 congvm
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import os


def __load_image_from_url(url, mode='cv', channel_format='rgb'):
    try:
        response = requests.get(url, timeout=10)
        img_pil = Image.open(BytesIO(response.content))
        img_pil = img_pil.convert('RGB')
        if mode == 'cv':
            img_arr = np.array(img_pil, np.uint8)
            if channel_format == 'bgr':
                img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
            return img_arr

        elif mode == 'pil':
            if channel_format == 'bgr':
                img_pil = img_pil.convert('BGR')
            return img_pil
    except Exception as e:
        raise ValueError(f'Cannot process `{url}`')


def load_image_from_url(url, mode='cv', channel_format='rgb', proxy=None):
    if proxy is not None:
        os.environ['https_proxy'] = proxy
        os.environ['http_proxy'] = proxy
    url = url.lower()
    mode = mode.lower()
    channel_format = channel_format.lower()
    assert mode in ['cv', 'pil'], ValueError(
        '`mode` must be in [\'cv\', \'pil\']')
    assert channel_format in ['rgb', 'bgr'], ValueError(
        '`mode` must be in [\'rgb\', \'bgr\']')
    if 'http://' in url or 'https://' in url:
        return __load_image_from_url(url, mode, channel_format)
    else:
        raise Exception("Invalid Url")


def bytify_image(image_arr, ext="png", format_params=None):
    byte_img = cv2.imencode("." + ext, image_arr,
                            params=format_params)[1].tostring()
    return byte_img


def debytify_image(byte_image, ext="png", format_params=None):
    img_arr = cv2.imdecode(np.frombuffer(byte_image, dtype=np.uint8), 1)
    return img_arr


def hexify_image(image_arr, ext="png", format_params=None):
    byte_img = cv2.imencode("." + ext, image_arr,
                            params=format_params)[1].tostring()
    img_as_hex = byte_img.hex()
    return img_as_hex


def dehexify_image(img_as_hex):
    byte_img = bytes.fromhex(img_as_hex)
    img_arr = cv2.imdecode(np.frombuffer(byte_img, dtype=np.uint8), 1)
    return img_arr


def encode_image(image_arr, encode_type='hex', ext="jpg", format_params=None):
    assert encode_type in ['hex', 'byte'], ValueError(
        "`encode_type` must be `hex` or `byte`")
    if encode_type == 'hex':
        return hexify_image(image_arr, ext=ext, format_params=format_params)
    else:
        return bytify_image(image_arr, ext=ext, format_params=format_params)


def decode_image(encoded_img, encode_type='hex'):
    assert encode_type in ['hex', 'byte'], ValueError(
        "`encode_type` must be `hex` or `byte`")
    if encode_type == 'hex':
        return dehexify_image(encoded_img)
    else:
        return debytify_image(encoded_img)
