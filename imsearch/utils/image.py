import sys
import base64
import requests
import cv2
from skimage import io, color

import numpy as np


def load_image(image_path):
    img = io.imread(image_path)
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img

def check_load_image(image_path):
    if requests.get(image_path).status_code != 200:
        return None

    img = io.imread(image_path)
    if len(img.shape) != 3:
        return None
    
    if img.shape[2] > 3:
        img = img[:, :, :3]
    
    return img        


def save_image(image, dest):
    cv2.imwrite(dest, image)


def base64_encode(a):
    a = a.copy(order='C')
    return base64.b64encode(a).decode("utf-8")


def base64_decode(a, shape=None, dtype=np.uint8):
    if sys.version_info.major == 3:
        a = bytes(a, encoding="utf-8")

    a = np.frombuffer(base64.decodestring(a), dtype=dtype)
    if shape is not None:
        a = a.reshape(shape)
    return a
