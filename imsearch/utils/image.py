import sys
import base64
import requests
import cv2
from PIL import Image
from io import BytesIO

import numpy as np


def check_load_image(image_path):
    if 'http' in image_path:
        r = requests.get(image_path)
        if r.status_code != 200:
            return None
        img = np.asarray(Image.open(BytesIO(r.content)))
    else:
        img = np.asarray(Image.open(image_path))

    if len(img.shape) != 3:
        return None

    if img.shape[2] > 3:
        img = img[:, :, :3]

    return img


def save_image(image, dest):
    image = Image.fromarray(image)
    image.save(dest)


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
