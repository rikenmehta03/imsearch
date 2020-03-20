import os
import sys
import shutil
import base64
import uuid
import json
import time
from PIL import Image
import numpy as np
import redis

from . import utils


class FeatureExtractor:
    def __init__(self, index_name):
        self.index_name = index_name
        self.redis_url = os.environ.get('REDIS_URI')
        self._redis_db = redis.StrictRedis.from_url(self.redis_url)
        os.makedirs(self._get_image_path(), exist_ok=True)

    def _get_image_path(self):
        home_dir = os.environ.get('HOME')
        return os.path.join(home_dir, '.imsearch', 'images', self.index_name)

    def _save_image(self, image, _id):
        dst = os.path.join(self._get_image_path(), '{}.jpg'.format(_id))
        utils.save_image(image, dst)
        return dst

    def _decode_redis_data(self, data):
        data = json.loads(data.decode('utf-8'))

        def _decode(d):
            d['features'] = utils.base64_decode(
                d['features'], dtype=np.float32)
            return d

        data['primary'] = list(map(_decode, data['primary']))
        data['secondary'] = utils.base64_decode(
            data['secondary'], dtype=np.float32)
        return data

    def clean(self):
        shutil.rmtree(self._get_image_path())
        os.makedirs(self._get_image_path(), exist_ok=True)

    def extract(self, image_path, save=True):
        image = utils.check_load_image(image_path)
        if image is None:
            return None
        _id = str(uuid.uuid4())
        data = {
            'id': _id,
            'image': utils.base64_encode(image),
            'shape': image.shape
        }
        self._redis_db.rpush('image_queue', json.dumps(data))

        result = None
        while result is None:
            time.sleep(0.01)
            result = self._redis_db.get(_id)

        result = self._decode_redis_data(result)
        result['id'] = _id
        if save:
            result['image'] = self._save_image(image, _id)

        if 'http' in image_path:
            result['url'] = image_path

        self._redis_db.delete(_id)
        return result
