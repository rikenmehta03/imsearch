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


class FeatureExtractor:
    def __init__(self, index_name):
        self.index_name = index_name
        self._redis_db = redis.StrictRedis(
            host='localhost', port='6379', db='0')

        if not os.path.exists(self._get_image_path()):
            os.makedirs(self._get_image_path())

    def _load_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = np.asarray(image).astype(np.float32)[:, :, ::-1]
        return image

    def _get_image_path(self):
        home_dir = os.environ.get('HOME')
        return os.path.join(home_dir, '.imsearch', 'images', self.index_name)

    def _save_image(self, src, _id):
        ext = src.split('.')[-1]
        dst = os.path.join(self._get_image_path(), '{}.{}'.format(_id, ext))
        shutil.copyfile(src, dst)
        return dst

    def _base64_encode(self, a):
        a = a.copy(order='C')
        return base64.b64encode(a).decode("utf-8")

    def _base64_decode(self, a, shape=None):
        if sys.version_info.major == 3:
            a = bytes(a, encoding="utf-8")

        a = np.frombuffer(base64.decodestring(a), dtype=np.float32)
        if shape is not None:
            a = a.reshape(shape)
        return a

    def _decode_redis_data(self, data):
        data = json.loads(data.decode('utf-8'))

        def _decode(d):
            d['features'] = self._base64_decode(d['features'])
            return d

        data['primary'] = map(_decode, data['primary'])
        data['secondary'] = self._base64_decode(data['secondary'])
        return data

    def clean(self):
        shutil.rmtree(self._get_image_path())
        os.makedirs(self._get_image_path())

    def extract(self, image_path, save=True):
        image = self._load_image(image_path)
        _id = str(uuid.uuid4())
        data = {
            'id': _id,
            'image': self._base64_encode(image),
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
            result['image'] = self._save_image(image_path, _id)
        self._redis_db.delete(_id)
        return result
