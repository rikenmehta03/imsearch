import os
import sys
import time
import json
import base64
import numpy as np
import redis

from object_detector import get_detector
from feature_extractor import extract_features
from imsearch import utils

REDIS_DB = redis.StrictRedis.from_url(
    os.environ.get('REDIS_URI', 'redis://localhost:6379/0'))
REDIS_QUEUE = 'image_queue'
BATCH_SIZE = 1


def main():
    detector = get_detector('yolo')
    while True:
        time.sleep(0.01)
        _queue = REDIS_DB.lrange(REDIS_QUEUE, 0, BATCH_SIZE - 1)
        for _q in _queue:
            all_features = {
                'primary': [],
                'object_bitmap': [0 for _ in range(len(detector.classes))]
            }
            _q = json.loads(_q.decode("utf-8"))
            img = utils.base64_decode(_q['image'], _q['shape'])

            all_features['secondary'] = extract_features(img.copy())

            response = detector.predict(img)
            for obj in response:
                box = obj['box']
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                if(x2-x1 >= 75 and y2-y1 >= 75):
                    features = extract_features(img[y1:y2, x1:x2])
                    all_features['primary'].append({
                        'features': features,
                        'label': obj['label'],
                        'name': obj['name'],
                        'box': obj['box']
                    })
                    all_features['object_bitmap'][obj['label']] = 1

            REDIS_DB.set(_q['id'], json.dumps(all_features))
        REDIS_DB.ltrim(REDIS_QUEUE, len(_queue), -1)


if __name__ == "__main__":
    print("Running extractor")
    main()
