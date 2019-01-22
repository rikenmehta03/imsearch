import cv2
import glob 
import os, sys

# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import imsearch

all_images = glob.glob(os.path.join(os.path.dirname(__file__), '..', 'images/*.jpg'))

index = imsearch.init('test')
index.cleanIndex()
index.addImageBatch(all_images[0:25])
index.createIndex()
similar = index.knnQuery(all_images[25], policy='object')
qImage = cv2.imread(all_images[25])
cv2.imshow('qImage', qImage)

for _i, _s in similar:
    rImage = cv2.imread(_i['image'])
    print([x['name'] for x in _i['primary']])
    print(_s)
    cv2.imshow('rImage', rImage)
    cv2.waitKey(0)