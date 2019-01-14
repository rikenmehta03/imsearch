import cv2
import glob 
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import imsearch

all_images = glob.glob('../images/*.jpg')

index = imsearch.init('test')
# index.addImageBatch(all_images[0:25])
index.createIndex()
similar = index.knnQuery(all_images[25])

qImage = cv2.imread(all_images[25])
cv2.imshow('qImage', qImage)

for _i, _s in similar:
    rImage = cv2.imread(_i['image'])
    print(_i['primary_label_names'])
    print(_s)
    cv2.imshow('rImage', rImage)
    cv2.waitKey(0)