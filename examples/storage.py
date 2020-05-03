import cv2
import glob
import os
import sys
import json

import imsearch

gcp_config = {
    'GOOGLE_APPLICATION_CREDENTIALS': '../.config/cloud-ml-f1954f23eaa8.json',
    'BUCKET_NAME': 'imsearch-testing',
    'STORAGE_MODE': 'gcp'
}

with open('../.config/aws-config.json', 'r') as fp:
    aws_config_file = json.load(fp)

aws_config = {
    'AWS_ACCESS_KEY_ID': aws_config_file['AWS_ACCESS_KEY_ID'],
    'AWS_SECRET_ACCESS_KEY': aws_config_file['AWS_SECRET_ACCESS_KEY'],
    'BUCKET_NAME': aws_config_file['BUCKET_NAME'],
    'STORAGE_MODE': 's3'
}


def show_results(similar, qImage):
    qImage = imsearch.utils.check_load_image(qImage)
    qImage = cv2.cvtColor(qImage, cv2.COLOR_RGB2BGR)
    cv2.imshow('qImage', qImage)
    for _i, _s in similar:
        rImage = cv2.cvtColor(imsearch.utils.check_load_image(
            _i['image']), cv2.COLOR_RGB2BGR)
        print([x['name'] for x in _i['primary']])
        print(_s)
        cv2.imshow('rImage', rImage)
        cv2.waitKey(0)


if __name__ == "__main__":
    all_images = glob.glob(os.path.join(
        os.path.dirname(__file__), '..', 'images/*.jpg'))
    index = imsearch.init(name='test', **aws_config)
    index.cleanIndex()
    index.addImageBatch(all_images)
    index.createIndex()

    # query with image URL
    img_url = 'https://www.wallpaperup.com/uploads/wallpapers/2014/04/14/332423/d5c09641cb3af3a18087937d55125ae3-700.jpg'
    similar, _ = index.knnQuery(image_path=img_url, k=10, policy='global')
    show_results(similar, img_url)
