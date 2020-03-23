import cv2
import glob
import os
import sys

import imsearch


def create_index(name, images):
    # Initialize the index
    index = imsearch.init(name)

    # Clear the index and data if any exists
    index.cleanIndex()

    # Add single image to index (image path locally stored)
    index.addImage(images[0])

    # Add image using URL with same interface
    index.addImage(
        "https://www.wallpaperup.com/uploads/wallpapers/2014/04/14/332423/d5c09641cb3af3a18087937d55125ae3-700.jpg")

    # Add images in batch (List of image paths locally stored)
    index.addImageBatch(images[1:])

    # Build the index
    index.createIndex()

    return index


def create_index_with_config(name):
    '''
    parameters:
        name: name of the index (unique identifier name)
        MONGO_URI
        REDIS_URI
        DETECTOR_MODE:  select 'local' or 'remote'. 
                        local: detector backend should be running on the same machine
                        remote: Process should not start detector backend
        pass any configuration you want to expose as environment variable. 
    '''
    index = imsearch.init(name=name,
                          MONGO_URI='mongodb://localhost:27017/',
                          REDIS_URI='redis://dummy:Welcome00@123.456.78.111:6379/0',
                          DETECTOR_MODE='local')


def show_results(similar, qImage):
    qImage = imsearch.utils.check_load_image(qImage)
    qImage = cv2.cvtColor(qImage, cv2.COLOR_RGB2BGR)
    cv2.imshow('qImage', qImage)
    for _i, _s in similar:
        rImage = cv2.imread(_i['image'])
        print([x['name'] for x in _i['primary']])
        print(_s)
        cv2.imshow('rImage', rImage)
        cv2.waitKey(0)


if __name__ == "__main__":
    all_images = glob.glob(os.path.join(
        os.path.dirname(__file__), '..', 'images/*.jpg'))
    index = create_index('test', all_images[:25])

    # query index with image path
    '''
    image_path: path to image or URL
    k: Number of results
    policy: choose policy from 'object' or 'global'. Search results will change accordingly.
    '''
    similar = index.knnQuery(image_path=all_images[25], k=10, policy='object')
    show_results(similar, all_images[25])

    # query with image URL
    img_url = 'https://www.wallpaperup.com/uploads/wallpapers/2014/04/14/332423/d5c09641cb3af3a18087937d55125ae3-700.jpg'
    similar = index.knnQuery(image_path=img_url, k=10, policy='global')
    show_results(similar, img_url)

    # Create index with configuration
    index = create_index_with_config('test')
