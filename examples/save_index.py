import cv2
import glob
import os
import sys

import imsearch


def create_and_save(name, images, file_path):
    index = imsearch.init(name)
    index.cleanIndex()
    index.addImageBatch(images)
    index.createIndex()
    index.saveIndex(file_path)
    index.cleanIndex()


def load_index(name, file_path):
    index = imsearch.init_from_file(file_path=file_path, name=name)
    index.createIndex()
    return index


if __name__ == "__main__":
    all_images = glob.glob(os.path.join(
        os.path.dirname(__file__), '..', 'images/*.jpg'))

    create_and_save('test', all_images, 'test_index.tar.gz')
    index = load_index('test_name', 'test_index.tar.gz')