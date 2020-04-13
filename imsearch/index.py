"""
index.py
====================================
The module contains the Index class 
"""

import os
import shutil
import numpy as np
import copy
import tarfile
import requests
import pickle
import uuid
import tempfile

from .nmslib import NMSLIBIndex, get_index_path
from .extractor import FeatureExtractor
from .repository import get_repository

from .backend import run

EPSILON = 10e-5


class Index:
    """
    The class to create the searchable index object.
    """
    fe = None
    object_counter = 0

    def __init__(self, name):
        """
        Create the index with its name.
        Parameters
        ---------
        name
            Unique indentifier for your searchable index object. 
        """
        Index.object_counter += 1
        if os.environ.get('DETECTOR_MODE') == 'local' and (Index.fe is None or Index.fe.poll() is not None):
            Index.fe = run()
        self.match_ratio = 0.3
        self.name = name
        self._nmslib_index = NMSLIBIndex(self.name)
        self._feature_extractor = FeatureExtractor(self.name)
        self._repository_db = get_repository(self.name, 'mongo')

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self._nmslib_index.createIndex()

    def __del__(self):
        Index.object_counter -= 1
        if Index.object_counter == 0 and Index.fe is not None:
            Index.fe.terminate()

    def _get_object_wise_similar(self, features, k):
        matches = {}
        for data in features['primary']:
            knn = self._nmslib_index.knnQuery(
                data['features'], 'primary', k=k)
            for x in knn:
                img_data = self._repository_db.find({
                    'primary': {
                        'index': x[0],
                        'label': data['label'],
                        'name': data['name']
                    }
                })
                if img_data is not None:
                    _id = img_data['_id']

                    if _id in matches:
                        matches[_id]['s_dist'] = x[1] + matches[_id]['s_dist']
                    else:
                        matches[_id] = {
                            'data': copy.deepcopy(img_data),
                            's_dist': x[1]
                        }

        knn = self._nmslib_index.knnQuery(
            features['secondary'], 'secondary', k=k)
        for x in knn:
            img_data = self._repository_db.find({'secondary_index': x[0]})
            if img_data is not None:
                _id = img_data['_id']
                if _id in matches:
                    matches[_id]['p_dist'] = x[1]
                else:
                    matches[_id] = {
                        'data': copy.deepcopy(img_data),
                        'p_dist': x[1]
                    }

        matches = list(matches.values())
        total_objects = float(len(features['primary']))

        def update_scores(data):
            score = (1 - self.match_ratio)*data.get('p_dist', 0) / \
                (total_objects + EPSILON) + \
                self.match_ratio*data.get('s_dist', 0)
            return (data['data'], score)

        matches = list(map(update_scores, matches))
        matches.sort(key=lambda x: x[1])
        return matches

    @classmethod
    def loadFromFile(cls, path='imsearch_index.tar.gz', name=None):
        dir_path = os.path.join('/tmp', str(uuid.uuid4()))
        os.makedirs(dir_path, exist_ok=True)
        with tarfile.open(path, 'r:gz') as tf:
            tf.extractall(dir_path)

        with open(os.path.join(dir_path, 'info.pkl'), 'rb') as fp:
            info = pickle.load(fp)

        if name is not None:
            info['name'] = name

        with open(os.path.join(dir_path, 'data.pkl'), 'rb') as fp:
            data = pickle.load(fp)

        index_path = get_index_path(info['name'])
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        shutil.copyfile(os.path.join(
            dir_path, 'index', 'index.h5'), index_path)

        images_path = FeatureExtractor._get_image_path(info['name'])
        os.makedirs(images_path, exist_ok=True)
        updated_data = []
        for entity in data:
            image = entity['image']
            if image != '' and 'http' not in image:
                image_name = os.path.basename(image)
                dst = os.path.join(images_path, image_name)
                src = os.path.join(dir_path, 'images', image_name)
                shutil.copyfile(src, dst)
                entity['image'] = dst
            updated_data.append(copy.deepcopy(entity))

        db = get_repository(info['name'], 'mongo')
        db.clean()
        db.insert(updated_data)

        return cls(info['name'])

    def saveIndex(self, path='imsearch_index.tar.gz'):
        data = self._repository_db.dump()
        info = {
            'name': self.name,
            'count': len(data)
        }
        with tarfile.open(path, 'w:gz') as tf:
            tf.add(get_index_path(self.name),
                   arcname='index/index.h5', recursive=False)

            pkl_path = os.path.join('/tmp', "{}.pkl".format(str(uuid.uuid4())))
            with open(pkl_path, 'wb') as temp:
                pickle.dump(data, temp)
            tf.add(pkl_path, arcname='data.pkl', recursive=False)

            with open(pkl_path, 'wb') as temp:
                pickle.dump(info, temp)
            tf.add(pkl_path, arcname='info.pkl', recursive=False)
            os.remove(pkl_path)

            tf.add(self._feature_extractor._get_image_path(
                self.name), arcname='images/')

    def cleanIndex(self):
        """
        Cleans the index. It will delete the images already added to the index. It will also remove the database entry for the same index.
        """
        self._feature_extractor.clean()
        self._nmslib_index.clean()
        self._repository_db.clean()

    def addImage(self, image_path, save=True):
        """
        Add a single image to the index. 
        Parameters
        ---------
        image_path
            The local path or url to the image to add to the index.
        """
        features = self._feature_extractor.extract(image_path, save=save)
        if features is None:
            return False
        reposiory_data = self._nmslib_index.addDataPoint(features)
        self._repository_db.insert(reposiory_data)
        return True

    def addImageBatch(self, image_list, save=True):
        """
        Add a multiple image to the index. 
        Parameters
        ---------
        image_list
            The list of the image paths or urls to add to the index.
        """
        response = []
        for image_path in image_list:
            response.append(self.addImage(image_path, save=save))
        return response

    def createIndex(self):
        """
        Creates the index. Set create time paramenters and query-time parameters for nmslib index. 
        """
        self._nmslib_index.createIndex()

    def knnQuery(self, image_path, k=10, policy='global'):
        """
        Query the index to search for k-nearest images in the database.
        Parameters
        ---------
        image_path
            The path to the query image.
        k=10
            Number of results.
        policy='global'
            choose policy from 'object' or 'global'. Search results will change accordingly.
            object: Object level matching. The engine will look for similarity at object level for every object detected in the image.
            global: Overall similarity using single feature space on the whole image.
        """

        features = self._feature_extractor.extract(image_path, save=False)
        matches = []
        if policy == 'object':
            matches = self._get_object_wise_similar(features, k)

        if not matches:
            knn = self._nmslib_index.knnQuery(
                features['secondary'], 'secondary', k=k)
            matches = [(self._repository_db.find(
                {'secondary_index': x[0]}, many=False), 1.0/(x[1] + EPSILON)) for x in knn]

        return matches[:k]
