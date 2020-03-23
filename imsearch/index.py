"""
index.py
====================================
The module contains the Index class 
"""

import os
import numpy as np
import copy
import requests

from .nmslib import NMSLIBIndex
from .extractor import FeatureExtractor
from .repository import get_repository

from .backend import run


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

    def _get_object_wise_similar(self, features):
        matches = {}
        for data in features['primary']:
            knn = self._nmslib_index.knnQuery(
                data['features'], 'primary', k=10)
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
            features['secondary'], 'secondary', k=10)
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
                (total_objects + 10e-5) + self.match_ratio*data.get('s_dist', 0)
            return (data['data'], score)

        matches = list(map(update_scores, matches))
        matches.sort(key=lambda x: x[1])
        return matches

    def cleanIndex(self):
        """
        Cleans the index. It will delete the images already added to the index. It will also remove the database entry for the same index.
        """
        self._feature_extractor.clean()
        self._nmslib_index.clean()
        self._repository_db.clean()

    def addImage(self, image_path):
        """
        Add a single image to the index. 
        Parameters
        ---------
        image_path
            The local path or url to the image to add to the index.
        """
        features = self._feature_extractor.extract(image_path)
        if features is None:
            return False
        reposiory_data = self._nmslib_index.addDataPoint(features)
        self._repository_db.insert(reposiory_data)
        return True

    def addImageBatch(self, image_list):
        """
        Add a multiple image to the index. 
        Parameters
        ---------
        image_list
            The list of the image paths or urls to add to the index.
        """
        response = []
        for image_path in image_list:
            response.append(self.addImage(image_path))
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
            matches = self._get_object_wise_similar(features)

        if not matches:
            knn = self._nmslib_index.knnQuery(
                features['secondary'], 'secondary', k=10)
            matches = [(self._repository_db.find(
                {'secondary_index': x[0]}, many=False), 1.0/x[1]) for x in knn]

        return matches[:k]
