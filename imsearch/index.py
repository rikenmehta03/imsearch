"""
index.py
====================================
The module contains the Index class 
"""

import numpy as np
import copy

from .nmslib import NMSLIBIndex
from .extractor import FeatureExtractor
from .repository import get_repository

from .feature_extractor import run


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
        if Index.fe is None or Index.fe.poll() is not None:
            Index.fe = run()
        self.name = name
        self._nmslib_index = NMSLIBIndex(self.name)
        self._feature_extractor = FeatureExtractor(self.name)
        self._repository_db = get_repository(self.name, 'mongo')

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
                        matches[_id] = (copy.deepcopy(img_data),
                                        0.5/x[1] + matches[_id][1])
                    else:
                        matches[_id] = (copy.deepcopy(img_data), 0.5/x[1])

        knn = self._nmslib_index.knnQuery(
            features['secondary'], 'secondary', k=10)
        for x in knn:
            img_data = self._repository_db.find({'secondary_index': x[0]})
            if img_data is not None:
                _id = img_data['_id']
                if _id in matches:
                    matches[_id] = (copy.deepcopy(img_data),
                                    0.5/x[1] + matches[_id][1])
                else:
                    matches[_id] = (copy.deepcopy(img_data), 0.5/x[1])

        matches = list(matches.values())

        def update_scores(data):
            score = 0.5*np.linalg.norm(features['secondary'] - self._nmslib_index.getDataById(
                data[0]['secondary_index'], 'secondary'))
            data = (data[0], score + data[1])
            return data

        matches = list(map(update_scores, matches))
        matches.sort(reverse=True, key=lambda x: x[1])

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
            The path to the image to add to the index.
        """
        features = self._feature_extractor.extract(image_path)
        reposiory_data = self._nmslib_index.addDataPoint(features)
        self._repository_db.insert(reposiory_data)

    def addImageBatch(self, image_list):
        """
        Add a multiple image to the index. 
        Parameters
        ---------
        image_list
            The list of the image paths to add to the index.
        """
        for image_path in image_list:
            self.addImage(image_path)

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
