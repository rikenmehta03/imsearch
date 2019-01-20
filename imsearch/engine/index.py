import numpy as np

from .nmslib import NMSLIBIndex
from .extractor import FeatureExtractor
from .repository import get_repository

from ..feature_extractor import run

class Index:
    fe = run()
    object_counter = 0
    def __init__(self, name):
        Index.object_counter += 1
        if Index.fe.poll() is not None:
            Index.fe = run()
        self.name = name
        self._nmslib_index = NMSLIBIndex(self.name)
        self._feature_extractor = FeatureExtractor(self.name)
        self._repository_db = get_repository(self.name, 'mongo')
    
    def __del__(self):
        Index.object_counter -= 1
        if Index.object_counter == 0:
            Index.fe.terminate()

    def _find_score(self, label, q_feature, match):
        all_dists = []
        for i, l in enumerate(match['primary_label']):
            if l == label:
                f = self._nmslib_index.getDataById(match['primary_id'][i], 'primary')
                all_dists.append(np.linalg.norm(q_feature-f))

        return min(all_dists) if all_dists else None

    def _sort_matches(self, features, matches):
        new_matches = [] 
        for _match, _score in matches:
            score = 0
            for label, v in features['primary'].items():
                q_feature = v[0]
                _s = self._find_score(int(label), q_feature, _match)
                if _s is not None:
                    score += _s
            score /= len(features['primary'])
            score = 0.5*score + 0.5*np.linalg.norm(features['secondary'] - self._nmslib_index.getDataById(_match['secondary_id'], 'secondary'))
            new_matches.append((_match, score))
        new_matches.sort(key=lambda x: x[1])
        return new_matches
    
    def cleanIndex(self):
        self._feature_extractor.clean()
        self._nmslib_index.clean()
        self._repository_db.clean()

    def addImage(self, image_path):
        features = self._feature_extractor.extract(image_path)
        reposiory_data = self._nmslib_index.addDataPoint(features)
        self._repository_db.insert(reposiory_data)

    def addImageBatch(self, image_list):
        for image_path in image_list:
            self.addImage(image_path)

    def createIndex(self):
        self._nmslib_index.createIndex()

    def knnQuery(self, image_path, k=10):
        features = self._feature_extractor.extract(image_path, save=False)
        # knn = self._nmslib_index.knnQuery(features['object_bitmap'], 'bitmap', k=10)
        # matches = [(self._repository_db.find(
        #     {'bitmap_id': x[0]}, many=False), x[1]) for x in knn]
        # return self._sort_matches(features, matches)

        knn = self._nmslib_index.knnQuery(features['secondary'], 'secondary', k=10)
        matches = [(self._repository_db.find({'secondary_id': x[0]}, many=False), x[1]) for x in knn]
        return matches
