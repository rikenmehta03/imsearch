import os
import shutil
import pandas as pd
import numpy as np
import nmslib

instance_map = {}


def NMSLIBIndex(index_name):
    if index_name not in instance_map:
        instance_map[index_name] = _nmslibIndex(index_name)
    return instance_map[index_name]


class _nmslibIndex:
    def __init__(self, name):
        self.index_name = name
        self._load_index()

    def _add_data(self, index, df):
        data = df.values.astype(np.float32)
        c = index.addDataPointBatch(data, range(data.shape[0]))
        return index, c

    def _get_index_path(self):
        home_dir = os.environ.get('HOME')
        return os.path.join(home_dir, '.imsearch', 'indices', self.index_name, 'index.h5')

    def _load_index(self):
        index_file = self._get_index_path()

        self.primary = nmslib.init(
            method='hnsw', space='l2', data_type=nmslib.DataType.DENSE_VECTOR)
        self.secondary = nmslib.init(
            method='hnsw', space='l2', data_type=nmslib.DataType.DENSE_VECTOR)
        self.bitmap = nmslib.init(
            method='hnsw', space='l2', data_type=nmslib.DataType.DENSE_VECTOR)

        if os.path.exists(index_file):
            self.primary_df = pd.read_hdf(index_file, 'primary')
            self.primary, self.primary_c = self._add_data(
                self.primary, self.primary_df)

            self.secondary_df = pd.read_hdf(index_file, 'secondary')
            self.secondary, self.secondary_c = self._add_data(
                self.secondary, self.secondary_df)

            self.bitmap_df = pd.read_hdf(index_file, 'bitmap')
            self.bitmap, self.bitmap_c = self._add_data(
                self.bitmap, self.bitmap_df)
        else:
            self.primary_df = None
            self.secondary_df = None
            self.bitmap_df = None

            self.primary_c, self.secondary_c, self.bitmap_c = 0, 0, 0

    def _add_data_point(self, index, count, data):
        _t = index.addDataPoint(count, data)
        return index, _t + 1

    def clean(self):
        index_file = self._get_index_path()
        if os.path.exists(os.path.dirname(index_file)):
            shutil.rmtree(os.path.dirname(index_file))
            self._load_index()

    def getDataByIds(self, id_list, _type='primary'):
        return [self.getDataById(x, _type) for x in id_list]

    def getDataById(self, _id, _type='primary'):
        return getattr(self, _type + '_df').iloc[_id].values

    def addDataPoint(self, data):
        primary = []
        for d in data['primary']:
            primary.append({
                'index': self.primary_c,
                'label': d['label'],
                'name': d['name'],
                'box': d['box']
            })
            v = d['features']
            if self.primary_df is None:
                self.primary_df = pd.DataFrame(columns=range(v.shape[0]))
            self.primary_df.loc[self.primary_c] = v
            self.primary, self.primary_c = self._add_data_point(
                self.primary, self.primary_c, v)

        db_data = {
            '_id': data['id'],
            'image': data['image'],
            'url': data.get('url', ''),
            'primary': primary,
            'secondary_index': self.secondary_c,
            'bitmap_index': self.bitmap_c
        }

        v = data['secondary']
        if self.secondary_df is None:
            self.secondary_df = pd.DataFrame(columns=range(v.shape[0]))
        self.secondary_df.loc[self.secondary_c] = v
        self.secondary, self.secondary_c = self._add_data_point(
            self.secondary, self.secondary_c, v)

        v = np.array(data['object_bitmap'], dtype=np.float32)
        if self.bitmap_df is None:
            self.bitmap_df = pd.DataFrame(columns=range(v.shape[0]))
        self.bitmap_df.loc[self.bitmap_c] = v
        self.bitmap, self.bitmap_c = self._add_data_point(
            self.bitmap, self.bitmap_c, v)

        return db_data

    def addDataPointBatch(self, data_list):
        return [self.addDataPoint(data) for data in data_list]

    def createIndex(self):
        M = 15
        efC = 100
        num_threads = 4
        index_time_params = {
            'M': M, 'indexThreadQty': num_threads, 'efConstruction': efC}
        self.primary.createIndex(index_time_params)
        self.secondary.createIndex(index_time_params)
        self.bitmap.createIndex(index_time_params)

        os.makedirs(os.path.dirname(self._get_index_path()), exist_ok=True)
        
        if self.primary_df is not None:
            self.primary_df.to_hdf(self._get_index_path(), 'primary')
        
        if self.secondary_df is not None:
            self.secondary_df.to_hdf(self._get_index_path(), 'secondary')
        
        if self.bitmap_df is not None:
            self.bitmap_df.to_hdf(self._get_index_path(), 'bitmap')

        efs = 100
        query_time_params = {'efSearch': efs}
        self.primary.setQueryTimeParams(query_time_params)
        self.secondary.setQueryTimeParams(query_time_params)
        self.bitmap.setQueryTimeParams(query_time_params)

    def knnQuery(self, data, _type='bitmap', k=10):
        response = getattr(self, _type).knnQuery(data, k=k)
        return [(int(i), float(d)) for i, d in zip(response[0], response[1])]
