import numpy as np
from imsearch import Index
import imsearch

import unittest
from unittest.mock import patch, Mock


class TestIndex(unittest.TestCase):
    @patch('imsearch.index.NMSLIBIndex')
    @patch('imsearch.index.FeatureExtractor')
    @patch('imsearch.index.get_repository')
    @patch('imsearch.index.os.environ')
    def setUp(self, mock_os_environ, mock_get_repository, mock_extractor, mock_nmslib):
        self.mock_nmslib_instance = mock_nmslib()
        self.mock_extractor_instance = mock_extractor()
        self.mock_get_repository = mock_get_repository
        self.mock_repository_instance = Mock()
        self.mock_get_repository.return_value = self.mock_repository_instance
        mock_os_environ.get.return_value = 'remote'

        self.test_index = Index('test')

    def test_constructor(self):
        self.assertIsInstance(self.test_index, Index)
        self.assertEqual(self.test_index.name, 'test')
        self.assertEqual(self.test_index._nmslib_index,
                         self.mock_nmslib_instance)
        self.assertEqual(self.test_index._feature_extractor,
                         self.mock_extractor_instance)
        self.mock_get_repository.assert_called_with('test', 'mongo')

    def test_add_image_valid(self):
        IMG_PATH = '../images/000000000139.jpg'
        mock_data = {
            'id': '12345',
            'image': IMG_PATH,
            'secondary': np.zeros(1024)
        }

        self.mock_extractor_instance.extract.return_value = mock_data
        self.mock_nmslib_instance.addDataPoint.return_value = mock_data

        self.assertEqual(self.test_index.addImage(IMG_PATH), True)
        self.mock_extractor_instance.extract.assert_called_with(IMG_PATH, save=True)
        self.mock_nmslib_instance.addDataPoint.assert_called_with(mock_data)
        self.mock_repository_instance.insert.assert_called_with(mock_data)

    def test_add_image_invalid(self):
        IMG_PATH = '../images/000000000139.jpg'

        self.mock_extractor_instance.extract.return_value = None

        self.assertEqual(self.test_index.addImage(IMG_PATH), False)
        self.mock_extractor_instance.extract.assert_called_with(IMG_PATH, save=True)
        self.mock_nmslib_instance.addDataPoint.assert_not_called()
        self.mock_repository_instance.insert.assert_not_called()

    def test_add_image_batch(self):
        list_length = 20
        IMG_PATH = '../images/000000000139.jpg'
        IMAGE_LIST = [IMG_PATH] * list_length
        mock_data = {
            'id': '12345',
            'image': IMG_PATH,
            'secondary': np.zeros(1024)
        }

        self.mock_extractor_instance.extract.return_value = mock_data
        self.mock_nmslib_instance.addDataPoint.return_value = mock_data

        self.assertEqual(self.test_index.addImageBatch(
            IMAGE_LIST), [True]*list_length)
        self.assertEqual(
            self.mock_extractor_instance.extract.call_count, list_length)
        self.assertEqual(
            self.mock_nmslib_instance.addDataPoint.call_count, list_length)
        self.assertEqual(
            self.mock_repository_instance.insert.call_count, list_length)

    def test_create_index(self):
        self.test_index.createIndex()
        self.mock_nmslib_instance.createIndex.assert_called()

    def test_knn_query(self):
        pass


if __name__ == "__main__":
    unittest.main()
