import imsearch

import unittest
from unittest.mock import patch


class TestInit(unittest.TestCase):

    @patch('imsearch.config_init')
    @patch('imsearch.Index')
    def test_init(self, mock_index, config_init_mock):
        instance = mock_index()
        instance.name = 'test'

        index = imsearch.init('test', MONGO_URI='mongodb://localhost:27017/')
        self.assertEqual(index, instance)
        self.assertEqual(index.name, 'test')
        config_init_mock.assert_called_with(
            {'MONGO_URI': 'mongodb://localhost:27017/'})

    @patch('imsearch.run')
    def test_detector(self, mock_backend):
        imsearch.run_detector('redis://dummy:Password@111.111.11.111:6379/0')
        mock_backend.assert_called_with(
            'redis://dummy:Password@111.111.11.111:6379/0')


if __name__ == "__main__":
    unittest.main()
