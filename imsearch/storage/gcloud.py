import os
from google.cloud import storage
from .abstract import AbstractStorage


class GcloudStorage(AbstractStorage):
    def __init__(self):
        if not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', False):
            raise Exception(
                'Google cloud credential config not provided: GOOGLE_APPLICATION_CREDENTIALS')

        self.bucket_name = os.environ.get('BUCKET_NAME', 'imsearch')
        self.client = storage.Client()

        try:
            bucket = self.client.lookup_bucket(self.bucket_name)
            if not bucket:
                self.bucket = self.client.create_bucket(self.bucket_name)
            else:
                self.bucket = bucket
        except Exception as e:
            raise e

    def upload(self, image_path, key):
        blob = self.bucket.blob(key)
        blob.upload_from_filename(image_path)
        blob.make_public()
        return blob.public_url
