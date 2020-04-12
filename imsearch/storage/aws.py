import os
import boto3
from .abstract import AbstractStorage


class S3Storage(AbstractStorage):
    def __init__(self):
        AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID', False)
        AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY', False)
        if not (AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY):
            raise Exception(
                'AWS s3 credential config not provided: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY')

        self.bucket_name = os.environ.get('BUCKET_NAME', 'imsearch')
        self.region = os.environ.get('S3_REGION', 'us-east-2')
        self.client = boto3.client('s3', region_name=self.region,
                                   aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

        try:
            self.client.create_bucket(Bucket=self.bucket_name, CreateBucketConfiguration={
                                      'LocationConstraint': self.region})
        except self.client.exceptions.BucketAlreadyExists:
            print('{}: Bucket already exists'.format(self.bucket_name))
        except self.client.exceptions.BucketAlreadyOwnedByYou:
            print('{}: Bucket already owned by you'.format(self.bucket_name))

    def upload(self, image_path, key):
        content_type = 'image/{}'.format(key.split('.')[-1])
        self.client.upload_file(image_path, self.bucket_name, key, ExtraArgs={
                                'ACL': 'public-read', 'ContentType': content_type})

        return 'https://s3.{}.amazonaws.com/{}/{}'.format(self.region, self.bucket_name, key)
