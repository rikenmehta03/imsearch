import requests

from .gcloud import GcloudStorage
from .aws import S3Storage

def get_storage_object(service_name='gcp'):
    if service_name == 'gcp':
        return GcloudStorage()
    
    if service_name == 's3':
        return S3Storage()
    
    return None