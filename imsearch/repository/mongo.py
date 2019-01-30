import os
from pymongo import MongoClient


class MongoRepository:
    def __init__(self, index_name):
        url = os.environ.get('MONGO_URI')
        self.db = MongoClient(url).imsearch[index_name]

    def clean(self):
        self.db.drop()

    def insert_one(self, data):
        return self.db.insert_one(data)

    def insert_many(self, data):
        return self.db.insert_many(data)

    def find_one(self, query):
        response = self.db.find_one(query)
        return response

    def find(self, query):
        return list(self.db.find(query))
