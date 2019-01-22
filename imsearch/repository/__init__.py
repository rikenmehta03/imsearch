from ..exception import InvalidAttributeError

from .mongo import MongoRepository


def get_repository(index_name, repo_type):
    repo = MongoRepository(index_name)
    return RepositoryWrapper(repo)


class RepositoryWrapper:
    def __init__(self, repo):
        self.db = repo

    def clean(self):
        self.db.clean()

    def insert(self, data):
        if isinstance(data, dict):
            return self.db.insert_one(data)

        if isinstance(data, list):
            return self.db.insert_many(data)

        raise InvalidAttributeError(
            data, 'data of type dict or list expected, got {}'.format(type(data)))

    def find(self, query, many=False):
        if many:
            return self.db.find(query)
        else:
            return self.db.find_one(query)
