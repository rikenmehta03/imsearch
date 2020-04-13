import os
import copy

default_config = {
    'MONGO_URI': 'mongodb://localhost:27017/',
    'REDIS_URI': 'redis://localhost:6379/0',
    'DETECTOR_MODE': 'local',
    'STORAGE_MODE': 'local'
}


def config_init(config):
    final_config = {}
    for k, v in default_config.items():
        final_config[k] = os.environ.get(k, v)

    for k, v in config.items():
        final_config[k] = v
    os.environ.update(final_config)
