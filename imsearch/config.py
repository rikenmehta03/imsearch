import os
import copy

default_config = {
    'MONGO_URI': 'mongodb://localhost:27017/',
    'REDIS_URI': 'redis://localhost:6379/0'
}


def config_init(config):
    final_config = copy.deepcopy(default_config)
    for k, v in config.items():
        final_config[k] = v
    if final_config['REDIS_URI'] == default_config['REDIS_URI']:
        final_config['DETECTOR_MODE'] = 'local'
    else:
        final_config['DETECTOR_MODE'] = 'remote'

    os.environ.update(final_config)
