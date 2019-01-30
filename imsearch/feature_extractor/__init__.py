import os
import subprocess

dir_path = os.path.dirname(os.path.realpath(__file__))


def run(redis_url=None):
    env = os.environ.copy()
    if redis_url is not None:
        env['REDIS_URI'] = redis_url
    return subprocess.Popen(['python', os.path.join(dir_path, 'extractor.py')], env=env)
