import os
import subprocess

dir_path = os.path.dirname(os.path.realpath(__file__))

def run():
    return subprocess.Popen(['python', os.path.join(dir_path, 'extractor.py')])