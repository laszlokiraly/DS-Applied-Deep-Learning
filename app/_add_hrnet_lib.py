from os import path
import sys


def path_to_sys_path(new_path):
    if new_path not in sys.path:
        sys.path.insert(0, new_path)

path_to_sys_path(path.join(path.dirname(__file__), '..', 'hrnet-imagenet-valid/lib'))
# path_to_sys_path(path.join(path.dirname(__file__), '..', 'transfer'))
