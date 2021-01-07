import tfl_man
import pickle
import os
from PIL import Image
import numpy as np


def get_pictures_paths():
    return list(filter(lambda x: x[-3:] == 'png', os.listdir(r'part3')))


def init(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data, get_pictures_paths()


def run():
    distances = None
    pkl, paths = init('part3/dusseldorf_000049.pkl')
    manager = tfl_man.Manager(pkl, 24)
    for path in paths:
        curr_frame = np.array(Image.open('part3/' + path))
        manager.set_curr_frame(curr_frame)
        distances = manager.on_frame()

    return distances


if __name__ == '__main__':
    run()
