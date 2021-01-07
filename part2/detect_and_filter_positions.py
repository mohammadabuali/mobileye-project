# from load_model import model
from part2.utils import crop_img, export_learn_data
import tensorflow as tf
import os
import logging

import matplotlib.pyplot as plt
import numpy as np


def detect_and_filter(image, positions, aux):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    logging.getLogger('tensorflow').setLevel(logging.FATAL)
    model = tf.keras.models.load_model('cnn/mod0.h5')
    filtered_positions = []
    filtered_aux = []
    bin_data = []
    for position in positions:
        center = (position[1], position[0])
        cropped = crop_img(image, center)
        bin_data.append(cropped)
    l_predictions = model.predict(np.array(bin_data))
    for i in range(len(positions)):

        if l_predictions[i][0] < 0.5:
            filtered_positions.append(positions[i])
            filtered_aux.append(aux[i])

    return filtered_positions, filtered_aux
