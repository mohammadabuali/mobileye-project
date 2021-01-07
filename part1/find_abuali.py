try:
    import os
    import json
    import glob
    import argparse
    import cv2
    import math
    import numpy as np
    from scipy import signal as sg
    import scipy.ndimage as ndimage
    from scipy.ndimage.filters import maximum_filter
    from PIL import Image
    import matplotlib.pyplot as plt
except ImportError:
    print("Need to fix the installation")
    raise


def filter_points(points, max_dist=30):
    point_set = set(points)
    set_points_copy1 = set(point_set)
    for point1 in set_points_copy1:
        if point1 not in point_set:
            continue
        set_points_copy2 = set(point_set)
        for point2 in set_points_copy2:
            if 0 < get_dist(point1, point2) < max_dist:
                point_set.remove(point2)
    return list(point_set)
def get_dist(point1, point2):
    if point1 == point2:
        return 0
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def find_tfl_lights(image: np.ndarray):
    """
    Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement
    :param c_image: The image itself as np.uint8, shape of (H, W, 3)
    :param kwargs: Whatever config you want to pass in here
    :return: 4-tuple of x_red, y_red, x_green, y_green
    """
    kernel = np.array(
        [[0, 0, 0],
         [0, 0, 0],
         [0, 1, 0],
         [1, 3, 1],
         [0, 1, 0]])

    kernel = kernel - kernel.mean()

    red_image = image.copy()
    red_image = red_image[:, :, 0]
    _, red_image = cv2.threshold(red_image, 200, 255, cv2.THRESH_BINARY)
    output = cv2.filter2D(red_image, -1, kernel)
    output_copy = output.copy()
    output = ndimage.maximum_filter(output, size=30)
    output = output - output_copy
    mask = ((output == 0) & (output_copy > 0))
    red_points = np.where(mask)
    positions = []
    final_red_points = []
    for point1 in range(len(red_points[0])):
        point = (red_points[0][point1], red_points[1][point1])
        pixel = image[point[0], point[1]]
        if (pixel[1] < 170 or pixel[2] < 120) and pixel[0] >= 200:
            final_red_points.append(point)
    final_red_points = filter_points(final_red_points)
    positions += final_red_points
    auxilary = ['r'] * len(positions)
    red_x = [val[1] for val in final_red_points]
    red_y = [val[0] for val in final_red_points]
    green_image = image.copy()
    green_image = green_image[:, :, 1]
    _, green_image = cv2.threshold(green_image, 190, 255, cv2.THRESH_BINARY)
    output = cv2.filter2D(green_image, -1, kernel)
    output_copy = output.copy()
    output = ndimage.maximum_filter(output, size=30)
    output = output - output_copy
    mask = ((output == 0) & (output_copy > 0))
    green_points = np.where(mask)
    final_green_points = []
    for point1 in range(len(green_points[0])):
        point = (green_points[0][point1], green_points[1][point1])
    pixel = image[point[0], point[1]]
    if pixel[0] <= 180 and pixel[1] >= 220 and pixel[2] >= 160:
        final_green_points.append(point)

    final_green_points = filter_points(final_green_points)
    positions += final_green_points
    auxilary += ['g'] * len(final_green_points)
    green_x = [val[1] for val in final_green_points]
    green_y = [val[0] for val in final_green_points]
    print(f"There are {len(green_x) + len(red_x)} points")
    return positions, auxilary
