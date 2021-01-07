import tfl_detection
import os
import json
import glob
import argparse
import cv2
import math
import numpy as np
import random
from scipy import signal as sg
import scipy.ndimage as ndimage
from scipy.ndimage.filters import maximum_filter
from PIL import Image
import matplotlib.pyplot as plt

label_base_path = 'gtFine_trainvaltest/gtFine/'
data_base_path = 'leftImg8bit_trainvaltest/leftImg8bit/'
label_folder = 'gtFine_trainvaltest/gtFine/train/aachen'
data_folder = 'leftImg8bit_trainvaltest/leftImg8bit/train/aachen'
strip_data_ = '_leftImg8bit.png'
strip_label_ = '_gtFine_labelIds.png'


def strip_data(title):
    return title[:len(title) - len(strip_data_)]


def strip_label(title):
    return title[:len(title) - len(strip_label_)]


def center_tfl(labeled_image, pixel):
    return pixel
    left, right, up, down = 0, 0, 0, 0
    check = 1
    try:
        while check:
            check = 0
            if pixel[0] - left > 0 and labeled_image[pixel[0] - left, pixel[1]] == 19:
                left += 1
                check = 1
            if pixel[0] + right < len(labeled_image) and labeled_image[pixel[0] + right, pixel[1]] == 19:
                right += 1
                check = 1
            if pixel[0] - down > 0 and labeled_image[pixel[0], pixel[1] - down] == 19:
                down += 1
                check = 1
            if pixel[0] + up < len(labeled_image[0]) and labeled_image[pixel[0], pixel[1] + up] == 19:
                up += 1
                check = 1
    except:
        print("Pixel: ", pixel)
        print("left: ", left)
        print("right: ", right)
        print("down: ", down)
        print("up: ", up)

    center_pixel_vertical = pixel[0] + (right - left)//2
    center_pixel_horizontal = pixel[1] + (up - down)//2
    return center_pixel_vertical, center_pixel_horizontal

def crop_img(image, center, value=1, phase='train/'):
    new_shape = tuple(sum(x) for x in zip(image.shape,(81,81,0)))
    new_shape = np.zeros(new_shape, dtype=int)
    new_shape[40: 40 + image.shape[0], 40: 40 + image.shape[1]] = image
    new_center = (center[0] + 40, center[1] + 40)
    z = new_shape[new_center[0] - 40: new_center[0] + 41, new_center[1] - 40: new_center[1] + 41]
    with open(f'Data/{phase}data.bin', 'ab') as f1:
        z.astype('uint8').tofile(f1)

    with open(f'Data/{phase}labels.bin', 'ab') as f2:
        np.array([value]).astype('uint8').tofile(f2)
    return z

def check_if_far_from_edges(image, center):
    if center[0] - 40 < 0 or center[0] >= len(image):
        return False
    if center[1] - 40 < 0 or center[0] >= len(image[0]):
        return False
    return True

def crop_non_tfl(image, center_lst, counter, phase='train/'):
    if counter == 0:
        return
    random.shuffle(center_lst)
    close_to_edge_points = []
    for center in center_lst:
        if check_if_far_from_edges(image, center):
            counter -= 1
            crop_img(image, center, value=0, phase=phase)
            if counter == 0:
                return
        else:
            close_to_edge_points.append(center)
    for index in range(min(counter, len(close_to_edge_points))):
        crop_img(image, close_to_edge_points[index], value=0, phase=phase)

    pass


def test_find_tfl_lights2(image_path, json_path=None, fig_num=None, phase='train/'):
    """
    Run the attention code
    """
    data_image_title = data_folder + '/' + image_path
    image = np.array(Image.open(data_image_title))
    labeled_image_title = label_folder + '/' + strip_data(image_path) + strip_label_
    labeled_image = np.array(Image.open(labeled_image_title))
    if json_path is None:
        objects = None
    else:
        gt_data = json.load(open(json_path))
        what = ['traffic light']
        objects = [o for o in gt_data['objects'] if o['label'] in what]

    tfl_counter = 0
    pixels = []
    not_tfl_lst = []
    red_x, red_y, green_x, green_y = tfl_detection.find_tfl_lights(image, some_threshold=42)
    for i, j in zip(red_x, red_y):
        if labeled_image[j, i] == 19:
            crop_img(image, center_tfl(labeled_image, (j, i)), phase=phase)
            tfl_counter += 1
        else:
            not_tfl_lst.append((j, i))

            # pixels.append(center_tfl(labeled_image, (j, i)))
            # tfl_detection.show_image_and_gt(crop_image(image, center_tfl(labeled_image, (j, i))), objects, fig_num)

    for i, j in zip(green_x, green_y):
        if labeled_image[j, i] == 19:
            tfl_counter += 1
            crop_img(image, center_tfl(labeled_image, (j, i)), phase=phase)
        else:
            not_tfl_lst.append((j, i))
    crop_non_tfl(image, not_tfl_lst, tfl_counter, phase=phase)
        # if labeled_image[j, i] == 19:
        #     pixels.append(center_tfl(labeled_image, (j, i)))
        #     tfl_detection.show_image_and_gt(crop_image(image, center_tfl(labeled_image, (j, i))), objects, fig_num)
    # p_x = [x[0] for x in pixels]
    # p_y = [y[1] for y in pixels]

    # plt.plot(red_x, red_y, '+', color='r', markersize=10)
    # if len(pixels) > 0:
    #     tfl_detection.show_image_and_gt(image, objects, fig_num)
    #     plt.plot(p_y, p_x, '.', color='g', markersize=10)


def main(argv=None):
    """It's nice to have a standalone tester for the algorithm.
    Consider looping over some images from here, so you can manually exmine the results
    Keep this functionality even after you have all system running, because you sometime want to debug/improve a module
    :param argv: In case you want to programmatically run this"""
    global data_folder
    global label_folder
    parser = argparse.ArgumentParser("Tsm")
    parser.add_argument('-i', '--image', type=str, help='Path to an image')
    parser.add_argument("-j", "--json", type=str, help="Path to json GT for comparison")
    parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
    args = parser.parse_args(argv)
    for phase in ['train/', 'test/', 'val/']:
        for city in os.listdir(data_base_path+phase):
            data_folder = data_base_path + phase + city
            label_folder = label_base_path + phase + city
            default_base = data_folder
            if args.dir is None:
                args.dir = default_base
            # flist = glob.glob(os.path.join(args.dir, '*_leftImg8bit.png'))
            flist = os.listdir(data_folder)
            for image in flist:
                json_fn = image.replace('_leftImg8bit.png', '_gtFine_polygons.json')
                if not os.path.exists(json_fn):
                    json_fn = None
                test_find_tfl_lights2(image, json_fn, phase=phase)
            if len(flist):
                print("You should now see some images, with the ground truth marked on them. Close all to quit.")
            else:
                print("Bad configuration?? Didn't find any picture to show")
    plt.show(block=True)


if __name__ == '__main__':
    # with open('Data/train/data.bin', 'wb') as f1:
    #     print("hi")
    #     pass
    # with open('Data/train/labels.bin', 'wb') as f2:
    #     print("hi")
    #     pass
    # with open('Data/val/data.bin', 'wb') as f3:
    #     print("hi")
    #     pass
    # with open('Data/val/labels.bin', 'wb') as f4:
    #     print("hi")
    #     pass
    with open('Data/test/data.bin', 'wb') as f5:
        print("hi")
        pass
    with open('Data/test/labels.bin', 'wb') as f6:
        print("hi")
        pass

    main()