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


def find_tfl_lights(image: np.ndarray, **kwargs):
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
    output = output-output_copy
    mask = ((output == 0) & (output_copy > 0))
    red_points = np.where(mask)

    final_red_points = []
    for point1 in range(len(red_points[0])):
        point = (red_points[0][point1], red_points[1][point1])
        pixel = image[point[0], point[1]]
        if (pixel[1] < 170 or pixel[2] < 120) and pixel[0] >= 200:
            final_red_points.append(point)
    final_red_points = filter_points(final_red_points)
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
    green_x = [val[1] for val in final_green_points]
    green_y = [val[0] for val in final_green_points]
    print(f"There are {len(green_x) + len(red_x)} points")
    candidates, aux = [], []
    for i,j in zip(red_x, red_y):
        candidates.append((red_x, red_y))
        aux.append('r')
    for i,j in zip(green_x, green_y):
        candidates.append((green_x, green_y))
        aux.append('g')
    # return red_x, red_y, green_x, green_y
    return candidates, aux


def show_image_and_gt(image, objs, fig_num=None):
    plt.figure(fig_num).clf()
    plt.imshow(image)
    labels = set()
    if objs is not None:
        for o in objs:
            poly = np.array(o['polygon'])[list(np.arange(len(o['polygon']))) + [0]]
            plt.plot(poly[:, 0], poly[:, 1], 'r', label=o['label'])
            labels.add(o['label'])
        if len(labels) > 1:
            plt.legend()


def test_find_tfl_lights2(image_path, json_path=None, fig_num=None):
    """
    Run the attention code
    """
    image = np.array(Image.open(image_path))
    if json_path is None:
        objects = None
    else:
        gt_data = json.load(open(json_path))
        what = ['traffic light']
        objects = [o for o in gt_data['objects'] if o['label'] in what]

    show_image_and_gt(image, objects, fig_num)

    red_x, red_y, green_x, green_y = find_tfl_lights(image, some_threshold=42)
    plt.plot(red_x, red_y, '+', color='r', markersize=10)
    plt.plot(green_x, green_y, '.', color='g', markersize=10)


def main(argv=None):
    """It's nice to have a standalone tester for the algorithm.
    Consider looping over some images from here, so you can manually exmine the results
    Keep this functionality even after you have all system running, because you sometime want to debug/improve a module
    :param argv: In case you want to programmatically run this"""
    parser = argparse.ArgumentParser("Test TFL attention mechanism")
    parser.add_argument('-i', '--image', type=str, help='Path to an image')
    parser.add_argument("-j", "--json", type=str, help="Path to json GT for comparison")
    parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
    args = parser.parse_args(argv)
    default_base = './berlin'
    if args.dir is None:
        args.dir = default_base
    flist = glob.glob(os.path.join(args.dir, '*_leftImg8bit.png'))
    for image in flist:
        json_fn = image.replace('_leftImg8bit.png', '_gtFine_polygons.json')
        if not os.path.exists(json_fn):
            json_fn = None
        test_find_tfl_lights2(image, json_fn)
    if len(flist):
        print("You should now see some images, with the ground truth marked on them. Close all to quit.")
    else:
        print("Bad configuration?? Didn't find any picture to show")
    plt.show(block=True)




if __name__ == '__main__':

    main()