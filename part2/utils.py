import numpy as np
import matplotlib.pyplot as plt

width = 2048
height = 1024
frame_side = 81
def get_ranges(x, max, shift):
    if x <= shift:
        min_x = 0
        max_x = frame_side
    elif x >= (max - shift):
        max_x = max
        min_x = max - frame_side
    else:
        min_x = x - shift
        max_x = x + shift + 1
    return min_x, max_x


def crop_img(image, center):
    # y = center[0]
    # x = center[1]
    new_shape = tuple(sum(x) for x in zip(image.shape, (81, 81, 0)))
    new_shape = np.zeros(new_shape, dtype=int)
    new_shape[40: 40 + image.shape[0], 40: 40 + image.shape[1]] = image
    new_center = (center[0] + 40, center[1] + 40)
    z = new_shape[new_center[0] - 40: new_center[0] + 41, new_center[1] - 40: new_center[1] + 41]
    return z
    img = z
    y = new_center[1]
    x = new_center[0]
    window_side_half = 40
    min_x, max_x = get_ranges(x, width, window_side_half)
    min_y, max_y = get_ranges(y, height, window_side_half)
    cropped = img[min_y:max_y, min_x:max_x]
    assert ((81,81,3) == cropped.shape)
    # plt.imshow(cropped)
    # plt.show()
    return cropped


def export_learn_data(path, bin_data,  bin_labels = None):
    with open(f'{path}data.bin', 'ab') as data_file:
        with open(f'{path}temp.bin', 'wb') as tmp:
            print(np.array(bin_data).shape)


            np.array(bin_data).reshape((-1,)).tofile(tmp)
        with open(f'{path}temp.bin', 'rb') as tmp:
            data_file.write(tmp.read())
    if bin_labels:
        with open(f'{path}labels.bin', 'ab') as labels_file:
            with open(f'{path}temp.bin', 'wb') as tmp:
                np.array(bin_labels).astype(np.uint8).tofile(tmp)
            with open(f'{path}/temp.bin', 'rb') as tmp:
                labels_file.write(tmp.read())

def export_false_positions(path, false_list):
    with open(f'{path}false_crops_data.txt', 'a') as data_file:
        for pic in false_list:
            data_file.write(pic + str(false_list[pic]) + '\n')