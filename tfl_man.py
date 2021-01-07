import numpy as np
import matplotlib.pyplot as plt
import part3.SFM_standAlone as sfm
from part1 import find_tfl
from part2 import detect_and_filter_positions
import time


class Container:
    def __init__(self, frame, index, tfl, aux):
        self.frame = frame
        self.index = index
        self.tfl = tfl
        self.aux = aux

    def set_tfl_and_aux(self, tfl, aux):
        self.tfl = tfl
        self.aux = aux


class Manager:
    def __init__(self, pkl, start_index):
        self.pkl_data = pkl
        self.start_index = start_index
        self.curr_container = None
        self.prev_container = None
        self.pp = self.get_pp_and_fl(self.pkl_data)[0]
        self.fl = self.get_pp_and_fl(self.pkl_data)[1]

    def set_curr_frame(self, frame):
        self.prev_container = self.curr_container
        curr_index = self.prev_container.index + 1 if self.prev_container else self.start_index
        self.curr_container = Container(frame, curr_index, None, None)

    def get_pkl_file_data(self):
        return self.pkl_data

    def get_EM(self, data):
        i_c, i_p = self.curr_container.index, self.prev_container.index
        EM = np.eye(4)
        prev_frame_id, curr_frame_id = i_p, i_c
        for i in range(prev_frame_id, curr_frame_id):
            EM = np.dot(data['egomotion_' + str(i) + '-' + str(i + 1)], EM)
        return EM

    def get_pp_and_fl(self, data):
        return data['principle_point'], data['flx']

    def get_light_sources(self, img):
        return find_tfl.find_tfl_lights(img)

    def split_data(self, candidates, auxiliary):
        red_x, red_y, green_x, green_y = [], [], [], []
        for pixel, color in zip(candidates, auxiliary):
            if color == 'r':
                red_x.append(pixel[0])
                red_y.append(pixel[1])
            else:
                green_x.append(pixel[0])
                green_y.append(pixel[1])
        return red_x, red_y, green_x, green_y

    def plot_distance_to_tfl(self, distances, plot):
        """

        :param distances:
        :param plot:
        :return:
        """
        image = self.curr_container.frame
        tfl = self.curr_container.tfl
        auxiliary = self.curr_container.aux
        self.plot_candidates(image, tfl, auxiliary, plot)
        for i in range(len(tfl)):
            plot.text(tfl[i][0], tfl[i][1],
                      r'{0:.1f}'.format(distances[i]), color='y')

    def plot_candidates(self, image, candidates, auxiliary, plot):
        plot.imshow(image)
        red_x, red_y, green_x, green_y = self.split_data(candidates, auxiliary)
        plot.plot(red_x, red_y, 'ro', color='r', markersize=4)
        plot.plot(green_x, green_y, '+', color='g', markersize=4)
        pass

    def crop(self, img):
        return img[50:(len(img) * 3) // 4, 20:-20]

    def trim_light_sources(self, img, candidates, auxiliary):
        """
        trims down the candidates to true traffic lights
        :param img:
        :param candidates:
        :param auxiliary:
        :return:
        """
        return detect_and_filter_positions.detect_and_filter(img, candidates, auxiliary)

    def get_dist(self, EM):
        """
        finds the distances to each traffic light
        :param EM:
        :return:
        """
        prev_img = self.prev_container.frame
        curr_img = self.curr_container.frame
        prev_tfl = self.prev_container.tfl
        curr_tfl = self.curr_container.tfl
        prev_aux = self.prev_container.aux
        curr_aux = self.curr_container.aux
        pp = self.pp
        fl = self.fl

        return sfm.get_distances(prev_img, curr_img, prev_tfl, curr_tfl, prev_aux, curr_aux, EM, pp, fl)

    def on_frame(self):
        """
        This function receives a new frame, and does the following:
        1) Calls a function which finds all the candidates that could be traffic lights.
            It enters a certain frame as an input, and receives candidates and auxiliary points as output

        2) Calls a function which runs a CNN over the data it received so that it determines which of the data
            truly is a traffic light. It enters the output of the previous function as input, and receives
            the traffic lights points, and their corresponding auxiliary as an output

        3) Calls a function which returns the distance from traffic lights. Receives the output of the second
            function as an input, and receives the distances of the traffic lights from the car as an output

        In addition to plotting a series of 3  pics each showing the result of its corresponding part.
        :return:
        """

        # part1
        pkl_file_data = self.get_pkl_file_data()
        curr_img = self.crop(self.curr_container.frame)
        start = time.time()
        candidates, auxiliary = self.get_light_sources(curr_img)
        print(f'Run time of phase 1 is: {time.time() - start}')
        if self.prev_container:
            fig, ax = plt.subplots(3, 1, figsize=(6, 18))
            self.plot_candidates(curr_img, candidates, auxiliary, ax[0])

        # part2
        start = time.time()
        traffic_lights, auxiliary = self.trim_light_sources(curr_img, candidates, auxiliary)
        print(f'Run time of phase 2 is: {time.time() - start}')

        if self.prev_container:
            self.plot_candidates(curr_img, traffic_lights, auxiliary, ax[1])
        self.curr_container.set_tfl_and_aux(traffic_lights, auxiliary)
        if not self.prev_container:
            print()
            return

        # part3
        EM = self.get_EM(pkl_file_data)
        start = time.time()
        distances = self.get_dist(EM)
        print(f'Run time of phase 3 is: {time.time() - start}\n')
        self.plot_distance_to_tfl(distances, ax[2])
        plt.show()

        return distances
