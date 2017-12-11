'''
segment.py
use this script to create segmented word images
'''

import os
from scipy import misc
from scipy.signal import argrelmin
import numpy as np
import matplotlib.pyplot as plt
import util


def main():
    image_path = 'data/page-1.gif'
    page_image = util.loadImage(image_path)
    binary_image = util.removeBackground(page_image)

    plt.imshow(binary_image,cmap='gray')

    line_locations = getHorizontalLines(binary_image)
    all_word_locations = []
    for i in range(len(line_locations) - 1):
        # plot the horizontal line
        plt.plot((0, binary_image.shape[1]), (line_locations[i],
            line_locations[i]), 'w-')
        line_image = binary_image[line_locations[i]:line_locations[i+1]]
        line_word_locations = getVerticalLines(line_image)
        for x in line_word_locations:
            plt.plot((x,x), (line_locations[i], line_locations[i+1]), 'r-')
        all_word_locations.append(line_word_locations)
    plt.show()
    exit()

    '''
    for line in line_locations:
        to_plot = np.array([line for _ in range(binary_image.shape[1])])
        plt.plot(to_plot, 'w')
    '''


def getVerticalLines(line_image):
    im_array = np.array(line_image)
    hist = np.sum(im_array, axis=0)
    print("Shape: {}".format(hist.shape))
    minimums = np.where(hist == 0)[0]
    return minimums


def getHorizontalLines(image):
    im_array = np.array(image)
    hist = np.sum(im_array, axis=1)
    mins = argrelmin(hist)
    
    approx_start = 60
    approx_ppl = 80
    approx_thresh = 80000
    text_lines = 21

    current_line = approx_start
    line_locations = [0]*(text_lines)
    for i in range(len(line_locations)):
        # approximate a boundary line between each line of text
        current_min = approx_thresh
        current_min_index = current_line
        while hist[current_line] < approx_thresh:
            if hist[current_line] < current_min:
                current_min = hist[current_line]
                current_min_index = current_line
            current_line += 1
        # move to next line above threshold
        while hist[current_line] > approx_thresh:
            current_line += 1
        line_locations[i] = current_min_index

    return line_locations




if __name__ == "__main__":
    main()
