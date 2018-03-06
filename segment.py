'''
segment.py
use this script to create segmented word images
this is unfinished
'''

import os
from scipy import misc
from scipy.signal import argrelmin
import numpy as np
import imageio
import matplotlib.pyplot as plt
import util


def main():
    image_path = 'data/sample_pages/Wycliffe-005-001r-2010.jpg'
    page_image = util.loadImage(image_path)
    binary_image = util.removeBackground(page_image)

    line_locations = getHorizontalLines(binary_image)
    all_word_locations = []
    plt.imshow(binary_image,cmap='gray')
    for i in range(len(line_locations) -1):
        # plot the horizontal line
        print("Working on line {}".format(i))
        plt.plot((0, binary_image.shape[1]), (line_locations[i],
            line_locations[i]), 'w-')

    plt.plot((0, binary_image.shape[1]), (line_locations[-1],
            line_locations[-1]), 'w-')
    plt.show()
    exit()



def findEnd(line_image, approx_end=1640):
    im_array = np.array(line_image)
    hist = np.sum(im_array, axis=0)
    approx_thresh = 2500
    approx_space_width = 16
    s_w = approx_space_width
    current_col = approx_end
    while np.sum(hist[current_col:current_col+s_w]) < (approx_thresh*s_w):
        current_col -= 1

    return min(current_col+approx_space_width, line_image.shape[1]-1)


def getVerticalLines(line_image, approx_start=75, approx_end=1640, n_words=0):
    #TODO make sure it's a sustained drop
    approx_thresh = 2500
    approx_space_width = 16
    s_w = approx_space_width
    min_word_width = 60
    max_word_width = 600

    im_array = np.array(line_image)
    hist = np.sum(im_array, axis=0)

    to_return = []
    current_col = approx_start
    for i in range(n_words):
        # reset for each word
        current_min = np.sum(hist[current_col-s_w:current_col])
        current_min_index = current_col
        while current_col < current_col + max_word_width:
            # perform the search
            area_sum = np.sum(hist[current_col-s_w:current_col])  
            if area_sum < current_min:
                current_min = area_sum
                current_min_index = current_col
            elif area_sum > (approx_thresh * s_w):
                # go to the next word if the threshold is crossed
                break
            current_col += 1
        
        print("Marking boundary at {}".format(current_min_index))
        to_return.append(current_min_index - int(s_w / 2))

        # move to next word
        current_col = min(current_col+min_word_width, hist.shape[0]-1)
        while np.sum(hist[current_col-s_w:current_col]) > (approx_thresh*s_w):
            if current_col < hist.shape[0]-1:
                current_col += 1
            else:
                break

    #work opposite direction for edge case
    current_col = approx_end
    while np.sum(hist[current_col:current_col+s_w]) < (approx_thresh*s_w):
        current_col -= 1
    to_return.append(current_col + int(s_w / 2))


    return to_return 



def getHorizontalLines(image):
    im_array = np.array(image)
    hist = np.sum(im_array, axis=1)
    mins = argrelmin(hist)
    
    approx_start = 60
    approx_ppl = 80
    approx_thresh = 100000
    text_lines = 45

    current_line = approx_start
    line_locations = [0]*(text_lines+1)
    for i in range(len(line_locations)-1):
        # approximate a boundary line between each line of text
        current_min = approx_thresh
        current_min_index = current_line
        while hist[current_line] < approx_thresh:
            if hist[current_line] < current_min:
                current_min = hist[current_line]
                current_min_index = current_line
            current_line += 1
        # move to next line above threshold
        if i < len(line_locations)-1:
            while hist[current_line] > approx_thresh:
                if current_line < hist.shape[0]-1:
                    current_line += 1
                else:
                    break
        line_locations[i] = current_min_index

    line_locations[-1] = 4560
    return line_locations


if __name__ == "__main__":
    main()
