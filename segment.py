'''
segment.py
use this script to create segmented word images
this is unfinished
'''

import os
from scipy import misc
from scipy.signal import argrelmin
import numpy as np
import matplotlib.pyplot as plt
import util


def main():
    image_path = 'data/page-1-full.gif'
    page_image = util.loadImage(image_path)
    binary_image = util.removeBackground(page_image)


    line_locations = getHorizontalLines(binary_image)
    all_word_locations = []
    plt.imshow(binary_image,cmap='gray')
    #for i in range(len(line_locations) - 1):
    for i in range(13,40):
        # plot the horizontal line
        print("Working on line {}".format(i))
        plt.plot((0, binary_image.shape[1]), (line_locations[i],
            line_locations[i]), 'w-')
        
        line_image = page_image[line_locations[i]:line_locations[i+1]]
        binary_line_image = binary_image[line_locations[i]:line_locations[i+1]]
        '''
        if i < 4:
            line_word_locations = getVerticalLines(binary_line_image, approx_start=460, n_words=wpl[i])
        else:
            line_word_locations = getVerticalLines(binary_line_image, n_words=wpl[i])
        '''
        for x in starts[i]:
            plt.plot((x,x), (line_locations[i], line_locations[i+1]), 'r-')
    plt.show()
    exit()

    '''
    for line in line_locations:
        to_plot = np.array([line for _ in range(binary_image.shape[1])])
        plt.plot(to_plot, 'w')
    '''


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
    text_lines = 41

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
        if i < len(line_locations)-1:
            while hist[current_line] > approx_thresh:
                if current_line < hist.shape[0]-1:
                    current_line += 1
                else:
                    break
        line_locations[i] = current_min_index

    return line_locations



wpl = [7, 8, 4, 5, 8, 7, 6, 6, 8, 8, 8, 8, 6, 6, 7, 6, 6, 7, 8, 6, 6, 6, 6, 7, 6, 6, 7, 8, 7, 6, 7, 5, 6, 6, 7, 9, 7, 8, 8, 6, 8, 8, 10, 8, 8]

starts = [
[],
[],
[],
[],
[],
[],
[],
[],
[],
[],
[],
[],
[300, 600, 850,1080,1340,1590],
[220, 380, 615, 990, 1370, 1600],
[285,520,720,940,1140,1370],
[370,600,830,1070,1300],
[330,560,930,1320,1545],
[260,510,740,970,1180,1400],
[220,330,490,750,850,950,1060],
[180,560,720,940,1070],
[170,550,860,1090,1390],
[220,430,810,1200,1430],
[330,570,920,1260,1490],
[200,380,615,870,1170,1360],
[350,590,800,1030,1260],
[350,580,900,1230,1470],
[210,450,670,930,1050,1470],
[360,490,750,890,1070,1320,1490],
[320,510,690,780,960,1420],
[440,530,760,930,1275],
[250,410,610,830,940,1050],
[185,580,780,1150],
[235,460,570,1180,1290],
[170,370,550,920,1450],
[195,620,725,905,1100,1300],
[330,445,670,775,910,1100,1390,1495],
[275,540,670,920,1260,1435],
[380,720,830,950,1150,1360,1450],
[320,530,790,920,1300,1430],
[250,480,560,795,950,1340]

]

if __name__ == "__main__":
    main()
