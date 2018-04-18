'''
segment.py
use this script to create segmented word images
this is unfinished
'''

import os
from scipy import misc
from scipy.signal import argrelmin
from sklearn import linear_model
import numpy as np
import imageio
import matplotlib.pyplot as plt
import util
import numpy as np
from scipy.optimize import leastsq
import pylab as plt
from scipy.ndimage.filters import gaussian_filter
from cv2 import erode,dilate
import pdb


def main():
    image_dir = '../data/aligned_sample_pages/'
    images = os.listdir(image_dir)
    output_dir = '../data/auto_segmented_samples/'
    if not (os.path.isdir(output_dir)):
        os.mkdir(output_dir)

    all_estimates = []
    for name in images:
        if name !='.DS_Store':
            all_estimates += getWordEstimates(image_dir+name, name,
                    image_num=images.index(name))
    '''
    with open('estimates.csv', 'w') as out:
        for est in all_estimates:
            out.write('{},{},{},{},{},{}\n'.format(
                est[0], est[1], est[2], est[3], est[4], est[5]))
    '''

def getWordEstimates(image_path, image_name, image_num):
    page_image = util.loadImage(image_path)
    binary_image = util.removeBackground(page_image)

    line_locations = smartHorizontalLines(binary_image)
    tilted_lines = tiltLines(binary_image, line_locations)
    avg_line_height = calculateLineHeight(line_locations)
    avg_line_split = int(avg_line_height / 2)
    print("Average height for page {}: {}".format(image_name, avg_line_height))

    all_word_locations = []

    previous_line_bound = 0
    #words = smartVerticalLines(binary_image[line_locations[5]:line_locations[6], :],test_plots=True)
    
    estimates = []
    line_num=0
    word_num=0
   
    edge = binary_image.shape[1]
    for tilted in tilted_lines:
        # clear out above and below the slanted text
        tmp_image = np.array(binary_image)
        for i in range(0,edge):
            center = int(tilted.predict(i))
            top = center - (avg_line_split)
            bot = center + (avg_line_split)
            tmp_image[:top,i] = 0
            tmp_image[bot:,i] = 0

        left = tilted.predict(0)
        right = tilted.predict(edge)
        line_top = int(min(left, right)) - (2*avg_line_split)
        line_bot = int(min(left, right)) + (2*avg_line_split)

        word_locations = smartVerticalLines(tmp_image[line_top:line_bot, :])
        previous_word_bound = 0
        for w in word_locations[:-1]:
            center = int(tilted.predict(w))
            #plt.plot((w,w), (center-avg_line_split, center+avg_line_split), 'r')
            #estimates.append([image_name,previous_line_bound,previous_word_bound,l,w,"unlabeled"])
            previous_word_bound = w

    #edge case: get the last line of text
    l = binary_image.shape[0]
    print("working at line {}".format(l))
    word_locations = smartVerticalLines(binary_image[line_locations[-1]:l, :])
    #for w in word_locations:
        #plt.plot((w,w), (previous_line_bound, l), 'w')

    #plt.axis('off')
    #plt.show()

    return estimates



def tiltLines(image, flat_line_locations, use_ransac=False, ransac_samples=20):
    ransac_lines = []
    lr_lines = []

    # front edge case
    word_area = image[0:flat_line_locations[0]]    
    fit_points = np.where(word_area > 0)
    y = fit_points[0] 
    X = fit_points[1].reshape(-1,1) 
    if use_ransac:
        ransac = linear_model.RANSACRegressor(min_samples=min(len(X),ransac_samples))
        ransac.fit(X, y)
        ransac_lines.append(ransac)
    lr = linear_model.LinearRegression()
    lr.fit(X, y)
    lr_lines.append(lr)

    for i in range(len(flat_line_locations)-1):
        line_loc = flat_line_locations[i]
        word_area = image[flat_line_locations[i]:flat_line_locations[i+1]]
        fit_points = np.where(word_area > 0)
        y = fit_points[0] + line_loc
        X = fit_points[1].reshape(-1,1) 

        # Fit line using all data
        lr = linear_model.LinearRegression()
        lr.fit(X, y)
        lr_lines.append(lr)

        if use_ransac:
            # Robustly fit linear model with RANSAC algorithm
            ransac = linear_model.RANSACRegressor(min_samples=min(len(X),ransac_samples))
            ransac.fit(X, y)
            ransac_lines.append(ransac)

    # back edge case
    word_area = image[flat_line_locations[-1]:]
    fit_points = np.where(word_area > 0)
    y = fit_points[0] + flat_line_locations[-1]
    X = fit_points[1].reshape(-1,1) 
    if use_ransac:
        ransac = linear_model.RANSACRegressor(min_samples=min(len(X),ransac_samples))
        ransac.fit(X, y)
        ransac_lines.append(ransac)
        return ransac_lines
    lr = linear_model.LinearRegression()
    lr.fit(X, y)
    lr_lines.append(lr)

    print("Found {} lines".format(len(lr_lines)))
    return lr_lines



def smartVerticalLines(image, test_plots=False):
    print("Vertical lines!")
    kernel = np.ones((5,5), np.uint8)
    dil_image = dilate(image, kernel, iterations=2)
    im_array = np.array(dil_image)
    hist = np.sum(im_array, axis=0)
    bl = gaussian_filter(hist, sigma=10)
    lows = argrelmin(bl)[0]
    if test_plots:
        plt.plot(hist)
        plt.show()


    est_mean = np.mean(hist)
    est_std = np.std(hist)
    est_low = int(est_mean - (0.7*est_std))
    if test_plots:
        plt.plot(bl)
        #plt.plot(lows,bl[lows],'bo')
        #plt.plot((0,image.shape[1]), (est_mean, est_mean))
        #plt.plot((0,image.shape[1]), (est_low, est_low))
        plt.show()

    bl = gaussian_filter(hist, sigma=5)
    final_lows = []

    # left bound
    left_bound = np.where(hist > est_low)[0][0] - 10
    final_lows.append(left_bound)
    if test_plots:
        plt.imshow(image,cmap='gray')
        plt.axis('off')
        plt.plot((left_bound,left_bound),(0,image.shape[0]), 'w')

    previous = 0
    for l in lows:
        if bl[l] < est_low:
            final_lows.append(l)
            previous = hist[l]
            if test_plots:
                plt.plot((l,l),(0,image.shape[0]), 'w')

    # right bound
    try:
        right_bound = lows[-1]
        reversed_edge_hist = np.fliplr([hist[right_bound:]])[0]
        cross_point = np.where(reversed_edge_hist > est_low)[0][0]
        right_bound = image.shape[1] - cross_point + 10
    except Exception:
        right_bound = image.shape[1] - 10

    if test_plots:
        plt.plot((right_bound,right_bound),(0,image.shape[0]), 'w')
        plt.show()

    return final_lows




def smartHorizontalLines(image, approx_start=0):
    im_array = np.array(image)
    hist = np.sum(im_array, axis=1)
    N = hist.shape[0]
    
    bl = gaussian_filter(hist, sigma=10)
    top = np.max(bl)
    lows = argrelmin(bl)[0]

    return lows



def calculateLineHeight(line_locations):
    total_diff = 0
    n_lines = 0
    for i in range(len(line_locations)-1):
        total_diff += (line_locations[i+1] - line_locations[i])
        n_lines += 1

    return int(total_diff / n_lines)




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
