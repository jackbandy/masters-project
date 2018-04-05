'''
batch_rotate_crop.py
use this script to create aligned images of columns
'''

import os
import numpy as np
import imageio
import util
from scipy.ndimage.interpolation import rotate
from scipy.signal import argrelmin


INPUT_DIR = 'data/sample_pages/'
OUTPUT_DIR = 'data/aligned_sample_pages/'

CLOCKWISE_DEGREES = 2.6
COUNTER_CLOCKWISE_DEGREES = .4

def main():
    image_names = os.listdir(INPUT_DIR)
    image_names.sort()
    os.mkdir(OUTPUT_DIR)
    for im_name in image_names:
        page_image = util.loadRawImage(INPUT_DIR+im_name)
        if 'v' in im_name:
            # left-side page, rotate a smidge counter-clockwise
            print("Rotating {} counter-clockwise".format(im_name))
            rotated_image = rotate(page_image, angle=(COUNTER_CLOCKWISE_DEGREES))
            col1, col2 = splitLeftPage(rotated_image)
        elif 'r' in im_name:
            # left-side page, rotate a smidge clockwise
            print("Rotating {} clockwise".format(im_name))
            rotated_image = rotate(page_image, angle=360.-CLOCKWISE_DEGREES)
            col1, col2 = splitRightPage(rotated_image)

        imageio.imwrite(OUTPUT_DIR+im_name[:-4]+'-col1.jpg', col1)
        imageio.imwrite(OUTPUT_DIR+im_name[:-4]+'-col2.jpg', col2)

    exit()



def splitRightPage(page_image):
    # estimated split between 2500, and 2670
    inv_image = util.invertImage(page_image)
    split_index = findBestSplit(inv_image, left_est=2480, right_est=2670)

    col1 = page_image[:,:split_index+10]
    col1 = trimLeftSide(col1, left_est=930)
    col1 = trimVertical(col1, top_est=1030, bottom_est=5550)

    col2 = page_image[:,split_index-10:]
    col2 = trimRightSide(col2, right_est=1760)
    col2 = trimVertical(col2, top_est=990, bottom_est=5550)

    return col1, col2



def splitLeftPage(page_image):
    # estimated split between 2780, and 2950 
    inv_image = util.invertImage(page_image)
    split_index = findBestSplit(inv_image, left_est=2780, right_est=2950)

    col1 = page_image[:,:split_index+10]
    col1 = trimLeftSide(col1, left_est=1210)
    col1 = trimVertical(col1, top_est=1420, bottom_est=5900)

    col2 = page_image[:,split_index-10:]
    col2 = trimRightSide(col2, right_est=1760)
    col2 = trimVertical(col2, top_est=1470, bottom_est=5900)

    return col1, col2



def trimVertical(col_image, top_est=0, bottom_est=5000):
    return col_image[top_est:bottom_est,:]



def trimLeftSide(col_image, left_est=0):
    # right side has already been trimmed
    return col_image[:,left_est:]



def trimRightSide(col_image, right_est=0):
    # left side has already been trimmed
    return col_image[:,:right_est]



def findBestSplit(bin_image, left_est, right_est, window=20):
    best_window_sum = float('inf')
    best_window_index = left_est
    for i in range(left_est, right_est-window):
        window_sum = np.sum(bin_image[:, i:i+window])
        if window_sum < best_window_sum:
            best_window_index = i
            best_window_sum = window_sum
    return (best_window_index + int(window / 2))



if __name__ == "__main__":
    main()
