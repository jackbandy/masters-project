'''
segment.py
use this script to create segmented word images
'''

import os
from scipy import misc
from PIL import Image, ImageOps
import PIL.ImageOps
import numpy as np
import matplotlib.pyplot as plt


def main():
    image_path = 'data/page-1.gif'
    page_image = loadImage(image_path)
    binary_image = binarizeImage(page_image)
	
    plt.imshow(binary_image)
    plt.show()



def loadImage(file_path):
    try:
        im = misc.imread(file_path, flatten=True)
        p_im = PIL.Image.fromarray(im).convert('L')
        invert = PIL.ImageOps.invert(p_im)
        return invert
    except:
        print("Unable to read {} as image".format(f))
        return None
    


def binarizeImage(image):
    image_array = np.array(image)
    print("image: {}".format(image_array.shape))
    hist, bin_edges = np.histogram(image_array, bins=2)
    threshold = bin_edges[1]
    binarized_image = np.where(image > threshold, 1, 0)
    return binarized_image



if __name__ == "__main__":
    main()
