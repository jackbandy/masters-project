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
import util


def main():
    image_path = 'data/page-1.gif'
    page_image = util.loadImage(image_path)
    binary_image = util.removeBackground(page_image)

    vector = binary_image[530]
    plt.plot(vector)
    plt.show()



def getHorizontalLines(image):
    # zoom in
    # find the gaps
    # return the line locations
    pass



if __name__ == "__main__":
    main()
