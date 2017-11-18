'''
util.py
Utility operations
'''

from scipy import misc
from PIL import Image, ImageOps
import PIL.ImageOps
import numpy as np



def loadImage(file_path):
    try:
        im = misc.imread(file_path, flatten=True)
        p_im = PIL.Image.fromarray(im).convert('L')
        invert = PIL.ImageOps.invert(p_im)
        return np.array(invert)
    except:
        print("Unable to read {} as image".format(f))
        return None
    


def binarizeImage(image):
    image_array = np.array(image)
    hist, bin_edges = np.histogram(image_array, bins=2)
    threshold = bin_edges[1]
    binarized_image = np.where(image > threshold, 1, 0)

    return binarized_image



def removeBackground(image):
    image_array = np.array(image)
    hist, bin_edges = np.histogram(image_array, bins=2)
    threshold = bin_edges[1]
    no_background_image = np.where(image > threshold, image, 0)

    return no_background_image
