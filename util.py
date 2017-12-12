'''
util.py
Utility operations
'''

from scipy import misc
from PIL import Image, ImageOps
import pickle
import PIL.ImageOps
import numpy as np
import imageio
import os
import matplotlib.pyplot as plt


def loadImage(file_path):
    try:
        im = misc.imread(file_path, flatten=True)
        p_im = PIL.Image.fromarray(im).convert('L')
        invert = PIL.ImageOps.invert(p_im)
        return np.array(invert)
    except:
        print("Unable to read image at {}".format(file_path))
        return None
    

def preProcessImages(images):
    to_return = []
    for im in images:
        to_return.append(removeBackground(im))
    return to_return



def binarizeImage(image):
    image_array = np.array(image)
    hist, bin_edges = np.histogram(image_array, bins=2)
    threshold = bin_edges[1]
    binarized_image = np.where(image > threshold, 1, 0)

    return binarized_image



def removeBackground(image):
    image_array = np.array(image)
    hist, bin_edges = np.histogram(image_array, bins='auto')
    # choose the bin edge (shortcut)
    halfway = int(bin_edges.shape[0] * .7)
    threshold = bin_edges[halfway]
    no_background_image = np.where(image > threshold, image, 0)

    return no_background_image



def saveImagesWithLabels(images, labels, directory='predictions'):
    if not (os.path.isdir(directory)):
        os.mkdir(directory)
    counts = [0]*(max(labels)+1)
    for i in range(images.shape[0]):
        if len(directory) > 0:
            imageio.imwrite('{}/{}-sample{}.jpg'.format(directory, labels[i],
                counts[labels[i]]), images[i])
        else:
            imageio.imwrite('{}-sample{}.jpg'.format(directory, labels[i],
                counts[labels[i]]), images[i])
        counts[labels[i]] += 1



def collectSamples(directory):
    file_names = os.listdir(directory)
    file_names.sort()
    images = []
    max_height = 0
    max_width = 0

    # first loop put all images in python list
    print("Collecing image files...")
    for f in file_names:
        inverted = loadImage(directory + '/' + f)
        if inverted is None:
            continue
        binarized = removeBackground(inverted)
        images.append(binarized)
        if binarized.shape[0] > max_height:
            max_height = binarized.shape[0]
        if binarized.shape[1] > max_width:
            max_width = binarized.shape[1]

    print("OLD max_width is {}, max_height is {}".format(max_width, max_height))
    while max_width % 16 != 0:
        max_width += 1
    while max_height % 16 != 0:
        max_height += 1
    print("NEW max_width is {}, max_height is {}".format(max_width, max_height))

    # second loop: center all images in a numpy array
    all_images = np.zeros((len(images), max_height, max_width),
            dtype=np.float32)
    print("Organizing image samples...")
    for i in range(len(images)):
        im = images[i]
        top = int((max_height - im.shape[0]) / 2)
        left = int((max_width - im.shape[1]) / 2)
        all_images[i, top:top+im.shape[0], left:left+im.shape[1]] = im

    return all_images, max_height, max_width
