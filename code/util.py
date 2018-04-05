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
import cv2


def loadRawImage(file_path):
    try:
        im = imageio.imread(file_path)
        return np.array(im)
    except:
        print("Unable to read image at {}".format(file_path))
        return None
   


def invertImage(im):
    p_im = PIL.Image.fromarray(im).convert('L')
    invert = PIL.ImageOps.invert(p_im)
    return np.array(invert)



def loadImage(file_path, invert=True):
    try:
        im = imageio.imread(file_path)
        p_im = PIL.Image.fromarray(im).convert('L')
        if invert:
            inverted = PIL.ImageOps.invert(p_im)
            return np.array(inverted)
        else:
            return np.array(p_im)
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
    blur = cv2.GaussianBlur(image_array,(5,5),0)
    cutoff, threshed = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    return threshed



def removeBackground(image):
    image_array = np.array(image)
    blur = cv2.GaussianBlur(image_array,(5,5),0)
    cutoff, threshed = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    return threshed



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



def collectSamples(directory, invert=True, binarize=True):
    file_names = os.listdir(directory)
    file_names.sort()
    images = []
    max_height = 0
    max_width = 0

    # first loop put all images in python list
    print("Collecing image files...")
    for f in file_names:
        im = loadImage(directory + '/' + f, invert=invert)
        if im is None:
            continue
        if binarize:
            im = removeBackground(im)
        images.append(im)
        if im.shape[0] > max_height:
            max_height = im.shape[0]
        if im.shape[1] > max_width:
            max_width = im.shape[1]

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
