'''
util.py
Utility operations
'''

from scipy import misc
from skimage.transform import resize
from skimage import feature
from PIL import Image, ImageOps
import pickle
import PIL.ImageOps
import numpy as np
import imageio
import os
import cv2
import collections
from keras.models import load_model



def buildDataset(words, n_words):
    """Process raw inputs into a dataset.
    http://adventuresinmachinelearning.com/word2vec-tutorial-tensorflow/"""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary



def getPixelFeatsForSamples(samples, scale=1):
    features = []
    for s in samples:
        if scale != 1:
            s = resize(s, (int(s.shape[0]/scale), int(s.shape[1]/scale)))
        features.append(s.flatten())

    return np.array(features)



def getVAEFeatsForSamples(samples, vae_encoder_path):
    encoder = load_model(vae_encoder_path)
    im_height = samples.shape[1]
    im_width = samples.shape[2]
    white_val = np.max(samples)
    input_images = samples.reshape(samples.shape[0], im_height, im_width, 1)/white_val
    encoded = encoder.predict(input_images)
    encoded = np.reshape(encoded, (encoded.shape[0], -1))
    return encoded



def getHandFeatsForSamples(samples):
    features = []
    for s in samples:
        s_features = []

        # global features
        mean_val = np.mean(s) 
        s_features.append(mean_val)
        max_val = np.max(s)
        s_features.append(max_val)
        std_val = np.std(s)
        s_features.append(std_val)

        # vertical profile features
        vertical = np.sum(s, axis=0)
        v_mean = np.mean(vertical)
        s_features.append(v_mean)
        v_max = np.max(vertical)
        s_features.append(v_max)
        v_std = np.std(vertical)
        s_features.append(v_std)
        v_cross = meanCrossRate(vertical, v_mean)
        s_features.append(v_cross)

        # horizontal profile features
        horizontal = np.sum(s, axis=1)
        h_mean = np.mean(horizontal)
        s_features.append(h_mean)
        h_max = np.max(horizontal)
        s_features.append(h_max)
        h_std = np.std(horizontal)
        s_features.append(h_std)
        h_cross = meanCrossRate(horizontal, h_mean)
        s_features.append(h_cross)

        features.append(s_features)

    return np.array(features)



def meanCrossRate(signal, mean):
    crosses = 0
    for i in range(1,len(signal)):
        if signal[i-1] < mean and signal[i] > mean:
            crosses += 1
        elif signal[i-1] > mean and signal[i] < mean:
            crosses += 1

    return (crosses / len(signal))



def getHogForSamples(samples, scale=1):
    hog_array = []
    for s in samples:
        if scale != 1:
            s = resize(s, (int(s.shape[0]/scale), int(s.shape[1]/scale)))
        f = feature.hog(s, orientations=9, pixels_per_cell=(8,8),
                cells_per_block=(2,2), feature_vector=True)
        hog_array.append(f)
    return np.array(hog_array)



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



def collectSamples(directory, invert=True, binarize=True, scale_to_fill=False,
        fixed_max_width=None, fixed_max_height=None):
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

    while max_width % 16 != 0:
        max_width += 1
    while max_height % 16 != 0:
        max_height += 1

    if fixed_max_width:
        max_width = fixed_max_width
    if fixed_max_height:
        max_height = fixed_max_height

    print("Dimensions: {} x {}".format(max_width, max_height))

    # second loop: center all images in a numpy array
    all_images = np.zeros((len(images), max_height, max_width),
            dtype=np.uint8)
    print("Organizing image samples...")
    for i in range(len(images)):
        if scale_to_fill:
            scaled_im = resize(images[i], (max_height, max_width))
            white_val = np.max(scaled_im)
            scaled_im = np.where(scaled_im > 0, white_val, 0)
            all_images[i] = scaled_im
        im = images[i]
        if im.shape[1] > max_width:
            top = int((max_height - im.shape[0]) / 2)
            all_images[i, top:top+im.shape[0], :] = im[:, :max_width]
            continue
        top = int((max_height - im.shape[0]) / 2)
        left = int((max_width - im.shape[1]) / 2)
        all_images[i, top:top+im.shape[0], left:left+im.shape[1]] = im

    return all_images, max_height, max_width



#https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p == actual[i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if actual is None:
        return 0.0

    return score / min(len(actual), k)



def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])
