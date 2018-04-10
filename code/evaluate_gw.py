'''
evaluate_gw.py
Evaluate performance on george washington dataset
'''

import os
import numpy as np
from PIL import Image 
import nltk
import util, cluster
from keras.models import load_model
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import pairwise_distances
import pdb


normal_images_path = '../gw-data/data/word_images_normalized/'
damaged_images_path = '../gw-data/data/word_images_damaged/'
model_path = 'vae_encoder_20epochs_8neurons.h5'
cluster_path = 'labeled_clusters.npy'
#clusterer_path = 'kmeans_50epochs_50clusters.sav'
clusterer_path = 'kmeans_20epochs_2200clusters.sav'
ground_truth_path = '../gw-data/ground_truth/just_words.txt'


def main():
    print("apk for damaged scaled pixels:\n {}".format(evaluateForData(damaged_images_path, feats='pixels2')))
    print("apk for damaged pixels:\n {}".format(evaluateForData(damaged_images_path, feats='pixels')))
    print("apk for damaged:\n {}".format(evaluateForData(damaged_images_path)))
    print("apk for data scaled pixels:\n {}".format(evaluateForData(normal_images_path, feats='pixels2')))
    print("apk for data pixels:\n {}".format(evaluateForData(normal_images_path, feats='pixels')))
    print("apk for data:\n {}".format(evaluateForData(normal_images_path)))



def evaluateForData(images_path, feats='hand'):
    # initialization
    samples, height, width = util.collectSamples(images_path, binarize=False,
            scale_to_fill=True)


    # first, get features
    if feats=='hand':
        sample_features = util.getHandFeatsForSamples(samples)
    elif feats=='pixels':
        sample_features = util.getPixelFeatsForSamples(samples)
    elif feats=='pixels2':
        sample_features = util.getPixelFeatsForSamples(samples, scale=2)
    print("Samples shape: {}".format(sample_features.shape))


    # create ground truth for the features
    print("Generating ground truth...")
    words = makeWordArray(ground_truth_path)
    n_words = len(words)
    n_unique_words = len(set(words))+1
    sample_gt, n, word_to_int, int_to_word = util.buildDataset(words, n_unique_words)
    sample_gt_np = np.array(sample_gt)


    # calculate distance matrix!
    print("Calculating pairwise distances...")
    distances = pairwise_distances(sample_features)
    print("distances shape: {}".format(distances.shape))

    # for each image
    apk_total = 0.0
    for i in range(n_words):
        n = 5  # find the n closest feature vectors
        dist_to_words = distances[i]
        word_id = sample_gt[i]
        # trim off the match with itself
        closest_n_inds = dist_to_words.argsort()[:n]
        actual = sample_gt_np[closest_n_inds]
        pred = [word_to_int[words[i]]]*n

        occurences = min(n,sample_gt.count(word_id))
        apk = util.apk(actual, pred, k=occurences)
        apk_total += apk
        '''
        print("--")
        print("Actual: {}".format(actual))
        print("Pred: {}".format(pred))
        print("Occurs: {}".format(occurences))
        print("apk at word {}:{:.3f} \r".format(i,apk))
        '''

    average_apk = apk_total / n_words
    return average_apk






def encodeImages(encoder, input_images, im_height, im_width):
    input_images = input_images.reshape(input_images.shape[0], im_height, im_width,
            1) 
    encoded = encoder.predict(input_images)
    encoded = np.reshape(encoded, (encoded.shape[0], -1))
    return encoded



def makeWordArray(text_file_path):
    txt = open(text_file_path).read()
    toks = nltk.word_tokenize(txt)
    return toks


if __name__ == '__main__':
    main()
