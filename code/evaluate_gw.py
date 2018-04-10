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

'''
# GW data
normal_images_path = '../gw-data/data/word_images_normalized/'
damaged_images_path = '../gw-data/data/word_images_damaged/'
reconstructed_images_path = '../gw-data/data/damaged_word_images_reconstructed/'
encoder_path = 'vae_encoder_gw_20epochs_8neurons.h5'
ground_truth_path = '../gw-data/ground_truth/just_words.txt'

# Wycliffe data
normal_images_path = '../data/data/first_page_samples/'
ground_truth_path = '../data/ground_truth/just_words.txt'

# PZ data
normal_images_path = '../pz-data/data/word_images_normalized/'
damaged_images_path = '../pz-data/data/word_images_damaged/'
reconstructed_images_path = '../pz-data/data/damaged_word_images_reconstructed/'
encoder_path = 'vae_encoder_dmpz_20epochs_16neurons.h5'
ground_truth_path = '../pz-data/ground_truth/words_only.txt'
'''

# WY data
normal_images_path = '../data/first_page_samples/'
reconstructed_images_path = '../data/word_images_reconstructed/'
encoder_path = 'vae_encoder_wy_100epochs_16neurons.h5'
ground_truth_path = '../data/words_only.txt'



def main():
    #print("apk for damaged pixels:\n {}".format(evaluateForData(damaged_images_path, feats='pixels2')))
    #print("apk for damaged hog:\n {}".format(evaluateForData(damaged_images_path, feats='hog')))
    #print("apk for damaged scaled hog:\n {}".format(evaluateForData(damaged_images_path, feats='hog2')))
    print("p@5, map for normal data vae:\n {}".format(evaluateForData(normal_images_path, feats='vae', distance_metric='cosine')))
    print("p@5, map for normal data hog:\n {}".format(evaluateForData(normal_images_path, feats='hog2', distance_metric='cosine')))
    print("p@5, map for reconstructed data vae:\n {}".format(evaluateForData(reconstructed_images_path, feats='vae', distance_metric='cosine')))
    print("p@5, map for reconstructed data hog:\n {}".format(evaluateForData(reconstructed_images_path, feats='hog2', distance_metric='cosine')))



def evaluateForData(images_path, feats='hand', distance_metric='euclidean'):
    # initialization
    samples, height, width = util.collectSamples(images_path, binarize=False, scale_to_fill=True)


    # first, get features
    if feats=='hand':
        sample_features = util.getHandFeatsForSamples(samples)
    elif feats=='pixels':
        sample_features = util.getPixelFeatsForSamples(samples)
    elif feats=='pixels2':
        sample_features = util.getPixelFeatsForSamples(samples, scale=2)
    elif feats=='hog':
        sample_features = util.getHogForSamples(samples)
    elif feats=='hog2':
        sample_features = util.getHogForSamples(samples, scale=2)
    elif feats=='vae':
        sample_features = util.getVAEFeatsForSamples(samples, vae_encoder_path=encoder_path)
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
    distances = pairwise_distances(sample_features, metric=distance_metric)

    # for each image
    apk_total = 0.0
    apk_count = 0
    pa5_total = 0.0
    pa5_count = 0
    for i in range(n_words):
        n = 5  # find the n closest feature vectors
        dist_to_words = distances[i]
        word_id = sample_gt[i]
        # trim off the match with itself
        closest_n_inds = dist_to_words.argsort()[:n]
        actual = sample_gt_np[closest_n_inds]
        pred = [word_id]*n

        occurences = min(n,sample_gt.count(word_id))
        if occurences > 4:
            pa5_count += 1
            pa5_total += actual.tolist().count(word_id) / 5

        w_k = max(1,sample_gt.count(word_id))
        if w_k > 1:
            apk = util.apk(actual, pred, k=w_k)
            apk_total += apk
            apk_count += 1
        '''
        print("--")
        print("Actual: {}".format(actual))
        print("Pred: {}".format(pred))
        print("Occurs: {}".format(occurences))
        print("apk at word {}:{:.3f} \r".format(i,apk))
        '''

    average_pa5 = pa5_total / pa5_count
    average_apk = apk_total / apk_count
    return average_pa5, average_apk






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
