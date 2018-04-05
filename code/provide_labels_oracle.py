'''
provide_labels_oracle.py
Run this script to automatically add ground-truth labels to the database
Requires a perfectly ordered image list and corresponding text
'''

import tkinter as tk
import os
import numpy as np
from PIL import Image, ImageTk
import nltk
import util, cluster
from keras.models import load_model
from sklearn.metrics import accuracy_score


images_path = 'data/ordered_samples'
model_path = 'vae_encoder_50epochs.h5'
cluster_path = 'labeled_clusters.npy'
#clusterer_path = 'kmeans_50epochs_50clusters.sav'
clusterer_path = 'kmeans_50epochs_50clusters.sav'
ground_truth_path = 'data/lined_samples.txt'

PUNCT = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'


def main():
    # initialization
    samples, height, width = util.collectSamples(images_path)
    clusterer = cluster.loadClusterer(clusterer_path)
    n_clusters = clusterer.cluster_centers_.shape[0]
    cluster_labels = [''] * n_clusters

    ground_truth = makeWordArray(ground_truth_path)
    print("GT: {}".format(ground_truth))
    encoder = load_model(model_path)

    # encoding and clustering
    print("Collected {} samples".format(len(samples)))
    encoded = encodeImages(encoder, samples, height, width)
    clustered = clusterer.predict(encoded)
    transcript = []


    # for each encoded word, if its cluster is not labeled, then label it
    for i in range(len(clustered)):
        cluster_num = clustered[i]
        cluster_word = cluster_labels[cluster_num]
        if cluster_word == '':
            label = ground_truth[i]
            cluster_labels[cluster_num] = label
            transcript.append(label)
            print("Labeling Cluster {} as {}".format(cluster_num, label))
        else:
            print("Cluster {} is {}".format(cluster_num, cluster_word))
            transcript.append(cluster_labels[cluster_num])

    print("\nTranscript: {}".format(transcript))
    print("Accuracy: {}".format(accuracy_score(ground_truth, transcript)))




def encodeImages(encoder, input_images, im_height, im_width):
    input_images = input_images.reshape(input_images.shape[0], im_height, im_width,
            1) / 255.
    encoded = encoder.predict(input_images)
    encoded = np.reshape(encoded, (encoded.shape[0], -1))
    return encoded



def makeWordArray(text_file_path):
    txt = open(text_file_path).read()
    toks = nltk.word_tokenize(txt)
    stripped_toks = [t for t in toks if t not in PUNCT]
    # dang I love python
    return stripped_toks 


if __name__ == '__main__':
    main()
