'''
train.py
use this script to train on images
'''

import os, keras
from keras import backend as K
from scipy import misc
from PIL import Image, ImageOps
import PIL.ImageOps
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
import matplotlib.pyplot as plt
import util, cluster

EPOCHS = [10]
N_CLUSTERS = [1500,2000]
NEURON_COUNT = [16,24]



def main():
    for ep in EPOCHS:
        for cl in N_CLUSTERS:
            for nc in NEURON_COUNT:
                run_instance(ep, cl, nc)


def run_instance(epochs, n_clusters, neuron_count):
    training_samples_path = '../gw-data/data/word_images_normalized/'
    vae_save_path = 'vae_encoder_{}epochs.h5'.format(epochs)
    kmeans_save_path = 'kmeans_{}epochs_{}clusters.sav'.format(epochs, n_clusters)

    # step 1: load in the samples
    training_images, im_height, im_width =util.collectSamples(training_samples_path, binarize=False, invert=False)
    n_samples = training_images.shape[0]
    training_images = training_images.reshape(n_samples, im_height, im_width,
            1) / 255.

    print("Collected {} images...".format(n_samples))

    # step 2: build the model
    vae = buildNetwork(input_height=im_height, input_width=im_width, neurons=[neuron_count]*4)
    vae.compile(optimizer='adadelta',loss='binary_crossentropy')
    vae.fit(training_images, training_images,
            epochs=epochs,
            batch_size=10,
            shuffle=True,
            validation_data=(training_images, training_images),
    )
            

    # step 3: visualize results
    predict_images = np.random.permutation(training_images)
    output_ims = vae.predict(predict_images)
    n = 5
    plt.figure(figsize=(8,4))
    for i in range(n):
        # original
        ax = plt.subplot(2, n, i+1)
        plt.imshow(predict_images[i].reshape(im_height, im_width))
        plt.gray()

        ax = plt.subplot(2, n, i+n+1)
        plt.imshow(output_ims[i].reshape(im_height, im_width))
        plt.gray()
    plt.savefig('results-{}epochs-{}clusters-{}neurons.png'.format(epochs, n_clusters, neuron_count))
    plt.clf()

    # step 4: cluster
    intermediate_layer_model = Model(inputs=vae.input, outputs=vae.layers[8].output)
    intermediate_layer_model.save(vae_save_path)
    encoded = intermediate_layer_model.predict(predict_images)
    encoded = np.reshape(encoded, (encoded.shape[0], -1))

    #kmeans = cluster.createNClusters(encoded, n_clusters)
    #labels = kmeans.predict(encoded)

    labels = cluster.predictAgglomNClusters(encoded, n_clusters)

    util.saveImagesWithLabels(images=predict_images, labels=labels,
                                directory='test-labels')
    cluster.saveClusters(centroids=kmeans.cluster_centers_)
    cluster.saveClusterer(model=kmeans, file_path=kmeans_save_path)
   



def collectSamples(directory):
    file_names = os.listdir(directory)
    images = []
    max_height = 0
    max_width = 0

    # first loop put all images in python list
    print("Collecing image files...")
    for f in file_names:
        inverted = util.loadImage(directory + '/' + f)
        if inverted is None:
            continue
        binarized = util.removeBackground(inverted)
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



def buildNetwork(input_height, input_width, neurons):
    input_img = Input(shape=(input_height, input_width, 1))
    x = Conv2D(neurons[0], (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(neurons[1], (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(neurons[2], (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(neurons[3], (3,3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(neurons[3], (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(neurons[2], (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(neurons[1], (3,3), activation='relu', padding='same')(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(neurons[0], (3,3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3,3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    return autoencoder


if __name__ == "__main__":
    main()
