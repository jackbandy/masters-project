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
import imageio

EPOCHS = [20]
N_CLUSTERS = [1800,2000,2200]
NEURON_COUNT = [8,16]



def main():
    for ep in EPOCHS:
        for nc in NEURON_COUNT:
            run_instance(ep, N_CLUSTERS, nc)


def run_instance(epochs, N_CLUSTERS, neuron_count):
    training_samples_path = '../gw-data/data/word_images_normalized/'
    vae_save_path = 'vae_encoder_{}epochs.h5'.format(epochs)

    # step 1: load in the samples
    training_images, im_height, im_width =util.collectSamples(training_samples_path, binarize=False, invert=True)
    n_samples = training_images.shape[0]
    white_val = np.max(training_images)
    print("White value is {}".format(white_val))
    training_images = training_images.reshape(n_samples, im_height, im_width,
            1) / white_val

    print("Collected {} images...".format(n_samples))

    # step 2: build and train the model
    print("Training network...")
    neuron_array = [4,4,8,neuron_count]
    vae = buildNetwork(input_height=im_height, input_width=im_width, neurons=neuron_array)
    vae.compile(optimizer='adadelta',loss='binary_crossentropy')
    vae.fit(training_images, training_images,
            epochs=epochs,
            batch_size=10,
            shuffle=True,
            validation_data=(training_images, training_images),
    )
            

    # step 3: visualize results
    print("Saving results from vae...")
    predict_images = np.random.permutation(training_images)
    output_ims = vae.predict(predict_images)
    n = 10
    #plt.figure(figsize=(8,4))
    instance_dir = 'results-{}epochs-{}neurons'.format(epochs, neuron_count)
    os.mkdir(instance_dir)
    for i in range(n):
        # original
        #ax = plt.subplot(2, n, i+1)
        #plt.imshow(predict_images[i].reshape(im_height, im_width))
        #plt.gray()
        imageio.imwrite(instance_dir+'/{}-raw.png'.format(i), predict_images[i])
        imageio.imwrite(instance_dir+'/{}-rec.png'.format(i), output_ims[i])
        #ax = plt.subplot(2, n, i+n+1)
        #plt.imshow(output_ims[i].reshape(im_height, im_width))
        #plt.gray()
    #plt.savefig('results-{}epochs-{}clusters-{}neurons.png'.format(epochs, n_clusters, neuron_count))
    #plt.clf()


    # step 4: cluster
    print("Building a cluster model...")
    intermediate_layer_model = Model(inputs=vae.input, outputs=vae.layers[8].output)
    intermediate_layer_model.save(vae_save_path)
    encoded = intermediate_layer_model.predict(predict_images)
    encoded = np.reshape(encoded, (encoded.shape[0], -1))

    for n_clusters in N_CLUSTERS:
        kmeans_save_path = 'kmeans_{}epochs_{}clusters.sav'.format(epochs, n_clusters)
        kmeans = cluster.createNClusters(encoded, n_clusters)
        labels = kmeans.predict(encoded)

        #labels = cluster.predictAgglomNClusters(encoded, n_clusters)

        #util.saveImagesWithLabels(images=predict_images, labels=labels,directory='test-labels')
        cluster.saveClusters(centroids=kmeans.cluster_centers_)
        cluster.saveClusterer(model=kmeans, file_path=kmeans_save_path)
       




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
