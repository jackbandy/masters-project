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
import pdb

EPOCHS = [100]
NEURON_COUNT = [16]



def main():
    for ep in EPOCHS:
        for nc in NEURON_COUNT:
            run_instance(ep, nc)


def run_instance(epochs, neuron_count):
    '''
    training_samples_path = '../gw-data/data/word_images_damaged/'
    gt_samples_path = '../gw-data/data/word_images_normalized/'
    output_images_path = '../gw-data/data/damaged_word_images_reconstructed/'
    vae_encoder_save_path = 'vae_encoder_dmgw_{}epochs_{}neurons.h5'.format(epochs, neuron_count)
    vae_save_path = 'vae_dmgw_{}epochs_{}neurons.h5'.format(epochs, neuron_count)

    training_samples_path = '../pz-data/data/word_images_damaged/'
    gt_samples_path = '../pz-data/data/word_images_normalized/'
    output_images_path = '../pz-data/data/damaged_word_images_reconstructed/'
    vae_encoder_save_path = 'vae_encoder_dmpz_{}epochs_{}neurons.h5'.format(epochs, neuron_count)
    vae_save_path = 'vae_dmpz_{}epochs_{}neurons.h5'.format(epochs, neuron_count)
    '''
    training_samples_path = '../data/first_page_samples/'
    gt_samples_path = '../data/first_page_samples/'
    output_images_path = '../data/word_images_reconstructed/'
    vae_encoder_save_path = 'vae_encoder_wy_{}epochs_{}neurons.h5'.format(epochs, neuron_count)
    vae_save_path = 'vae_wy_{}epochs_{}neurons.h5'.format(epochs, neuron_count)

    if not (os.path.isdir(output_images_path)):
        os.mkdir(output_images_path)
    file_names = os.listdir(training_samples_path)
    file_names.sort()

    # step 1: load in the samples
    training_images, im_height, im_width =util.collectSamples(training_samples_path, binarize=True, invert=True)
    gt_images, im_height, im_width =util.collectSamples(gt_samples_path, binarize=True, invert=True)
    n_samples = training_images.shape[0]
    white_val = np.max(training_images)
    white_val_gt = np.max(gt_images)
    print("White value is {}".format(white_val))
    training_images = training_images.reshape(n_samples, im_height, im_width,
            1) / white_val
    gt_images = gt_images.reshape(n_samples, im_height, im_width,
            1) / white_val_gt
    print("Collected {} images...".format(n_samples))

    # step 2: build and train the model
    print("Training network...")
    neuron_array = [4,4,8,neuron_count]
    validation_images = (training_images)[-20:]
    validation_gt = (gt_images)[-20:]
    vae = buildNetwork(input_height=im_height, input_width=im_width, neurons=neuron_array)
    vae.compile(optimizer='adadelta',loss='binary_crossentropy')
    vae.fit(training_images, gt_images,
            epochs=epochs,
            batch_size=30,
            shuffle=True,
            validation_data=(validation_images, validation_gt),
    )
            

    # step 3: visualize results
    print("Generating results from vae...")
    #output_ims = vae.predict(training_images)
    #predict_images = np.random.permutation(training_images)
    output_ims = vae.predict(training_images)
    n = 10
    print("Saving result images...")
    for i in range(n_samples):
        # original
        #ax = plt.subplot(2, n, i+1)
        #plt.imshow(predict_images[i].reshape(im_height, im_width))
        #plt.gray()
        inv = util.invertImage(output_ims[i,:,:,0]*white_val)
        imageio.imwrite(output_images_path+'/{}'.format(file_names[i]), inv)
        #ax = plt.subplot(2, n, i+n+1)
        #plt.imshow(output_ims[i].reshape(im_height, im_width))
        #plt.gray()
    #plt.savefig('results-{}epochs-{}clusters-{}neurons.png'.format(epochs, n_clusters, neuron_count))
    #plt.clf()
    print("Saving networks...")
    intermediate_layer_model = Model(inputs=vae.input, outputs=vae.layers[8].output)
    intermediate_layer_model.save(vae_encoder_save_path)
    vae.save(vae_save_path)


    '''
    # step 4: cluster
    print("Building a cluster model...")
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
    '''
       




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
