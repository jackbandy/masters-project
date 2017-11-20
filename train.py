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


def main():
    # step 1: load in the samples
    training_images, im_height, im_width  = collectSamples('data/lined_samples')
    n_samples = training_images.shape[0]
    training_images = training_images.reshape(n_samples, im_height, im_width,
            1) / 255.

    print("Collected {} images...".format(n_samples))

    # step 2: build the model
    vae = buildNetwork(input_height=im_height, input_width=im_width)
    vae.compile(optimizer='adadelta',loss='binary_crossentropy')
    vae.fit(training_images, training_images,
            epochs=20,
            batch_size=10,
            shuffle=True,
            validation_data=(training_images, training_images),
    )
            

    # step 3: cluster
    predict_images = np.random.permutation(training_images[0:30])
    intermediate_layer_model = Model(inputs=vae.input, outputs=vae.layers[8].output)
    encoded = intermediate_layer_model.predict(predict_images)
    encoded = np.reshape(encoded, (encoded.shape[0], -1))
    print("Layer 7 shape: {}".format(encoded.shape))
    print("Layer 7 sample: {}".format(encoded[0]))

    kmeans = cluster.createNClusters(encoded, 20)
    labels = kmeans.predict(encoded)

    util.saveImagesWithLabels(images=predict_images, labels=labels,
                                directory='test-labels')
    
    # step 4: visualize results
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
    plt.show()



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



def buildNetwork(input_height, input_width):
    neurons = [2, 4, 8, 16]
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
