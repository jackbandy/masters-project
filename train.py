'''
train.py
use this script to train on images
'''

import os, keras
from scipy import misc
from PIL import Image, ImageOps
import PIL.ImageOps
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
import matplotlib.pyplot as plt


def main():
    # step 1: load in the samples
    training_images, im_height, im_width  = collectSamples('data/individual_samples')
    n_samples = training_images.shape[0]
    training_images = training_images.reshape(n_samples, im_height, im_width,
            1) / 255.

    print("Collected {} images...".format(n_samples))

    # step 2: build the model
    vae = buildNetwork(input_height=im_height, input_width=im_width)
    vae.compile(optimizer='adadelta',loss='binary_crossentropy')
    vae.fit(training_images, training_images,
            epochs=2,
            batch_size=n_samples,
            shuffle=True,
            validation_data=(training_images, training_images),
            )
            
    layer_shapes = [layer.output.get_shape() for layer in vae.layers]
    print("Shapes:\n{}".format(layer_shapes))
    layer_seven = vae.layers[7].output
    print("Layer 7: {}".format(layer_seven))

    # step 3: visualize results
    output_ims = vae.predict(training_images)
    n = 3
    plt.figure(figsize=(8,4))
    for i in range(n):
        # original
        ax = plt.subplot(2, n, i+1)
        plt.imshow(training_images[i].reshape(im_height, im_width))
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
        try:
            im = misc.imread(directory + '/' + f, flatten=True)
            p_im = PIL.Image.fromarray(im).convert('L')
            invert = PIL.ImageOps.invert(p_im)
        except:
            print("Unable to read {} as image".format(f))
            continue

        images.append(invert)
        if im.shape[0] > max_height:
            max_height = im.shape[0]
        if im.shape[1] > max_width:
            max_width = im.shape[1]
    print("OLD max_width is {}, max_height is {}".format(max_width, max_height))
    while max_width % 8 != 0:
        max_width += 1
    while max_height % 8 != 0:
        max_height += 1
    print("NEW max_width is {}, max_height is {}".format(max_width, max_height))

    # second loop: center all images in a numpy array
    all_images = np.zeros((len(images), max_height, max_width),
            dtype=np.float32)
    print("Organizing image samples...")
    for i in range(len(images)):
        im = images[i]
        top = int((max_height - im.height) / 2)
        left = int((max_width - im.width) / 2)
        all_images[i, top:top+im.height, left:left+im.width] = im

    return all_images, max_height, max_width



def buildNetwork(input_height, input_width):
    input_img = Input(shape=(input_height, input_width, 1))
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3,3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(16, (3,3), activation='relu', padding='same')(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3,3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    return autoencoder


if __name__ == "__main__":
    main()
