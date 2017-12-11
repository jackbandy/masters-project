'''
transcribe.py
input: a sequence of images, a vae model, and a cluster model
output: a text transcription
'''

import os, keras
import numpy as np
import util, cluster
from keras.models import load_model, Model

samples_path = 'data/individual samples'
vae_path = 'vae_encoder_50epochs.h5'
clusters_path = 'labeled-clusters.npy'


def main():
    # step 1: load in the samples
    input_images, im_height, im_width = util.collectSamples(samples_path)
    n_samples = input_images.shape[0]
    input_images = input_images.reshape(n_samples, im_height, im_width,
            1) / 255.
    print("Collected {} images...".format(n_samples))

    # step 2: load the vae model and clusterererer
    encoder = load_model(vae_path)
    kmeans = cluster.createModelFromFile(clusters_path)

    # step 3: get the encoded representation of the input
    intermediate_layer_model = Model(inputs=vae.input, outputs=vae.layers[8].output)
    encoded = intermediate_layer_model.predict(input_images)
    encoded = np.reshape(encoded, (encoded.shape[0], -1)) 

    # step 4: predict!
    transcript = ['']*n_samples
    clustered = kmeans.predict(encoded)
    for i in range(len(clustered)):
        cluster_number = clustered[i]
        transcript[i] = cluster_labels[cluster_number]['word']

    print(transcript)





if __name__ == "__main__":
    main()
