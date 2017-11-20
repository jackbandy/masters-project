'''
cluster.py
Cluster VAE-encoded features into words
'''


import numpy as np
from sklearn.cluster import KMeans


def createClusters(features):
    # how many clusters though
    pass


def createNClusters(features, n_clusters):
    #scikit learn
    model = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
    return model
