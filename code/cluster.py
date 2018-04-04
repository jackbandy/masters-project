'''
cluster.py
Cluster VAE-encoded features into words
'''


import numpy as np
from sklearn.externals import joblib
import pickle
from sklearn.cluster import KMeans, AgglomerativeClustering


def createClusters(features):
    # how many clusters though
    pass


def predictAgglomNClusters(features, n_clusters):
    model = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(features)
    return model


def createNClusters(features, n_clusters):
    #scikit learn
    model = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
    return model



def createModelFromCentroids(centroids):
    # the scikit thing requires a model be fitted before using it
    # so, "fit" the model to the centroids it's initalized with
    model = KMeans(init=centroids, n_clusters=centroids.shape[0], n_init=1,
            max_iter=1).fit(centroids)
    # unfortunately this tweaks the model
    # so for restoration use saveClsuterer and loadClusterer 
    return model



def createModelFromFile(file_path):
    cluster_labels = np.load(file_path)
    centroids_array = cluster_labels[:]['centroid']
    return createModelFromCentroids(centroids_array)



def saveClusterer(model, file_path):
    pickle.dump(model, open(file_path, 'wb'))



def loadClusterer(file_path):
    loaded_model = pickle.load(open(file_path, 'rb'))
    return loaded_model



def saveClusters(centroids, n_clusters, n_neurons, directory=''):
    # save a NumPy structured array
    # each index has a centroid and a word label
    # initially, every word label is 'unknown'
    clusters = [('unknown', centroids[i]) for i in range(len(centroids))]
    np_clusters = np.array(clusters, dtype=[('word', 'S20'), ('centroid',
        np.float32, (centroids.shape[1]))])
    if len(directory) > 0:
        if not (os.path.isdir(directory)):
            os.mkdir(directory)
        np.save('{}/labeled_clusters_{}clusters_{}neurons.npy'.format(directory, n_clusters, n_neurons), np_clusters)
    else:
        np.save('labeled_clusters_{}clusters_{}neurons.npy'.format(n_clusters, n_neurons), np_clusters)
