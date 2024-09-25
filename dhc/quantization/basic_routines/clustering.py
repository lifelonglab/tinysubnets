import numpy as np
from sklearn.cluster import KMeans


def cluster_kmeans(data_in, word_len=2):
    grid = pow(2, word_len)
    kmeans = KMeans(n_clusters=grid).fit(data_in.flatten().reshape(-1, 1))
    data_out = np.vectorize(lambda item: kmeans.cluster_centers_[kmeans.predict(item)][0])(data_in)
    return data_out


def cluster_kmeans_cuda(data_in, word_len=2):
    pass


