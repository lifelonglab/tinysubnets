import numpy as np
import random as random


def get_edgepoints_from_centroids(midpoints, bin_edges,  centroids, hist):
    edgepoints = np.empty([len(centroids)+1])
    edgepoints[0] = bin_edges[0]
    for clusterid  in range(len(centroids)-1):
        edgepoints[clusterid+1] = (centroids[clusterid] + centroids[clusterid+1])/2
    edgepoints[len(centroids)] = bin_edges[-1]
    return edgepoints


def get_centroids_from_edgepoints(midpoints, bin_edges, edgepoints, hist):
    centroids = np.empty([len(edgepoints)-1])
    for clusterid in range(len(edgepoints)-1):
        clusterlow = edgepoints[clusterid]
        clusterhigh = edgepoints[clusterid+1]
        condition = np.logical_and(midpoints >= clusterlow, midpoints <= clusterhigh)
        clust_arr = np.extract(condition, midpoints)
        hist_arr = np.extract(condition, hist)
        histsumval = np.sum(hist_arr)
        #print("clusterid = {}".format(clusterid))
        #print("histsumval = {}".format(histsumval))
        #print("lenhistsumval = {}".format(hist_arr.size))
        centroids[clusterid] = np.dot(clust_arr, hist_arr)/histsumval
    return centroids
          

def kmeans_scaler(midpoints, hist, bin_edges, n_clusters, n_iterations=10):
    max_val = bin_edges[-1]
    min_val = bin_edges[0]
    percentile_steps = 1/float(n_clusters)
    percentile_to_use = 1/(2*float(n_clusters))
    cumsumhist = np.cumsum(hist)#/np.sum(hist)
    print("CUMSUMHIST={}".format(cumsumhist))
    centroids = np.zeros([n_clusters])
    for iter in range(n_clusters):
        idx = np.searchsorted(cumsumhist,percentile_to_use)
        idx = np.clip(idx, 0, midpoints.size - 1)
        centroids[iter] = midpoints[idx]
        #print("{} => {} {}".format(idx, centroids[iter], percentile_to_use))
        percentile_to_use = percentile_to_use + percentile_steps
    print("initialized centroids = {}".format(centroids))
    for iter in range(n_iterations):
        edgepoints = get_edgepoints_from_centroids(midpoints, bin_edges, centroids, hist)
        centroids  = get_centroids_from_edgepoints(midpoints, bin_edges, edgepoints, hist)
    print("$$$$$$$$$$$ ITER = {} $$$$$$$$$$$".format(iter))
    print("edgepoints = {}".format(edgepoints))
    print("centroids = {}".format(centroids))
    return centroids, edgepoints


def kmeans_scaler_hist(inp, n_clusters, n_iterations=4):
    hist, bin_edges = np.histogram(inp, bins=10000)
    hist = hist/np.sum(hist)
    print("hist = {}, bin_edges = {}".format(hist, bin_edges))
    midpoints = (bin_edges[0:-2] + bin_edges[1:-1])/2
    centroids, edgepoints = kmeans_scaler(midpoints, hist, bin_edges, 16, n_iterations=200)
    return centroids, edgepoints


def vq_and_back(filt, n_clusters, sparsity_threshold=0):
    X = filt.reshape((-1, 1))  # We need an (n_sample, n_feature) array
    sparsity_enabled=(sparsity_threshold!=0)
    clusters_used = n_clusters
    k_means = cluster.KMeans(n_clusters=clusters_used, n_init=1, verbose=0, n_jobs=-1)
    k_means.fit(X)
    values = k_means.cluster_centers_.squeeze()
    labels = k_means.labels_
    if sparsity_enabled:
        min_idx = np.argmin(values)
        values[min_idx] = 0

    # create an array from labels and values
    #out = np.choose(labels, values)
    print("Cluster Values = {}".format(values))
    out = np.take(values, labels)
    out.shape = filt.shape
    return out


def vq_and_back_fast(filt, n_clusters, sparsity_threshold=0):
    X = filt.reshape((-1, 1))  # We need an (n_sample, n_feature) array
    sparsity_enabled=(sparsity_threshold!=0)
    #print("X.Shape")
    #print(X.shape)
    clusters_used = n_clusters
    k_means = cluster.KMeans(n_clusters=clusters_used, n_init=1, verbose=0, n_jobs=-1)
    sz = X.shape
    print(sz)
    if False:#sz[0] > 1000000:
        idx = np.random.choice(sz[0],100000)
        x_short = X[idx,:]
    else:
        x_short = X
    k_means.fit(x_short)
    values = k_means.cluster_centers_#.squeeze()
    labels = k_means.labels_
    if sparsity_enabled:
        min_idx = np.argmin(values)
        values[min_idx] = 0
    #    for ix in range(len(values)):
    #        if values[ix] < sparsity_threshold:
    #            values[ix] = 0

    # create an array from labels and values
    #out = np.choose(labels, values)
    print("Cluster Values = {}".format(values))
    print("shape x")
    print(X.shape)
    print("shape values")
    print(values.shape)
    labels, dist = vq.vq(X, values)
    print("shape labels")
    print(labels)
    out = np.take(values, labels)
    out.shape = filt.shape
    return out


def vq_and_back_fastest(filt, n_clusters, sparsity_threshold=0):
    X = filt.reshape((-1, 1))  # We need an (n_sample, n_feature) array
    sparsity_enabled=(sparsity_threshold!=0)
    clusters_used = n_clusters
    sz = X.shape
    print(sz)
    idx = np.random.choice(sz[0],100000)
    x_short = X[idx,:]
    values, edges = kmeans_scaler_hist(x_short, clusters_used)
    if sparsity_enabled:
        min_idx = np.argmin(values)
        values[min_idx] = 0

    print("Cluster Values = {}".format(values))
    print("shape x")
    print(X.shape)
    print("shape values")
    print(values.shape)
    labels, dist = vq.vq(X.flatten(), values)
    print("shape labels")
    print(labels)
    ids, counts = np.unique(labels, return_counts=True)
    print("Counts")
    print(counts)
    out = np.take(values, labels)
    out.shape = filt.shape
    return out


def vquant(in_tensor, n_clusters=16, sparsity_threshold=0, fast=False):
    in_np = in_tensor.cpu().numpy()
    np.random.seed(0)
    shape = in_np.shape
    out_combined = np.zeros(in_np.shape)
    if False: #in_np.ndim == 4:
        for itr in range(shape[0]):
            print(str(itr) + ': shape' + str(in_np.shape))
            filt = in_np[itr,:,:,:]
            out = vq_and_back(filt, n_clusters)
            out.shape = filt.shape
            out_combined[itr,:,:,:] = out
    else: #in_np.ndim == 2:
        print('shape' + str(in_np.shape))
        filt = in_np
        if fast == True:
            out = vq_and_back_fastest(filt, n_clusters, sparsity_threshold=sparsity_threshold)
        else:
            out = vq_and_back(filt, n_clusters, sparsity_threshold=sparsity_threshold)
        out_combined = out
    #else:
    #   raise Exception('We Should not be here')

    out_tensor = torch.from_numpy(out_combined)

    return out_tensor.cuda()



if __name__ == '__main__':
    arr = np.random.rand(1,100000)
    centroids, edgepoints = kmeans_scaler_hist(arr, 16)
    print(centroids)
    print(edgepoints)
 
